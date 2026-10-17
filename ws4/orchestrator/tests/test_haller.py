"""Sunday demo path on the green Haller, with run_single_arm.py's fake arm and a fake policy (no hardware, no pod)."""

import threading
import time
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from orchestrator.haller import OtherArmPose
from orchestrator.monitors import KeepOutMonitor, StallMonitor
from orchestrator.plan import Plan, Step, validate_plan
from orchestrator.planner import StubPlanner
from orchestrator.session import Session, SessionConfig, load_trial_log
from orchestrator.skills import DEFAULT_VOCABULARY, load_vocabulary
from orchestrator.verifier import Verdict

PUT = "put the red pen in its slot in the cardboard box"
TAKE = "take the red pen out of its slot and put it on the mat"


@pytest.fixture(scope="module")
def pen():
    return load_vocabulary(DEFAULT_VOCABULARY)


def test_vocabulary_is_the_final_demo_task(pen):
    """The FLUX prompts are the datasets' exact strings; one task arm (robot left); home is the demos' start pose."""
    assert pen["put_pen_in_slot"].prompt == PUT
    assert pen["take_pen_out"].prompt == TAKE
    assert pen["put_pen_in_slot"].success_check == "The red pen is standing in its slot in the cardboard box."
    assert pen["take_pen_out"].success_check == "The red pen is lying on the mat, outside the box."
    assert pen.station.home_pose == (0, 0, 0, 0, 0, 100)
    assert pen.station.arms_available == ("left",)
    assert all(s.arms == ("left",) for s in pen.skills.values())
    assert {s.verifier_model for s in pen.skills.values() if s.is_flux} == {"claude-haiku-5-5"}
    assert all(s.start_pose == "home" for s in pen.skills.values() if s.is_flux)


def test_stub_chains_and_cycles(pen):
    planner = StubPlanner(pen)
    plan = planner.plan("put the pen in the box, take it out, put it back").plan
    assert [s.skill for s in plan.steps] == ["put_pen_in_slot", "take_pen_out", "put_pen_in_slot"]
    assert [s.depends_on for s in plan.steps] == [[], [plan.steps[0].id], [plan.steps[1].id]]
    assert all(s.arm == "left" for s in plan.steps)
    cycles = planner.plan("put the pen in its slot, then take it out, 3 times").plan
    assert [s.skill for s in cycles.steps] == ["put_pen_in_slot", "take_pen_out"] * 3
    assert validate_plan(cycles, pen) == []


def test_right_arm_is_refused(pen):
    plan = Plan([Step("a", "right", "put_pen_in_slot", success_check="x", max_attempts=1)])
    assert any("can only run on ['left']" in e for e in validate_plan(plan, pen))


def short_vocab(tmp_path, timeout=1.5):
    data = yaml.safe_load(DEFAULT_VOCABULARY.read_text())
    for name in ("put_pen_in_slot", "take_pen_out"):
        data["skills"][name]["timeout_s"] = timeout
        data["skills"][name]["verify_every_chunks"] = 1
    path = tmp_path / "skills.yaml"
    path.write_text(yaml.safe_dump(data))
    return str(path)


class PassOnAttempt:
    """Verifier stand-in: a step's check passes on its n-th attempt (1 = first), never before."""

    name, fast = "scripted-test", True

    def __init__(self, passes_on=None):
        self.passes_on = passes_on or {}
        self.calls = Counter()
        self.attempts = {}

    def check(self, step, skill, obs):
        self.calls[step.id] += 1
        need = self.passes_on.get(step.id, 1)
        ok = self.attempts.get(step.id, 1) >= need
        return Verdict(ok, 0.95, "pen where it should be" if ok else "not yet", self.name)


@pytest.fixture
def haller(tmp_path):
    cfg = SessionConfig(env="haller", robot="fake", policy="fake", planner="stub", verifier="manual",
                        realtime=False, task="pen_chain", vocabulary=short_vocab(tmp_path), max_run_s=60,
                        trial_log=str(tmp_path / "trials.jsonl"), runs_dir=str(tmp_path / "runs"))
    s = Session(cfg)
    yield s
    s.close()


def test_chain_homes_before_every_step_and_logs_each_attempt(haller, tmp_path):
    verifier = PassOnAttempt({"take_pen_out": 2})
    haller.verifier = verifier
    plan = haller.plan("put the pen in the box, take it out, put it back").result.plan
    seen_start = []
    original = haller.runner.start

    def spy(target, task):
        if not target.endswith(":prim"):  # the moment a FLUX policy starts, the arm must be at home
            seen_start.append((target, task, np.round(haller.rig.robot.joints(), 1).tolist()))
        if target.endswith(":prim") is False and "take" in task:
            verifier.attempts["take_pen_out"] = verifier.attempts.get("take_pen_out", 0) + 1
        return original(target, task)

    haller.runner.start = spy
    result = haller.execute(plan)
    assert result.status == "succeeded" and (result.steps_done, result.steps_total) == (3, 3)
    assert [t for t, _, _ in seen_start] == ["left:so101"] * 4  # put, take (fails once), take again, put
    assert [task for _, task, _ in seen_start] == [PUT, TAKE, TAKE, PUT]
    for _, _, q in seen_start:
        assert np.abs(np.array(q) - [0, 0, 0, 0, 0, 100]).max() <= 3.0
    trials = load_trial_log().load(tmp_path / "trials.jsonl")
    attempts = [t for t in trials if t.task != "pen_chain"]
    assert [(t.task, t.steps_done) for t in attempts] == [
        ("put_pen_in_slot", 1), ("take_pen_out", 0), ("take_pen_out", 1), ("put_pen_in_slot", 1)]
    assert "timeout" in attempts[1].failure and attempts[1].attempt == 1 and attempts[2].attempt == 2
    assert PUT in attempts[0].notes
    (run_line,) = [t for t in trials if t.task == "pen_chain"]
    assert run_line.success and run_line.steps_done == 3


def test_monitor_stop_holds_the_arm_and_freezes(haller):
    haller.verifier = PassOnAttempt({"put_pen_in_slot": 99})
    fired = {"n": 0}

    def trip(report, obs):
        fired["n"] += 1
        return "test: stall" if fired["n"] == 40 else None

    haller.rig.monitors.append(trip)
    holds = []
    haller.rig.hold = lambda reason="": holds.append(reason)
    plan = haller.plan("put the pen in the box").result.plan
    out = {}
    thread = threading.Thread(target=lambda: out.setdefault("r", haller.execute(plan)))
    thread.start()
    deadline = time.time() + 20
    while (haller.executor is None or haller.executor.state != "estopped") and time.time() < deadline:
        time.sleep(0.01)
    assert haller.executor.state == "estopped"
    assert holds and "stall" in holds[0]
    assert any("safety monitor: test: stall" in e["msg"] for e in haller.events.recent())
    haller.abort()
    thread.join(timeout=10)
    assert out["r"].status == "aborted"


def test_stall_and_keepout_monitors():
    stall = StallMonitor("left", stall_deg=8.0, stall_ticks=3)
    cmd = np.zeros(12)
    obs = {"observation.state": np.zeros(12)}
    obs["observation.state"][2] = 20.0  # elbow lags its command by 20 deg
    reasons = [stall(SimpleNamespace(tick=i, command=cmd), obs) for i in range(1, 5)]
    assert reasons[:2] == [None, None] and "elbow_flex" in reasons[2]
    obs["observation.state"][5] = 60.0  # the gripper lag alone never trips it
    obs["observation.state"][2] = 0.0
    assert all(stall(SimpleNamespace(tick=10 + i, command=cmd), obs) is None for i in range(5))
    guard = SimpleNamespace(last_zone="table")
    keep = KeepOutMonitor(guard, ticks=2)
    assert keep(None, {}) is None and "table" in keep(None, {})
    guard.last_zone = ""
    assert keep(None, {}) is None


def test_fake_arm_never_writes_the_graded_log(tmp_path):
    cfg = SessionConfig(env="haller", robot="fake", policy="fake", verifier="manual", realtime=False,
                        runs_dir=str(tmp_path / "runs"))
    s = Session(cfg)
    try:
        assert s.trial_log_path().name == "trials_mock.jsonl"
        assert s.gate is None and s.rig is not None and not s.rig.real
        assert sorted(s.target_backend) == ["left:prim", "left:so101"]  # the parked right arm gets no target
    finally:
        s.close()


def test_keepout_monitor_stops_the_ramp_at_the_first_refusal():
    guard = SimpleNamespace(last_zone="base")
    ramp = {"on": True}
    keep = KeepOutMonitor(guard, ticks=45, ramping=lambda: ramp["on"])
    assert "refused the ramp" in keep(None, {})  # move_to semantics: one refusal ends the ramp
    ramp["on"] = False
    assert all(keep(None, {}) is None for _ in range(44))  # under the policy, 45 in a row are needed
    assert "45 ticks" in keep(None, {})


def test_other_arm_pose_is_read_only_when_the_port_is_free(monkeypatch):
    from orchestrator import haller as haller_mod

    haller_mod.import_single_arm()
    import keepout

    ko = SimpleNamespace(other_q=None)
    now = {"t": 1000.0}
    rig = haller_mod.HallerRig(sar=None, so101=None, robot=None, guard=SimpleNamespace(keepout=ko), monitors=[],
                               rest=np.zeros(6), real=True, observe=False, clock=lambda: now["t"])
    reads = []
    good = np.arange(6.0)
    monkeypatch.setattr(keepout, "read_arm_joints", lambda port, rid: reads.append((port, rid)) or good)
    monkeypatch.setattr(haller_mod, "port_users", lambda port: [])
    pose = rig.refresh_other_arm()
    assert not pose.block and pose.note == "read"
    assert reads == [("/dev/haller_arm_left", "haller_left")] and ko.other_q == (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)

    def broken(port, rid):
        raise OSError("no such device")

    monkeypatch.setattr(keepout, "read_arm_joints", broken)
    now["t"] += 120  # a failed read reuses a pose younger than 180 s
    pose = rig.refresh_other_arm()
    assert not pose.block and pose.note == "cached 120 s" and ko.other_q == (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
    now["t"] += 61  # ... and blocks once it is older
    pose = rig.refresh_other_arm()
    assert "OSError" in pose.block and "181 s old" in pose.block and ko.other_q is None
    monkeypatch.setattr(keepout, "read_arm_joints", lambda port, rid: good)
    assert not rig.refresh_other_arm().block
    monkeypatch.setattr(haller_mod, "port_users", lambda port: [4242])  # someone may be driving the parked arm
    pose = rig.refresh_other_arm()
    assert "held by pid [4242]" in pose.block and ko.other_q is None
    monkeypatch.setattr(haller_mod, "port_users", lambda port: [])
    monkeypatch.setattr(keepout, "read_arm_joints", broken)  # a held port also drops the cached pose
    assert "no earlier pose" in rig.refresh_other_arm().block
    rig.read_other_arm = False
    pose = rig.refresh_other_arm()
    assert not pose.block and pose.note.startswith("DISABLED") and ko.other_q is None
    rig.real = False  # the fake arm never touches a bus
    assert rig.refresh_other_arm().note == "fake arm"


def test_port_users_finds_an_open_file(tmp_path):
    import subprocess
    import sys

    from orchestrator.haller import port_users

    path = tmp_path / "fake_port"
    path.write_text("")
    holder = subprocess.Popen([sys.executable, "-c", f"f = open({str(path)!r}); import time; time.sleep(30)"])
    try:
        deadline = time.time() + 10
        while holder.pid not in port_users(str(path)) and time.time() < deadline:
            time.sleep(0.05)
        assert holder.pid in port_users(str(path))
    finally:
        holder.kill()
        holder.wait()
    assert port_users(str(tmp_path / "nobody_has_this")) == []


def test_other_arm_read_before_every_ramp(haller, tmp_path):
    """As run_single_arm.run_attempt: read the parked arm first, so the ramp is checked against it too."""
    haller.verifier = PassOnAttempt()
    seq = []

    def refresh():
        seq.append("read")
        return OtherArmPose("read", "other arm pose for the keep-out: [0, 0, 0, 0, 0, 0]")

    haller.rig.refresh_other_arm = refresh
    original = haller.runner.start
    haller.runner.start = lambda target, task: seq.append(task) or original(target, task)
    plan = haller.plan("put the pen in the box, take it out").result.plan
    assert haller.execute(plan).status == "succeeded"
    assert seq == ["read", "home", PUT, "read", "home", TAKE]
    assert sum("other arm pose" in e["msg"] for e in haller.events.recent()) == 2
    attempts = [t for t in load_trial_log().load(tmp_path / "trials.jsonl") if t.task != "pen_chain"]
    assert len(attempts) == 2 and all("other_arm=read" in t.notes for t in attempts)


def test_unknown_other_arm_pose_blocks_the_attempt_until_resumed(haller):
    haller.verifier = PassOnAttempt()
    poses = [OtherArmPose("blocked: port held", "held", "other arm: /dev/haller_arm_left is held by pid [7]"),
             OtherArmPose("read", "other arm pose for the keep-out: [0, 0, 0, 0, 0, 0]")]
    haller.rig.refresh_other_arm = lambda: poses.pop(0)
    holds, starts = [], []
    haller.rig.hold = lambda reason="": holds.append(reason)
    original = haller.runner.start
    haller.runner.start = lambda target, task: starts.append(task) or original(target, task)
    plan = haller.plan("put the pen in the box").result.plan
    out = {}
    thread = threading.Thread(target=lambda: out.setdefault("r", haller.execute(plan)))
    thread.start()
    deadline = time.time() + 20
    while (haller.executor is None or haller.executor.state != "estopped") and time.time() < deadline:
        time.sleep(0.01)
    assert haller.executor.state == "estopped"
    assert holds and "held by pid [7]" in holds[0]
    assert starts == []  # neither the ramp nor FLUX starts on an unknown parked-arm pose
    haller.release()
    while haller.executor.state != "paused" and time.time() < deadline:
        time.sleep(0.01)
    haller.resume()
    thread.join(timeout=20)
    assert out["r"].status == "succeeded" and starts == ["home", PUT] and not poses


def test_keepout_refusal_during_the_ramp_stops_and_holds(haller):
    haller.verifier = PassOnAttempt()
    haller.rig.full_keepout.without_envelope = lambda: SimpleNamespace(refuse=lambda target, q: (True, "table"))
    holds = []
    haller.rig.hold = lambda reason="": holds.append(reason)
    plan = haller.plan("put the pen in the box").result.plan
    out = {}
    thread = threading.Thread(target=lambda: out.setdefault("r", haller.execute(plan)))
    thread.start()
    deadline = time.time() + 20
    while (haller.executor is None or haller.executor.state != "estopped") and time.time() < deadline:
        time.sleep(0.01)
    assert haller.executor.state == "estopped"
    assert holds and "refused the ramp" in holds[0]
    (run,) = haller.executor.runs.values()
    assert run.homing and not run.policy_started  # FLUX never started
    haller.abort()
    thread.join(timeout=10)
    assert out["r"].status == "aborted"


FOLDED_REST = [0.0, -96.0, 97.0, 1.0, 0.0, 2.0]


def ramp_refusal(keepout, start, goal, step=1.0):
    """First refused step of a 1 deg/tick joint ramp (move_to's stepping), or None."""
    cmd, goal = np.array(start, float), np.array(goal, float)
    limit = np.r_[np.full(5, step), 2 * step]
    n = 0
    while not np.allclose(cmd, goal):
        nxt = cmd + np.clip(goal - cmd, -limit, limit)
        n += 1
        if keepout.refuse(nxt, cmd)[0]:
            return n
        cmd = nxt
    return None


def test_ramp_from_the_folded_rest_uses_the_station_zones_only(haller):
    """The demonstrated envelope bounds the policy, not the ramp: unfolding from the folded rest leaves the demonstrated
    TCP workspace for a moment (so101-zeroshot, 2026-10-17). The policy still runs under every zone."""
    full = haller.rig.full_keepout
    if getattr(full, "env_cells", None) is None:
        pytest.skip("workspace file without a demonstrated envelope")
    home = [0, 0, 0, 0, 0, 100]
    assert ramp_refusal(full, FOLDED_REST, home) is not None
    assert ramp_refusal(full.without_envelope(), FOLDED_REST, home) is None
    haller.verifier = PassOnAttempt()
    haller.rig.so101.q = np.array(FOLDED_REST)
    seen = {}
    original = haller.runner.start

    def spy(target, task):
        guard_ko = haller.rig.guard.keepout
        seen[task] = (guard_ko is full, getattr(guard_ko, "env_cells", None) is None)
        return original(target, task)

    haller.runner.start = spy
    plan = haller.plan("put the pen in the box").result.plan
    assert haller.execute(plan).status == "succeeded"
    assert seen["home"] == (False, True)  # ramp: station zones only
    assert seen[PUT] == (True, False)  # policy: the full keep-out with the envelope
    assert haller.rig.guard.keepout is full
    assert not any("refused the ramp" in e["msg"] for e in haller.events.recent())


def test_observe_mode_skips_the_ramp(tmp_path):
    """Observe sends nothing, so a ramp could never arrive; run_single_arm skips it there too."""
    cfg = SessionConfig(env="haller", robot="fake", observe=True, policy="fake", planner="stub", verifier="manual",
                        realtime=False, task="pen_chain", vocabulary=short_vocab(tmp_path), max_run_s=60,
                        trial_log=str(tmp_path / "trials.jsonl"), runs_dir=str(tmp_path / "runs"))
    s = Session(cfg)
    try:
        s.verifier = PassOnAttempt()
        starts = []
        original = s.runner.start
        s.runner.start = lambda target, task: starts.append(task) or original(target, task)
        plan = s.plan("put the pen in the box").result.plan
        assert s.execute(plan).status == "succeeded"
        assert starts == [PUT]
    finally:
        s.close()
