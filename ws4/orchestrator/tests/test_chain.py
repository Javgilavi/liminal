"""The final demo chain on the Haller (skills.yaml): robot-right arm opens the box, then the robot-left arm puts the
green pen, the yellow cutter and the red pen in their slots. Both arms are run_single_arm's fake arms; each skill has
its own package on flux.remote's real HTTP server (localhost, token), served by a fake policy. No hardware, no pod.

Each step runs with what ws4/single_arm learned on the real arm (18 Oct): its kit's workspace (skills/<kit>/), the
box-arm start gate before every pen-arm step, the cutter's slow zone, the box arm held for the rest of the run, action
horizon 16, lift stall 25 deg, ARRIVE_TOL and the joint-by-joint ramp fallback."""

import dataclasses
import json
import threading
import time
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from flux.backends import FakeBackend
from flux.contract import OBS_STATE, SO101_CONTRACT, arm_slice
from orchestrator.monitors import LoadMonitor, StallMonitor
from orchestrator.session import Session, SessionConfig, load_trial_log
from orchestrator.skills import DEFAULT_VOCABULARY, SINGLE_ARM, load_vocabulary
from orchestrator.verifier import Verdict

GOAL = ("open the box, put the green pen in the green slot, put the yellow cutter in the yellow slot, "
        "put the red pen in its slot")
CHAIN = ["open_box", "put_green_pen_in_slot", "put_yellow_cutter_in_slot", "put_pen_in_slot"]
# The fixed names on the 8790 demo server (ws4-datasets maps them in /workspace/serve/demo_packages.txt).
PACKAGES = {"open_box": "skill_open_box", "put_green_pen_in_slot": "skill_green_pen",
            "put_yellow_cutter_in_slot": "skill_yellow_cutter", "put_pen_in_slot": "skill_red_pen"}
STRINGS = {"open_box": "open the box by pulling its handle",
           "put_green_pen_in_slot": "put the green pen in the green slot",
           "put_yellow_cutter_in_slot": "put the yellow cutter in the yellow slot",
           "put_pen_in_slot": "put the red pen in its slot in the cardboard box"}
KITS = {"open_box": "box", "put_green_pen_in_slot": "green", "put_yellow_cutter_in_slot": "cutter",
        "put_pen_in_slot": "pen_lid"}
# Box-arm poses holding the lid (single_arm tests/test_skills.py): in the green and cutter takes (the start gate allows
# green, cutter and pen_lid there) and in the first red-pen session (refused for green: envelope voxels inside it).
GREEN_HOLD = [12.13, -13.80, 12.26, 37.54, 29.93, 0.87]
RED_PEN_HOLD = [0.88, 15.12, -5.85, 32.62, 14.64, 0.87]
PEN_HOME = np.array([0, 0, 0, 0, 0, 100.0])


def test_vocabulary_is_the_final_chain():
    v = load_vocabulary(DEFAULT_VOCABULARY)
    assert v.station.arms_available == ("left", "right")
    assert {name: v[name].prompt for name in CHAIN} == STRINGS
    box = v["open_box"]
    assert box.arms == ("right",) and box.other_arm_pose == "tucked" and not box.park_after  # it holds the lid
    assert all(v[name].arms == ("left",) for name in CHAIN[1:])
    assert all(v[name].start_pose == "home" for name in CHAIN)
    assert all(v[n].precheck == "open_box" and not v[n].park_after and not v[n].other_arm_pose for n in CHAIN[1:])
    assert [v[n].finish for n in CHAIN[1:]] == ["home", "home", ""]  # the red pen step ends holding at the slot
    assert "tuck" not in v  # no scripted ramp out of the box at the end
    assert v.station.home_for("left") == (0, 0, 0, 0, 0, 100)
    assert v.station.home_for("right") == (0, 0.8, 2.0, 0.4, 0, 99.7)
    assert set(v.station.poses_for("left")) == set(v.station.poses_for("right")) == {"home", "tucked"}
    assert box.finish == "hold" and {n: v[n].package for n in CHAIN} == PACKAGES


def test_each_chain_skill_runs_on_its_kit():
    """keepout: points at ws4/single_arm/skills/<kit>/workspace.json; the skill.json next to it gives the runner
    flags and the start gate, and its verbatim task is the FLUX prompt."""
    from orchestrator.haller import import_single_arm

    _, KeepOut = import_single_arm()
    v = load_vocabulary(DEFAULT_VOCABULARY)
    for name, kit in KITS.items():
        s = v[name]
        info = json.loads((SINGLE_ARM / "skills" / kit / "skill.json").read_text())
        assert s.keepout == f"skills/{kit}/workspace.json" and s.kit.endswith(f"skills/{kit}/skill.json")
        assert s.prompt == info["task"] and s.runner_args == tuple(info.get("runner_args") or ())
        assert bool(s.start_gate) == (s.arms == ("left",))  # the gate is for the pen-arm skills
    assert v["put_yellow_cutter_in_slot"].runner == {"slow_zone": 0.04, "slow_step": 0.5, "slow_load_sustain": 0.5,
                                                     "slow_load_ticks": 6}
    assert not any(v[n].runner for n in CHAIN if n != "put_yellow_cutter_in_slot")
    ws = {n: KeepOut.load(SINGLE_ARM / v[n].keepout) for n in CHAIN}
    assert ws["open_box"].midline_y == -0.17 and ws["open_box"].side == -1  # box: right-arm midline +0.17 m
    assert ws["put_green_pen_in_slot"].midline_y == -0.23 and ws["put_green_pen_in_slot"].other_clearance == 0.04
    cutter = ws["put_yellow_cutter_in_slot"]
    assert cutter.midline_y == -0.21 and cutter.other_clearance == 0.01  # 1 cm box-arm clearance, with the slow zone
    assert ws["put_pen_in_slot"].torso_margin == 0.014 and ws["put_pen_in_slot"].midline_y == -0.15
    assert all(k.env_cells for k in ws.values())  # each with its demonstrated envelope


def test_a_kit_that_does_not_match_fails_the_vocabulary(tmp_path):
    data = yaml.safe_load(DEFAULT_VOCABULARY.read_text())
    data["skills"]["put_green_pen_in_slot"]["prompt"] = "put the green pen in its slot"  # not the dataset's caption
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="is not the kit's task"):
        load_vocabulary(path)
    data = yaml.safe_load(DEFAULT_VOCABULARY.read_text())
    data["skills"]["put_pen_in_slot"]["keepout"] = "skills/pen_lid/no_such_workspace.json"
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="not found"):
        load_vocabulary(path)
    from orchestrator.skills import parse_runner_args

    with pytest.raises(ValueError, match="does not apply"):
        parse_runner_args(["--slow-zone", "0.04", "--max-step", "3"])


def box_at(obs, pose, tol=1.0):
    return float(np.abs(np.asarray(obs[OBS_STATE])[arm_slice("right")][:5] - np.asarray(pose)[:5]).max()) <= tol


class ScriptedChecks:
    """Verifier stand-in: per skill, a list of verdicts consumed in order (then always True). The lid counts as open
    only once the box arm holds it at ``box_pose`` (where the fake box policy takes it): until then open_box answers
    "not yet" without consuming its script (``waits``)."""

    name, fast = "scripted-test", True

    def __init__(self, script=None, box_pose=GREEN_HOLD):
        self.script = defaultdict(list, {k: list(v) for k, v in (script or {}).items()})
        self.box_pose = box_pose
        self.calls, self.waits = [], 0

    def check(self, step, skill, obs):
        if skill.name == "open_box" and not box_at(obs, self.box_pose):
            self.waits += 1
            return Verdict(False, 0.9, "the lid is still closed", self.name)
        self.calls.append(skill.name)
        ok = self.script[skill.name].pop(0) if self.script[skill.name] else True
        return Verdict(ok, 0.9, "as scripted", self.name)


class ToPose(FakeBackend):
    """Fake policy that takes its arm to ``pose`` (model units = robot units here) at ``step`` deg per command, then
    holds there: the box skill ends with the box arm holding the lid at a recorded pose."""

    def __init__(self, pose, step=2.0, grip_step=5.0):
        super().__init__(SO101_CONTRACT, velocity=0.0, delay_s=0.005)
        self.pose, self.limit = np.asarray(pose, dtype=float), np.r_[np.full(5, step), grip_step]

    def predict(self, request):
        response = super().predict(request)
        cur, frames = np.asarray(request.commands[-1], dtype=float), []
        for _ in range(len(response.commands)):
            cur = cur + np.clip(self.pose - cur, -self.limit, self.limit)
            frames.append(cur.copy())
        return dataclasses.replace(response, commands=np.asarray(frames))


def serve(tmp_path, box_pose=GREEN_HOLD, policies=None, omit=()):
    """flux.remote's HTTP server on localhost with one fake package per chain skill: the box policy goes to
    ``box_pose``, the pen-arm policies hold still unless ``policies`` (skill -> backend) says otherwise; ``omit``:
    skills whose package the server does not serve."""
    from flux.remote import make_server

    backends = {pkg: FakeBackend(SO101_CONTRACT, velocity=0.0, delay_s=0.005) for pkg in PACKAGES.values()}
    backends[PACKAGES["open_box"]] = ToPose(box_pose)
    for skill, backend in (policies or {}).items():
        backends[PACKAGES[skill]] = backend
    for skill in omit:
        del backends[PACKAGES[skill]]
    server = make_server(backends, "127.0.0.1", 0, token="chain-token")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    token = tmp_path / "flux_token"
    token.write_text("chain-token\n")
    return server, f"http://127.0.0.1:{server.server_address[1]}", str(token), backends


@pytest.fixture
def chain_server(tmp_path):
    server, url, token, backends = serve(tmp_path)
    yield url, token, backends
    server.shutdown()
    server.server_close()


def short_chain_vocab(tmp_path, timeout=3.0, replay=None):
    """``replay``: the open_box replay entry, or {skill: entry}."""
    data = yaml.safe_load(DEFAULT_VOCABULARY.read_text())
    for name in CHAIN:
        data["skills"][name]["timeout_s"] = timeout
        data["skills"][name]["verify_every_chunks"] = 1
    if replay is not None:
        for name, entry in (replay.items() if "episode" not in replay else [("open_box", replay)]):
            data["skills"][name]["replay"] = entry
    path = tmp_path / "skills.yaml"
    path.write_text(yaml.safe_dump(data))
    return str(path)


def chain_session(tmp_path, server, token, step_timeout=3.0, replay=None, **kw):
    """The packages come from skills.yaml (the fixed demo names), not from --skill-package."""
    cfg = SessionConfig(env="haller", robot="fake", policy="remote", server=server, token_file=token,
                        planner="stub", verifier="manual", realtime=False,
                        task="final_chain", vocabulary=short_chain_vocab(tmp_path, step_timeout, replay), max_run_s=120,
                        condition="post_training", variant="lora_chain", checkpoint="ws4/chain-loras@test",
                        trial_log=str(tmp_path / "trials.jsonl"), runs_dir=str(tmp_path / "runs"), **kw)
    return Session(cfg)


def keepout_view(s, arm):
    """What the guard enforces on ``arm`` right now: its kit's overrides, envelope on or off, slow zone."""
    ko = s.rig.guard.keepouts[arm]
    return {"midline": ko.midline_y, "clearance": ko.other_clearance, "torso": ko.torso_margin,
            "envelope": bool(ko.env_cells), "slow": dict(s.rig.guard.slow.get(arm) or {})}


def test_final_chain_end_to_end_over_the_policy_server(tmp_path, chain_server):
    server, token, backends = chain_server
    s = chain_session(tmp_path, server, token)
    try:
        assert s.rig.both and sorted(s.rig.arms) == ["left", "right"]
        assert sorted(s.target_backend) == ["left:prim", "left:skill_green_pen", "left:skill_red_pen",
                                            "left:skill_yellow_cutter", "right:prim", "right:skill_open_box"]
        s.verifier = checks = ScriptedChecks()
        plan = s.plan(GOAL).result.plan
        assert [(st.skill, st.arm) for st in plan.steps] == [
            ("open_box", "right"), ("put_green_pen_in_slot", "left"), ("put_yellow_cutter_in_slot", "left"),
            ("put_pen_in_slot", "left")]
        assert [st.depends_on for st in plan.steps] == [[]] + [[plan.steps[i].id] for i in range(3)]
        starts = []
        original = s.runner.start

        views = []

        def spy(target, task):
            joints = {a: x.robot.joints().round(1) for a, x in s.rig.arms.items()}
            starts.append((target, task, joints, list(s.runner.active_targets())))
            views.append((target, keepout_view(s, target.split(":")[0]), s.rig.guard.config))
            return original(target, task)

        s.runner.start = spy
        result = s.execute(plan)
        assert result.status == "succeeded" and (result.steps_done, result.steps_total) == (4, 4)
        assert result.interventions == 0
        # Each policy under its kit's workspace with its envelope (and the cutter's slow zone), each ramp under the
        # same zones without the envelope and without a slow zone (run_single_arm: policy commands only).
        policy = [(t, v) for t, v, _ in views if not t.endswith(":prim")]
        assert [(v["midline"], v["clearance"], v["torso"], v["envelope"]) for _, v in policy] == [
            (-0.17, 0.04, 0.02, True), (-0.23, 0.04, 0.02, True), (-0.21, 0.01, 0.02, True), (-0.15, 0.04, 0.014, True)]
        assert [bool(v["slow"]) for _, v in policy] == [False, False, True, False]
        assert policy[2][1]["slow"] == {"slow_zone": 0.04, "slow_step": 0.5, "slow_load_sustain": 0.5,
                                        "slow_load_ticks": 6}
        ramps = [v for t, v, _ in views if t.endswith(":prim")]
        assert not any(v["envelope"] or v["slow"] for v in ramps)
        pen_ramps = [v for (t, v, _) in views if t == "left:prim"][1:]  # tucked -> home before each pen skill
        assert [(v["midline"], v["clearance"]) for v in pen_ramps] == [(-0.23, 0.04), (-0.21, 0.01), (-0.15, 0.04)]
        # the start gate ran with the box arm's live pose before each pen-arm step, and allowed it
        gate_ok = [e["msg"] for e in s.events.recent(500) if "start gate ok" in e["msg"]]
        assert [m.split("start gate ok for ")[1].split(" ")[0] for m in gate_ok] == CHAIN[1:]
        assert all(f"right arm at {np.round(GREEN_HOLD[:5], 1).tolist()}" in m for m in gate_ok)
        # the box arm holds the lid from open_box on and keeps holding (torque on) when the session closes
        assert s.executor.held == {"right"} and s.rig.holding == {"right"}
        assert not s.rig.torque_off_at_close(s.rig.arms["right"])
        # Box: pen arm tucked, box arm home, open, then the box arm HOLDS (no ramp). Each pen skill: lid precheck, pen
        # arm home, skill; green and yellow come back home under their own policy, the red pen step ends at the slot.
        # No scripted ramp ever leaves the box. One arm moving at a time.
        expected = [("left:prim", "tucked"), ("right:prim", "home"), ("right:skill_open_box", STRINGS["open_box"])]
        for skill in CHAIN[1:]:
            expected += [("left:prim", "home"), (f"left:{PACKAGES[skill]}", STRINGS[skill])]
        assert [(t, task) for t, task, _, _ in starts] == expected
        assert all(active == [] for _, _, _, active in starts)  # nothing else moves when a ramp or skill starts
        poses = s.vocab.station.poses
        box_q = starts[3][2]["right"]  # the box arm when the first pen ramp starts: holding the lid
        for target, _, joints, _ in starts[3:]:
            assert np.abs(joints["right"] - box_q).max() <= 0.5  # it never moves again
            if not target.endswith(":prim"):
                assert np.abs(joints["left"] - poses["left"]["home"]).max() <= 3.0, (target, joints)
        assert np.abs(starts[2][2]["left"] - poses["left"]["tucked"]).max() <= 3.0  # pen arm tucked for the box
        # each package served only its own skill (after one warm-up request per prompt)
        for skill, pkg in PACKAGES.items():
            assert {r.task for r in backends[pkg].requests} == {STRINGS[skill]}
        assert checks.calls.count("open_box") == 4  # its own check, then the lid precheck before each pen skill
        assert [checks.calls.count(skill) for skill in CHAIN[1:]] == [2, 2, 1]  # again once back home; red: once
        assert s.executor.known_pose == {"left": False, "right": False}  # both arms end where their policies left them
        trials = load_trial_log().load(tmp_path / "trials.jsonl")
        assert [t.task for t in trials if t.task != "final_chain"] == CHAIN
        attempts = [t for t in trials if t.task != "final_chain"]
        assert all("action horizon 16" in t.notes and "lift stall 25.0 deg" in t.notes and "prefetch off" in t.notes
                   for t in attempts)
        assert ["start gate ok" in t.notes for t in attempts] == [False, True, True, True]
        # open_box: the tucked pen arm's pose at its start, against the box takes' range (logged, not checked)
        tucked = np.round(s.vocab.station.poses["left"]["tucked"][:5], 1).tolist()
        started = [e["msg"] for e in s.events.recent(500) if "open_box starts with the left arm (tucked)" in e["msg"]]
        assert len(started) == 1 and f"at {tucked}" in started[0] and "0.0 deg outside" in started[0], started
        assert "left arm at start" in attempts[0].notes and not any("arm at start" in t.notes for t in attempts[1:])
        assert "kit runner_args --slow-zone 0.04" in attempts[2].notes
        (chain,) = [t for t in trials if t.task == "final_chain"]
        assert chain.success and chain.steps_done == chain.steps_total == 4
        snap = s.executor.snapshot()
        assert [st["status"] for st in snap["steps"]] == ["done"] * 4
        assert [st["arm"] for st in snap["steps"]] == ["right", "left", "left", "left"]
    finally:
        s.close()


def test_pen_skill_holds_when_the_lid_is_closed(tmp_path, chain_server):
    """Before each pen skill the lid check runs again; a closed lid stops everything in a safe hold before the pen arm
    moves (parent 2026-10-18). Resume checks again."""
    server, token, _ = chain_server
    s = chain_session(tmp_path, server, token)
    try:
        s.verifier = checks = ScriptedChecks({"open_box": [True, False]})  # opened, then found closed
        holds, starts = [], []
        s.rig.hold = lambda reason="": holds.append(reason)
        original = s.runner.start
        s.runner.start = lambda target, task: starts.append((target, task)) or original(target, task)
        plan = s.plan("open the box, put the green pen in the green slot").result.plan
        out = {}
        thread = threading.Thread(target=lambda: out.setdefault("r", s.execute(plan)))
        thread.start()
        deadline = time.time() + 30
        while (s.executor is None or s.executor.state != "estopped") and time.time() < deadline:
            time.sleep(0.01)
        assert s.executor.state == "estopped" and holds and "open_box no longer holds" in holds[0]
        assert starts == [("left:prim", "tucked"), ("right:prim", "home"),
                          ("right:skill_open_box", STRINGS["open_box"])]  # the pen arm never moved
        green = s.executor.runs[plan.steps[1].id]
        assert green.status == "interrupted" and not green.policy_started
        s.release()
        while s.executor.state != "paused" and time.time() < deadline:
            time.sleep(0.01)
        s.resume()  # the lid check runs again (and passes now)
        thread.join(timeout=30)
        assert out["r"].status == "succeeded" and checks.calls.count("open_box") == 3
    finally:
        s.close()


def test_left_arm_keepout_sees_the_right_arm_live(tmp_path, chain_server):
    """Both arms connected: each arm's other_arm zone follows the other arm's measured pose every tick, no bus read."""
    server, token, _ = chain_server
    s = chain_session(tmp_path, server, token)
    try:
        s.runner.step()
        right_q = tuple(float(v) for v in s.rig.arms["right"].robot.last_q)
        left_q = tuple(float(v) for v in s.rig.arms["left"].robot.last_q)
        assert s.rig.guard.keepouts["left"].other_q == right_q
        assert s.rig.guard.keepouts["right"].other_q == left_q
        assert s.rig.refresh_other_arm().note == "live"
    finally:
        s.close()


# ---------------------------------------------------------------------------------------------- imagination panel
def _pack_jpegs(frames):
    import io

    from PIL import Image

    chunks = []
    for f in frames:
        buf = io.BytesIO()
        Image.fromarray(f).save(buf, format="JPEG", quality=95)
        chunks.append(buf.getvalue())
    return np.frombuffer(b"".join(chunks), dtype=np.uint8), [len(c) for c in chunks]


def imagining_backend():
    """FakeBackend that also 'imagines' like ws4/imagine/serving.py in lazy mode: an id in the chunk's response, the
    frames (41 canvases of scene | wrist, offsets +2..+42, frame k filled with grey level 5 * offset) on fetch."""
    import dataclasses

    from flux.backends import FakeBackend
    from flux.contract import SO101_CONTRACT

    class Imagining(FakeBackend):
        store: dict = {}

        def predict(self, request):
            resp = super().predict(request)
            if not request.imagine:
                return resp
            iid = f"im{len(self.store)}"
            offsets = list(range(2, 43))
            blob, lengths = _pack_jpegs([np.full((256, 512, 3), 5 * o, np.uint8) for o in offsets])
            meta = {"id": iid, "mode": request.imagine, "frame_offsets": offsets}
            self.store[iid] = ({**meta, "lengths": {"frames": lengths}}, {"frames": blob})
            return dataclasses.replace(resp, imagination={"meta": meta, "arrays": {}})

        def fetch_imagination(self, iid, wait_s):
            return self.store[iid]

    return Imagining(SO101_CONTRACT, velocity=0.0)


def test_imagination_panel_pairs_imagined_and_actual(tmp_path):
    import io
    import time
    from types import SimpleNamespace

    from flux.backends import PolicyRequest
    from flux.contract import IMG_HEAD, WRIST_KEY
    from flux.remote import RemoteBackend, make_server
    from orchestrator.imagination_panel import ImaginationPanel
    from PIL import Image

    server = make_server({"lora_open_box": imagining_backend()}, "127.0.0.1", 0, token="t")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}"
        backend = RemoteBackend(url, "lora_open_box", token="t", imagine="lazy")
        from flux.contract import SO101_CONTRACT

        c = SO101_CONTRACT
        images = {k: np.zeros((c.history_snapshots, *c.image_hw, 3), np.uint8) for k in c.camera_keys}
        resp = backend.predict(PolicyRequest("open the box by pulling its handle", np.zeros((8, 6)), np.zeros((8, 6)),
                                             images, c.snapshots, target="right:lora_open_box"))
        assert resp.info["imagination_id"] == "im0"
        runner = SimpleNamespace(_targets={"right:lora_open_box": SimpleNamespace(last_response=resp)},
                                 active_targets=lambda: ["right:lora_open_box"])
        panel = ImaginationPanel(runner, {"right:lora_open_box": backend})
        assert panel.state() == {"enabled": True, "ready": False, "fetched": 0, "error": ""}
        frame = np.ones((480, 640, 3), np.uint8)
        obs = {IMG_HEAD: 200 * frame, WRIST_KEY["right"]: 50 * frame, WRIST_KEY["left"]: 0 * frame}
        panel.on_frame(obs)
        deadline = time.time() + 10
        while not panel.state()["ready"] and time.time() < deadline:
            time.sleep(0.02)
        st = panel.state()
        assert st["ready"] and st["arm"] == "right" and st["offset"] == 32 and st["ahead_s"] == pytest.approx(1.07)
        imagined = np.asarray(Image.open(io.BytesIO(panel.frame("imagined"))))
        assert imagined.shape == (256, 512, 3) and abs(int(imagined.mean()) - 160) <= 3  # the +32 frame
        actual = np.asarray(Image.open(io.BytesIO(panel.frame("actual"))))
        assert actual.shape == (256, 512, 3)  # head | the right arm's wrist
        assert abs(int(actual[:, :256].mean()) - 200) <= 3 and abs(int(actual[:, 256:].mean()) - 50) <= 3
        panel.on_frame(obs)  # same chunk: no second fetch
        time.sleep(0.1)
        assert panel.state()["fetched"] == 1
        panel.close()
    finally:
        server.shutdown()
        server.server_close()


def test_session_wires_imagination(tmp_path, chain_server):
    server, token, _ = chain_server
    s = chain_session(tmp_path, server, token, imagine="lazy")
    try:
        assert s.imagination is not None and all(be.imagine == "lazy" for t, be in s.target_backend.items()
                                                 if not t.endswith(":prim"))
        assert s.snapshot()["imagination"]["enabled"] and s.snapshot()["arms"] == ["left", "right"]
        assert s.frame("imagined") is None  # nothing fetched yet
    finally:
        s.close()


def test_cli_builds_the_chain_config():
    import argparse

    from orchestrator.__main__ import _config, _session_args

    p = argparse.ArgumentParser()
    _session_args(p)
    a = p.parse_args(["--env", "haller", "--skill-package", "open_box=lora_open_box", "--skill-package",
                      "put_pen_in_slot = lora_red_pen", "--imagine", "lazy", "--server", "http://127.0.0.1:8790"])
    cfg = _config(a)
    assert cfg.skill_packages == {"open_box": "lora_open_box", "put_pen_in_slot": "lora_red_pen"}
    assert cfg.imagine == "lazy" and cfg.policy == "remote" and cfg.task == "haller_chain"
    assert cfg.right_arm_port == "/dev/haller_arm_left" and cfg.right_robot_id == "haller_left"
    with pytest.raises(SystemExit):
        _config(p.parse_args(["--env", "haller", "--skill-package", "open_box"]))


def drifting_chain(tmp_path, **kw):
    """The chain server, except that the green policy rolls the wrist 30 deg away from home and stays there (inside
    green's envelope: no keep-out refusal involved)."""
    server, url, token, _ = serve(tmp_path, policies={"put_green_pen_in_slot": ToPose(PEN_HOME + [0, 0, 0, 0, 30, 0])})
    return server, chain_session(tmp_path, url, token, **kw)


def run_until_estop(s, plan, deadline_s=60):
    out = {}
    thread = threading.Thread(target=lambda: out.setdefault("r", s.execute(plan)))
    thread.start()
    deadline = time.time() + deadline_s
    while (s.executor is None or s.executor.state != "estopped") and time.time() < deadline:
        time.sleep(0.01)
    return thread, out, deadline


def resume_and_finish(s, thread, out, deadline):
    s.release()
    while s.executor.state != "paused" and time.time() < deadline:
        time.sleep(0.01)
    s.resume()
    thread.join(timeout=60)
    return out["r"]


def test_return_timeout_then_operator_confirms_the_ramp(tmp_path):
    """finish: home: if the policy does not bring the arm home, the step still counts (its check passed) and the arm
    holds; the next ramp of that arm from there asks the operator first (e-stop), Resume confirms it."""
    server, s = drifting_chain(tmp_path, step_timeout=10.0, return_timeout_s=1.0)
    try:
        s.verifier = ScriptedChecks({"put_green_pen_in_slot": [False] * 2})  # passes with the wrist well off home
        holds = []
        s.rig.hold = lambda reason="": holds.append(reason)
        plan = s.plan("open the box, put the green pen in the green slot, put the yellow cutter in the yellow slot")
        thread, out, deadline = run_until_estop(s, plan.result.plan)
        assert s.executor.state == "estopped" and "would ramp" in holds[0] and "left arm" in holds[0]
        green, yellow = (s.executor.runs[st.id] for st in plan.result.plan.steps[1:])
        assert green.status == "done" and green.attempt == 1 and not yellow.policy_started
        assert any("did not bring the left arm home" in e["msg"] for e in s.events.recent())
        assert resume_and_finish(s, thread, out, deadline).status == "succeeded"  # the operator confirmed the ramp
    finally:
        s.close()
        server.shutdown()
        server.server_close()


def test_retry_from_an_unknown_pose_asks_the_operator(tmp_path):
    """A failed attempt leaves the arm where the policy left it: the retry's ramp home waits for the operator."""
    server, s = drifting_chain(tmp_path, step_timeout=2.0)
    try:
        s.verifier = ScriptedChecks({"put_green_pen_in_slot": [False] * 50})
        holds = []
        s.rig.hold = lambda reason="": holds.append(reason)
        plan = s.plan("open the box, put the green pen in the green slot").result.plan
        thread, out, deadline = run_until_estop(s, plan)
        assert s.executor.state == "estopped" and "would ramp" in holds[0]
        assert any("attempt 1 failed" in e["msg"] for e in s.events.recent())
        s.abort()
        thread.join(timeout=30)
        assert out["r"].status == "aborted"
    finally:
        s.close()
        server.shutdown()
        server.server_close()


def test_hold_keeps_the_grip_and_holding_arms_do_not_stall():
    from types import SimpleNamespace

    from orchestrator.haller import freeze_keep_grip
    from orchestrator.monitors import StallMonitor

    sent = []
    robot = SimpleNamespace(slot="right", last_sent=np.array([0, 0, 0, 0, 0, 5.0]),
                            joints=lambda: np.array([1.0, 2, 3, 4, 5, 30.0]),
                            send_joints=lambda q, measure=False: sent.append(np.array(q)))
    freeze_keep_grip(robot)  # joints frozen where they are, the gripper keeps squeezing the lid's tab
    assert np.allclose(sent[0], [1, 2, 3, 4, 5, 5.0])
    driven = {"right": False}
    stall = StallMonitor("right", stall_deg=8.0, stall_ticks=2, active=lambda arm: driven[arm])
    obs = {"observation.state": np.zeros(12)}
    obs["observation.state"][6 + 1] = 20.0  # lift lags 20 deg under the lid's weight
    cmd = np.zeros(12)
    assert all(stall(SimpleNamespace(tick=t, command=cmd), obs) is None for t in range(1, 6))  # holding: exempt
    driven["right"] = True
    assert stall(SimpleNamespace(tick=6, command=cmd), obs) is None and "shoulder_lift" in stall(
        SimpleNamespace(tick=7, command=cmd), obs)


# ---------------------------------------------------------------------------------------------- tonight's runner work
def test_start_gate_refusal_ends_the_chain_with_both_arms_holding(tmp_path):
    """The box skill ends with the box arm holding the lid where green's demos never had it (the first red-pen
    session's hold: envelope voxels inside the box arm). The gate refuses before the pen arm moves; the chain ends,
    failed, with the reason in the HMI's events, no retry, no replan, and both arms keep holding."""
    server, url, token, backends = serve(tmp_path, box_pose=RED_PEN_HOLD)
    s = chain_session(tmp_path, url, token)
    try:
        s.verifier = checks = ScriptedChecks(box_pose=RED_PEN_HOLD)
        replans, holds, starts = [], [], []
        s._on_replan = lambda state: replans.append(state)
        s.rig.hold = lambda reason="": holds.append(reason)
        original = s.runner.start
        s.runner.start = lambda target, task: starts.append((target, task)) or original(target, task)
        plan = s.plan(GOAL).result.plan
        result = s.execute(plan)
        assert result.status == "failed" and result.steps_done == 1 and result.interventions == 0
        assert "start gate REFUSED put_green_pen_in_slot" in result.failure and "refused" in result.failure
        assert starts == [("left:prim", "tucked"), ("right:prim", "home"),
                          ("right:skill_open_box", STRINGS["open_box"])]  # the pen arm never left its tuck
        green = s.executor.runs[plan.steps[1].id]
        assert green.status == "failed" and green.attempt == 1 and not green.policy_started
        assert all(s.executor.runs[st.id].status == "pending" for st in plan.steps[2:])
        assert checks.calls == ["open_box"] and not replans and not holds  # no lid precheck, no e-stop
        msgs = [e["msg"] for e in s.events.recent(500)]
        assert any("REFUSED before it moved" in m and "both arms hold where they are" in m and "gate.py report" in m
                   for m in msgs)
        snap = s.snapshot()
        assert snap["run"]["state"] == "failed" and "start gate REFUSED" in snap["run"]["failure"]
        # both arms hold: nothing drives them and each keeps its last command, the box arm at the lid
        assert s.runner.active_targets() == [] and not s.runner.guard.estopped
        for arm, x in s.rig.arms.items():
            assert np.allclose(x.robot.last_sent[:5], x.robot.joints()[:5]), arm
        assert box_at({OBS_STATE: s.robot.get_observation()[OBS_STATE]}, RED_PEN_HOLD)
        assert np.abs(s.rig.arms["left"].robot.joints() - s.vocab.station.poses["left"]["tucked"]).max() <= 3.0
        assert s.rig.holding == {"right"} and not s.rig.torque_off_at_close(s.rig.arms["right"])
        trials = load_trial_log().load(tmp_path / "trials.jsonl")
        assert [t.task for t in trials] == ["open_box", "final_chain"]  # no attempt line for the refused step
        assert "start gate REFUSED" in trials[-1].failure
        assert {r.task for r in backends[PACKAGES["put_green_pen_in_slot"]].requests} == {STRINGS[CHAIN[1]]}
        assert len(backends[PACKAGES["put_green_pen_in_slot"]].requests) == 1  # its warm-up only
    finally:
        s.close()
        server.shutdown()
        server.server_close()


def test_cutter_slow_zone_end_to_end(tmp_path, monkeypatch):
    """The cutter's kit records --slow-zone 0.04 --slow-step 0.5 --slow-load-sustain 0.5 --slow-load-ticks 6: with
    the pen arm within 4 cm of the box arm its joints move at most 0.5 deg per tick under the cutter policy and the
    load freeze trips at 50 % for 6 ticks; green's identical policy is not slowed and nothing applies to ramps."""
    pan10 = ToPose(PEN_HOME + [10, 0, 0, 0, 0, 0])  # 2 deg per command toward pan 10
    server, url, token, _ = serve(tmp_path, policies={"put_yellow_cutter_in_slot": pan10,
                                                      "put_green_pen_in_slot": ToPose(PEN_HOME + [10, 0, 0, 0, 0, 0])})
    s = chain_session(tmp_path, url, token, return_timeout_s=1.0)
    try:
        s.verifier = ScriptedChecks()
        cutter = s.vocab["put_yellow_cutter_in_slot"]
        ko = s.rig.skill_keepout("left", cutter)
        # as if the box arm held the lid 3 cm from the pen arm (capsule surfaces): above the 1 cm clearance, so the
        # keep-out itself does not refuse, inside the 4 cm slow zone
        monkeypatch.setattr(ko, "other_arm_gap", lambda q: np.full(np.shape(q)[:-1], 0.03))
        steps = defaultdict(list)
        send = s.rig.robot.send_action
        last = {}

        def spy(command):
            owner = s.runner._owner.get("left")
            if owner and "left" in last:
                steps[owner].append(float(np.abs(command[arm_slice("left")][:5] - last["left"][:5]).max()))
            last["left"] = np.asarray(command, float)[arm_slice("left")].copy()
            return send(command)

        s.rig.robot.send_action = spy
        result = s.execute(s.plan("open the box, put the green pen in the green slot, put the yellow cutter in the "
                                  "yellow slot").result.plan)
        assert result.status == "succeeded", result.failure
        assert max(steps["left:skill_yellow_cutter"]) <= 0.5 + 1e-9 < max(steps["left:skill_green_pen"])
        assert max(steps["left:prim"]) > 0.5 + 1e-9  # ramps keep their speed (run_single_arm: policy only)
        assert s.rig.guard.slow_ticks["left"] > 0 and s.rig.guard.slow_ticks["right"] == 0
        # the load freeze in the slow zone: 55 % of the cap trips there after 6 ticks, never outside (70 %, 9 ticks)
        s.rig.set_ramp("left", False, cutter)
        s.rig.arms["left"].so101.load = np.full(5, 550.0)  # fake cap: 100 % = 1000
        load = next(m for m in s.rig.monitors if isinstance(m, LoadMonitor) and m.slot == "left")
        cmd = s.robot.get_observation()[OBS_STATE]
        s.rig.guard.filter(cmd, cmd)  # in the slow zone now
        reasons = [load(SimpleNamespace(tick=t, command=cmd), {}) for t in range(1, 7)]
        assert reasons[:5] == [None] * 5 and "in the slow zone" in reasons[5]
        s.rig.set_ramp("left", False, s.vocab["put_green_pen_in_slot"])  # green: no slow zone
        s.rig.guard.filter(cmd, cmd)
        assert not s.rig.guard.in_slow_zone["left"]
        assert all(load(SimpleNamespace(tick=t, command=cmd), {}) is None for t in range(10, 40))
    finally:
        s.close()
        server.shutdown()
        server.server_close()


def test_runner_settings_of_the_first_real_inserts(tmp_path, chain_server, monkeypatch):
    """Action horizon 16, shoulder-lift stall 25 deg, run_single_arm's ARRIVE_TOL (lift 8) for every ramp, chunk
    prefetch wired but off; the CLI flags and WS4_* variables override them as in the runner."""
    server, token, _ = chain_server
    for name in ("WS4_ACTION_STEPS", "WS4_LIFT_STALL_DEG", "WS4_PREFETCH_TICKS"):
        monkeypatch.delenv(name, raising=False)
    s = chain_session(tmp_path, server, token)
    try:
        assert (s.cfg.action_steps, s.cfg.lift_stall_deg, s.cfg.prefetch_ticks) == (16, 25.0, 0)
        assert s.runner.prefetch_ticks == 0 and s.cfg.max_step == 1.0
        horizons = {t: ts.contract.n_action_steps for t, ts in s.runner._targets.items()}
        assert horizons == {t: (32 if t.endswith(":prim") else 16) for t in s.target_backend}
        stalls = [m for m in s.rig.monitors if isinstance(m, StallMonitor)]
        assert len(stalls) == 2 and all(m.stall_deg.tolist() == [8, 25, 8, 8, 8] for m in stalls)
        assert all(np.allclose(s.target_backend[f"{a}:prim"].arrive_tol, s.rig.sar.ARRIVE_TOL) for a in s.rig.arms)
        assert s.rig.sar.ARRIVE_TOL.tolist() == [6, 8, 6, 6, 6]
        prim = s.target_backend["left:prim"]  # a lift 7 deg short of home counts as arrived, 9 does not
        assert prim.reached("home", PEN_HOME, PEN_HOME + [0, -7, 0, 0, 0, 0])
        assert not prim.reached("home", PEN_HOME, PEN_HOME + [0, -9, 0, 0, 0, 0])
        assert "action horizon 16, lift stall 25.0 deg (others 8.0), prefetch off" in s.runner_settings()
    finally:
        s.close()
    monkeypatch.setenv("WS4_ACTION_STEPS", "8")
    monkeypatch.setenv("WS4_LIFT_STALL_DEG", "20")
    s = chain_session(tmp_path, server, token, prefetch_ticks=8)  # explicit: wins over WS4_PREFETCH_TICKS
    try:
        assert (s.cfg.action_steps, s.cfg.lift_stall_deg, s.runner.prefetch_ticks) == (8, 20.0, 8)
        s.verifier = ScriptedChecks()
        assert s.execute(s.plan("open the box").result.plan).status == "succeeded"  # prefetch through the gated runner
    finally:
        s.close()


class PanFirst:
    """Keep-out stand-in that refuses any lift move while the pan is still out (the torso zone after an insert,
    18 Oct; run_single_arm's own test)."""

    other_q, env_cells, calibrated = None, None, True

    def refuse(self, nxt, cur):
        bad = abs(nxt[1] - cur[1]) > 1e-6 and abs(cur[0]) > 5
        return bad, "torso" if bad else ""

    def without_envelope(self):
        return self


class NoLift(PanFirst):
    def refuse(self, nxt, cur):
        bad = abs(nxt[1] - cur[1]) > 1e-6
        return bad, "torso" if bad else ""


def home_step_session(tmp_path, server, token, keepout):
    s = chain_session(tmp_path, server, token, fake_start=(24.0, 10.0, 50.0, 80.0, 35.0, 2.0))
    s.rig.arms["left"].station_keepout = keepout
    s.rig.guard.keepouts["left"] = keepout
    plan = s.load_plan_json({"steps": [{"id": "s1", "arm": "left", "skill": "home"}]}, "home").result.plan
    return s, plan


def test_ramp_goes_joint_by_joint_after_a_keepout_refusal(tmp_path, chain_server):
    """run_single_arm.move_to's fallback: the straight ramp home is refused, the arm holds, then it moves one joint
    at a time in the first order the keep-out accepts (pan first here), and arrives."""
    server, token, _ = chain_server
    s, plan = home_step_session(tmp_path, server, token, PanFirst())
    try:
        holds = []
        s.rig.hold = lambda reason="": holds.append(reason)
        result = s.execute(plan)
        assert result.status == "succeeded" and not holds, result.failure
        msgs = [e["msg"] for e in s.events.recent(500)]
        assert any("refused the left arm's ramp to its home pose; going joint by joint instead: shoulder_pan" in m
                   for m in msgs)
        assert np.allclose(s.rig.arms["left"].robot.joints(), PEN_HOME)
        assert s.target_backend["left:prim"].fallback is None  # the next ramp starts straight again
    finally:
        s.close()


def test_ramp_without_a_joint_order_stops_and_holds(tmp_path, chain_server):
    server, token, _ = chain_server
    s, plan = home_step_session(tmp_path, server, token, NoLift())
    try:
        holds = []
        s.rig.hold = lambda reason="": holds.append(reason)
        thread, out, _ = run_until_estop(s, plan)
        assert s.executor.state == "estopped" and "refused the left arm's ramp" in holds[0]
        assert any("no joint-by-joint order clears it either" in e["msg"] for e in s.events.recent(500))
        s.abort()
        thread.join(timeout=30)
        assert out["r"].status == "aborted"
    finally:
        s.close()


def test_box_arm_handed_over_holding_is_never_driven(tmp_path, chain_server):
    """--hold-arm right (single_arm README "Handover from the HMI"): the box arm comes in holding the lid; it is held
    where it is (the runner's first command is its measured pose), never commanded elsewhere, and keeps its torque at
    close even at its launch pose. A pen skill alone runs against it (gate and lid precheck with its live pose); a
    step that would drive it is refused before anything moves."""
    server, token, _ = chain_server
    plain = chain_session(tmp_path, server, token, right_fake_start=tuple(GREEN_HOLD))
    try:
        assert plain.rig.torque_off_at_close(plain.rig.arms["right"])  # without --hold-arm: launch pose, torque off
    finally:
        plain.close()
    s = chain_session(tmp_path, server, token, right_fake_start=tuple(GREEN_HOLD), hold_arms=("right",))
    try:
        s.verifier = checks = ScriptedChecks()
        sent = []
        send = s.rig.robot.send_action
        s.rig.robot.send_action = lambda command: sent.append(np.array(command)) or send(command)
        result = s.execute(s.plan("put the green pen in the green slot").result.plan)
        assert result.status == "succeeded", result.failure
        assert checks.calls == ["open_box", "put_green_pen_in_slot", "put_green_pen_in_slot"]  # lid precheck first
        assert sent and all(np.allclose(c[arm_slice("right")], GREEN_HOLD) for c in sent)  # never moved
        assert s.rig.guard.keepouts["right"] is None  # no keep-out on its own constant command (OtherArmHolder)
        right = slice(6, 12)
        assert np.allclose(s.rig.guard.config.lower[right], s.rig.arms["right"].ramp_limits.lower[right])
        assert any("right arm handed over holding (--hold-arm): held at" in e["msg"] for e in s.events.recent(500))
        assert not s.rig.torque_off_at_close(s.rig.arms["right"])
        assert "held from launch: right" in s.runner_settings()
        refused = s.execute(s.plan("open the box").result.plan)
        assert refused.status == "failed" and "right arm keeps holding" in refused.failure
        assert all(np.allclose(c[arm_slice("right")], GREEN_HOLD) for c in sent)
    finally:
        s.close()


def test_finish_hold_keeps_the_box_arm_out_of_later_steps(tmp_path, chain_server):
    """After open_box passed, no step of the same run may drive the box arm (e.g. a planned home ramp)."""
    server, token, _ = chain_server
    s = chain_session(tmp_path, server, token)
    try:
        s.verifier = ScriptedChecks()
        plan = s.load_plan_json({"steps": [{"id": "s1", "arm": "right", "skill": "open_box"},
                                           {"id": "s2", "arm": "right", "skill": "home", "depends_on": ["s1"]}]},
                                "open the box").result.plan
        result = s.execute(plan)
        assert result.status == "failed" and "s2 (home) refused: the right arm keeps holding" in result.failure
        assert box_at(s.robot.get_observation(), GREEN_HOLD)
    finally:
        s.close()


def test_next_run_asks_before_ramping_an_arm_a_policy_left(tmp_path, chain_server):
    """Skills rehearsed one run at a time in one HMI session: after open_box the box arm holds the lid where its policy
    left it, so a second "open the box" must not ramp it home on its own (it would pull it out under the lid)."""
    server, token, _ = chain_server
    s = chain_session(tmp_path, server, token)
    try:
        s.verifier = ScriptedChecks()
        assert s.execute(s.plan("open the box").result.plan).status == "succeeded"
        assert s.known_pose == {"left": True, "right": False}
        holds = []
        s.rig.hold = lambda reason="": holds.append(reason)
        thread, out, _ = run_until_estop(s, s.plan("open the box").result.plan)
        assert s.executor.state == "estopped" and "right arm would ramp" in holds[0]
        assert box_at(s.robot.get_observation(), GREEN_HOLD)  # it did not move
        s.abort()
        thread.join(timeout=30)
    finally:
        s.close()


def test_cli_wires_the_runner_settings_and_the_handover():
    import argparse

    from orchestrator.__main__ import _config, _session_args

    p = argparse.ArgumentParser()
    _session_args(p)
    cfg = _config(p.parse_args(["--env", "haller", "--hold-arm", "right", "--action-steps", "12",
                                "--lift-stall-deg", "20", "--prefetch-ticks", "16"]))
    assert cfg.hold_arms == ("right",) and (cfg.action_steps, cfg.lift_stall_deg, cfg.prefetch_ticks) == (12, 20, 16)
    assert cfg.server == "http://127.0.0.1:8790" and cfg.max_run_s == 1200 and cfg.skill_packages == {}
    cfg = _config(p.parse_args(["--env", "haller"]))
    assert (cfg.action_steps, cfg.lift_stall_deg, cfg.prefetch_ticks, cfg.hold_arms) == (None, None, None, ())
    sim = _config(p.parse_args([]))
    assert sim.server == "http://127.0.0.1:8765" and sim.max_run_s == 600


# ---------------------------------------------------------------------------------------------- box fallback: replay
def write_box_take(root, episode, take, columns="left"):
    """A LeRobot v3 parquet with one take: the box arm's actions in the left_* columns, the pen arm tucked (or, with
    ``columns="right"``, a pen-arm take with the box arm holding the lid)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if columns == "left":
        rows = np.hstack([np.asarray(take, float), np.tile([3, -83.6, 81.9, 82.8, -10.2, 0.9], (len(take), 1))])
    else:
        rows = np.hstack([np.tile(GREEN_HOLD, (len(take), 1)), np.asarray(take, float)])
    (root / "data" / "chunk-000").mkdir(parents=True)
    pq.write_table(pa.table({"episode_index": pa.array([episode] * len(rows), pa.int64()),
                             "frame_index": pa.array(range(len(rows)), pa.int64()),
                             "action": pa.array(rows.tolist(), pa.list_(pa.float32()))}),
                   root / "data" / "chunk-000" / "file-000.parquet")
    return root


def box_take():
    """From the box home out to the hold the start gate allows, at up to 4 deg per frame (faster than the guard)."""
    home, hold = np.array([0, 0, 0, 0, 0, 100.0]), np.array(GREEN_HOLD)
    return np.vstack([home + (hold - home) * k / 12 for k in range(13)] + [hold] * 8)


def test_box_replay_runs_the_chain_and_is_never_called_flux(tmp_path):
    """--skill-backend open_box=replay: the box step replays a recorded take through the same stack (1 deg/tick,
    the box kit's keep-out, the hold), checked once the take has ended; the box package is not even served. It is
    labelled a scripted replay in the events, the HMI state and the trial log (condition "scripted")."""
    from orchestrator.haller import import_single_arm

    import_single_arm()
    import replay as replay_module

    root = write_box_take(tmp_path / "box", 7, box_take())
    server, url, token, backends = serve(tmp_path, omit=("open_box",))
    s = chain_session(tmp_path, url, token, skill_backends={"open_box": "replay"}, prefetch_ticks=8,
                      replay={"episode": 7, "root": str(root), "columns": "left"})
    try:
        box = s.vocab["open_box"]
        assert box.backend == "replay" and box.driver == replay_module.label(7) == "scripted replay of recorded demo 7"
        assert s.cfg.prefetch_ticks == 0 and s.runner.prefetch_ticks == 0  # forced off: it would skip recorded steps
        assert any("chunk prefetch off (was 8)" in e["msg"] for e in s.events.recent(200))
        assert "right:replay_open_box" in s.target_backend and "right:skill_open_box" not in s.target_backend
        s.verifier = checks = ScriptedChecks()
        right = []
        send = s.rig.robot.send_action
        s.rig.robot.send_action = lambda c: right.append((s.runner._owner.get("right"), np.array(c[6:12]))) or send(c)
        snaps = []
        original = s.runner.start
        s.runner.start = lambda target, task: snaps.append(s.executor.snapshot()) or original(target, task)
        result = s.execute(s.plan(GOAL).result.plan)
        assert result.status == "succeeded" and (result.steps_done, result.steps_total) == (4, 4), result.failure
        assert checks.waits == 0 and checks.calls.count("open_box") == 4  # checked once, at its end; 3 prechecks
        replayed = np.array([c for owner, c in right if owner == "right:replay_open_box"])
        assert np.abs(np.diff(replayed[:, :5], axis=0)).max() <= 1.0 + 1e-9  # the take time-warped onto 1 deg/tick
        assert np.allclose(s.rig.arms["right"].robot.joints()[:5], GREEN_HOLD[:5], atol=1e-6)  # holds at its end
        assert s.rig.holding == {"right"}
        msgs = [e["msg"] for e in s.events.recent(800)]
        start = [m for m in msgs if "start on right arm" in m]
        assert start and all("SCRIPTED REPLAY OF RECORDED DEMO 7, NOT FLUX" in m and "FLUX prompt" not in m
                             for m in start)
        assert any("open_box: scripted replay of recorded demo 7 (not FLUX)" in m for m in msgs)  # runner settings
        card = [st for snap in snaps for st in snap["steps"] if st["skill"] == "open_box"]
        assert card and all(st["driver"] == "scripted replay of recorded demo 7" for st in card)
        assert {v["name"]: v["driver"] for v in s.snapshot()["vocabulary"]}["open_box"] == box.driver
        trials = load_trial_log().load(tmp_path / "trials.jsonl")
        assert [t.task for t in trials] == CHAIN + ["final_chain"]
        box_row = trials[0]  # its own condition and variant: never mixed with the FLUX rows
        assert (box_row.condition, box_row.variant) == ("scripted", "scripted-replay-ep7") and box_row.success
        assert box_row.checkpoint.startswith("scripted replay of recorded demo 7 (") and "left_* columns" in \
            box_row.checkpoint
        assert all(t.condition == "post_training" and t.variant == "lora_chain" for t in trials[1:])
        chain = trials[-1]
        assert "NOT FLUX: open_box: scripted replay of recorded demo 7" in chain.notes
        assert chain.checkpoint == "ws4/chain-loras@test (open_box: scripted replay of recorded demo 7, not FLUX)"
        assert PACKAGES["open_box"] not in backends
    finally:
        s.close()
        server.shutdown()
        server.server_close()


def test_replay_backend_needs_a_replay_entry(tmp_path, chain_server):
    server, token, _ = chain_server
    with pytest.raises(ValueError, match="skill-backend put_green_pen_in_slot=replay"):
        chain_session(tmp_path, server, token, skill_backends={"put_green_pen_in_slot": "replay"})
    import argparse

    from orchestrator.__main__ import _config, _session_args

    p = argparse.ArgumentParser()
    _session_args(p)
    cfg = _config(p.parse_args(["--env", "haller", "--skill-backend", "open_box=replay", "--replay-episode", "40"]))
    assert cfg.skill_backends == {"open_box": "replay"} and cfg.replay_episodes == {"*": 40}
    cfg = _config(p.parse_args(["--env", "haller", "--replay-episode", "open_box=72", "--replay-episode",
                                "put_yellow_cutter_in_slot = 67"]))
    assert cfg.replay_episodes == {"open_box": 72, "put_yellow_cutter_in_slot": 67}
    s = chain_session(tmp_path, server, token, replay_episodes={"open_box": 72})
    try:  # one skill's take only: the cutter keeps its own
        episodes = {n: s.vocab[n].replay["episode"] for n in ("open_box", "put_yellow_cutter_in_slot")}
        assert episodes == {"open_box": 72, "put_yellow_cutter_in_slot": 67}
    finally:
        s.close()
    with pytest.raises(ValueError, match="has no replay entry"):
        chain_session(tmp_path, server, token, replay_episodes={"put_green_pen_in_slot": 3})
    v = load_vocabulary(DEFAULT_VOCABULARY)
    assert v["open_box"].backend == "flux" and v["open_box"].replay["episode"] == 72  # FLUX unless asked


def test_cutter_replay_in_the_chain_slow_near_the_box_arm(tmp_path, monkeypatch):
    """--skill-backend put_yellow_cutter_in_slot=replay (the last-resort cutter fallback): its recorded take on the pen
    arm, under the cutter kit's keep-out and slow zone with the box arm held, warped at the slow step where the box
    arm is within the slow zone (faked at 3 cm here), so the 0.5 deg/tick cap never clips it; it ends home, its row
    is condition scripted, and the events name the take's first head frame for placing the cutter."""
    from orchestrator.haller import import_single_arm

    _, KeepOut = import_single_arm()
    real_gap = KeepOut.other_arm_gap

    def gap(self, q):  # only the cutter's keep-out (1 cm clearance) sees the box arm 3 cm away
        if self.other_clearance == 0.01 and self.side == 1:
            return np.full(np.shape(q)[:-1], 0.03)
        return real_gap(self, q)

    monkeypatch.setattr(KeepOut, "other_arm_gap", gap)
    home = PEN_HOME
    take = np.vstack([home + [k, 0, 0, 0, 0, 0] for k in range(0, 11, 2)] + [home + [10, 0, 0, 0, 0, 0]] * 3
                     + [home + [k, 0, 0, 0, 0, 0] for k in range(10, -1, -2)])
    root = write_box_take(tmp_path / "cutter", 67, take, columns="right")
    server, url, token, _ = serve(tmp_path, omit=("put_yellow_cutter_in_slot",))
    s = chain_session(tmp_path, url, token, skill_backends={"put_yellow_cutter_in_slot": "replay"},
                      replay={"put_yellow_cutter_in_slot": {"episode": 67, "root": str(root), "columns": "right"}})
    try:
        cutter_target = "left:replay_put_yellow_cutter_in_slot"
        assert s.target_backend[cutter_target].step_at is not None
        msgs = [e["msg"] for e in s.events.recent(200)]
        assert any("put_yellow_cutter_in_slot: scripted replay of recorded demo 67, not FLUX; place the objects" in m
                   and m.endswith("skills/cutter/layouts/P3_ep067_head.jpg (its first head frame)") for m in msgs)
        s.verifier = checks = ScriptedChecks()
        sent, clipped = [], []
        send = s.rig.robot.send_action
        s.rig.robot.send_action = lambda c: sent.append((s.runner._owner.get("left"), np.array(c[:6]))) or send(c)
        step = s.runner.step

        def spy_step():
            report = step()
            if s.runner._owner.get("left") == cutter_target and report.guard_intervened:
                clipped.append(report.tick)
            return report

        s.runner.step = spy_step
        result = s.execute(s.plan("open the box, put the green pen in the green slot, put the yellow cutter in the "
                                  "yellow slot").result.plan)
        assert result.status == "succeeded", result.failure
        cut = np.array([c for owner, c in sent if owner == cutter_target])
        assert len(cut) and np.abs(np.diff(cut[:, :5], axis=0)).max() <= 0.5 + 1e-9 and not clipped
        assert s.rig.guard.slow_ticks["left"] > 0
        assert checks.calls.count("put_yellow_cutter_in_slot") == 2  # once at the take's end, once back home
        trials = load_trial_log().load(tmp_path / "trials.jsonl")
        row = next(t for t in trials if t.task == "put_yellow_cutter_in_slot")
        assert (row.condition, row.variant) == ("scripted", "scripted-replay-ep67")
        green = next(t for t in trials if t.task == "put_green_pen_in_slot")
        assert green.condition == "post_training"  # the pen skills stay FLUX
    finally:
        s.close()
        server.shutdown()
        server.server_close()


@pytest.mark.parametrize("recheck, status", [((True, 0.6), "succeeded"), ((False, 0.3), "succeeded"),
                                             ((False, 0.9), "failed")])
def test_unclear_recheck_after_home_keeps_the_pass(tmp_path, chain_server, recheck, status):
    """finish: home re-checks once the policy is back home. Live 18 Oct the green insert passed at 0.70, then the
    re-check said "upright in the green holder, seating not fully visible" at 0.60 and failed it. Now only a confident
    "not done" fails the step; an unclear re-check keeps the pass, and the trial row lists both verdicts."""
    server, token, _ = chain_server
    s = chain_session(tmp_path, server, token)
    try:
        checks = ScriptedChecks()
        green = iter([Verdict(True, 0.7, "upright in the green holder", "test"),
                      Verdict(*recheck, "seating not fully visible", "test")]
                     + [Verdict(False, 0.9, "not in the slot", "test")] * 200)  # a retry after a contrary re-check
        base = checks.check
        checks.check = lambda step, skill, obs: (next(green) if skill.name == "put_green_pen_in_slot"
                                                 else base(step, skill, obs))
        s.verifier = checks
        result = s.execute(s.plan("open the box, put the green pen in the green slot").result.plan)
        assert result.status == status, result.failure
        trials = load_trial_log().load(tmp_path / "trials.jsonl")
        green_row = [t for t in trials if t.task == "put_green_pen_in_slot"][0]
        assert f"True 0.70, {recheck[0]} {recheck[1]:.2f}; last check" in green_row.notes  # pass, then re-check
        assert green_row.success == (status == "succeeded")
    finally:
        s.close()
