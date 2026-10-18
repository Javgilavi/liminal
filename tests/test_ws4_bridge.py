"""The liminal tree on the Haller through the ws4 bridge, on ws4's fake arms (run_single_arm.FakeSO101) and a fake
policy behind flux.remote's real HTTP server: no serial port, camera, GPU or network beyond localhost."""

import asyncio
import dataclasses
import threading
import time
from pathlib import Path

import pytest
import yaml

np = pytest.importorskip("numpy")
pytest.importorskip("PIL")

from liminal.app import build_context  # noqa: E402
from liminal.hardware.runtime import connected  # noqa: E402
from liminal.orchestration import load_task, run_task  # noqa: E402
from liminal.perception.config import load_perception_config  # noqa: E402
from liminal.safety.supervisor import EmergencyStop  # noqa: E402
from liminal.ws4_bridge import import_orchestrator  # noqa: E402

ROBOT = Path("configs/robot/haller_ws4.yaml")
POLICY = Path("configs/policy/haller_ws4.yaml")
TASK = Path("configs/tasks/assemble_components_haller.yaml")
PERCEPTION = Path("configs/perception/haller_haiku.yaml")
CHAIN = [("open_box", "right"), ("put_green_pen_in_slot", "left"), ("put_yellow_cutter_in_slot", "left"),
         ("put_pen_in_slot", "left")]
PACKAGES = {"open_box": "skill_open_box", "put_green_pen_in_slot": "skill_green_pen",
            "put_yellow_cutter_in_slot": "skill_yellow_cutter", "put_pen_in_slot": "skill_red_pen"}
# The box arm holding the lid as in the green and cutter takes: the start gate allows every pen skill there.
GREEN_HOLD = [12.13, -13.80, 12.26, 37.54, 29.93, 0.87]


@pytest.fixture(autouse=True)
def no_api_key(monkeypatch, tmp_path):
    """Offline: no Claude call may leave this process (as ws4/orchestrator/tests/conftest.py)."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY_FILE", str(tmp_path / "no-key.env"))


@pytest.fixture(scope="module")
def ws4():
    import_orchestrator()
    from flux.backends import FakeBackend
    from flux.contract import OBS_STATE, SO101_CONTRACT, arm_slice
    from orchestrator.skills import DEFAULT_VOCABULARY
    from orchestrator.verifier import Verdict

    class ToPose(FakeBackend):
        """Takes its arm to ``pose`` and holds there (the box policy ends holding the lid)."""

        def __init__(self, pose):
            super().__init__(SO101_CONTRACT, velocity=0.0, delay_s=0.005)
            self.pose = np.asarray(pose, dtype=float)

        def predict(self, request):
            response = super().predict(request)
            cur, frames = np.asarray(request.commands[-1], dtype=float), []
            for _ in range(len(response.commands)):
                cur = cur + np.clip(self.pose - cur, -2.0, 2.0)
                frames.append(cur.copy())
            return dataclasses.replace(response, commands=np.asarray(frames))

    class Checks:
        """ws4's step verifier: the lid is open once the box arm holds it at GREEN_HOLD; pen steps pass."""

        name, fast = "scripted-test", True

        def __init__(self):
            self.calls = []

        def check(self, step, skill, obs):
            self.calls.append(skill.name)
            if skill.name == "open_box":
                box = np.asarray(obs[OBS_STATE])[arm_slice("right")][:5]
                return Verdict(bool(np.abs(box - GREEN_HOLD[:5]).max() <= 1.0), 0.9, "scripted", self.name)
            return Verdict(True, 0.9, "scripted", self.name)

    return {"FakeBackend": FakeBackend, "ToPose": ToPose, "Checks": Checks, "SO101": SO101_CONTRACT,
            "vocabulary": DEFAULT_VOCABULARY}


@pytest.fixture
def policy_server(ws4, tmp_path):
    from flux.remote import make_server

    backends = {pkg: ws4["FakeBackend"](ws4["SO101"], velocity=0.0, delay_s=0.005) for pkg in PACKAGES.values()}
    backends["skill_open_box"] = ws4["ToPose"](GREEN_HOLD)
    server = make_server(backends, "127.0.0.1", 0, token="bridge-token")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    token = tmp_path / "flux_token"
    token.write_text("bridge-token\n")
    yield f"http://127.0.0.1:{server.server_address[1]}", str(token), backends
    server.shutdown()
    server.server_close()


def robot_config(ws4, tmp_path, server, token, **session):
    """haller_ws4.yaml with ws4's fake arms, the local policy server and short step timeouts."""
    vocab = yaml.safe_load(ws4["vocabulary"].read_text())
    for name, _ in CHAIN:
        vocab["skills"][name].update(timeout_s=3.0, verify_every_chunks=1)
    (tmp_path / "skills.yaml").write_text(yaml.safe_dump(vocab))
    cfg = yaml.safe_load(ROBOT.read_text())
    cfg["ws4"]["hmi_port"] = 0
    cfg["ws4"]["session"].update({"robot": "fake", "server": server, "token_file": token, "verifier": "manual",
                                  "skill_backends": {},  # FLUX for every skill unless a test asks for the replay
                                  "realtime": False, "vocabulary": str(tmp_path / "skills.yaml"),
                                  "runs_dir": str(tmp_path / "runs"), "trial_log": str(tmp_path / "trials.jsonl"),
                                  "max_run_s": 120, **session})
    return cfg


class SceneFromWs4:
    """The tree's perception on the fake rig: the lid is open while ws4 holds the box arm, and an item is in its slot
    once its ws4 step succeeded. Answers through ``inspect_request`` like the ws4_haiku model."""

    ITEMS = {"green_pen_in_slot": "put_green_pen_in_slot", "yellow_boxcutter_in_slot": "put_yellow_cutter_in_slot",
             "red_pen_in_slot": "put_pen_in_slot"}

    def __init__(self, bridge):
        self.bridge, self.done, self.requests = bridge, set(), []

    def truth(self, predicate):
        if predicate == "lid_open":
            return self.bridge.session is not None and self.bridge.holding("right")
        if predicate == "components_assembled":
            return all(self.truth(p) for p in self.ITEMS)
        return self.ITEMS[predicate] in self.done

    async def inspect_request(self, request, images):
        self.requests.append(request.predicates)
        assert len(images) == len(request.cameras) and all(getattr(i, "ndim", 0) == 3 for i in images)
        return {"predicates": {p: {"value": self.truth(p), "evidence": "fake rig"} for p in request.predicates}}

    async def inspect(self, prompt, images):
        raise AssertionError("SceneInspector must use inspect_request")


def bridge_context(ws4, tmp_path, server, token, **session):
    perception = load_perception_config(PERCEPTION)
    ctx = build_context(robot_config(ws4, tmp_path, server, token, **session), backend="ws4", policy=POLICY,
                        log_path=tmp_path / "events.jsonl", vision_model=None, inspection=perception.inspection)
    bridge = ctx.arms["left"].ws4_bridge
    assert ctx.arms["right"].ws4_bridge is bridge and ctx.cameras["top"].ws4_bridge is bridge
    scene = SceneFromWs4(bridge)
    from liminal.perception.inspector import SceneInspector

    ctx.inspector = SceneInspector(scene, ctx.cameras, ctx.world, confirmations=perception.confirmations)
    original = bridge.run_step

    async def spy(skill, arm, **kw):
        result = await original(skill, arm, **kw)
        scene.steps.append((skill, arm, result.get("status")))
        if result.get("status") == "succeeded":
            scene.done.add(skill)
        return result

    scene.steps = []
    bridge.run_step = spy
    return ctx, bridge, scene


def test_assembly_task_runs_each_skill_as_a_ws4_step(ws4, tmp_path, policy_server):
    server, token, backends = policy_server
    ctx, bridge, scene = bridge_context(ws4, tmp_path, server, token)
    bridge.connect()
    try:
        session = bridge.session
        session.verifier = checks = ws4["Checks"]()
        task = load_task(TASK, arms=ctx.arm_names)
        with connected(ctx):
            report = asyncio.run(run_task(task, ctx))
        assert report.goal_verified, (report, [e for e in ctx.world.events if e["kind"].startswith("ws4")])
        # the tree chose the steps and the order; ws4 ran each one on the robot side the vocabulary allows
        assert scene.steps == [(s, a, "succeeded") for s, a in CHAIN]
        # each LoRA package served only its own skill's training caption
        for skill, pkg in PACKAGES.items():
            assert {r.task for r in backends[pkg].requests} == {session.vocab[skill].prompt}
        # the box arm holds the lid from open_box on and keeps its torque at close
        assert session.rig.holding == {"right"} and not session.rig.torque_off_at_close(session.rig.arms["right"])
        assert "open_box" in checks.calls and checks.calls.count("open_box") >= 4  # its check + 3 lid prechecks
        # the tree verified every success predicate on its own fresh evidence
        kinds = [e["kind"] for e in ctx.world.events]
        assert kinds.count("ws4_result") == 4 and "verify_failed" not in kinds
        assert ("lid_open",) in scene.requests and any("components_assembled" in r for r in scene.requests)
        assert not ctx.safety.estopped and not session.runner.guard.estopped
    finally:
        bridge.disconnect()
    assert bridge.session is None  # closed on the last disconnect (ws4 e-stop first: both arms freeze holding)


def test_liminal_estop_freezes_ws4_and_fails_the_tree(ws4, tmp_path, policy_server):
    server, token, _ = policy_server
    ctx, bridge, scene = bridge_context(ws4, tmp_path, server, token)
    bridge.connect()
    try:
        bridge.session.verifier = ws4["Checks"]()
        session = bridge.session
        holds = []
        session.rig.hold = lambda reason="": holds.append(reason)  # what the real rig does: freeze_keep_grip
        task = load_task(TASK, arms=ctx.arm_names)

        async def run_and_estop():
            run = asyncio.create_task(run_task(task, ctx))
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline and not (
                    session.executor is not None and session.executor.state == "running"
                    and any(r.status == "running" for r in session.executor.runs.values())):
                await asyncio.sleep(0.005)
            assert bridge.running == "right"  # open_box is moving (ramps or policy)
            ctx.safety.estop()  # e.g. POST /estop on `liminal serve`
            return await run

        with connected(ctx):
            report = asyncio.run(run_and_estop())
        assert not report.goal_verified and "EmergencyStop" in (report.error or "")
        assert session.runner.guard.estopped and holds  # ws4 froze both arms (torque on, grips kept)
        assert session.executor.state == "aborted"
    finally:
        bridge.disconnect()


def test_backend_refuses_what_the_loras_were_not_trained_for(ws4, tmp_path, policy_server):
    from liminal.core.types import Skill, Status

    server, token, _ = policy_server
    ctx, bridge, scene = bridge_context(ws4, tmp_path, server, token)
    bridge.connect()
    try:
        backend = ctx.skills.policy

        def run(skill):
            return asyncio.run(backend.execute(skill, ctx))

        wrong_arm = run(Skill("left", "open_box", prompt="open the box by pulling its handle"))
        assert wrong_arm.status is Status.FAILURE and "robot's right arm" in wrong_arm.message
        wrong_prompt = run(Skill("left", "put_green_pen_in_slot", prompt="put the green pen in its slot"))
        assert wrong_prompt.status is Status.FAILURE and "trained on" in wrong_prompt.message
        unknown = run(Skill("left", "wave", prompt="wave"))
        assert unknown.status is Status.FAILURE and "no ws4 skill" in unknown.message
        assert scene.steps == []  # nothing reached ws4
        bridge.session.rig.mark_holding("right")
        held = run(Skill("right", "open_box", prompt="open the box by pulling its handle"))
        assert held.status is Status.FAILURE and "holds the lid" in held.message
    finally:
        bridge.disconnect()


def test_ws4_arms_refuse_direct_motor_commands(ws4, tmp_path, policy_server):
    from liminal.core.types import Skill

    server, token, _ = policy_server
    ctx, bridge, _ = bridge_context(ws4, tmp_path, server, token)
    with connected(ctx):
        assert len(ctx.arms["left"].read_state()) == 6 and ctx.cameras["top"].read().ndim == 3
        with pytest.raises(RuntimeError, match="move only through ws4"):
            ctx.arms["left"].send_action([0.0] * 6)
        # a primitive would command the arm directly: the supervisor refuses and latches its e-stop
        with pytest.raises(RuntimeError, match="move only through ws4"):
            asyncio.run(ctx.skills.execute(Skill("left", "open_gripper"), ctx))
        assert ctx.safety.estopped and bridge.session.runner.guard.estopped
        with pytest.raises(EmergencyStop):
            ctx.safety.check()
        with pytest.raises(EmergencyStop):  # no ws4 step starts while the e-stop is latched
            asyncio.run(ctx.skills.policy.execute(Skill("left", "put_green_pen_in_slot",
                                                        prompt="put the green pen in the green slot"), ctx))
        assert bridge.session.last_result is None


def test_haiku_vision_maps_ws4_verdicts_to_predicates(ws4):
    from orchestrator.verifier import Verdict

    from liminal.perception.types import InspectionRequest
    from liminal.ws4_bridge.vision import Ws4HaikuVisionModel

    answers = {"open_box": (True, 0.95), "put_green_pen_in_slot": (True, 0.9),
               "put_yellow_cutter_in_slot": (False, 0.4), "put_pen_in_slot": (False, 0.9)}

    class Fake:
        def __init__(self):
            self.seen = []

        def check(self, step, skill, obs):
            self.seen.append((skill.name, step.success_check, set(obs)))
            done, conf = answers[skill.name]
            return Verdict(done, conf, f"{skill.name} seen", "fake")

    fake = Fake()
    model = Ws4HaikuVisionModel(verifier=fake)
    request = InspectionRequest(("lid_open", "yellow_boxcutter_in_slot", "components_assembled", "main_arm_home"),
                                cameras=("top", "left_wrist"))
    top, wrist = np.zeros((48, 64, 3), np.uint8), np.ones((48, 64, 3), np.uint8)
    out = asyncio.run(model.inspect_request(request, [top, wrist]))["predicates"]
    assert out["lid_open"]["value"] == "true"
    assert out["yellow_boxcutter_in_slot"]["value"] == "unknown"  # below the 0.7 confidence of the skill
    assert out["components_assembled"]["value"] == "false"  # the red pen is confidently not in its slot
    assert out["main_arm_home"]["value"] == "unknown"  # no ws4 check: fail closed
    # one check per needed predicate, each with its ws4 skill's own condition, on the head frame only
    assert sorted(name for name, _, _ in fake.seen) == sorted(answers)
    assert all(keys == {"observation.images.head"} for _, _, keys in fake.seen)
    assert ("open_box", "The lid of the box is open.", {"observation.images.head"}) in fake.seen


def test_xlerobot_config_uses_the_robot_frame():
    """The task files and registry give `left` the pen skills: `left` must be the robot-LEFT arm, board 5B14030445
    (= /dev/haller_arm_right per /etc/udev/rules.d/99-haller-devices.rules), not the box arm 5B79018460."""
    cfg = yaml.safe_load(Path("configs/robot/xlerobot.yaml").read_text())
    assert "5B14030445" in cfg["arms"]["left"]["port"] and "5B79018460" in cfg["arms"]["right"]["port"]
    haller = yaml.safe_load(ROBOT.read_text())["ws4"]["session"]
    assert haller["port"] == "/dev/haller_arm_right" and haller["right_arm_port"] == "/dev/haller_arm_left"
    assert haller["wrist_cam"].endswith("usb-0:7.2:1.0-video-index0")
    assert haller["right_wrist_cam"].endswith("usb-0:7.3:1.0-video-index0")


def test_liminal_run_cli_drives_the_fake_haller(ws4, tmp_path, policy_server, monkeypatch, capsys):
    """`liminal run` with the shipped Haller configs. The Claude-backed parts stay real code down to the API call:
    ClaudeVerifier.check is the only stand-in (it judges the fake rig's state; fake camera frames show nothing), for
    both ws4's step checks and the ws4_haiku scene predicates."""
    import json

    from orchestrator import verifier as ws4_verifier
    from orchestrator.verifier import Verdict

    from liminal.ui.cli import main
    from liminal.ws4_bridge import session as bridge_session
    from liminal.ws4_bridge import vision

    server, token, backends = policy_server
    bridges, done, calls = [], set(), []
    connect = bridge_session.Ws4Bridge.connect

    def tracking_connect(self):
        bridges.append(self)
        connect(self)

    def fake_check(self, step, skill, obs):
        calls.append(("scene" if step.id in vision.HALLER_CHECKS else "ws4", skill.name))
        rig = bridges[-1].session.rig
        box = np.asarray(rig.arms["right"].robot.joints())[:5]
        if skill.name == "open_box":
            return Verdict(bool(np.abs(box - GREEN_HOLD[:5]).max() <= 1.0), 0.9, "fake rig", "fake")
        if step.id not in vision.HALLER_CHECKS:  # ws4's own check after the step: the fake policy "placed" it
            done.add(skill.name)
        return Verdict(skill.name in done, 0.9, "fake rig", "fake")

    monkeypatch.setattr(bridge_session.Ws4Bridge, "connect", tracking_connect)
    monkeypatch.setattr(ws4_verifier.ClaudeVerifier, "check", fake_check)
    monkeypatch.setattr("orchestrator.claude.api_key_available", lambda: True)
    robot = tmp_path / "haller_fake.yaml"
    robot.write_text(yaml.safe_dump(robot_config(ws4, tmp_path, server, token, verifier="claude")))
    code = main(["run", str(TASK), "--robot", str(robot), "--backend", "ws4", "--policy", str(POLICY),
                 "--perception", str(PERCEPTION), "--log-dir", str(tmp_path / "runs")])
    out = capsys.readouterr().out
    report = json.loads(out[out.index("{\n") : out.rindex("}") + 1])
    assert code == 0 and report["goal_verified"] and report["status"] == "success", out
    assert {skill for kind, skill in calls if kind == "ws4"} == {s for s, _ in CHAIN}
    assert {skill for kind, skill in calls if kind == "scene"} == {s for s, _ in CHAIN}
    assert all(len(backends[pkg].requests) > 1 for pkg in PACKAGES.values())  # warm-up plus the policy's chunks
    assert bridges and bridges[-1].session is None  # closed at the end of the run


def test_session_settings_per_run(monkeypatch, tmp_path):
    from liminal.ws4_bridge.session import Ws4Bridge

    cfg = yaml.safe_load(ROBOT.read_text())["ws4"]
    monkeypatch.setenv("WS4_TOKEN_CMD", "ssh -p 1 root@pod cat /workspace/flux_token")
    monkeypatch.setenv("LIMINAL_WS4_SESSION", '{"observe": true, "hold_arms": ["right"], "operator": "oscar"}')
    out = Ws4Bridge(cfg).session_config()
    assert out["observe"] is True and out["hold_arms"] == ("right",) and out["operator"] == "oscar"
    assert out["token_cmd"].endswith("cat /workspace/flux_token") and out["server"] == "http://127.0.0.1:8790"
    assert (out["port"], out["right_arm_port"]) == ("/dev/haller_arm_right", "/dev/haller_arm_left")
    monkeypatch.setenv("LIMINAL_WS4_SESSION", "[1]")
    with pytest.raises(ValueError, match="JSON object"):
        Ws4Bridge(cfg).session_config()


def test_unclear_lid_view_keeps_the_last_confident_verdict_for_keep_s(ws4):
    from orchestrator.verifier import Verdict

    from liminal.perception.types import InspectionRequest
    from liminal.ws4_bridge.vision import Ws4HaikuVisionModel

    answers, now = [], [0.0]

    class Fake:
        def check(self, step, skill, obs):
            return answers.pop(0)

    model = Ws4HaikuVisionModel(verifier=Fake(), keep_s={"lid_open": 30}, clock=lambda: now[0])
    request = InspectionRequest(("lid_open",), cameras=("top",))
    frame = [np.zeros((48, 64, 3), np.uint8)]

    def lid(done, conf, t):
        answers.append(Verdict(done, conf, "seen", "fake"))
        now[0] = t
        return asyncio.run(model.inspect_request(request, frame))["predicates"]["lid_open"]

    assert lid(False, 0.4, 0)["value"] == "unknown"  # nothing confident yet: fail closed
    assert lid(True, 0.9, 5)["value"] == "true"
    kept = lid(False, 0.35, 20)  # the pen arm hides the box
    assert kept["value"] == "true" and "kept the confident verdict from 15 s ago" in kept["evidence"]
    assert lid(False, 0.35, 36)["value"] == "unknown"  # 31 s since the last confident view
    assert lid(False, 0.9, 40)["value"] == "false"  # a confident closed lid always counts
    assert lid(True, 0.5, 45)["value"] == "false"  # and is what an unclear view keeps



def test_a_failed_ws4_step_ends_without_waiting_for_a_ws4_replan(ws4, tmp_path, policy_server):
    """Live 18 Oct: a failed open_box parked ws4 in awaiting_replan (no goal, no approver) until the tree's timeout."""
    from orchestrator.verifier import Verdict

    server, token, _ = policy_server
    ctx, bridge, scene = bridge_context(ws4, tmp_path, server, token)
    bridge.connect()
    try:
        bridge.session.verifier = ws4["Checks"]()
        bridge.session.verifier.check = lambda step, skill, obs: Verdict(False, 0.9, "lid still closed", "test")
        t0 = time.monotonic()
        with connected(ctx):
            report = asyncio.run(run_task(load_task(TASK, arms=ctx.arm_names), ctx))
        assert not report.goal_verified and time.monotonic() - t0 < 60
        assert scene.steps == [("open_box", "right", "failed")]
        assert bridge.session.executor.state == "failed" and bridge.session.cfg.max_replans == 0
    finally:
        bridge.disconnect()


def test_a_slow_check_counts_as_unclear_and_keeps_the_lid(ws4):
    """Live 18 Oct run 3: one slow Opus call made SceneInspector publish lid_open unknown and broke the WHILE."""
    import threading as th

    from orchestrator.verifier import Verdict

    from liminal.perception.types import InspectionRequest
    from liminal.ws4_bridge.vision import Ws4HaikuVisionModel

    release = th.Event()
    answers = [Verdict(True, 0.9, "open", "fake")]

    class Fake:
        def check(self, step, skill, obs):
            if answers:
                return answers.pop(0)
            release.wait(5)  # slower than the request allows
            return Verdict(False, 0.1, "late", "fake")

    model = Ws4HaikuVisionModel(verifier=Fake(), keep_s={"lid_open": 45})
    request = InspectionRequest(("lid_open",), cameras=("top",), timeout_s=0.5)
    frame = [np.zeros((48, 64, 3), np.uint8)]
    assert asyncio.run(model.inspect_request(request, frame))["predicates"]["lid_open"]["value"] == "true"
    slow = asyncio.run(model.inspect_request(request, frame))["predicates"]["lid_open"]
    release.set()
    assert slow["value"] == "true" and "longer than" in slow["evidence"] and "kept the confident" in slow["evidence"]
