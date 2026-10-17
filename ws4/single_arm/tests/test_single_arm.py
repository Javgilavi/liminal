"""Single-arm runner on a fake arm: fake policy, and the real HTTP client against a local fake policy server."""

import json
import sys
import threading
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import run_single_arm as rsa  # noqa: E402
from flux.backends import FakeBackend  # noqa: E402
from flux.contract import SO101_CONTRACT  # noqa: E402
from flux.remote import make_server  # noqa: E402


def _run(tmp_path, monkeypatch, *extra):
    monkeypatch.setattr(rsa, "RUNS", tmp_path)
    monkeypatch.setattr(rsa, "STOP_FILE", tmp_path / "STOP")
    monkeypatch.setattr(rsa, "KEY_FILE", tmp_path / "KEY")
    rsa.main(["--robot", "fake", "--seconds", "2", "--yes", "--no-video", "--fake-delay", "0.05", *extra])
    metas = sorted(tmp_path.glob("*/attempt*/meta.json"))
    return [json.loads(m.read_text()) for m in metas]


def test_fake_backend_attempt_is_recorded_and_guarded(tmp_path, monkeypatch):
    (meta,) = _run(tmp_path, monkeypatch, "--max-step", "0.1")
    assert meta["chunks"] >= 1 and meta["end_reason"] == "timeout"
    trace = np.load(next(tmp_path.glob("*/attempt01/trace.npz")))
    steps = np.abs(np.diff(trace["command"][:, :5], axis=0))
    assert steps.max() <= 0.1 + 1e-9  # the guard's per-tick limit holds on every joint
    assert trace["guard"].any()


def test_remote_client_against_local_fake_server(tmp_path, monkeypatch):
    backend = FakeBackend(SO101_CONTRACT, velocity=0.1)
    server = make_server({"so101": backend}, port=0, token="secret")
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        metas = _run(tmp_path, monkeypatch, "--backend", "remote", "--server", url, "--token-cmd", "echo secret",
                     "--no-log", "--task", "stack the cups", "--task", "put the left cup into the right cup",
                     "--attempts", "2")  # fmt: skip
    finally:
        server.shutdown()
    assert [m["task"] for m in metas] == ["stack the cups", "put the left cup into the right cup"]
    assert all(m["chunks"] >= 1 for m in metas)
    tasks = [r.task for r in backend.requests]
    assert tasks[:2] == ["put the left cup into the right cup", "stack the cups"]  # warm-up, sorted
    assert "token_cmd" not in metas[0]["args"]


def test_stop_file_freezes_the_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(rsa, "RUNS", tmp_path)
    monkeypatch.setattr(rsa, "STOP_FILE", tmp_path / "STOP")
    monkeypatch.setattr(rsa, "KEY_FILE", tmp_path / "KEY")
    real_step = rsa.DualArmRunner.step
    calls = {"n": 0}

    def step(self):
        calls["n"] += 1
        if calls["n"] == 10:
            (tmp_path / "STOP").touch()
        return real_step(self)

    monkeypatch.setattr(rsa.DualArmRunner, "step", step)
    rsa.main(["--robot", "fake", "--seconds", "5", "--yes", "--no-video", "--fake-delay", "0.0"])
    meta = json.loads(next(tmp_path.glob("*/attempt01/meta.json")).read_text())
    assert meta["end_reason"].startswith("stop") and meta["ticks"] == 10


class FlakyBus:
    """A bus whose sync_read fails on demand; records goals the way the real bus would receive them."""

    def __init__(self):
        self.q = {"shoulder_pan": 0.0, "shoulder_lift": -95.0, "elbow_flex": 95.0, "wrist_flex": 70.0,
                  "wrist_roll": 0.0, "gripper": 2.0}  # fmt: skip
        self.fail_reads = 0
        self.goals = []

    def sync_read(self, name, num_retry=0):
        if self.fail_reads:
            self.fail_reads -= 1
            raise ConnectionError("Incorrect status packet")
        return dict(self.q)

    def sync_write(self, name, goal, num_retry=0):
        self.goals.append(dict(goal))
        self.q.update(goal)


class FakeCam:
    def read_latest(self):
        return np.zeros((480, 640, 3), np.uint8)


class DirectRobot:
    def __init__(self):
        self.bus = FlakyBus()
        self.cameras = {"scene": FakeCam(), "wrist": FakeCam()}


def test_direct_bus_path_caps_goals_like_lerobot_and_survives_a_bus_error():
    robot = rsa.SingleArmRobot(DirectRobot(), max_relative_target=5.0)
    robot.get_observation()
    robot.send_joints(np.array([20.0, -95.0, 95.0, 70.0, 0.0, 2.0]))
    assert robot.robot.bus.goals[-1]["shoulder_pan"] == 5.0 and robot.capped == 1  # LeRobot's own cap
    backend = FakeBackend(SO101_CONTRACT, velocity=0.1, delay_s=0.0)
    target = rsa.per_arm_target("left", "policy", calibration=rsa.ArmCalibration())
    runner = rsa.DualArmRunner(robot, [target], {"policy": backend}, rsa.SafetyGuard(rsa.SafetyConfig()))
    runner.start("left", "stack the cups")
    runner.step()
    robot.robot.bus.fail_reads = 1
    try:
        runner.step()
        raised = False
    except ConnectionError:
        raised = True
    assert raised  # the runner itself propagates; run_attempt turns it into a freeze (end_reason "stop (error ...)")
