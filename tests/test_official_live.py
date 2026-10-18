"""Official runner gates; mock motors only, no network/GPU/serial."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from liminal.core.types import SO101_JOINTS
from liminal.hardware.official_so101 import OfficialSO101
from liminal.policy.official_live import check_chunk


def arm():
    handle = OfficialSO101.__new__(OfficialSO101)
    handle.name = "right"
    handle.engaged = False
    handle.current = [0.0] * 6
    bus = MagicMock()
    bus.read.return_value = 0
    bus.sync_read.return_value = dict.fromkeys(SO101_JOINTS, 2000)
    handle.robot = SimpleNamespace(bus=bus, cameras={}, is_calibrated=True,
                                   configure=MagicMock(), send_action=MagicMock())
    return handle


def test_out_of_travel_chunk_rejected_without_modifying_targets():
    actions = [[0.0] * 6 for _ in range(42)]
    actions[31][3] = 132
    with pytest.raises(RuntimeError, match="wrist_flex=132"):
        check_chunk(actions, [-100] * 6, [100] * 6, 32)
    assert actions[31][3] == 132
    check_chunk(actions, [-100] * 6, [100] * 6, 16)


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_rejected_even_outside_executed_horizon(value):
    actions = [[0.0] * 6 for _ in range(42)]
    actions[-1][0] = value
    with pytest.raises(ValueError):
        check_chunk(actions, [-100] * 6, [100] * 6, 32)


def test_read_only_connection_never_configures_or_enables_motors(monkeypatch):
    fake_socket = MagicMock()
    fake_socket.return_value.__enter__.return_value.connect_ex.return_value = 1
    monkeypatch.setattr("liminal.hardware.official_so101.socket.socket", fake_socket)
    handle = arm()
    handle.connect_read_only()
    handle.robot.configure.assert_not_called()
    handle.robot.bus.write.assert_not_called()
    handle.robot.bus.sync_write.assert_not_called()


def test_haller_listener_blocks_serial_open(monkeypatch):
    fake_socket = MagicMock()
    fake_socket.return_value.__enter__.return_value.connect_ex.return_value = 0
    monkeypatch.setattr("liminal.hardware.official_so101.socket.socket", fake_socket)
    handle = arm()
    with pytest.raises(RuntimeError, match="Stop Haller"):
        handle.connect_read_only()
    handle.robot.bus.connect.assert_not_called()


def test_targets_unchanged_and_release_attempts_every_motor():
    handle = arm()
    handle.engaged = True
    target = [10, 20, 30, 40, 50, 60]
    expected = dict(zip((j + ".pos" for j in SO101_JOINTS), target, strict=True))
    handle.robot.send_action.return_value = expected
    handle.send_action(target)
    handle.robot.send_action.assert_called_once_with(expected)
    handle.robot.bus.write.side_effect = [RuntimeError("first motor failed"), None, None, None, None, None]
    with pytest.raises(RuntimeError, match="TORQUE RELEASE UNCONFIRMED"):
        handle.stop()
    assert handle.robot.bus.write.call_count == 6


def test_engagement_parks_current_encoder_goals_before_stock_configure():
    handle = arm()
    order = []
    handle.robot.bus.sync_write.side_effect = lambda *a, **k: order.append("park")
    handle.robot.configure.side_effect = lambda: order.append("configure")
    handle.engage()
    assert order == ["park", "configure"]
    assert handle.engaged


@pytest.mark.parametrize("mode", ["student", "edit", "eef", "stop"])
def test_direct_runner_hybrid_full_cycle_uses_post_ack_measurements(tmp_path, monkeypatch, mode):
    import asyncio
    import json
    import sys

    pytest.importorskip("numpy")
    from test_hybrid import decision

    from liminal.policy import official_live
    from liminal.policy.hybrid import HybridConfig

    sent, reasoner_calls = [], []

    class FakeArm:
        name = "right"
        num_joints = 6
        engaged = False
        current = [0, 0, 0, 0, 0, 50]
        pending = None

        def __init__(self, cfg):
            pass

        def connect_read_only(self):
            pass

        def limits(self):
            return [-100]*5+[0], [100]*6

        def observe(self):
            if self.pending is not None:
                self.current = list(self.pending)
            image = SimpleNamespace(std=lambda: 10)
            return {"scene": image, "wrist": image}, dict.fromkeys(SO101_JOINTS, 0)

        def read_state(self):
            return list(self.current)

        def engage(self):
            self.engaged = True

        def send_action(self, target):
            assert self.engaged
            sent.append(list(target))
            self.pending = list(target)

        def stop(self):
            self.engaged = False

        def close(self):
            self.stop()

    class Reasoner:
        async def decide(self, payload, images, schema):
            reasoner_calls.append(payload)
            return decision(payload, mode if len(reasoner_calls) == 1 else "stop")

    class Writer:
        def write(self, value):
            pass

        async def drain(self):
            pass

        def close(self):
            pass

    class Process:
        stdin = Writer()
        async def wait(self):
            return 0

    messages = iter([
        {"type": "ready", "integration": "lerobot.Flux3Policy", "inference": {"n_action_steps": 32, "fps": 30}},
        *[{"type": "actions", "actions": [[i/10]*5+[50] for i in range(42)], "inference_seconds": .1}]*2,
    ])

    async def reply(*args):
        return next(messages)

    async def spawn(*args, **kwargs):
        return Process()

    monkeypatch.setitem(sys.modules, "cv2", SimpleNamespace(
        COLOR_RGB2BGR=0, cvtColor=lambda img, mode: img, imencode=lambda ext, img: (True, b"jpeg")))
    monkeypatch.setattr(official_live, "OfficialSO101", FakeArm)
    monkeypatch.setattr(official_live, "worker_reply", reply)
    monkeypatch.setattr(official_live.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr("liminal.policy.hybrid.ClaudeReasoner", lambda cfg: Reasoner())
    cfg = {"arm": "right", "prompt": "put pen in slot", "python": "python", "worker": "worker",
           "vm": "fake", "zone": "fake", "project": "fake",
           "hybrid": vars(HybridConfig(enabled=True, kinematics_verified=True))}
    asyncio.run(official_live.run(cfg, tmp_path, True))
    assert len(sent) == (0 if mode == "stop" else 2)
    if mode == "student":
        assert sent == [[0]*5+[50], [.1]*5+[50]]
    if mode == "eef":
        assert [a[5] for a in sent] == [60, 70]  # second IK step uses the actual first result
    if sent:
        assert reasoner_calls[1]["episode_history"][0]["after"] == sent[-1]
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["status"] == "hybrid_stopped_task_success_unverified"
