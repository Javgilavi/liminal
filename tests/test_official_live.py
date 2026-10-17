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
