"""Fake-bus tests only: no serial ports, cameras, or optional hardware imports."""
import sys
import threading
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

from liminal.core.types import SO101_JOINTS
from liminal.hardware.so101 import SO101Arm
from liminal.safety import EmergencyStop, SafetySupervisor


@pytest.fixture
def rig(monkeypatch, tmp_path):
    robot = MagicMock()
    robot.calibration = {}
    calibration = {j: SimpleNamespace(range_min=100, range_max=3900) for j in SO101_JOINTS}
    robot.bus.read_calibration.return_value = calibration
    robot.bus.read.side_effect = lambda register, *a, **kw: {
        "Torque_Enable": 0, "Operating_Mode": 0, "Unloading_Condition": 44,
        "Max_Torque_Limit": 500, "Protection_Current": 250,
    }[register]
    robot.bus.sync_read.side_effect = lambda register, **kw: dict.fromkeys(SO101_JOINTS, {
        "Present_Temperature": 24, "Present_Voltage": 73, "Status": 0, "Present_Position": 2000,
    }[register])
    robot.get_observation.return_value = {f"{j}.pos": 0.0 for j in SO101_JOINTS}
    package = ModuleType("lerobot.robots.so_follower")
    package.SO101Follower = MagicMock(return_value=robot)
    package.SO101FollowerConfig = lambda **kw: SimpleNamespace(**kw)
    for name in ["lerobot", "lerobot.robots"]:
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, "lerobot.robots.so_follower", package)
    monkeypatch.setattr("liminal.hardware.so101.subprocess.run", lambda *a, **kw: SimpleNamespace(returncode=1))
    monkeypatch.setattr("liminal.hardware.so101.Path.home", lambda: tmp_path)
    device = tmp_path / "fake-device"
    device.touch()
    arm = SO101Arm("left", str(device))
    yield arm, robot, package
    if arm._robot is not None:
        arm.disconnect()


def test_connect_does_not_enable_configure_or_write_calibration(rig):
    arm, robot, package = rig
    arm.connect()
    assert not arm.torque_enabled
    assert package.SO101Follower.call_args.args[0].use_degrees
    robot.connect.assert_not_called()
    robot.configure.assert_not_called()
    robot.calibrate.assert_not_called()
    robot.bus.write_calibration.assert_not_called()
    robot.bus.write.assert_not_called()
    robot.bus.sync_write.assert_not_called()
    robot.bus.enable_torque.assert_not_called()
    robot._save_calibration.assert_called_once()


def test_busy_port_is_rejected_before_constructing_robot(rig, monkeypatch):
    arm, _, package = rig
    monkeypatch.setattr("liminal.hardware.so101.subprocess.run", lambda *a, **kw: SimpleNamespace(returncode=0))
    with pytest.raises(RuntimeError, match="busy"):
        arm.connect()
    package.SO101Follower.assert_not_called()
    assert arm._lease is None


def test_calibration_mismatch_closes_port_without_writes(rig):
    arm, robot, _ = rig
    robot.calibration = {"wrong": "file"}
    with pytest.raises(RuntimeError, match="Calibration file differs"):
        arm.connect()
    robot.bus.disconnect.assert_called_once_with(disable_torque=False)
    robot.bus.write.assert_not_called()
    robot.bus.write_calibration.assert_not_called()


def test_factory_default_range_is_not_adopted_as_calibration(rig):
    arm, robot, _ = rig
    calibration = robot.bus.read_calibration.return_value
    calibration["shoulder_lift"] = SimpleNamespace(range_min=0, range_max=4095)
    with pytest.raises(RuntimeError, match="not been swept"):
        arm.connect()
    robot._save_calibration.assert_not_called()
    robot.bus.write.assert_not_called()


def test_startup_cleanup_releases_lease_even_when_port_close_fails(rig):
    arm, robot, _ = rig
    robot.calibration = {"wrong": "file"}
    robot.bus.disconnect.side_effect = RuntimeError("USB disconnected")
    with pytest.raises(RuntimeError, match="Calibration file differs"):
        arm.connect()
    assert arm._lease is None


def test_engage_parks_raw_position_before_enabling(rig):
    arm, robot, _ = rig
    arm.connect()
    arm.set_torque(True)
    calls = robot.bus.mock_calls
    park = next(i for i, c in enumerate(calls) if c[0] == "sync_write")
    engage = next(i for i, c in enumerate(calls) if c[0] == "enable_torque")
    assert park < engage
    robot.bus.sync_write.assert_called_once_with("Goal_Position", dict.fromkeys(SO101_JOINTS, 2000), normalize=False)
    assert arm.torque_enabled


def test_unstable_encoder_never_enables_torque(rig):
    arm, robot, _ = rig
    arm.connect()
    original = robot.bus.sync_read.side_effect
    ticks = iter([2000, 100, 2000])
    robot.bus.sync_read.side_effect = lambda reg, **kw: (
        dict.fromkeys(SO101_JOINTS, next(ticks)) if reg == "Present_Position" else original(reg, **kw)
    )
    with pytest.raises(RuntimeError, match="unstable"):
        arm.set_torque(True)
    robot.bus.enable_torque.assert_not_called()


def test_release_continues_after_one_motor_refuses_and_closes_port(rig):
    arm, robot, _ = rig
    arm.connect()
    robot.bus.write.side_effect = [RuntimeError("alarm"), None, None, None, None, None]
    with pytest.raises(RuntimeError, match="unconfirmed"):
        arm.disconnect()
    assert robot.bus.write.call_count == 6
    robot.bus.disconnect.assert_called_once_with(disable_torque=False)
    assert arm._lease is None
    assert arm._robot is None


def test_torque_off_refuses_action(rig):
    arm, robot, _ = rig
    arm.connect()
    with pytest.raises(RuntimeError, match="torque off"):
        arm.send_action([0] * 6)
    robot.send_action.assert_not_called()


def test_rate_cap_uses_elapsed_time_and_cannot_accumulate_idle_budget(rig, monkeypatch):
    arm, robot, _ = rig
    arm.connect()
    arm.set_torque(True)
    arm._last_action = 10
    monkeypatch.setattr("liminal.hardware.so101.time.monotonic", lambda: 100)
    arm.send_action([100] * 6)
    assert max(robot.send_action.call_args.args[0].values()) == 2
    # Repeating a command with no elapsed time does not buy another step.
    arm.send_action([100] * 6)
    assert max(robot.send_action.call_args.args[0].values()) == 2


def test_hot_motor_prevents_engage(rig):
    arm, robot, _ = rig
    arm.connect()
    original = robot.bus.sync_read.side_effect
    robot.bus.sync_read.side_effect = lambda reg, **kw: (
        dict.fromkeys(SO101_JOINTS, 56) if reg == "Present_Temperature" else original(reg, **kw)
    )
    with pytest.raises(RuntimeError, match="temperature"):
        arm.set_torque(True)
    robot.bus.enable_torque.assert_not_called()


def test_health_monitor_latches_fault_and_releases_all_motors(rig):
    arm, robot, _ = rig
    arm.connect()
    arm._watch_stop.set()
    arm._watch_thread.join(timeout=2)
    arm._watch_stop = MagicMock()
    arm._watch_stop.wait.side_effect = [False]
    robot.bus.sync_read.side_effect = RuntimeError("lost telemetry")
    arm.on_fault = MagicMock()
    arm._watch()
    assert arm.fault == "lost telemetry"
    assert robot.bus.write.call_count == 6
    arm.on_fault.assert_called_once()
    with pytest.raises(RuntimeError, match="inspect hardware"):
        arm.set_torque(True)


def test_estop_latch_blocks_queued_engagement_and_keeps_other_arm_releasing():
    left, right = MagicMock(name="left"), MagicMock(name="right")
    left.emergency_stop.side_effect = RuntimeError("left alarm")
    safety = SafetySupervisor({"left": left, "right": right})
    safety.estop()
    right.emergency_stop.assert_called_once()
    with pytest.raises(EmergencyStop):
        safety.set_torque("right", True)
    right.set_torque.assert_not_called()


def test_estop_waits_for_inflight_command_then_releases():
    from liminal.hardware.mock_arm import MockArm

    arm = MockArm("left")
    safety = SafetySupervisor({"left": arm})
    started, finish = threading.Event(), threading.Event()
    original = arm.send_action

    def slow_send(target):
        started.set()
        assert finish.wait(2)
        original(target)

    arm.send_action = slow_send
    worker = threading.Thread(target=safety.send, args=("left", [1] * 6))
    worker.start()
    assert started.wait(2)
    emergency = threading.Thread(target=safety.estop)
    emergency.start()
    finish.set()
    worker.join(2)
    emergency.join(2)
    assert safety.estopped and arm.stopped
    with pytest.raises(EmergencyStop):
        safety.send("left", [2] * 6)


def test_disconnected_emergency_release_is_harmless():
    arm = SO101Arm("left", "/not-used")
    arm.emergency_stop()
    assert not arm.torque_enabled
    with pytest.raises(RuntimeError, match="not connected"):
        arm.set_torque(True)


def test_camera_only_lifecycle_connects_and_disconnects_camera():
    from liminal.hardware.runtime import connected_cameras

    camera = MagicMock(name="top")
    camera.name = "top"
    with connected_cameras({"top": camera}):
        camera.connect.assert_called_once()

    camera.disconnect.assert_called_once()


@pytest.mark.parametrize("operation", ["read_state", "send_action"])
def test_transport_failure_stops_both_arms(operation):
    from liminal.hardware.mock_arm import MockArm

    left, right = MockArm("left"), MockArm("right")
    setattr(left, operation, MagicMock(side_effect=RuntimeError("lost bus")))
    safety = SafetySupervisor({"left": left, "right": right})
    with pytest.raises(RuntimeError, match="lost bus"):
        safety.send("left", [1] * 6)
    assert safety.estopped
    assert right.stopped


def test_normal_stop_preserves_loaded_gripper_goal(rig):
    arm, robot, _ = rig
    arm.connect()
    arm.set_torque(True)
    arm._target = [0, 0, 0, 0, 0, 3]
    robot.get_observation.return_value["gripper.pos"] = 17.0
    robot.bus.sync_write.reset_mock()
    arm.stop()
    command = robot.bus.sync_write.call_args.args[1]
    assert "gripper" not in command
    assert set(command) == set(SO101_JOINTS) - {"gripper"}
    assert arm.read_command()[-1] == 3
    assert arm.torque_enabled


def test_freeze_estop_keeps_torque_and_the_gripper_goal(rig):
    arm, robot, _ = rig
    arm.estop_mode = "freeze"  # robot config `estop: freeze`, e.g. an arm holding a box lid open
    arm.connect()
    arm.set_torque(True)
    arm._target = [0, 0, 0, 0, 0, 3]
    robot.bus.sync_write.reset_mock()
    robot.bus.write.reset_mock()
    SafetySupervisor({"left": arm}).estop()
    assert arm.torque_enabled and robot.bus.write.call_count == 0  # no Torque_Enable write
    assert "gripper" not in robot.bus.sync_write.call_args.args[1] and arm.read_command()[-1] == 3
    arm.fault = "overheated"  # a faulty arm still releases
    arm.emergency_stop()
    assert not arm.torque_enabled


def test_estop_mode_is_validated():
    with pytest.raises(ValueError, match="estop"):
        SO101Arm("left", "/not-used", estop="sag")
