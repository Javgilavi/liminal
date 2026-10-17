"""SO-101 transport: read-only attach, explicit torque, serialized bus access.

Five joints use degrees; the gripper uses 0–100 percent. Never run LeRobot's
configure/calibrate on connect: both can change EEPROM, and configure enables torque.
"""
from __future__ import annotations

import logging
import math
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from liminal.core.types import SO101_JOINTS, JointPositions

log = logging.getLogger(__name__)


class SO101Arm:
    num_joints = len(SO101_JOINTS)

    def __init__(self, name: str, port: str, robot_id: str | None = None,
                 hold_rate_hz: float = 30.0, max_relative_target: float | None = None,
                 max_speed: float = 20.0, temperature_limit: float = 55.0):
        if not math.isfinite(max_speed) or max_speed <= 0:
            raise ValueError("max_speed must be finite and positive")
        if max_relative_target is not None and (not math.isfinite(max_relative_target) or max_relative_target <= 0):
            raise ValueError("max_relative_target must be finite and positive")
        if not 20 <= temperature_limit <= 60:
            raise ValueError("temperature_limit must be between 20 and 60 C")
        self.name, self.port, self.robot_id = name, port, robot_id or name
        self.hold_rate_hz = hold_rate_hz
        self.max_relative_target = max_relative_target
        self.max_speed, self.temperature_limit = max_speed, temperature_limit
        self.torque_enabled = False
        self.fault: str | None = None
        self.on_fault: Callable[[], None] | None = None
        self.joint_limits: tuple[list[float], list[float]] | None = None
        self._robot: Any = None
        self._lease: Any = None
        self._lock = threading.RLock()
        self._watch_stop = threading.Event()
        self._watch_thread: threading.Thread | None = None
        self._last_action = 0.0
        self._target: JointPositions | None = None
        self._health: dict[str, Any] = {}

    def connect(self) -> None:
        import fcntl

        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        if self._robot is not None:
            raise RuntimeError(f"{self.name}: already connected")
        device = Path(self.port).resolve(strict=True)
        leases = Path.home() / ".cache/liminal/ports"
        leases.mkdir(parents=True, exist_ok=True)
        self._lease = (leases / device.name).open("a")
        try:
            fcntl.flock(self._lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Also refuse controllers that don't participate in our flock protocol.
            busy = subprocess.run(["fuser", str(device)], capture_output=True, check=False)
            if busy.returncode != 1:
                raise RuntimeError(f"{device} is busy or cannot be checked; stop its controller first")
            cfg = SO101FollowerConfig(port=self.port, id=self.robot_id, use_degrees=True)
            self._robot = SO101Follower(cfg)
            bus = self._robot.bus
            bus.connect()
            for joint in SO101_JOINTS:
                if bus.read("Torque_Enable", joint, normalize=False):
                    raise RuntimeError(f"{joint}: torque is already enabled; rest and release the arm first")
                if bus.read("Operating_Mode", joint, normalize=False) != 0:
                    raise RuntimeError(f"{joint}: not in position mode")
            stored = bus.read_calibration()
            if set(stored) != set(SO101_JOINTS):
                raise RuntimeError("Stored calibration is incomplete")
            if self._robot.calibration and self._robot.calibration != stored:
                raise RuntimeError("Calibration file differs from motor EEPROM; refusing to overwrite either")
            # Harvest existing calibration, never invent offsets or sweep the motors.
            for joint, c in stored.items():
                if not 0 <= c.range_min < c.range_max <= 4095:
                    raise RuntimeError(f"{joint}: invalid stored calibration")
                if joint != "wrist_roll" and not 100 <= c.range_max - c.range_min <= 3800:
                    raise RuntimeError(f"{joint}: calibration range is too small or has not been swept")
            self._robot.calibration = bus.calibration = stored
            self._robot._save_calibration()
            lower, upper = [], []
            for joint in SO101_JOINTS:
                c = stored[joint]
                half = (c.range_max - c.range_min) * 180 / 4095
                lower.append(0.0 if joint == "gripper" else -half)
                upper.append(100.0 if joint == "gripper" else half)
            self.joint_limits = lower, upper
            for joint in SO101_JOINTS:
                unloading = bus.read("Unloading_Condition", joint, normalize=False)
                if unloading & 0x28 != 0x28:
                    raise RuntimeError(f"{joint}: hardware overheat/overload release is not enabled")
            if (bus.read("Max_Torque_Limit", "gripper", normalize=False) > 500
                    or bus.read("Protection_Current", "gripper", normalize=False) > 250):
                raise RuntimeError("Gripper protection is not configured; inspect setup before enabling")
            self._check_health()
            self._watch_stop.clear()
            self._watch_thread = threading.Thread(target=self._watch, name=f"health-{self.name}", daemon=True)
            self._watch_thread.start()
        except BaseException:
            try:
                if self._robot is not None and self._robot.bus.is_connected:
                    self._robot.bus.disconnect(disable_torque=False)
            except Exception:
                log.exception("%s: port close failed during startup cleanup", self.name)
            finally:
                self._robot = None
                self._lease.close()
                self._lease = None
            raise

    def disconnect(self) -> None:
        self._watch_stop.set()
        if self._watch_thread:
            self._watch_thread.join(timeout=3)
        with self._lock:
            try:
                if self._robot is not None:
                    try:
                        self.set_torque(False)
                    finally:
                        self._robot.bus.disconnect(disable_torque=False)
            finally:
                self._robot = None
                if self._lease:
                    self._lease.close()
                    self._lease = None

    def read_state(self) -> JointPositions:
        with self._lock:
            obs = self._robot.get_observation()
            values = [float(obs[f"{j}.pos"]) for j in SO101_JOINTS]
            if not all(math.isfinite(v) for v in values):
                raise RuntimeError(f"{self.name}: invalid position feedback")
            return values

    def _park(self) -> None:
        bus = self._robot.bus
        samples = [bus.sync_read("Present_Position", normalize=False) for _ in range(3)]
        present = {}
        for joint in SO101_JOINTS:
            ticks = sorted(int(sample[joint]) for sample in samples)
            if ticks[0] < 0 or ticks[-1] > 4095 or ticks[-1] - ticks[0] > 12:
                raise RuntimeError(f"{joint}: unstable encoder readings; refusing torque enable")
            present[joint] = ticks[1]
        bus.sync_write("Goal_Position", present, normalize=False)

    def set_torque(self, enabled: bool) -> None:
        with self._lock:
            if self._robot is None:
                if enabled:
                    raise RuntimeError(f"{self.name}: not connected")
                self.torque_enabled = False
                self._target = None
                return
            if enabled:
                if self.fault:
                    raise RuntimeError(f"{self.name}: {self.fault}; inspect hardware before restarting")
                self._check_health()
                current = self.read_state()
                lower, upper = self.joint_limits
                if any(p < lo or p > hi for p, lo, hi in zip(current, lower, upper, strict=True)):
                    raise RuntimeError(f"{self.name}: rest pose is outside the software limits")
                self._park()  # stale goals must never become a movement on enable
                try:
                    self._robot.bus.enable_torque()
                    self.torque_enabled = True
                    self._target = self.read_state()
                    self._last_action = time.monotonic()
                except Exception:
                    self.set_torque(False)
                    raise
            else:
                refused = []
                for joint in SO101_JOINTS:
                    try:
                        self._robot.bus.write("Torque_Enable", joint, 0, normalize=False)
                    except Exception:
                        refused.append(joint)
                self.torque_enabled = False
                self._target = None
                if refused:
                    self.fault = f"torque release unconfirmed on {', '.join(refused)}; cut motor power"
                    raise RuntimeError(self.fault)

    def send_action(self, positions: JointPositions) -> None:
        if len(positions) != self.num_joints or not all(math.isfinite(p) for p in positions):
            raise ValueError(f"{self.name}: expected six finite joint values")
        with self._lock:
            if not self.torque_enabled or self.fault:
                raise RuntimeError(f"{self.name}: torque off or hardware fault; engage explicitly first")
            now = time.monotonic()
            budget = self.max_speed * max(0, min(now - self._last_action, 0.1))
            if self.max_relative_target is not None:
                budget = min(budget, self.max_relative_target)
            current = self.read_state()
            lower, upper = self.joint_limits
            # Refuse a sensor glitch rather than using it as a new command anchor.
            if any(p < lo - 2 or p > hi + 2 for p, lo, hi in zip(current, lower, upper, strict=True)):
                self.emergency_stop()
                raise RuntimeError(f"{self.name}: position feedback outside calibrated limits")
            target = [min(hi, max(lo, p)) for p, lo, hi in zip(positions, lower, upper, strict=True)]
            anchor = self._target if self._target is not None else current
            target = [a + max(-budget, min(budget, p - a)) for p, a in zip(target, anchor, strict=True)]
            self._robot.send_action(dict(zip((f"{j}.pos" for j in SO101_JOINTS), target, strict=True)))
            self._target, self._last_action = target, now

    def hold(self, positions: JointPositions | None = None) -> None:
        # Servos already hold a parked position without a host-side write loop.
        if positions is not None:
            self.send_action(positions)
        else:
            self.stop()

    def release_hold(self) -> None:
        """No host hold loop to release; torque is controlled explicitly."""

    def stop(self) -> None:
        """Freeze an engaged arm, as required by the Arm protocol."""
        with self._lock:
            if self.torque_enabled:
                self._park()
                self._target = self.read_state()
                self._last_action = time.monotonic()

    def emergency_stop(self) -> None:
        self.set_torque(False)

    def health(self) -> dict[str, Any]:
        with self._lock:
            return {**self._health, "torque": self.torque_enabled, "fault": self.fault}

    def _check_health(self) -> None:
        bus = self._robot.bus
        temps = bus.sync_read("Present_Temperature", normalize=False)
        volts = bus.sync_read("Present_Voltage", normalize=False)
        status = bus.sync_read("Status", normalize=False)
        for values in (temps, volts, status):
            if set(values) != set(SO101_JOINTS) or not all(math.isfinite(v) for v in values.values()):
                raise RuntimeError(f"{self.name}: incomplete or invalid health feedback")
        if any(t >= self.temperature_limit for t in temps.values()):
            raise RuntimeError(f"{self.name}: temperature limit {self.temperature_limit} C reached")
        if any(s for s in status.values()):
            raise RuntimeError(f"{self.name}: motor alarm {status}")
        if any(not 60 <= v <= 84 for v in volts.values()):
            raise RuntimeError(f"{self.name}: supply outside 6.0–8.4 V")
        self._health = {"temperature_c": temps, "voltage_v": {j: v / 10 for j, v in volts.items()}}

    def _watch(self) -> None:
        while not self._watch_stop.wait(1):
            failed = False
            with self._lock:
                try:
                    self._check_health()
                except Exception as exc:
                    self.fault = str(exc)
                    log.exception("%s hardware health check failed", self.name)
                    try:
                        self.emergency_stop()
                    except Exception:
                        log.exception("%s: torque release failed; cut motor power", self.name)
                    failed = True
            if failed:
                # Never acquire the supervisor's lock while holding our bus lock:
                # a command acquires those locks in the opposite order.
                if self.on_fault:
                    self.on_fault()
                return
