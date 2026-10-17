"""SO-101 follower arm backed by LeRobot (owned by WS5).

LeRobot's module paths and observation keys have changed between releases. This
adapter isolates that: if `import` fails or keys differ, fix it here only.
Verify against your installed version with `lerobot-calibrate --help`.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from liminal.core.types import SO101_JOINTS, JointPositions


class SO101Arm:
    num_joints = len(SO101_JOINTS)

    def __init__(
        self,
        name: str,
        port: str,
        robot_id: str | None = None,
        hold_rate_hz: float = 30.0,
        max_relative_target: float | None = None,
        use_degrees: bool | None = None,
    ):
        self.name = name
        self.port = port
        self.robot_id = robot_id or name
        self.hold_rate_hz = hold_rate_hz
        self.max_relative_target = max_relative_target
        self.use_degrees = use_degrees
        self._robot: Any = None
        self._lock = threading.Lock()
        self._hold_target: JointPositions | None = None
        self._hold_thread: threading.Thread | None = None
        self._hold_stop = threading.Event()

    # -- lifecycle ---------------------------------------------------------------------

    def connect(self) -> None:
        try:
            from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
        except ImportError:
            from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
        units = {} if self.use_degrees is None else {"use_degrees": self.use_degrees}
        cfg = SO101FollowerConfig(port=self.port, id=self.robot_id,
                                  max_relative_target=self.max_relative_target, **units)
        self._robot = SO101Follower(cfg)
        self._robot.connect()

    def disconnect(self) -> None:
        self.release_hold()
        if self._robot is not None:
            self._robot.disconnect()
            self._robot = None

    # -- io ----------------------------------------------------------------------------

    def read_state(self) -> JointPositions:
        with self._lock:
            obs = self._robot.get_observation()
        return [float(obs[f"{j}.pos"]) for j in SO101_JOINTS]

    def send_action(self, positions: JointPositions) -> None:
        if len(positions) != self.num_joints:
            raise ValueError(f"{self.name}: expected {self.num_joints} joints, got {len(positions)}")
        with self._lock:
            self._robot.send_action({f"{j}.pos": float(p) for j, p in zip(SO101_JOINTS, positions, strict=True)})

    # -- holding -----------------------------------------------------------------------

    def hold(self, positions: JointPositions | None = None) -> None:
        """Position-hold in a background thread so the arm keeps its pose while the
        shared FLUX model is busy with the other arm. This is *position* holding, not
        torque control - keep physical loads light (see docs/architecture.md)."""
        self._hold_target = list(positions) if positions else self.read_state()
        if self._hold_thread and self._hold_thread.is_alive():
            return
        self._hold_stop.clear()
        self._hold_thread = threading.Thread(target=self._hold_loop, name=f"hold-{self.name}", daemon=True)
        self._hold_thread.start()

    def _hold_loop(self) -> None:
        period = 1.0 / self.hold_rate_hz
        while not self._hold_stop.is_set():
            target = self._hold_target
            if target is not None:
                self.send_action(target)
            time.sleep(period)

    def release_hold(self) -> None:
        self._hold_stop.set()
        if self._hold_thread:
            self._hold_thread.join(timeout=1.0)
        self._hold_thread = None
        self._hold_target = None

    @property
    def holding(self) -> bool:
        return self._hold_thread is not None and self._hold_thread.is_alive() and not self._hold_stop.is_set()

    def stop(self) -> None:
        self.hold()
