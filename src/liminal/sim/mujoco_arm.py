"""MuJoCo-backed arms sharing one physics world (owned by WS4).

`MujocoWorld` owns the model and steps physics in a background thread; each
`MujocoArm` is a view over a subset of actuators selected by name prefix, so one MJCF
with `left_*` and `right_*` actuators (dual SO-101, or XLeRobot's two arms) gives two
`Arm`s with the exact interface the real hardware has.

Candidate scenes (check licences/quality before adopting; see docs/research/04):
  - a community dual-SO-101 MuJoCo environment
  - XLeRobot's simulation assets
  - MuJoCo Menagerie / LeRobot SO-101 MJCF for single-arm
"""

from __future__ import annotations

import atexit
import math
import threading
import time
from typing import Any

from liminal.core.types import JointPositions


class MujocoWorld:
    def __init__(self, xml_path: str, sim_hz: float = 500.0, render_cameras: tuple[str, ...] = ()):
        import mujoco

        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.sim_hz = sim_hz
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._renderer: Any = None
        self.render_cameras = render_cameras

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="mujoco", daemon=True)
        self._thread.start()
        # Nothing else stops the world (arms share it, disconnect is a no-op): without this the daemon
        # thread can still be in mj_step while the interpreter frees the model, a segfault at exit.
        atexit.register(self.stop)

    def _run(self) -> None:
        dt = 1.0 / self.sim_hz
        while not self._stop.is_set():
            with self.lock:
                self.mj.mj_step(self.model, self.data)
            time.sleep(dt)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def actuator_ids(self, prefix: str) -> list[int]:
        ids = [i for i in range(self.model.nu) if (self.model.actuator(i).name or "").startswith(prefix)]
        if not ids:
            raise ValueError(f"no actuators with prefix {prefix!r}")
        return ids

    def render(self, camera: str, width: int = 640, height: int = 480) -> Any:
        if self._renderer is None:
            self._renderer = self.mj.Renderer(self.model, height=height, width=width)
        with self.lock:
            self._renderer.update_scene(self.data, camera=camera)
            return self._renderer.render()


class MujocoArm:
    """Same units as the real SO-101 (`hardware/so101.py`, LeRobot degrees mode), so robot configs, task
    trees and policies are shared with hardware: arm joints in degrees (the MJCF's zero is the middle of the
    range, like LeRobot's calibration), gripper 0-100 % across its joint range (100 = open). The MJCF works
    in radians; the conversion is the one examples/dual_so101_sim/twin/expert.py uses."""

    def __init__(self, name: str, world: MujocoWorld, actuator_prefix: str):
        self.name = name
        self.world = world
        self.ids = world.actuator_ids(actuator_prefix)
        self.num_joints = len(self.ids)
        model = world.model
        # Position actuators: the joint each one drives.
        joints = [int(model.actuator_trnid[i, 0]) for i in self.ids]
        self.qpos_adr = [int(model.jnt_qposadr[j]) for j in joints]
        # Gripper: joint range (radians) that maps to 0-100 %.
        self.grip_range = {k: (float(model.jnt_range[j][0]), float(model.jnt_range[j][1]))
                           for k, (i, j) in enumerate(zip(self.ids, joints, strict=True))
                           if model.actuator(i).name.removeprefix(actuator_prefix) == "gripper"}

    def _to_robot(self, k: int, rad: float) -> float:
        if k in self.grip_range:
            lo, hi = self.grip_range[k]
            return (rad - lo) / (hi - lo) * 100.0
        return math.degrees(rad)

    def _to_mujoco(self, k: int, value: float) -> float:
        if k in self.grip_range:
            lo, hi = self.grip_range[k]
            return lo + min(max(value, 0.0), 100.0) / 100.0 * (hi - lo)
        return math.radians(value)

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def read_state(self) -> JointPositions:
        with self.world.lock:
            return [self._to_robot(k, float(self.world.data.qpos[a])) for k, a in enumerate(self.qpos_adr)]

    def send_action(self, positions: JointPositions) -> None:
        if len(positions) != self.num_joints:
            raise ValueError(f"{self.name}: expected {self.num_joints} joints, got {len(positions)}")
        with self.world.lock:
            for k, (i, p) in enumerate(zip(self.ids, positions, strict=True)):
                self.world.data.ctrl[i] = self._to_mujoco(k, float(p))

    def hold(self, positions: JointPositions | None = None) -> None:
        # Position actuators already hold the last ctrl target.
        self.send_action(positions or self.read_state())

    def release_hold(self) -> None:
        pass

    def stop(self) -> None:
        self.hold()


class MujocoCamera:
    def __init__(self, name: str, world: MujocoWorld, mj_camera: str, width: int = 640, height: int = 480):
        self.name, self.world, self.mj_camera, self.width, self.height = name, world, mj_camera, width, height

    def read(self) -> Any:
        return self.world.render(self.mj_camera, self.width, self.height)
