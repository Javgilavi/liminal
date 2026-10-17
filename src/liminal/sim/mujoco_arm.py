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
    """Units are whatever the MJCF actuators use (usually radians). Map to the real
    arm's units (LeRobot degrees / normalized range) in configs, not here."""

    def __init__(self, name: str, world: MujocoWorld, actuator_prefix: str):
        self.name = name
        self.world = world
        self.ids = world.actuator_ids(actuator_prefix)
        self.num_joints = len(self.ids)
        # Position actuators: the joint each one drives.
        self.qpos_adr = [int(world.model.jnt_qposadr[world.model.actuator_trnid[i, 0]]) for i in self.ids]

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def read_state(self) -> JointPositions:
        with self.world.lock:
            return [float(self.world.data.qpos[a]) for a in self.qpos_adr]

    def send_action(self, positions: JointPositions) -> None:
        with self.world.lock:
            for i, p in zip(self.ids, positions, strict=True):
                self.world.data.ctrl[i] = p

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
