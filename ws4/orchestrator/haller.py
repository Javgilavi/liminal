"""Sunday demo backend: the green Haller's robot-LEFT arm through ``ws4/single_arm/run_single_arm.py`` (read-only).

The orchestrator reuses the single-arm runner's robot side instead of its attempt loop:
- ``connect_so101``: connect without LeRobot's ``connect()`` (no torque drop on an arm that holds a pose), the
  wrist camera rotated 180 deg and the head unrotated, both with OpenCV's default fourcc (camera parity with the VR
  recording, single_arm/README.md "Haller").
- ``SingleArmRobot``: the task arm in the runner's ``left`` slot, the other slot a never-commanded phantom (the
  robot-right arm stays parked and holding and is not connected by this process); bus retries and LeRobot's
  ``max_relative_target`` cap on every goal.
- ``build_guard`` + ``KeepOutGuard``: joint limits from the arm's calibration, a per-tick step limit, and the FK
  keep-out (table, base, fold, torso, mast, midline) from ``calibration/haller_right_workspace.json``.
- ``hold_here`` on every stop (freeze at the MEASURED pose), ``warm_up`` (pre-encode prompts on the server), and the
  torque-safe shutdown of its ``main``.
The per-tick stall and keep-out stops are rebuilt in ``monitors.py`` with the same defaults.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import WS4
from .monitors import KeepOutMonitor, StallMonitor

SINGLE_ARM = WS4 / "single_arm"
SLOT = "left"  # the task arm is the robot's left arm
DEFAULT_PORT = "/dev/haller_arm_right"  # operator-frame name: it is the ROBOT's left arm
DEFAULT_ROBOT_ID = "haller_right"  # LeRobot calibration so_follower/haller_right.json
DEFAULT_HEAD_CAM = "/dev/haller_cam_mast"
DEFAULT_WRIST_CAM = "/dev/v4l/by-path/pci-0000:00:14.0-usb-0:7.2:1.0-video-index0"
DEFAULT_KEEPOUT = SINGLE_ARM / "calibration" / "haller_right_workspace.json"


def import_single_arm():
    """``run_single_arm`` imports ``keepout`` as a top-level module, so its folder goes on ``sys.path``."""
    if str(SINGLE_ARM) not in sys.path:
        sys.path.insert(0, str(SINGLE_ARM))
    import run_single_arm  # noqa: PLC0415
    from keepout import KeepOut  # noqa: PLC0415

    return run_single_arm, KeepOut


@dataclass
class HallerRig:
    sar: Any  # the run_single_arm module
    so101: Any  # LeRobot SOFollower, or run_single_arm.FakeSO101
    robot: Any  # run_single_arm.SingleArmRobot (RobotInterface)
    guard: Any  # run_single_arm.KeepOutGuard
    monitors: list
    rest: np.ndarray  # measured pose at launch
    real: bool
    observe: bool

    def hold(self, reason: str = "") -> None:
        """Freeze at the measured pose so a blocked joint stops pushing (the runner alone would hold the command)."""
        if self.real and not self.observe:
            self.sar.hold_here(self.robot)

    def warm_up(self, backend: Any, target: Any, prompts: list[str]) -> None:
        self.sar.warm_up(backend, self.robot, target, prompts)

    def close(self) -> str:
        """Same rule as run_single_arm.main: switch torque off only if it was off at launch and the arm is back at its
        launch pose; otherwise keep holding (dropping an extended arm onto the box is worse)."""
        if not self.real:
            return "fake arm"
        at_rest = np.allclose(self.robot.joints()[:5], self.rest[:5], atol=8.0)
        if self.observe:
            self.sar.disconnect_so101(self.so101, torque_off=False)
            return "observe: torque unchanged"
        if at_rest and not getattr(self.so101, "ws4_launch_torque", False):
            self.sar.disconnect_so101(self.so101)
            return "at its launch pose: torque off"
        for camera in self.so101.cameras.values():
            camera.disconnect()
        self.so101.bus.disconnect(disable_torque=False)
        return "arm still holds with torque on; support it before switching the power off"


def build(cfg: Any) -> HallerRig:
    sar, keepout_cls = import_single_arm()
    ns = argparse.Namespace(
        port=cfg.port, robot_id=cfg.robot_id, scene_cam=cfg.head_cam, wrist_cam=cfg.wrist_cam, scene_rotate=0,
        wrist_rotate=cfg.wrist_rotate, scene_fourcc=cfg.scene_fourcc, wrist_fourcc=cfg.wrist_fourcc, parity_side=None,
        max_relative_target=cfg.max_relative_target, observe=cfg.observe, allow_torque_off=cfg.allow_torque_off,
        torque_limit=cfg.torque_limit, limit_margin=cfg.limit_margin, max_step=cfg.max_step,
        max_gripper_step=cfg.max_gripper_step,
    )
    real = cfg.robot == "so101"
    if real:
        if not cfg.head_cam:
            raise SystemExit("--head-cam is required on the real arm")
        so101 = sar.connect_so101(ns)
        calib = so101.calibration_fpath
    else:
        so101, calib = sar.FakeSO101(), None
    robot = sar.SingleArmRobot(so101, SLOT, send=not cfg.observe, max_relative_target=cfg.max_relative_target)
    rest = robot.joints()
    guard_cfg = sar.build_guard(ns, SLOT, calib, rest)
    keepout = keepout_cls.load(Path(cfg.keepout_config))
    if real and not keepout.calibrated:
        print("NOTE: table height not calibrated (keepout.py --touch); using the UNVERIFIED default", flush=True)
    guard = sar.KeepOutGuard(guard_cfg, keepout, SLOT, lambda: robot.last_q)
    monitors = [StallMonitor(SLOT, cfg.stall_deg, cfg.stall_ticks), KeepOutMonitor(guard, cfg.keepout_ticks)]
    return HallerRig(sar, so101, robot, guard, monitors, rest, real, cfg.observe)
