"""Run FLUX on the real XLeRobot (two SO-101 arms on the torso) through LeRobot's ``bi_so_follower``.

Why ``bi_so_follower`` and not the XLeRobot plugin: the plugin reports +-100 ranges by default and will not
connect without the mobile-base wheel motors (see ws4/research/flux_dual_arm.md, "Units"). ``bi_so_follower``
(lerobot@146d847:src/lerobot/robots/bi_so_follower/config_bi_so_follower.py:25-36) takes two ``SOFollowerConfig``
arms (``use_degrees=True`` by default, config_so_follower.py:42) plus unprefixed top-level cameras; per-arm
cameras get a ``left_`` / ``right_`` prefix. Observation keys are ``left_shoulder_pan.pos`` ... ``right_gripper.pos``.

Robot PC, with the policy server reachable on localhost (ssh -L, see GPU_RUNBOOK.md):
    PYTHONPATH=ws4 python -m flux.run_xlerobot --left-port /dev/ttyACM0 --right-port /dev/ttyACM1 \
        --head-cam <realsense-serial> --left-wrist-cam /dev/video2 --right-wrist-cam /dev/video4 \
        --left-task "push the red box out of the way" --seconds 20
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from .contract import ARM_JOINTS, ARMS, IMG_HEAD, IMG_LEFT_WRIST, IMG_RIGHT_WRIST, OBS_STATE

STATE_KEYS = [f"{arm}_{joint}.pos" for arm in ARMS for joint in ARM_JOINTS]
CAMERA_KEYS = {IMG_HEAD: "head", IMG_LEFT_WRIST: "left_wrist", IMG_RIGHT_WRIST: "right_wrist"}


class LeRobotBimanualRobot:
    """``RobotInterface`` over a connected LeRobot robot with ``bi_so_follower`` keys."""

    def __init__(self, robot: Any, camera_keys: dict[str, str] | None = None):
        self.robot = robot
        self.camera_keys = camera_keys or dict(CAMERA_KEYS)

    def get_observation(self) -> dict[str, Any]:
        raw = self.robot.get_observation()
        obs: dict[str, Any] = {OBS_STATE: np.array([float(raw[k]) for k in STATE_KEYS])}
        for ours, theirs in self.camera_keys.items():
            obs[ours] = np.asarray(raw[theirs])
        return obs

    def send_action(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        self.robot.send_action({k: float(v) for k, v in zip(STATE_KEYS, action, strict=True)})


def build_robot(args: argparse.Namespace) -> Any:
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
    from lerobot.robots.bi_so_follower.config_bi_so_follower import BiSOFollowerConfig
    from lerobot.robots.so_follower.config_so_follower import SOFollowerConfig
    from lerobot.robots.utils import make_robot_from_config

    def wrist(spec: str):
        index = int(spec) if spec.isdigit() else spec
        return {"wrist": OpenCVCameraConfig(index_or_path=index, fps=30, width=640, height=480)}

    config = BiSOFollowerConfig(
        id=args.robot_id,
        left_arm_config=SOFollowerConfig(port=args.left_port, cameras=wrist(args.left_wrist_cam), use_degrees=True),
        right_arm_config=SOFollowerConfig(port=args.right_port, cameras=wrist(args.right_wrist_cam), use_degrees=True),
        cameras={"head": RealSenseCameraConfig(serial_number_or_name=args.head_cam, fps=30, width=640, height=480)},
    )
    robot = make_robot_from_config(config)
    robot.connect()
    return robot


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--robot-id", default="xlerobot", help="calibration files <id>_left.json / <id>_right.json")
    parser.add_argument("--left-port", required=True)
    parser.add_argument("--right-port", required=True)
    parser.add_argument("--head-cam", required=True, help="RealSense serial number or name")
    parser.add_argument("--left-wrist-cam", required=True)
    parser.add_argument("--right-wrist-cam", required=True)
    parser.add_argument("--server", default="http://127.0.0.1:8765")
    parser.add_argument("--token", default=None)
    parser.add_argument("--package", default="so101", help="package name on the server, used by both arms")
    parser.add_argument("--left-package", default=None)
    parser.add_argument("--right-package", default=None)
    parser.add_argument("--left-task", default="")
    parser.add_argument("--right-task", default="")
    parser.add_argument("--scene-crop", choices=("full", "half"), default="full", help="head view per arm")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--calibration-limits", action="store_true", help="joint limits from LeRobot calibration")
    args = parser.parse_args(argv)

    from .cameras import CameraView
    from .dual_arm_runner import DualArmRunner, per_arm_target
    from .remote import RemoteBackend
    from .safety import SafetyConfig, SafetyGuard

    packages = {"left": args.left_package or args.package, "right": args.right_package or args.package}
    backends = {name: RemoteBackend(args.server, name, token=args.token) for name in set(packages.values())}
    crops = {"left": (0.0, 0.0, 0.5, 1.0), "right": (0.5, 0.0, 1.0, 1.0)}  # left arm on the image's left: UNVERIFIED
    targets = [
        per_arm_target(arm, packages[arm], CameraView(IMG_HEAD, crop=crops[arm] if args.scene_crop == "half" else None))
        for arm in ARMS
    ]
    robot = build_robot(args)
    guard = SafetyGuard(SafetyConfig())
    if args.calibration_limits:
        from lerobot.utils.constants import HF_LEROBOT_CALIBRATION

        from .safety import limits_from_lerobot_calibration

        for arm in ARMS:
            path = HF_LEROBOT_CALIBRATION / "robots" / "so_follower" / f"{args.robot_id}_{arm}.json"
            guard.config = guard.config.with_arm_limits(arm, *limits_from_lerobot_calibration(path))
    runner = DualArmRunner(LeRobotBimanualRobot(robot), targets, backends, guard, async_inference=True)
    for arm, task in (("left", args.left_task), ("right", args.right_task)):
        if task:
            runner.start(arm, task)
    try:
        runner.run(args.seconds, on_tick=lambda r: print(r.tick, r.status, r.latencies, flush=True))
    finally:
        runner.close()
        robot.disconnect()


if __name__ == "__main__":
    main()
