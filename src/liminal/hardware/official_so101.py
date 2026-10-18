"""Single-owner LeRobot SO-101 adapter; no Haller imports or trajectory shaping."""
from __future__ import annotations

import socket
from pathlib import Path

from liminal.core.types import SO101_JOINTS


class OfficialSO101:
    num_joints = 6

    def __init__(self, cfg):
        # Hardware dependencies stay out of imports and mock tests.
        from lerobot.cameras.configs import Cv2Rotation
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        self.name = cfg["arm"]
        self.current = None
        self.robot = SO101Follower(SO101FollowerConfig(
            port=cfg["expected_port"], id=cfg["calibration_id"], use_degrees=True,
            max_relative_target=None, disable_torque_on_disconnect=True,
            cameras={
                "scene": OpenCVCameraConfig(index_or_path=Path(cfg["scene_device"]),
                                           fps=30, width=640, height=480),
                "wrist": OpenCVCameraConfig(index_or_path=Path(cfg["wrist_device"]),
                                           fps=30, width=640, height=480, fourcc="MJPG",
                                           rotation=Cv2Rotation.ROTATE_180),
            }))
        self.engaged = False

    def connect_read_only(self):
        for port in (8000, 8781):
            with socket.socket() as sock:
                if sock.connect_ex(("127.0.0.1", port)) == 0:
                    raise RuntimeError("Stop Haller before opening the LeRobot motor/camera owner")
        self.robot.bus.connect()
        try:
            if not self.robot.is_calibrated:
                raise RuntimeError("saved calibration disagrees with motors; no calibration was changed")
            if any(self.robot.bus.read("Torque_Enable", j, normalize=False) for j in SO101_JOINTS):
                raise RuntimeError("motors must already be parked with torque OFF")
            for camera in self.robot.cameras.values():
                camera.connect()
        except BaseException:
            self.close()
            raise

    def engage(self):
        # Avoid enabling against an old servo goal. This initialization is the
        # only addition to stock configure(); runtime targets remain unchanged.
        raw = self.robot.bus.sync_read("Present_Position", normalize=False)
        self.robot.bus.sync_write("Goal_Position", raw, normalize=False)
        self.engaged = True  # cleanup must release even if configure partly fails
        self.robot.configure()

    def limits(self):
        lo, hi = [], []
        for joint in SO101_JOINTS:
            cal = self.robot.calibration[joint]
            span = (cal.range_max - cal.range_min) * 180 / 4095
            lo.append(0.0 if joint == "gripper" else -span)
            hi.append(100.0 if joint == "gripper" else span)
        return lo, hi

    def observe(self):
        obs = self.robot.get_observation()
        self.current = [float(obs[j + ".pos"]) for j in SO101_JOINTS]
        loads = self.robot.bus.sync_read("Present_Load", normalize=False)
        return obs, {j: float(loads[j]) / 1000 for j in SO101_JOINTS}

    def read_state(self):
        if self.current is None:
            raise RuntimeError("no measured state")
        return list(self.current)

    def send_action(self, target):
        if not self.engaged:
            raise RuntimeError("not engaged")
        sent = self.robot.send_action(dict(zip((j + ".pos" for j in SO101_JOINTS), target, strict=True)))
        if any(abs(sent[j + ".pos"] - q) > 1e-5 for j, q in zip(SO101_JOINTS, target, strict=True)):
            raise RuntimeError("LeRobot altered a target; stopping instead of silently changing trajectory")

    def stop(self):
        if not self.engaged:
            return
        failed = []
        for joint in SO101_JOINTS:
            try:
                self.robot.bus.write("Torque_Enable", joint, 0, normalize=False, num_retry=2)
                if self.robot.bus.read("Torque_Enable", joint, normalize=False):
                    failed.append(joint)
            except Exception:
                failed.append(joint)
        self.engaged = False
        if failed:
            raise RuntimeError(f"TORQUE RELEASE UNCONFIRMED: {failed}; support arm and cut motor power")

    def close(self):
        try:
            self.stop()
        finally:
            for camera in self.robot.cameras.values():
                if camera.is_connected:
                    camera.disconnect()
            if self.robot.bus.is_connected:
                self.robot.bus.disconnect(disable_torque=False)
