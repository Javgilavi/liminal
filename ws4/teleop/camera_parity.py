"""Make deployment frames match recorded frames for the station cameras.

The haller recorder does three things to camera frames that plain LeRobot `OpenCVCamera` does not
(haller_ws hmi/backend/haller_hmi/cameras.py `_read_rotated` / `_apply_color_gains`, v4l2_controls.py):
  1. rotates each gripper camera 180 degrees (both modules are mounted upside down),
  2. multiplies R, G, B by per-camera `color_gains`, in float, clipped to 0..255,
  3. writes V4L2 controls (fixed exposure, gain, white balance) every time the device opens; they do not survive
     a replug and LeRobot never sets them.
A policy trained on the recorded frames and fed raw OpenCV frames at deployment sees upside-down, colour-shifted
wrist views. Use this module in the deployment loop:

    from teleop.camera_parity import StationCameras
    cams = StationCameras.load()                       # reads ws4/teleop/configs/station-real.yaml
    cams.apply_controls()                              # once, after LeRobot opened the devices
    frame = cams.transform("left_wrist", rgb_frame)    # FLUX/robot-side key, RGB HxWx3 uint8

Keys here are ROBOT sides (ws4/FLUX): `left_wrist` is the robot's left gripper camera, which haller names
`wrist_right` (/dev/haller_cam_wrist_right). See ws4/teleop/calib.py for the side table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CONFIG = Path(__file__).resolve().parent / "configs" / "station-real.yaml"
# robot-side FLUX key -> haller dataset_key recorded
ROBOT_TO_HALLER_KEY = {"head": "head", "left_wrist": "right_wrist", "right_wrist": "left_wrist"}


@dataclass(frozen=True)
class CameraSpec:
    key: str                     # robot-side key: head, left_wrist, right_wrist
    device: str
    width: int
    height: int
    rotation: int = 0
    color_gains: tuple[float, float, float] | None = None
    controls: dict[str, int] = field(default_factory=dict)


def rotate(frame: np.ndarray, rotation: int) -> np.ndarray:
    """cv2.rotate equivalents: 90 = clockwise, 270 = counter-clockwise (haller cameras._ROTATIONS)."""
    if rotation in (0, None):
        return frame
    k = {90: -1, 180: 2, 270: 1}[int(rotation)]
    return np.ascontiguousarray(np.rot90(frame, k=k))


def apply_gains(frame: np.ndarray, gains) -> np.ndarray:
    if not gains:
        return frame
    out = frame.astype(np.float32) * np.asarray(gains, dtype=np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


@dataclass
class StationCameras:
    specs: dict[str, CameraSpec]

    @classmethod
    def load(cls, path: Path = CONFIG) -> StationCameras:
        import yaml

        cfg = yaml.safe_load(Path(path).read_text())
        by_key = {c.get("dataset_key") or c["id"]: c for c in cfg["cameras"] if c.get("record", True)}
        specs = {}
        for robot_key, haller_key in ROBOT_TO_HALLER_KEY.items():
            c = by_key[haller_key]
            gains = tuple(float(g) for g in c["color_gains"]) if c.get("color_gains") else None
            specs[robot_key] = CameraSpec(robot_key, str(c["index_or_path"]), int(c["width"]), int(c["height"]),
                                          int(c.get("rotation", 0) or 0), gains, dict(c.get("controls") or {}))
        return cls(specs)

    def transform(self, key: str, frame_rgb: np.ndarray) -> np.ndarray:
        s = self.specs[key]
        return apply_gains(rotate(frame_rgb, s.rotation), s.color_gains)

    def apply_controls(self, setter=None) -> dict[str, dict[str, int]]:
        """Write each camera's V4L2 controls (haller_ws v4l2_controls.apply, standard library only)."""
        if setter is None:
            import sys

            sys.path.insert(0, str(Path.home() / "haller_ws/hmi/backend"))
            from haller_hmi import v4l2_controls

            setter = v4l2_controls.apply
        return {k: setter(s.device, s.controls) for k, s in self.specs.items() if s.controls}
