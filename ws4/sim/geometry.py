"""Read station_geometry.yaml and derive poses from it (numpy only, no MuJoCo)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

SIM_DIR = Path(__file__).resolve().parent
GEOMETRY_YAML = SIM_DIR / "station_geometry.yaml"
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
SIDES = ("left", "right")


@lru_cache(maxsize=4)
def _load(path: str) -> dict:
    import yaml

    return yaml.safe_load(Path(path).read_text())


def load_geometry(path: str | Path = GEOMETRY_YAML) -> dict:
    """The parsed yaml (cached). Leaves are {value, verified, source} dicts; use `val` to read them."""
    return _load(str(path))


def val(entry):
    """Value of a {value, verified, source} leaf (plain values pass through)."""
    return entry["value"] if isinstance(entry, dict) and "value" in entry else entry


def quat_from_mat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion [w, x, y, z] with w >= 0."""
    R = np.asarray(R, dtype=float)
    t = np.trace(R)
    if t > 0:
        s = 2.0 * np.sqrt(1.0 + t)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def mat_from_quat(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def rot_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def mujoco_to_opencv(R_mj: np.ndarray) -> np.ndarray:
    """MuJoCo camera axes (look -z, y up) -> OpenCV camera axes (look +z, y down)."""
    return R_mj @ np.diag([1.0, -1.0, -1.0])


def head_camera_pose(geom: dict | None = None, pitch_down_deg: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """World pose of the head colour camera in MuJoCo camera convention: (pos, R) with R columns = camera axes.

    The camera sits in the gimbal pitch holder, which the mock tilt servo holds at a fixed pitch about the
    tilt pivot (a +y axis). At zero pitch the optical axis is +x (forward, horizontal).
    """
    g = geom or load_geometry()
    h = g["cameras"]["head"]
    pitch = np.deg2rad(val(h["pitch_down_deg"]) if pitch_down_deg is None else pitch_down_deg)
    pivot = np.array(val(h["tilt_pivot"]), dtype=float)
    rel0 = np.array(val(h["optical_centre_rel_pivot"]), dtype=float)
    Rp = rot_y(pitch)  # positive rotation about +y turns +x toward -z, i.e. pitches the view down
    pos = pivot + Rp @ rel0
    # MuJoCo camera at zero pitch: looks along +x, image right = -y, image up = +z.
    R0 = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    return pos, Rp @ R0


def wrist_camera_pose_in_gripper(side: str, geom: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Pose of `<side>_wrist` in the SO-101 gripper body frame, MuJoCo camera convention (pos, R)."""
    g = geom or load_geometry()
    c = g["cameras"][f"{side}_wrist"]
    pos = np.array(val(c["pos_in_gripper"]), dtype=float)
    fwd = np.array(val(c["optical_axis_in_gripper"]), dtype=float)
    fwd /= np.linalg.norm(fwd)
    # Image right = gripper +x (toward the moving finger), so the image is upright when the arm is at q = 0.
    right = np.array([1.0, 0.0, 0.0]) - fwd * fwd[0]
    right /= np.linalg.norm(right)
    up = np.cross(-fwd, right)
    return pos, np.column_stack([right, up, -fwd])


def intrinsics(width: int, height: int, fovy_deg: float) -> dict:
    """Pinhole intrinsics of a MuJoCo camera (square pixels, principal point at the image centre)."""
    f = 0.5 * height / np.tan(np.deg2rad(fovy_deg) / 2)
    return {"fx": f, "fy": f, "cx": (width - 1) / 2, "cy": (height - 1) / 2, "width": width, "height": height}
