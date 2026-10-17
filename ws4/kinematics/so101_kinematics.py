"""SO-101 forward kinematics, units and the XLeRobot station (two arms on one torso).

Source of the kinematic chain: TheRobotStudio/SO-ARM100@a758567c3978dfeefe282ede0500085a48fe8f78,
``Simulation/SO101/so101_new_calib.urdf`` (joint lines cited below). "New calibration" means each joint's
zero is the middle of its range (``Simulation/SO101/README.md``), which is the same zero LeRobot uses in
``MotorNormMode.DEGREES`` (``lerobot==0.5.1``, ``motors/motors_bus.py:858-861``: ``(raw - mid) * 360 / 4095``
with ``mid`` the middle of the calibrated range). LeRobot's own kinematics converts those degrees to
radians with a plain ``np.deg2rad`` against this URDF (``lerobot/model/kinematics.py:66-71``), so we do too.

The FK matches MuJoCo on ``so101_new_calib.xml`` to 4e-6 m (see ``tests/test_so101_kinematics.py``).

Frames used throughout WS4 kinematics:

* ``base``: the URDF ``base_link`` of one arm. z up along the pan axis, x is where the arm points at pan 0.
* ``flange``: the URDF ``gripper_frame_link`` (LeRobot's default IK target frame). Its z axis is the
  approach direction of the stock gripper, it sits between the stock jaws, 7.9 mm off the roll axis.
* ``tcp``: ``flange`` pushed along its z axis by ``tcp_extra`` metres, for the custom fin-ray fingers.
* ``station``: origin on the table under the middle of the torso, x forward (towards the workspace),
  y to the robot's left, z up. Arm bases are placed in it by :class:`Station`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .se3 import inv_pose, make_pose, rot_z, rpy_matrix

JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
N_JOINTS = 5

# (xyz, rpy) of each revolute joint's origin in its parent link; every axis is local +z.
# so101_new_calib.urdf lines 434-439, 414-419, 394-399, 373-378, 353-358.
_JOINT_ORIGINS = (
    ((0.0388353, -8.97657e-09, 0.0624), (3.14159, 4.18253e-17, -3.14159)),
    ((-0.0303992, -0.0182778, -0.0542), (-1.5708, -1.5708, 0.0)),
    ((-0.11257, -0.028, 1.73763e-16), (-3.63608e-16, 8.74301e-16, 1.5708)),
    ((-0.1349, 0.0052, 3.62355e-17), (4.02456e-15, 8.67362e-16, -1.5708)),
    ((5.55112e-17, -0.0611, 0.0181), (1.5708, 0.0486795, 3.14159)),
)
# gripper_frame_joint (fixed, gripper_link -> gripper_frame_link), lines 302-306.
_FLANGE_ORIGIN = ((-0.0079, -0.000218121, -0.0981274), (0.0, 3.14159, 0.0))
# Joint limits in radians, same lines (lower, upper).
JOINT_LIMITS = np.array(
    [
        [-1.91986, 1.91986],
        [-1.74533, 1.74533],
        [-1.69, 1.69],
        [-1.65806, 1.65806],
        [-2.74385, 2.84121],
    ]
)
# Gripper joint limits (rad), lines 333-338. Only used to map the LeRobot 0..100 gripper to an angle.
GRIPPER_LIMITS = (-0.174533, 1.74533)

# Degrees per LeRobot RANGE_M100_100 unit, from Oscar's follower calibration
# (~/.cache/huggingface/lerobot/calibration/robots/so_follower/haller_follower.json, 2026-05-22):
# half the calibrated tick range * 360 / 4095 / 100. Used only for datasets recorded before LeRobot's
# DEGREES mode. Every arm is calibrated differently, so this is approximate by construction.
M100_DEG_PER_UNIT = np.array([1.1956, 1.0440, 0.9723, 1.0277, 1.8000])


def _origin(xyz, rpy) -> np.ndarray:
    return make_pose(rpy_matrix(*rpy), np.array(xyz))


_T_ORIGINS = np.stack([_origin(*o) for o in _JOINT_ORIGINS])
_T_FLANGE = _origin(*_FLANGE_ORIGIN)


def lerobot_deg_to_rad(q_deg: np.ndarray) -> np.ndarray:
    """LeRobot ``MotorNormMode.DEGREES`` joint values -> URDF radians (same zero, same sign)."""
    return np.deg2rad(np.asarray(q_deg, dtype=float))


def rad_to_lerobot_deg(q_rad: np.ndarray) -> np.ndarray:
    return np.rad2deg(np.asarray(q_rad, dtype=float))


def lerobot_m100_to_rad(q_m100: np.ndarray, deg_per_unit: np.ndarray = M100_DEG_PER_UNIT) -> np.ndarray:
    """LeRobot ``RANGE_M100_100`` (older recordings) -> radians, assuming a mid-range zero. Approximate."""
    return np.deg2rad(np.asarray(q_m100, dtype=float)[..., :N_JOINTS] * deg_per_unit)


def lerobot_gripper_to_open(g: np.ndarray) -> np.ndarray:
    """LeRobot gripper (``RANGE_0_100``: 0 closed, 100 open) -> EE50 convention (0 closed, 1 open)."""
    return np.clip(np.asarray(g, dtype=float) / 100.0, 0.0, 1.0)


def open_to_lerobot_gripper(g: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(g, dtype=float), 0.0, 1.0) * 100.0


# Fin-ray TCP from ws4/sim/station_geometry.yaml (tcp.pos_in_gripper, verified: false): on the wrist-roll
# axis, 0.140 m out along the approach, midway between the fingertip pads of the 123.4 mm TPU fingers.
FINRAY_TCP_IN_GRIPPER = (0.0, 0.0, -0.140)


@dataclass(frozen=True)
class ArmModel:
    """One SO-101 arm.

    The TCP keeps the flange orientation (z = approach) and sits at ``tcp_in_gripper`` in the URDF
    ``gripper_link`` frame (default: the stock flange point), then ``tcp_extra`` further along the approach.
    """

    tcp_extra: float = 0.0
    tcp_in_gripper: tuple[float, float, float] | None = None
    limits: np.ndarray = field(default_factory=lambda: JOINT_LIMITS.copy())

    @property
    def T_flange_tcp(self) -> np.ndarray:
        p = np.array([0.0, 0.0, self.tcp_extra])
        if self.tcp_in_gripper is not None:
            # express the requested point in the flange frame: p_f = R_gf^T (p_g - t_gf)
            p = p + _T_FLANGE[:3, :3].T @ (np.asarray(self.tcp_in_gripper, dtype=float) - _T_FLANGE[:3, 3])
        return make_pose(np.eye(3), p)

    def joint_frames(self, q: np.ndarray) -> np.ndarray:
        """Frames of the 5 joints (after rotation) in ``base``: (..., 5, 4, 4)."""
        q = np.asarray(q, dtype=float)
        batch = q.shape[:-1]
        out = np.empty(batch + (N_JOINTS, 4, 4))
        T = np.broadcast_to(np.eye(4), batch + (4, 4)).copy()
        Rz = make_pose(rot_z(q), np.zeros(batch + (N_JOINTS, 3)))
        for i in range(N_JOINTS):
            T = T @ _T_ORIGINS[i] @ Rz[..., i, :, :]
            out[..., i, :, :] = T
        return out

    def fk(self, q: np.ndarray, frame: str = "tcp") -> np.ndarray:
        """Pose of ``frame`` (``tcp`` or ``flange``) in ``base`` for joints ``q`` (..., 5) in radians."""
        T5 = self.joint_frames(q)[..., -1, :, :]
        T = T5 @ _T_FLANGE
        if frame == "tcp":
            T = T @ self.T_flange_tcp
        elif frame != "flange":
            raise ValueError(f"unknown frame {frame!r}")
        return T

    def joint_frames_single(self, q: np.ndarray) -> list[np.ndarray]:
        """Unbatched :meth:`joint_frames` without the stacking overhead (the IK hot path)."""
        frames = []
        T = np.eye(4)
        for i in range(N_JOINTS):
            c, s = math.cos(q[i]), math.sin(q[i])
            Rz = np.array([[c, -s, 0.0, 0.0], [s, c, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
            T = T @ _T_ORIGINS[i] @ Rz
            frames.append(T)
        return frames

    def fk_single(self, q: np.ndarray) -> np.ndarray:
        return self.joint_frames_single(q)[-1] @ self._T_tool

    @property
    def _T_tool(self) -> np.ndarray:
        return _T_FLANGE @ self.T_flange_tcp

    def jacobian(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Geometric Jacobian of the TCP in ``base`` (rows: v then w), and the TCP pose. Single q only."""
        frames = self.joint_frames_single(q)
        T = frames[-1] @ self._T_tool
        p = T[:3, 3]
        z = np.array([f[:3, 2] for f in frames])
        o = np.array([f[:3, 3] for f in frames])
        J = np.empty((6, N_JOINTS))
        J[:3] = np.cross(z, p - o).T
        J[3:] = z.T
        return J, T

    def within_limits(self, q: np.ndarray, tol: float = 1e-9) -> np.ndarray:
        return np.all((q >= self.limits[:, 0] - tol) & (q <= self.limits[:, 1] + tol), axis=-1)

    def clip(self, q: np.ndarray) -> np.ndarray:
        return np.clip(q, self.limits[:, 0], self.limits[:, 1])


# ---------------------------------------------------------------------------------------------------------
# Station: two arms on the XLeRobot torso.

# Defaults until ws4/sim publishes station_geometry.yaml. Lateral spacing from the XLeRobot URDF
# (Vector-Wangel/XLeRobot@b017b5e, simulation/xlerobot/xlerobot.urdf: Base at y=-0.11 and Base_2 at y=+0.11
# with the pan joint a further 0.0452 m outboard, so pan axes ~0.31 m apart; the MuJoCo model
# simulation/mujoco/xlerobot.xml puts them at y=+-0.15). Mount height and forward offset are UNVERIFIED
# guesses from the photo (the torso box is roughly 10 cm tall); both arms face forward.
DEFAULT_ARM_Y = 0.155
DEFAULT_ARM_X = 0.0
DEFAULT_ARM_Z = 0.10
DEFAULT_TORSO = {"center": [0.0, 0.0, 0.05], "half_extents": [0.09, 0.22, 0.05]}  # UNVERIFIED box
DEFAULT_MAST = {"p0": [-0.02, 0.0, 0.10], "p1": [-0.02, 0.0, 0.55], "radius": 0.04}  # UNVERIFIED

STATION_GEOMETRY_PATH = Path(__file__).resolve().parents[1] / "sim" / "station_geometry.yaml"


def _val(x):
    return x["value"] if isinstance(x, dict) and "value" in x else x


def quat_wxyz_to_matrix(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


@dataclass
class Station:
    """Two arms in the ``station`` frame. ``T_station_base[name]`` places each arm's ``base_link``."""

    T_station_base: dict[str, np.ndarray]
    arm: ArmModel = field(default_factory=ArmModel)
    table_z: float = 0.0
    torso: dict = field(default_factory=lambda: dict(DEFAULT_TORSO))
    mast: dict = field(default_factory=lambda: dict(DEFAULT_MAST))
    source: str = "defaults (UNVERIFIED)"

    @classmethod
    def default(cls, tcp_extra: float = 0.0) -> Station:
        T = {
            "left": make_pose(np.eye(3), np.array([DEFAULT_ARM_X, DEFAULT_ARM_Y, DEFAULT_ARM_Z])),
            "right": make_pose(np.eye(3), np.array([DEFAULT_ARM_X, -DEFAULT_ARM_Y, DEFAULT_ARM_Z])),
        }
        return cls(T_station_base=T, arm=ArmModel(tcp_extra=tcp_extra))

    @classmethod
    def load(cls, path: Path | None = None, tcp_extra: float = 0.0, finray: bool = True) -> Station:
        """Read ``ws4/sim/station_geometry.yaml`` (ws4-sim schema v1) if it exists, else :meth:`default`.

        Uses ``arms.<side>.base_pos`` / ``base_quat`` ([w, x, y, z]), ``table.top_z``, the torso bounding box
        and mast top, and ``tcp.pos_in_gripper`` (when ``finray``). Every leaf is ``{value, verified, source}``;
        plain values are accepted too. Missing keys keep the defaults; parse failures are reported in ``source``.
        """
        path = STATION_GEOMETRY_PATH if path is None else Path(path)
        st = cls.default(tcp_extra)
        if not path.exists():
            return st
        try:
            import yaml

            data = yaml.safe_load(path.read_text()) or {}
            arms = data.get("arms", {})
            for name in ("left", "right"):
                a = arms.get(name) or {}
                if "base_pos" in a:
                    quat = _val(a.get("base_quat", [1.0, 0.0, 0.0, 0.0]))
                    st.T_station_base[name] = make_pose(quat_wxyz_to_matrix(quat), np.array(_val(a["base_pos"]), float))
                elif "base_pose" in a:  # simple schema: {xyz, rpy}
                    pose = a["base_pose"]
                    R = rpy_matrix(*pose.get("rpy", [0, 0, 0]))
                    st.T_station_base[name] = make_pose(R, np.array(pose["xyz"], float))
            table = data.get("table", {})
            st.table_z = float(_val(table.get("top_z", data.get("table_z", st.table_z))))
            torso = data.get("torso", {})
            if "armbase_box_min" in torso:
                lo, hi = np.array(_val(torso["armbase_box_min"])), np.array(_val(torso["armbase_box_max"]))
                st.torso = {"center": ((lo + hi) / 2).tolist(), "half_extents": ((hi - lo) / 2).tolist()}
            if "mast_top_z" in torso:
                base_z = float(_val(torso.get("centre_shell_top_z", 0.135)))
                st.mast = {"p0": [0.0, 0.0, 0.0], "p1": [0.0, 0.0, float(_val(torso["mast_top_z"]))], "radius": 0.05,
                           "shell_top_z": base_z}
            tcp = (data.get("tcp") or {}).get("pos_in_gripper")
            if finray and tcp is not None:
                st.arm = ArmModel(tcp_extra=tcp_extra, tcp_in_gripper=tuple(float(x) for x in _val(tcp)))
            st.source = str(path)
        except Exception as exc:  # malformed or different schema: keep defaults, say why
            st.source = f"defaults (could not parse {path}: {exc})"
        return st

    def fk(self, name: str, q: np.ndarray, frame: str = "tcp") -> np.ndarray:
        return self.T_station_base[name] @ self.arm.fk(q, frame)

    def to_base(self, name: str, T_station: np.ndarray) -> np.ndarray:
        return inv_pose(self.T_station_base[name]) @ T_station

    def to_station(self, name: str, T_base: np.ndarray) -> np.ndarray:
        return self.T_station_base[name] @ T_base
