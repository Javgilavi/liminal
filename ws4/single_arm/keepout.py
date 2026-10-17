"""Geometric keep-out for one SO-101 arm: refuse commanded targets that drive the gripper into the table, the arm's
own base, the station (XLeRobot torso box, head mast) or across the midline toward the other arm, checked by forward
kinematics on every tick.

Why: in A0 on the red arm the wrist folded back toward the robot's base and the elbow pushed against it until the
servo was damaged (2026-10-17, attempts A0 #2 and #3). Joint limits and a step limit cannot see that; FK can.

Kinematics: ``ws4/kinematics/so101_kinematics.py`` (stock SO-101 URDF chain, LeRobot degrees with zero at mid-range).
Checked points: the TCP (stock flange between the jaws, or the fin-ray TCP from ``ws4/sim/station_geometry.yaml``)
and the wrist flex joint. Arm-local zones use the arm's URDF ``base_link`` frame (pan axis vertical through
(0.0388, 0)); station zones use the station frame (table top z = 0, x forward, y to the robot's left), with the arm
base placed at ``base_pos`` and no rotation (both Haller arms face +x, station_geometry.yaml ``base_quat``).

A target is refused when it violates a zone AND goes deeper into it than the measured pose already is, so the arm can
always move out of a zone (for example from a rest pose near the base) but never further in:

(a) table: TCP below ``table_z + table_margin`` or wrist below ``table_z + wrist_table_margin`` (station z; the
    Haller pen task grasps a pen lying on the mat, so its TCP margin is negative and the wrist margin carries the
    protection);
(b) base: TCP inside a cylinder of ``base_radius`` around the own pan axis, below ``shoulder_z + base_z_margin``;
(c) fold-back: the gripper's horizontal approach direction points back at the own pan axis (cosine to the outward
    radial direction below ``fold_cos``) while the TCP is within ``fold_radius`` of the axis;
(d) torso: TCP or wrist inside the station's torso box grown by ``torso_margin`` (Haller only);
(e) mast: TCP or wrist within ``mast_radius + mast_margin`` of the head mast axis, below its top (Haller only);
(g) other_arm: any link capsule of this arm within ``other_clearance`` (surface to surface) of the PARKED other arm,
    posed at its measured joints (``other_q``), capsules from ``ws4/kinematics/collision.py`` (Haller only);
(h) joint_env: an arm joint outside the per-joint range of the commanded actions in the good training demos, widened
    by ``env_joint_margin`` degrees (depth in degrees);
(i) tcp_env: the TCP outside the demonstrated TCP workspace, a voxel grid of the training demos' TCP positions
    dilated by ``env_dilate`` (depth = distance in metres to the nearest kept voxel). (h) and (i) keep a fine-tuned
    or zero-shot FLUX inside what was demonstrated; they are built by ``--envelope`` from a recorded dataset;
(f) midline: TCP or wrist too far toward the other arm (Haller only; ``side`` +1 = robot left arm must stay at
    y >= midline_y, -1 = robot right arm at y <= -midline_y).

Geometry values are UNVERIFIED estimates until measured: calibrate the table with one touch
(``python ws4/single_arm/keepout.py --touch --config <file>`` with the gripper tips resting on the table).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WS4 = HERE.parent
if str(WS4) not in sys.path:
    sys.path.insert(0, str(WS4))

from kinematics.so101_kinematics import ArmModel, lerobot_deg_to_rad  # noqa: E402

PAN_AXIS_XY = np.array([0.0388353, 0.0])  # so101_new_calib.urdf shoulder_pan origin (kinematics _JOINT_ORIGINS[0])
WORKSPACE_FILE = HERE / "calibration" / "red_follower_workspace.json"
ZONES = ("table", "base", "fold", "torso", "mast", "midline", "other_arm", "joint_env", "tcp_env")


@dataclass
class KeepOut:
    table_z: float = 0.0  # station frame; for the red arm the station frame is its own base frame
    table_margin: float = 0.02
    wrist_table_margin: float | None = None  # None = same as table_margin
    base_radius: float = 0.12
    shoulder_z: float = 0.117  # FK height of the shoulder-lift joint above base_link
    base_z_margin: float = 0.05
    fold_cos: float = -0.5
    fold_radius: float = 0.25
    min_approach_xy: float = 0.3  # ignore the fold test when the gripper points almost straight up or down
    base_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)  # arm base_link origin in the station frame
    tcp_in_gripper: tuple[float, float, float] | None = None  # None = stock flange; Haller: fin-ray TCP
    torso_min: tuple[float, float, float] | None = None
    torso_max: tuple[float, float, float] | None = None
    torso_margin: float = 0.03
    mast_xy: tuple[float, float] | None = None
    mast_radius: float = 0.05
    mast_margin: float = 0.04
    mast_top_z: float = 0.0
    midline_y: float | None = None
    side: int = 1  # +1: this arm must stay at y >= midline_y (robot left); -1: y <= -midline_y side
    other_base_pos: tuple[float, float, float] | None = None  # parked other arm's base in the station frame
    other_q: tuple[float, ...] | None = None  # its measured joints (LeRobot degrees); set per session / attempt
    other_clearance: float = 0.04
    env_joint_lo: tuple[float, ...] | None = None  # 5 arm joints, LeRobot degrees, margin included
    env_joint_hi: tuple[float, ...] | None = None
    env_voxel: float = 0.025
    env_origin: tuple[float, float, float] | None = None  # station frame, corner of voxel (0, 0, 0)
    env_cells: tuple | None = None  # kept (dilated) voxel indices (i, j, k)
    env_source: str = ""
    calibrated: bool = False
    source: str = "defaults"

    @classmethod
    def load(cls, path: Path = WORKSPACE_FILE) -> KeepOut:
        if Path(path).exists():
            data = json.loads(Path(path).read_text())
            data = {k: tuple(v) if isinstance(v, list) else v for k, v in data.items() if k in cls.__dataclass_fields__}
            if data.get("env_cells") is not None:
                data["env_cells"] = tuple(map(tuple, data["env_cells"]))
            return cls(**data)
        return cls()

    def save(self, path: Path = WORKSPACE_FILE) -> None:
        data = asdict(self)
        cells = data.pop("env_cells")
        text = json.dumps(data, indent=2)
        # one line for the voxel list, so the file stays readable
        cells_json = json.dumps(None if cells is None else [list(c) for c in cells])
        text = text[:-2] + ',\n  "env_cells": ' + cells_json + "\n}"
        Path(path).write_text(text + "\n")

    @classmethod
    def for_haller(cls, side: str) -> KeepOut:
        """Haller (XLeRobot tower) arm on the robot's ``side`` ("left" / "right"), from station_geometry.yaml."""
        from kinematics.so101_kinematics import Station

        st = Station.load(finray=True)
        T = st.T_station_base[side]
        if not np.allclose(T[:3, :3], np.eye(3), atol=1e-6):
            raise ValueError("keep-out assumes unrotated arm bases (station_geometry.yaml base_quat identity)")
        torso_c, torso_h = np.array(st.torso["center"]), np.array(st.torso["half_extents"])
        return cls(
            table_z=float(st.table_z),
            base_pos=tuple(float(v) for v in T[:3, 3]),
            tcp_in_gripper=st.arm.tcp_in_gripper,
            torso_min=tuple((torso_c - torso_h).tolist()),
            torso_max=tuple((torso_c + torso_h).tolist()),
            mast_xy=tuple(float(v) for v in st.mast["p0"][:2]),
            mast_radius=float(st.mast.get("radius", 0.05)),
            mast_top_z=float(st.mast["p1"][2]),
            # Fitted to the 10 layout-A VR demos of "put the red pen in its slot in the cardboard box" (2026-10-17):
            # TCP down to z = 0.002 (pen grasped from the mat) while the wrist stays >= 0.124; TCP to y = -0.074
            # measured / -0.084 commanded. The midline is only a backstop: the parked-arm capsules are the real guard.
            table_margin=-0.005,
            wrist_table_margin=0.05,
            midline_y=-0.11,
            # Takes 17 and 20 grasp the pen close to the robot: TCP 7 cm in front of the pan axis, 6 cm above the mat,
            # gripper pointing back at the base. The red arm's 12 cm base cylinder and fold-back zone would refuse
            # that; here the torso box (+2 cm) covers the robot body and the base cylinder only the turntable.
            base_radius=0.06,
            fold_cos=-1.01,  # disabled: a back-pointing grasp near the base is a demonstrated motion on the Haller
            torso_margin=0.02,
            other_base_pos=tuple(float(v) for v in st.T_station_base["right" if side == "left" else "left"][:3, 3]),
            side=1 if side == "left" else -1,
            source=f"station_geometry ({st.source}), side {side}",
        )

    # ------------------------------------------------------------------ geometry
    def _points(self, q_deg: np.ndarray):
        q = lerobot_deg_to_rad(np.asarray(q_deg, dtype=float)[..., :5])
        arm = ArmModel(tcp_in_gripper=self.tcp_in_gripper)
        frames = arm.joint_frames(q)
        T = arm.fk(q, "tcp")
        return frames[..., 3, :3, 3], T[..., :3, 3], T[..., :3, 2]  # wrist, tcp, approach (base frame)

    def depths(self, q_deg: np.ndarray) -> np.ndarray:
        """Violation depth per zone (metres; cosine units for fold), shape (..., len(ZONES)); 0 = clear."""
        wrist, tcp, approach = self._points(q_deg)
        off = np.asarray(self.base_pos, dtype=float)
        pts = np.stack([tcp + off, wrist + off], axis=-2)  # station frame, (..., 2, 3)
        out = np.zeros(tcp.shape[:-1] + (len(ZONES),))
        wrist_margin = self.table_margin if self.wrist_table_margin is None else self.wrist_table_margin
        out[..., 0] = np.maximum(np.maximum(0.0, self.table_z + self.table_margin - pts[..., 0, 2]),
                                 self.table_z + wrist_margin - pts[..., 1, 2])  # fmt: skip
        r = np.linalg.norm(tcp[..., :2] - PAN_AXIS_XY, axis=-1)
        below = tcp[..., 2] < self.shoulder_z + self.base_z_margin
        out[..., 1] = np.where(below, np.maximum(0.0, self.base_radius - r), 0.0)
        radial = (tcp[..., :2] - PAN_AXIS_XY) / np.maximum(r, 1e-6)[..., None]
        axy = approach[..., :2]
        n = np.linalg.norm(axy, axis=-1)
        cos = np.sum(axy / np.maximum(n, 1e-6)[..., None] * radial, axis=-1)
        near = (n > self.min_approach_xy) & (r < self.fold_radius)
        out[..., 2] = np.where(near, np.maximum(0.0, self.fold_cos - cos), 0.0)
        if self.torso_min is not None:
            lo = np.asarray(self.torso_min) - self.torso_margin
            hi = np.asarray(self.torso_max) + self.torso_margin
            inside = np.all((pts > lo) & (pts < hi), axis=-1)
            exit_depth = np.minimum(pts - lo, hi - pts).min(axis=-1)
            out[..., 3] = np.where(inside, exit_depth, 0.0).max(axis=-1)
        if self.mast_xy is not None:
            rm = np.linalg.norm(pts[..., :2] - np.asarray(self.mast_xy), axis=-1)
            reach = self.mast_radius + self.mast_margin
            out[..., 4] = np.where(pts[..., 2] < self.mast_top_z + self.mast_margin, np.maximum(0.0, reach - rm),
                                   0.0).max(axis=-1)  # fmt: skip
        if self.midline_y is not None:
            out[..., 5] = np.maximum(0.0, self.side * (self.side * self.midline_y - pts[..., 1])).max(axis=-1)
        if self.other_q is not None and self.other_base_pos is not None:
            out[..., 6] = self.other_arm_depth(q_deg)
        if self.env_joint_lo is not None:
            q5 = np.asarray(q_deg, dtype=float)[..., :5]
            lo, hi = np.asarray(self.env_joint_lo), np.asarray(self.env_joint_hi)
            out[..., 7] = np.maximum(np.maximum(lo - q5, q5 - hi), 0.0).max(axis=-1)
        if self.env_cells is not None:
            out[..., 8] = self._tcp_env_depth(pts[..., 0, :])
        return out

    def _env_grid(self):
        cache = getattr(self, "_grid_cache", None)
        if cache is None:
            cells = np.asarray(self.env_cells, dtype=int)
            shape = cells.max(axis=0) + 1
            grid = np.zeros(shape, dtype=bool)
            grid[tuple(cells.T)] = True
            centers = np.asarray(self.env_origin) + (cells + 0.5) * self.env_voxel
            cache = (grid, centers)
            object.__setattr__(self, "_grid_cache", cache)
        return cache

    def _tcp_env_depth(self, tcp: np.ndarray) -> np.ndarray:
        grid, centers = self._env_grid()
        flat = tcp.reshape(-1, 3)
        idx = np.floor((flat - np.asarray(self.env_origin)) / self.env_voxel).astype(int)
        inside = np.all((idx >= 0) & (idx < grid.shape), axis=1)
        inside[inside] = grid[tuple(idx[inside].T)]
        depth = np.zeros(len(flat))
        for start in range(0, len(flat), 2048):
            sl = slice(start, start + 2048)
            out = ~inside[sl]
            if out.any():
                d = np.linalg.norm(flat[sl][out][:, None] - centers[None], axis=-1).min(axis=1)
                depth[np.flatnonzero(out) + start] = np.maximum(d - self.env_voxel / 2, 1e-3)
        return depth.reshape(tcp.shape[:-1])

    def build_envelope(self, actions: np.ndarray, joint_margin: float = 10.0, voxel: float = 0.025,
                       dilate: float = 0.05, source: str = "") -> None:  # fmt: skip
        """Joint and TCP envelope of demonstrated commanded actions (N, 6), LeRobot degrees."""
        a = np.asarray(actions, dtype=float)
        self.env_joint_lo = tuple((a[:, :5].min(axis=0) - joint_margin).tolist())
        self.env_joint_hi = tuple((a[:, :5].max(axis=0) + joint_margin).tolist())
        tcp = self.tcp_station(a)
        r = int(np.ceil(dilate / voxel))
        origin = tcp.min(axis=0) - (r + 1) * voxel
        idx = np.floor((tcp - origin) / voxel).astype(int)
        grid = np.zeros(idx.max(axis=0) + r + 2, dtype=bool)
        grid[tuple(idx.T)] = True
        offsets = [(i, j, k) for i in range(-r, r + 1) for j in range(-r, r + 1) for k in range(-r, r + 1)
                   if (i * i + j * j + k * k) * voxel * voxel <= dilate * dilate + 1e-12]  # fmt: skip
        dil = np.zeros_like(grid)
        for i, j, k in offsets:
            dil |= np.roll(np.roll(np.roll(grid, i, 0), j, 1), k, 2)
        self.env_voxel, self.env_origin = voxel, tuple(origin.tolist())
        self.env_cells = tuple(map(tuple, np.argwhere(dil).tolist()))
        self.env_source = source
        object.__setattr__(self, "_grid_cache", None)

    def _capsule_points(self, q_deg: np.ndarray, base_pos) -> np.ndarray:
        """Capsule end points (..., 7, 3) in the station frame: base, j0..j4, tcp (collision.LINKS order)."""
        q = lerobot_deg_to_rad(np.asarray(q_deg, dtype=float)[..., :5])
        arm = ArmModel(tcp_in_gripper=self.tcp_in_gripper)
        frames = arm.joint_frames(q)
        tcp = arm.fk(q, "tcp")[..., :3, 3]
        off = np.asarray(base_pos, dtype=float)
        base = np.broadcast_to(np.zeros(3), tcp.shape)
        return np.stack([base, *(frames[..., i, :3, 3] for i in range(5)), tcp], axis=-2) + off

    def other_arm_depth(self, q_deg: np.ndarray) -> np.ndarray:
        from kinematics.collision import LINKS, segment_distance

        names = ["base", "j0", "j1", "j2", "j3", "j4", "tcp"]
        idx = [(names.index(a), names.index(b), r) for a, b, r in LINKS]
        mine = self._capsule_points(q_deg, self.base_pos)
        other = self._capsule_points(np.asarray(self.other_q, dtype=float), self.other_base_pos)
        gap = np.full(mine.shape[:-2], np.inf)
        for a, b, r in idx[1:]:  # skip this arm's base capsule: it sits fixed on the torso
            for c, d, r2 in idx:
                dist = segment_distance(mine[..., a, :], mine[..., b, :], other[c], other[d]) - r - r2
                gap = np.minimum(gap, dist)
        return np.maximum(0.0, self.other_clearance - gap)

    def refuse(self, target_deg: np.ndarray, measured_deg: np.ndarray, eps: float = 1e-4) -> tuple[bool, str]:
        """True (and the zone) when ``target`` is in a zone and deeper than the measured pose."""
        dt, dm = self.depths(target_deg), self.depths(measured_deg)
        worse = (dt > 0) & (dt > dm + eps)
        if worse.any():
            return True, ZONES[int(np.argmax(worse))]
        return False, ""

    def without_envelope(self) -> KeepOut:
        """The station zones only, for scripted ramps (to the start pose, back to rest): the envelope bounds what the
        POLICY may command; an unfolding ramp from a folded rest briefly leaves the demonstrated TCP workspace."""
        from dataclasses import replace

        ko = replace(self, env_joint_lo=None, env_joint_hi=None, env_cells=None)
        ko.other_q = self.other_q
        return ko

    def tcp_station(self, q_deg: np.ndarray) -> np.ndarray:
        return self._points(q_deg)[1] + np.asarray(self.base_pos, dtype=float)


def read_arm_joints(port: str, robot_id: str) -> np.ndarray:
    """Measured joints of an arm in LeRobot degrees (gripper 0..100), read-only (no register writes, torque untouched).
    Only while no other process (the haller HMI) holds the port."""
    sys.path.insert(0, str(HERE))
    from show_joints import JOINTS, open_bus

    bus = open_bus(port, robot_id)
    try:
        q = bus.sync_read("Present_Position", list(JOINTS), num_retry=3)
    finally:
        bus.port_handler.closePort()
    return np.array([q[j] for j in JOINTS])


def touch(args: argparse.Namespace) -> None:
    """Read the joints (torque off) with the gripper tips resting on the table and store that height as the table."""
    from show_joints import open_bus

    path = Path(args.config)
    ko = KeepOut.load(path)
    bus = open_bus(args.port, args.robot_id)
    try:
        joints = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
        q = bus.sync_read("Present_Position", joints, num_retry=3)
    finally:
        bus.port_handler.closePort()
    qd = np.array([q[j] for j in joints])
    tcp = ko.tcp_station(qd)
    print(f"joints {np.round(qd, 1)} -> TCP at {np.round(tcp, 3)} m (station); previous table_z {ko.table_z}")
    ko.table_z, ko.calibrated = float(tcp[2]), True
    ko.save(path)
    print(f"saved table_z = {ko.table_z:.3f} m to {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", default=str(WORKSPACE_FILE), help="workspace JSON of this arm")
    p.add_argument("--haller", choices=("left", "right"), help="write --config from station_geometry.yaml")
    p.add_argument("--touch", action="store_true", help="calibrate the table height from the gripper tips on it")
    p.add_argument("--envelope", default=None, metavar="DATASET_ROOT",
                   help="add the demonstrated-workspace envelope from a recorded 12-D or per-arm dataset")
    p.add_argument("--envelope-arm", default="right", help="arm columns of a 12-D dataset (haller right = robot left)")
    p.add_argument("--exclude", type=int, nargs="*", default=[], help="held-out and dropped episodes")
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--robot-id", default="red_follower")
    args = p.parse_args()
    sys.path.insert(0, str(HERE))
    if args.haller:
        ko = KeepOut.for_haller(args.haller)
        ko.save(Path(args.config))
        print(f"wrote {args.config}: {json.dumps(asdict(ko))}")
    elif args.envelope:
        from flux.package_tools import arm_columns, dataset_arrays

        ko = KeepOut.load(Path(args.config))
        arrays = dataset_arrays("local/envelope", root=args.envelope)
        keep = ~np.isin(arrays["episode_index"], args.exclude)
        actions = arrays["action"][keep]
        if actions.shape[1] == 12:
            actions = arm_columns(actions, args.envelope_arm)
        eps = sorted(set(arrays["episode_index"][keep].tolist()))
        ko.build_envelope(actions, source=f"{args.envelope} episodes {eps} (excluded {args.exclude})")
        ko.save(Path(args.config))
        print(f"envelope from {keep.sum()} frames of {len(eps)} episodes: joints {np.round(ko.env_joint_lo, 1)} .. "
              f"{np.round(ko.env_joint_hi, 1)}, {len(ko.env_cells)} TCP voxels of {ko.env_voxel} m")  # fmt: skip
    elif args.touch:
        touch(args)
    else:
        print(json.dumps(asdict(KeepOut.load(Path(args.config))), indent=2))


if __name__ == "__main__":
    main()
