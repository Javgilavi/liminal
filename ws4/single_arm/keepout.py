"""Geometric keep-out for one SO-101 follower: refuse commanded targets that drive the gripper into the table or
into the arm's own base, checked by forward kinematics on every tick.

Why: in A0 the wrist folded back toward the robot's base and the elbow pushed against it until the servo was damaged
(2026-10-17, attempts A0 #2 and #3). Joint limits and a step limit cannot see that; FK can.

Kinematics: ``ws4/kinematics/so101_kinematics.py`` (stock SO-101 URDF chain, LeRobot degrees with zero at mid-range).
The red arm has the stock gripper, so the checked points are the stock ``flange`` (between the jaws) and the wrist
flex joint. Frame: URDF ``base_link`` of this arm; the pan axis is vertical through (0.0388, 0).

A target is refused when it violates a zone AND goes deeper into it than the measured pose already is, so the arm can
always move out of a zone (for example from a rest pose near the base) but never further in:

(a) table: the TCP or the wrist below ``table_z + table_margin``;
(b) base: the TCP inside a cylinder of ``base_radius`` around the pan axis, below ``shoulder_z + base_z_margin``;
(c) fold-back: the gripper's horizontal approach direction points back at the pan axis (cosine to the outward radial
    direction below ``fold_cos``) while the TCP is within ``fold_radius`` of the axis.

Geometry defaults are UNVERIFIED estimates: ``table_z`` 0 assumes the base plate sits on the table (the lowest TCP in
Oscar's hand-guided demos was z = 0.002 m), the base cylinder covers the base and shoulder servos. Calibrate the
table with one measured point: ``python ws4/single_arm/keepout.py --touch`` with the jaw tips resting on the table.
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
ZONES = ("table", "base", "fold")


@dataclass
class KeepOut:
    table_z: float = 0.0
    table_margin: float = 0.02
    base_radius: float = 0.12
    shoulder_z: float = 0.117  # FK height of the shoulder-lift joint above base_link
    base_z_margin: float = 0.05
    fold_cos: float = -0.5
    fold_radius: float = 0.25
    min_approach_xy: float = 0.3  # ignore the fold test when the gripper points almost straight up or down
    calibrated: bool = False

    @classmethod
    def load(cls, path: Path = WORKSPACE_FILE) -> KeepOut:
        if path.exists():
            data = json.loads(path.read_text())
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        return cls()

    def save(self, path: Path = WORKSPACE_FILE) -> None:
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")

    def depths(self, q_deg: np.ndarray) -> np.ndarray:
        """Violation depth per zone (metres for table / base, cosine units for fold), shape (..., 3); 0 = clear."""
        q = lerobot_deg_to_rad(np.asarray(q_deg, dtype=float)[..., :5])
        arm = ArmModel()
        frames = arm.joint_frames(q)
        T = arm.fk(q, "flange")
        wrist, tcp, approach = frames[..., 3, :3, 3], T[..., :3, 3], T[..., :3, 2]
        floor = self.table_z + self.table_margin
        table = np.maximum(0.0, floor - np.minimum(tcp[..., 2], wrist[..., 2]))
        r = np.linalg.norm(tcp[..., :2] - PAN_AXIS_XY, axis=-1)
        below = tcp[..., 2] < self.shoulder_z + self.base_z_margin
        base = np.where(below, np.maximum(0.0, self.base_radius - r), 0.0)
        radial = (tcp[..., :2] - PAN_AXIS_XY) / np.maximum(r, 1e-6)[..., None]
        axy = approach[..., :2]
        n = np.linalg.norm(axy, axis=-1)
        cos = np.sum(axy / np.maximum(n, 1e-6)[..., None] * radial, axis=-1)
        fold = np.where((n > self.min_approach_xy) & (r < self.fold_radius), np.maximum(0.0, self.fold_cos - cos), 0.0)
        return np.stack([table, base, fold], axis=-1)

    def refuse(self, target_deg: np.ndarray, measured_deg: np.ndarray, eps: float = 1e-4) -> tuple[bool, str]:
        """True (and the zone) when ``target`` is in a zone and deeper than the measured pose."""
        dt, dm = self.depths(target_deg), self.depths(measured_deg)
        worse = (dt > 0) & (dt > dm + eps)
        if worse.any():
            return True, ZONES[int(np.argmax(worse))]
        return False, ""


def touch(args: argparse.Namespace) -> None:
    """Read the joints (torque off) with the jaw tips resting on the table and store that height as the table."""
    from show_joints import open_bus

    bus = open_bus(args.port, args.robot_id)
    try:
        q = bus.sync_read("Present_Position", ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
                                                "wrist_roll"], num_retry=3)  # fmt: skip
    finally:
        bus.port_handler.closePort()
    qd = np.array(list(q.values()))
    tcp = ArmModel().fk(lerobot_deg_to_rad(qd), "flange")[:3, 3]
    ko = KeepOut.load()
    print(f"joints {np.round(qd, 1)} -> flange at {np.round(tcp, 3)} m; previous table_z {ko.table_z}")
    ko.table_z, ko.calibrated = float(tcp[2]), True
    ko.save()
    print(f"saved table_z = {ko.table_z:.3f} m to {WORKSPACE_FILE}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--touch", action="store_true", help="calibrate the table height from the jaw tips on the table")
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--robot-id", default="red_follower")
    args = p.parse_args()
    if args.touch:
        sys.path.insert(0, str(HERE))
        touch(args)
    else:
        print(json.dumps(asdict(KeepOut.load()), indent=2))


if __name__ == "__main__":
    main()
