"""Inverse kinematics for one SO-101 arm of the twin: a jaw-centred tool point and gripper orientation.

The `gripperframe` site sits at the fixed jaw's fingertip with its x axis along the fingers and z axis
toward the moving jaw. An object of thickness w held 12 mm behind the tips is centred at
(-0.012, 0, w/2) in that frame, which is the tool point used here.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


def frame(approach, jaw_axis):
    """Rotation whose x axis is `approach` and z axis is `jaw_axis` made orthogonal to it."""
    x = np.asarray(approach, float) / np.linalg.norm(approach)
    z = np.asarray(jaw_axis, float) - np.dot(jaw_axis, x) * x
    z /= np.linalg.norm(z)
    return np.column_stack((x, np.cross(z, x), z))


class ArmIK:
    def __init__(self, model, arm: str):
        self.m, self.arm = model, arm
        self.ids = [model.joint(f"{arm}_{j}").id for j in ARM_JOINTS]
        self.qadr = [model.jnt_qposadr[i] for i in self.ids]
        self.lo, self.hi = model.jnt_range[self.ids, 0], model.jnt_range[self.ids, 1]
        self.site = model.site(f"{arm}_gripperframe").id
        self.scratch = mujoco.MjData(model)

    def tool(self, q, qpos_rest, offset):
        d = self.scratch
        d.qpos[:] = qpos_rest
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, d)
        R = d.site_xmat[self.site].reshape(3, 3)
        return d.site_xpos[self.site] + R @ offset, R

    def solve(self, qpos, target, R_target, offset=(-0.012, 0.0, 0.0), q0=None, iters=80, rot_weight=0.25):
        """Joint angles (rad, 5 arm joints) placing the tool point at `target` with orientation `R_target`.

        Returns (q, position_error_m, orientation_error_rad)."""
        offset = np.asarray(offset, float)
        q = np.array(q0 if q0 is not None else qpos[self.qadr], float)
        for _ in range(iters):
            p, R = self.tool(q, qpos, offset)
            e_rot = 0.5 * sum(np.cross(R[:, k], R_target[:, k]) for k in range(3))
            e = np.concatenate([target - p, rot_weight * e_rot])
            if np.linalg.norm(e[:3]) < 5e-4 and np.linalg.norm(e_rot) < 1e-2:
                break
            J = np.zeros((6, 5))
            for k in range(5):
                dq = np.zeros(5)
                dq[k] = 1e-5
                p2, R2 = self.tool(q + dq, qpos, offset)
                e2 = np.concatenate([target - p2, rot_weight * 0.5 * sum(np.cross(R2[:, i], R_target[:, i]) for i in range(3))])
                J[:, k] = (e - e2) / 1e-5
            step = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
            q = np.clip(q + np.clip(step, -0.2, 0.2), self.lo, self.hi)
        p, R = self.tool(q, qpos, offset)
        e_rot = 0.5 * sum(np.cross(R[:, k], R_target[:, k]) for k in range(3))
        return q, float(np.linalg.norm(target - p)), float(np.linalg.norm(e_rot))

    def seeds(self, qpos, target):
        """Starting poses for the solver: the current pose plus bent-elbow poses aimed at the target."""
        d = self.scratch
        d.qpos[:] = qpos
        mujoco.mj_kinematics(self.m, d)
        b = d.xpos[self.m.body(f"{self.arm}_shoulder").id]
        # Pan zero points the arm along world -y; try both turning directions toward the target.
        bearing = math.atan2(target[0] - b[0], -(target[1] - b[1]))
        out = [np.array(qpos[self.qadr], float)]
        for pan in (bearing, -bearing):
            for lift, elbow, flex in ((-20, 40, 60), (20, 20, 50), (40, -10, 60), (-50, 80, 40), (10, 60, -10)):
                for roll in (0, 90, -90):
                    out.append(np.clip(np.radians([math.degrees(pan), lift, elbow, flex, roll]), self.lo, self.hi))
        return out

    def solve_robust(self, qpos, target, R_target, offset=(-0.012, 0.0, 0.0), q0=None, tol=0.002):
        """`solve` from several starting poses; returns the best result."""
        best = None
        starts = ([np.asarray(q0, float)] if q0 is not None else []) + self.seeds(qpos, target)
        for start in starts:
            res = self.solve(qpos, np.asarray(target, float), R_target, offset=offset, q0=start)
            if best is None or res[1] + 0.05 * res[2] < best[1] + 0.05 * best[2]:
                best = res
            if best[1] < tol and best[2] < 0.02:
                break
        return best

    def solve_topdown(self, qpos, target, yaw, width, q0=None):
        """Gripper pointing straight down, jaws closing across direction `yaw` (rad, world z).

        Tries both jaw orientations and keeps the one with the smaller error."""
        best = None
        for flip in (0.0, math.pi):
            jaw = (math.cos(yaw + flip), math.sin(yaw + flip), 0.0)
            res = self.solve_robust(qpos, target, frame((0, 0, -1), jaw), offset=(-0.012, 0.0, width / 2), q0=q0)
            if best is None or res[1] + 0.05 * res[2] < best[1] + 0.05 * best[2]:
                best = res
        return best
