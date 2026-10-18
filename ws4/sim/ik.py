"""Small damped-least-squares IK on the station MJCF, for scripted tests and demos.

This is a sim test utility, not the team's kinematics: ws4/kinematics owns the real IK / EE50 work.
"""

from __future__ import annotations

import numpy as np

from .geometry import ARM_JOINTS

BODY = ARM_JOINTS[:5]


def top_down_rotation(yaw: float) -> np.ndarray:
    """TCP orientation for a vertical grasp: site +z (back toward the wrist) = world +z, fingers open along yaw."""
    c, s = np.cos(yaw), np.sin(yaw)
    x = np.array([c, s, 0.0])
    z = np.array([0.0, 0.0, 1.0])
    return np.column_stack([x, np.cross(z, x), z])


class ArmIK:
    def __init__(self, model, side: str, site: str | None = None) -> None:
        import mujoco

        self.mj, self.m = mujoco, model
        self.d = mujoco.MjData(model)
        self.site = model.site(site or f"{side}_ee").id
        self.jids = [model.joint(f"{side}_{j}").id for j in BODY]
        self.qadr = np.array([model.jnt_qposadr[j] for j in self.jids])
        self.dadr = np.array([model.jnt_dofadr[j] for j in self.jids])
        self.lo, self.hi = model.jnt_range[self.jids].T

    def fk(self, qpos_full: np.ndarray, q5: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.d.qpos[:] = qpos_full
        self.d.qpos[self.qadr] = q5
        self.mj.mj_kinematics(self.m, self.d)
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3).copy()

    def solve(self, qpos_full: np.ndarray, pos: np.ndarray, rot: np.ndarray | None = None, q_init=None,
              iters: int = 300, rot_weight: float = 0.3, damping: float = 1e-3, tol: float = 1e-4):
        """Returns (q5 radians, position error m, orientation error rad)."""
        q = np.array(q_init if q_init is not None else qpos_full[self.qadr], dtype=float)
        jacp = np.zeros((3, self.m.nv))
        jacr = np.zeros((3, self.m.nv))
        for _ in range(iters):
            p, R = self.fk(qpos_full, q)
            e = pos - p
            er = np.zeros(3)
            if rot is not None:
                er = 0.5 * sum(np.cross(R[:, i], rot[:, i]) for i in range(3))
            if np.linalg.norm(e) < tol and np.linalg.norm(er) < 10 * tol:
                break
            self.mj.mj_comPos(self.m, self.d)
            self.mj.mj_jacSite(self.m, self.d, jacp, jacr, self.site)
            J = jacp[:, self.dadr]
            err = e
            if rot is not None:
                J = np.vstack([J, rot_weight * jacr[:, self.dadr]])
                err = np.concatenate([e, rot_weight * er])
            dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(J.shape[0]), err)
            q = np.clip(q + np.clip(dq, -0.2, 0.2), self.lo, self.hi)
        p, R = self.fk(qpos_full, q)
        er = 0.0 if rot is None else float(np.linalg.norm(0.5 * sum(np.cross(R[:, i], rot[:, i]) for i in range(3))))
        return q, float(np.linalg.norm(pos - p)), er
