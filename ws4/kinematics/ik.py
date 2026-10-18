"""Constrained IK for one 5-DoF SO-101 arm: exact position plus the reachable part of the orientation.

Structure of the arm (from the URDF, checked numerically in ``tests/test_ik.py``): a vertical pan axis,
three parallel pitch axes (shoulder_lift, elbow_flex, wrist_flex) perpendicular to the arm plane, and a
roll axis along the approach direction. By the product of exponentials the TCP orientation is exactly

    R(q) = Rz(-q1) Ry(q2 + q3 + q4) Rx(-q5) R0,        R0 = R(0)

so the reachable orientations at a given pan angle form the 2-parameter family Rz(a) Ry(b) Rx(g) R0:
the approach axis must lie in the vertical plane of the arm (whose angle is fixed by the target position),
the wrist roll is free. A full 6-DoF target loses exactly one rotational DoF: the yaw of the approach
direction out of the arm plane. ``project_orientation`` removes it; everything else is executed.

Two solvers:

* :class:`AnalyticIK`: closed form (pan from the position with the lateral TCP offset, orientation
  projection, planar 2-link for lift/elbow, flex and roll from the projected orientation), one Newton polish.
* :class:`DLSIK`: task-priority damped least squares (position first, orientation in the null space),
  warm-started from the previous solution. The numeric reference the analytic solver is compared against.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .se3 import make_pose, matrix_to_rotvec, rot_x, rot_y, rot_z, wrap_angle
from .so101_kinematics import ArmModel


@dataclass
class IKResult:
    q: np.ndarray
    T: np.ndarray  # achieved TCP pose in base
    pos_err: float  # metres, achieved vs requested position
    rot_err: float  # rad, geodesic, achieved vs requested orientation (includes the unreachable part)
    approach_err: float  # rad, angle between achieved and requested approach (tool z) axes
    proj_rot_err: float  # rad, requested vs its projection onto the reachable family (irreducible part)
    reachable: bool  # position inside the workspace (before joint limits)
    clamped: bool  # a joint limit was active
    degenerate: bool  # approach nearly normal to the arm plane: projection ill-conditioned
    iters: int = 0

    @property
    def ok(self) -> bool:
        return self.pos_err < 5e-3 and self.reachable


def _log(R: np.ndarray) -> np.ndarray:
    """Rotation vector of R (scalar fast path; falls back to the robust batched version near pi)."""
    cos = max(-1.0, min(1.0, (R[0, 0] + R[1, 1] + R[2, 2] - 1.0) / 2.0))
    angle = math.acos(cos)
    if angle > math.pi - 1e-4:
        return matrix_to_rotvec(R)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w * (0.5 if angle < 1e-6 else angle / (2.0 * math.sin(angle)))


def _rot_angle(R: np.ndarray) -> float:
    return math.acos(max(-1.0, min(1.0, (R[0, 0] + R[1, 1] + R[2, 2] - 1.0) / 2.0)))


def _errors(arm: ArmModel, q, T_target) -> tuple[np.ndarray, float, float, float]:
    T = arm.fk_single(q)
    pos_err = float(np.linalg.norm(T[:3, 3] - T_target[:3, 3]))
    rot_err = _rot_angle(T[:3, :3].T @ T_target[:3, :3])
    app_err = math.acos(max(-1.0, min(1.0, float(T[:3, 2] @ T_target[:3, 2]))))
    return T, pos_err, rot_err, app_err


class _Geometry:
    """Constants of the planar decomposition, read off the FK model at q = 0 (so they track the URDF)."""

    def __init__(self, arm: ArmModel):
        frames = arm.joint_frames(np.zeros(5))
        T0 = arm.fk(np.zeros(5))
        self.c = frames[0, :3, 3]  # point on the pan axis
        self.S = frames[1, :3, 3]  # shoulder_lift
        self.E = frames[2, :3, 3]  # elbow_flex
        self.F = frames[3, :3, 3]  # wrist_flex
        self.P = frames[4, :3, 3]  # point on the roll axis
        self.R0 = T0[:3, :3]
        self.tcp0 = T0[:3, 3]
        self.u = self.tcp0 - self.P  # TCP offset from the roll-axis point (rotates with roll)
        self.v = self.P - self.F  # roll-axis point from the flex axis (rotates with pitch)
        L1, L2 = self.E - self.S, self.F - self.E
        self.l1, self.l2 = np.hypot(L1[0], L1[2]), np.hypot(L2[0], L2[2])
        self.a1, self.a2 = np.arctan2(L1[2], L1[0]), np.arctan2(L2[2], L2[0])
        axes = frames[:, :3, 2]
        expected = np.array([[0, 0, -1], [0, 1, 0], [0, 1, 0], [0, 1, 0], [-1, 0, 0]], dtype=float)
        self.axis_dev = float(np.max(np.abs(axes - expected)))  # ~1e-5 from the URDF's 3.14159 literals

    def lateral_offset(self, g: float) -> float:
        """y of the TCP in the pan-0 frame relative to the pan axis, for roll parameter g (= -q5)."""
        return float(self.P[1] + self.u[1] * np.cos(g) - self.u[2] * np.sin(g) - self.c[1])


def project_family(N: np.ndarray, b_prev: float | None = None, iters: int = 3, eps: float = 0.05):
    """Closest Ry(b) Rx(g) to ``N`` (Frobenius), started from the approach (first column) projection.

    Returns (b, g, degenerate). ``degenerate`` when the approach is within ~3 deg of the arm-plane normal;
    then b is taken from ``b_prev`` (continuity) because the projection is ill-conditioned there.
    """
    n = N[:, 0]
    h = np.hypot(n[0], n[2])
    degenerate = h < eps
    b = float(np.arctan2(-n[2], n[0])) if not (degenerate and b_prev is not None) else float(b_prev)
    g = 0.0
    for _ in range(max(iters, 1)):
        K = _ry(b).T @ N
        g = float(np.arctan2(K[2, 1] - K[1, 2], K[1, 1] + K[2, 2]))
        if degenerate and b_prev is not None:
            break
        L = N @ _rx(g).T
        b = float(np.arctan2(L[0, 2] - L[2, 0], L[0, 0] + L[2, 2]))
    return b, g, bool(degenerate)


def _dpinv(A: np.ndarray, lam2: float) -> np.ndarray:
    return A.T @ np.linalg.inv(A @ A.T + lam2 * np.eye(A.shape[0]))


def priority_step(arm: ArmModel, q: np.ndarray, p_t: np.ndarray, R_t: np.ndarray, lam2: float = 1e-6,
                  max_step: float = 0.3, lam2_rot: float = 1e-4) -> tuple[np.ndarray, float]:
    """One task-priority DLS step: position first, orientation in the position null space.

    ``R_t`` must already be reachable (projected onto the arm's orientation family at the current pan):
    an unreachable orientation component lands on a near-zero singular value of the secondary task and
    produces huge steps that wreck the position. Callers project first, see ``AnalyticIK.project_at_pan``.

    Joint limits use a small active set: a joint the step would push past its limit is moved onto the limit
    and removed from both tasks, and the rest is re-solved. Plain clipping instead lets the orientation task
    drive a joint into its stop and silently lose the position.
    """
    J, T = arm.jacobian(q)
    ep = p_t - T[:3, 3]
    eo = _log(R_t @ T[:3, :3].T)
    lo, hi = arm.limits[:, 0], arm.limits[:, 1]
    free = np.ones(5, dtype=bool)
    fixed = np.zeros(5)
    dq = np.zeros(5)
    for _ in range(5):
        Jf = J * free
        Jp, Jo = Jf[:3], Jf[3:]
        r_p = ep - J[:3] @ fixed
        r_o = eo - J[3:] @ fixed
        Jp_pinv = _dpinv(Jp, lam2)
        dq1 = Jp_pinv @ r_p
        Nul = np.eye(5) - Jp_pinv @ Jp
        dq = dq1 + Nul @ (_dpinv(Jo @ Nul, lam2_rot) @ (r_o - Jo @ dq1))
        dq = dq * free + fixed
        n = float(np.linalg.norm(dq))
        if n > max_step:
            dq *= max_step / n
        over = free & ((q + dq < lo) | (q + dq > hi))
        if not over.any():
            break
        free &= ~over
        fixed[over] = np.clip(q + dq, lo, hi)[over] - q[over]
    return arm.clip(q + dq), float(np.linalg.norm(dq))


def _rx(t: float) -> np.ndarray:
    c, s = np.cos(t), np.sin(t)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _ry(t: float) -> np.ndarray:
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rz(t: float) -> np.ndarray:
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class AnalyticIK:
    """Closed-form IK. Enumerates both pan branches (reaching forward, or back over the shoulder) and the
    elbow branches (only the data's branch by default, ``elbow_up``), then keeps the candidate with the lowest
    (limit violation, position error, orientation error, distance to the previous solution) cost."""

    def __init__(self, arm: ArmModel | None = None, polish: int = 4, frobenius_iters: int = 3,
                 fixed_point_iters: int = 6, max_jump: float = 0.3, elbow_up: bool = True):
        self.arm = arm or ArmModel()
        self.geo = _Geometry(self.arm)
        self.polish = polish
        self.frobenius_iters = frobenius_iters
        self.fixed_point_iters = fixed_point_iters
        self._scan = np.deg2rad(np.arange(-180.0, 180.0, 1.0))  # coarse wrist-pitch grid; bisection refines it
        self.max_jump = max_jump  # rad per call; larger joint changes are penalised like 10 cm of position error
        # Only the elbow branch the recorded SO-101/XLeRobot data uses (e < 0 in >99% of frames, see the
        # report). Allowing both lets noisy targets drift onto the other branch and later force a flip.
        self.elbow_up = elbow_up

    def _branch(self, R, hx, hy, r, branch, q_prev):
        geo = self.geo
        g = -float(q_prev[4]) if q_prev is not None else 0.0
        b_prev = float(q_prev[1] + q_prev[2] + q_prev[3]) if q_prev is not None else None
        reachable = True
        a = b = 0.0
        degenerate = False
        phi = np.arctan2(hy, hx)
        a_last = None
        for _ in range(self.fixed_point_iters):
            d = geo.lateral_offset(g)
            if r <= abs(d) + 1e-9:  # on the pan axis: any pan works, keep the previous one
                reachable = False
                a = -float(q_prev[0]) if q_prev is not None else 0.0
            else:
                s = np.arcsin(d / r)
                a = float(phi - s) if branch == 0 else float(wrap_angle(phi - np.pi + s))
            if a_last is not None and abs(a - a_last) < 1e-10:
                break  # the roll-dependent lateral offset has converged
            a_last = a
            N = _rz(a).T @ R @ geo.R0.T
            b, g, degenerate = project_family(N, b_prev, self.frobenius_iters)
        return a, b, g, degenerate, reachable

    def project(self, T_target: np.ndarray, q_prev: np.ndarray | None = None, branch: int = 0):
        """Reachable orientation closest to the target at the target position, for one pan branch."""
        geo = self.geo
        p, R = T_target[:3, 3], T_target[:3, :3]
        hx, hy = p[0] - geo.c[0], p[1] - geo.c[1]
        a, b, g, degenerate, reachable = self._branch(R, hx, hy, float(np.hypot(hx, hy)), branch, q_prev)
        R_proj = _rz(a) @ _ry(b) @ _rx(g) @ geo.R0
        return R_proj, {"a": a, "b": b, "g": g, "degenerate": degenerate, "reachable": reachable}

    def project_at_pan(self, R: np.ndarray, q1: float, b_prev: float | None = None) -> np.ndarray:
        """Closest reachable orientation to ``R`` with the pan joint at ``q1`` (no position involved)."""
        N = _rz(-q1).T @ R @ self.geo.R0.T
        b, g, _ = project_family(N, b_prev, self.frobenius_iters)
        return _rz(-q1) @ _ry(b) @ _rx(g) @ self.geo.R0

    def _planar(self, p0: np.ndarray, b: np.ndarray, g: float):
        """Both elbow solutions for wrist pitch(es) ``b`` (array) at roll ``g``: q2, q3, q4 arrays (2, n), ok."""
        geo = self.geo
        g = np.asarray(g, dtype=float)
        cg, sg = np.cos(g), np.sin(g)
        # w = v + Rx(g) u (x and z components), broadcast over g
        wx = geo.v[0] + geo.u[0]
        wz = geo.v[2] + sg * geo.u[1] + cg * geo.u[2]
        cb, sb = np.cos(b), np.sin(b)
        # Ry(b) @ w, x and z components
        Fx = p0[0] - (cb * wx + sb * wz)
        Fz = p0[2] - (-sb * wx + cb * wz)
        Dx, Dz = Fx - geo.S[0], Fz - geo.S[2]
        cos_e = (Dx * Dx + Dz * Dz - geo.l1**2 - geo.l2**2) / (2 * geo.l1 * geo.l2)
        inside = np.abs(cos_e) <= 1.0
        e = np.arccos(np.clip(cos_e, -1.0, 1.0))
        e = np.stack([e, -e])
        inside = np.broadcast_to(inside, e.shape)
        q3 = geo.a2 - geo.a1 - e
        q2 = geo.a1 + np.arctan2(geo.l2 * np.sin(e), geo.l1 + geo.l2 * np.cos(e)) - np.arctan2(Dz, Dx)
        q2, q3 = wrap_angle(q2), wrap_angle(q3)
        q4 = wrap_angle(b - q2 - q3)
        lim = self.arm.limits
        tol = 1e-7  # joints sitting exactly on a limit (clipped recordings) must count as feasible
        ok = inside & (q2 >= lim[1, 0] - tol) & (q2 <= lim[1, 1] + tol) & (q3 >= lim[2, 0] - tol)
        ok &= (q3 <= lim[2, 1] + tol) & (q4 >= lim[3, 0] - tol) & (q4 <= lim[3, 1] + tol)
        return q2, q3, q4, ok, inside, cos_e

    def candidates(self, T_target: np.ndarray, q_prev: np.ndarray | None = None) -> list[dict]:
        """Per pan branch: the exact solution for the projected orientation if the position allows it, else
        the feasible wrist pitch closest to the projected one (1-D scan), so position stays exact. The scan
        only runs when no branch has an exact solution (the usual case for on-manifold targets is one scan-free
        exact branch)."""
        geo = self.geo
        p, R = T_target[:3, 3], T_target[:3, :3]
        hx, hy = p[0] - geo.c[0], p[1] - geo.c[1]
        r = float(np.hypot(hx, hy))
        branches = []
        for branch in (0, 1):
            a, b_star, g, degenerate, reachable = self._branch(R, hx, hy, r, branch, q_prev)
            lim = self.arm.limits
            # Wrist roll beyond its range: clamp it here (costs orientation only, the position is re-solved
            # with the clamped roll below). Pan beyond its range: this branch cannot be exact at all.
            g_free = g
            g = -float(np.clip(wrap_angle(-g), lim[4, 0], lim[4, 1]))
            roll_clamped = abs(wrap_angle(g - g_free)) > 1e-9
            pan_ok = lim[0, 0] - 1e-7 <= wrap_angle(-a) <= lim[0, 1] + 1e-7
            p0 = _rz(a).T @ (p - geo.c) + geo.c  # target in the pan-0 frame
            q2, q3, q4, ok, inside, cos_e = self._planar(p0, np.array([b_star]), g)
            ok = ok & pan_ok
            if self.elbow_up:
                ok[0] = False
            branches.append([a, b_star, g, degenerate, reachable, p0, q2, q3, q4, ok, inside, cos_e, roll_clamped,
                             pan_ok, bool(ok.any())])
        any_exact = any(br[-1] for br in branches)
        out = []
        lim = self.arm.limits
        for br in branches:
            a, b_star, g, degenerate, reachable, p0, q2, q3, q4, ok, inside, cos_e, roll_clamped, pan_ok, exact = br
            b = b_star
            if not exact and not any_exact:
                N = _rz(a).T @ R @ geo.R0.T

                def best_roll(bb, N=N):
                    # argmax_g tr(Rx(g)^T Ry(bb)^T N), vectorized over bb, clamped to the roll range
                    cb, sb = np.cos(bb), np.sin(bb)
                    k11 = N[1, 1]
                    k22 = sb * N[0, 2] + cb * N[2, 2]
                    k21 = sb * N[0, 1] + cb * N[2, 1]
                    k12 = N[1, 2]
                    gg = np.arctan2(k21 - k12, k11 + k22)
                    return -np.clip(wrap_angle(-gg), lim[4, 0], lim[4, 1])

                grid = b_star + self._scan
                okg = self._planar(p0, grid, best_roll(grid))[3]
                feas = (okg[1] if self.elbow_up else okg.any(axis=0)) & pan_ok
                if feas.any():
                    j = int(np.argmin(np.where(feas, np.abs(self._scan), np.inf)))
                    # refine the feasibility boundary between the grid point and b* by bisection
                    lo_b, hi_b = float(grid[j]), b_star
                    for _ in range(25):
                        mid = 0.5 * (lo_b + hi_b)
                        okm = self._planar(p0, np.array([mid]), best_roll(np.array([mid])))[3]
                        if (okm[1] if self.elbow_up else okm.any()):
                            lo_b = mid
                        else:
                            hi_b = mid
                    b = lo_b
                    g = float(best_roll(np.array([b]))[0])
                    g_free = g
                    g = -float(np.clip(wrap_angle(-g), lim[4, 0], lim[4, 1]))
                    roll_clamped = abs(wrap_angle(g - g_free)) > 1e-9
                    q2, q3, q4, ok, inside, cos_e = self._planar(p0, np.array([b]), g)
                    ok = ok & pan_ok
                    exact = bool(ok.any())  # position-exact, with the orientation as close as the limits allow
                else:
                    reachable = False
            R_proj = _rz(a) @ _ry(b) @ _rx(g) @ geo.R0
            for k in range(2):
                if self.elbow_up and k == 0 and inside[k, 0] and np.arccos(np.clip(cos_e[0], -1, 1)) > 1e-6:
                    continue  # e > 0 branch: elbow folded the other way, absent from the recorded SO-101 data
                q = wrap_angle(np.array([-a, q2[k, 0], q3[k, 0], q4[k, 0], -g]))
                out.append({"q": q, "R_proj": R_proj, "degenerate": degenerate, "roll_clamped": roll_clamped,
                            "reachable": reachable and bool(inside[k, 0]), "exact": exact and bool(ok[k, 0])})
        return out

    def solve(self, T_target: np.ndarray, q_prev: np.ndarray | None = None) -> IKResult:
        arm = self.arm
        lim = arm.limits
        p_t = T_target[:3, 3]
        best, best_cost, best_pe = None, np.inf, np.inf
        cands = self.candidates(T_target, q_prev)
        exact = [c for c in cands if c["exact"]]
        pool = exact or cands  # position is the primary task: never trade it for orientation
        if q_prev is not None:
            near = [c for c in exact if np.abs(wrap_angle(c["q"] - q_prev)).max() <= self.max_jump]
            if near:
                pool = near
            else:
                # No exact solution near the previous joints (target outside the workspace or the limits, or
                # only reachable on another branch): also offer the local solution reached from q_prev, so
                # tracking degrades smoothly instead of flipping branches.
                q_loc = arm.clip(np.asarray(q_prev, dtype=float))
                R_goal = T_target[:3, :3]
                for _ in range(8):
                    R_goal = self.project_at_pan(T_target[:3, :3], q_loc[0], float(q_loc[1] + q_loc[2] + q_loc[3]))
                    q_loc, n = priority_step(arm, q_loc, p_t, R_goal)
                    if n < 1e-6:
                        break
                pool = cands + [{"q": q_loc, "R_proj": R_goal, "degenerate": False, "reachable": False,
                                 "exact": False}]
        for c in pool:
            q = c["q"]
            viol = float(np.sum(np.maximum(lim[:, 0] - q, 0) + np.maximum(q - lim[:, 1], 0)))
            Tc = arm.fk_single(arm.clip(q))
            pe = float(np.linalg.norm(Tc[:3, 3] - p_t))
            re = _rot_angle(Tc[:3, :3].T @ T_target[:3, :3])
            cont, jump = 0.0, 0.0
            if q_prev is not None:
                dq = np.abs(wrap_angle(q - q_prev))
                cont, jump = float(np.sum(dq)), max(0.0, float(dq.max()) - self.max_jump)
            cost = 10.0 * viol + 100.0 * pe + 10.0 * re + cont + 10.0 * jump
            if cost < best_cost:
                best, best_cost, best_pe = c, cost, pe
        q_raw = best["q"]
        q = arm.clip(q_raw)
        clamped = bool(np.any(np.abs(q - q_raw) > 1e-6))
        iters = 0
        # Polish away the idealization error (axes off by ~1e-5 rad, the fixed point near the pan axis).
        # Steps are accepted only if they reduce the position error, so it can never make things worse.
        pe = best_pe
        for _ in range(self.polish):
            if pe < 1e-7:
                break
            iters += 1
            q_new, n = priority_step(arm, q, p_t, best["R_proj"])
            pe_new = float(np.linalg.norm(arm.fk_single(q_new)[:3, 3] - p_t))
            if pe_new >= pe:
                break
            q, pe = q_new, pe_new
        clamped = clamped or best.get("roll_clamped", False)
        clamped = clamped or bool(np.any((q <= lim[:, 0] + 1e-9) | (q >= lim[:, 1] - 1e-9)))
        T, pos_err, rot_err, app_err = _errors(arm, q, T_target)
        R_proj = self.project_at_pan(T_target[:3, :3], q[0], float(q[1] + q[2] + q[3]))
        proj_err = _rot_angle(R_proj.T @ T_target[:3, :3])
        return IKResult(q, T, pos_err, rot_err, app_err, proj_err, best["reachable"], clamped,
                        best["degenerate"], iters)


class DLSIK:
    """Task-priority damped least squares. Position is the primary task, orientation the secondary."""

    def __init__(self, arm: ArmModel | None = None, damping: float = 1e-3, max_iters: int = 100,
                 tol: float = 1e-6, max_step: float = 0.3):
        self.arm = arm or ArmModel()
        self.lam2 = damping**2
        self.max_iters = max_iters
        self.tol = tol
        self.max_step = max_step
        self._analytic = AnalyticIK(self.arm, polish=0)

    def solve(self, T_target: np.ndarray, q_prev: np.ndarray | None = None) -> IKResult:
        arm = self.arm
        q = np.zeros(5) if q_prev is None else np.array(q_prev, dtype=float)
        p_t, R_t = T_target[:3, 3], T_target[:3, :3]
        it = 0
        # Merit for a backtracking line search: position dominates (1 rad of orientation = 2 cm). Without it the
        # secondary task can oscillate between two configurations near ill-conditioned wrist poses.
        def merit(qq, R_goal):
            T = arm.fk_single(qq)
            return float(np.linalg.norm(p_t - T[:3, 3])) + 0.02 * _rot_angle(R_goal.T @ T[:3, :3])

        for _ in range(self.max_iters):
            it += 1
            R_goal = self._analytic.project_at_pan(R_t, q[0], float(q[1] + q[2] + q[3]))
            m0 = merit(q, R_goal)
            step = self.max_step
            for _ in range(6):
                q_new, n = priority_step(arm, q, p_t, R_goal, self.lam2, step)
                if merit(q_new, R_goal) < m0:
                    break
                step = min(step, n) / 2
            else:
                break  # no descent direction left: converged (or stuck at a limit)
            q = q_new
            if n < self.tol:
                break
        # The secondary task converges to the reachable orientation; report it like the analytic solver.
        R_proj = self._analytic.project_at_pan(R_t, q[0], float(q[1] + q[2] + q[3]))
        _, s = self._analytic.project(T_target, q)
        T, pos_err, rot_err, app_err = _errors(arm, q, T_target)
        clamped = bool(np.any((q <= arm.limits[:, 0] + 1e-9) | (q >= arm.limits[:, 1] - 1e-9)))
        proj_err = _rot_angle(R_proj.T @ R_t)
        return IKResult(q, T, pos_err, rot_err, app_err, proj_err, pos_err < 5e-3, clamped, s["degenerate"], it)


def solve_trajectory(solver, T_targets: np.ndarray, q0: np.ndarray | None = None) -> list[IKResult]:
    """Sequential IK with warm start / branch continuity from the previous solution."""
    out = []
    q = q0
    for T in T_targets:
        r = solver.solve(T, q)
        out.append(r)
        q = r.q
    return out


def pose_from_joint_family(arm: ArmModel, a: float, b: float, g: float, p: np.ndarray) -> np.ndarray:
    """Helper for tests: a pose whose orientation is in the reachable family at pan -a."""
    geo = _Geometry(arm)
    return make_pose(rot_z(a) @ rot_y(b) @ rot_x(g) @ geo.R0, p)
