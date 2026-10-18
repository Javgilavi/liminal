import time

import numpy as np
import pytest
from kinematics.ik import DLSIK, AnalyticIK, _Geometry, pose_from_joint_family, solve_trajectory
from kinematics.se3 import make_pose, rot_x, rot_y, rot_z, rotvec_to_matrix
from kinematics.so101_kinematics import ArmModel

ARM = ArmModel()


def _random_q(n, seed=0, margin=0.95):
    lim = ARM.limits
    return np.random.default_rng(seed).uniform(lim[:, 0] * margin, lim[:, 1] * margin, (n, 5))


def test_product_of_exponentials_structure():
    # R(q) = Rz(-q1) Ry(q2 + q3 + q4) Rx(-q5) R0: the basis of the analytic solver and of the projection.
    geo = _Geometry(ARM)
    assert geo.axis_dev < 1e-4
    for q in _random_q(50, 1):
        R = ARM.fk_single(q)[:3, :3]
        R_poe = rot_z(-q[0]) @ rot_y(q[1] + q[2] + q[3]) @ rot_x(-q[4]) @ geo.R0
        assert np.allclose(R, R_poe, atol=1e-4)


@pytest.mark.parametrize("solver", [AnalyticIK(ARM, elbow_up=False), DLSIK(ARM)], ids=["analytic", "dls"])
def test_round_trip_on_manifold_with_warm_start(solver):
    rng = np.random.default_rng(2)
    for q in _random_q(300, 2):
        T = ARM.fk_single(q)
        if np.hypot(T[0, 3] - 0.0388, T[1, 3]) < 0.02:
            continue  # TCP over the base: pan ill-conditioned (see the cold-start test)
        r = solver.solve(T, q + rng.normal(0, 0.05, 5))
        assert r.pos_err < 1e-4
        assert r.rot_err < 2e-3
        assert r.proj_rot_err < 2e-3


def test_analytic_cold_start_is_global_away_from_pan_axis():
    an = AnalyticIK(ARM, elbow_up=False)
    for q in _random_q(500, 3):
        T = ARM.fk_single(q)
        if np.hypot(T[0, 3] - 0.0388, T[1, 3]) < 0.02:
            continue  # TCP over the base: pan is ill-defined, never used for manipulation
        r = an.solve(T, None)
        assert r.pos_err < 1e-4 and r.rot_err < 2e-3


def test_top_down_grasp_reachable_at_any_yaw():
    # Tool z straight down lies in every vertical plane, so any yaw maps onto wrist roll. Only the wrist roll
    # range (-157..163 deg, 320 of 360) can block it; there the joint clamps.
    an = AnalyticIK(ARM, elbow_up=False)
    q = None
    ok = 0
    yaws = np.linspace(-np.pi, np.pi, 73)
    for yaw in yaws:
        R = rot_z(yaw) @ np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])  # tool z down
        r = an.solve(make_pose(R, np.array([0.25, 0.05, 0.03])), q)
        assert r.proj_rot_err < 1e-3, yaw  # in the reachable family: nothing lost before limits
        if not r.clamped:
            assert r.pos_err < 1e-4 and r.rot_err < 1e-3
            ok += 1
        q = r.q
    assert ok / len(yaws) > 0.85


def test_out_of_plane_approach_loses_exactly_that_angle():
    an = AnalyticIK(ARM, elbow_up=False)
    p = np.array([0.30, 0.0, 0.10])  # straight ahead: the arm plane is the base xz plane
    for tilt in np.deg2rad([5, 15, 30, 45]):
        T = pose_from_joint_family(ARM, 0.0, 0.3, 0.2, p)
        T[:3, :3] = rot_z(tilt) @ T[:3, :3]  # yaw the approach out of the plane
        r = an.solve(T, None)
        assert r.pos_err < 1e-4
        # the out-of-plane yaw cannot be executed; the geodesic-optimal projection loses at most that angle
        # (slightly more than the tilt is possible: rolling moves the off-axis TCP, which moves the pan)
        assert 0.5 * tilt < r.rot_err <= 1.05 * tilt + 1e-3
        assert r.approach_err <= 1.05 * tilt + 1e-3


def test_projection_is_idempotent():
    an = AnalyticIK(ARM, elbow_up=False)
    rng = np.random.default_rng(4)
    for _ in range(50):
        T = make_pose(rotvec_to_matrix(rng.normal(0, 1.5, 3)), np.array([0.25, 0.1, 0.08]))
        R1, _ = an.project(T)
        R2, _ = an.project(make_pose(R1, T[:3, 3]))
        assert np.allclose(R1, R2, atol=1e-6)


def test_dls_and_analytic_agree_off_manifold():
    # Off-manifold targets: the analytic solver keeps the position exact; the numeric reference converges to
    # (nearly) the same orientation error, i.e. both find the same best reachable orientation.
    # single-shot solves: the seed is not a 30 Hz predecessor, so no jump penalty
    an, dls = AnalyticIK(ARM, elbow_up=False, max_jump=np.inf), DLSIK(ARM)
    geo = _Geometry(ARM)
    rng = np.random.default_rng(5)
    diffs = []
    for q in _random_q(100, 6, margin=0.7):
        T = ARM.fk_single(q)
        T[:3, :3] = rotvec_to_matrix(rng.normal(0, 0.15, 3)) @ T[:3, :3]
        ra, rd = an.solve(T, q), dls.solve(T, q)
        e = np.angle(np.exp(1j * (geo.a2 - geo.a1 - q[2])))
        if ra.degenerate or ra.clamped or rd.clamped or np.hypot(T[0, 3] - 0.0388, T[1, 3]) < 0.02 or abs(e) < 0.4:
            continue  # |e| small: arm nearly stretched, the workspace edge where position and wrist pitch couple
        assert ra.pos_err < 1e-4
        assert rd.pos_err < 3e-3
        diffs.append(ra.rot_err - rd.rot_err)
    assert abs(np.median(diffs)) < np.deg2rad(0.1)


def test_trajectory_continuity_and_speed():
    rng = np.random.default_rng(7)
    q = np.array([0.1, -0.3, 0.6, 0.4, 0.2])
    Q = q + np.cumsum(rng.normal(0, 0.01, (200, 5)), axis=0)
    Q = ARM.clip(Q)
    Ts = ARM.fk(Q)
    for solver in (AnalyticIK(ARM, elbow_up=False), DLSIK(ARM)):
        t0 = time.perf_counter()
        res = solve_trajectory(solver, Ts, Q[0])
        dt = (time.perf_counter() - t0) / len(Ts)
        qs = np.array([r.q for r in res])
        assert np.max(np.abs(np.diff(qs, axis=0))) < 0.1
        assert np.max(np.abs(qs - Q)) < 1e-3
        assert dt < 0.015  # generous for CI; real numbers are in the report


def test_elbow_up_default_tracks_data_branch_without_flips():
    # The default solver only uses the elbow branch of the recorded data (e < 0), so noisy targets cannot
    # drift onto the other branch. Configurations on that branch must still round-trip exactly.
    geo = _Geometry(ARM)
    an = AnalyticIK(ARM)
    rng = np.random.default_rng(8)
    n = 0
    for q in _random_q(400, 9, margin=0.9):
        e = np.angle(np.exp(1j * (geo.a2 - geo.a1 - q[2])))
        T = ARM.fk_single(q)
        if e > -0.05 or np.hypot(T[0, 3] - 0.0388, T[1, 3]) < 0.02:
            continue
        r = an.solve(T, q + rng.normal(0, 0.05, 5))
        assert r.pos_err < 1e-4 and r.rot_err < 2e-3
        n += 1
    assert n > 100
