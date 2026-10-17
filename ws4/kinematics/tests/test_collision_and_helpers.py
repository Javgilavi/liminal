import numpy as np
import pytest
from kinematics import collision
from kinematics.ik import AnalyticIK
from kinematics.mj_collision import SCENE
from kinematics.se3 import make_pose, rotvec_to_matrix
from kinematics.so101_kinematics import Station
from kinematics.validate import interp_poses, smooth_noise


def test_segment_distance_matches_brute_force():
    rng = np.random.default_rng(0)
    p1, q1, p2, q2 = (rng.normal(size=(200, 3)) for _ in range(4))
    d = collision.segment_distance(p1, q1, p2, q2)
    s = np.linspace(0, 1, 201)
    for i in range(0, 200, 20):
        a = p1[i] + (q1[i] - p1[i]) * s[:, None]
        b = p2[i] + (q2[i] - p2[i]) * s[:, None]
        brute = np.min(np.linalg.norm(a[:, None] - b[None], axis=-1))
        assert d[i] <= brute + 1e-9 and brute - d[i] < 0.02


def test_capsule_check_shapes_and_clear_zero_pose():
    st = Station.default()
    q = {s: np.zeros((3, 5)) for s in ("left", "right")}
    out = collision.check(st, q)
    assert out["inter_arm"].shape == (3,) and np.all(out["inter_arm"] > 0)


@pytest.mark.skipif(not SCENE.exists(), reason="ws4/sim scene not present")
def test_mujoco_station_zero_pose_is_collision_free():
    pytest.importorskip("mujoco")
    from kinematics.mj_collision import MjCollision

    try:
        mc = MjCollision()
    except ValueError as exc:  # ws4/sim regenerating its model (another workstream): not our failure
        pytest.skip(f"sim scene does not load right now: {exc}")
    assert mc.contacts({"left": np.zeros(5), "right": np.zeros(5)}) == {}
    # both TCPs sent to the same point in front of the torso: the grippers must collide
    st = Station.load()
    ik = AnalyticIK(st.arm)
    R = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
    q = {s: ik.solve(st.to_base(s, make_pose(R, np.array([0.2, 0.0, 0.08]))), None).q for s in ("left", "right")}
    assert "inter_arm" in mc.contacts(q)


def test_smooth_noise_std_and_interp_endpoints():
    n = smooth_noise(np.random.default_rng(0), 20000, 3, 0.01)
    assert np.allclose(n.std(axis=0), 0.01, rtol=0.1)
    T = make_pose(rotvec_to_matrix(np.array([[0, 0, 0], [0.3, 0, 0], [0.3, 0.2, 0]])), np.eye(3))
    U = interp_poses(T, 3)
    assert len(U) == 7 and np.allclose(U[0], T[0]) and np.allclose(U[3], T[1]) and np.allclose(U[-1], T[-1])
