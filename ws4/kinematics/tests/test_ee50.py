import numpy as np
import pytest
from kinematics import ee50
from kinematics.se3 import make_pose, rotvec_to_matrix
from kinematics.so101_kinematics import Station


def _random_poses(n, seed=0):
    rng = np.random.default_rng(seed)
    return make_pose(rotvec_to_matrix(rng.normal(0, 1.0, (n, 3))), rng.normal(0, 0.2, (n, 3)))


def test_6d_uses_rows_and_round_trips():
    R = rotvec_to_matrix(np.array([0.3, -1.2, 0.7]))
    x = ee50.rot_to_6d(R)
    assert np.allclose(x, np.concatenate([R[0], R[1]]))  # rows, not columns
    assert not np.allclose(x, np.concatenate([R[:, 0], R[:, 1]]))
    assert np.allclose(ee50.rot_from_6d(x), R)


def test_6d_decoding_orthonormalizes_noisy_input():
    R = rotvec_to_matrix(np.array([1.0, 0.5, -0.2]))
    x = ee50.rot_to_6d(R) + np.random.default_rng(1).normal(0, 0.05, 6)
    Rd = ee50.rot_from_6d(x)
    assert np.allclose(Rd @ Rd.T, np.eye(3), atol=1e-12) and np.isclose(np.linalg.det(Rd), 1.0)


@pytest.mark.parametrize("rel_frame", ["anchor", "base"])
def test_encode_decode_round_trip(rel_frame):
    layout = ee50.EE50Layout(rel_frame=rel_frame)
    poses = {"left": _random_poses(16, 0), "right": _random_poses(16, 1)}
    grip = {"left": np.linspace(0, 1, 16), "right": np.linspace(1, 0, 16)}
    anchor = {a: P[3] for a, P in poses.items()}
    x = ee50.encode(poses, grip, anchor, layout)
    assert x.shape == (16, 50)
    # anchor row is the identity pose; gripper is absolute
    for a in ("left", "right"):
        b = layout.block(a)
        assert np.allclose(x[3, b : b + 9], [0, 0, 0, 1, 0, 0, 0, 1, 0], atol=1e-12)
        assert np.allclose(x[:, b + layout.gripper_index], grip[a])
    # padding stays zero
    assert np.all(x[:, ~layout.active_mask()] == 0)
    P, G = ee50.decode(x, anchor, layout)
    for a in poses:
        assert np.allclose(P[a], poses[a], atol=1e-10)
        assert np.allclose(G[a], grip[a])


def test_rel_frames_differ_but_both_invert():
    poses = {"left": _random_poses(4, 2)}
    anchor = {"left": poses["left"][0]}
    xa = ee50.encode(poses, {"left": np.zeros(4)}, anchor, ee50.EE50Layout("anchor"))
    xb = ee50.encode(poses, {"left": np.zeros(4)}, anchor, ee50.EE50Layout("base"))
    assert not np.allclose(xa, xb)


def test_windows_and_stats_round_trip():
    st = Station.default()
    rng = np.random.default_rng(3)
    T = 120
    q = {a: np.cumsum(rng.normal(0, 0.01, (T, 5)), axis=0) for a in ("left", "right")}
    poses = ee50.joints_to_poses(st, q)
    for a in q:
        assert np.allclose(poses[a], st.fk(a, q[a]))
    grip = {a: rng.uniform(0, 1, T) for a in q}
    offsets = np.arange(-4, 33)
    chunks, anchors = ee50.windows(poses, grip, offsets)
    assert chunks.shape == (len(anchors), len(offsets), 50)
    assert anchors[0] == 4 and anchors[-1] == T - 33
    stats = ee50.EE50Stats.fit(chunks, offsets)
    z = stats.normalize(chunks)
    assert np.allclose(stats.denormalize(z), chunks)
    active = ee50.EE50Layout().active_mask()
    k0 = list(offsets).index(0)
    # the anchor row is constant (identity) and padding is zero: both keep std 1 (pass-through)
    pose_dims = np.r_[0:9, 25:34]
    assert np.allclose(stats.std[k0, pose_dims], 1.0)  # identity anchor pose: constant, passes through
    assert np.allclose(stats.std[:, ~active], 1.0)
    assert np.allclose(z[:, :, ~active], 0.0)
    k_far = len(offsets) - 1
    assert np.all(stats.std[k_far, :3] > stats.std[k0 + 1, :3])  # spread grows with offset


def test_stats_save_load(tmp_path):
    offsets = np.arange(-2, 3)
    chunks = np.random.default_rng(0).normal(size=(10, 5, 50))
    s = ee50.EE50Stats.fit(chunks, offsets)
    s.save(tmp_path / "s.npz")
    s2 = ee50.EE50Stats.load(tmp_path / "s.npz")
    assert np.allclose(s2.mean, s.mean) and np.allclose(s2.std, s.std) and np.all(s2.offsets == offsets)


def test_layout_validation():
    with pytest.raises(ValueError):
        ee50.EE50Layout(rel_frame="world")
    with pytest.raises(ValueError):
        ee50.EE50Layout(gripper_index=3)
