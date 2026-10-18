import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # ws4/, home of the ws4-ik `kinematics` package

pytest.importorskip("kinematics", reason="needs ws4/kinematics (ws4-ik)")

import ee_convert  # noqa: E402
from kinematics.se3 import rotvec_to_matrix  # noqa: E402


def _random_rotations(n, seed=0):
    rng = np.random.default_rng(seed)
    return rotvec_to_matrix(rng.uniform(-1.5, 1.5, size=(n, 3)))


def _matrix_to_quat_xyzw(R):
    # Shepperd-free version, fine for the well-conditioned test rotations (trace > -1)
    w = np.sqrt(np.maximum(1 + np.trace(R, axis1=-2, axis2=-1), 1e-12)) / 2
    x = (R[..., 2, 1] - R[..., 1, 2]) / (4 * w)
    y = (R[..., 0, 2] - R[..., 2, 0]) / (4 * w)
    z = (R[..., 1, 0] - R[..., 0, 1]) / (4 * w)
    return np.stack([x, y, z, w], -1)


def test_quat_and_rot6d_conventions():
    R = _random_rotations(50)
    np.testing.assert_allclose(ee_convert.quat_xyzw_to_matrix(_matrix_to_quat_xyzw(R)), R, atol=1e-9)
    rows = R[:, :2, :].reshape(-1, 6)
    cols = np.swapaxes(R, -1, -2)[:, :2, :].reshape(-1, 6)
    np.testing.assert_allclose(ee_convert.rot6d_to_matrix(rows, "rows"), R, atol=1e-9)
    np.testing.assert_allclose(ee_convert.rot6d_to_matrix(cols, "cols"), R, atol=1e-9)


def test_ee20_roundtrip_and_tool_fix():
    R = _random_rotations(20, seed=1)
    p = np.random.default_rng(2).normal(size=(20, 3))
    poses = {a: np.tile(np.eye(4), (20, 1, 1)) for a in ee_convert.ARMS}
    for a in poses:
        poses[a][:, :3, :3], poses[a][:, :3, 3] = R, p
    g = {a: np.linspace(0, 1, 20) for a in ee_convert.ARMS}
    back, gb = ee_convert.ee20_to_poses(ee_convert.poses_to_ee20(poses, g))
    np.testing.assert_allclose(back["left"], poses["left"], atol=1e-9)
    np.testing.assert_allclose(gb["right"], g["right"])
    fixed = ee_convert.apply_tool_fix(poses, np.diag([1.0, -1.0, -1.0]))
    np.testing.assert_allclose(fixed["left"][:, :3, 3], p)  # a tool rotation does not move the TCP


def _write_ee_dataset(root: Path, X: np.ndarray, ep: np.ndarray):
    (root / "data/chunk-000").mkdir(parents=True)
    pd.DataFrame({"observation.state": list(X.astype(np.float32)), "episode_index": ep}).to_parquet(
        root / "data/chunk-000/file-000.parquet", index=False)


def test_read_roco_preset(tmp_path):
    n = 60
    R = _random_rotations(n, seed=3)
    q = _matrix_to_quat_xyzw(R)
    X = np.zeros((n, 16))
    X[:, 0:3], X[:, 3:7] = 0.1, q
    X[:, 7:10], X[:, 10:14] = -0.2, q
    X[:, 14], X[:, 15] = np.linspace(0.25, 1.0, n), 0.5
    _write_ee_dataset(tmp_path / "roco", X, np.repeat([0, 1], n // 2))
    ee, ep = ee_convert.read_lerobot_ee(tmp_path / "roco", "roco")
    assert ee.shape == (n, 20) and set(ep) == {0, 1}
    np.testing.assert_allclose(ee[:, 0:3], 0.1, atol=1e-6)
    np.testing.assert_allclose(ee[:, 3:9], R[:, :2, :].reshape(-1, 6), atol=1e-5)
    assert ee[:, 9].min() == pytest.approx(0.0) and ee[:, 9].max() == pytest.approx(1.0)


def test_so101_joints_fk_and_stats(tmp_path, bi_dataset):
    root, arrays = bi_dataset
    ee, ep = ee_convert.read_so101_joints(root)
    assert ee.shape == (len(arrays["action"]), 20)
    # TCP stays within an SO-101's reach (< 0.6 m from its base in a station a few decimetres wide)
    assert np.abs(ee[:, [0, 1, 2, 10, 11, 12]]).max() < 1.0
    stats = ee_convert.fit_ee50_stats(ee, ep, np.arange(-2, 5), stride=1)
    assert stats.mean.shape == (7, 50)
    i0 = list(stats.offsets).index(0)
    np.testing.assert_allclose(stats.mean[i0, 0:3], 0.0, atol=1e-9)  # the anchor encodes to the identity


def test_cli_writes_npz(tmp_path, bi_dataset):
    root, _ = bi_dataset
    out = tmp_path / "ee.npz"
    assert ee_convert.main([str(root), "--preset", "so101_joints", "--out", str(out)]) == 0
    d = np.load(out)
    assert d["ee20"].shape[1] == 20 and len(d["episode"]) == len(d["ee20"])
    json.dumps({"ok": True})
