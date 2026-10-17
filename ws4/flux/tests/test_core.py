"""Contract, history, cameras, safety and quantiles: numpy only."""

import numpy as np
import pytest
from flux.cameras import CameraView, prepare_image
from flux.contract import (
    BIMANUAL_GRID_CONTRACT,
    SO101_CONTRACT,
    merge_bimanual,
    snapshot_indices,
    split_bimanual,
)
from flux.history import HistoryBuffer
from flux.package_tools import compute_quantiles
from flux.safety import SafetyConfig, SafetyGuard, limits_from_lerobot_calibration


def test_split_merge_roundtrip():
    x = np.arange(24, dtype=float).reshape(2, 12)
    parts = split_bimanual(x)
    assert parts["left"].shape == (2, 6) and parts["right"][0, 0] == 6
    assert np.array_equal(merge_bimanual(parts), x)
    with pytest.raises(ValueError):
        split_bimanual(np.zeros(6))


def test_snapshot_indices_match_lerobot_packer_rule():
    assert snapshot_indices(8, 2) == (0, 7)  # SO-101 package: oldest and current tick
    assert snapshot_indices(8, 1) == (7,)
    assert snapshot_indices(5, 3) == (0, 2, 4)
    assert SO101_CONTRACT.snapshots == (0, 7)
    assert BIMANUAL_GRID_CONTRACT.absolute_dims == (5, 11)


def test_history_pads_with_first_observation_and_anchors_on_state():
    h = HistoryBuffer(n_obs_steps=4, action_dim=2)
    img = {"cam": np.zeros((2, 2, 3), np.uint8)}
    h.push(np.array([1.0, 2.0]), img)
    w = h.window()
    assert np.array_equal(w.states, np.tile([1.0, 2.0], (4, 1)))
    assert np.array_equal(w.commands, w.states)  # no command yet: the state is the previous command
    h.record_command(np.array([1.5, 2.5]))
    h.push(np.array([1.4, 2.4]), {"cam": np.ones((2, 2, 3), np.uint8)})
    w = h.window(image_ticks=(0, 3))
    assert np.array_equal(w.states[-1], [1.4, 2.4]) and np.array_equal(w.states[0], [1.0, 2.0])
    assert np.array_equal(w.commands[-1], [1.5, 2.5])  # command issued before the newest observation
    assert np.array_equal(w.commands[-2], [1.0, 2.0])
    assert w.images["cam"].shape == (2, 2, 2, 3) and w.images["cam"][1].max() == 1


def test_prepare_image_crop_mirror_resize():
    img = np.zeros((480, 640, 3), np.uint8)
    img[:, 320:] = 255  # right half white
    out = prepare_image(img, CameraView("x", crop=(0.5, 0.0, 1.0, 1.0)), (64, 64))
    assert out.shape == (64, 64, 3) and out.min() == 255
    mirrored = prepare_image(img, CameraView("x", mirror=True), (48, 64))
    assert mirrored[:, :20].min() == 255 and mirrored[:, -20:].max() == 0
    wide = np.zeros((480, 848, 3), np.uint8)
    wide[:, :100] = 255  # only in the 16:9 margin
    cropped = prepare_image(wide, CameraView("x", aspect=4 / 3), (48, 64))
    assert cropped.max() == 0
    floats = prepare_image(np.ones((10, 10, 3), np.float32), CameraView("x"), (10, 10))
    assert floats.dtype == np.uint8 and floats.min() == 255


def test_safety_guard_steps_clamps_and_holds_nan():
    cfg = SafetyConfig(lower=-np.ones(12) * 10, upper=np.ones(12) * 10, max_step=np.ones(12))
    guard = SafetyGuard(cfg)
    prev = np.zeros(12)
    cmd = np.zeros(12)
    cmd[0], cmd[1], cmd[2] = 5.0, np.nan, -0.5
    out, report = guard.filter(cmd, prev)
    assert out[0] == 1.0 and out[1] == 0.0 and out[2] == -0.5
    assert report.nonfinite and report.intervened
    out, report = guard.filter(np.full(12, 10.5), np.full(12, 10.0))
    assert np.allclose(out, 10.0) and report.clamped.all()


def test_limits_from_calibration(tmp_path):
    joints = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    calib = {j: {"range_min": 1000, "range_max": 3047} for j in joints}
    path = tmp_path / "c.json"
    path.write_text(__import__("json").dumps(calib))
    lo, hi = limits_from_lerobot_calibration(path, margin_deg=0.0)
    assert np.allclose(hi[:5], 2047 / 2 * 360 / 4095) and np.allclose(lo[:5], -hi[:5])
    assert lo[5] == 0 and hi[5] == 100


def test_quantiles_follow_export_base_delta_rule():
    commands = np.array([[0.0, 10.0], [1.0, 20.0], [3.0, 30.0], [100.0, 40.0], [101.0, 50.0]])
    states = commands + 0.5
    episodes = np.array([0, 0, 0, 1, 1])
    stats = compute_quantiles(commands, states, episodes, absolute_dims=(-1,))
    # Deltas of channel 0: 0 (episode start), 1, 2, 0 (episode start), 1. Channel 1 stays absolute.
    deltas = np.array([0.0, 1.0, 2.0, 0.0, 1.0])
    assert np.isclose(stats["action"]["q99"][0], np.percentile(deltas, 99))
    assert np.isclose(stats["action"]["q01"][1], np.percentile(commands[:, 1], 1))
    assert np.isclose(stats["state"]["q01"][0], np.percentile(states[:, 0], 1))


def test_crosscheck_observation_and_compare(tmp_path):
    from flux.crosscheck_so101 import NPZ_SCENE, NPZ_WRIST, compare, write_observation

    write_observation(str(tmp_path / "obs.npz"), None, None, [0, 0, 0, 0, 0, 50])
    with np.load(tmp_path / "obs.npz") as data:
        assert data[NPZ_SCENE].shape == (256, 256, 3) and data[NPZ_SCENE].dtype == np.uint8
        assert data[NPZ_WRIST].shape == (256, 256, 3) and data["state"].dtype == np.float32
    a = np.zeros((1, 42, 6))
    np.save(tmp_path / "a.npy", a)
    np.save(tmp_path / "b.npy", a + 1.0)
    report = compare(str(tmp_path / "a.npy"), str(tmp_path / "b.npy"))
    assert report["max_abs_diff"] == [1.0] * 6
