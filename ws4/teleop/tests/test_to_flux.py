import json

import numpy as np
import pandas as pd
import to_flux


def _state(root):
    df = pd.concat(pd.read_parquet(p) for p in sorted((root / "data").rglob("*.parquet")))
    return np.stack(df["observation.state"].to_numpy())


def test_per_arm_swaps_haller_sides_to_robot_sides(haller_ds, tmp_path):
    out = to_flux.convert(haller_ds, tmp_path / "out", "rec", ("per_arm",))
    src = _state(haller_ds)
    # robot left = haller right = columns 6..11; robot right = haller left = columns 0..5
    assert np.allclose(_state(out["left"]), src[:, 6:])
    assert np.allclose(_state(out["right"]), src[:, :6])
    info = json.loads((out["left"] / "meta/info.json").read_text())
    assert info["features"]["observation.state"]["names"][0] == "shoulder_pan.pos"
    assert info["robot_type"] == "so_follower"
    assert set(k for k in info["features"] if "images" in k) == {"observation.images.scene", "observation.images.wrist"}
    assert info["ws4_teleop"]["recorded_side"] == "right" and info["ws4_teleop"]["robot_side"] == "left"
    # robot-left wrist video is the haller right_wrist file (hard link or copy, same bytes)
    a = (out["left"] / "videos/observation.images.wrist/chunk-000/file-000.mp4").read_bytes()
    b = (haller_ds / "videos/observation.images.right_wrist/chunk-000/file-000.mp4").read_bytes()
    assert a == b
    tasks = pd.read_parquet(out["left"] / "meta/tasks.parquet").index.tolist()
    assert tasks == ["Open the box with the left arm"]
    for k in ("observation.effort", "observation.wall_clock"):
        assert k not in info["features"]


def test_xle12_robot_order_and_names(haller_ds, tmp_path):
    out = to_flux.convert(haller_ds, tmp_path / "out", "rec", ("xle12",))["xle12"]
    src, got = _state(haller_ds), _state(out)
    assert np.allclose(got[:, :6], src[:, 6:]) and np.allclose(got[:, 6:], src[:, :6])
    info = json.loads((out / "meta/info.json").read_text())
    names = info["features"]["action"]["names"]
    assert names[0] == "left_shoulder_pan.pos" and names[6] == "right_shoulder_pan.pos"
    lw = (out / "videos/observation.images.left_wrist/chunk-000/file-000.mp4").read_bytes()
    assert lw == (haller_ds / "videos/observation.images.right_wrist/chunk-000/file-000.mp4").read_bytes()
    stats = json.loads((out / "meta/stats.json").read_text())
    assert np.allclose(stats["observation.state"]["mean"][:6], np.asarray(
        json.loads((haller_ds / "meta/stats.json").read_text())["observation.state"]["mean"])[6:], atol=1e-4)


def test_sides_as_recorded_keeps_order(haller_ds, tmp_path):
    out = to_flux.convert(haller_ds, tmp_path / "out", "rec", ("per_arm",), sides_as_recorded=True)
    assert np.allclose(_state(out["left"]), _state(haller_ds)[:, :6])


def test_merge_precheck_against_reference(haller_ds, tmp_path):
    out = to_flux.convert(haller_ds, tmp_path / "out", "rec", ("per_arm",))
    res = to_flux.precheck([out["left"]], [out["right"]])
    assert all(not why for _, why in res)


def test_units_must_be_known(haller_ds, tmp_path):
    info = json.loads((haller_ds / "meta/info.json").read_text())
    del info["haller_joint_calibration"]
    (haller_ds / "meta/info.json").write_text(json.dumps(info))
    try:
        to_flux.convert(haller_ds, tmp_path / "out", "rec", ("per_arm",))
    except SystemExit as e:
        assert "--src-units" in str(e)
    else:
        raise AssertionError("expected SystemExit")
