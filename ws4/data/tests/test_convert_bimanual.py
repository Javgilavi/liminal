import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contracts import ARM_JOINTS, FLUX_SO101, XLE_12D, m100_to_degrees  # noqa: E402  # noqa: E402
from convert_bimanual import build_index, main, remap, transform_stats  # noqa: E402
from inspect_dataset import describe, diff_against, load_meta  # noqa: E402


def _read(root: Path):
    df = pd.read_parquet(root / "data/chunk-000/file-000.parquet")
    return np.stack(df["action"].to_numpy()), np.stack(df["observation.state"].to_numpy()), df


def test_build_index_orders_by_contract():
    names = [f"right_arm_{j}.pos" for j in ARM_JOINTS] + [f"left_arm_{j}.pos" for j in ARM_JOINTS] + ["x.vel"]
    assert build_index(names, ("left", "right")) == list(range(6, 12)) + list(range(0, 6))
    with pytest.raises(KeyError):
        build_index(names[:5], ("right",))


def test_transform_stats_slices_and_scales():
    st = {"min": [1.0, 2.0, 3.0], "max": [4.0, 5.0, 6.0], "count": [10, 10, 10], "q01": [0.0, 1.0, 2.0]}
    out = transform_stats(st, [2, 0], [2.0, 1.0])
    assert out["min"] == [6.0, 1.0] and out["max"] == [12.0, 4.0] and out["count"] == [10, 10]


def test_inspect_flags_bimanual(bi_dataset):
    root, _ = bi_dataset
    desc = describe(load_meta(root))
    assert desc["arm_units"] == {"left": "degrees", "right": "degrees"}
    per_arm = diff_against(desc, FLUX_SO101)
    assert per_arm["ok"] and set(per_arm["action_index"]) == {"left", "right"}
    xle = diff_against(desc, XLE_12D)
    assert xle["ok"] and xle["camera_map"]["observation.images.head"] == "observation.images.left_top"


def test_per_arm_split(bi_dataset, tmp_path):
    root, arrays = bi_dataset
    dst = tmp_path / "left"
    main([str(root), str(dst), "--to", "per_arm", "--side", "left", "--copy"])
    act, state, df = _read(dst)
    np.testing.assert_allclose(act, arrays["action"][:, :6], rtol=1e-6)
    np.testing.assert_allclose(state, arrays["state"][:, :6], rtol=1e-6)
    info = json.loads((dst / "meta/info.json").read_text())
    assert info["features"]["action"]["names"] == [f"{j}.pos" for j in ARM_JOINTS]
    assert set(k for k, f in info["features"].items() if f["dtype"] == "video") == {
        "observation.images.scene", "observation.images.wrist"}
    assert (dst / "videos/observation.images.wrist/chunk-000/file-000.mp4").read_bytes() == b"fake video left_wrist"
    assert (dst / "videos/observation.images.scene/chunk-000/file-000.mp4").read_bytes() == b"fake video left_top"
    ep = pd.read_parquet(dst / "meta/episodes/chunk-000/file-000.parquet")
    assert "videos/observation.images.wrist/from_timestamp" in ep.columns
    assert not any("right_wrist" in c or "effort" in c for c in ep.columns)
    assert len(ep["stats/action/min"].iloc[0]) == 6
    stats = json.loads((dst / "meta/stats.json").read_text())
    np.testing.assert_allclose(stats["action"]["max"], arrays["action"][:, :6].max(0), rtol=1e-5)
    assert set(df.columns) == {"action", "observation.state", "timestamp", "frame_index", "episode_index", "index",
                               "task_index"}


def test_xle12_drops_extras_and_converts_m100(xle_dataset, tmp_path):
    root, arrays = xle_dataset
    dst = tmp_path / "xle12"
    main([str(root), str(dst), "--to", "xle12", "--src-units", "m100", "--copy"])
    act, _, _ = _read(dst)
    assert act.shape[1] == 12
    expected = arrays["action"][:, :12].copy()
    for i, j in enumerate(list(ARM_JOINTS) * 2):
        if j != "gripper":
            expected[:, i] = m100_to_degrees(expected[:, i], j)
    np.testing.assert_allclose(act, expected, rtol=1e-5)
    info = json.loads((dst / "meta/info.json").read_text())
    assert info["features"]["action"]["names"][0] == "left_shoulder_pan.pos"
    assert info["ws4_conversion"]["source_arm_units"] == "m100"
    cams = {k for k, f in info["features"].items() if f["dtype"] == "video"}
    assert cams == set(XLE_12D.cameras)


def test_refuses_wrong_fps(tmp_path):
    from conftest import BI_NAMES, make_v3_dataset

    root = tmp_path / "bi60"
    make_v3_dataset(root, BI_NAMES, ["top", "left_wrist", "right_wrist"], fps=60)
    with pytest.raises(SystemExit):
        remap(root, tmp_path / "out", ("left",), {"observation.images.scene": "observation.images.top"}, "degrees",
              "so101_follower")


def test_flip_stats_swaps_quantiles():
    from convert_bimanual import flip_stats

    st = {"min": [0.0, 10.0], "max": [5.0, 60.0], "mean": [2.0, 20.0], "std": [1.0, 3.0],
          "q01": [0.1, 11.0], "q99": [4.9, 59.0]}
    out = flip_stats(st, [1])
    assert out["min"] == [0.0, 40.0] and out["max"] == [5.0, 90.0]
    assert out["q01"] == [0.1, 41.0] and out["q99"] == [4.9, 89.0]
    assert out["mean"] == [2.0, 80.0] and out["std"] == [1.0, 3.0]


def test_per_arm_flip_gripper_and_task_suffix(bi_dataset, tmp_path):
    root, arrays = bi_dataset
    dst = tmp_path / "right"
    main([str(root), str(dst), "--to", "per_arm", "--side", "right", "--copy", "--flip-action-gripper"])
    act, state, _ = _read(dst)
    np.testing.assert_allclose(act[:, 5], 100.0 - arrays["action"][:, 11], rtol=1e-5)
    np.testing.assert_allclose(state[:, 5], arrays["state"][:, 11], rtol=1e-5)  # state is never flipped
    tasks = pd.read_parquet(dst / "meta/tasks.parquet")
    assert list(tasks.index) == ["do the thing with the right arm"]
    ep = pd.read_parquet(dst / "meta/episodes/chunk-000/file-000.parquet")
    assert list(ep["tasks"].iloc[0]) == ["do the thing with the right arm"]
    stats = json.loads((dst / "meta/stats.json").read_text())
    assert stats["action"]["max"][5] == pytest.approx(100.0 - arrays["action"][:, 11].min(), rel=1e-5)


def test_per_arm_negate_joints(bi_dataset, tmp_path):
    root, arrays = bi_dataset
    dst = tmp_path / "right_d"
    main([str(root), str(dst), "--to", "per_arm", "--side", "right", "--copy", "--src-units", "degrees",
          "--negate", "shoulder_lift", "wrist_roll"])
    act, state, _ = _read(dst)
    for out, src in ((act, arrays["action"]), (state, arrays["state"])):
        np.testing.assert_allclose(out[:, [1, 4]], -src[:, [7, 10]], rtol=1e-5)
        np.testing.assert_allclose(out[:, [0, 2, 3, 5]], src[:, [6, 8, 9, 11]], rtol=1e-5)
    stats = json.loads((dst / "meta/stats.json").read_text())
    for key, src in (("action", arrays["action"]), ("observation.state", arrays["state"])):
        assert stats[key]["min"][1] == pytest.approx(-src[:, 7].max(), rel=1e-5)
        assert stats[key]["q01"][4] == pytest.approx(-np.percentile(src[:, 10], 99), rel=1e-4)
        assert stats[key]["mean"][1] == pytest.approx(-src[:, 7].mean(), rel=1e-4)
    ep = pd.read_parquet(dst / "meta/episodes/chunk-000/file-000.parquet")
    first = state[: int(ep["length"].iloc[0])]
    for stat, fn in (("max", np.max), ("min", np.min)):
        assert ep[f"stats/observation.state/{stat}"].iloc[0][1] == pytest.approx(fn(first[:, 1]), rel=1e-5)
    info = json.loads((dst / "meta/info.json").read_text())
    assert info["ws4_conversion"]["negated"] == ["shoulder_lift", "wrist_roll"]


def test_merge_precheck(bi_dataset, tmp_path):
    from merge_datasets import compatibility

    root, _ = bi_dataset
    main([str(root), str(tmp_path / "l"), "--to", "per_arm", "--side", "left", "--copy"])
    main([str(root), str(tmp_path / "r"), "--to", "per_arm", "--side", "right", "--copy"])
    infos = [json.loads((tmp_path / s / "meta/info.json").read_text()) for s in ("l", "r")]
    assert compatibility(infos) == [[], []]
    other = json.loads(json.dumps(infos[1]))
    other["features"]["observation.images.wrist"]["info"]["video.codec"] = "h264"
    other["features"]["observation.images.wrist"]["info"]["video.crf"] = 99  # encoder settings may differ
    other["robot_type"] = "bi_so_follower"
    why = compatibility([infos[0], other])[1]
    assert any("robot_type" in w for w in why)
    assert any("info.video.codec" in w for w in why) and not any("crf" in w for w in why)
