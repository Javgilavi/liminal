"""Synthetic haller-recorder datasets for CPU tests: real LeRobot v3.0 layout, tiny real mp4 files (PyAV, mpeg4)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

TELEOP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TELEOP))
sys.path.insert(0, str(TELEOP.parent / "data"))

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
NAMES = [f"{s}_{j}" for s in ("left", "right") for j in JOINTS]
CAMS = ("head", "left_wrist", "right_wrist")
CAL_FILE = {j: {"id": i + 1, "drive_mode": 0, "homing_offset": -1000 + 10 * i, "range_min": 800 + i,
                "range_max": 3200 - i} for i, j in enumerate(JOINTS)}


def write_mp4(path: Path, n: int, hw=(480, 640)) -> None:
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as c:
        s = c.add_stream("mpeg4", rate=30)
        s.height, s.width = hw
        s.pix_fmt = "yuv420p"
        for i in range(n):
            img = np.full((hw[0], hw[1], 3), (i * 7) % 255, np.uint8)
            for pkt in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(pkt)
        for pkt in s.encode():
            c.mux(pkt)


def _stats(arr: np.ndarray) -> dict:
    return {"min": arr.min(0).tolist(), "max": arr.max(0).tolist(), "mean": arr.mean(0).tolist(),
            "std": arr.std(0).tolist(), "count": [len(arr)],
            "q01": np.percentile(arr, 1, 0).tolist(), "q99": np.percentile(arr, 99, 0).tolist()}


def make_haller_dataset(root: Path, *, n_eps: int = 2, ep_len: int = 24, fps: int = 30, codec: str = "av1",
                        rig: str = "real", tasks: tuple[str, ...] = ("Open the box",), action_eq_state: bool = False,
                        short_video: int = 0, measured_hz: float = 30.01, calib: dict | None = None,
                        hw=(480, 640)) -> Path:
    import pandas as pd

    rng = np.random.default_rng(0)
    n = n_eps * ep_len
    # column i of observation.state encodes (side, joint): 10*i + small motion, gripper inside 0..100
    state = np.tile(np.arange(12, dtype=np.float32) * 10 - 55, (n, 1)) + rng.uniform(-3, 3, (n, 12)).astype(np.float32)
    state[:, 5] = rng.uniform(5, 60, n)
    state[:, 11] = rng.uniform(5, 60, n)
    action = state.copy() if action_eq_state else state + 0.5
    ep = np.repeat(np.arange(n_eps), ep_len)
    fr = np.tile(np.arange(ep_len), n_eps)
    df = pd.DataFrame({
        "observation.state": list(state), "action": list(action),
        "observation.effort": list(np.zeros((n, 12), np.float32)),
        "observation.wall_clock": list((fr / fps).astype(np.float32)[:, None]),
        "timestamp": (fr / fps).astype(np.float32), "frame_index": fr, "episode_index": ep,
        "index": np.arange(n), "task_index": ep % len(tasks),
    })
    (root / "data/chunk-000").mkdir(parents=True)
    df.to_parquet(root / "data/chunk-000/file-000.parquet", index=False)
    rows = []
    for e in range(n_eps):
        row = {"episode_index": e, "tasks": [tasks[e % len(tasks)]], "length": ep_len,
               "dataset_from_index": e * ep_len, "dataset_to_index": (e + 1) * ep_len}
        sl = slice(e * ep_len, (e + 1) * ep_len)
        for v, arr in (("observation.state", state), ("action", action)):
            for k, val in _stats(arr[sl]).items():
                row[f"stats/{v}/{k}"] = np.asarray(val)
        for cam in CAMS:
            key = f"observation.images.{cam}"
            frames = ep_len - (short_video if e == n_eps - 1 and cam == "left_wrist" else 0)
            write_mp4(root / f"videos/{key}/chunk-000/file-{e:03d}.mp4", frames, hw)
            row |= {f"videos/{key}/chunk_index": 0, f"videos/{key}/file_index": e,
                    f"videos/{key}/from_timestamp": 0.0, f"videos/{key}/to_timestamp": ep_len / fps}
            row[f"stats/{key}/mean"] = [[[0.5]], [[0.5]], [[0.5]]]
        rows.append(row)
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(root / "meta/episodes/chunk-000/file-000.parquet", index=False)
    pd.DataFrame({"task_index": range(len(tasks))}, index=pd.Index(list(tasks), name="task")).to_parquet(
        root / "meta/tasks.parquet")
    vid_info = {"video.height": hw[0], "video.width": hw[1], "video.codec": codec, "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False, "video.fps": fps, "video.channels": 3, "has_audio": False}
    feats = {
        "observation.state": {"dtype": "float32", "shape": [12], "names": NAMES},
        "action": {"dtype": "float32", "shape": [12], "names": NAMES},
        "observation.effort": {"dtype": "float32", "shape": [12], "names": NAMES},
        "observation.wall_clock": {"dtype": "float32", "shape": [1], "names": ["t"]},
        **{f"observation.images.{c}": {"dtype": "video", "shape": [hw[0], hw[1], 3],
                                       "names": ["height", "width", "channels"], "info": dict(vid_info)}
           for c in CAMS},
        **{k: {"dtype": t, "shape": [1], "names": None} for k, t in
           (("timestamp", "float32"), ("frame_index", "int64"), ("episode_index", "int64"), ("index", "int64"),
            ("task_index", "int64"))},
    }
    cal = calib if calib is not None else {"left": CAL_FILE, "right": CAL_FILE}
    joints = {}
    for side in ("left", "right"):
        for j in JOINTS:
            if rig == "real":
                c = cal[side][j]
                joints[f"{side}_{j}"] = {"source": "feetech_calibration", "range_min_ticks": c["range_min"],
                                         "range_max_ticks": c["range_max"], "homing_offset": c["homing_offset"],
                                         "drive_mode": c["drive_mode"]}
            else:
                joints[f"{side}_{j}"] = {"source": "declared_joint_range", "range_min_ticks": None}
    info = {"codebase_version": "v3.0", "robot_type": "haller_bimanual", "fps": fps, "total_episodes": n_eps,
            "total_frames": n, "total_tasks": len(tasks), "chunks_size": 1000, "data_files_size_in_mb": 100,
            "video_files_size_in_mb": 0.0001,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            "features": feats, "haller_joint_calibration": {"state_unit": "deg", "joints": joints},
            "haller_rate": {"fps_written": fps, "measured_hz": measured_hz}}
    (root / "meta/info.json").write_text(json.dumps(info, indent=1))
    stats = {v: _stats(a) for v, a in (("observation.state", state), ("action", action))}
    (root / "meta/stats.json").write_text(json.dumps(stats))
    return root


@pytest.fixture
def haller_ds(tmp_path):
    return make_haller_dataset(tmp_path / "rec")


@pytest.fixture
def calib_dir(tmp_path):
    d = tmp_path / "calib"
    d.mkdir()
    for cid in ("haller_left", "haller_right"):
        (d / f"{cid}.json").write_text(json.dumps(CAL_FILE))
    return d
