"""Tiny synthetic LeRobot v3.0 datasets for CPU tests (no network, no real video)."""

import json
from pathlib import Path

import numpy as np
import pytest

BI_NAMES = [f"{s}_{j}.pos" for s in ("left", "right") for j in
            ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")]
XLE_NAMES = [n.replace("left_", "left_arm_").replace("right_", "right_arm_") for n in BI_NAMES] + [
    "head_motor_1.pos", "head_motor_2.pos", "x.vel", "y.vel", "theta.vel"]


def _stats(arr: np.ndarray) -> dict:
    return {
        "min": arr.min(0).tolist(), "max": arr.max(0).tolist(), "mean": arr.mean(0).tolist(),
        "std": arr.std(0).tolist(), "count": [len(arr)],
        "q01": np.percentile(arr, 1, axis=0).tolist(), "q99": np.percentile(arr, 99, axis=0).tolist(),
    }


def make_v3_dataset(root: Path, names: list[str], cameras: list[str], fps: int = 30, n_eps: int = 2,
                    ep_len: int = 20, scale: float = 115.0, seed: int = 0) -> dict:
    """Write a minimal v3.0 layout. Joint channel i holds values that encode (episode, frame, channel)."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    d = len(names)
    n = n_eps * ep_len
    state = (rng.uniform(-1, 1, size=(n, d)) * scale).astype(np.float32)
    for i, nm in enumerate(names):
        if nm.endswith("gripper.pos"):
            state[:, i] = rng.uniform(0, 60, size=n)
    action = state + 1.0
    ep_index = np.repeat(np.arange(n_eps), ep_len)
    frame_index = np.tile(np.arange(ep_len), n_eps)
    df = pd.DataFrame({
        "observation.state": list(state), "action": list(action),
        "observation.effort": list(np.zeros((n, d), np.float32)),
        "timestamp": (frame_index / fps).astype(np.float32), "frame_index": frame_index,
        "episode_index": ep_index, "index": np.arange(n), "task_index": np.zeros(n, dtype=np.int64),
    })
    (root / "data/chunk-000").mkdir(parents=True)
    df.to_parquet(root / "data/chunk-000/file-000.parquet", index=False)

    feats = {
        "observation.state": {"dtype": "float32", "shape": [d], "names": names},
        "action": {"dtype": "float32", "shape": [d], "names": names},
        "observation.effort": {"dtype": "float32", "shape": [d], "names": names},
    }
    for k in ("timestamp",):
        feats[k] = {"dtype": "float32", "shape": [1], "names": None}
    for k in ("frame_index", "episode_index", "index", "task_index"):
        feats[k] = {"dtype": "int64", "shape": [1], "names": None}
    ep_cols: dict = {"episode_index": np.arange(n_eps), "tasks": [["do the thing"]] * n_eps,
                     "length": [ep_len] * n_eps, "data/chunk_index": [0] * n_eps, "data/file_index": [0] * n_eps,
                     "dataset_from_index": np.arange(n_eps) * ep_len,
                     "dataset_to_index": (np.arange(n_eps) + 1) * ep_len}
    stats = {"observation.state": _stats(state), "action": _stats(action),
             "observation.effort": _stats(np.zeros((n, d)))}
    for cam in cameras:
        key = f"observation.images.{cam}"
        feats[key] = {"dtype": "video", "shape": [480, 640, 3], "names": ["height", "width", "channels"],
                      "info": {"video.fps": fps, "video.codec": "av1", "video.is_depth_map": False}}
        vdir = root / "videos" / key / "chunk-000"
        vdir.mkdir(parents=True)
        (vdir / "file-000.mp4").write_bytes(f"fake video {cam}".encode())
        ep_cols[f"videos/{key}/chunk_index"] = [0] * n_eps
        ep_cols[f"videos/{key}/file_index"] = [0] * n_eps
        ep_cols[f"videos/{key}/from_timestamp"] = np.arange(n_eps) * ep_len / fps
        ep_cols[f"videos/{key}/to_timestamp"] = (np.arange(n_eps) + 1) * ep_len / fps
        stats[key] = {"min": [[[0.0]]] * 3, "max": [[[1.0]]] * 3}
    for feat in ("observation.state", "action"):
        arr = state if feat == "observation.state" else action
        per_ep = [_stats(arr[ep_index == e]) for e in range(n_eps)]
        for stat in per_ep[0]:
            ep_cols[f"stats/{feat}/{stat}"] = [np.asarray(p[stat]) for p in per_ep]
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    pd.DataFrame(ep_cols).to_parquet(root / "meta/episodes/chunk-000/file-000.parquet", index=False)
    pd.DataFrame({"task_index": [0]}, index=pd.Index(["do the thing"], name="task")).to_parquet(
        root / "meta/tasks.parquet")
    info = {"codebase_version": "v3.0", "robot_type": "bi_so_follower", "total_episodes": n_eps,
            "total_frames": n, "total_tasks": 1, "fps": fps, "features": feats}
    (root / "meta/info.json").write_text(json.dumps(info))
    (root / "meta/stats.json").write_text(json.dumps(stats))
    return {"state": state, "action": action}


@pytest.fixture
def bi_dataset(tmp_path):
    root = tmp_path / "bi"
    arrays = make_v3_dataset(root, BI_NAMES, ["left_top", "left_wrist", "right_wrist", "right_front"])
    return root, arrays


@pytest.fixture
def xle_dataset(tmp_path):
    root = tmp_path / "xle"
    arrays = make_v3_dataset(root, XLE_NAMES, ["head", "left_wrist", "right_wrist"], scale=90.0)
    return root, arrays
