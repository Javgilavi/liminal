"""Bring handheld (UMI-style) and robot datasets into one end-effector format, then into EE50 chunks.

Common intermediate, `ee20`, one row per frame, absolute in the source's own fixed frame:
    [left: position (3, metres), rotation 6D = first two ROWS of R (6), gripper open fraction 0..1 (1) |
     right: same 10]
EE50 itself is relative to an anchor frame and z-scored per embodiment, so it is built per training window from
ee20 with ws4/kinematics/ee50.py (read-only reuse of the ws4-ik codec): `ee50.windows` + `ee50.EE50Stats.fit`.
With the codec's default rel_frame="anchor" (UNVERIFIED as BFL's choice) the source world frame cancels out:
only each tool frame's axis convention matters, which is the `tool_fix` rotation below.

Sources (presets):
  roco          rocochallenge2025/roco_iros2026_umi_dataset: [l xyz, l quat xyzw, r xyz, r quat xyzw, l grip, r grip]
  axibo         axiboai/umi_stack_sohaib_no_roll: [l xyz, l rot6d, l grip, r xyz, r rot6d, r grip]
                (rot6d convention not documented: columns assumed, UNVERIFIED)
  umi_zarr      original UMI replay buffer (github.com/real-stanford/universal_manipulation_interface):
                robot{i}_eef_pos, robot{i}_eef_rot_axis_angle, robot{i}_gripper_width, meta/episode_ends
                (key names from the UMI codebase, UNVERIFIED here: zarr is not installed on this laptop)
  so101_joints  any LeRobot bimanual SO-101 dataset in degrees: FK through ws4/kinematics Station

Usage:
    python -I ws4/data/ee_convert.py SRC --preset roco --out ws4/.cache/ee/roco.npz [--episodes 50]
    python -I ws4/data/ee_convert.py SRC --preset so101_joints --out ws4/.cache/ee/zonglin.npz --stats-out s.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))  # ws4/ so that `import kinematics` works (ws4-ik package, read-only)

ARMS = ("left", "right")
EE20 = 20

PRESETS: dict[str, dict] = {
    "roco": {"pos": {"left": slice(0, 3), "right": slice(7, 10)}, "rot": {"left": slice(3, 7), "right": slice(10, 14)},
             "rot_format": "quat_xyzw", "grip": {"left": 14, "right": 15}, "grip_scale": "minmax",
             "column": "observation.state"},
    "axibo": {"pos": {"left": slice(0, 3), "right": slice(10, 13)},
              "rot": {"left": slice(3, 9), "right": slice(13, 19)}, "rot_format": "rot6d_cols",
              "grip": {"left": 9, "right": 19}, "grip_scale": "minmax", "column": "observation.state"},
}


# ------------------------------------------------------------------------------------------- rotations


def quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def rot6d_to_matrix(x: np.ndarray, layout: str) -> np.ndarray:
    """6D -> R by Gram-Schmidt. layout 'rows' = first two rows (BFL EE50), 'cols' = first two columns (Zhou)."""
    from kinematics.ee50 import rot_from_6d

    R = rot_from_6d(x)  # rows convention
    return R if layout == "rows" else np.swapaxes(R, -1, -2)


def to_matrix(r: np.ndarray, fmt: str) -> np.ndarray:
    from kinematics.se3 import rotvec_to_matrix

    if fmt == "quat_xyzw":
        return quat_xyzw_to_matrix(r)
    if fmt == "rotvec":
        return rotvec_to_matrix(r)
    if fmt == "rot6d_rows":
        return rot6d_to_matrix(r, "rows")
    if fmt == "rot6d_cols":
        return rot6d_to_matrix(r, "cols")
    raise ValueError(fmt)


# ------------------------------------------------------------------------------------------- ee20


def poses_to_ee20(poses: dict[str, np.ndarray], grippers: dict[str, np.ndarray]) -> np.ndarray:
    """poses[arm] (T, 4, 4), grippers[arm] (T,) open fraction -> (T, 20). Missing arms stay zero."""
    from kinematics.ee50 import rot_to_6d

    T = len(next(iter(poses.values())))
    out = np.zeros((T, EE20))
    for i, arm in enumerate(ARMS):
        if arm not in poses:
            continue
        b = 10 * i
        out[:, b : b + 3] = poses[arm][:, :3, 3]
        out[:, b + 3 : b + 9] = rot_to_6d(poses[arm][:, :3, :3])
        out[:, b + 9] = grippers[arm]
    return out


def ee20_to_poses(x: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    from kinematics.se3 import make_pose

    poses, grips = {}, {}
    for i, arm in enumerate(ARMS):
        b = 10 * i
        poses[arm] = make_pose(rot6d_to_matrix(x[:, b + 3 : b + 9], "rows"), x[:, b : b + 3])
        grips[arm] = x[:, b + 9]
    return poses, grips


def apply_tool_fix(poses: dict[str, np.ndarray], tool_fix: np.ndarray | None) -> dict[str, np.ndarray]:
    """Right-multiply a constant tool-frame rotation so every embodiment uses one tool-axis convention."""
    if tool_fix is None:
        return poses
    F = np.eye(4)
    F[:3, :3] = tool_fix
    return {a: P @ F for a, P in poses.items()}


# ------------------------------------------------------------------------------------------- readers


def _load_columns(root: Path, columns: list[str], max_episodes: int | None):
    import pandas as pd

    files = sorted((root / "data").rglob("*.parquet"))
    frames = []
    for f in files:
        df = pd.read_parquet(f, columns=[*columns, "episode_index"])
        frames.append(df)
        if max_episodes and pd.concat(frames)["episode_index"].nunique() > max_episodes:
            break
    df = pd.concat(frames)
    if max_episodes:
        keep = sorted(df["episode_index"].unique())[:max_episodes]
        df = df[df["episode_index"].isin(keep)]
    return df


def _gripper_open(g: np.ndarray, how: str) -> np.ndarray:
    if how == "minmax":  # per-dataset 1st..99th percentile -> 0..1; which end is open is UNVERIFIED per dataset
        lo, hi = np.percentile(g, 1), np.percentile(g, 99)
        return np.clip((g - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    if how == "pct":
        return np.clip(g / 100.0, 0.0, 1.0)
    if how == "umi_width":  # UMI gripper width in metres, ~0.08 m fully open (UNVERIFIED)
        return np.clip(g / 0.08, 0.0, 1.0)
    raise ValueError(how)


def read_lerobot_ee(root: Path, preset: str, max_episodes: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """ee20 rows and episode index from a LeRobot dataset that stores EE poses."""
    p = PRESETS[preset]
    df = _load_columns(root, [p["column"]], max_episodes)
    X = np.stack(df[p["column"]].to_numpy()).astype(float)
    poses, grips = {}, {}
    for arm in ARMS:
        R = to_matrix(X[:, p["rot"][arm]], p["rot_format"])
        T = np.tile(np.eye(4), (len(X), 1, 1))
        T[:, :3, :3] = R
        T[:, :3, 3] = X[:, p["pos"][arm]]
        poses[arm] = T
        grips[arm] = _gripper_open(X[:, p["grip"][arm]], p["grip_scale"])
    return poses_to_ee20(poses, grips), df["episode_index"].to_numpy()


def read_umi_zarr(path: Path, arm_of_robot: dict[int, str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Original UMI replay buffer (dataset.zarr or .zarr.zip). Skeleton: key names UNVERIFIED, needs `zarr`."""
    import zarr  # optional dependency, not installed on the laptop

    arm_of_robot = arm_of_robot or {0: "left", 1: "right"}  # UNVERIFIED: depends on how the demos were recorded
    root = zarr.open(str(path), mode="r")
    data, ends = root["data"], np.asarray(root["meta"]["episode_ends"])
    n = int(ends[-1])
    poses, grips = {}, {}
    for i, arm in arm_of_robot.items():
        if f"robot{i}_eef_pos" not in data:
            continue
        T = np.tile(np.eye(4), (n, 1, 1))
        T[:, :3, :3] = to_matrix(np.asarray(data[f"robot{i}_eef_rot_axis_angle"]), "rotvec")
        T[:, :3, 3] = np.asarray(data[f"robot{i}_eef_pos"])
        poses[arm] = T
        grips[arm] = _gripper_open(np.asarray(data[f"robot{i}_gripper_width"])[:, 0], "umi_width")
    episode = np.searchsorted(ends, np.arange(n), side="right")
    return poses_to_ee20(poses, grips), episode


def read_so101_joints(root: Path, station=None, max_episodes: int | None = None, column: str = "action"):
    """Bimanual SO-101 joints (LeRobot DEGREES, mid-range zero) -> station-frame TCP poses through FK."""
    from contracts import ARM_JOINTS, canonical_joint
    from kinematics.so101_kinematics import Station, lerobot_deg_to_rad, lerobot_gripper_to_open

    station = station or Station.load()
    info = json.loads((root / "meta/info.json").read_text())
    names = info["features"][column]["names"]
    lookup = {canonical_joint(n): i for i, n in enumerate(names)}
    df = _load_columns(root, [column], max_episodes)
    X = np.stack(df[column].to_numpy()).astype(float)
    poses, grips = {}, {}
    for arm in ARMS:
        idx = [lookup[(arm, j)] for j in ARM_JOINTS]
        q = lerobot_deg_to_rad(X[:, idx[:5]])
        poses[arm] = station.fk(arm, q)
        grips[arm] = lerobot_gripper_to_open(X[:, idx[5]])
    return poses_to_ee20(poses, grips), df["episode_index"].to_numpy()


# ------------------------------------------------------------------------------------------- EE50


def fit_ee50_stats(ee20: np.ndarray, episode: np.ndarray, offsets: np.ndarray, stride: int = 4):
    """Per-embodiment EE50 z-score stats indexed by signed offset from the anchor (ws4-ik codec)."""
    from kinematics import ee50

    chunks = []
    for e in np.unique(episode):
        poses, grips = ee20_to_poses(ee20[episode == e])
        if len(grips["left"]) <= offsets.max() - offsets.min():
            continue
        w, _ = ee50.windows(poses, grips, offsets, stride=stride)
        chunks.append(w)
    return ee50.EE50Stats.fit(np.concatenate(chunks), offsets)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("--preset", choices=[*PRESETS, "umi_zarr", "so101_joints"], required=True)
    ap.add_argument("--out", type=Path, required=True, help="npz with ee20 (N, 20) and episode (N,)")
    ap.add_argument("--episodes", type=int, default=None, help="only the first N episodes")
    ap.add_argument("--stats-out", type=Path, default=None, help="also fit EE50 z-score stats")
    ap.add_argument("--offsets", default="-7:43", help="signed anchor offsets for the stats, python range lo:hi")
    args = ap.parse_args(argv)

    if args.preset in PRESETS:
        ee, ep = read_lerobot_ee(args.src, args.preset, args.episodes)
    elif args.preset == "umi_zarr":
        ee, ep = read_umi_zarr(args.src)
    else:
        ee, ep = read_so101_joints(args.src, max_episodes=args.episodes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, ee20=ee.astype(np.float32), episode=ep)
    summary = {"frames": len(ee), "episodes": int(len(np.unique(ep))),
               "pos_range_m": [float(ee[:, [0, 1, 2, 10, 11, 12]].min()), float(ee[:, [0, 1, 2, 10, 11, 12]].max())],
               "grip_mean": [float(ee[:, 9].mean()), float(ee[:, 19].mean())]}
    if args.stats_out:
        lo, hi = (int(v) for v in args.offsets.split(":"))
        stats = fit_ee50_stats(ee, ep, np.arange(lo, hi))
        stats.save(args.stats_out)
        summary["stats"] = str(args.stats_out)
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    _rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_rc)  # skip interpreter teardown: pyarrow deadlocked there in 2 of ~12 laptop runs
