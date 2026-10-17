"""Convert a LeRobot v3.0 bimanual SO-10x / XLeRobot dataset without re-encoding any video.

Two targets:
  per_arm  one arm's 6 joints + (scene, that arm's wrist) cameras -> FLUX SO-101 contract (use 1, task LoRA on
           flux-3-action-so101). Run once per side; merge the outputs with lerobot's merge_datasets if wanted.
  xle12    both arms [left 6 | right 6] + (head, left_wrist, right_wrist) -> our 12-D contract (use 3); drops base
           velocities, head motors and extra cameras from XLeRobot datasets.

Usage:
    python -I ws4/data/convert_bimanual.py SRC DST --to per_arm --side left [--scene KEY --wrist KEY]
    python -I ws4/data/convert_bimanual.py SRC DST --to xle12 [--cam head=KEY --cam left_wrist=KEY ...]
    add --src-units m100 to convert range_m100_100 arm joints to degrees (default: auto from stats)

Only parquet columns and metadata are rewritten. Video files are hard-linked (or copied) under the new camera
keys, so the source must already be at the target fps (30). v2.x sources: convert to v3.0 with lerobot's
convert_dataset_v21_to_v30 first.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from contracts import ARM_JOINTS, FLUX_SO101, XLE_12D, canonical_joint, guess_units, m100_to_degrees  # noqa: E402
from inspect_dataset import describe, diff_against, load_meta  # noqa: E402

VECTORS = ("action", "observation.state")
KEEP_SCALARS = ("timestamp", "frame_index", "episode_index", "index", "task_index")


def _vec_names(info: dict, key: str) -> list[str]:
    names = info["features"][key].get("names")
    if isinstance(names, dict):
        names = next(iter(names.values()))
    return list(names)


def build_index(names: list[str], sides: tuple[str | None, ...]) -> list[int]:
    """Indices of the 6 arm joints for each requested side, in contract order."""
    lookup = {canonical_joint(n): i for i, n in enumerate(names)}
    out = []
    for side in sides:
        for j in ARM_JOINTS:
            if (side, j) not in lookup:
                raise KeyError(f"no channel for {side} {j} in {names}")
            out.append(lookup[(side, j)])
    return out


def unit_scales(joints: list[str], src_units: str) -> list[float]:
    """Per-channel multiplier taking source units to degrees (arm) / range_0_100 (gripper)."""
    scales = []
    for j in joints:
        if j == "gripper" or src_units == "degrees":
            scales.append(1.0)
        elif src_units == "m100":
            scales.append(float(m100_to_degrees(1.0, j)))
        else:
            raise ValueError(f"unknown source units {src_units}")
    return scales


def detect_src_units(meta: dict) -> str:
    desc = describe(meta)
    units = {c.get("units") for c in desc["vectors"].get("action", []) if c["joint"] and c["joint"] != "gripper"}
    if "degrees" in units:
        return "degrees"
    if units == {"range_m100_100_or_degrees"}:
        raise SystemExit(
            "arm joints stay inside +-100: range_m100_100 or small-motion degrees cannot be told apart from "
            "metadata. Look at a few frames vs values, then pass --src-units m100 or --src-units degrees."
        )
    raise SystemExit(f"unsupported arm units {units}; pass --src-units explicitly")


def transform_stats(stats: dict, idx: list[int], scales: list[float]) -> dict:
    """Slice and rescale one feature's per-dim stats. Scales are positive so min/max/quantiles keep their order."""
    out = {}
    for k, v in stats.items():
        if not isinstance(v, list) or len(v) <= max(idx):
            out[k] = v
            continue
        sliced = [v[i] for i in idx]
        if k == "count":
            out[k] = sliced
        else:
            pairs = zip(sliced, scales, strict=True)
            out[k] = [x * s if not isinstance(x, list) else [y * s for y in x] for x, s in pairs]
    return out


def flip_stats(stats: dict, dims: list[int], full: float = 100.0) -> dict:
    """Stats of x' = full - x on `dims`: min/max and quantile pairs swap, mean mirrors, std and count stay."""
    out = {k: (list(v) if isinstance(v, list) else v) for k, v in stats.items()}
    for d in dims:
        if "min" in stats and "max" in stats:
            out["min"][d], out["max"][d] = full - stats["max"][d], full - stats["min"][d]
        if "mean" in stats:
            out["mean"][d] = full - stats["mean"][d]
        for k in stats:
            if k.startswith("q") and k[1:].isdigit() and f"q{100 - int(k[1:]):02d}" in stats:
                out[k][d] = full - stats[f"q{100 - int(k[1:]):02d}"][d]
    return out


def _link_or_copy(src: Path, dst: Path, link: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if link:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def remap(
    src: Path,
    dst: Path,
    sides: tuple[str | None, ...],
    camera_map: dict[str, str],
    src_units: str,
    robot_type: str,
    link: bool = True,
    target_fps: int = 30,
    flip_action_gripper: bool = False,
    task_suffix: str = "",
    negate: tuple[str, ...] = (),
) -> dict:
    """Write a new v3.0 dataset at `dst` with the selected arm channels and renamed cameras.

    camera_map: new feature key -> source feature key. flip_action_gripper maps the action gripper g -> 100 - g
    (for datasets whose leader gripper is recorded inverted). task_suffix is appended to every task string.
    negate: arm joints whose state and action become -x, for a model whose joint sign convention differs from the
    robot's (variant D of ws4/single_arm: shoulder_lift and wrist_roll). The runner has to apply the same map.
    """
    import numpy as np
    import pandas as pd

    meta = load_meta(src)
    info = meta["info"]
    if info.get("codebase_version") != "v3.0":
        raise SystemExit(f"{src}: codebase {info.get('codebase_version')}, convert to v3.0 first")
    if info.get("fps") != target_fps:
        raise SystemExit(f"{src}: fps {info.get('fps')} != {target_fps}; this converter does not re-encode video")
    if dst.exists():
        raise SystemExit(f"{dst} exists")

    joints = [j for _ in sides for j in ARM_JOINTS]
    scales = unit_scales(joints, src_units)
    idx = {v: build_index(_vec_names(info, v), sides) for v in VECTORS}
    scale_arr = np.asarray(scales, dtype=np.float32)
    new_names = [f"{s}_{j}.pos" if s else f"{j}.pos" for s in sides for j in ARM_JOINTS]
    cam_rename = {v: k for k, v in camera_map.items()}
    grip_dims = [i for i, j in enumerate(joints) if j == "gripper"] if flip_action_gripper else []
    neg_dims = [i for i, j in enumerate(joints) if j in negate]

    # data parquet files
    for f in sorted((src / "data").rglob("*.parquet")):
        df = pd.read_parquet(f)
        out = pd.DataFrame({c: df[c] for c in KEEP_SCALARS if c in df.columns})
        for v in VECTORS:
            arr = np.stack(df[v].to_numpy())[:, idx[v]].astype(np.float32) * scale_arr
            if v == "action" and grip_dims:
                arr[:, grip_dims] = 100.0 - arr[:, grip_dims]
            arr[:, neg_dims] *= -1.0
            out[v] = list(arr)
        out = out[[*VECTORS, *[c for c in KEEP_SCALARS if c in out.columns]]]
        target = dst / f.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(target, index=False)

    # videos: same files, new key directories
    for new_key, old_key in camera_map.items():
        vdir = src / "videos" / old_key
        for f in sorted(vdir.rglob("*.mp4")):
            _link_or_copy(f, dst / "videos" / new_key / f.relative_to(vdir), link)

    # episodes metadata
    for f in sorted((src / "meta" / "episodes").rglob("*.parquet")):
        ep = pd.read_parquet(f)
        cols = {}
        for c in ep.columns:
            if c.startswith("videos/"):
                key = c.split("/")[1]
                if key in cam_rename:
                    cols[c.replace(f"videos/{key}/", f"videos/{cam_rename[key]}/")] = ep[c]
            elif c.startswith("stats/"):
                _, feat, stat = c.split("/", 2)
                if feat in VECTORS:
                    vals = []
                    for cell in ep[c]:
                        arr = np.asarray(cell, dtype=np.float64)
                        if arr.shape[0] <= max(idx[feat]):  # per-feature scalars such as count
                            vals.append(arr)
                            continue
                        sub = arr[idx[feat]]
                        vals.append(sub if stat == "count" else sub * scale_arr)
                    cols[c] = vals
                elif feat in cam_rename:
                    cols[f"stats/{cam_rename[feat]}/{stat}"] = ep[c]
                elif feat in KEEP_SCALARS:
                    cols[c] = ep[c]
            elif c == "tasks" and task_suffix:
                cols[c] = [[t + task_suffix for t in ts] for ts in ep[c]]
            else:
                cols[c] = ep[c]
        mirrors = [(v, neg_dims, 0.0) for v in VECTORS if neg_dims]  # x -> -x; the gripper is never negated
        if grip_dims:
            mirrors.append(("action", grip_dims, 100.0))
        for feat, dims, full in mirrors:
            stat_cols = {c.split("/")[2]: c for c in cols if c.startswith(f"stats/{feat}/")}
            for row in range(len(ep)):
                cell = {k: list(np.asarray(cols[c][row], dtype=np.float64)) for k, c in stat_cols.items()}
                flipped = flip_stats(cell, dims, full)
                for k, c in stat_cols.items():
                    cols[c][row] = np.asarray(flipped[k])
        target = dst / f.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(cols).to_parquet(target, index=False)

    # tasks
    if (src / "meta" / "tasks.parquet").exists():
        tasks = pd.read_parquet(src / "meta" / "tasks.parquet")
        if task_suffix:
            tasks.index = [str(t) + task_suffix for t in tasks.index]
            tasks.index.name = "task"
        tasks.to_parquet(dst / "meta" / "tasks.parquet")

    # stats.json
    stats = meta["stats"]
    new_stats = {}
    for k, v in stats.items():
        if k in VECTORS:
            new_stats[k] = transform_stats(v, idx[k], scales)
            if k == "action" and grip_dims:
                new_stats[k] = flip_stats(new_stats[k], grip_dims)
            if neg_dims:
                new_stats[k] = flip_stats(new_stats[k], neg_dims, 0.0)  # x -> -x
        elif k in cam_rename:
            new_stats[cam_rename[k]] = v
        elif k in KEEP_SCALARS:
            new_stats[k] = v
    (dst / "meta" / "stats.json").write_text(json.dumps(new_stats, indent=4))

    # info.json
    feats = {}
    for k, v in info["features"].items():
        if k in VECTORS:
            feats[k] = {**v, "shape": [len(new_names)], "names": new_names}
        elif k in cam_rename:
            feats[cam_rename[k]] = v
        elif k in KEEP_SCALARS:
            feats[k] = v
    new_info = {**info, "features": feats, "robot_type": robot_type}
    new_info["ws4_conversion"] = {
        "source": str(src),
        "sides": list(sides),
        "camera_map": camera_map,
        "source_arm_units": src_units,
        "arm_units": "degrees",
        "gripper_units": "range_0_100",
        "flip_action_gripper": flip_action_gripper,
        "negated": list(negate),
        "task_suffix": task_suffix,
    }
    (dst / "meta" / "info.json").write_text(json.dumps(new_info, indent=4))
    return new_info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--to", choices=("per_arm", "xle12"), required=True)
    ap.add_argument("--side", choices=("left", "right"), help="per_arm: which arm")
    ap.add_argument("--scene", help="per_arm: source key of the scene/head camera")
    ap.add_argument("--wrist", help="per_arm: source key of that arm's wrist camera")
    ap.add_argument("--cam", action="append", default=[], help="xle12: head=KEY, left_wrist=KEY, right_wrist=KEY")
    ap.add_argument("--src-units", choices=("auto", "degrees", "m100"), default="auto")
    ap.add_argument("--robot-type", default=None,
                    help="robot_type written to info.json; must equal our own recordings' for merging. Defaults: "
                         "per_arm 'so_follower', xle12 'bi_so_follower' (LeRobot main class names, check own data)")
    ap.add_argument("--copy", action="store_true", help="copy video files instead of hard-linking")
    ap.add_argument("--flip-action-gripper", action="store_true", help="action gripper g -> 100 - g")
    ap.add_argument("--negate", nargs="+", default=[], choices=[j for j in ARM_JOINTS if j != "gripper"],
                    help="arm joints whose state and action become -x (e.g. variant D: shoulder_lift wrist_roll)")
    ap.add_argument("--task-suffix", default=None,
                    help="appended to task strings; per_arm default ' with the <side> arm', xle12 default none")
    args = ap.parse_args(argv)

    meta = load_meta(args.src)
    src_units = detect_src_units(meta) if args.src_units == "auto" else args.src_units
    desc = describe(meta)
    if args.to == "per_arm":
        if not args.side:
            ap.error("--side is required for per_arm")
        roles = {k: c["role"] for k, c in desc["cameras"].items() if not c["depth"]}
        scene = args.scene or next((k for k, r in roles.items() if r == "scene"), None)
        wrist = args.wrist or next((k for k, r in roles.items() if r == f"wrist:{args.side}"), None)
        if not scene or not wrist:
            ap.error(f"cannot pick cameras from {roles}; pass --scene/--wrist")
        cam_map = {"observation.images.scene": scene, "observation.images.wrist": wrist}
        suffix = f" with the {args.side} arm" if args.task_suffix is None else args.task_suffix
        info = remap(args.src, args.dst, (args.side,), cam_map, src_units, args.robot_type or "so_follower",
                     not args.copy,
                     flip_action_gripper=args.flip_action_gripper, task_suffix=suffix, negate=tuple(args.negate))
        # per-arm output uses bare joint names like the FLUX SO-101 contract
        feats = info["features"]
        for v in VECTORS:
            feats[v]["names"] = [f"{j}.pos" for j in ARM_JOINTS]
        (args.dst / "meta" / "info.json").write_text(json.dumps(info, indent=4))
        target = FLUX_SO101
    else:
        plan = diff_against(desc, XLE_12D)
        cam_map = {k: v for k, v in plan["camera_map"].items() if v}
        for kv in args.cam:
            k, v = kv.split("=", 1)
            cam_map[f"observation.images.{k}"] = v if v.startswith("observation.") else f"observation.images.{v}"
        missing = [k for k in XLE_12D.cameras if k not in cam_map]
        if missing:
            ap.error(f"no source camera for {missing}; pass --cam")
        remap(args.src, args.dst, ("left", "right"), cam_map, src_units, args.robot_type or "bi_so_follower",
              not args.copy,
              flip_action_gripper=args.flip_action_gripper, task_suffix=args.task_suffix or "",
              negate=tuple(args.negate))
        target = XLE_12D
    out = describe(load_meta(args.dst))
    print(json.dumps({"written": str(args.dst), "contract": target.name, "camera_map": cam_map,
                      "source_arm_units": src_units, "fps": out["fps"], "episodes": out["episodes"]}, indent=1))
    units = {c.get("units") for c in out["vectors"]["action"] if c["joint"] and c["joint"] != "gripper"}
    if units - {"degrees", "range_m100_100_or_degrees"}:
        print(f"warning: output arm units look like {units}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    _rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_rc)  # skip interpreter teardown: pyarrow deadlocked there in 2 of ~12 laptop runs


__all__ = ["build_index", "remap", "transform_stats", "unit_scales", "guess_units"]
