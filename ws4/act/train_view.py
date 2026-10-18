"""Make a read-only view of a LeRobot v3 dataset whose normalization statistics come from the train episodes only.

``lerobot_train --dataset.episodes`` selects the frames but still normalizes with ``meta/stats.json`` of the whole
dataset, held-out takes included. The view copies ``meta/`` (a few MB), replaces the ``observation.state`` and
``action`` statistics with the aggregate of the train episodes' own per-episode statistics (``meta/episodes``), and
symlinks ``data/`` and ``videos/`` so nothing is duplicated or written next to the source. Images keep LeRobot's
ImageNet statistics (``dataset.use_imagenet_stats``, the default).

    python ws4/act/train_view.py --src /workspace/real_lora/pen_in_right_final --dst /workspace/act/data/pen84 \\
        --episodes $TRAIN_EPS
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
from pathlib import Path

import numpy as np

FEATURES = ("observation.state", "action")


def episode_stats(meta: Path, episodes: list[int]) -> list[dict[str, dict[str, np.ndarray]]]:
    import pandas as pd

    frames = [pd.read_parquet(f) for f in sorted(glob.glob(str(meta / "episodes" / "*" / "*.parquet")))]
    table = pd.concat(frames, ignore_index=True).set_index("episode_index")
    missing = sorted(set(episodes) - set(table.index))
    if missing:
        raise SystemExit(f"episodes not in {meta}: {missing}")
    out = []
    for ep in episodes:
        row = table.loc[ep]
        stats = {}
        for feat in FEATURES:
            prefix = f"stats/{feat}/"
            stats[feat] = {c.removeprefix(prefix): np.asarray(row[c], dtype=np.float64)
                           for c in table.columns if c.startswith(prefix)}  # fmt: skip
        out.append(stats)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--src", required=True)
    p.add_argument("--dst", required=True)
    p.add_argument("--episodes", type=int, nargs="+", required=True)
    args = p.parse_args()
    from lerobot.datasets.compute_stats import aggregate_stats

    src, dst = Path(args.src).resolve(), Path(args.dst)
    if dst.exists():
        raise SystemExit(f"{dst} exists; remove it first")
    (dst / "meta").parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src / "meta", dst / "meta")
    for sub in ("data", "videos"):
        (dst / sub).symlink_to(src / sub)
    stats = json.loads((src / "meta" / "stats.json").read_text())
    train = aggregate_stats(episode_stats(src / "meta", sorted(args.episodes)))
    for feat in FEATURES:
        stats[feat] = {k: np.asarray(v).tolist() for k, v in train[feat].items()}
    (dst / "meta" / "stats.json").write_text(json.dumps(stats, indent=2))
    (dst / "meta" / "train_episodes.json").write_text(json.dumps(sorted(args.episodes)))
    print(f"{dst}: {len(args.episodes)} train episodes; state mean {np.round(stats['observation.state']['mean'], 2)}"
          f", action std {np.round(stats['action']['std'], 2)}")  # fmt: skip


if __name__ == "__main__":
    main()
