"""A0r statistics from a recorded LeRobot dataset: q01/q99 of the follower state and of the per-tick command deltas
(gripper absolute), the same rule as BFL's export (``flux.package_tools.compute_quantiles``).

Input: a per-arm FLUX SO-101 dataset (6-D state / action, e.g. ``ws4/teleop/to_flux.py`` output) or a 12-D
bimanual one with ``--arm left|right`` (robot sides). Held-out episodes can be excluded with ``--exclude``.

    python ws4/single_arm/a0r_stats.py --root <dataset dir> --repo-id <name> \\
        --out ws4/single_arm/calibration/a0r_haller_left.json

The JSON's ``state`` / ``action`` keys go to ``flux.package_tools.renormalize_package`` on the pod.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--repo-id", required=True)
    p.add_argument("--root", default=None, help="local dataset directory")
    p.add_argument("--arm", choices=("left", "right"), default=None, help="pick one arm of a 12-D dataset")
    p.add_argument("--exclude", type=int, nargs="*", default=[], help="episode indices to leave out (held-out)")
    p.add_argument("--min-state-width", type=float, default=20.0)
    p.add_argument("--min-action-width", type=float, default=1.0)
    p.add_argument("--scale", type=float, nargs=6, default=None,
                   help="model = scale * robot before computing the quantiles (variant D: 1 -1 1 1 -1 1)")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    from flux.package_tools import arm_columns, compute_quantiles, dataset_arrays, describe_stats
    from record_handguided import widen

    arrays = dataset_arrays(args.repo_id, root=args.root)
    ep = arrays["episode_index"]
    keep = ~np.isin(ep, args.exclude)
    states, actions = arrays["observation.state"][keep], arrays["action"][keep]
    if states.shape[1] == 12:
        if args.arm is None:
            raise SystemExit("12-D dataset: pass --arm left|right (robot side)")
        states, actions = arm_columns(states, args.arm), arm_columns(actions, args.arm)
    if states.shape[1] != 6:
        raise SystemExit(f"expected 6-D state/action, got {states.shape}")
    if args.scale:
        scale = np.asarray(args.scale, dtype=float)
        states, actions = states * scale, actions * scale
    stats = compute_quantiles(commands=actions, states=states, episode_index=ep[keep], absolute_dims=(-1,))
    widened = widen(stats, args.min_state_width, args.min_action_width)
    print(describe_stats(stats, list(JOINTS)))
    if widened:
        print("widened:", ", ".join(widened))
    meta = {"repo_id": args.repo_id, "root": args.root, "arm": args.arm, "excluded": args.exclude, "scale": args.scale,
            "episodes": int(len(np.unique(ep[keep]))), "frames": int(keep.sum()), "widened": widened}  # fmt: skip
    Path(args.out).write_text(json.dumps({**stats, "meta": meta}, indent=2))
    print(f"wrote {args.out}: {meta['episodes']} episodes, {meta['frames']} frames")


if __name__ == "__main__":
    main()
