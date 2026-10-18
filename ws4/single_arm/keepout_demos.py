"""Replay recorded demos through the FK keep-out: every demonstrated motion must pass, or the zones are wrong for the
task (for example a box near the torso, or a pen grasped flat on the mat inside the table margin).

For each consecutive pair of recorded states, ``refuse(state[t+1], state[t])`` must be False. Reports refusals per
zone and episode, the lowest TCP height over the table, the closest approach to each station zone, and the commanded
actions as well (the follower may lag them). In a 12-D recording the other arm's recorded state is the parked arm's
pose for the other-arm zone, frame by frame (Haller: ``--arm right`` = haller right = robot left task arm).

    python ws4/single_arm/keepout_demos.py --root <per-arm dataset dir> --repo-id <name> \\
        --config ws4/single_arm/calibration/haller_right_workspace.json
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from keepout import ZONES, KeepOut  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--repo-id", required=True)
    p.add_argument("--root", default=None)
    p.add_argument("--config", required=True)
    p.add_argument("--arm", choices=("left", "right"), default=None, help="pick one arm of a 12-D dataset")
    p.add_argument("--exclude", type=int, nargs="*", default=[], help="episodes to skip (junk takes)")
    args = p.parse_args()
    from flux.package_tools import arm_columns, dataset_arrays

    ko = KeepOut.load(Path(args.config))
    arrays = dataset_arrays(args.repo_id, root=args.root)
    keep = ~np.isin(arrays["episode_index"], args.exclude)
    states, actions, ep = arrays["observation.state"][keep], arrays["action"][keep], arrays["episode_index"][keep]
    other = None
    if states.shape[1] == 12:
        other = arm_columns(states, "left" if args.arm == "right" else "right")  # the parked arm, per frame
        states, actions = arm_columns(states, args.arm), arm_columns(actions, args.arm)
    for name, traj in (("state", states), ("action", actions)):
        refusals: Counter = Counter()
        per_episode: Counter = Counter()
        for i in range(len(traj) - 1):
            if ep[i + 1] != ep[i]:
                continue
            if other is not None and ko.other_base_pos is not None:
                ko.other_q = tuple(other[i])
            refuse, zone = ko.refuse(traj[i + 1], states[i])
            if refuse:
                refusals[zone] += 1
                per_episode[int(ep[i])] += 1
        print(f"{name}: {sum(refusals.values())} refused steps of {len(traj) - 1} {dict(refusals)}; "
              f"episodes with refusals {dict(per_episode)}")  # fmt: skip
    ko.other_q = None  # per-frame other-arm poses differ; the station zones below are pose-independent
    depths = ko.depths(states)
    tcp = ko.tcp_station(states)
    print(f"lowest TCP z over all demos: {tcp[:, 2].min():.3f} m (table_z {ko.table_z}, margin {ko.table_margin})")
    print("max depth per zone over the measured demo states (0 = never entered):",
          {z: round(float(d), 3) for z, d in zip(ZONES, depths.max(axis=0), strict=True)})  # fmt: skip


if __name__ == "__main__":
    main()
