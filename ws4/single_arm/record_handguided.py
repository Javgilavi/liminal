"""Record hand-guided joint motion with torque OFF and turn it into A0r normalization statistics.

A0r keeps the released SO-101 weights and swaps only the history quantiles (q01/q99 of the state and of the
per-tick command deltas, gripper absolute) for ones measured on our own arm, because the package's quantiles are
one legacy-degree dataset's (``VoicAndrei/so100_kitchen``; ws4/flux/checkpoint_facts.md). Without a leader arm,
the follower's own measured motion stands in for teleop commands: command(t) = state(t), so deltas are state
deltas. UNVERIFIED how close that is to teleop deltas (leader commands lead the follower by a few ticks).

    python ws4/single_arm/record_handguided.py --seconds 60      # repeat 3 to 5 times, one episode each
    python ws4/single_arm/record_handguided.py --stats            # q01/q99 over every recorded episode

Read-only on the motors: torque is never enabled, nothing is written.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "handguided"
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def record(args: argparse.Namespace) -> Path:
    sys.path.insert(0, str(HERE))
    from show_joints import open_bus

    bus = open_bus(args.port, args.robot_id)
    rows, stamps = [], []
    period, start = 1 / 30, time.monotonic()
    next_tick = start
    try:
        print(f"recording {args.seconds:.0f} s at 30 Hz, torque off: move the arm by hand now", flush=True)
        while time.monotonic() - start < args.seconds:
            q = bus.sync_read("Present_Position", list(JOINTS), num_retry=3)
            rows.append([float(q[j]) for j in JOINTS])
            stamps.append(time.monotonic() - start)
            if len(rows) % 150 == 0:
                print(f"t={stamps[-1]:5.1f}s " + " ".join(f"{v:7.1f}" for v in rows[-1]), flush=True)
            next_tick += period
            time.sleep(max(0.0, next_tick - time.monotonic()))
    finally:
        bus.port_handler.closePort()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{datetime.now():%Y%m%d_%H%M%S}.npz"
    np.savez_compressed(path, t=np.array(stamps), q=np.array(rows), joints=np.array(JOINTS))
    q = np.array(rows)
    print(f"saved {path}: {len(q)} ticks, {len(q) / max(stamps[-1], 1e-9):.1f} Hz")
    print("range per joint:", " ".join(f"{j}={q[:, i].min():.0f}..{q[:, i].max():.0f}" for i, j in enumerate(JOINTS)))
    return path


def widen(stats: dict, min_state: float, min_action: float) -> list[str]:
    """Give every arm joint at least ``min_state`` deg of state range and ``min_action`` deg/tick of delta range,
    centred on what was measured. A joint the demos never moved (wrist roll here) otherwise gets q01 == q99, so
    any sensor noise maps to a huge normalized value and the model can never command it."""
    out = []
    for i, joint in enumerate(JOINTS[:-1]):  # the gripper stays absolute, as measured
        for key, width in (("state", min_state), ("action", min_action)):
            lo, hi = stats[key]["q01"][i], stats[key]["q99"][i]
            if hi - lo < width:
                mid = (lo + hi) / 2
                stats[key]["q01"][i], stats[key]["q99"][i] = mid - width / 2, mid + width / 2
                out.append(f"{joint}.{key}")
    return out


def stats(args: argparse.Namespace) -> Path:
    sys.path.insert(0, str(HERE.parent))
    from flux.package_tools import compute_quantiles, describe_stats

    files = sorted(OUT.glob("*.npz"))
    if not files:
        raise SystemExit(f"no recordings in {OUT}")
    qs, eps = [], []
    for i, f in enumerate(files):
        q = np.load(f)["q"]
        qs.append(q)
        eps.append(np.full(len(q), i))
    q, ep = np.concatenate(qs), np.concatenate(eps)
    out = compute_quantiles(commands=q, states=q, episode_index=ep, absolute_dims=(-1,))
    widened = widen(out, args.min_state_width, args.min_action_width)
    print(describe_stats(out, list(JOINTS)))
    if widened:
        print(f"widened (joints the demos barely moved, kept usable): {', '.join(widened)}")
    path = OUT / f"stats_{args.robot_id}.json"
    meta = {"source": [f.name for f in files], "ticks": int(len(q)), "robot_id": args.robot_id, "widened": widened,
            "min_state_width": args.min_state_width, "min_action_width": args.min_action_width,
            "note": "hand-guided, torque off; command = measured state; gripper absolute"}  # fmt: skip
    path.write_text(json.dumps({**out, "meta": meta}, indent=2))
    print(f"wrote {path} from {len(files)} episodes, {len(q)} ticks")
    return path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--robot-id", default="red_follower")
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--stats", action="store_true", help="compute q01/q99 from all recordings instead")
    p.add_argument("--min-state-width", type=float, default=20.0, help="deg, per arm joint")
    p.add_argument("--min-action-width", type=float, default=1.0, help="deg per tick, per arm joint")
    args = p.parse_args()
    stats(args) if args.stats else record(args)


if __name__ == "__main__":
    main()
