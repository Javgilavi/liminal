"""Single-joint health test: does one servo's output follow a small, slow command?

The STS3215 encoder reads the output shaft (UNVERIFIED for this exact servo, standard for the series), so a damaged
gear train shows up as the measured position not following the command, jumping, or load without motion. Only the
tested joint gets torque, capped at --torque-limit percent; every other joint stays limp. The arm should rest folded.

    python ws4/single_arm/joint_test.py --joint elbow_flex --amplitude -10 --cycles 2
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
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--robot-id", default="red_follower")
    p.add_argument("--joint", default="elbow_flex", choices=JOINTS)
    p.add_argument("--amplitude", type=float, default=-10.0, help="deg away from the start position (sign matters)")
    p.add_argument("--speed", type=float, default=0.5, help="deg per tick at 30 Hz")
    p.add_argument("--cycles", type=int, default=2)
    p.add_argument("--torque-limit", type=float, default=30.0, help="percent of max torque")
    p.add_argument("--abort-lag", type=float, default=15.0)
    args = p.parse_args()
    sys.path.insert(0, str(HERE))
    from show_joints import open_bus

    bus = open_bus(args.port, args.robot_id)
    j = args.joint
    rows = []
    try:
        for m in JOINTS:
            bus.write("Torque_Enable", m, 0, num_retry=3)
        start = float(bus.read("Present_Position", j, num_retry=3))
        status0 = bus.read("Status", j, normalize=False, num_retry=3)
        print(f"{j}: start {start:.1f} deg, status {status0}, temperature "
              f"{bus.read('Present_Temperature', j, normalize=False, num_retry=3)} C")  # fmt: skip
        bus.write("Torque_Limit", j, int(args.torque_limit * 10), num_retry=3)
        bus.write("Goal_Position", j, start, num_retry=3)  # also switches torque on, holding where it is
        bus.write("Torque_Enable", j, 1, num_retry=3)
        n = max(1, int(abs(args.amplitude) / args.speed))
        ramp = np.linspace(0.0, args.amplitude, n + 1)
        path = np.concatenate([np.concatenate([ramp, ramp[::-1]]) for _ in range(args.cycles)] + [np.zeros(30)])
        t0 = time.monotonic()
        for k, offset in enumerate(path):
            goal = start + offset
            bus.write("Goal_Position", j, goal, num_retry=3)
            time.sleep(max(0.0, t0 + (k + 1) / 30 - time.monotonic()))
            pos = float(bus.read("Present_Position", j, num_retry=3))
            load = int(bus.read("Present_Load", j, normalize=False, num_retry=3))
            status = int(bus.read("Status", j, normalize=False, num_retry=3))
            rows.append((time.monotonic() - t0, goal, pos, load, status))
            if status or abs(goal - pos) > args.abort_lag:
                print(f"ABORT at tick {k}: goal {goal:.1f} measured {pos:.1f} load {load} status {status}")
                break
    finally:
        bus.write("Torque_Enable", j, 0, num_retry=5)
        bus.port_handler.closePort()
    a = np.array(rows)
    err = a[:, 1] - a[:, 2]
    steps = np.abs(np.diff(a[:, 2]))
    print(f"ticks {len(a)}; lag median {np.median(np.abs(err)):.2f} max {np.abs(err).max():.2f} deg; "
          f"largest single-tick jump {steps.max():.2f} deg; load min/max {a[:, 3].min():.0f}/{a[:, 3].max():.0f}; "
          f"status bits seen {sorted(set(a[:, 4].astype(int)))}")  # fmt: skip
    print("goal / measured / load every 0.5 s:")
    for row in a[::15]:
        print(f"  t={row[0]:4.1f}s goal {row[1]:7.1f} measured {row[2]:7.1f} load {row[3]:5.0f}")
    out = HERE / "runs" / f"jointtest_{j}_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "rows": a.tolist()}))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
