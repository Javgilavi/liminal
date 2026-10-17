"""Record scripted-fallback keyframes (WS5): move the arm by hand / leader arm,
press Enter at each keyframe, 'q' to finish. Appends to data/waypoints/<arm>.yaml."""

import argparse
from pathlib import Path

import yaml

from liminal.hardware.so101 import SO101Arm

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True)
ap.add_argument("--port", required=True)
ap.add_argument("--skill", required=True)
args = ap.parse_args()

arm = SO101Arm(args.arm, args.port)
arm.connect()
poses = []
try:
    while input(f"[{len(poses)}] Enter = capture, q = done: ").strip() != "q":
        poses.append([round(x, 2) for x in arm.read_state()])
        print(poses[-1])
finally:
    arm.disconnect()

path = Path("data/waypoints") / f"{args.arm}.yaml"
data = yaml.safe_load(path.read_text()) if path.exists() else {}
data[args.skill] = poses
path.write_text(yaml.safe_dump(data))
print(f"saved {len(poses)} keyframes for {args.arm}:{args.skill} -> {path}")
