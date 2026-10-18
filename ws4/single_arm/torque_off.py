"""Switch one arm's torque off after a typed confirmation. SUPPORT THE ARM FIRST: it goes limp. Used to place the box
arm by hand (holding the lid as in the skill's takes) before the runner takes it over again with --hold-other-arm.
Writes Torque_Enable = 0 and nothing else; refuses a port another process holds.

    python ws4/single_arm/torque_off.py --port /dev/haller_arm_left --robot-id haller_left
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--port", required=True)
    p.add_argument("--robot-id", required=True, help="calibration id (haller_left = the box arm)")
    args = p.parse_args()
    held = subprocess.run(["fuser", os.path.realpath(args.port)], capture_output=True, text=True).stdout.split()
    if held:
        raise SystemExit(f"{args.port} is held by pid {' '.join(held)}: stop that process first")
    from show_joints import JOINTS, open_bus

    bus = open_bus(args.port, args.robot_id)
    try:
        q = bus.sync_read("Present_Position", list(JOINTS), num_retry=3)
        print(f"{args.port} pose: {[round(float(q[j]), 1) for j in JOINTS]}", flush=True)
        answer = input(">>> SUPPORT THE ARM, it goes limp. Type off + Enter to switch its torque off: ")
        if answer.strip().lower() != "off":
            print("nothing changed")
            return
        bus.disable_torque(num_retry=5)
        print(f"torque off: {bus.sync_read('Torque_Enable', normalize=False, num_retry=3)}", flush=True)
    finally:
        bus.port_handler.closePort()


if __name__ == "__main__":
    main()
