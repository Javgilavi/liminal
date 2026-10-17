"""Which port and which wrist camera belong to the arm Oscar is moving? Read-only on the motors.

Streams both Haller buses (positions only, no writes) and both wrist cameras for a few seconds and reports the
joint travel per port and the frame-to-frame motion per camera. Move ONE arm by hand (a limp one) while it runs.

    python ws4/single_arm/haller_identify.py --seconds 10
"""

from __future__ import annotations

import argparse
import time

import numpy as np

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
PORTS = {"/dev/haller_arm_left": "haller_left", "/dev/haller_arm_right": "haller_right"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--cams", nargs="+", default=["/dev/video4", "/dev/video6"])
    args = p.parse_args()
    import cv2
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    buses = {}
    for port in PORTS:
        motors = {m: Motor(i + 1, "sts3215", MotorNormMode.RANGE_M100_100) for i, m in enumerate(JOINTS)}
        bus = FeetechMotorsBus(port, motors)
        bus.connect(handshake=False)
        buses[port] = bus
    caps = {}
    for dev in args.cams:
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        caps[dev] = cap
    travel = {port: 0.0 for port in buses}
    motion = {dev: [] for dev in caps}
    last_q, last_f = {}, {}
    try:
        print(f"move ONE arm by hand now ({args.seconds:.0f} s) ...", flush=True)
        end = time.monotonic() + args.seconds
        while time.monotonic() < end:
            for port, bus in buses.items():
                raw = bus.sync_read("Present_Position", JOINTS, normalize=False, num_retry=2)
                q = np.array(list(raw.values()), float)
                if port in last_q:
                    travel[port] += np.abs(q - last_q[port]).sum() * 360 / 4095
                last_q[port] = q
            for dev, cap in caps.items():
                ok, f = cap.read()
                if ok:
                    g = cv2.cvtColor(cv2.resize(f, (160, 120)), cv2.COLOR_BGR2GRAY).astype(np.float32)
                    if dev in last_f:
                        motion[dev].append(float(np.abs(g - last_f[dev]).mean()))
                    last_f[dev] = g
    finally:
        for bus in buses.values():
            bus.port_handler.closePort()
        for cap in caps.values():
            cap.release()
    for port, deg in travel.items():
        print(f"{port} ({PORTS[port]}): joint travel {deg:7.1f} deg")
    for dev, m in motion.items():
        m = m or [float("nan")]
        print(f"{dev}: mean frame change {np.mean(m):5.2f}, p90 {np.percentile(m, 90):5.2f}")


if __name__ == "__main__":
    main()
