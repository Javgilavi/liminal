"""Print the follower's joints in LeRobot degrees (gripper 0..100) with torque OFF; optionally save camera frames.

Read-only on the motors (no register writes, torque never enabled). Move the arm by hand to check joint signs and
what the cameras see:

    python ws4/single_arm/show_joints.py --seconds 10
    python ws4/single_arm/show_joints.py --once --frames ws4/single_arm/frames/pose_name --wrist-cam /dev/video4
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
SHORT = ("pan", "lift", "elbow", "wflex", "roll", "grip")
CALIB_DIR = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower"


def open_bus(port: str, robot_id: str):
    from lerobot.motors import Motor, MotorCalibration, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    calib = json.loads((CALIB_DIR / f"{robot_id}.json").read_text())
    motors = {
        j: Motor(i + 1, "sts3215", MotorNormMode.RANGE_0_100 if j == "gripper" else MotorNormMode.DEGREES)
        for i, j in enumerate(JOINTS)
    }
    bus = FeetechMotorsBus(port, motors, calibration={j: MotorCalibration(**calib[j]) for j in JOINTS})
    bus.connect(handshake=False)
    torque = bus.sync_read("Torque_Enable", list(JOINTS), normalize=False)
    if any(torque.values()):
        print(f"note: torque is ON on {[j for j, t in torque.items() if t]}")
    return bus


def grab(spec: str, out: Path) -> None:
    import cv2

    cap = cv2.VideoCapture(int(spec) if spec.isdigit() else spec, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    for _ in range(15):  # let auto exposure settle
        ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"no frame from {spec}")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), frame)
    print(f"saved {out} {frame.shape[1]}x{frame.shape[0]}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--robot-id", default="red_follower")
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--once", action="store_true")
    p.add_argument("--frames", default="", help="path prefix; saves <prefix>_wrist.jpg / <prefix>_scene.jpg")
    p.add_argument("--wrist-cam", default="/dev/video4")
    p.add_argument("--scene-cam", default="")
    args = p.parse_args()
    bus = open_bus(args.port, args.robot_id)
    try:
        end = time.monotonic() + (0 if args.once else args.seconds)
        while True:
            q = bus.sync_read("Present_Position", list(JOINTS))
            print("  ".join(f"{n} {q[j]:7.1f}" for n, j in zip(SHORT, JOINTS, strict=True)), flush=True)
            if time.monotonic() >= end:
                break
            time.sleep(0.5)
    finally:
        bus.port_handler.closePort()  # plain close: disconnect() would write Torque_Enable
    if args.frames:
        grab(args.wrist_cam, Path(f"{args.frames}_wrist.jpg"))
        if args.scene_cam:
            grab(args.scene_cam, Path(f"{args.frames}_scene.jpg"))


if __name__ == "__main__":
    main()
