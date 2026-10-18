"""One calibration per physical arm, used by recording, training and deployment alike.

LeRobot degrees put zero at the middle of each joint's calibrated range, so the calibration file IS the unit
definition: a different file for the same arm shifts every recorded angle. On connect, LeRobot compares the file
with the servos' EEPROM (range_min, range_max, homing_offset) and on a mismatch WRITES THE FILE INTO THE SERVOS
(haller_ws config.desktop-real.yaml, arm notes; lerobot FeetechMotorsBus.is_calibrated). A wrong file is
therefore not just a wrong label, it rewrites the arm. This tool never writes to a servo.

Names, three conventions for the same two arms:

    robot side (ws4, FLUX)   haller id (operator frame)   port                    calibration file
    left                     right                        /dev/haller_arm_right   haller_right.json
    right                    left                         /dev/haller_arm_left    haller_left.json

ws4/flux/run_xlerobot.py drives the torso as LeRobot `bi_so_follower` with --robot-id xlerobot, which loads
xlerobot_left.json / xlerobot_right.json (robot sides). `alias` writes those as byte copies of the haller files.

    python ws4/teleop/calib.py harvest            # read both arms' EEPROM (read only) -> haller_<side>.json
    python ws4/teleop/calib.py check              # files vs EEPROM, both arms, read only
    python ws4/teleop/calib.py alias              # xlerobot_<robot side>.json = copies of haller_<side>.json
    python ws4/teleop/calib.py status             # what is on disk, no hardware
    python ws4/teleop/calib.py snapshot DATASET   # copy the files in use into DATASET/meta/ws4_calibration/

harvest/check need the haller-hmi or LeRobot venv and the arms powered, with nothing else holding the ports
(stop the HMI backend first).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

CALIB_DIR = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower"
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
FIELDS = ("homing_offset", "range_min", "range_max")
# robot side -> (haller id, port, haller calibration id, deploy calibration id)
ARMS = {
    "left": ("right", "/dev/haller_arm_right", "haller_right", "xlerobot_left"),
    "right": ("left", "/dev/haller_arm_left", "haller_left", "xlerobot_right"),
}


def read_eeprom(port: str, bus_factory=None) -> dict:
    """Calibration as the servos hold it, in LeRobot's JSON layout. Reads only."""
    if bus_factory is None:
        from lerobot.motors import Motor, MotorNormMode
        from lerobot.motors.feetech import FeetechMotorsBus

        def bus_factory(p):
            motors = {j: Motor(i + 1, "sts3215", MotorNormMode.RANGE_0_100 if j == "gripper"
                               else MotorNormMode.DEGREES) for i, j in enumerate(JOINTS)}
            return FeetechMotorsBus(port=p, motors=motors)

    bus = bus_factory(port)
    bus.connect()
    try:
        cal = bus.read_calibration()
    finally:
        bus.disconnect(disable_torque=False) if _takes_kw(bus.disconnect) else bus.disconnect()
    return {j: {"id": int(c.id), "drive_mode": int(c.drive_mode), "homing_offset": int(c.homing_offset),
                "range_min": int(c.range_min), "range_max": int(c.range_max)} for j, c in cal.items()}


def _takes_kw(fn) -> bool:
    import inspect

    try:
        return "disable_torque" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def diff(a: dict, b: dict) -> list[str]:
    out = []
    for j in JOINTS:
        for f in FIELDS:
            if int(a[j][f]) != int(b[j][f]):
                out.append(f"{j}.{f}: {a[j][f]} != {b[j][f]}")
    return out


def span_report(cal: dict) -> str:
    return ", ".join(f"{j} {(cal[j]['range_max'] - cal[j]['range_min']) * 360 / 4095:.0f}deg" for j in JOINTS)


def write_new(path: Path, cal: dict) -> str:
    """Write `cal` to `path`; an existing different file is moved to a timestamped backup first."""
    if path.exists():
        old = json.loads(path.read_text())
        if not diff(old, cal):
            return "unchanged"
        backup = path.with_name(f"{path.name}.bak-{time.strftime('%Y%m%d-%H%M%S')}")
        shutil.move(path, backup)
        path.write_text(json.dumps(cal, indent=4))
        return f"replaced (old kept as {backup.name})"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cal, indent=4))
    return "written"


def alias(calib_dir: Path) -> list[str]:
    msgs = []
    for side, (_hid, _port, src_id, dst_id) in ARMS.items():
        src, dst = calib_dir / f"{src_id}.json", calib_dir / f"{dst_id}.json"
        if not src.exists():
            msgs.append(f"robot {side}: {src.name} missing, run harvest first")
            continue
        msgs.append(f"robot {side}: {dst.name} {write_new(dst, json.loads(src.read_text()))} from {src.name}")
    return msgs


def status(calib_dir: Path) -> list[str]:
    msgs = []
    for side, (hid, port, src_id, dst_id) in ARMS.items():
        files = {k: calib_dir / f"{k}.json" for k in (src_id, dst_id)}
        have = {k: json.loads(p.read_text()) for k, p in files.items() if p.exists()}
        line = f"robot {side} (haller {hid}, {port}): " + ", ".join(
            f"{k}.json {'present' if k in have else 'MISSING'}" for k in files)
        if len(have) == 2:
            d = diff(have[src_id], have[dst_id])
            line += "; identical" if not d else f"; DIFFER ({len(d)} fields)"
        if src_id in have:
            line += f"\n    spans: {span_report(have[src_id])}"
        msgs.append(line)
    return msgs


def snapshot(dataset: Path, calib_dir: Path) -> Path:
    out = dataset / "meta" / "ws4_calibration"
    out.mkdir(parents=True, exist_ok=True)
    for _side, (_hid, _port, src_id, dst_id) in ARMS.items():
        for cid in (src_id, dst_id):
            p = calib_dir / f"{cid}.json"
            if p.exists():
                shutil.copy2(p, out / p.name)
    (out / "README.txt").write_text(
        "LeRobot so_follower calibration files in use when this dataset was recorded (ws4/teleop/calib.py).\n"
        "robot left = haller right = haller_right.json = xlerobot_left.json\n"
        "robot right = haller left = haller_left.json = xlerobot_right.json\n")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("harvest", "check", "alias", "status", "snapshot"))
    ap.add_argument("dataset", nargs="?", type=Path)
    ap.add_argument("--calib-dir", type=Path, default=CALIB_DIR)
    ap.add_argument("--side", choices=("left", "right", "both"), default="both", help="robot side")
    args = ap.parse_args(argv)
    sides = list(ARMS) if args.side == "both" else [args.side]
    rc = 0
    if args.cmd in ("harvest", "check"):
        for side in sides:
            hid, port, src_id, _ = ARMS[side]
            cal = read_eeprom(port)
            print(f"robot {side} (haller {hid}, {port}) EEPROM: {span_report(cal)}")
            path = args.calib_dir / f"{src_id}.json"
            if args.cmd == "harvest":
                print(f"    {path.name}: {write_new(path, cal)}")
            elif not path.exists():
                print(f"    {path.name}: MISSING")
                rc = 1
            else:
                d = diff(json.loads(path.read_text()), cal)
                print(f"    {path.name}: {'matches EEPROM' if not d else 'DIFFERS: ' + '; '.join(d)}")
                rc |= bool(d)
    elif args.cmd == "alias":
        print("\n".join(alias(args.calib_dir)))
    elif args.cmd == "status":
        print("\n".join(status(args.calib_dir)))
    elif args.cmd == "snapshot":
        if args.dataset is None:
            ap.error("snapshot needs DATASET")
        print(f"copied to {snapshot(args.dataset, args.calib_dir)}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
