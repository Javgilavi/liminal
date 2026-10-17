"""Read-only checks before each post-training eval, and whenever someone else has used the robot. In order:

1. ports: no process holds /dev/haller_arm_right or /dev/haller_arm_left (the HMI or a recorder would);
2. calibration: both arms' EEPROM vs the files the runner loads (``ws4/teleop/calib.py check``), and those files vs
   the copies in ``calibration/`` that the dataset was recorded and the LoRA trained with (sha256);
3. task arm health: servo IDs, supply voltage against limits, temperature (``probe_bus.py``);
4. camera parity: one live frame from the scene and the wrist camera through the runner's camera path (LeRobot
   OpenCVCamera, 640x480, wrist rotated 180) against reference frames: tonight's recorder references when given,
   otherwise a held-out take's start frame. Colour and brightness; wrist orientation on the gripper band (fixed in
   the wrist view); head camera shift on the robot's base plate (fixed to the mast camera).

Never writes to a servo and never moves the arm; the cameras are open for a few seconds. Exit 1 if a check fails.
Thresholds are UNVERIFIED first guesses (brightness within 20 %, base-plate shift within 12 px and 1.5 deg).

    python ws4/single_arm/preflight.py                                  # held-out start frames as reference
    python ws4/single_arm/preflight.py --ref-dir ws4/single_arm/frames/reference_2026-10-17_night
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WS4 = HERE.parent
sys.path.insert(0, str(HERE))

PORTS = ("/dev/haller_arm_right", "/dev/haller_arm_left")
CALIB_DIR = Path.home() / ".cache/huggingface/lerobot/calibration/robots/so_follower"
SCENE_CAM = "/dev/haller_cam_mast"
WRIST_CAM = "/dev/v4l/by-path/pci-0000:00:14.0-usb-0:7.2:1.0-video-index0"
GRIPPER_BAND = (0.7, 1.0)  # wrist view rows with the gripper body and finger


def check_ports() -> bool:
    ok = True
    for link in PORTS:
        target = os.path.realpath(link)
        held = subprocess.run(["fuser", target], capture_output=True, text=True).stdout.split()
        print(f"  {link} -> {target}: {'held by pid ' + ' '.join(held) if held else 'free'}")
        ok &= not held
    return ok


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def check_calibration() -> bool:
    ok = True
    for name in ("haller_right", "haller_left"):
        used, repo = CALIB_DIR / f"{name}.json", HERE / "calibration" / f"{name}.json"
        same = used.exists() and sha(used) == sha(repo)
        print(f"  {used.name}: runner file {sha(used) if used.exists() else 'MISSING'}, repo copy {sha(repo)}"
              f" -> {'same' if same else 'DIFFERENT'}")  # fmt: skip
        ok &= same
    rc = subprocess.run([sys.executable, str(WS4 / "teleop" / "calib.py"), "check"], capture_output=True, text=True)
    print("  " + (rc.stdout.strip() or rc.stderr.strip()[-800:]).replace("\n", "\n  "))
    return ok and rc.returncode == 0


def check_health(port: str) -> bool:
    rc = subprocess.run([sys.executable, str(HERE / "probe_bus.py"), "--port", port], capture_output=True, text=True)
    print("  " + (rc.stdout.strip() or rc.stderr.strip()[-800:]).replace("\n", "\n  "))
    return rc.returncode == 0


def band_mad(a: np.ndarray, b: np.ndarray, band: tuple[float, float]) -> float:
    h = a.shape[0]
    rows = slice(int(band[0] * h), int(band[1] * h))
    return float(np.abs(a[rows].astype(float) - b[rows]).mean())


def parity(live: np.ndarray, ref: np.ndarray, camera: str) -> tuple[bool, list[str]]:
    """Colour/brightness for both cameras; orientation (wrist) or base-plate shift (head)."""
    import cv2
    from placement import ref_shift

    if live.shape != ref.shape:
        live = cv2.resize(live, (ref.shape[1], ref.shape[0]))
    lines, ok = [], True
    ratio = live.mean() / max(ref.mean(), 1e-6)
    lines.append(f"brightness live/ref {ratio:.2f}, RGB means live {live.reshape(-1, 3).mean(0).round(0)} "
                 f"ref {ref.reshape(-1, 3).mean(0).round(0)}")  # fmt: skip
    ok &= 0.8 <= ratio <= 1.2
    if camera == "wrist":
        as_is, rotated = band_mad(live, ref, GRIPPER_BAND), band_mad(np.rot90(live, 2), ref, GRIPPER_BAND)
        lines.append(f"gripper band |live - ref| {as_is:.1f}, live rotated 180 {rotated:.1f} -> "
                     f"{'orientation matches' if as_is < rotated else 'ROTATED vs the recording'}")  # fmt: skip
        ok &= as_is < rotated
    else:
        try:
            dx, dy, rot = ref_shift(live, ref)
            lines.append(f"base-plate shift {dx:+.1f} px, {dy:+.1f} px, {rot:+.2f} deg")
            ok &= abs(dx) <= 12 and abs(dy) <= 12 and abs(rot) <= 1.5
        except cv2.error:
            lines.append("base-plate alignment FAILED (camera moved a lot, or a different view)")
            ok = False
    return bool(ok), lines


def pick_ref(ref_dir: Path, camera: str, take: int | None = None) -> Path:
    """``*head*``/``*scene*`` or ``*wrist*`` image in ``ref_dir``: the given take (``ep<NNN>`` in the name), else the
    last by name (tonight's set: take 103, the one nobody walks through)."""
    names = ("head", "scene") if camera == "head" else ("wrist",)
    hits = sorted(p for p in ref_dir.glob("*.*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")
                  and any(n in p.name.lower() for n in names)
                  and (take is None or f"ep{take:03d}" in p.name))  # fmt: skip
    if not hits:
        raise SystemExit(f"no {camera} reference image{f' of take {take}' if take else ''} in {ref_dir}")
    return hits[-1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ref-dir", default=None, help="reference frames (*head* or *scene*, *wrist*); default: held-out")
    p.add_argument("--ref-take", type=int, default=None, help="take number in --ref-dir (default: the last by name)")
    p.add_argument("--layout-ref", default=str(HERE / "layouts" / "A_ep020_head.jpg"),
                   help="held-out start frame used when --ref-dir is not given (its _wrist twin too)")
    p.add_argument("--scene-cam", default=SCENE_CAM)
    p.add_argument("--wrist-cam", default=WRIST_CAM)
    p.add_argument("--skip", nargs="*", default=[], choices=("ports", "calibration", "health", "cameras"))
    p.add_argument("--out", default=str(HERE / "frames" / "haller" / f"preflight_{time.strftime('%Y%m%d_%H%M%S')}.jpg"))
    args = p.parse_args()
    results: dict[str, bool] = {}
    print("1. ports")
    results["ports"] = check_ports()
    if not results["ports"]:
        print("STOP: a port is held; nothing else is opened. Stop the HMI or recorder first.")
        sys.exit(1)
    if "calibration" not in args.skip:
        print("2. calibration (EEPROM vs files, read only)")
        results["calibration"] = check_calibration()
    if "health" not in args.skip:
        print("3. task arm health (/dev/haller_arm_right)")
        results["health"] = check_health(PORTS[0])
    if "cameras" not in args.skip:
        import cv2
        from compare_frames import live_frame
        from placement import read_rgb

        print("4. camera parity")
        if args.ref_dir:
            refs = {cam: pick_ref(Path(args.ref_dir), cam, args.ref_take) for cam in ("head", "wrist")}
        else:
            head = Path(args.layout_ref)
            refs = {"head": head, "wrist": head.with_name(head.name.replace("_head", "_wrist"))}
        tiles = []
        for cam, device, rotate in (("head", args.scene_cam, 0), ("wrist", args.wrist_cam, 180)):
            ref, live = read_rgb(refs[cam]), live_frame(device, None, rotate)
            ok, lines = parity(live, ref, cam)
            print(f"  {cam} ({device}) vs {refs[cam].name}: {'ok' if ok else 'CHECK'}")
            for line in lines:
                print(f"    {line}")
            results[f"camera {cam}"] = ok
            tiles.append(np.concatenate([ref, cv2.resize(live, (ref.shape[1], ref.shape[0]))], axis=1))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.out, cv2.cvtColor(np.concatenate(tiles, axis=0), cv2.COLOR_RGB2BGR))
        print(f"  saved {args.out} (rows head, wrist; reference | live)")
    failed = [k for k, v in results.items() if not v]
    print("PREFLIGHT " + ("PASS" if not failed else "FAIL: " + ", ".join(failed)))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
