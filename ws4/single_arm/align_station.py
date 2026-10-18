"""Live station alignment against a recording's home view: cameras only, no arm port, no motion.

FLUX's red-pen LoRA only starts its reach when the wrist camera at home sees the room as it did in the training takes
(ws4 single_arm handover, 18 Oct). This page helps put the station (robot and table) back on that spot: the head and
wrist views next to the reference take's frame 0 and a 50/50 blend, refreshed about once a second, with the offset of
the room as seen by the wrist camera (ORB features above the gripper, RANSAC similarity: centre offset in px, scale,
inliers) and a hint which way to move. The task arm must be at home, as in the reference.

    python ws4/single_arm/align_station.py          # http://127.0.0.1:8899, Ctrl-C to stop

Good enough (UNVERIFIED thresholds, the FLUX camera check decides): wrist centre offset under 15 px both ways, scale
0.95..1.05, at least 40 inliers, wrist brightness 0.85..1.2 of the reference.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

REF_DIR = HERE / "frames" / "reference_2026-10-17_night"
WRIST_CAM = "/dev/v4l/by-path/pci-0000:00:14.0-usb-0:7.2:1.0-video-index0"
ROOM_ROWS = (0, 330)  # wrist rows above the gripper: the room only
GOOD = {"offset_px": 15.0, "scale": (0.95, 1.05), "inliers": 40, "brightness": (0.85, 1.2)}


def view_offset(ref: np.ndarray, live: np.ndarray, rows: tuple[int, int] = ROOM_ROWS) -> dict | None:
    """Similarity from ref to live on the given rows: centre offset (live - ref, px), scale, angle, inliers.
    None when too few features match (the views barely overlap)."""
    import cv2

    mask = np.zeros(ref.shape[:2], np.uint8)
    mask[rows[0] : rows[1]] = 255
    orb = cv2.ORB_create(3000)
    k1, d1 = orb.detectAndCompute(cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY), mask)
    k2, d2 = orb.detectAndCompute(cv2.cvtColor(live, cv2.COLOR_RGB2GRAY), mask)
    if d1 is None or d2 is None:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1, d2, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 8:
        return None
    src = np.float32([k1[m.queryIdx].pt for m in good])
    dst = np.float32([k2[m.trainIdx].pt for m in good])
    warp, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0)
    if warp is None or inl is None or int(inl.sum()) < 8:
        return None
    centre = np.array([ref.shape[1] / 2, (rows[0] + rows[1]) / 2, 1.0])
    moved = warp @ centre
    return {"dx": float(moved[0] - centre[0]), "dy": float(moved[1] - centre[1]),
            "scale": float(np.hypot(warp[0, 0], warp[1, 0])),
            "angle": float(np.degrees(np.arctan2(warp[1, 0], warp[0, 0]))), "inliers": int(inl.sum())}


def hint(off: dict | None, brightness: float) -> tuple[str, bool]:
    """Plain-language move for the station, and whether everything is within GOOD."""
    if off is None:
        return "wrist view does not overlap the recording: turn the station toward the long counter", False
    moves = []
    if off["dx"] < -GOOD["offset_px"]:
        moves.append("turn LEFT")  # the room sits left of where it was: the camera points too far right
    elif off["dx"] > GOOD["offset_px"]:
        moves.append("turn RIGHT")
    if off["scale"] < GOOD["scale"][0]:
        moves.append("move FORWARD (room looks smaller)")
    elif off["scale"] > GOOD["scale"][1]:
        moves.append("move BACK (room looks larger)")
    if abs(off["dy"]) > GOOD["offset_px"]:
        moves.append("room sits " + ("higher" if off["dy"] < 0 else "lower") + " than recorded")
    lo, hi = GOOD["brightness"]
    if brightness < lo:
        moves.append("more light (wrist view darker than recorded)")
    elif brightness > hi:
        moves.append("less light")
    ok = not moves and off["inliers"] >= GOOD["inliers"]
    return ("GOOD: within the alignment thresholds" if ok else ", ".join(moves) or "few inliers: hold still"), ok


def panel(ref_head, head, ref_wrist, wrist, lines: list[str]) -> np.ndarray:
    import cv2

    def small(img, text):
        out = cv2.resize(img, (320, 240))
        cv2.putText(out, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
        return out

    def blend(a, b):
        return (0.5 * a.astype(np.float32) + 0.5 * b.astype(np.float32)).astype(np.uint8)

    rows = [np.hstack([small(ref_head, "recording head"), small(head, "live head"),
                       small(blend(ref_head, head), "blend")]),
            np.hstack([small(ref_wrist, "recording wrist"), small(wrist, "live wrist"),
                       small(blend(ref_wrist, wrist), "blend")])]  # fmt: skip
    text = np.zeros((30 + 26 * len(lines), 960, 3), np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(text, line, (10, 26 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1, cv2.LINE_AA)
    return np.ascontiguousarray(np.vstack(rows + [text]))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ref-dir", type=Path, default=REF_DIR)
    p.add_argument("--take", default="103", help="reference take (frame 0) in --ref-dir")
    p.add_argument("--scene-cam", default="/dev/haller_cam_mast")
    p.add_argument("--wrist-cam", default=WRIST_CAM)
    p.add_argument("--port", type=int, default=8899)
    p.add_argument("--period", type=float, default=1.0)
    args = p.parse_args()
    busy = subprocess.run(["fuser", os.path.realpath(args.scene_cam), os.path.realpath(args.wrist_cam)],
                          capture_output=True, text=True).stdout.split()  # fmt: skip
    if busy:
        raise SystemExit(f"cameras held by pid {' '.join(busy)}: stop that process first")
    from lerobot.cameras.configs import Cv2Rotation
    from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from placement import PlacementView, read_rgb

    ref_head = read_rgb(args.ref_dir / f"ep{args.take}_f0000_head.png")
    ref_wrist = read_rgb(args.ref_dir / f"ep{args.take}_f0000_task_wrist_rot180.png")

    def cam(path: str, rotation) -> OpenCVCamera:  # the runner's camera config (run_single_arm.connect_so101)
        return OpenCVCamera(OpenCVCameraConfig(index_or_path=Path(path), fps=30, width=640, height=480,
                                               rotation=rotation, fourcc=None, warmup_s=3))  # fmt: skip

    cams = {"head": cam(args.scene_cam, Cv2Rotation.NO_ROTATION), "wrist": cam(args.wrist_cam, Cv2Rotation.ROTATE_180)}
    for c in cams.values():
        c.connect()
    view = PlacementView(args.port)
    print(f"alignment page: {view.url} (reference take {args.take}); task arm at home; Ctrl-C to stop", flush=True)
    try:
        while True:
            head, wrist = (cams[k].read_latest() for k in ("head", "wrist"))
            off = view_offset(ref_wrist, wrist)
            bright = float(wrist.mean() / max(ref_wrist.mean(), 1.0))
            advice, ok = hint(off, bright)
            nums = ("wrist: no overlap" if off is None else
                    f"wrist: offset {off['dx']:+.0f} px, {off['dy']:+.0f} px, scale {off['scale']:.2f}, "
                    f"angle {off['angle']:+.1f} deg, {off['inliers']} inliers")  # fmt: skip
            lines = [nums + f", brightness {bright:.2f}", ("OK  " if ok else "->  ") + advice,
                     f"good enough: |offset| < {GOOD['offset_px']:.0f} px, scale {GOOD['scale'][0]}.."
                     f"{GOOD['scale'][1]}, >= {GOOD['inliers']} inliers, brightness {GOOD['brightness'][0]}.."
                     f"{GOOD['brightness'][1]}"]
            view.show(panel(ref_head, head, ref_wrist, wrist, lines))
            print(time.strftime("%H:%M:%S"), lines[0], "|", lines[1], flush=True)
            time.sleep(args.period)
    except KeyboardInterrupt:
        pass
    finally:
        view.close()
        for c in cams.values():
            c.disconnect()
        print("cameras released", flush=True)


if __name__ == "__main__":
    main()
