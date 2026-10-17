"""At the start pose, before the first attempt: do the arm and its cameras look as they did in the recording?

The runner's live frames (its exact camera path) against reference frames recorded with the arm at home (ws4-vr2:
held-out takes 101 to 103, frame 0, wrist already rotated 180, ``frames/reference_2026-10-17_night``), and the
measured joints against the recorded home state from the references' ``manifest.json``:

- head camera: brightness and the base-plate shift (``preflight.parity``: has the mast camera moved?);
- wrist camera: brightness and orientation on the gripper band;
- the task arm in the head view: its green pixels in the upper-left of the image, live vs reference (overlap and
  centroid shift). A different calibration puts "home" somewhere else, so the arm is drawn elsewhere;
- joints: measured at the commanded home vs the recorded home state (VR teleop held home within 3 deg).

Thresholds are UNVERIFIED first guesses: arm overlap >= 0.6 and centroid within 10 px, joints within 6 deg.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ARM_BOX = (0.0, 0.75, 0.0, 0.45)  # rows y0..y1, cols x0..x1 (fractions) of the head view holding the task arm at home


def green_mask(img: np.ndarray) -> np.ndarray:
    """The robot's green PLA inside ARM_BOX (the parked arm, the base plate and the box are outside it)."""
    import cv2

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (40, 80, 40), (90, 255, 255)) > 0
    h, w = mask.shape
    box = np.zeros_like(mask)
    box[int(ARM_BOX[0] * h):int(ARM_BOX[1] * h), int(ARM_BOX[2] * w):int(ARM_BOX[3] * w)] = True
    return mask & box


def arm_overlap(live: np.ndarray, ref: np.ndarray) -> tuple[float, float, float]:
    """(IoU, dx px, dy px) of the task arm's green pixels in the head view, live vs reference."""
    a, b = green_mask(live), green_mask(ref)
    union = (a | b).sum()
    if not union or not a.any() or not b.any():
        return 0.0, float("nan"), float("nan")
    ya, xa = np.nonzero(a)
    yb, xb = np.nonzero(b)
    return float((a & b).sum() / union), float(xa.mean() - xb.mean()), float(ya.mean() - yb.mean())


def load_refs(ref_dir: Path, take: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, str]:
    """Head and wrist reference images of one take (default: the last by name), and its recorded joint state."""
    from placement import read_rgb
    from preflight import pick_ref

    head_path, wrist_path = pick_ref(ref_dir, "head", take), pick_ref(ref_dir, "wrist", take)
    state = None
    manifest = ref_dir / "manifest.json"
    if manifest.exists():
        ep = head_path.name.split("_")[0].removeprefix("ep").lstrip("0")
        entry = json.loads(manifest.read_text()).get(ep, {})
        state = np.array(entry["state_f0_right"]) if "state_f0_right" in entry else None
    return read_rgb(head_path), read_rgb(wrist_path), state, head_path.name


def home_check(head: np.ndarray, wrist: np.ndarray, q: np.ndarray, ref_head: np.ndarray, ref_wrist: np.ndarray,
               ref_state: np.ndarray | None) -> tuple[bool, list[str], np.ndarray]:  # fmt: skip
    """(ok, report lines, image: head ref | live | arm masks over wrist ref | live | blank)."""
    import cv2
    from preflight import parity

    lines, ok = [], True
    for name, live, ref in (("head", head, ref_head), ("wrist", wrist, ref_wrist)):
        cam_ok, cam_lines = parity(live, ref, name)
        lines += [f"{name}: {line}" for line in cam_lines]
        ok &= cam_ok
    if head.shape != ref_head.shape:
        head = cv2.resize(head, (ref_head.shape[1], ref_head.shape[0]))
    iou, dx, dy = arm_overlap(head, ref_head)
    arm_ok = iou >= 0.6 and abs(dx) <= 10 and abs(dy) <= 10
    lines.append(f"task arm in the head view: overlap {iou:.2f}, centroid shift {dx:+.1f} px, {dy:+.1f} px "
                 f"-> {'as recorded' if arm_ok else 'NOT where the recording had home (calibration?)'}")  # fmt: skip
    ok &= arm_ok
    if ref_state is not None:
        err = np.asarray(q[:5], float) - ref_state[:5]
        joints_ok = bool(np.all(np.abs(err) <= 6.0))
        lines.append(f"joints at home {np.round(q[:5], 1)} vs recorded {np.round(ref_state[:5], 1)}: max error "
                     f"{np.abs(err).max():.1f} deg")  # fmt: skip
        ok &= joints_ok
    masks = np.zeros_like(head)
    masks[green_mask(ref_head)] = (255, 0, 255)
    masks[green_mask(head)] = (0, 255, 0)
    masks[green_mask(ref_head) & green_mask(head)] = (255, 255, 255)
    if wrist.shape != ref_wrist.shape:
        wrist = cv2.resize(wrist, (ref_wrist.shape[1], ref_wrist.shape[0]))
    img = np.concatenate([np.concatenate([ref_head, head, masks], axis=1),
                          np.concatenate([ref_wrist, wrist, np.zeros_like(wrist)], axis=1)], axis=0)  # fmt: skip
    return bool(ok), lines, img
