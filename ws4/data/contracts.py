"""Data contracts that public datasets are compared against and converted to.

Sources (read 2026-10-17):
- FLUX SO-101: https://docs.bfl.ai/flux_3/flux3_action_so101 ("LeRobot v3 dataset at 30 Hz, with six absolute
  commanded actions, six measured-state values"; "degrees for the five arm joints and percentage points for the
  gripper"; cameras observation.images.scene (left) + observation.images.wrist (right), each resized to 256x256)
  and the model card https://huggingface.co/black-forest-labs/flux-3-action-so101.
- LeRobot units: lerobot 0.5.1 motors/motors_bus.py `_normalize` (DEGREES: (raw - mid) * 360 / 4095 with
  mid = (range_min + range_max) / 2; RANGE_M100_100: (raw - min) / (max - min) * 200 - 100; gripper is always
  RANGE_0_100, robots/so_follower/so_follower.py:59).
- XLeRobot 12-D: our own station (two SO-101 arms, head camera, two wrist cameras). The [left 6 | right 6] layout
  matches lerobot's bi_so_follower feature naming (robots/bi_so_follower/bi_so_follower.py:74).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

ARM_JOINTS: tuple[str, ...] = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
STS3215_RESOLUTION = 4096

# Joint spans (range_max - range_min, in encoder ticks) from Oscar's own SO-101 follower calibration
# (~/.cache/huggingface/lerobot/calibration/robots/so_follower/haller_follower.json). LeRobot calibration records
# the mechanical range, so other SO-101 builds are close but not identical: converting someone else's
# range_m100_100 data to degrees with these spans has a few percent scale error per joint (UNVERIFIED size).
DEFAULT_SPAN_TICKS: dict[str, int] = {
    "shoulder_pan": 3510 - 790,
    "shoulder_lift": 3264 - 889,
    "elbow_flex": 3025 - 813,
    "wrist_flex": 3212 - 874,
    "wrist_roll": 4095 - 0,
}


@dataclass(frozen=True)
class Contract:
    name: str
    fps: int
    joints: tuple[str, ...]  # canonical "<side>_<joint>" or "<joint>" names, in order
    cameras: dict[str, str]  # feature key -> role ("scene" or "wrist:<side>")
    arm_units: str = "degrees"
    gripper_units: str = "range_0_100"
    codebase_version: str = "v3.0"
    view_hw: tuple[int, int] | None = None  # size each view is resized to inside the model
    notes: str = ""
    sides: tuple[str, ...] = field(default=())


FLUX_SO101 = Contract(
    name="flux3_so101",
    fps=30,
    joints=ARM_JOINTS,
    cameras={"observation.images.scene": "scene", "observation.images.wrist": "wrist:self"},
    view_hw=(256, 256),
    notes="actions are absolute commanded joints in the dataset; the policy predicts deltas with absolute gripper",
    sides=("self",),
)

XLE_12D = Contract(
    name="xlerobot_12d",
    fps=30,
    joints=tuple(f"{s}_{j}" for s in ("left", "right") for j in ARM_JOINTS),
    cameras={
        "observation.images.head": "scene",
        "observation.images.left_wrist": "wrist:left",
        "observation.images.right_wrist": "wrist:right",
    },
    notes="[left 6 | right 6]; head camera is the RealSense on the mast (colour stream only)",
    sides=("left", "right"),
)

CONTRACTS = {c.name: c for c in (FLUX_SO101, XLE_12D)}

_SIDE_RE = re.compile(r"^(left|right|l|r)(?:_arm)?[_.]", re.I)
_SIDE_SUFFIX_RE = re.compile(r"[_.](left|right)$", re.I)


def canonical_joint(name: str) -> tuple[str | None, str | None]:
    """Map a dataset's motor name to (side, canonical joint), e.g. 'left_arm_wrist_roll.pos' -> ('left', 'wrist_roll').

    Returns (side, None) for non-arm channels (base velocity, head motors, effort) so callers can drop them.
    """
    n = name.strip().lower()
    for suffix in (".pos", "_pos", ".position"):
        if n.endswith(suffix):
            n = n[: -len(suffix)]
    side = None
    m = _SIDE_RE.match(n)
    if m:
        side = {"l": "left", "r": "right"}.get(m.group(1), m.group(1))
        n = n[m.end() :]
    else:
        m = _SIDE_SUFFIX_RE.search(n)
        if m:
            side = m.group(1)
            n = n[: m.start()]
    for prefix in ("main_", "follower_", "arm_"):
        if n.startswith(prefix):
            n = n[len(prefix) :]
    if n in ARM_JOINTS:
        return side, n
    if n in ("gripper_pos", "jaw", "gripper_width"):
        return side, "gripper"
    return side, None


_WRIST_WORDS = ("wrist", "hand", "gripper", "eef", "follower")
_SCENE_WORDS = ("head", "top", "up", "overhead", "front", "base", "main", "center", "global", "scene", "laptop",
                "phone", "webcam", "high", "cam_high", "image", "side", "exo", "third")


def camera_role(key: str) -> str:
    """Guess a camera's role from its feature key: 'scene', 'wrist:left', 'wrist:right', 'wrist:?' or 'unknown:<side>'.

    bi_so_follower prefixes every camera with the arm it is attached to (left_top, right_front), so a side prefix
    alone does not make a camera a wrist camera. Bare 'left'/'right' keys are usually wrist cameras in community
    bimanual datasets, but check frames before trusting that.
    """
    k = key.split("observation.images.")[-1].lower()
    tokens = re.split(r"[_.\-]", k)
    sides = [t for t in tokens if t in ("left", "right")]
    if any(w in k for w in _WRIST_WORDS):
        if sides:
            # left_left_wrist -> left; wrist_right -> right; left_follower_left -> left
            return f"wrist:{sides[-1]}"
        return "wrist:?"
    if any(t in _SCENE_WORDS for t in tokens) or any(w in k for w in ("head", "top", "overhead")):
        return "scene"
    if tokens in (["left"], ["right"]):
        return f"unknown:{tokens[0]}"
    return "unknown"


def guess_units(joint: str, lo: float, hi: float, mean: float | None = None) -> str:
    """Guess the unit of one joint channel from its dataset-wide min/max.

    'degrees' is certain when an arm joint exceeds +-100.5. Inside +-100 the data is either range_m100_100 or
    degrees with a small range of motion, which one channel alone cannot separate (see `arm_units`).
    'degrees_legacy' is the pre-2025 LeRobot SO-100 calibration (zero at a folded pose, values up to ~190).
    """
    if hi - lo < 1e-3:
        return "constant"
    if joint == "gripper":
        if lo >= -5 and hi <= 105:
            return "range_0_100"
        if hi <= 1.5 and lo >= -0.5:
            return "fraction_0_1"
        return "unknown"
    half = DEFAULT_SPAN_TICKS.get(joint, STS3215_RESOLUTION) / 2 * 360 / (STS3215_RESOLUTION - 1)
    if max(abs(lo), abs(hi)) > half + 35:  # beyond what a midpoint-zero calibration of this joint can produce
        return "degrees_legacy"
    if max(abs(lo), abs(hi)) > 100.5:
        return "degrees"
    if max(abs(lo), abs(hi)) <= 3.5:
        return "radians?"
    return "range_m100_100_or_degrees"


def arm_units(channels: list[tuple[str, float, float]]) -> str:
    """Decide one arm's unit from all its (joint, min, max) channels; a recorder uses one setting per arm.

    Any channel beyond +-100.5 proves degrees. Two or more channels that stop within 0.6 of +-100 are the
    saturation signature of range_m100_100 (calibrated ends map to exactly +-100).
    """
    labels = [guess_units(j, lo, hi) for j, lo, hi in channels if j != "gripper"]
    if "degrees_legacy" in labels:
        return "degrees_legacy"
    if "degrees" in labels:
        return "degrees"
    moving = [(lo, hi) for j, lo, hi in channels if j != "gripper" and hi - lo >= 1e-3]
    if not moving:
        return "constant"
    saturated = sum(1 for lo, hi in moving for v in (lo, hi) if abs(abs(v) - 100.0) < 0.6)
    if saturated >= 2:
        return "range_m100_100"
    return "ambiguous"


def m100_to_degrees(value, joint: str, span_ticks: dict[str, int] | None = None):
    """Convert a RANGE_M100_100 joint value to DEGREES. Both share the zero at the calibration midpoint."""
    span = (span_ticks or DEFAULT_SPAN_TICKS)[joint]
    return value * (span / 200.0) * (360.0 / (STS3215_RESOLUTION - 1))


def degrees_to_m100(value, joint: str, span_ticks: dict[str, int] | None = None):
    span = (span_ticks or DEFAULT_SPAN_TICKS)[joint]
    return value / ((span / 200.0) * (360.0 / (STS3215_RESOLUTION - 1)))
