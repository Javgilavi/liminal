"""LeRobot SO-101 calibrated joint units <-> MJCF joint radians, per joint.

What LeRobot does (lerobot 0.5.1, read from source):
- `so_follower` (registered as so100_follower / so101_follower) uses `MotorNormMode.DEGREES` for the five arm
  joints when `use_degrees=True`, the default since v0.4.4 (robots/so_follower/config_so_follower.py:42,
  so_follower.py:50-60). Older releases and XLeRobot's own plugin default to `RANGE_M100_100`
  (XLeRobot software/src/robots/xlerobot/config_xlerobot.py:66). The gripper always uses `RANGE_0_100`.
- DEGREES: value = (raw - mid) * 360 / 4095 with mid = (range_min + range_max) / 2 of the range recorded at
  calibration (motors/motors_bus.py `_normalize`). No clamping, drive_mode ignored.
- RANGE_M100_100: value = (clip(raw) - min) / (max - min) * 200 - 100, negated if drive_mode (SO follower:
  drive_mode is always 0). So 0 is the same mid-range point as DEGREES and +-100 are the recorded ends.
- RANGE_0_100 (gripper): value = (clip(raw) - min) / (max - min) * 100, 0 = range_min. SO-ARM100 documents
  0 = fully closed, 100 = fully open (Simulation/SO101/README.md "Gripper Note").
- One STS3215 tick is 360/4096 deg, so a DEGREES value is (4095/4096) of the physical angle from mid-range.

What the MJCF uses: so101_new_calib joint zero = middle of each joint range ("new calibration", SO-ARM100
Simulation/SO101/README.md), and LeRobot's own kinematics passes DEGREES straight through np.deg2rad into
that URDF (lerobot 0.5.1 model/kinematics.py:67). Hence the default per joint is sign +1, offset 0.
Both still need a hardware check (`verified: false` in station_geometry.yaml): jog one joint and compare.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .geometry import ARM_JOINTS, load_geometry, val

BODY_JOINTS = ARM_JOINTS[:5]
TICK_DEG = 360.0 / 4096.0
LEROBOT_DEG_PER_TICK = 360.0 / 4095.0
MODES = ("degrees", "range_m100_100", "radians")


@dataclass
class ArmUnits:
    """Converts one arm's 6-vector [pan, lift, elbow, wrist_flex, wrist_roll, gripper].

    mode: "degrees" (LeRobot >= 0.4.4 default, FLUX SO-101 checkpoint), "range_m100_100" (older LeRobot,
    XLeRobot plugin default) or "radians" (no conversion, for debugging).
    half_range_ticks: per body joint, (range_max - range_min) / 2 in encoder ticks. Only used by
    range_m100_100. Defaults to the MJCF joint range, i.e. it assumes calibration swept the full range.
    """

    mode: str = "degrees"
    sign: dict = field(default_factory=lambda: {j: 1.0 for j in BODY_JOINTS})
    offset_rad: dict = field(default_factory=lambda: {j: 0.0 for j in BODY_JOINTS})
    gripper_closed_rad: float = 0.0
    gripper_open_rad: float = 1.74533
    half_range_ticks: dict | None = None
    joint_range_rad: dict | None = None

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.joint_range_rad is None:
            self.joint_range_rad = {k: tuple(v) for k, v in val(load_geometry()["arms"]["joint_range_rad"]).items()}
        if self.half_range_ticks is None:
            self.half_range_ticks = {j: np.rad2deg((hi - lo) / 2) / TICK_DEG
                                     for j, (lo, hi) in self.joint_range_rad.items() if j in BODY_JOINTS}

    @classmethod
    def from_geometry(cls, mode: str = "degrees", calibration: dict | str | Path | None = None) -> ArmUnits:
        g = load_geometry()
        lu, gr = g["lerobot_units"], g["gripper"]
        u = cls(mode=mode, sign={k: float(v) for k, v in val(lu["sign"]).items()},
                offset_rad={k: float(v) for k, v in val(lu["offset_rad"]).items()},
                gripper_closed_rad=float(val(gr["closed_rad"])), gripper_open_rad=float(val(gr["open_rad"])))
        if calibration is not None:
            u.half_range_ticks = half_range_from_calibration(calibration)
        return u

    # -- body joints ------------------------------------------------------------------------------
    def _to_physical_deg(self, j: str, v: float) -> float:
        if self.mode == "degrees":
            return v / LEROBOT_DEG_PER_TICK * TICK_DEG  # LeRobot divides by 4095, the encoder has 4096 ticks
        if self.mode == "range_m100_100":
            return v / 100.0 * self.half_range_ticks[j] * TICK_DEG
        return float(np.rad2deg(v))

    def _from_physical_deg(self, j: str, d: float) -> float:
        if self.mode == "degrees":
            return d / TICK_DEG * LEROBOT_DEG_PER_TICK
        if self.mode == "range_m100_100":
            return d / TICK_DEG / self.half_range_ticks[j] * 100.0
        return float(np.deg2rad(d))

    def to_radians(self, values) -> np.ndarray:
        v = np.asarray(values, dtype=float)
        assert v.shape == (6,), v.shape
        q = np.empty(6)
        for i, j in enumerate(BODY_JOINTS):
            q[i] = self.sign[j] * np.deg2rad(self._to_physical_deg(j, v[i])) + self.offset_rad[j]
        q[5] = v[5] if self.mode == "radians" else self.gripper_to_radians(v[5])
        return q

    def from_radians(self, q) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        assert q.shape == (6,), q.shape
        v = np.empty(6)
        for i, j in enumerate(BODY_JOINTS):
            v[i] = self._from_physical_deg(j, np.rad2deg((q[i] - self.offset_rad[j]) / self.sign[j]))
        v[5] = q[5] if self.mode == "radians" else self.gripper_from_radians(q[5])
        return v

    # -- gripper ----------------------------------------------------------------------------------
    def gripper_to_radians(self, pct: float) -> float:
        return self.gripper_closed_rad + pct / 100.0 * (self.gripper_open_rad - self.gripper_closed_rad)

    def gripper_from_radians(self, q: float) -> float:
        return (q - self.gripper_closed_rad) / (self.gripper_open_rad - self.gripper_closed_rad) * 100.0


def half_range_from_calibration(calibration: dict | str | Path) -> dict:
    """Half range in ticks per body joint from a LeRobot calibration JSON ({motor: {range_min, range_max, ...}}).

    Accepts an arm file (keys `shoulder_pan`, ...) or an XLeRobot file (keys `left_arm_shoulder_pan`, ...),
    in which case pass the already-filtered per-arm dict.
    """
    if not isinstance(calibration, dict):
        calibration = json.loads(Path(calibration).read_text())
    out = {}
    for j in BODY_JOINTS:
        key = next(k for k in calibration if k == j or k.endswith("_" + j))
        c = calibration[key]
        out[j] = (c["range_max"] - c["range_min"]) / 2.0
    return out


class StationUnits:
    """Both arms: 12-vector [left 6 | right 6] in LeRobot units <-> MJCF radians."""

    def __init__(self, mode: str = "degrees", calibration: dict | None = None) -> None:
        calibration = calibration or {}
        self.arms = {s: ArmUnits.from_geometry(mode, calibration.get(s)) for s in ("left", "right")}
        self.mode = mode

    def to_radians(self, values12) -> np.ndarray:
        v = np.asarray(values12, dtype=float)
        return np.concatenate([self.arms["left"].to_radians(v[:6]), self.arms["right"].to_radians(v[6:])])

    def from_radians(self, q12) -> np.ndarray:
        q = np.asarray(q12, dtype=float)
        return np.concatenate([self.arms["left"].from_radians(q[:6]), self.arms["right"].from_radians(q[6:])])
