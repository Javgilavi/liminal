"""Command guards between the policy and the motors.

The FLUX 3 Action model card says the model outputs joint targets with no velocity, force or workspace bounds
(https://huggingface.co/black-forest-labs/flux-3-action-so101, "Out-of-scope use"). Every command the runner sends
passes ``SafetyGuard.filter``: non-finite values hold the previous command, then a per-tick step limit, then joint
limits. Units are the robot's units (LeRobot degrees mode: joints in degrees, gripper 0..100).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .contract import ARM_DIM, ARM_JOINTS, ARMS

STS3215_TICKS = 4095  # LeRobot DEGREES mode maps (raw - mid) * 360 / 4095 (lerobot motors_bus.py _normalize)

# Conservative defaults, not measured on our arms (UNVERIFIED): +-110 deg on the arm joints, gripper 0..100.
DEFAULT_LOWER = np.array([-110.0, -110.0, -110.0, -110.0, -170.0, 0.0])
DEFAULT_UPPER = np.array([110.0, 110.0, 110.0, 110.0, 170.0, 100.0])
# Per-tick step at 30 Hz (UNVERIFIED tuning values): 4 deg/tick = 120 deg/s on joints, 10 %/tick on the gripper.
DEFAULT_MAX_STEP = np.array([4.0, 4.0, 4.0, 4.0, 6.0, 10.0])


@dataclass
class SafetyConfig:
    lower: np.ndarray = field(default_factory=lambda: np.tile(DEFAULT_LOWER, len(ARMS)))
    upper: np.ndarray = field(default_factory=lambda: np.tile(DEFAULT_UPPER, len(ARMS)))
    max_step: np.ndarray = field(default_factory=lambda: np.tile(DEFAULT_MAX_STEP, len(ARMS)))

    def __post_init__(self) -> None:
        for name in ("lower", "upper", "max_step"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if value.shape != (ARM_DIM * len(ARMS),) or not np.isfinite(value).all():
                raise ValueError(f"{name} must hold {ARM_DIM * len(ARMS)} finite values")
            setattr(self, name, value)
        if (self.upper < self.lower).any() or (self.max_step <= 0).any():
            raise ValueError("need lower <= upper and positive max_step")

    def with_arm_limits(self, arm: str, lower, upper) -> SafetyConfig:
        i = ARMS.index(arm) * ARM_DIM
        lo, hi = self.lower.copy(), self.upper.copy()
        lo[i : i + ARM_DIM], hi[i : i + ARM_DIM] = lower, upper
        return SafetyConfig(lo, hi, self.max_step.copy())


def limits_from_lerobot_calibration(path: str | Path, margin_deg: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """Joint limits in LeRobot degrees mode from a LeRobot calibration JSON (range_min / range_max ticks).

    Degrees mode centres each joint on the middle of its calibrated range, so the limits are
    +-(range_max - range_min) / 2 * 360 / 4095, shrunk by ``margin_deg``. The gripper stays 0..100.
    """
    calib = json.loads(Path(path).read_text())
    lower, upper = np.zeros(ARM_DIM), np.zeros(ARM_DIM)
    for i, joint in enumerate(ARM_JOINTS):
        if joint == "gripper":
            lower[i], upper[i] = 0.0, 100.0
            continue
        half = (calib[joint]["range_max"] - calib[joint]["range_min"]) / 2 * 360.0 / STS3215_TICKS
        lower[i], upper[i] = -half + margin_deg, half - margin_deg
    return lower, upper


@dataclass
class GuardReport:
    nonfinite: bool = False
    step_limited: np.ndarray | None = None
    clamped: np.ndarray | None = None

    @property
    def intervened(self) -> bool:
        return bool(
            self.nonfinite
            or (self.step_limited is not None and self.step_limited.any())
            or (self.clamped is not None and self.clamped.any())
        )


class SafetyGuard:
    """Stateless filter plus an e-stop latch; the caller keeps the previous command."""

    def __init__(self, config: SafetyConfig | None = None):
        self.config = config or SafetyConfig()
        self.estopped = False

    def estop(self) -> None:
        self.estopped = True

    def release(self) -> None:
        self.estopped = False

    def filter(self, command: np.ndarray, previous: np.ndarray) -> tuple[np.ndarray, GuardReport]:
        cfg = self.config
        command = np.asarray(command, dtype=np.float64)
        previous = np.asarray(previous, dtype=np.float64)
        report = GuardReport()
        if not np.isfinite(command).all():
            report.nonfinite = True
            command = np.where(np.isfinite(command), command, previous)
        step = np.clip(command - previous, -cfg.max_step, cfg.max_step)
        report.step_limited = np.abs(command - previous) > cfg.max_step
        limited = previous + step
        out = np.clip(limited, cfg.lower, cfg.upper)
        report.clamped = out != limited
        return out, report
