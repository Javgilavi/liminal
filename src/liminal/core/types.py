"""Shared data types. Every module talks in these; change them only with team agreement."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# SO-101 joint order used everywhere in this repo (matches LeRobot's so101_follower).
SO101_JOINTS: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

JointPositions = list[float]


class Status(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"

    @property
    def ok(self) -> bool:
        return self is Status.SUCCESS


@dataclass
class Skill:
    """A bounded, verifiable unit of work for one arm.

    `success` is a world-state predicate that must become true for the skill to count
    as done. Execution backends may finish early when it does; the orchestrator always
    re-checks it independently (never trust the policy's own opinion of success).
    """

    arm: str
    name: str
    prompt: str | None = None
    target: str | None = None
    success: str | None = None
    timeout_s: float = 15.0
    verify_timeout_s: float = 2.0
    zones: tuple[str, ...] = ()
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"{self.arm}:{self.name}" + (f"({self.target})" if self.target else "")


@dataclass
class SkillResult:
    status: Status
    skill: Skill
    message: str = ""
    duration_s: float = 0.0


@dataclass
class Observation:
    """One synchronized snapshot for one arm.

    `images` maps a camera role ("scene", "wrist") to a frame (HxWx3 uint8 array).
    """

    arm: str
    state: JointPositions
    images: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.monotonic)
