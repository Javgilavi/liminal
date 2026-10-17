from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from liminal.core.interfaces import Arm
from liminal.orchestration.attempt import AttemptFactory, AttemptScope
from liminal.orchestration.skills import SkillRouter
from liminal.perception.world_state import WorldState
from liminal.safety.supervisor import SafetySupervisor


@dataclass
class ExecutionContext:
    """Everything a behavior-tree node may touch. Built once per run (see liminal.app)."""

    arms: dict[str, Arm]
    world: WorldState
    safety: SafetySupervisor
    skills: SkillRouter
    cameras: dict[str, Any] = field(default_factory=dict)
    # Named sets of arms a single skill may drive, e.g. {"both": ["left", "right"]} for a
    # bimanual (option B) policy. Order matters: it is the action-vector order.
    arm_groups: dict[str, list[str]] = field(default_factory=dict)
    # Present only for the strict, evidence-backed episode path. Trusted authored
    # task fixtures retain the existing symbolic backend behavior.
    verification: AttemptScope | None = None
    # Evidence-backed authored trees (V1): a fresh attempt per verified skill run, plus
    # an observation refresh so checks and the final goal see current evidence.
    attempts: AttemptFactory | None = None
    observe: Callable[[], Awaitable[None]] | None = None

    def members(self, arm: str) -> list[str]:
        """An arm name or a group name -> the physical arms it covers."""
        return list(self.arm_groups.get(arm, [arm]))

    @property
    def arm_names(self) -> list[str]:
        """Every name a task file may use in `arm:` (physical arms and groups)."""
        return [*self.arms, *self.arm_groups]
