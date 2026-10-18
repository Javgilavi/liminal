from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from liminal.core.interfaces import Arm
from liminal.orchestration.skills import SkillRouter
from liminal.perception.inspector import SceneInspector
from liminal.perception.types import InspectionProfile
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
    inspector: SceneInspector | None = None
    inspection: InspectionProfile = field(default_factory=InspectionProfile)
    # Named sets of arms a single skill may drive, e.g. {"both": ["left", "right"]} for a
    # bimanual (option B) policy. Order matters: it is the action-vector order.
    arm_groups: dict[str, list[str]] = field(default_factory=dict)
    _invariant_counts: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def members(self, arm: str) -> list[str]:
        """An arm name or a group name -> the physical arms it covers."""
        return list(self.arm_groups.get(arm, [arm]))

    @property
    def arm_names(self) -> list[str]:
        """Every name a task file may use in `arm:` (physical arms and groups)."""
        return [*self.arms, *self.arm_groups]

    @property
    def active_invariants(self) -> tuple[str, ...]:
        """Conditions that an executing skill must include in scene inspection."""

        return tuple(sorted(self._invariant_counts))

    @contextmanager
    def watch_invariant(self, condition: str) -> Iterator[None]:
        """Reference-count an invariant across nested or concurrent BT nodes."""

        self._invariant_counts[condition] = self._invariant_counts.get(condition, 0) + 1
        try:
            yield
        finally:
            remaining = self._invariant_counts[condition] - 1
            if remaining:
                self._invariant_counts[condition] = remaining
            else:
                self._invariant_counts.pop(condition)
