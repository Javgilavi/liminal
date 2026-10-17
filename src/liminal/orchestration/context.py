from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from liminal.core.interfaces import Arm
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
