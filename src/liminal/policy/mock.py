"""Symbolic mock backend: no motion, just world-state effects. Lets WS3 develop and
test whole task trees (including failures) with zero hardware."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from liminal.core.types import Skill, SkillResult, Status

if TYPE_CHECKING:
    from liminal.orchestration.context import ExecutionContext


@dataclass
class Fault:
    """Inject a failure: on the `on_attempt`-th run of `skill`, either fail outright
    or, mid-skill, set `facts` (e.g. {"lid_open": False} to simulate a slipping lid)."""

    skill: str
    on_attempt: int = 1
    fail: bool = False
    facts: dict[str, Any] = field(default_factory=dict)


class MockBackend:
    def __init__(self, duration_s: float = 0.05, effects: dict[str, dict[str, Any]] | None = None,
                 faults: list[Fault] | None = None):
        self.duration_s = duration_s
        self.effects = effects or {}  # skill name -> extra facts set on success
        self.faults = faults or []
        self.calls: list[str] = []

    async def execute(self, skill: Skill, ctx: ExecutionContext) -> SkillResult:
        self.calls.append(skill.label)
        attempt = sum(1 for c in self.calls if c == skill.label)
        for f in self.faults:
            if f.skill == skill.name and f.on_attempt == attempt:
                if f.facts:
                    await asyncio.sleep(self.duration_s / 2)
                    ctx.world.update(f.facts, source="mock_fault")
                if f.fail:
                    return SkillResult(Status.FAILURE, skill, "injected failure")
        await asyncio.sleep(self.duration_s)
        if skill.success:
            ctx.world.set(skill.success.lstrip("!"), not skill.success.startswith("!"), source="mock")
        ctx.world.update(self.effects.get(skill.name, {}), source="mock")
        return SkillResult(Status.SUCCESS, skill)
