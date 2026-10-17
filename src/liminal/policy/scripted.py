"""Deterministic fallback backend: replay recorded joint waypoints per skill.

This is the hour-6 escape hatch (TASK.md): if FLUX is not running yet, the rest of
the team keeps integrating against this. Record waypoints with
scripts/record_waypoints.py (teleop, press enter at each keyframe).
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from liminal.core.types import Skill, SkillResult, Status

if TYPE_CHECKING:
    from liminal.orchestration.context import ExecutionContext


class ScriptedBackend:
    """waypoints[arm][skill] -> list of joint poses, interpolated at `rate_hz`."""

    def __init__(self, waypoints: dict[str, dict[str, list[list[float]]]], steps_per_segment: int = 30,
                 rate_hz: float = 30.0):
        self.waypoints = waypoints
        self.steps_per_segment = steps_per_segment
        self.rate_hz = rate_hz

    @classmethod
    def from_dir(cls, directory: str | Path, **kw: object) -> ScriptedBackend:
        """Load every `<arm>.yaml` in `directory` (mapping skill -> [pose, ...])."""
        wp: dict[str, dict[str, list[list[float]]]] = {}
        for f in sorted(Path(directory).glob("*.yaml")):
            wp[f.stem] = yaml.safe_load(f.read_text()) or {}
        return cls(wp, **kw)  # type: ignore[arg-type]

    async def execute(self, skill: Skill, ctx: ExecutionContext) -> SkillResult:
        t0 = time.monotonic()
        poses = self.waypoints.get(skill.arm, {}).get(skill.name)
        if not poses:
            return SkillResult(Status.FAILURE, skill, f"no waypoints for {skill.arm}:{skill.name}")
        for target in poses:
            start = ctx.arms[skill.arm].read_state()
            for i in range(1, self.steps_per_segment + 1):
                a = i / self.steps_per_segment
                ctx.safety.send(skill.arm, [s + a * (t - s) for s, t in zip(start, target, strict=True)])
                await asyncio.sleep(1.0 / self.rate_hz)
        return SkillResult(Status.SUCCESS, skill, duration_s=time.monotonic() - t0)
