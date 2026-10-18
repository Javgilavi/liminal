"""Skill routing: decides which backend executes a named skill.

Primitives (hold / release / home / gripper) run on the local controller and never
touch the VLA. Everything else goes to the learned-policy backend (FLUX, scripted
fallback, or mock), optionally overridden per skill name.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from liminal.core.interfaces import SkillBackend
from liminal.core.types import Skill, SkillResult, Status

if TYPE_CHECKING:
    from liminal.orchestration.context import ExecutionContext


class PrimitiveBackend:
    """Hardware-level skills that must work even when FLUX is down (WS5)."""

    NAMES = frozenset({"hold", "release", "home", "open_gripper", "close_gripper"})

    def __init__(self, home_poses: dict[str, list[float]] | None = None, gripper_open: float = 50.0,
                 gripper_closed: float = 0.0, move_steps: int = 30, rate_hz: float = 30.0):
        self.home_poses = home_poses or {}
        self.gripper_open = gripper_open
        self.gripper_closed = gripper_closed
        self.move_steps = move_steps
        self.rate_hz = rate_hz

    async def execute(self, skill: Skill, ctx: ExecutionContext) -> SkillResult:
        t0 = time.monotonic()
        if skill.name not in self.NAMES:
            return SkillResult(Status.FAILURE, skill, f"unknown primitive {skill.name}")
        # A group ("both") runs the primitive on every member arm at the same time.
        await asyncio.gather(*(self._run_one(skill, arm, ctx) for arm in ctx.members(skill.arm)))
        return SkillResult(Status.SUCCESS, skill, duration_s=time.monotonic() - t0)

    async def _run_one(self, skill: Skill, arm_name: str, ctx: ExecutionContext) -> None:
        arm = ctx.arms[arm_name]
        if skill.name == "hold":
            arm.hold(skill.params.get("pose"))
        elif skill.name == "release":
            arm.release_hold()
            if skill.params.get("open_gripper", True):
                await self._move_gripper(arm_name, self.gripper_open, ctx)
        elif skill.name == "home":
            if arm_name not in self.home_poses and getattr(arm, "joint_limits", None) is not None:
                raise ValueError(f"{arm_name}: no measured home pose configured")
            arm.release_hold()
            await self._move_to(arm_name, self.home_poses.get(arm_name, [0.0] * arm.num_joints), ctx)
        elif skill.name == "open_gripper":
            await self._move_gripper(arm_name, self.gripper_open, ctx)
        elif skill.name == "close_gripper":
            await self._move_gripper(arm_name, self.gripper_closed, ctx)

    async def _move_gripper(self, arm_name: str, value: float, ctx: ExecutionContext) -> None:
        target = ctx.arms[arm_name].read_state()
        target[-1] = value
        await self._move_to(arm_name, target, ctx)

    async def _move_to(self, arm_name: str, target: list[float], ctx: ExecutionContext) -> None:
        start = ctx.arms[arm_name].read_state()
        for i in range(1, self.move_steps + 1):
            a = i / self.move_steps
            ctx.safety.send(arm_name, [s + a * (t - s) for s, t in zip(start, target, strict=True)])
            await asyncio.sleep(1.0 / self.rate_hz)


class SkillRouter:
    """`arm_policies` gives an arm its own policy backend (current architecture: one
    FLUX instance per arm). Arms not listed use the shared `policy`."""

    def __init__(self, policy: SkillBackend, primitives: SkillBackend | None = None,
                 overrides: dict[str, SkillBackend] | None = None,
                 arm_policies: dict[str, SkillBackend] | None = None):
        self.policy = policy
        self.primitives = primitives or PrimitiveBackend()
        self.overrides = overrides or {}
        self.arm_policies = arm_policies or {}

    def backend_for(self, skill: Skill) -> SkillBackend:
        if skill.name in self.overrides:
            return self.overrides[skill.name]
        if skill.name in PrimitiveBackend.NAMES:
            return self.primitives
        return self.arm_policies.get(skill.arm, self.policy)

    async def execute(self, skill: Skill, ctx: ExecutionContext) -> SkillResult:
        return await self.backend_for(skill).execute(skill, ctx)
