"""SkillBackend that runs each liminal skill as one ws4 step on the Haller."""

from __future__ import annotations

import time
from collections.abc import Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from liminal.core.types import Skill, SkillResult, Status
from liminal.safety.supervisor import EmergencyStop

if TYPE_CHECKING:
    from liminal.orchestration.context import ExecutionContext


@dataclass(frozen=True)
class Ws4Skill:
    ws4_skill: str  # name in the ws4 vocabulary (ws4/orchestrator/skills.yaml)
    package: str = ""  # the policy-server package it must be served by (checked against the vocabulary)
    checkpoint: str = ""  # provenance, logged only
    max_attempts: int | None = None


class Ws4ChainBackend:
    """The tree decides; ws4 executes. ``execute`` maps the skill to its ws4 skill, refuses anything that does not
    match what the LoRA was trained and served for (prompt, arm, package), claims the other arm when the ws4 skill
    also poses it (open_box tucks the pen arm), runs the step through ``Ws4Bridge.run_step`` and then has the
    harness's own perception look again, so SkillNode verifies ``success`` on fresh evidence rather than on ws4's
    word."""

    def __init__(self, skills: Mapping[str, Ws4Skill], max_attempts: int = 1):
        self.skills = dict(skills)
        self.max_attempts = int(max_attempts)
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> Ws4ChainBackend:
        unknown = set(cfg) - {"skills", "max_attempts"}
        if unknown:
            raise ValueError(f"unknown ws4 policy settings {sorted(unknown)}")
        skills = {name: Ws4Skill(**dict(v)) for name, v in (cfg.get("skills") or {}).items()}
        if not skills:
            raise ValueError("the ws4 policy config needs a skills: map (liminal skill -> ws4_skill)")
        return cls(skills, cfg.get("max_attempts", 1))

    async def execute(self, skill: Skill, ctx: ExecutionContext) -> SkillResult:
        t0 = time.monotonic()

        def fail(message: str) -> SkillResult:
            ctx.world.log("ws4_refused", skill=skill.label, reason=message)
            return SkillResult(Status.FAILURE, skill, message, time.monotonic() - t0)

        entry = self.skills.get(skill.name)
        if entry is None:
            return fail(f"no ws4 skill for {skill.name!r} (have {sorted(self.skills)})")
        if len(ctx.members(skill.arm)) != 1:
            return fail(f"{skill.arm!r} is an arm group; ws4 skills run on one arm")
        arm = ctx.arms[skill.arm]
        bridge = getattr(arm, "ws4_bridge", None)
        if bridge is None:
            return fail(f"{skill.arm!r} is not a ws4 arm (robot config type: ws4)")
        try:
            spec = bridge.vocab_skill(entry.ws4_skill)
        except KeyError as exc:
            return fail(str(exc))
        if skill.prompt and skill.prompt != spec.prompt:
            return fail(f"prompt {skill.prompt!r} is not the one {entry.ws4_skill} was trained on ({spec.prompt!r})")
        if arm.ws4_arm not in spec.arms:
            return fail(f"{entry.ws4_skill} runs on the robot's {'/'.join(spec.arms)} arm, not on {arm.ws4_arm} "
                        f"(liminal {skill.arm!r})")
        if entry.package and spec.package != entry.package:
            return fail(f"{entry.ws4_skill} is served as {spec.package!r}, the policy config expects {entry.package!r}")
        if bridge.holding(arm.ws4_arm):
            return fail(f"the {arm.ws4_arm} arm holds the lid (open_box finish: hold, or hold_arms): no step may "
                        "drive it again")
        by_side = {getattr(a, "ws4_arm", None): name for name, a in ctx.arms.items()
                   if getattr(a, "ws4_bridge", None) is bridge}
        other = by_side.get("right" if arm.ws4_arm == "left" else "left")
        ctx.safety.check()
        async with AsyncExitStack() as stack:
            if spec.other_arm_pose and other is not None:
                # ws4 moves the other arm too (to its `other_arm_pose`): no other node may own it meanwhile
                owner = f"{skill.label}:ws4 {spec.other_arm_pose}"
                await stack.enter_async_context(ctx.safety.claim(other, owner=owner))
            ctx.world.log("ws4_step", skill=skill.label, ws4_skill=entry.ws4_skill, ws4_arm=arm.ws4_arm,
                          package=spec.package, checkpoint=entry.checkpoint)
            result = await bridge.run_step(entry.ws4_skill, arm.ws4_arm, label=skill.label,
                                           max_attempts=entry.max_attempts or self.max_attempts,
                                           should_stop=lambda: ctx.safety.estopped)
        ctx.world.log("ws4_result", skill=skill.label, status=result.get("status"), failure=result.get("failure"),
                      run_dir=result.get("run_dir"), steps_done=result.get("steps_done"),
                      interventions=result.get("interventions"))
        if ctx.safety.estopped:
            raise EmergencyStop("e-stop during a ws4 step")
        if result.get("status") != "succeeded":
            return SkillResult(Status.FAILURE, skill, f"ws4 {result.get('status')}: {result.get('failure', '')}",
                               time.monotonic() - t0)
        if ctx.inspector is not None and skill.success:
            # ws4's Haiku check passed; the tree's own perception looks again so SkillNode sees fresh evidence
            request = ctx.inspection.request((skill.success, *ctx.active_invariants),
                                             context=f"The robot finished the bounded skill {skill.name!r}.")
            await ctx.inspector.inspect_until_known(request, timeout_s=ctx.inspection.timeout_s,
                                                    retry_delay_s=ctx.inspection.retry_delay_s)
        return SkillResult(Status.SUCCESS, skill, f"ws4 step succeeded ({result.get('run_dir')})",
                           time.monotonic() - t0)
