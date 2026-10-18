"""FLUX 3 Action skill backend (owned by WS1). Supports both embodiment options in
PLAN.md with the same class; the config decides:

* Option A, per-arm (`action_dim: 6`): skills target one arm. A1 = one shared model
  with per-arm sessions (optionally a per-arm LoRA in `arm_adapters`); A2 = one
  backend instance per arm (`instances: per_arm` in the policy config, see app.py).
* Option B, bimanual (`action_dim: 12`): skills target an arm group such as `both`.
  One fine-tuned model sees both arms and predicts one concatenated action vector,
  split back per arm in group order.

Either way:

* each skill target (arm or group) keeps its *own* observation history
* inference is single-flight per instance behind an asyncio.Lock (this is also what
  makes per-arm adapter switching safe); idle arms sit on their hold controller
* actions are executed in chunks at `control_hz`, every one through the safety
  supervisor, and the skill finishes early as soon as its success predicate holds

Model loading / pre-processing is intentionally left as the one TODO below: wire it
against the official BFL integration docs and keep every number in
configs/policy/*.yaml (all marked VERIFY until checked).
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from liminal.core.types import Observation, Skill, SkillResult, Status

if TYPE_CHECKING:
    from liminal.orchestration.context import ExecutionContext


@dataclass
class FluxConfig:
    checkpoint: str = "black-forest-labs/flux-3-action-so101"
    device: str = "cuda"
    dtype: str = "bfloat16"
    action_dim: int = 6  # 6 = per-arm (option A), 12 = bimanual (option B)
    chunk_size: int = 42  # actions predicted per inference
    execute_steps: int = 32  # actions executed per chunk before re-planning
    control_hz: float = 30.0
    obs_history: int = 8
    camera_roles: tuple[str, ...] = ("scene", "wrist")
    # skill target -> {role -> camera name}, e.g. {"right": {"scene": "top", "wrist": "right_wrist"}}
    # or, for option B, {"both": {"scene": "top", "left_wrist": "left_wrist", "right_wrist": "right_wrist"}}
    camera_map: dict[str, dict[str, str]] = field(default_factory=dict)
    lora_adapter: str | None = None  # applied to every arm
    arm_adapters: dict[str, str] = field(default_factory=dict)  # option A1: arm -> LoRA, overrides lora_adapter


class FluxPolicyBackend:
    def __init__(self, cfg: FluxConfig):
        self.cfg = cfg
        self._model: Any = None
        self._processor: Any = None
        self._infer_lock = asyncio.Lock()
        self._history: dict[str, deque[Observation]] = {}
        self.latencies_s: list[float] = []

    # -- model -------------------------------------------------------------------------

    def load(self) -> None:
        """TODO(WS1): load checkpoint + saved processors (+ LoRA adapters).

        Expected shape of the work (confirm every detail against the BFL docs):
          1. load `cfg.checkpoint` and its saved pre/post-processing on `cfg.device`
          2. load `cfg.lora_adapter` and/or every adapter in `cfg.arm_adapters`
          3. run `predict` on a saved observation and record latency before touching
             hardware (scripts/flux_smoke_test.py)
        Option B: `cfg.checkpoint` is the bimanual fine-tune from WS4, whose action
        head outputs `action_dim` = 12.
        """
        raise NotImplementedError(
            "FLUX 3 Action loading is not wired yet - see FluxPolicyBackend.load and TASK.md WS1. "
            "Use --backend scripted or --backend mock meanwhile."
        )

    def predict(self, history: list[Observation], instruction: str, target: str) -> list[list[float]]:
        """Return a chunk of `chunk_size` absolute joint targets, each `action_dim` long.
        `target` is the arm or group; use it to activate `cfg.arm_adapters[target]`.
        TODO(WS1): build model inputs from `history` (camera frames by role, joint state)."""
        raise NotImplementedError

    # -- execution ---------------------------------------------------------------------

    def _observe(self, target: str, ctx: ExecutionContext) -> Observation:
        roles = self.cfg.camera_map.get(target, {})
        images = {role: ctx.cameras[cam].read() for role, cam in roles.items() if cam in ctx.cameras}
        # A group's state is its members' joint vectors concatenated in group order.
        state = [q for arm in ctx.members(target) for q in ctx.arms[arm].read_state()]
        return Observation(arm=target, state=state, images=images)

    def _send(self, target: str, action: list[float], ctx: ExecutionContext) -> None:
        i = 0
        for arm in ctx.members(target):
            n = ctx.arms[arm].num_joints
            ctx.safety.send(arm, action[i : i + n])
            i += n

    def reset(self, target: str) -> None:
        self._history.pop(target, None)

    async def execute(self, skill: Skill, ctx: ExecutionContext) -> SkillResult:
        if not skill.prompt:
            return SkillResult(Status.FAILURE, skill, "policy skills need a text prompt")
        dof = sum(ctx.arms[a].num_joints for a in ctx.members(skill.arm))
        if dof != self.cfg.action_dim:
            return SkillResult(Status.FAILURE, skill,
                               f"{skill.arm!r} has {dof} joints but this policy outputs {self.cfg.action_dim}-D "
                               "actions (6 = one arm, option A; 12 = arm group, option B)")
        if self._model is None:
            self.load()
        t0 = time.monotonic()
        history = self._history.setdefault(skill.arm, deque(maxlen=self.cfg.obs_history))
        period = 1.0 / self.cfg.control_hz

        while True:
            await self._infer_lock.acquire()
            t_inf = time.monotonic()
            # Cancellation cannot stop a thread. Keep the model locked until it really
            # exits, but discard its output immediately when the skill is cancelled.
            try:
                # Another target or a cancelled inference may have held the lock;
                # observe only now so resumption uses the current robot pose.
                history.append(self._observe(skill.arm, ctx))
                prediction = asyncio.create_task(
                    asyncio.to_thread(self.predict, list(history), skill.prompt, skill.arm)
                )
            except BaseException:
                self._infer_lock.release()
                raise

            def finished(future):
                self._infer_lock.release()
                if not future.cancelled():
                    future.exception()  # retrieve errors even when the caller was cancelled

            prediction.add_done_callback(finished)
            chunk = await asyncio.shield(prediction)
            self.latencies_s.append(time.monotonic() - t_inf)
            for action in chunk[: self.cfg.execute_steps]:
                self._send(skill.arm, action, ctx)
                await asyncio.sleep(period)
                if skill.success and ctx.world.holds(skill.success):
                    return SkillResult(Status.SUCCESS, skill, duration_s=time.monotonic() - t0)
            if not skill.success:
                # No predicate to wait for: one chunk is the whole skill.
                return SkillResult(Status.SUCCESS, skill, duration_s=time.monotonic() - t0)
            # Otherwise keep re-planning; SkillNode's timeout bounds this loop.
