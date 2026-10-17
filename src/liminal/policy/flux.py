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
import math
import time
from collections import deque
from dataclasses import dataclass, field
from numbers import Real
from typing import TYPE_CHECKING, Any

from liminal.core.types import Observation, Skill, SkillResult, Status

if TYPE_CHECKING:
    from liminal.orchestration.context import ExecutionContext


class _InvalidActionChunk(ValueError):
    pass


class _InferenceTimeout(TimeoutError):
    pass


@dataclass
class FluxConfig:
    checkpoint: str = "black-forest-labs/flux-3-action-so101"
    device: str = "cuda"
    dtype: str = "bfloat16"
    action_dim: int = 6  # 6 = per-arm (option A), 12 = bimanual (option B)
    chunk_size: int = 42  # actions predicted per inference
    execute_steps: int = 12  # short stop-inspect-continue window
    control_hz: float = 30.0
    inference_timeout_s: float = 30.0
    obs_history: int = 8
    camera_roles: tuple[str, ...] = ("scene", "wrist")
    # skill target -> {role -> camera name}, e.g. {"right": {"scene": "top", "wrist": "right_wrist"}}
    # or, for option B, {"both": {"scene": "top", "left_wrist": "left_wrist", "right_wrist": "right_wrist"}}
    camera_map: dict[str, dict[str, str]] = field(default_factory=dict)
    lora_adapter: str | None = None  # applied to every arm
    arm_adapters: dict[str, str] = field(default_factory=dict)  # option A1: arm -> LoRA, overrides lora_adapter

    def __post_init__(self) -> None:
        integer_fields = {
            "action_dim": self.action_dim,
            "chunk_size": self.chunk_size,
            "execute_steps": self.execute_steps,
            "obs_history": self.obs_history,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.execute_steps > self.chunk_size:
            raise ValueError("execute_steps cannot exceed chunk_size")
        for name, value in {
            "control_hz": self.control_hz,
            "inference_timeout_s": self.inference_timeout_s,
        }.items():
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")


class FluxPolicyBackend:
    def __init__(self, cfg: FluxConfig):
        self.cfg = cfg
        self._model: Any = None
        self._processor: Any = None
        self._infer_lock = asyncio.Lock()
        self._history: dict[str, deque[Observation]] = {}
        # A timed-out/cancelled worker thread cannot be killed safely. Quarantine this
        # model instance so a retry can never overlap that orphaned inference.
        self._unhealthy_reason: str | None = None
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

    def _stop(self, target: str, ctx: ExecutionContext) -> None:
        for arm in ctx.members(target):
            ctx.arms[arm].stop()

    def _validate_chunk(self, chunk: Any) -> list[list[float]]:
        """Validate a complete model response before issuing its first command."""

        try:
            actions = list(chunk)
        except TypeError as exc:
            raise _InvalidActionChunk("FLUX output must be an iterable action chunk") from exc
        if len(actions) != self.cfg.chunk_size:
            raise _InvalidActionChunk(
                f"FLUX returned {len(actions)} actions; expected exactly {self.cfg.chunk_size}"
            )

        validated: list[list[float]] = []
        for action_index, action in enumerate(actions):
            try:
                values = list(action)
            except TypeError as exc:
                raise _InvalidActionChunk(f"FLUX action {action_index} is not iterable") from exc
            if len(values) != self.cfg.action_dim:
                raise _InvalidActionChunk(
                    f"FLUX action {action_index} has {len(values)} values; "
                    f"expected {self.cfg.action_dim}"
                )
            if any(isinstance(value, bool) or not isinstance(value, Real) for value in values):
                raise _InvalidActionChunk(f"FLUX action {action_index} contains a non-numeric value")
            numeric = [float(value) for value in values]
            if not all(math.isfinite(value) for value in numeric):
                raise _InvalidActionChunk(f"FLUX action {action_index} contains a non-finite value")
            validated.append(numeric)
        return validated

    async def _predict_chunk(
        self,
        history: list[Observation],
        instruction: str,
        target: str,
    ) -> list[list[float]]:
        async with self._infer_lock:
            started_at = time.monotonic()
            try:
                chunk = await asyncio.wait_for(
                    asyncio.to_thread(self.predict, history, instruction, target),
                    timeout=self.cfg.inference_timeout_s,
                )
            except TimeoutError as exc:
                self._unhealthy_reason = (
                    f"inference exceeded {self.cfg.inference_timeout_s:g}s; "
                    "restart the backend before retrying"
                )
                raise _InferenceTimeout from exc
            except asyncio.CancelledError:
                self._unhealthy_reason = (
                    "execution was cancelled during inference; restart the backend before retrying"
                )
                raise
            else:
                self.latencies_s.append(time.monotonic() - started_at)
        return self._validate_chunk(chunk)

    async def _inspect_progress(
        self,
        skill: Skill,
        ctx: ExecutionContext,
        started_at: float,
    ) -> SkillResult | None:
        """Pause and ask perception whether to finish, continue, or abort.

        A ``None`` result means every invariant is confirmed and the success
        predicate is confirmed false, so another FLUX chunk may run.
        """

        invariants = ctx.active_invariants
        conditions = tuple(condition for condition in (skill.success, *invariants) if condition)
        if not conditions:
            return None
        if ctx.inspector is None:
            violated = tuple(condition for condition in invariants if ctx.world.evaluate(condition) is not True)
            if violated:
                self._stop(skill.arm, ctx)
                return SkillResult(
                    Status.FAILURE,
                    skill,
                    f"active invariant violated: {', '.join(violated)}",
                    time.monotonic() - started_at,
                )
            return None

        self._stop(skill.arm, ctx)
        try:
            request = ctx.inspection.request(
                conditions,
                context=f"The robot is executing the bounded skill {skill.name!r} on {skill.arm!r}.",
            )
        except (TypeError, ValueError) as exc:
            return SkillResult(Status.FAILURE, skill, f"invalid inspection configuration: {exc}")

        assessment = await ctx.inspector.inspect_until_known(
            request,
            timeout_s=ctx.inspection.timeout_s,
            retry_delay_s=ctx.inspection.retry_delay_s,
        )
        values = {key: observation.value.value for key, observation in assessment.observations.items()}
        ctx.world.log(
            "skill_inspection",
            skill=skill.label,
            predicates=values,
            invariants=invariants,
            parse_error=assessment.parse_error,
        )
        unknown = tuple(
            predicate for predicate in request.predicates if not assessment.observation(predicate).known
        )
        if unknown:
            return SkillResult(
                Status.FAILURE,
                skill,
                f"scene inspection inconclusive for {', '.join(unknown)}",
                time.monotonic() - started_at,
            )

        violated = tuple(condition for condition in invariants if ctx.world.evaluate(condition) is not True)
        if violated:
            ctx.world.log("skill_invariant_violated", skill=skill.label, conditions=violated)
            return SkillResult(
                Status.FAILURE,
                skill,
                f"active invariant violated: {', '.join(violated)}",
                time.monotonic() - started_at,
            )
        if skill.success and ctx.world.evaluate(skill.success) is True:
            ctx.world.log("skill_terminated", skill=skill.label, condition=skill.success, reason="vlm_confirmed")
            return SkillResult(
                Status.SUCCESS,
                skill,
                f"confirmed {skill.success}",
                time.monotonic() - started_at,
            )
        return None

    def reset(self, target: str) -> None:
        self._history.pop(target, None)

    async def execute(self, skill: Skill, ctx: ExecutionContext) -> SkillResult:
        """Execute a bounded skill and freeze its arm on every exit path."""

        started_at = time.monotonic()
        try:
            return await self._execute(skill, ctx, started_at)
        except asyncio.CancelledError:
            ctx.world.log("flux_cancelled", skill=skill.label)
            raise
        except _InferenceTimeout:
            return SkillResult(
                Status.TIMEOUT,
                skill,
                self._unhealthy_reason or "FLUX inference timed out",
                time.monotonic() - started_at,
            )
        except _InvalidActionChunk as exc:
            return SkillResult(
                Status.FAILURE,
                skill,
                f"invalid FLUX output: {exc}",
                time.monotonic() - started_at,
            )
        finally:
            self._stop(skill.arm, ctx)

    async def _execute(
        self,
        skill: Skill,
        ctx: ExecutionContext,
        started_at: float,
    ) -> SkillResult:
        if not skill.prompt:
            return SkillResult(Status.FAILURE, skill, "policy skills need a text prompt")
        if self._unhealthy_reason is not None:
            return SkillResult(
                Status.FAILURE,
                skill,
                f"FLUX backend is quarantined: {self._unhealthy_reason}",
                time.monotonic() - started_at,
            )
        dof = sum(ctx.arms[a].num_joints for a in ctx.members(skill.arm))
        if dof != self.cfg.action_dim:
            return SkillResult(Status.FAILURE, skill,
                               f"{skill.arm!r} has {dof} joints but this policy outputs {self.cfg.action_dim}-D "
                               "actions (6 = one arm, option A; 12 = arm group, option B)")
        if self._model is None:
            self.load()
        history = self._history.setdefault(skill.arm, deque(maxlen=self.cfg.obs_history))
        period = 1.0 / self.cfg.control_hz
        success_version = ctx.world.observation_version(skill.success) if skill.success else 0

        # Inspect before the first motion. This catches an already-completed skill and
        # refuses to enter a WHILE body when its physical invariant is not observable.
        result = await self._inspect_progress(skill, ctx, started_at)
        if result is not None:
            return result

        while True:
            history.append(self._observe(skill.arm, ctx))
            chunk = await self._predict_chunk(list(history), skill.prompt, skill.arm)
            for action in chunk[: self.cfg.execute_steps]:
                self._send(skill.arm, action, ctx)
                await asyncio.sleep(period)
                violated = tuple(
                    condition for condition in ctx.active_invariants
                    if ctx.world.evaluate(condition) is not True
                )
                if violated:
                    self._stop(skill.arm, ctx)
                    return SkillResult(
                        Status.FAILURE,
                        skill,
                        f"active invariant violated: {', '.join(violated)}",
                        time.monotonic() - started_at,
                    )
                observed_success = (
                    ctx.world.holds(skill.success, after_version=success_version)
                    if ctx.inspector is not None else ctx.world.holds(skill.success)
                ) if skill.success else False
                if observed_success:
                    self._stop(skill.arm, ctx)
                    return SkillResult(Status.SUCCESS, skill, duration_s=time.monotonic() - started_at)

            result = await self._inspect_progress(skill, ctx, started_at)
            if result is not None:
                return result
            if not skill.success:
                # No predicate to wait for: one chunk is the whole skill.
                return SkillResult(Status.SUCCESS, skill, duration_s=time.monotonic() - started_at)
            # Otherwise keep re-planning; SkillNode's timeout bounds this loop.
