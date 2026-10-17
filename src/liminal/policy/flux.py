"""FLUX 3 Action skill backend (owned by WS1).

One model instance, shared by per-arm sessions:

* each arm keeps its *own* observation history - never mix the two arms' queues
* inference is single-flight (one request at a time) behind an asyncio.Lock; the
  arm that is not running inference sits on its local position-hold controller
* actions are executed in chunks at `control_hz`, every one through the safety
  supervisor, and the skill finishes early as soon as its success predicate holds

Model loading / pre-processing is intentionally left as the one TODO below: wire it
against the official BFL integration docs for the SO-101 checkpoint and keep every
number in configs/policy/flux3_so101.yaml (all marked VERIFY until checked).
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
    action_dim: int = 6
    chunk_size: int = 42  # actions predicted per inference
    execute_steps: int = 32  # actions executed per chunk before re-planning
    control_hz: float = 30.0
    obs_history: int = 8
    camera_roles: tuple[str, ...] = ("scene", "wrist")
    # arm -> {role -> camera name}, e.g. {"right": {"scene": "top", "wrist": "right_wrist"}}
    camera_map: dict[str, dict[str, str]] = field(default_factory=dict)
    lora_adapter: str | None = None


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
        """TODO(WS1): load checkpoint + saved processors (+ optional LoRA adapter).

        Expected shape of the work (confirm every detail against the BFL docs):
          1. load the checkpoint and its saved pre/post-processing on `cfg.device`
          2. apply `cfg.lora_adapter` if set
          3. run `predict` on a saved observation and record latency before touching
             hardware (scripts/flux_smoke_test.py)
        """
        raise NotImplementedError(
            "FLUX 3 Action loading is not wired yet - see FluxPolicyBackend.load and TASK.md WS1. "
            "Use --backend scripted or --backend mock meanwhile."
        )

    def predict(self, history: list[Observation], instruction: str) -> list[list[float]]:
        """Return a chunk of `chunk_size` absolute joint targets for one arm.
        TODO(WS1): build model inputs from `history` (scene + wrist frames, joint state)."""
        raise NotImplementedError

    # -- execution ---------------------------------------------------------------------

    def _observe(self, arm: str, ctx: ExecutionContext) -> Observation:
        roles = self.cfg.camera_map.get(arm, {})
        images = {role: ctx.cameras[cam].read() for role, cam in roles.items() if cam in ctx.cameras}
        return Observation(arm=arm, state=ctx.arms[arm].read_state(), images=images)

    def reset(self, arm: str) -> None:
        self._history.pop(arm, None)

    async def execute(self, skill: Skill, ctx: ExecutionContext) -> SkillResult:
        if self._model is None:
            self.load()
        if not skill.prompt:
            return SkillResult(Status.FAILURE, skill, "policy skills need a text prompt")
        t0 = time.monotonic()
        history = self._history.setdefault(skill.arm, deque(maxlen=self.cfg.obs_history))
        period = 1.0 / self.cfg.control_hz

        while True:
            history.append(self._observe(skill.arm, ctx))
            async with self._infer_lock:
                t_inf = time.monotonic()
                chunk = await asyncio.to_thread(self.predict, list(history), skill.prompt)
                self.latencies_s.append(time.monotonic() - t_inf)
            for action in chunk[: self.cfg.execute_steps]:
                ctx.safety.send(skill.arm, action)
                await asyncio.sleep(period)
                if skill.success and ctx.world.holds(skill.success):
                    return SkillResult(Status.SUCCESS, skill, duration_s=time.monotonic() - t0)
            history.append(self._observe(skill.arm, ctx))
            if not skill.success:
                # No predicate to wait for: one chunk is the whole skill.
                return SkillResult(Status.SUCCESS, skill, duration_s=time.monotonic() - t0)
            # Otherwise keep re-planning; SkillNode's timeout bounds this loop.
