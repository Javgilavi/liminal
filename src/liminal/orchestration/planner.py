"""High-level planners that turn a goal into a task tree (owned by WS3).

All planners emit the task_spec grammar, which is then compiled and validated. The
LLM / VLM never sends motor commands and never decides on its own that a task
succeeded - the behavior tree and perception do. Options being compared in
docs/research/03_long_horizon_orchestration.md:

* TemplatePlanner - hand-written trees from configs/tasks (baseline, always works)
* LLMPlanner      - frontier LLM / VLM (Claude, Gemini Robotics-ER, ...) proposes a
                    tree; compile errors are fed back for self-correction
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import yaml

from liminal.orchestration import task_spec
from liminal.orchestration.plan_validation import (
    PlanningContext,
    PlanValidationError,
    ValidatedPlan,
    parse_proposal_json,
    validate_proposal,
)
from liminal.orchestration.task_spec import TaskSpec, TaskSpecError, load_task, parse_task

GRAMMAR = task_spec.__doc__ or ""


class TextModel(Protocol):
    def complete(self, system: str, user: str, images: Sequence[bytes] = ()) -> str: ...


class TemplatePlanner:
    """Pick a pre-written task by name or keyword. The reliable demo path."""

    def __init__(self, task_dir: str | Path = "configs/tasks"):
        self.tasks = {p.stem: p for p in Path(task_dir).glob("*.yaml")}

    def plan(self, goal: str, **_: Any) -> TaskSpec:
        key = goal.strip().lower().replace(" ", "_")
        if key in self.tasks:
            return load_task(self.tasks[key])
        words = set(re.findall(r"[a-z]+", goal.lower()))
        scored = sorted(((len(words & set(stem.split("_"))), stem) for stem in self.tasks), reverse=True)
        if scored and scored[0][0] > 0:
            return load_task(self.tasks[scored[0][1]])
        raise KeyError(f"no template for {goal!r}; have {sorted(self.tasks)}")


SYSTEM_PROMPT = """You are the task planner for a dual-arm SO-101 robot.
Decompose the user's goal into a behavior tree using ONLY the grammar below.
Rules:
- Use only the listed arms and skills. Policy skills need a short imperative `prompt`
  describing one bounded manipulation (the VLA executes it).
- Give every skill that changes the scene a `success` predicate the camera can verify.
- If one arm must keep something in place while the other works, use `while`.
- Wrap fragile manipulation in `retry`.
- End with `check:` on the goal predicate.
Reply with a single YAML document (keys: name, goal, initial_state, tree) and nothing else.

GRAMMAR:
<<GRAMMAR>>
"""


class LLMPlanner:
    def __init__(self, model: TextModel, arms: Sequence[str], skills: dict[str, str],
                 predicates: Sequence[str], max_repairs: int = 2):
        self.model = model
        self.arms = list(arms)
        self.skills = skills  # name -> one-line description
        self.predicates = list(predicates)
        self.max_repairs = max_repairs

    def _user_prompt(self, goal: str, world: dict[str, Any]) -> str:
        return (
            f"GOAL: {goal}\n"
            f"ARMS: {self.arms}\n"
            f"SKILLS:\n" + "\n".join(f"- {k}: {v}" for k, v in self.skills.items()) + "\n"
            f"OBSERVABLE PREDICATES: {self.predicates}\n"
            f"CURRENT WORLD STATE: {json.dumps(world, default=str)}\n"
        )

    def plan(self, goal: str, world: dict[str, Any] | None = None, images: Sequence[bytes] = ()) -> TaskSpec:
        system = SYSTEM_PROMPT.replace("<<GRAMMAR>>", GRAMMAR)
        user = self._user_prompt(goal, world or {})
        last_error = ""
        for _ in range(self.max_repairs + 1):
            text = self.model.complete(system, user + (f"\nYOUR PREVIOUS PLAN WAS INVALID: {last_error}\n"
                                                       if last_error else ""), images)
            try:
                if not isinstance(text, str):
                    raise PlanValidationError("model response missing or exceeds byte budget")
                try:
                    encoded_size = len(text.encode("utf-8"))
                except UnicodeEncodeError as exc:
                    raise PlanValidationError("model response contains invalid Unicode") from exc
                if encoded_size > 16_384:
                    raise PlanValidationError("model response missing or exceeds byte budget")
                data = yaml.safe_load(_strip_fences(text))
                _reject_model_state_writes(data)
                return parse_task(data, arms=self.arms, skills=self.skills)
            except (TaskSpecError, yaml.YAMLError, PlanValidationError) as e:
                last_error = str(e)
        raise TaskSpecError(f"planner failed after {self.max_repairs + 1} attempts: {last_error}")


def _strip_fences(text: str) -> str:
    m = re.search(r"```(?:ya?ml)?\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1) if m else text


def _reject_model_state_writes(value: Any, path: str = "plan") -> None:
    """Keep legacy planner output from asserting physical facts through task YAML."""
    stack = [(path, value, 0)]
    visited: set[int] = set()
    while stack:
        current_path, current, depth = stack.pop()
        if depth > 16 or len(visited) > 2_000:
            raise PlanValidationError(f"{current_path}: model plan structure exceeds bounds")
        if isinstance(current, (dict, list)):
            identity = id(current)
            if identity in visited:
                raise PlanValidationError(f"{current_path}: recursive YAML aliases are not allowed")
            visited.add(identity)
        if isinstance(current, dict):
            for key, child in current.items():
                if isinstance(key, str) and key.lower() in {"initial_state", "set"}:
                    raise PlanValidationError(f"{current_path}.{key}: model plans cannot write physical state")
                stack.append((f"{current_path}.{key}", child, depth + 1))
        elif isinstance(current, list):
            for index, child in enumerate(current):
                stack.append((f"{current_path}[{index}]", child, depth + 1))


PROPOSAL_SYSTEM_PROMPT = """You propose one bounded next manipulation for a robot task.
Return exactly one JSON object matching the schema in the user message. Use only listed
runtime capabilities, objects, arms/groups, effects and evidence references. The trusted
goal obligations must be copied exactly. Do not report facts, write initial state, use
set nodes, provide code, or claim task completion. Historical images are context only;
use current labeled frames for present object bindings. The dependency sketch may describe
the complete ordering, but steps must contain only the bounded next invocation(s). The
step prompt field is an explanation only; trusted code renders the executable instruction
from the capability prompt template. Preconditions come only from the capability registry.
Optional "memory" is a record built from verified ledger facts: only its "completed" list is done, and
everything else must be re-checked on the current frames. Optional "experience" holds advice from earlier
runs; it is never evidence about the present scene and never completes an obligation. When the response
shape lists keyframe_nominations, you may nominate up to four of the listed frame_ids (with a short
reason) that are worth keeping as history; nominations are advisory.
"""


class ConstrainedPlanner:
    """One provider-neutral bounded proposal path for typed multimodal contexts."""

    def __init__(self, model: TextModel, max_repairs: int = 1):
        self.model = model
        self.max_repairs = max(0, max_repairs)
        self._count_lock = threading.Lock()
        self.last_model_call_count = 0

    def _prompt(self, context: PlanningContext, previous_error: str = "") -> str:
        frames = [{"frame_id": frame.frame_id, "camera_role": frame.camera_role,
                   "captured_at": frame.captured_at, "temporal_role": frame.temporal_role,
                   "media_type": frame.media_type} for frame in context.observations]
        registry = [{"skill_id": skill_id, "arms": cap.arms,
                     "effect_templates": cap.effect_templates,
                     "prompt_template": cap.prompt_template, "preconditions": cap.preconditions,
                     "max_timeout_s": cap.max_timeout_s, "max_attempts": cap.max_attempts,
                     "resources": cap.resources, "changes_scene": cap.changes_scene}
                    for skill_id, cap in context.capabilities.skills.items()]
        schema = {
            "schema_version": 1, "run_id": context.run_id, "base_revision": context.revision,
            "goal_id": context.goal.goal_id, "goal_obligations": list(context.goal.obligations),
            "dependency_sketch": ["short string"], "evidence_refs": ["frame or evidence id"],
            "steps": [{"id": "short id", "skill": "registered skill id", "object_id": "known id",
                       "destination_id": "known id or empty string", "arm": "available arm/group",
                       "prompt": "bounded explanation; not executed",
                       "effects": ["exact registered bound effect"], "evidence_refs": ["known id"],
                       "timeout_s": 10, "attempts": 1}],
        }
        if context.nominate_keyframes:
            schema["keyframe_nominations"] = [{"frame_id": "a listed frame_id", "reason": "why it is worth keeping"}]
        payload = {"run_id": context.run_id, "revision": context.revision,
                   "trusted_goal": {"id": context.goal.goal_id, "text": context.goal.text,
                                    "obligations": context.goal.obligations},
                   "frames": frames,
                   "evidence": context.evidence,
                   "objects": context.object_labels,
                   "progress": context.progress,
                   **({"memory": context.memory} if context.memory else {}),
                   **({"experience": context.experience} if context.experience else {}),
                   "runtime_capabilities": registry,
                   "available_arms": context.capabilities.available_arms,
                   "arm_groups": context.capabilities.arm_groups,
                   "budgets": context.budget,
                   "required_response_shape": schema}
        try:
            user = json.dumps(payload, default=lambda item: item.__dict__, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError, RecursionError) as exc:
            raise PlanValidationError("planning context is not bounded JSON data") from exc
        if previous_error:
            user += "\nRepair the previous invalid response. Validation error: " + previous_error[:800]
        return user

    def plan(self, context: PlanningContext) -> ValidatedPlan:
        """Call the model within its configured call budget and validate every response."""
        return self._plan(context)

    def _plan(self, context: PlanningContext, deadline: float | None = None) -> ValidatedPlan:
        last_error = ""
        call_limit = min(context.budget.max_model_calls, self.max_repairs + 1)
        if call_limit < 1:
            raise PlanValidationError("planner model-call budget is zero")
        with self._count_lock:
            self.last_model_call_count = 0
        images = tuple(frame.data for frame in context.observations)
        for attempt in range(call_limit):
            if deadline is not None and time.monotonic() >= deadline:
                raise PlanValidationError("planning deadline exceeded; no further model call started")
            user = self._prompt(context, last_error)
            complete_context = getattr(self.model, "complete_context", None)
            with self._count_lock:
                self.last_model_call_count = attempt + 1
            if complete_context is not None:
                text = complete_context(PROPOSAL_SYSTEM_PROMPT, user, context.observations)
            else:
                text = self.model.complete(PROPOSAL_SYSTEM_PROMPT, user, images)
            if deadline is not None and time.monotonic() >= deadline:
                raise PlanValidationError("planning deadline exceeded; late model result discarded")
            try:
                proposal = parse_proposal_json(text, context.budget.max_proposal_bytes)
                return validate_proposal(proposal, context)
            except (PlanValidationError, TypeError, ValueError) as exc:
                last_error = str(exc)
        raise PlanValidationError(f"planner failed after {call_limit} model calls: {last_error}")

    async def plan_async(self, context: PlanningContext, deadline_s: float | None = None) -> ValidatedPlan:
        """Run provider I/O off-loop; timeout prevents a late proposal being consumed."""
        if deadline_s is None:
            return await asyncio.to_thread(self.plan, context)
        if deadline_s <= 0:
            raise PlanValidationError("planning deadline must be positive")
        deadline = time.monotonic() + deadline_s
        try:
            return await asyncio.wait_for(asyncio.to_thread(self._plan, context, deadline), timeout=deadline_s)
        except TimeoutError as exc:
            raise PlanValidationError("planning deadline exceeded; late result discarded") from exc


# -- model adapters (thin; swap freely) ------------------------------------------------


class ClaudeModel:
    def __init__(self, model: str | None = None, max_tokens: int = 1400, timeout_s: float = 20.0):
        import anthropic

        self.client = anthropic.Anthropic(timeout=timeout_s, max_retries=0)
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5")
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str, images: Sequence[bytes] = ()) -> str:
        import base64

        content: list[dict[str, Any]] = self._image_content(base64, images, ["image/jpeg"] * len(images))
        content.append({"type": "text", "text": user})
        msg = self.client.messages.create(model=self.model, max_tokens=self.max_tokens, system=system,
                                          messages=[{"role": "user", "content": content}])
        return "".join(b.text for b in msg.content if b.type == "text")

    def complete_context(self, system: str, user: str, frames: Sequence[Any]) -> str:
        import base64

        content: list[dict[str, Any]] = []
        for frame in frames:
            content.append({"type": "text", "text": f"FRAME {frame.frame_id} ({frame.temporal_role}, "
                                                       f"camera={frame.camera_role}, captured_at={frame.captured_at})"})
            content.extend(self._image_content(base64, [frame.data], [frame.media_type]))
        content.append({"type": "text", "text": user})
        msg = self.client.messages.create(model=self.model, max_tokens=self.max_tokens, system=system,
                                          messages=[{"role": "user", "content": content}])
        return "".join(block.text for block in msg.content if block.type == "text")

    @staticmethod
    def _image_content(base64: Any, images: Sequence[bytes], media_types: Sequence[str]) -> list[dict[str, Any]]:
        return [{"type": "image", "source": {"type": "base64", "media_type": media_type,
                                                  "data": base64.b64encode(image).decode()}}
                for image, media_type in zip(images, media_types, strict=True)]


class GeminiModel:
    """Gemini (incl. Robotics-ER) via google-genai. Set the model id in configs."""

    def __init__(self, model: str):
        from google import genai

        self.client = genai.Client()
        self.model = model

    def complete(self, system: str, user: str, images: Sequence[bytes] = ()) -> str:
        from google.genai import types

        parts: list[Any] = [types.Part.from_bytes(data=img, mime_type="image/jpeg") for img in images]
        parts.append(user)
        resp = self.client.models.generate_content(
            model=self.model, contents=parts, config=types.GenerateContentConfig(system_instruction=system)
        )
        return resp.text or ""
