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

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import yaml

from liminal.orchestration import task_spec
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
                data = yaml.safe_load(_strip_fences(text))
                return parse_task(data, arms=self.arms, skills=self.skills)
            except (TaskSpecError, yaml.YAMLError) as e:
                last_error = str(e)
        raise TaskSpecError(f"planner failed after {self.max_repairs + 1} attempts: {last_error}")


def _strip_fences(text: str) -> str:
    m = re.search(r"```(?:ya?ml)?\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1) if m else text


# -- model adapters (thin; swap freely) ------------------------------------------------


class ClaudeModel:
    def __init__(self, model: str = "claude-sonnet-5-5", max_tokens: int = 2048):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model, self.max_tokens = model, max_tokens

    def complete(self, system: str, user: str, images: Sequence[bytes] = ()) -> str:
        import base64

        content: list[dict[str, Any]] = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                         "data": base64.b64encode(img).decode()}}
            for img in images
        ]
        content.append({"type": "text", "text": user})
        msg = self.client.messages.create(model=self.model, max_tokens=self.max_tokens, system=system,
                                          messages=[{"role": "user", "content": content}])
        return "".join(b.text for b in msg.content if b.type == "text")


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
