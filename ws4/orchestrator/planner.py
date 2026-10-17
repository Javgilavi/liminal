"""Planners: goal (+ head image, + progress when replanning) -> validated plan.

``ClaudePlanner`` asks Claude for a plan through structured output, validates it, and feeds validator errors back
at most ``feedback_rounds`` times. ``StubPlanner`` is deterministic and offline (tests, laptop demo without a key).
Neither output reaches the motors: a human approves every plan and the executor runs vocabulary skills only.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .claude import ClaudeJSON, image_block
from .plan import Plan, Step, fill_defaults, parse_plan, plan_json_schema, validate_plan
from .skills import ARMS, Skill, Vocabulary

DEFAULT_PLANNER_MODEL = "claude-opus-5-5"


@dataclass
class PlanResult:
    plan: Plan | None
    errors: list[str]
    attempts: int
    source: str
    latency_s: float = 0.0
    cost_usd: float = 0.0
    raw: list[Any] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.plan is not None and not self.errors


def _done_ids(state: dict | None) -> set[str]:
    return {d["id"] for d in (state or {}).get("done", [])}


class StubPlanner:
    """Keyword planner. Clauses split on "then" become stages; each stage depends on the previous one; skills inside
    a stage alternate arms (or follow ``arm_hint``) so they can run concurrently. A skill with ``hold_after`` gets a
    hold step that lasts through the next stage, whose steps then go to the other arm."""

    name = "stub"

    def __init__(self, vocab: Vocabulary, arm_hint: Callable[[Skill], str | None] | None = None):
        self.vocab = vocab
        self.arm_hint = arm_hint

    def _clause_skills(self, clause: str) -> list[Skill]:
        found: list[tuple[int, Skill]] = []
        for skill in self.vocab.skills.values():
            if skill.kind not in ("flux", "operator"):
                continue
            hits = [clause.find(k) for k in skill.keywords if k in clause]
            if hits:
                found.append((min(hits), skill))
        return [s for _, s in sorted(found, key=lambda t: t[0])]

    def plan(self, goal: str, *, image: np.ndarray | None = None, state: dict | None = None) -> PlanResult:
        start = time.perf_counter()
        done = _done_ids(state)
        done_skills = {d["skill"] for d in (state or {}).get("done", [])}
        text = goal.lower()
        repeat = 1
        m = re.search(r"\b(\d+)\s*(?:x|times|cycles?|rounds?)\b", text)
        if m:
            repeat = max(1, min(int(m.group(1)), 10))
            text = text[:m.start()] + text[m.end():]
        clauses = [c.strip() for c in re.split(r"\bthen\b|\band then\b|;|,|\.(?:\s|$)", text) if c.strip()]
        stages = [[s for s in self._clause_skills(c) if s.name not in done_skills] for c in clauses]
        stages = [s for s in stages if s] * repeat
        while stages and all(s.kind == "operator" for s in stages[-1]):
            stages.pop()  # "..., I reset it each time": a reset only between repetitions, not after the last one
        if not stages:
            return PlanResult(None, [f"no vocabulary skill matches the goal {goal!r}"], 1, self.name)
        used: set[str] = set(done)
        steps: list[Step] = []
        last_on_arm: dict[str, str] = {}
        prev_stage: list[str] = []
        held: dict[str, Step] = {}  # arm -> hold step waiting for the current stage

        def new_id(base: str) -> str:
            sid, n = base, 2
            while sid in used:
                sid, n = f"{base}_{n}", n + 1
            used.add(sid)
            return sid

        for k, stage in enumerate(stages):
            stage_ids: list[str] = []
            for i, skill in enumerate(stage):
                free = [a for a in skill.arms if a not in held]
                if not free:
                    continue
                hint = self.arm_hint(skill) if self.arm_hint else None
                arm = hint if hint in free else free[i % len(free)] if len(free) > 1 else free[0]
                sid = new_id(skill.name)
                deps = list(prev_stage)
                if arm in last_on_arm and last_on_arm[arm] not in deps:
                    deps.append(last_on_arm[arm])
                steps.append(Step(sid, arm, skill.name, deps, [], skill.success_check, skill.max_attempts))
                last_on_arm[arm] = sid
                stage_ids.append(sid)
            for arm, hold in list(held.items()):
                hold.hold_until = [sid for sid in stage_ids if self._arm_of(steps, sid) != arm]
                if not hold.hold_until:
                    steps.remove(hold)
                    used.discard(hold.id)
                    last_on_arm[arm] = hold.depends_on[0]
                del held[arm]
            if k + 1 < len(stages):
                for sid in stage_ids:
                    step = self._step(steps, sid)
                    if self.vocab[step.skill].hold_after and "hold" in self.vocab:
                        hold = Step(new_id(f"hold_{step.arm}"), step.arm, "hold", [sid], [], "", 1)
                        steps.append(hold)
                        held[step.arm] = hold
                        last_on_arm[step.arm] = hold.id
            prev_stage = stage_ids or prev_stage
        plan = Plan(steps, goal=goal, rationale="keyword stub planner", source=self.name)
        errors = validate_plan(plan, self.vocab, done)
        return PlanResult(None if errors else plan, errors, 1, self.name, time.perf_counter() - start)

    @staticmethod
    def _step(steps: list[Step], sid: str) -> Step:
        return next(s for s in steps if s.id == sid)

    def _arm_of(self, steps: list[Step], sid: str) -> str:
        return self._step(steps, sid).arm


SYSTEM_PROMPT = """You are the task planner for a two-arm robot station: an XLeRobot torso on a table with two SO-101 \
arms, "left" and "right" (the robot's own left and right), a fixed head camera above and behind them, and a camera on \
each gripper. In the head camera image the left arm is on the image's left side.

Each arm runs one skill at a time. FLUX skills are executed by a learned visuomotor policy that is conditioned on the \
skill's exact text prompt; the policy does not stop by itself, so every FLUX step needs a success_check: one short, \
concrete sentence a vision model can verify from the head camera image alone (objects, places, open or closed). \
Primitive skills are scripted; operator skills are done by a person while the arm waits, and the person confirms \
when they are done (they need no success_check). You choose skills only from the vocabulary below; you never write \
motor commands or new prompts.

How the plan runs: a step starts as soon as every step in its depends_on is done and its arm is free. Steps on \
different arms with no dependency between them run at the same time, which saves time, so put independent work on \
different arms when each arm can do its part (pick the arm closer to the object). Steps on the same arm must be \
ordered through depends_on. If one arm must keep something in place while the other arm works (for example holding \
a lid open), add a "hold" step on that arm right after it and list the other arm's steps in hold_until; the hold ends \
when they are done. Only hold steps use hold_until; for every other step it is []. Give a later step a dependency on \
anything it physically needs first (for example, close a lid only after everything is inside). max_attempts is 1 to \
3 (2 is a good default for FLUX steps, 1 for primitives). Use short readable ids.

If progress is given, plan only the remaining work: do not repeat finished steps. Finished step ids may appear in \
depends_on. If a step failed, think about why from the image and the reason given; you may retry it, use the other \
arm, or add a step that clears the way.

Keep the rationale to two or three sentences.

Skill vocabulary (JSON):
"""


class ClaudePlanner:
    name = "claude"

    def __init__(self, vocab: Vocabulary, llm: ClaudeJSON, model: str = DEFAULT_PLANNER_MODEL,
                 effort: str = "medium", feedback_rounds: int = 2):
        self.vocab = vocab
        self.llm = llm
        self.model = model
        self.effort = effort
        self.feedback_rounds = feedback_rounds
        self.schema = plan_json_schema(vocab)

    def system_prompt(self) -> str:
        station = self.vocab.station
        note = f"Station ({station.name}): {station.notes}\n\n" if station.notes else ""
        return SYSTEM_PROMPT + note + json.dumps(self.vocab.planner_view(), indent=1, sort_keys=True)

    def user_content(self, goal: str, image: np.ndarray | None, state: dict | None) -> list[dict]:
        content: list[dict] = []
        if image is not None:  # images before text work best (vision docs)
            content += [{"type": "text", "text": "Current head camera view:"}, image_block(image)]
        content.append({"type": "text", "text": f"Goal from the operator: {goal}"})
        if state:
            content.append({"type": "text", "text": "Progress so far (JSON):\n" + json.dumps(state, indent=1)})
        content.append({"type": "text", "text": "Return the plan."})
        return content

    def plan(self, goal: str, *, image: np.ndarray | None = None, state: dict | None = None) -> PlanResult:
        start = time.perf_counter()
        spent0 = self.llm.meter.total_usd
        done = _done_ids(state)
        messages: list[dict] = [{"role": "user", "content": self.user_content(goal, image, state)}]
        raw: list[Any] = []
        errors: list[str] = []
        attempts = 0
        for _ in range(1 + self.feedback_rounds):
            attempts += 1
            data, response = self.llm.call(purpose="plan", model=self.model, system=self.system_prompt(),
                                           messages=messages, schema=self.schema, effort=self.effort)
            raw.append(data)
            plan, errors = parse_plan(data, goal=goal, source=self.model)
            if plan is not None:
                fill_defaults(plan, self.vocab)
                errors = validate_plan(plan, self.vocab, done)
            if not errors:
                return PlanResult(plan, [], attempts, self.model, time.perf_counter() - start,
                                  self.llm.meter.total_usd - spent0, raw)
            # Append-only history: the full assistant content (thinking included), then the validator's verdict.
            messages += [
                {"role": "assistant", "content": response.content},
                {"role": "user", "content": "The validator rejected this plan:\n- " + "\n- ".join(errors)
                 + "\nReturn a corrected complete plan."},
            ]
        return PlanResult(None, errors, attempts, self.model, time.perf_counter() - start,
                          self.llm.meter.total_usd - spent0, raw)


def arm_by_side(positions: Callable[[], dict[str, np.ndarray]]) -> Callable[[Skill], str | None]:
    """Stub arm hint for the sim: the arm on the object's side of the robot (world +y is the robot's left)."""

    def hint(skill: Skill) -> str | None:
        obj = (skill.sim or {}).get("object") or ("lid" if (skill.sim or {}).get("action") in ("place_lid",
                                                                                               "remove_lid") else None)
        pos = positions().get(obj) if obj else None
        if pos is None:
            return None
        return ARMS[0] if pos[1] >= 0 else ARMS[1]

    return hint
