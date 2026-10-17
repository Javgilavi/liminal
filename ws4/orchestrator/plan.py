"""Plan format, JSON schema and validation. The same JSON is what WS3's planner can emit (ws4/research/orchestrator.md).

A plan is a list of steps. Each step runs one vocabulary skill on one arm. ``depends_on`` orders steps; steps whose
dependencies are done start at once, one step per arm at a time, so independent steps on the two arms run
concurrently. ``hold`` keeps an arm at its last command until every step in ``hold_until`` (on the other arm) is done.

Validation rejects what the executor could not run safely or could deadlock on:
schema, unknown skills, an arm the skill does not allow, unknown or cyclic dependencies, two steps on one arm with no
order between them, and holds whose release depends on work queued behind the hold.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .skills import ARMS, Vocabulary

MAX_ATTEMPTS = 3
MAX_STEPS = 24
ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,40}$")
STEP_KEYS = ("id", "arm", "skill", "depends_on", "hold_until", "success_check", "max_attempts")


@dataclass
class Step:
    id: str
    arm: str
    skill: str
    depends_on: list[str] = field(default_factory=list)
    hold_until: list[str] = field(default_factory=list)
    success_check: str = ""
    max_attempts: int = 1


@dataclass
class Plan:
    steps: list[Step]
    goal: str = ""
    rationale: str = ""
    source: str = ""
    version: int = 1

    def step(self, step_id: str) -> Step:
        for s in self.steps:
            if s.id == step_id:
                return s
        raise KeyError(step_id)

    @property
    def ids(self) -> list[str]:
        return [s.id for s in self.steps]

    def to_json(self) -> dict[str, Any]:
        return {"goal": self.goal, "rationale": self.rationale, "source": self.source, "version": self.version,
                "steps": [asdict(s) for s in self.steps]}

    def dumps(self) -> str:
        return json.dumps(self.to_json(), indent=2)


def plan_json_schema(vocab: Vocabulary | None = None) -> dict[str, Any]:
    """JSON schema for structured output (and for other planners). Every object is closed and lists every key as
    required, as structured outputs expect. Numeric bounds are checked by :func:`validate_plan` instead."""
    skill: dict[str, Any] = {"type": "string"}
    if vocab is not None:
        skill["enum"] = vocab.names()
    ids = {"type": "array", "items": {"type": "string"}}
    return {
        "type": "object",
        "properties": {
            "rationale": {"type": "string"},
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "arm": {"type": "string", "enum": list(ARMS)},
                        "skill": skill,
                        "depends_on": ids,
                        "hold_until": ids,
                        "success_check": {"type": "string"},
                        "max_attempts": {"type": "integer"},
                    },
                    "required": list(STEP_KEYS),
                    "additionalProperties": False,
                },
            },
        },
        "required": ["rationale", "steps"],
        "additionalProperties": False,
    }


def parse_plan(data: dict | str, goal: str = "", source: str = "") -> tuple[Plan | None, list[str]]:
    """Structural check of raw planner output. Returns the plan (or None) and schema errors."""
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError as exc:
            return None, [f"not valid JSON: {exc}"]
    if not isinstance(data, dict):
        return None, ["top level must be an object with 'steps'"]
    raw_steps = data.get("steps")
    if not isinstance(raw_steps, list):
        return None, ["'steps' must be a list"]
    errors: list[str] = []
    steps: list[Step] = []
    for i, raw in enumerate(raw_steps):
        where = f"steps[{i}]"
        if not isinstance(raw, dict):
            errors.append(f"{where} must be an object")
            continue
        extra = sorted(set(raw) - set(STEP_KEYS))
        if extra:
            errors.append(f"{where} has unknown keys {extra}")
        for key in ("id", "arm", "skill"):
            if not isinstance(raw.get(key), str) or not raw.get(key):
                errors.append(f"{where}.{key} must be a non-empty string")
        lists = {}
        for key in ("depends_on", "hold_until"):
            value = raw.get(key, [])
            if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                errors.append(f"{where}.{key} must be a list of step ids")
                value = []
            lists[key] = list(value)
        attempts = raw.get("max_attempts", 1)
        if isinstance(attempts, bool) or not isinstance(attempts, int):
            errors.append(f"{where}.max_attempts must be an integer")
            attempts = 1
        check = raw.get("success_check", "")
        if not isinstance(check, str):
            errors.append(f"{where}.success_check must be a string")
            check = ""
        steps.append(Step(str(raw.get("id", "")), str(raw.get("arm", "")), str(raw.get("skill", "")),
                          lists["depends_on"], lists["hold_until"], check.strip(), attempts))
    rationale = data.get("rationale", "")
    plan = Plan(steps, goal=goal or str(data.get("goal", "")), rationale=str(rationale), source=source,
                version=int(data.get("version", 1)) if isinstance(data.get("version", 1), int) else 1)
    return (None if errors else plan), errors


def fill_defaults(plan: Plan, vocab: Vocabulary) -> Plan:
    """Default success checks from the vocabulary; cap attempts at the skill's default when 0 or missing."""
    for s in plan.steps:
        skill = vocab.skills.get(s.skill)
        if skill is None:
            continue
        if not s.success_check and skill.is_flux:
            s.success_check = skill.success_check
        if s.max_attempts == 0:
            s.max_attempts = skill.max_attempts
    return plan


def _ancestors(plan: Plan, known: set[str]) -> dict[str, set[str]]:
    deps = {s.id: [d for d in s.depends_on if d in known] for s in plan.steps}
    memo: dict[str, set[str]] = {}

    def visit(node: str, stack: tuple[str, ...]) -> set[str]:
        if node in memo:
            return memo[node]
        if node in stack:  # cycles are reported separately
            return set()
        out: set[str] = set()
        for d in deps.get(node, []):
            out.add(d)
            out |= visit(d, stack + (node,))
        memo[node] = out
        return out

    return {s.id: visit(s.id, ()) for s in plan.steps}


def _find_cycle(edges: dict[str, list[str]]) -> list[str] | None:
    color: dict[str, int] = {}
    path: list[str] = []

    def dfs(node: str) -> list[str] | None:
        color[node] = 1
        path.append(node)
        for nxt in edges.get(node, []):
            if color.get(nxt, 0) == 1:
                return path[path.index(nxt):] + [nxt]
            if color.get(nxt, 0) == 0:
                found = dfs(nxt)
                if found:
                    return found
        path.pop()
        color[node] = 2
        return None

    for node in list(edges):
        if color.get(node, 0) == 0:
            found = dfs(node)
            if found:
                return found
    return None


def validate_plan(plan: Plan, vocab: Vocabulary, done: set[str] | frozenset[str] = frozenset()) -> list[str]:
    """Semantic errors, phrased so they can be fed back to the planner verbatim. ``done`` are ids of steps that
    already finished (when replanning); new steps may depend on them."""
    errors: list[str] = []
    if not plan.steps:
        return ["the plan has no steps"]
    if len(plan.steps) > MAX_STEPS:
        errors.append(f"the plan has {len(plan.steps)} steps; the limit is {MAX_STEPS}")
    ids = [s.id for s in plan.steps]
    seen: set[str] = set()
    for i in ids:
        if not ID_RE.match(i):
            errors.append(f"step id {i!r} must be 1-40 characters of letters, digits, '_', '-' or '.'")
        if i in seen:
            errors.append(f"step id {i!r} is used twice")
        if i in done:
            errors.append(f"step id {i!r} is already used by a finished step; pick a new id")
        seen.add(i)
    known = set(ids)
    by_id = {s.id: s for s in plan.steps}
    for s in plan.steps:
        where = f"step {s.id!r}"
        skill = vocab.skills.get(s.skill)
        if skill is None:
            errors.append(f"{where}: skill {s.skill!r} is not in the vocabulary ({', '.join(vocab.names())})")
        elif s.arm not in skill.arms:
            errors.append(f"{where}: skill {s.skill!r} can only run on {list(skill.arms)}, not on {s.arm!r}")
        if s.arm not in ARMS:
            errors.append(f"{where}: arm must be one of {list(ARMS)}")
        if not 1 <= s.max_attempts <= MAX_ATTEMPTS:
            errors.append(f"{where}: max_attempts must be between 1 and {MAX_ATTEMPTS}")
        if len(set(s.depends_on)) != len(s.depends_on):
            errors.append(f"{where}: depends_on lists a step twice")
        for d in s.depends_on:
            if d == s.id:
                errors.append(f"{where} depends on itself")
            elif d not in known and d not in done:
                errors.append(f"{where} depends on unknown step {d!r}")
        if skill is not None and skill.is_flux and not s.success_check:
            errors.append(f"{where}: a FLUX step needs a success_check the camera can verify")
        if s.skill == "hold":
            if not s.hold_until:
                errors.append(f"{where}: a hold step must list the steps it waits for in hold_until")
            for u in s.hold_until:
                if u not in known:
                    errors.append(f"{where}: hold_until names unknown step {u!r}"
                                  + (" (already finished)" if u in done else ""))
                elif by_id[u].arm == s.arm:
                    errors.append(f"{where}: hold_until step {u!r} is on the same arm; a held arm cannot run it")
                elif u == s.id:
                    errors.append(f"{where}: hold_until names itself")
        elif s.hold_until:
            errors.append(f"{where}: only hold steps may have hold_until")
    if errors:
        return errors

    cycle = _find_cycle({s.id: [d for d in s.depends_on if d in known] for s in plan.steps})
    if cycle:
        return [f"dependency cycle: {' -> '.join(cycle)}"]
    anc = _ancestors(plan, known)
    for arm in ARMS:
        on_arm = [s.id for s in plan.steps if s.arm == arm]
        for i, a in enumerate(on_arm):
            for b in on_arm[i + 1:]:
                if a not in anc[b] and b not in anc[a]:
                    errors.append(f"steps {a!r} and {b!r} both use the {arm} arm but neither depends on the other; "
                                  "add a depends_on entry to order them")
    # Event graph: a step starts after its dependencies end; a hold ends after its hold_until steps end.
    edges: dict[str, list[str]] = {}
    for s in plan.steps:
        edges.setdefault(f"start:{s.id}", []).append(f"end:{s.id}")
        for d in s.depends_on:
            if d in known:
                edges.setdefault(f"end:{d}", []).append(f"start:{s.id}")
        for u in s.hold_until:
            edges.setdefault(f"end:{u}", []).append(f"end:{s.id}")
            if u in anc[s.id]:
                errors.append(f"step {s.id!r}: hold_until step {u!r} finishes before the hold starts, so the hold "
                              "would end at once")
    cycle = _find_cycle(edges)
    if cycle:
        names = [c.split(":", 1)[1] for c in cycle if c.startswith("end:")]
        errors.append("deadlock: a hold waits for work that can only start after the hold ends "
                      f"({' -> '.join(dict.fromkeys(names))})")
    return errors


def lanes(plan: Plan) -> dict[str, list[str]]:
    return {arm: [s.id for s in plan.steps if s.arm == arm] for arm in ARMS}


def levels(plan: Plan) -> dict[str, int]:
    """Earliest start row of each step (longest dependency chain), for drawing lanes side by side."""
    known = set(plan.ids)
    out: dict[str, int] = {}

    def level(sid: str, stack: tuple[str, ...] = ()) -> int:
        if sid in out:
            return out[sid]
        if sid in stack:
            return 0
        deps = [d for d in plan.step(sid).depends_on if d in known]
        out[sid] = 1 + max((level(d, stack + (sid,)) for d in deps), default=-1)
        return out[sid]

    for s in plan.steps:
        level(s.id)
    return out


def concurrent_pairs(plan: Plan) -> list[tuple[str, str]]:
    """Pairs of steps on different arms with no ordering between them: these may run at the same time."""
    known = set(plan.ids)
    anc = _ancestors(plan, known)
    left = [s.id for s in plan.steps if s.arm == "left"]
    right = [s.id for s in plan.steps if s.arm == "right"]
    return [(a, b) for a in left for b in right if a not in anc[b] and b not in anc[a]]
