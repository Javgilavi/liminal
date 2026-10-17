"""Task files (YAML/JSON) -> behavior trees.

This format is the contract between *planners* (hand-written, LLM, Gemini ER, VLM,
bimanual obstacle planner) and the *executor*. Every planner emits this; one compiler
validates it. Grammar (each node is a single-key mapping):

    skill:    {arm, name, prompt?, target?, success?, timeout_s?, zones?, params?}
    sequence: [node, ...]                       # AFTER
    parallel: [node, ...]
    fallback: [node, ...]
    while:    {arm, maintain, until?, do: node} # WHILE / UNTIL
    retry:    {attempts?, do: node, recovery?: node}
    timeout:  {seconds, do: node}
    wait:     {condition, timeout_s?}           # UNTIL a condition
    check:    condition
    set:      {key, value?}

See configs/tasks/*.yaml for examples.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from liminal.core.types import Skill
from liminal.orchestration import bt


class TaskSpecError(ValueError):
    """Raised with a path-qualified message so an LLM planner can self-correct."""


@dataclass
class TaskSpec:
    name: str
    tree: bt.Node
    goal: str | None = None
    description: str = ""
    initial_state: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


_SKILL_FIELDS = {"arm", "name", "prompt", "target", "success", "timeout_s", "verify_timeout_s", "zones", "params"}


def load_task(path: str | Path, **kw: Any) -> TaskSpec:
    with open(path) as f:
        return parse_task(yaml.safe_load(f), **kw)


def parse_task(data: dict[str, Any], arms: Iterable[str] | None = None,
               skills: Iterable[str] | None = None) -> TaskSpec:
    if not isinstance(data, dict) or "tree" not in data:
        raise TaskSpecError("task must be a mapping with a 'tree' key")
    _fields(data, {"name", "goal", "description", "initial_state", "tree"}, "task")
    for key in ("name", "description", "goal"):
        if key in data and data[key] is not None and not isinstance(data[key], str):
            raise TaskSpecError(f"task.{key}: expected a string")
    initial = data.get("initial_state", {})
    if not isinstance(initial, dict) or any(not isinstance(k, str) for k in initial):
        raise TaskSpecError("task.initial_state: expected a mapping with string keys")
    compiler = _Compiler(set(arms) if arms is not None else None,
                         set(skills) if skills is not None else None)
    tree = compiler.node(data["tree"], "tree")
    return TaskSpec(
        name=data.get("name", "task"),
        tree=tree,
        goal=data.get("goal"),
        description=data.get("description", ""),
        initial_state=dict(initial),
        raw=data,
    )


def _fields(body: dict, allowed: set[str], path: str) -> None:
    unknown = set(body) - allowed
    if unknown:
        raise TaskSpecError(f"{path}: unknown fields {sorted(map(str, unknown))}")


def _positive(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TaskSpecError(f"{path}: expected a positive finite number")
    if not math.isfinite(value) or value <= 0:
        raise TaskSpecError(f"{path}: expected a positive finite number")
    return float(value)


def _condition(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip() or value == "!":
        raise TaskSpecError(f"{path}: expected a non-empty condition string")
    return value


_NODE_KINDS = frozenset({
    "skill", "sequence", "parallel", "fallback", "while", "retry", "timeout", "wait", "check", "set",
})


class _Compiler:
    def __init__(self, arms: set[str] | None, skills: set[str] | None):
        self.arms, self.skills = arms, skills
        self._depth = 0
        self._nodes = 0

    def node(self, d: Any, path: str) -> bt.Node:
        self._depth += 1
        self._nodes += 1
        try:
            if self._depth > 32 or self._nodes > 256:
                raise TaskSpecError(f"{path}: tree exceeds depth/node budget")
            return self._node(d, path)
        finally:
            self._depth -= 1

    def _node(self, d: Any, path: str) -> bt.Node:
        if not isinstance(d, dict):
            raise TaskSpecError(f"{path}: expected a node mapping, got {type(d).__name__}")
        d = dict(d)
        name = d.pop("name", None)
        if name is not None and not isinstance(name, str):
            raise TaskSpecError(f"{path}.name: expected a string")
        if len(d) != 1:
            raise TaskSpecError(f"{path}: node must have exactly one type key, got {sorted(map(str, d))}")
        kind, body = next(iter(d.items()))
        if kind not in _NODE_KINDS:
            raise TaskSpecError(f"{path}: unknown node type {kind!r}")
        fn = getattr(self, f"_{kind}", None)
        if fn is None:
            raise TaskSpecError(f"{path}: unknown node type {kind!r}")
        return fn(body, f"{path}.{kind}", name)

    def _children(self, body: Any, path: str) -> list[bt.Node]:
        if not isinstance(body, list) or not body:
            raise TaskSpecError(f"{path}: expected a non-empty list of nodes")
        return [self.node(c, f"{path}[{i}]") for i, c in enumerate(body)]

    def _arm(self, arm: Any, path: str) -> str:
        if not isinstance(arm, str):
            raise TaskSpecError(f"{path}: 'arm' is required")
        if self.arms is not None and arm not in self.arms:
            raise TaskSpecError(f"{path}: unknown arm {arm!r}; available: {sorted(self.arms)}")
        return arm

    def _skill(self, b: Any, path: str, name: str | None) -> bt.Node:
        if not isinstance(b, dict):
            raise TaskSpecError(f"{path}: expected a mapping")
        unknown = set(b) - _SKILL_FIELDS
        if unknown:
            raise TaskSpecError(f"{path}: unknown fields {sorted(unknown)}")
        if "name" not in b:
            raise TaskSpecError(f"{path}: 'name' is required")
        if not isinstance(b["name"], str) or not b["name"].strip():
            raise TaskSpecError(f"{path}.name: expected a non-empty string")
        if self.skills is not None and b["name"] not in self.skills:
            raise TaskSpecError(f"{path}: unknown skill {b['name']!r}; available: {sorted(self.skills)}")
        kw = {k: v for k, v in b.items() if k not in {"arm", "name", "zones"}}
        for field_name in ("prompt", "target", "success"):
            if field_name in b and (not isinstance(b[field_name], str) or not b[field_name].strip()):
                raise TaskSpecError(f"{path}.{field_name}: expected a non-empty string")
        for field_name in ("timeout_s", "verify_timeout_s"):
            if field_name in kw:
                kw[field_name] = _positive(kw[field_name], f"{path}.{field_name}")
        zones = b.get("zones", ())
        if not isinstance(zones, (list, tuple)) or any(not isinstance(z, str) or not z for z in zones):
            raise TaskSpecError(f"{path}.zones: expected a list of names")
        if "params" in b and (not isinstance(b["params"], dict)
                              or any(not isinstance(k, str) for k in b["params"])):
            raise TaskSpecError(f"{path}.params: expected a mapping with string keys")
        skill = Skill(arm=self._arm(b.get("arm"), path), name=b["name"], zones=tuple(b.get("zones", ())), **kw)
        return bt.SkillNode(skill, name=name)

    def _sequence(self, b: Any, path: str, name: str | None) -> bt.Node:
        return bt.Sequence(self._children(b, path), name=name)

    def _parallel(self, b: Any, path: str, name: str | None) -> bt.Node:
        return bt.Parallel(self._children(b, path), name=name)

    def _fallback(self, b: Any, path: str, name: str | None) -> bt.Node:
        return bt.Fallback(self._children(b, path), name=name)

    def _while(self, b: Any, path: str, name: str | None) -> bt.Node:
        if not isinstance(b, dict) or "maintain" not in b or "do" not in b:
            raise TaskSpecError(f"{path}: 'while' needs arm, maintain and do")
        _fields(b, {"arm", "maintain", "do", "until", "establish_timeout_s"}, path)
        return bt.While(
            support_arm=self._arm(b.get("arm"), path),
            maintain=_condition(b["maintain"], f"{path}.maintain"),
            until=_condition(b["until"], f"{path}.until") if "until" in b else None,
            body=self.node(b["do"], f"{path}.do"),
            establish_timeout_s=_positive(b.get("establish_timeout_s", 5.0), f"{path}.establish_timeout_s"),
            name=name,
        )

    def _retry(self, b: Any, path: str, name: str | None) -> bt.Node:
        if not isinstance(b, dict) or "do" not in b:
            raise TaskSpecError(f"{path}: 'retry' needs do")
        _fields(b, {"attempts", "do", "recovery"}, path)
        attempts = b.get("attempts", 2)
        if isinstance(attempts, bool) or not isinstance(attempts, int) or not 1 <= attempts <= 100:
            raise TaskSpecError(f"{path}.attempts: expected an integer from 1 to 100")
        recovery = self.node(b["recovery"], f"{path}.recovery") if "recovery" in b else None
        return bt.Retry(self.node(b["do"], f"{path}.do"), attempts=attempts,
                        recovery=recovery, name=name)

    def _timeout(self, b: Any, path: str, name: str | None) -> bt.Node:
        if not isinstance(b, dict) or "seconds" not in b or "do" not in b:
            raise TaskSpecError(f"{path}: 'timeout' needs seconds and do")
        _fields(b, {"seconds", "do"}, path)
        return bt.Timeout(self.node(b["do"], f"{path}.do"), _positive(b["seconds"], f"{path}.seconds"), name=name)

    def _wait(self, b: Any, path: str, name: str | None) -> bt.Node:
        if isinstance(b, str):
            return bt.WaitFor(_condition(b, path), name=name)
        if not isinstance(b, dict) or "condition" not in b:
            raise TaskSpecError(f"{path}: 'wait' needs condition")
        _fields(b, {"condition", "timeout_s"}, path)
        timeout = b.get("timeout_s", 30.0)
        if timeout is not None:
            timeout = _positive(timeout, f"{path}.timeout_s")
        return bt.WaitFor(_condition(b["condition"], path), timeout_s=timeout, name=name)

    def _check(self, b: Any, path: str, name: str | None) -> bt.Node:
        return bt.Check(_condition(b, path), name=name)

    def _set(self, b: Any, path: str, name: str | None) -> bt.Node:
        if not isinstance(b, dict) or "key" not in b:
            raise TaskSpecError(f"{path}: 'set' needs key")
        _fields(b, {"key", "value"}, path)
        if not isinstance(b["key"], str) or not b["key"]:
            raise TaskSpecError(f"{path}.key: expected a non-empty string")
        return bt.SetFact(b["key"], b.get("value", True), name=name)
