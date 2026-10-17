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
    compiler = _Compiler(set(arms) if arms else None, set(skills) if skills else None)
    tree = compiler.node(data["tree"], "tree")
    return TaskSpec(
        name=data.get("name", "task"),
        tree=tree,
        goal=data.get("goal"),
        description=data.get("description", ""),
        initial_state=dict(data.get("initial_state", {})),
        raw=data,
    )


class _Compiler:
    def __init__(self, arms: set[str] | None, skills: set[str] | None):
        self.arms, self.skills = arms, skills

    def node(self, d: Any, path: str) -> bt.Node:
        if not isinstance(d, dict):
            raise TaskSpecError(f"{path}: expected a node mapping, got {type(d).__name__}")
        d = dict(d)
        name = d.pop("name", None)
        if len(d) != 1:
            raise TaskSpecError(f"{path}: node must have exactly one type key, got {sorted(d)}")
        kind, body = next(iter(d.items()))
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
        if self.skills is not None and b["name"] not in self.skills:
            raise TaskSpecError(f"{path}: unknown skill {b['name']!r}; available: {sorted(self.skills)}")
        kw = {k: v for k, v in b.items() if k not in {"arm", "name", "zones"}}
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
        return bt.While(
            support_arm=self._arm(b.get("arm"), path),
            maintain=b["maintain"],
            until=b.get("until"),
            body=self.node(b["do"], f"{path}.do"),
            establish_timeout_s=float(b.get("establish_timeout_s", 5.0)),
            name=name,
        )

    def _retry(self, b: Any, path: str, name: str | None) -> bt.Node:
        if not isinstance(b, dict) or "do" not in b:
            raise TaskSpecError(f"{path}: 'retry' needs do")
        recovery = self.node(b["recovery"], f"{path}.recovery") if "recovery" in b else None
        return bt.Retry(self.node(b["do"], f"{path}.do"), attempts=int(b.get("attempts", 2)),
                        recovery=recovery, name=name)

    def _timeout(self, b: Any, path: str, name: str | None) -> bt.Node:
        if not isinstance(b, dict) or "seconds" not in b or "do" not in b:
            raise TaskSpecError(f"{path}: 'timeout' needs seconds and do")
        return bt.Timeout(self.node(b["do"], f"{path}.do"), float(b["seconds"]), name=name)

    def _wait(self, b: Any, path: str, name: str | None) -> bt.Node:
        if isinstance(b, str):
            return bt.WaitFor(b, name=name)
        if not isinstance(b, dict) or "condition" not in b:
            raise TaskSpecError(f"{path}: 'wait' needs condition")
        return bt.WaitFor(b["condition"], timeout_s=b.get("timeout_s", 30.0), name=name)

    def _check(self, b: Any, path: str, name: str | None) -> bt.Node:
        if not isinstance(b, str):
            raise TaskSpecError(f"{path}: 'check' takes a condition string")
        return bt.Check(b, name=name)

    def _set(self, b: Any, path: str, name: str | None) -> bt.Node:
        if not isinstance(b, dict) or "key" not in b:
            raise TaskSpecError(f"{path}: 'set' needs key")
        return bt.SetFact(b["key"], b.get("value", True), name=name)
