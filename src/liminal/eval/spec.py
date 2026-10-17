"""Experiment files: scenarios x conditions x repeats under one matched budget.

    name: orchestration
    budgets: {max_attempts: 6, max_model_calls: 20}   # identical for every condition, or refuse
    repeats: 2
    seeds: [1, 2]                                      # optional; one per repeat
    scenarios:
      - {name: two_blocks, scenario: configs/replay/workbench.yaml, goal: "...",
         fail_first: false, lie_first: false, interrupt_after: 1}
    conditions:
      - {name: authored, mode: authored}
      - {name: planner, mode: planner, model: replay, memory: {keyframes: event}}

Scenarios own the injections (so every condition sees the same failures); conditions own only
mode / model / memory / robot. Memory string values may use "{work}" for the run's own tmp dir
(e.g. experience_db: "{work}/experience.sqlite3"), so no run shares state with another.
A condition that sets max_attempts or max_model_calls to a value different from ``budgets`` makes the
whole experiment unmatched and is refused.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BUDGET_KEYS = ("max_attempts", "max_model_calls")
SCENARIO_KEYS = {"name", "scenario", "goal", "task", "fail_first", "lie_first", "interrupt_after"}
CONDITION_KEYS = {"name", "mode", "model", "memory", "robot", *BUDGET_KEYS}


class ExperimentError(ValueError):
    """The experiment file is malformed or would not be a fair comparison."""


class UnmatchedBudgets(ExperimentError):
    pass


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    scenario: str
    goal: str | None = None
    task: str | None = None
    fail_first: bool = False
    lie_first: bool = False
    interrupt_after: int | None = None


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    mode: str = "planner"
    model: str = "replay"
    memory: dict[str, Any] = field(default_factory=dict)
    robot: str | None = None


@dataclass(frozen=True)
class Experiment:
    name: str
    description: str
    max_attempts: int
    max_model_calls: int
    repeats: int
    seeds: tuple[int, ...]
    scenarios: tuple[ScenarioSpec, ...]
    conditions: tuple[ConditionSpec, ...]
    allow_paid: bool = False


def _unique_names(items: list[Any], what: str) -> None:
    names = [i.name for i in items]
    if len(set(names)) != len(names):
        raise ExperimentError(f"duplicate {what} names: {sorted(n for n in set(names) if names.count(n) > 1)}")


def _check_keys(raw: dict[str, Any], allowed: set[str], where: str) -> None:
    extra = set(raw) - allowed
    if extra:
        raise ExperimentError(f"{where}: unknown fields {sorted(extra)}")


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ExperimentError(f"{where} must be a positive integer, got {value!r}")
    return value


def parse_experiment(raw: Any, default_name: str = "experiment") -> Experiment:
    if not isinstance(raw, dict):
        raise ExperimentError("experiment file must be a mapping")
    _check_keys(raw, {"name", "description", "budgets", "repeats", "seeds", "scenarios", "conditions",
                      "allow_paid"}, "experiment")
    budgets = raw.get("budgets")
    if not isinstance(budgets, dict) or set(budgets) != set(BUDGET_KEYS):
        raise ExperimentError(f"budgets must set exactly {list(BUDGET_KEYS)}")
    attempts = _positive_int(budgets["max_attempts"], "budgets.max_attempts")
    calls = _positive_int(budgets["max_model_calls"], "budgets.max_model_calls")
    repeats = _positive_int(raw.get("repeats", 1), "repeats")
    seeds = tuple(raw.get("seeds") or range(1, repeats + 1))
    if len(seeds) != repeats or not all(isinstance(s, int) and not isinstance(s, bool) for s in seeds):
        raise ExperimentError("seeds must be a list of integers, one per repeat")

    scenarios = []
    for i, s in enumerate(raw.get("scenarios") or []):
        if not isinstance(s, dict):
            raise ExperimentError(f"scenarios[{i}] must be a mapping")
        _check_keys(s, SCENARIO_KEYS, f"scenarios[{i}]")
        if not s.get("name") or not s.get("scenario"):
            raise ExperimentError(f"scenarios[{i}] needs name and scenario")
        after = s.get("interrupt_after")
        if after is not None:
            _positive_int(after, f"scenarios[{i}].interrupt_after")
        scenarios.append(ScenarioSpec(str(s["name"]), str(s["scenario"]), s.get("goal"), s.get("task"),
                                      bool(s.get("fail_first", False)), bool(s.get("lie_first", False)), after))
    conditions = []
    for i, c in enumerate(raw.get("conditions") or []):
        if not isinstance(c, dict):
            raise ExperimentError(f"conditions[{i}] must be a mapping")
        _check_keys(c, CONDITION_KEYS, f"conditions[{i}]")
        if not c.get("name"):
            raise ExperimentError(f"conditions[{i}] needs a name")
        for key in BUDGET_KEYS:
            if key in c and c[key] != budgets[key]:
                raise UnmatchedBudgets(
                    f"condition {c['name']!r} sets {key}={c[key]!r} but the matched budget is {budgets[key]!r}; "
                    "budgets must be identical across conditions")
        memory = c.get("memory") or {}
        if not isinstance(memory, dict):
            raise ExperimentError(f"conditions[{i}].memory must be a mapping")
        conditions.append(ConditionSpec(str(c["name"]), str(c.get("mode", "planner")), str(c.get("model", "replay")),
                                        dict(memory), c.get("robot")))
    if not scenarios or not conditions:
        raise ExperimentError("an experiment needs at least one scenario and one condition")
    _unique_names(scenarios, "scenario")
    _unique_names(conditions, "condition")
    return Experiment(str(raw.get("name") or default_name), str(raw.get("description") or "").strip(), attempts,
                      calls, repeats, seeds, tuple(scenarios), tuple(conditions), bool(raw.get("allow_paid", False)))


def load_experiment(path: str | Path) -> Experiment:
    import yaml

    p = Path(path)
    return parse_experiment(yaml.safe_load(p.read_text(encoding="utf-8")), p.stem)
