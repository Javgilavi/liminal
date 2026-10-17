"""Run every condition x scenario x repeat in its own tmp db and log dir, then score and write results."""

from __future__ import annotations

import asyncio
import inspect
import json
import random
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from liminal.eval.metrics import RunMetrics, aggregate, score_run
from liminal.eval.spec import ConditionSpec, Experiment, ExperimentError, ScenarioSpec, load_experiment

LABEL = "REPLAY/MOCK, not hardware"
Execute = Callable[[Any], Awaitable[Any]]


@dataclass
class ExperimentResult:
    experiment: str
    description: str
    label: str
    budgets: dict[str, int]
    repeats: int
    seeds: list[int]
    runs: list[dict[str, Any]] = field(default_factory=list)  # one per attempted run
    skipped: list[dict[str, Any]] = field(default_factory=list)  # one per skipped cell (never in denominators)
    conditions: dict[str, dict[str, Any]] = field(default_factory=dict)
    by_scenario: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)
    out_dir: str = ""
    results_path: str = ""
    report_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in (
            "experiment", "description", "label", "budgets", "repeats", "seeds", "conditions", "by_scenario",
            "runs", "skipped")}


def unavailable_memory_options(memory: dict[str, Any]) -> list[str]:
    """Memory kwargs that the current ``run_episode`` does not accept (the memory session adds them)."""
    from liminal.orchestration.episode import run_episode

    params = inspect.signature(run_episode).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return []
    return sorted(k for k in memory if k not in params)


def _skip_reason(cond: ConditionSpec, scen: ScenarioSpec) -> str | None:
    if cond.mode == "planner":
        missing = unavailable_memory_options(cond.memory)
        if missing:
            return f"skipped: option unavailable ({', '.join(missing)})"
    if cond.mode == "authored":
        if scen.interrupt_after is not None:
            return "skipped: authored runs have no ledger, so no interrupt/resume"
        if not scen.goal and not scen.task:
            return "skipped: authored mode needs a goal or task"
    return None


def _read_trace(log_dir: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for path in sorted(log_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(json.loads(line))
    return events


def _final_facts(scene: Any, obligations: tuple[str, ...]) -> dict[str, bool | None]:
    """Fresh observation of the replay scene after the run: the independent referee."""
    from liminal.perception.evidence import Truth

    facts = scene.capture().facts
    return {o: (True if facts.get(o) is Truth.TRUE else False if facts.get(o) is Truth.FALSE else None)
            for o in obligations}


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", text).strip("-")


def run_cell(exp: Experiment, cond: ConditionSpec, scen: ScenarioSpec, repeat: int, seed: int, work: Path,
             execute: Execute | None = None) -> RunMetrics:
    """One run (two sessions when the scenario interrupts and resumes). Never raises: errors are scored."""
    from liminal.orchestration import service

    execute = execute or service.execute
    work.mkdir(parents=True, exist_ok=True)
    log_dir = work / "logs"
    run_id = f"{_slug(cond.name)}-{_slug(scen.name)}-r{repeat}-s{seed}"
    common: dict[str, Any] = dict(
        scenario=scen.scenario, mode=cond.mode, model=cond.model, goal=scen.goal, task=scen.task,
        db=str(work / "ledger.sqlite3"), run_id=run_id, fail_first=scen.fail_first, lie_first=scen.lie_first,
        max_attempts=exp.max_attempts, max_model_calls=exp.max_model_calls, log_dir=str(log_dir),
        scene_state=str(work / "scene.json"),
        memory={k: v.replace("{work}", str(work)) if isinstance(v, str) else v for k, v in cond.memory.items()})
    if cond.robot:
        common["robot"] = cond.robot
    started = time.monotonic()
    report, prepared, error = None, None, None
    try:
        sessions = [service.RunOptions(**common, interrupt_after=scen.interrupt_after)]
        if scen.interrupt_after is not None:
            sessions.append(service.RunOptions(**common, resume=True))
        for i, options in enumerate(sessions):
            prepared = service.prepare(options)
            report = asyncio.run(execute(prepared))
            if report.completed or i == len(sessions) - 1:
                break
            # interrupted: the second session resumes the same run, ledger and scene
    except Exception as exc:  # noqa: BLE001 - a crashed run is a failed run in the denominator
        error = f"{type(exc).__name__}: {exc}"
        report = None
    wall = time.monotonic() - started
    obligations = tuple(prepared.scenario.goal.obligations) if prepared is not None else ()
    facts = _final_facts(prepared.scene, obligations) if prepared is not None else {}
    trace = _read_trace(log_dir) if log_dir.exists() else []
    return score_run(report, facts, trace, obligations, wall_s=wall, error=error)


def run_experiment(path: str | Path, out_dir: str | Path, *, execute: Execute | None = None) -> ExperimentResult:
    exp = load_experiment(path)  # refuses unmatched budgets
    paid = [c.name for c in exp.conditions if c.model != "replay"]
    if paid and not exp.allow_paid:
        raise ExperimentError(f"conditions {paid} use a paid model; set allow_paid: true to run them")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cells = [(c, s, r, seed) for c in exp.conditions for s in exp.scenarios for r, seed in enumerate(exp.seeds)]
    random.Random(exp.seeds[0]).shuffle(cells)  # counterbalance run order, reproducibly
    result = ExperimentResult(exp.name, exp.description, LABEL, {"max_attempts": exp.max_attempts,
                                                                  "max_model_calls": exp.max_model_calls},
                              exp.repeats, list(exp.seeds), out_dir=str(out))
    metrics: dict[str, list[RunMetrics]] = {c.name: [] for c in exp.conditions}
    per_scen: dict[str, dict[str, list[RunMetrics]]] = {c.name: {s.name: [] for s in exp.scenarios}
                                                         for c in exp.conditions}
    skipped_seen: set[tuple[str, str]] = set()
    for cond, scen, repeat, seed in cells:
        reason = _skip_reason(cond, scen)
        if reason:
            if (cond.name, scen.name) not in skipped_seen:
                skipped_seen.add((cond.name, scen.name))
                result.skipped.append({"condition": cond.name, "scenario": scen.name, "reason": reason})
            continue
        work = out / "runs" / f"{_slug(cond.name)}__{_slug(scen.name)}__r{repeat}"
        m = run_cell(exp, cond, scen, repeat, seed, work, execute)
        metrics[cond.name].append(m)
        per_scen[cond.name][scen.name].append(m)
        result.runs.append({"condition": cond.name, "scenario": scen.name, "repeat": repeat, "seed": seed,
                            "work_dir": str(work), **m.to_dict()})
    result.conditions = {name: aggregate(runs) for name, runs in metrics.items()}
    result.by_scenario = {c: {s: aggregate(r) for s, r in d.items() if r} for c, d in per_scen.items()}
    from liminal.eval.report import render_report

    result.results_path = str(out / "results.json")
    result.report_path = str(out / "report.md")
    Path(result.results_path).write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    Path(result.report_path).write_text(render_report(result), encoding="utf-8")
    return result
