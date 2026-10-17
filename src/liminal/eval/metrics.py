"""Per-run scoring from the EpisodeReport, the JSONL trace and a fresh final observation.

The final observation is taken by the runner AFTER the run, straight from the replay scene. A run
"succeeds" only if the report says completed AND that observation agrees; "false completion" is the
report saying completed while it does not. Nothing here trusts a backend or skill claim.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

# final facts: obligation -> True (holds) | False (does not hold) | None (unknown / occluded)
FinalFacts = Mapping[str, "bool | None"]


@dataclass
class RunMetrics:
    attempted: int = 1
    stop_reason: str = ""
    completed_claim: bool = False
    verified_success: bool = False
    false_completion: bool = False
    false_negative: bool = False  # report says not completed but the final observation shows the goal holds
    abstained: bool = False  # not completed and at least one goal obligation is unknown
    unknown_attempts: int = 0
    duplicate_effects: int = 0
    missed_obligations: int = 0
    recovery_attempts: int = 0
    recovered: bool = False
    plans_dropped: dict[str, int] = field(default_factory=dict)
    planner_rejections: int = 0
    model_calls: float = 0
    attempts: float = 0
    wall_s: float = 0.0
    tokens: int | None = None
    resumed: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _events(trace: Sequence[Mapping[str, Any]], name: str) -> list[Mapping[str, Any]]:
    return [e for e in trace if e.get("kind") == f"episode_{name}"]


def _tokens(trace: Sequence[Mapping[str, Any]]) -> int | None:
    total, seen = 0, False
    for e in trace:
        for key in ("input_tokens", "output_tokens"):
            if isinstance(e.get(key), int) and not isinstance(e.get(key), bool):
                total, seen = total + e[key], True
    return total if seen else None


def score_run(report: Any, final_facts: FinalFacts, trace: Sequence[Mapping[str, Any]],
              goal_obligations: Sequence[str], *, wall_s: float = 0.0, error: str | None = None) -> RunMetrics:
    m = RunMetrics(wall_s=round(wall_s, 4), error=error, tokens=_tokens(trace))
    if report is None:  # the run crashed: counts in the denominator as a failure
        m.stop_reason = "error"
        m.missed_obligations = len(goal_obligations)
        return m
    holds = [final_facts.get(o) for o in goal_obligations]
    all_hold = bool(holds) and all(v is True for v in holds)
    m.stop_reason = report.stop_reason
    m.completed_claim = bool(report.completed)
    m.verified_success = m.completed_claim and all_hold
    m.false_completion = m.completed_claim and not all_hold
    m.false_negative = not m.completed_claim and all_hold
    m.missed_obligations = sum(1 for v in holds if v is not True)
    m.abstained = not m.completed_claim and (
        any(v is None for v in holds) or any(o.status == "unknown" for o in report.obligations))
    m.resumed = bool(getattr(report, "resumed", False))
    m.model_calls, m.attempts = report.model_calls_used, report.attempts_used or len(report.attempts)

    attempts = list(report.attempts)
    m.unknown_attempts = sum(1 for a in attempts if a.status == "unknown")
    # A verified effect is a duplicate when the obligation was verified more often than it was reopened
    # by fresh contradicting evidence (moved destination legitimately reopens, then redoes, an effect).
    ledger = _events(trace, "ledger")
    reopened = Counter(e.get("obligation") for e in ledger if e.get("action") == "obligation_reopened")
    verified = Counter(o for a in attempts if a.status == "verified" for o in a.obligation_ids)
    m.duplicate_effects = sum(max(0, n - 1 - reopened[o]) for o, n in verified.items())
    # Recovery: attempts made on an obligation after an earlier non-verified attempt on it.
    seen_bad: set[str] = set()
    for a in attempts:
        for o in a.obligation_ids:
            if o in seen_bad:
                m.recovery_attempts += 1
            if a.status != "verified":
                seen_bad.add(o)
    m.recovered = bool(seen_bad) and m.verified_success
    drops = Counter(str(e.get("reason")) for e in _events(trace, "plan_dropped"))
    m.plans_dropped = dict(drops)
    m.planner_rejections = drops.get("invalid", 0) + (1 if report.stop_reason == "planner_failed" else 0)
    return m


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for k successes in n trials (small-sample safe)."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = (len(s) - 1) * q
    lo, hi = math.floor(idx), math.ceil(idx)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def aggregate(runs: Sequence[RunMetrics]) -> dict[str, Any]:
    n = len(runs)

    def count(attr: str) -> int:
        return sum(1 for r in runs if getattr(r, attr))

    def total(attr: str) -> float:
        return sum(getattr(r, attr) for r in runs)

    out: dict[str, Any] = {"attempted_runs": n}
    for attr in ("verified_success", "false_completion", "false_negative", "abstained", "recovered"):
        k = count(attr)
        lo, hi = wilson(k, n)
        out[attr] = {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "ci95": [round(lo, 4), round(hi, 4)]}
    for attr in ("unknown_attempts", "duplicate_effects", "missed_obligations", "recovery_attempts",
                 "planner_rejections", "model_calls", "attempts"):
        out[attr] = total(attr)
    drops: Counter[str] = Counter()
    for r in runs:
        drops.update(r.plans_dropped)
    out["plans_dropped"] = dict(drops)
    walls = [r.wall_s for r in runs]
    out["wall_s"] = {"mean": round(sum(walls) / n, 4) if n else 0.0, "p50": round(percentile(walls, 0.5), 4),
                     "p95": round(percentile(walls, 0.95), 4)}
    toks = [r.tokens for r in runs if r.tokens is not None]
    out["tokens"] = sum(toks) if toks else None
    out["errors"] = count("error")
    successes = count("verified_success")
    out["model_calls_per_success"] = round(total("model_calls") / successes, 3) if successes else None
    return out
