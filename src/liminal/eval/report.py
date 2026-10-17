"""Readable report.md: per-condition table with denominators, uncertainty and honest labels."""

from __future__ import annotations

from typing import Any


def _rate(d: dict[str, Any]) -> str:
    lo, hi = d["ci95"]
    return f"{d['k']}/{d['n']} ({100 * d['rate']:.0f}%, 95% CI {100 * lo:.0f}-{100 * hi:.0f}%)" if d["n"] else "n/a"


def _row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def render_report(result: Any) -> str:
    runs_total = sum(c["attempted_runs"] for c in result.conditions.values())
    L = [f"# Eval: {result.experiment}", "",
         f"**{result.label}.** Scene evidence is simulated; nothing here is a hardware result.", ""]
    if result.description:
        L += [result.description, ""]
    b = result.budgets
    L += [f"Matched budgets for every condition: max_attempts={b['max_attempts']}, "
          f"max_model_calls={b['max_model_calls']}. Repeats: {result.repeats}, seeds: {result.seeds}. "
          f"Attempted runs: {runs_total}.", "",
          "## Per condition", "",
          _row(["condition", "runs", "verified success", "false completion", "abstained", "duplicates",
                "missed oblig.", "recovered", "retry attempts", "plans dropped", "planner rejections",
                "model calls", "attempts", "wall s (p50/p95)"]),
          _row(["---"] * 14)]
    for name, a in result.conditions.items():
        if not a["attempted_runs"]:
            reasons = {s["reason"] for s in result.skipped if s["condition"] == name}
            L.append(_row([name, "0", *["-"] * 11, "; ".join(sorted(reasons)) or "skipped"]))
            continue
        drops = ", ".join(f"{k}={v}" for k, v in sorted(a["plans_dropped"].items())) or "0"
        L.append(_row([name, str(a["attempted_runs"]), _rate(a["verified_success"]), _rate(a["false_completion"]),
                       _rate(a["abstained"]), str(a["duplicate_effects"]), str(a["missed_obligations"]),
                       _rate(a["recovered"]), str(a["recovery_attempts"]), drops, str(a["planner_rejections"]),
                       f"{a['model_calls']:g}", f"{a['attempts']:g}",
                       f"{a['wall_s']['p50']:.2f}/{a['wall_s']['p95']:.2f}"]))
    L += ["", "Denominator is every attempted run, including crashes and timeouts. Skipped cells are never counted.",
          "Verified success = the report says completed AND a fresh observation of the scene after the run "
          "agrees. False completion = the report says completed but that observation disagrees.", ""]

    L += ["## Per scenario (verified success, k/n)", ""]
    scenarios = sorted({s for d in result.by_scenario.values() for s in d})
    L += [_row(["scenario", *result.conditions]), _row(["---"] * (len(result.conditions) + 1))]
    for s in scenarios:
        cells = []
        for c in result.conditions:
            a = result.by_scenario.get(c, {}).get(s)
            if a:
                cells.append(f"{a['verified_success']['k']}/{a['verified_success']['n']}")
            else:
                cells.append(next((x["reason"] for x in result.skipped if x["condition"] == c and x["scenario"] == s),
                                  "-"))
        L.append(_row([s, *cells]))

    if result.skipped:
        L += ["", "## Skipped", ""]
        L += [f"- {x['condition']} / {x['scenario']}: {x['reason']}" for x in result.skipped]

    errors = [r for r in result.runs if r.get("error")]
    if errors:
        L += ["", "## Crashed runs", ""] + [f"- {r['condition']} / {r['scenario']} r{r['repeat']}: {r['error']}"
                                            for r in errors]
    tokens = [c["tokens"] for c in result.conditions.values() if c["tokens"] is not None]
    token_note = "reported per run in results.json" if tokens else "not reported (offline model; placeholder)"
    L += ["", "## Notes", "", f"- API tokens: {token_note}.",
          "- Replay runs are deterministic, so repeats and seeds mostly check reproducibility and run order, "
          "not sampling variance; the intervals show how little a handful of runs can establish.",
          "- Small samples are exploratory. Do not call a difference here a statistically established "
          "improvement; a refinement earns its complexity only with higher verified success, no more false "
          "completion, under the same budgets, on held-out scenes and on hardware.", ""]
    return "\n".join(L)
