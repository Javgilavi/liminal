"""V6 eval runner: replay/mock only, no network, no hardware."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from liminal.eval import ExperimentError, UnmatchedBudgets, eval_main, load_experiment, run_experiment
from liminal.eval.metrics import score_run, wilson
from liminal.eval.runner import unavailable_memory_options

SMOKE = "configs/eval/smoke.yaml"


@pytest.fixture(scope="module")
def smoke(tmp_path_factory):
    out = tmp_path_factory.mktemp("smoke")
    return run_experiment(SMOKE, out), out


def _write(tmp_path: Path, **over) -> Path:
    raw = yaml.safe_load(Path(SMOKE).read_text())
    raw.update(over)
    path = tmp_path / "exp.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def _cell(result, condition, scenario):
    return next(r for r in result.runs if r["condition"] == condition and r["scenario"] == scenario)


def test_smoke_writes_results_and_report(smoke):
    result, out = smoke
    data = json.loads((out / "results.json").read_text())
    assert data["label"] == "REPLAY/MOCK, not hardware" and len(data["runs"]) == 5
    report = (out / "report.md").read_text()
    assert "REPLAY/MOCK, not hardware" in report and "3/3" in report and "95% CI" in report
    assert all(r["verified_success"] and not r["false_completion"] for r in result.runs)


def test_lie_first_costs_one_failed_attempt_and_still_completes(smoke):
    result, _ = smoke
    for cond in ("planner", "authored"):
        r = _cell(result, cond, "lie_first")
        assert r["verified_success"] and r["recovery_attempts"] == 1 and r["recovered"]
        assert r["attempts"] == 2 and r["duplicate_effects"] == 0
    assert _cell(result, "planner", "one_block")["recovery_attempts"] == 0


def test_interrupt_resume_has_no_duplicates(smoke):
    result, _ = smoke
    r = _cell(result, "planner", "interrupt_resume")
    assert r["resumed"] and r["verified_success"] and r["duplicate_effects"] == 0 and r["attempts"] == 2
    skipped = [s for s in result.skipped if s["scenario"] == "interrupt_resume"]
    assert [s["condition"] for s in skipped] == ["authored"]


def test_unmatched_budgets_are_refused(tmp_path):
    raw = yaml.safe_load(Path(SMOKE).read_text())
    raw["conditions"][1]["max_attempts"] = 3
    path = _write(tmp_path, conditions=raw["conditions"])
    with pytest.raises(UnmatchedBudgets, match="identical"):
        run_experiment(path, tmp_path / "out")
    assert eval_main([str(path), "--out", str(tmp_path / "o2")]) == 2
    raw["conditions"][1]["max_attempts"] = 6  # equal to the matched budget is fine
    assert load_experiment(_write(tmp_path, conditions=raw["conditions"]))


def test_paid_models_need_an_explicit_opt_in(tmp_path):
    raw = yaml.safe_load(Path(SMOKE).read_text())
    raw["conditions"][1]["model"] = "claude"
    with pytest.raises(ExperimentError, match="allow_paid"):
        run_experiment(_write(tmp_path, conditions=raw["conditions"]), tmp_path / "out")


def test_unavailable_memory_option_is_reported_skipped(tmp_path):
    assert unavailable_memory_options({"definitely_not_a_kwarg": 1}) == ["definitely_not_a_kwarg"]
    cond = [{"name": "base", "mode": "planner"}, {"name": "future", "mode": "planner",
                                                   "memory": {"definitely_not_a_kwarg": "x"}}]
    scen = [{"name": "one", "scenario": "configs/replay/workbench.yaml", "goal": "Put the red block in the box"}]
    result = run_experiment(_write(tmp_path, conditions=cond, scenarios=scen), tmp_path / "out")
    assert result.conditions["future"]["attempted_runs"] == 0
    assert result.skipped[0]["reason"].startswith("skipped: option unavailable")
    assert "skipped: option unavailable" in Path(result.report_path).read_text()
    assert result.conditions["base"]["attempted_runs"] == 1  # the other condition is unaffected


def test_false_completion_is_detected_when_the_final_observation_disagrees():
    lying = SimpleNamespace(stop_reason="completed", completed=True, obligations=(), attempts=(),
                            model_calls_used=0, attempts_used=1, resumed=False)
    m = score_run(lying, {"red_in_box": False}, [], ["red_in_box"])
    assert m.false_completion and not m.verified_success and m.missed_obligations == 1
    honest = score_run(lying, {"red_in_box": True}, [], ["red_in_box"])
    assert honest.verified_success and not honest.false_completion
    occluded = score_run(lying, {"red_in_box": None}, [], ["red_in_box"])
    assert occluded.false_completion  # unknown is not agreement


def test_runner_catches_a_report_that_lies_end_to_end(tmp_path):
    async def broken(prepared):
        report = SimpleNamespace(stop_reason="completed", completed=True, obligations=(), attempts=(),
                                 model_calls_used=0, attempts_used=1, resumed=False)
        return report  # nothing was ever moved in the scene

    scen = [{"name": "one", "scenario": "configs/replay/workbench.yaml", "goal": "Put the red block in the box"}]
    cond = [{"name": "broken", "mode": "planner"}]
    result = run_experiment(_write(tmp_path, scenarios=scen, conditions=cond), tmp_path / "out", execute=broken)
    agg = result.conditions["broken"]
    assert agg["false_completion"]["k"] == 1 and agg["verified_success"]["k"] == 0


def test_crash_counts_in_the_denominator(tmp_path):
    async def boom(prepared):
        raise RuntimeError("model unavailable")

    scen = [{"name": "one", "scenario": "configs/replay/workbench.yaml", "goal": "Put the red block in the box"}]
    result = run_experiment(_write(tmp_path, scenarios=scen, conditions=[{"name": "c", "mode": "planner"}]),
                            tmp_path / "out", execute=boom)
    assert result.conditions["c"]["attempted_runs"] == 1 and result.conditions["c"]["errors"] == 1
    assert "model unavailable" in Path(result.report_path).read_text()


def test_wilson_interval_is_wide_for_small_samples():
    lo, hi = wilson(3, 3)
    assert 0.4 < lo < 0.5 and hi == 1.0
    assert wilson(0, 0) == (0.0, 1.0)
