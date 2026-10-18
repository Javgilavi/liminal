import importlib.util
import sys
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("trial_log", Path(__file__).resolve().parents[1] / "trial_log.py")
trial_log = importlib.util.module_from_spec(_spec)
sys.modules["trial_log"] = trial_log  # dataclasses look their module up in sys.modules
_spec.loader.exec_module(trial_log)


def _add(log, *extra):
    base = ["--log", str(log), "add", "--task", "pack_box", "--checkpoint", "so101@abc", "--steps-total", "5"]
    assert trial_log.main(base + list(extra)) == 0


def test_attempts_number_per_task_and_variant(tmp_path):
    log = tmp_path / "t.jsonl"
    _add(log, "--condition", "zero_shot", "--variant", "A0", "--steps-done", "1")
    _add(log, "--condition", "zero_shot", "--variant", "A0", "--steps-done", "2")
    _add(log, "--condition", "post_training", "--variant", "lora_A", "--steps-done", "5")
    trials = trial_log.load(log)
    assert [(t.variant, t.attempt) for t in trials] == [("A0", 1), ("A0", 2), ("lora_A", 1)]


def test_auto_success_needs_all_steps_and_no_help(tmp_path):
    log = tmp_path / "t.jsonl"
    _add(log, "--condition", "post_training", "--variant", "lora_A", "--steps-done", "5")
    _add(log, "--condition", "post_training", "--variant", "lora_A", "--steps-done", "5", "--interventions", "1")
    _add(log, "--condition", "post_training", "--variant", "lora_A", "--steps-done", "4", "--success", "yes")
    assert [t.success for t in trial_log.load(log)] == [True, False, True]


def test_summary_orders_zero_shot_first_and_counts(tmp_path):
    log = tmp_path / "t.jsonl"
    _add(log, "--condition", "post_training", "--variant", "lora_A", "--steps-done", "5")
    _add(log, "--condition", "zero_shot", "--variant", "A0", "--steps-done", "1", "--failure", "grasp miss")
    _add(log, "--condition", "zero_shot", "--variant", "A0", "--steps-done", "3", "--failure", "grasp miss",
         "--interventions", "2")
    rows = trial_log.summarize(trial_log.load(log))
    assert [r.condition for r in rows] == ["zero_shot", "post_training"]
    zs = rows[0]
    assert (zs.attempts, zs.successes, zs.mean_steps_done, zs.with_intervention) == (2, 0, 2.0, 1)
    assert zs.top_failures == [("grasp miss", 2)]
    table = trial_log.render(rows, markdown=True)
    assert table.splitlines()[0].startswith("| task |") and "0/2 (0%)" in table and "1/1 (100%)" in table


def test_rejects_bad_records(tmp_path):
    with pytest.raises(ValueError):
        trial_log.Trial(task="t", condition="later", variant="v", checkpoint="c", steps_total=3, steps_done=1,
                        success=False)
    with pytest.raises(ValueError):
        trial_log.Trial(task="t", condition="zero_shot", variant="v", checkpoint="c", steps_total=3, steps_done=4,
                        success=False)


def test_summary_on_empty_log_returns_1(tmp_path):
    assert trial_log.main(["--log", str(tmp_path / "none.jsonl"), "summary"]) == 1


def test_undo_is_dry_run_without_yes_and_keeps_removed(tmp_path):
    log = tmp_path / "t.jsonl"
    _add(log, "--condition", "zero_shot", "--variant", "A0", "--steps-done", "1")
    _add(log, "--condition", "zero_shot", "--variant", "A0", "--steps-done", "2")
    assert trial_log.main(["--log", str(log), "undo"]) == 0
    assert len(trial_log.load(log)) == 2
    assert trial_log.main(["--log", str(log), "undo", "--yes"]) == 0
    assert [t.steps_done for t in trial_log.load(log)] == [1]
    assert [t.steps_done for t in trial_log.load(tmp_path / "t.removed.jsonl")] == [2]
    _add(log, "--condition", "zero_shot", "--variant", "A0", "--steps-done", "3")
    assert [t.attempt for t in trial_log.load(log)] == [1, 2]
