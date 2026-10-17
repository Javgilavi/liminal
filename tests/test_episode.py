"""Episode runner: mock/replay only (tmp_path SQLite, no hardware, no network, no key)."""

import asyncio
import json
from pathlib import Path

import pytest

from liminal.app import build_context
from liminal.memory import MemoryStore
from liminal.orchestration.episode import make_model, run_episode
from liminal.orchestration.replay import (
    ReplayBackend,
    ReplayModel,
    ReplayObserver,
    ReplayScene,
    load_scenario,
)
from liminal.ui.cli import main

CONFIGS = Path(__file__).parent.parent / "configs" / "replay"


class Rig:
    """One process worth of state: scene file + db persist across Rigs, everything else is fresh."""

    def __init__(self, tmp_path, name="two_blocks", *, model=None, **backend_kw):
        self.scenario = load_scenario(CONFIGS / f"{name}.yaml")
        self.scene = ReplayScene(self.scenario, tmp_path / "scene.json")
        self.backend = ReplayBackend(self.scene, **backend_kw)
        self.ctx = build_context("configs/robot/mock_dual.yaml", policy_backend=self.backend)
        self.observer = ReplayObserver(self.ctx.world, self.scene, ttl_s=30.0)
        self.model = model or ReplayModel(self.scenario)
        self.db = tmp_path / "ep.sqlite3"

    def run(self, **kw):
        kw.setdefault("run_id", "run-1")
        return asyncio.run(run_episode(
            self.scenario, self.ctx, self.model, self.db, self.observer.observe,
            observer_ids=(ReplayObserver.OBSERVER_ID,), **kw))

    def kinds(self, kind):
        return [e for e in self.ctx.world.events if e["kind"] == f"episode_{kind}"]


def _statuses(report):
    return {o.obligation_id: o.status for o in report.obligations}


def test_two_obligations_complete_with_two_verified_attempts(tmp_path):
    rig = Rig(tmp_path)
    report = rig.run()
    assert report.completed and report.stop_reason == "completed"
    assert _statuses(report) == {"red_in_box": "verified", "blue_in_box": "verified"}
    assert [a.status for a in report.attempts] == ["verified", "verified"]
    assert sorted(report.verified_effects) == ["blue_in_box", "red_in_box"]
    assert len(rig.backend.calls) == 2
    assert all(o.evidence_ids for o in report.obligations)
    assert report.mode == "replay" and "not hardware" in report.summary()


def test_trace_events_cover_the_decision_chain(tmp_path):
    rig = Rig(tmp_path)
    rig.run()
    for kind in ("start", "observation", "context", "proposal", "dispatch", "command_outcome",
                 "verification", "memory", "stop"):
        assert rig.kinds(kind), kind
    dispatch = rig.kinds("dispatch")[0]
    assert dispatch["arm"] in ("left", "right") and "red block" in dispatch["prompt"]
    assert rig.kinds("verification")[0]["evidence_ids"]
    assert rig.kinds("proposal")[0]["evidence_refs"]
    assert all(e["mode"] == "replay" for e in rig.kinds("start"))


def test_lie_first_backend_does_not_count(tmp_path):
    rig = Rig(tmp_path, lie_first=True)
    report = rig.run(max_attempts=1)
    assert report.stop_reason == "budget_attempts"
    assert not report.completed
    assert _statuses(report) == {"red_in_box": "pending", "blue_in_box": "pending"}
    assert report.verified_effects == ()
    # the backend said SUCCESS, the fresh observation said the scene did not change
    assert rig.backend.calls[0].result.ok and rig.scene.revision == 0
    assert [a.status for a in report.attempts] == ["failed"]


def test_lie_first_then_honest_retry_verifies_once(tmp_path):
    rig = Rig(tmp_path, lie_first=True)
    report = rig.run()
    assert report.completed
    assert [a.status for a in report.attempts] == ["failed", "verified", "verified"]
    assert sorted(report.verified_effects) == ["blue_in_box", "red_in_box"]


def test_fail_first_retries_within_budget(tmp_path):
    rig = Rig(tmp_path, fail_first=True)
    report = rig.run(max_attempts=3)
    assert report.completed and report.attempts_used == 3
    assert [a.status for a in report.attempts] == ["failed", "verified", "verified"]
    rig2 = Rig(tmp_path / "b", fail_first=True)
    capped = rig2.run(max_attempts=2)
    assert capped.stop_reason == "budget_attempts" and not capped.completed
    assert sorted(capped.verified_effects) == ["red_in_box"]


def test_interrupt_then_resume_finishes_only_the_remaining_obligation(tmp_path):
    first = Rig(tmp_path)
    interrupted = first.run(interrupt_after=1)
    assert interrupted.stop_reason == "interrupted" and not interrupted.completed
    assert _statuses(interrupted) == {"red_in_box": "verified", "blue_in_box": "pending"}

    second = Rig(tmp_path)  # new process: fresh world/backend/model, same scene + db
    resumed = second.run()
    assert resumed.resumed and resumed.completed
    assert len(second.backend.calls) == 1 and second.backend.calls[0].object_id == "blue"
    assert sorted(resumed.verified_effects) == ["blue_in_box", "red_in_box"]  # no duplicate effect
    assert [a.status for a in resumed.attempts] == ["verified", "verified"]
    with MemoryStore(second.db) as store:
        assert len(store.verified_effects("run-1")) == 2


def _crash_mid_attempt(rig, obligation, *, actuate):
    """Simulate a process death after dispatch began: attempt 'started' in SQLite, never finished."""
    with MemoryStore(rig.db) as store:
        store.create_run("run-1", {"goal_id": rig.scenario.goal.goal_id, "text": rig.scenario.goal.text,
                                   "obligations": list(rig.scenario.goal.obligations)},
                         {"scenario": rig.scenario.name, "registry_version": rig.scenario.registry.version})
        for name in rig.scenario.goal.obligations:
            store.add_obligation("run-1", name, {"predicate": name})
        store.start_attempt("run-1", "att-001", [obligation], {"skill": "pick_place", "arm": "left"})
    if actuate:
        binding = rig.scenario.obligations[obligation]
        rig.scene.locations[binding["object_id"]] = binding["destination_id"]
        rig.scene.revision += 1
        rig.scene._save_state()


def test_crash_after_actuation_is_reconciled_not_resent(tmp_path):
    _crash_mid_attempt(Rig(tmp_path), "red_in_box", actuate=True)
    reopened = Rig(tmp_path)
    report = reopened.run()
    assert [c.object_id for c in reopened.backend.calls] == ["blue"]  # red was never resent
    assert [a.status for a in report.attempts] == ["verified", "verified"]
    assert report.attempts[0].attempt_id == "att-001"
    assert sorted(report.verified_effects) == ["blue_in_box", "red_in_box"]
    assert any(e["action"] == "attempt_reconciled" and e["resent"] is False for e in reopened.kinds("ledger"))


def test_crash_without_effect_is_failed_then_redone_once(tmp_path):
    rig = Rig(tmp_path)
    _crash_mid_attempt(rig, "red_in_box", actuate=False)
    reopened = Rig(tmp_path)
    report = reopened.run()
    assert report.completed
    assert [a.status for a in report.attempts] == ["failed", "verified", "verified"]
    assert sorted(report.verified_effects) == ["blue_in_box", "red_in_box"]
    assert len(reopened.backend.calls) == 2


def test_stale_plan_is_dropped_before_dispatch(tmp_path):
    class Disturb:
        """Delegates to the replay model, then the scene changes while 'the model was thinking'."""

        def __init__(self, inner, scene):
            self.inner, self.scene, self.n = inner, scene, 0

        def complete_context(self, system, user, frames):
            reply = self.inner.complete_context(system, user, frames)
            self.n += 1
            if self.n == 1:
                self.scene.occlude("blue")  # bumps the scene revision
            return reply

    rig = Rig(tmp_path)
    rig.model = Disturb(rig.model, rig.scene)
    report = rig.run()
    dropped = rig.kinds("plan_dropped")
    assert dropped and dropped[0]["reason"] == "stale_revision"
    assert dropped[0]["plan_revision"] != dropped[0]["live_revision"]
    first_dispatch = rig.kinds("dispatch")[0]["attempt_id"]
    assert first_dispatch == "att-001"  # the stale plan consumed no attempt
    assert report.completed and report.attempts_used == 2
    assert report.model_calls_used == 3


def test_model_call_budget_stops_the_run(tmp_path):
    rig = Rig(tmp_path)
    report = rig.run(max_model_calls=1)
    assert report.stop_reason == "budget_model_calls" and not report.completed
    assert _statuses(report) == {"red_in_box": "verified", "blue_in_box": "pending"}
    assert rig.model.calls == 1


def test_attempt_budget_stops_the_run(tmp_path):
    rig = Rig(tmp_path)
    report = rig.run(max_attempts=1)
    assert report.stop_reason == "budget_attempts" and report.attempts_used == 1
    assert len(rig.backend.calls) == 1


def test_budget_survives_a_resume(tmp_path):
    Rig(tmp_path).run(max_attempts=1)
    again = Rig(tmp_path)
    report = again.run(max_attempts=5)  # limits can only shrink for an existing run
    assert report.stop_reason == "budget_attempts" and not again.backend.calls


class Rogue:
    """A model that tries every way of declaring victory without evidence."""

    def __init__(self, scenario):
        self.scenario, self.calls = scenario, 0
        self.replies = [
            {"goal_complete": True},
            {"schema_version": 1, "initial_state": {"red_in_box": True}},
            {"schema_version": 1, "facts": {"red_in_box": True, "blue_in_box": True}, "steps": []},
        ]

    def complete(self, system, user, images=()):
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        return json.dumps(reply)


def test_planner_cannot_mark_a_goal_complete(tmp_path):
    scenario = load_scenario(CONFIGS / "two_blocks.yaml")
    rig = Rig(tmp_path, model=Rogue(scenario))
    report = rig.run()
    assert report.stop_reason == "planner_failed" and not report.completed
    assert _statuses(report) == {"red_in_box": "pending", "blue_in_box": "pending"}
    assert report.attempts == () and not rig.backend.calls and rig.scene.revision == 0
    assert report.verified_effects == ()


def test_proposing_a_completed_effect_is_not_dispatched(tmp_path):
    class Stubborn(ReplayModel):
        def _proposal(self, user):
            payload = json.loads(user.split("\n")[0])  # repair prompts append prose
            payload["progress"] = {"pending_effects": ["red_in_box"], "completed_effects": []}
            return super()._proposal(json.dumps(payload))

    rig = Rig(tmp_path)
    rig.model = Stubborn(rig.scenario)
    report = rig.run(max_model_calls=6)
    assert report.verified_effects.count("red_in_box") == 1  # never re-done, never duplicated
    assert len([c for c in rig.backend.calls if c.object_id == "red"]) == 1
    assert not report.completed and report.stop_reason in {"planner_failed", "budget_model_calls", "max_cycles"}


def test_occluded_object_is_unknown_until_handled(tmp_path):
    rig = Rig(tmp_path, "occluded_reference")
    report = rig.run()
    assert report.completed and len(rig.backend.calls) == 1
    first_context = rig.kinds("context")[0]
    assert any(f["frame_id"].startswith("reference:") and f["role"] == "historical"
               for f in first_context["frames"]), "goal reference frame must come from memory"
    assert rig.kinds("observation")[0]["facts"] == {"red_in_box": "unknown"}


def test_externally_undone_obligation_is_redone(tmp_path):
    rig = Rig(tmp_path, "moved_destination")
    assert rig.run(interrupt_after=1).stop_reason == "interrupted"
    done = next(b for b in rig.scenario.obligations.values()
                if rig.scene.locations[b["object_id"]] == b["destination_id"])
    rig.scene.relocate(done["object_id"], "table")
    again = Rig(tmp_path, "moved_destination")
    report = again.run()
    assert report.completed
    assert any(e["action"] == "obligation_reopened" for e in again.kinds("ledger"))
    assert len(again.backend.calls) == 2
    assert len(report.verified_effects) == 3  # the undone placement is a new, separately verified effect


def test_make_model_replay_needs_no_key_and_claude_demands_one(monkeypatch):
    scenario = load_scenario(CONFIGS / "two_blocks.yaml")
    assert isinstance(make_model("replay", scenario), ReplayModel)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.chdir(Path(__file__).parent)  # no .env here
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        make_model("claude")
    with pytest.raises(ValueError):
        make_model("gpt")


def test_cli_episode_interrupt_and_resume(tmp_path, capsys):
    scenario = str(CONFIGS / "two_blocks.yaml")
    db = str(tmp_path / "x.sqlite3")
    common = [scenario, "--db", db, "--log-dir", str(tmp_path / "logs")]
    assert main(["episode", *common, "--interrupt-after", "1"]) == 1
    out = capsys.readouterr().out
    run_id = next(line.split()[-1] for line in out.splitlines() if line.startswith("resume:"))
    assert "red_in_box: verified" in out and "blue_in_box: pending" in out and "not hardware" in out
    assert main(["episode", *common, "--resume", run_id]) == 0
    assert "completed" in capsys.readouterr().out
    log = next((tmp_path / "logs").glob(f"*{run_id}.jsonl"))
    assert any(json.loads(line)["kind"] == "episode_verification" for line in log.read_text().splitlines())


# -- V1: authored tree via `liminal episode --task` ---------------------------------------------

TASK = Path(__file__).parent.parent / "configs" / "tasks" / "replay_two_blocks_v1.yaml"


def _authored(tmp_path, capsys, *flags):
    code = main(["episode", str(CONFIGS / "two_blocks.yaml"), "--task", str(TASK),
                 "--log-dir", str(tmp_path / "logs"), *flags])
    out = capsys.readouterr().out
    log = next((tmp_path / "logs").glob("*.jsonl"))
    events = [json.loads(line) for line in log.read_text().splitlines()]
    return code, out, events


def test_cli_authored_task_runs_without_planner_or_db(tmp_path, capsys):
    code, out, events = _authored(tmp_path, capsys)
    assert code == 0 and "completed" in out and "not hardware" in out
    kinds = [e["kind"] for e in events]
    assert "attempt_start" in kinds and "verification" in kinds
    assert not any(k in ("episode_proposal", "episode_context", "episode_dispatch") for k in kinds)
    assert len([e for e in events if e["kind"] == "attempt_end" and e["decision"] == "true"]) == 2


@pytest.mark.parametrize("flag, first", [("--lie-first", "failed"), ("--fail-first", "failed")])
def test_cli_authored_task_catches_lie_and_failure_then_retries(tmp_path, capsys, flag, first):
    code, out, events = _authored(tmp_path, capsys, flag)
    assert code == 0
    ends = [e for e in events if e["kind"] == "attempt_end"]
    assert len(ends) == 3  # red tried twice (first unverified), blue once
    assert ends[0]["decision"] != "true" and ends[1]["decision"] == "true"
    assert len({e["attempt_id"] for e in ends}) == 3  # fresh attempt per try
    assert first in out


def test_cli_authored_task_rejects_planner_only_flags(tmp_path, capsys):
    assert main(["episode", str(CONFIGS / "two_blocks.yaml"), "--task", str(TASK), "--resume", "run-x"]) == 2
    assert main(["episode", str(CONFIGS / "two_blocks.yaml")]) == 2  # planner path still needs --db


def test_run_authored_never_sets_both_scope_and_factory(tmp_path):
    from liminal.orchestration import load_task
    from liminal.orchestration.episode import run_authored

    rig = Rig(tmp_path)
    rig.ctx.verification = object()  # type: ignore[assignment]
    with pytest.raises(ValueError, match="either"):
        asyncio.run(run_authored(rig.scenario, rig.ctx, load_task(TASK), rig.observer))


# -- scheduled scene change: moved destination in the CLI run ---------------------------------------


def test_cli_moved_destination_drops_stale_plan_and_redoes_placement(tmp_path, capsys):
    scenario = str(CONFIGS / "moved_destination.yaml")
    code = main(["episode", scenario, "--db", str(tmp_path / "m.sqlite3"), "--log-dir", str(tmp_path / "logs")])
    out = capsys.readouterr().out
    assert code == 0 and "not hardware" in out
    log = next((tmp_path / "logs").glob("*.jsonl"))
    events = [json.loads(line) for line in log.read_text().splitlines()]
    dropped = [e for e in events if e["kind"] == "episode_plan_dropped"]
    assert dropped and dropped[0]["reason"] == "stale_revision"
    assert any(e["kind"] == "episode_ledger" and e["action"] == "obligation_reopened" for e in events)
    assert out.count(": verified pick_place") == 3  # red placed, taken out by the event, placed again, then blue


# -- V4 memory: MemER-style keyframes, ledger summary, cross-run experience ---------------------------------


class SpyModel:
    """Wraps the replay proposal fixture: records every prompt and optionally adds keyframe nominations."""

    def __init__(self, inner, *, nominate=False):
        self.inner, self.nominate, self.prompts = inner, nominate, []

    def _reply(self, user):
        payload = json.loads(user)
        self.prompts.append(payload)
        reply = json.loads(self.inner.complete("", user))
        if self.nominate and "keyframe_nominations" in payload["required_response_shape"]:
            shown = [f["frame_id"] for f in payload["frames"]]
            reply["keyframe_nominations"] = [{"frame_id": shown[0], "reason": "scene before the move"},
                                             {"frame_id": "invented-frame", "reason": "not shown"}]
        return json.dumps(reply)

    def complete(self, system, user, images=()):
        return self._reply(user)

    def complete_context(self, system, user, frames):
        return self._reply(user)


def _spy(rig, **kw):
    rig.model = SpyModel(rig.model, **kw)
    return rig.model


@pytest.mark.parametrize("scenario", ["two_blocks", "occluded_reference"])
@pytest.mark.parametrize("keyframes", ["event", "nominated", "none"])
def test_every_keyframe_mode_completes_and_is_inspectable(tmp_path, scenario, keyframes):
    from liminal.memory import inspect_run

    rig = Rig(tmp_path, scenario)
    spy = _spy(rig, nominate=True)
    report = rig.run(keyframes=keyframes)
    assert report.completed
    seen = inspect_run(rig.db, "run-1")
    assert seen["config"]["keyframes"] == keyframes and seen["keyframes"]["mode"] == keyframes
    assert all(o["status"] == "completed" for o in seen["ledger"]["obligations"])
    assert seen["summary"].startswith("Ledger progress") and "Not done yet" not in seen["summary"]
    assert seen["keyframes"]["contexts"] and seen["experience"]["mode"] == "off"
    refs = seen["keyframes"]["references"]
    if scenario == "occluded_reference":
        assert refs and all(r in [f["frame_id"] for f in seen["keyframes"]["selected"]] for r in refs)
    selected = [e["source"] for e in seen["keyframes"]["selected"] if e["source"] != "reference"]
    if keyframes == "event":
        assert selected and set(selected) == {"event"} and rig.kinds("keyframes")
        assert all("nominate" not in json.dumps(p["required_response_shape"]) for p in spy.prompts)
    elif keyframes == "nominated":
        assert set(selected) <= {"planner"} and rig.kinds("keyframes")[0]["action"] == "nominations"
        assert "keyframe_nominations" in spy.prompts[0]["required_response_shape"]
        assert seen["keyframes"]["rejected"][0]["reason"] == "frame id was not shown to the planner"
        assert seen["keyframes"]["nominations"]
    else:
        assert not selected and not rig.kinds("keyframes")
    for prompt in spy.prompts:  # recent frames + at most the context budget, references labelled historical
        assert len(prompt["frames"]) <= 6


def test_default_options_keep_current_behavior(tmp_path):
    rig = Rig(tmp_path)
    spy = _spy(rig)
    report = rig.run()
    assert report.completed and not (tmp_path / "experience.sqlite3").exists()
    assert not rig.kinds("experience") and all("experience" not in p for p in spy.prompts)
    assert all("keyframe_nominations" not in p["required_response_shape"] for p in spy.prompts)


def test_planner_memory_is_a_ledger_summary_not_a_prediction(tmp_path):
    rig = Rig(tmp_path, lie_first=True)
    spy = _spy(rig)
    rig.run()
    first, second, third = (p["memory"] for p in spy.prompts[:3])
    assert first["progress_record"]["completed"] == [] and "0 of 2" in first["summary"]
    # after the false backend claim, memory records a failed attempt and still lists red as not done
    assert second["progress_record"]["failed_attempts"][0]["failure_kind"] == "backend_claimed_success_unverified"
    assert "Verified:" not in second["summary"] and "red_in_box" in second["summary"].split("Not done yet:")[1]
    assert [c["obligation"] for c in third["progress_record"]["completed"]] == ["red_in_box"]
    assert "blue_in_box" in third["summary"].split("Not done yet:")[1]
    assert all(h["kind"] in {"goal_reference", "keyframe", "recent"} for h in third["historical_frames"])
    # the trusted progress block is separate and unchanged in meaning
    assert spy.prompts[2]["progress"]["completed_effects"] == ["red_in_box"]


def _experience_rig(tmp_path, name, **backend):
    path = tmp_path / name
    path.mkdir()
    rig = Rig(path, **backend)
    return rig, _spy(rig)


def test_experience_written_then_retrieved_by_a_compatible_run(tmp_path):
    from liminal.memory import ExperienceStore, inspect_run

    exp = tmp_path / "runs" / "experience.sqlite3"
    first, _ = _experience_rig(tmp_path, "a", lie_first=True)
    assert first.run(experience_db=exp, experience_mode="readwrite").completed
    assert first.kinds("experience")[-1]["action"] == "written"
    with ExperienceStore(exp, writable=False) as store:
        assert store.counts()["traces"] == 1
        failure_kinds = {f["kind"] for f in store.retrieve(registry_version="replay-v1", arms=("left", "right"),
                                                           roles=("red", "box"))["failures"]}
    assert failure_kinds == {"backend_claimed_success_unverified"}

    second, spy = _experience_rig(tmp_path, "b")
    report = second.run(experience_db=exp, experience_mode="readwrite", run_id="run-2")
    assert report.completed
    experience = spy.prompts[0]["experience"]
    assert experience["procedures"][0]["source_run"] == "run-1"
    assert [s["object_id"] for s in experience["procedures"][0]["steps"]] == ["red", "blue"]
    assert experience["failures"][0]["kind"] == "backend_claimed_success_unverified"
    assert "NOT evidence" in experience["advisory"]
    assert second.kinds("experience")[0]["action"] == "retrieved" and second.kinds("experience")[0]["hit_ids"]
    assert inspect_run(second.db, "run-2")["experience"]["hit_ids"] == second.kinds("experience")[0]["hit_ids"]


def test_experience_never_completes_an_obligation(tmp_path):
    from liminal.memory import ExperienceStore

    exp = tmp_path / "exp.sqlite3"
    with ExperienceStore(exp) as store:  # a corpus that "knows" both blocks end up in the box
        steps = [{"skill": "pick_place", "arm": "left", "object_id": o, "destination_id": "box",
                  "effect": f"{o}_in_box", "retries": 0, "duration_s": 1.0} for o in ("red", "blue")]
        store.record_run("old", scenario="two_blocks", registry_version="replay-v1", arms=("left", "right"),
                         goal_text="g", completed=True, steps=steps, failures=[])
    rig, spy = _experience_rig(tmp_path, "a", lie_first=True)
    report = rig.run(experience_db=exp, experience_mode="read", max_attempts=1)
    assert spy.prompts[0]["experience"]["procedures"] and spy.prompts[0]["progress"]["completed_effects"] == []
    assert not report.completed and report.verified_effects == ()
    assert _statuses(report) == {"red_in_box": "pending", "blue_in_box": "pending"}


def test_experience_from_an_incompatible_registry_is_not_retrieved(tmp_path):
    from liminal.memory import ExperienceStore

    exp = tmp_path / "exp.sqlite3"
    with ExperienceStore(exp) as store:
        store.record_run("old", scenario="two_blocks", registry_version="some-other-registry", arms=("left", "right"),
                         goal_text="g", completed=True, failures=[], steps=[
                             {"skill": "pick_place", "arm": "left", "object_id": "red", "destination_id": "box"}])
    rig, spy = _experience_rig(tmp_path, "a")
    assert rig.run(experience_db=exp, experience_mode="read").completed
    assert all("experience" not in p for p in spy.prompts)
    assert rig.kinds("experience")[0]["hit_ids"] == []


def test_frozen_and_read_modes_leave_the_corpus_untouched(tmp_path):
    from liminal.memory import ExperienceStore

    exp = tmp_path / "exp.sqlite3"
    seed, _ = _experience_rig(tmp_path, "seed")
    seed.run(experience_db=exp, experience_mode="readwrite")
    with ExperienceStore(exp, writable=False) as store:
        before = store.corpus_hash()
    for index, (mode, frozen) in enumerate([("readwrite", True), ("read", False)]):
        rig, _ = _experience_rig(tmp_path, f"r{index}")
        assert rig.run(experience_db=exp, experience_mode=mode, experience_frozen=frozen, run_id="held-out").completed
        assert rig.kinds("experience")[-1]["action"] == "not_written"
        with ExperienceStore(exp, writable=False) as store:
            assert store.corpus_hash() == before and store.counts()["traces"] == 1


def test_interrupted_run_writes_experience_only_when_it_finishes(tmp_path):
    from liminal.memory import ExperienceStore

    exp = tmp_path / "exp.sqlite3"
    first = Rig(tmp_path)
    assert first.run(interrupt_after=1, experience_db=exp, experience_mode="readwrite").stop_reason == "interrupted"
    if exp.exists():
        with ExperienceStore(exp, writable=False) as store:
            assert store.counts()["traces"] == 0
    assert Rig(tmp_path).run(experience_db=exp, experience_mode="readwrite").completed
    with ExperienceStore(exp, writable=False) as store:
        assert store.counts()["traces"] == 1


def test_invalid_memory_options_are_refused(tmp_path):
    rig = Rig(tmp_path)
    with pytest.raises(ValueError, match="keyframes"):
        rig.run(keyframes="sometimes")
    with pytest.raises(ValueError, match="experience_mode"):
        rig.run(experience_mode="write")
    with pytest.raises(ValueError, match="experience_db"):
        rig.run(experience_mode="read")
