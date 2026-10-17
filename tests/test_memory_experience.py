"""Cross-run experience memory (Harness-VLA style): own tmp_path SQLite file, advisory only."""

import sqlite3

import pytest

from liminal.memory import ExperienceStore, MemoryStore
from liminal.memory.experience import classify_failure, collect_run_experience, hit_ids

STEP = {"skill": "pick_place", "arm": "left", "object_id": "red", "destination_id": "box",
        "effect": "red_in_box", "retries": 1, "duration_s": 2.5}
FAIL = {"skill": "pick_place", "arm": "left", "object_id": "red", "destination_id": "box", "effect": "red_in_box",
        "kind": "backend_claimed_success_unverified", "detail": "block not in box"}


def _write(path, run_id="old", version="v1", arms=("left", "right"), steps=(STEP,), failures=(FAIL,)):
    with ExperienceStore(path) as store:
        assert store.record_run(run_id, scenario="two_blocks", registry_version=version, arms=arms,
                                goal_text="put red in box", completed=True, steps=list(steps),
                                failures=list(failures))


def _ask(path, *, version="v1", arms=("left", "right"), roles=("red", "box"), **kw):
    with ExperienceStore(path, writable=False) as store:
        return store.retrieve(registry_version=version, arms=arms, roles=roles, **kw)


def test_write_then_retrieve_on_a_compatible_run(tmp_path):
    path = tmp_path / "exp.sqlite3"
    _write(path)
    hits = _ask(path, roles=("red", "box", "blue"))
    procedure = hits["procedures"][0]
    assert procedure["steps"][0]["retries"] == 1 and procedure["role_overlap"] == ["box", "red"]
    assert procedure["source_run"] == "old"
    failure = hits["failures"][0]
    assert failure["kind"] == "backend_claimed_success_unverified" and failure["count"] == 1
    assert "NOT evidence" in hits["advisory"]
    assert hit_ids(hits) == ["trace:old", "failure:pick_place:red:backend_claimed_success_unverified"]
    assert _ask(path, exclude_run_id="old")["procedures"] == []  # a run never retrieves itself


def test_incompatible_registry_embodiment_or_roles_are_filtered_out(tmp_path):
    path = tmp_path / "exp.sqlite3"
    _write(path)
    for kwargs in ({"version": "v2"}, {"arms": ("left",)}, {"roles": ("green", "tray")}):
        hits = _ask(path, **kwargs)
        assert hits["procedures"] == [] and hits["failures"] == [], kwargs


def test_frozen_and_read_only_modes_never_write(tmp_path):
    path = tmp_path / "exp.sqlite3"
    _write(path)
    before = ExperienceStore(path, writable=False).corpus_hash()
    for store in (ExperienceStore(path, writable=False), ExperienceStore(path, frozen=True)):
        assert not store.writable
        assert store.record_run("new", scenario="s", registry_version="v1", arms=("left", "right"), goal_text="g",
                                completed=True, steps=[STEP], failures=[]) is False
        assert store.counts() == {"traces": 1, "failures": 1}
        store.close()
    assert ExperienceStore(path, writable=False).corpus_hash() == before
    # a read-only store over a missing file is an empty corpus and creates nothing
    missing = tmp_path / "nope.sqlite3"
    assert _ask(missing)["procedures"] == [] and not missing.exists()
    with pytest.raises(sqlite3.OperationalError):
        sqlite3.connect(f"file:{path}?mode=ro", uri=True).execute("DELETE FROM traces")


def test_rewriting_a_run_replaces_its_records(tmp_path):
    path = tmp_path / "exp.sqlite3"
    _write(path)
    _write(path, steps=(STEP, {**STEP, "object_id": "blue", "effect": "blue_in_box"}), failures=())
    with ExperienceStore(path, writable=False) as store:
        assert store.counts() == {"traces": 1, "failures": 0}


def test_collect_run_experience_distils_the_verified_ledger(tmp_path):
    with MemoryStore(tmp_path / "m.sqlite3") as store:
        store.create_run("r", {"text": "g"}, {})
        store.add_obligation("r", "red_in_box", {})
        store.start_attempt("r", "att-001", ["red_in_box"], {"skill": "pick_place", "arm": "left", "object_id": "red",
                                                              "destination_id": "box", "effect": "red_in_box"})
        store.record_event("r", "att-001:outcome", "attempt_outcome",
                           {"failure_kind": "grasp_failed", "reason": "no grasp"}, attempt_id="att-001")
        store.finish_attempt("r", "att-001", "failed")
        store.record_event("r", "drop-001", "plan_dropped", {"reason": "stale_revision", "skill": "pick_place",
                                                              "object_id": "red", "effect": "red_in_box"})
        store.start_attempt("r", "att-002", ["red_in_box"], {"skill": "pick_place", "arm": "right", "object_id": "red",
                                                              "destination_id": "box", "effect": "red_in_box"})
        store.record_evidence("r", "ev", "verification", {})
        store.finish_attempt("r", "att-002", "verified", evidence_ids=["ev"], verified_obligation_ids=["red_in_box"])
        steps, failures, completed = collect_run_experience(store, "r")
    assert completed and [s["retries"] for s in steps] == [1] and steps[0]["arm"] == "right"
    assert [f["kind"] for f in failures] == ["grasp_failed", "stale_plan"]


def test_failure_kinds():
    assert classify_failure(decision=None, error="grasp lost") == "grasp_failed"
    assert classify_failure(decision=None, error=None) == "backend_failed"
    assert classify_failure(decision="false", error=None) == "backend_claimed_success_unverified"
    assert classify_failure(decision="unknown", error=None) == "unknown_evidence"
    assert classify_failure(decision=None, error="EmergencyStop: x") == "emergency_stop"
