from __future__ import annotations

import pytest

from liminal.memory import MemoryStore


class FakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float = 1.0) -> None:
        self.value += seconds


def test_restart_keeps_obligation_pending_and_attempt_unresolved(tmp_path) -> None:
    path = tmp_path / "episode.sqlite3"
    clock = FakeClock()
    store = MemoryStore(path, clock=clock)
    store.create_run("r1", {"goal": "put both blocks in box"})
    store.add_obligation("r1", "red", {"object": "red block", "destination": "box"})
    store.add_obligation("r1", "blue", {"object": "blue block", "destination": "box"})
    store.start_attempt("r1", "a1", ["red"], {"skill": "pick_place"})
    store.consume_budget("r1", "model_calls", limit=4)
    store.close()

    inspection = MemoryStore(path, clock=clock)
    assert inspection.get_attempt("r1", "a1").status == "started"
    assert inspection.reconcile("r1").unresolved_attempt_ids == ("a1",)
    inspection.close()

    resumed = MemoryStore(path, clock=clock, recover_open_attempts=True)
    reconciliation = resumed.reconcile("r1")
    assert reconciliation.interrupted_attempt_ids == ("a1",)
    assert reconciliation.unresolved_attempt_ids == ("a1",)
    assert reconciliation.pending_obligation_ids == ("blue", "red")
    assert reconciliation.auto_resend is False
    assert resumed.get_attempt("r1", "a1").status == "interrupted"
    assert resumed.budget("r1", "model_calls") == (1.0, 4.0)
    clock.advance()
    resumed.record_evidence("r1", "fresh-scene", "observation", {"captured": True})
    resolved = resumed.reconcile_attempt("r1", "a1", "unknown", evidence_ids=["fresh-scene"])
    assert resolved.status == "unknown"
    assert resumed.reconcile("r1").unresolved_attempt_ids == ("a1",)
    resumed.close()


def test_verified_effect_is_unique_and_completes_only_with_evidence(tmp_path) -> None:
    clock = FakeClock()
    store = MemoryStore(tmp_path / "episode.sqlite3", clock=clock)
    store.create_run("r1", {"goal": "place block"})
    store.add_obligation("r1", "place-red", {"object": "red"})
    store.start_attempt("r1", "a1", ["place-red"])
    clock.advance()
    store.record_evidence("r1", "ev-1", "visual", {"relation": "inside"})

    with pytest.raises(ValueError, match="evidence ID"):
        store.complete_obligation("r1", "place-red", [])

    done = store.finish_attempt(
        "r1", "a1", "verified", evidence_ids=["ev-1"],
        verified_obligation_ids=["place-red"], effect_key="placed:red:box",
    )
    assert done.status == "verified"
    repeated = store.finish_attempt(
        "r1", "a1", "verified", evidence_ids=["ev-1"],
        verified_obligation_ids=["place-red"], effect_key="placed:red:box",
    )
    assert repeated == done
    assert store.get_obligation("r1", "place-red").status == "completed"
    assert len(store.verified_effects("r1")) == 1

    # A duplicate callback with a new event ID returns the existing unique effect.
    duplicate = store.record_event(
        "r1", "late-callback", "placed", {"source": "duplicate"},
        effect_key="placed:red:box", verified=True, evidence_ids=["ev-1"],
    )
    assert duplicate.event_id == "a1:effect"
    assert len(store.verified_effects("r1")) == 1
    with pytest.raises(ValueError, match="fresh evidence"):
        store.set_obligation_status("r1", "place-red", "pending")
    clock.advance()
    store.record_evidence("r1", "ev-2", "visual", {"relation": "outside"})
    reopened = store.set_obligation_status("r1", "place-red", "pending", evidence_ids=["ev-2"])
    assert reopened.status == "pending"
    store.close()


def test_payloads_are_detached_immutable_and_run_goal_is_checked(tmp_path) -> None:
    store = MemoryStore(tmp_path / "episode.sqlite3")
    goal = {"objects": ["red", "blue"]}
    record = store.create_run("run", goal, {"rig": "mock"})
    goal["objects"].append("green")
    assert record.goal["objects"] == ("red", "blue")
    with pytest.raises(TypeError):
        record.goal["objects"] = ()

    assert store.create_run("run", {"objects": ["red", "blue"]}, {"rig": "mock"}).status == "active"
    with pytest.raises(ValueError, match="different goal"):
        store.create_run("run", {"objects": ["green"]}, {"rig": "mock"})
    store.close()


def test_frame_store_bounds_bytes_and_persists_context_selection(tmp_path) -> None:
    path = tmp_path / "episode.sqlite3"
    clock = FakeClock()
    store = MemoryStore(
        path, clock=clock, max_recent_frames=2, max_stored_frames=3,
        max_stored_frame_bytes=8, max_selected_frames=2,
    )
    store.create_run("r1", {"goal": "remember target"})
    store.add_frame("r1", "f1", b"111", captured_at=1, camera="scene", event="goal-reference")
    store.select_frames("r1", ["f1"], reason="indicated target", priority=10)
    clock.advance()
    store.add_frame("r1", "f2", b"222", captured_at=2, camera="wrist", event="pre-action")
    clock.advance()
    store.add_frame("r1", "f3", b"333", captured_at=3, camera="scene", event="post-action")
    assert [frame.frame_id for frame in store.recent_frames("r1")] == ["f3"]

    # Selected evidence survives count and byte trimming.
    assert store.get_frame("r1", "f1").selected
    context = store.select_context("r1", max_frames=2, reason="planner request", context_id="ctx-1")
    assert [frame.frame_id for frame in context.frames] == ["f1", "f3"]
    store.close()

    reopened = MemoryStore(path, clock=clock, max_recent_frames=2, max_stored_frames=3, max_stored_frame_bytes=8)
    assert reopened.get_frame("r1", "f1").data == b"111"
    assert reopened.selected_frames("r1")[0].selection_reason == "indicated target"
    restored = reopened.get_context("r1", "ctx-1")
    assert [frame.frame_id for frame in restored.frames] == ["f1", "f3"]
    with pytest.raises(ValueError, match="byte limit"):
        reopened.add_frame("r1", "too-large", b"123456789", captured_at=4, camera="scene", event="failure")
    reopened.close()


def test_consumed_budget_is_enforced_and_persists(tmp_path) -> None:
    path = tmp_path / "episode.sqlite3"
    store = MemoryStore(path, clock=FakeClock())
    store.create_run("r1", {"goal": "test"})
    assert store.consume_budget("r1", "retries", amount=2, limit=3) == 2
    store.close()

    store = MemoryStore(path, clock=FakeClock())
    with pytest.raises(ValueError, match="exhausted"):
        store.consume_budget("r1", "retries", amount=2)
    with pytest.raises(ValueError, match="cannot be increased"):
        store.consume_budget("r1", "retries", amount=1, limit=4)
    for invalid in (float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="budget amount"):
            store.consume_budget("r1", "bad", amount=invalid)
    assert store.budget("r1", "retries") == (2.0, 3.0)
    store.close()


def test_persisted_context_reports_evicted_frame_ids(tmp_path) -> None:
    store = MemoryStore(
        tmp_path / "episode.sqlite3", max_recent_frames=2,
        max_stored_frames=2, max_stored_frame_bytes=32,
    )
    store.create_run("r1", {"goal": "remember sequence"})
    store.add_frame("r1", "f1", b"one", captured_at=1, camera="scene", event="before")
    store.add_frame("r1", "f2", b"two", captured_at=2, camera="scene", event="after")
    store.select_context("r1", max_frames=2, reason="replay", context_id="ctx")
    store.add_frame("r1", "f3", b"tri", captured_at=3, camera="scene", event="newer")

    snapshot = store.get_context("r1", "ctx")
    assert snapshot.complete is False
    assert snapshot.missing_frame_ids == ("f1",)
    assert [frame.frame_id for frame in snapshot.frames] == ["f2"]
    store.close()


def test_reselection_rejects_pins_over_current_byte_quota(tmp_path) -> None:
    path = tmp_path / "episode.sqlite3"
    store = MemoryStore(path, max_stored_frame_bytes=8)
    store.create_run("r1", {"goal": "pin references"})
    store.add_frame("r1", "f1", b"12345", captured_at=1, camera="scene", event="goal")
    store.add_frame("r1", "f2", b"678", captured_at=2, camera="scene", event="pre-action")
    store.close()

    smaller_quota = MemoryStore(path, max_stored_frame_bytes=7)
    with pytest.raises(ValueError, match="byte limit"):
        smaller_quota.select_frames("r1", ["f1", "f2"], reason="too many pins")
    smaller_quota.close()
