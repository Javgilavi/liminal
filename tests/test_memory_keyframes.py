"""MemER-style keyframes and the ledger-only progress summary: tmp_path SQLite, no hardware, no network."""

import pytest

from liminal.memory import KeyframeMemory, MemoryStore, Nomination, build_progress, cluster_nominations, inspect_run


def _store(tmp_path, **kw):
    store = MemoryStore(tmp_path / "m.sqlite3", **kw)
    store.create_run("r", {"goal_id": "g", "text": "put red in box"}, {})
    return store


def _frames(store, count, *, start=1, event="observation"):
    for seq in range(start, start + count):
        store.add_frame("r", f"f{seq}", f"png{seq}".encode(), captured_at=float(seq), camera="scene", event=event,
                        metadata={"media_type": "image/png", "revision": seq, "seq": seq})


def _nom(frame_id, seq, reason="x"):
    return Nomination(frame_id, reason, "planner", "planner_nomination", seq)


def test_clustering_groups_adjacent_nominations_and_keeps_the_median_frame():
    clusters = cluster_nominations([_nom("f1", 1), _nom("f2", 2), _nom("f3", 3), _nom("f9", 9)], gap=1)
    assert [[m.frame_id for m in c.members] for c in clusters] == [["f1", "f2", "f3"], ["f9"]]
    assert [c.keyframe.frame_id for c in clusters] == ["f2", "f9"]
    # even-sized clusters take the lower median, deterministically
    assert cluster_nominations([_nom("a", 1), _nom("b", 2)], gap=1)[0].keyframe.frame_id == "a"
    # a wider gap merges more; duplicate nominations of one frame merge their reasons
    wide = cluster_nominations([_nom("f1", 1, "pre"), _nom("f1", 1, "post"), _nom("f4", 4)], gap=3)
    assert len(wide) == 1 and "pre" in wide[0].reason and "post" in wide[0].reason
    assert cluster_nominations([], 1) == []


def test_nominated_mode_clusters_persists_and_restores(tmp_path):
    with _store(tmp_path) as store:
        _frames(store, 8)
        kf = KeyframeMemory(store, "r", "nominated", max_keyframes=2)
        kf.accept_planner_nominations([{"frame_id": "f1", "reason": "start"}, {"frame_id": "f2", "reason": "grasp"},
                                       {"frame_id": "f3", "reason": "lift"}])
        kf.accept_planner_nominations([{"frame_id": "f6", "reason": "placed"}])
        selected = kf.apply()
        assert [e["frame_id"] for e in selected] == ["f2", "f6"]
        assert selected[0]["cluster_size"] == 3 and "grasp" in selected[0]["reason"]
        assert {f.frame_id for f in store.selected_frames("r")} == {"f2", "f6"}
    with MemoryStore(tmp_path / "m.sqlite3") as reopened:  # a restart reloads nominations and reasons
        again = KeyframeMemory(reopened, "r", "nominated", max_keyframes=2)
        assert [c.keyframe.frame_id for c in again.keyframes()] == ["f2", "f6"]
        assert [n["frame_id"] for n in again.describe()["selected"]] == ["f2", "f6"]


def test_nominations_are_advisory_and_bounded(tmp_path):
    with _store(tmp_path) as store:
        _frames(store, 12)
        kf = KeyframeMemory(store, "r", "nominated", max_keyframes=3)
        kf.accept_planner_nominations([{"frame_id": "ghost", "reason": "never stored"}])  # skipped, not an error
        for seq in (1, 4, 7, 10):  # four separate clusters, only the newest three survive
            kf.accept_planner_nominations([{"frame_id": f"f{seq}", "reason": f"moment {seq}"}])
        kf.apply()
        assert [c.keyframe.frame_id for c in kf.keyframes()] == ["f4", "f7", "f10"]
        assert kf.describe()["rejected"][0]["frame_id"] == "ghost"
        assert KeyframeMemory(store, "r", "event").note_event(["f1"], "x", "y") == ["f1"]
        # event-mode memory ignores planner nominations and vice versa
        assert KeyframeMemory(store, "r", "event").accept_planner_nominations([{"frame_id": "f1"}]) == []
        assert kf.note_event(["f1"], "post_verified", "never in nominated mode") == []


def test_none_mode_keeps_only_references_and_recent_frames(tmp_path):
    with _store(tmp_path) as store:
        store.add_frame("r", "ref", b"ref", captured_at=0.0, camera="scene", event="goal_reference",
                        metadata={"revision": None, "seq": 0})
        _frames(store, 5)
        kf = KeyframeMemory(store, "r", "none")
        kf.note_event(["f2"], "post_verified", "x")
        selected = kf.apply(add_references=["ref"])
        assert [e["frame_id"] for e in selected] == ["ref"]
        assert [f.frame_id for f in kf.context_frames(3)] == ["ref", "f4", "f5"]


def test_event_mode_is_bounded_and_context_budget_keeps_references(tmp_path):
    with _store(tmp_path, max_recent_frames=3, max_stored_frames=6) as store:
        store.add_frame("r", "ref", b"ref", captured_at=0.0, camera="scene", event="goal_reference",
                        metadata={"revision": None, "seq": 0})
        kf = KeyframeMemory(store, "r", "event", max_keyframes=2)
        for seq in range(1, 21):
            _frames(store, 1, start=seq)
            if seq % 4 == 0:
                kf.note_event([f"f{seq}"], "post_verified", f"attempt {seq}")
            kf.apply(add_references=["ref"])
        # storage stays bounded (6 stored + selected), the two newest event frames and the reference survive
        assert {f.frame_id for f in store.selected_frames("r")} == {"ref", "f16", "f20"}
        assert len(store.recent_frames("r")) <= 3
        # a tiny context budget still shows the reference first, then the newest keyframe
        assert [f.frame_id for f in kf.context_frames(2)] == ["ref", "f20"]
        assert kf.context_frames(0) == []
        with pytest.raises(ValueError):
            KeyframeMemory(store, "r", "sometimes")


def test_progress_summary_comes_only_from_verified_ledger_facts(tmp_path):
    with _store(tmp_path) as store:
        store.add_obligation("r", "red_in_box", {})
        store.add_obligation("r", "blue_in_box", {})
        store.start_attempt("r", "att-001", ["red_in_box"], {"skill": "pick_place", "arm": "left"})
        store.record_evidence("r", "ev-1", "verification", {})
        store.finish_attempt("r", "att-001", "verified", evidence_ids=["ev-1"], verified_obligation_ids=["red_in_box"],
                             effect_key="red_in_box#1", effect_payload={"predicate": "red_in_box"})
        store.start_attempt("r", "att-002", ["blue_in_box"],
                            {"skill": "pick_place", "arm": "right", "object_id": "blue", "effect": "blue_in_box"})
        store.record_event("r", "att-002:outcome", "attempt_outcome",
                           {"failure_kind": "backend_claimed_success_unverified", "reason": "block not in box"},
                           attempt_id="att-002")
        store.finish_attempt("r", "att-002", "failed")
        # a planner-claimed completion lives only in a proposal event: it must not appear as done
        store.record_event("r", "att-003:proposal", "proposal", {"claims": "blue_in_box is done"})
        record = build_progress(store, "r")
        assert [c["obligation"] for c in record["completed"]] == ["red_in_box"]
        assert record["completed"][0]["evidence_ids"] == ["ev-1"]
        assert [p["obligation"] for p in record["pending"]] == ["blue_in_box"]
        assert record["failed_attempts"][0]["failure_kind"] == "backend_claimed_success_unverified"
        text = record["summary"]
        assert "1 of 2" in text and "Verified: red_in_box" in text and "Not done yet: blue_in_box" in text
        assert "backend_claimed_success_unverified" in text and "claims" not in text
        seen = inspect_run(store, "r")
        assert seen["summary"] == text and seen["progress"]["completed"][0]["obligation"] == "red_in_box"
        assert seen["ledger"]["attempts"][1]["outcome"]["failure_kind"] == "backend_claimed_success_unverified"
