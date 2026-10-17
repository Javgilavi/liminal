"""Read-only view of one run's memory for the UI and for evaluation: ``inspect_run(db, run_id)``."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .keyframes import SELECTED_KIND, KeyframeMemory, plain
from .progress import attempt_outcomes, build_progress
from .store import MemoryStore

CONFIG_KIND = "memory_config"
EXPERIENCE_KIND = "experience_hits"
SNAPSHOT_KIND = "progress_snapshot"


def inspect_run(db_path: str | Path | MemoryStore, run_id: str) -> dict[str, Any]:
    """JSON-able memory picture of a run: ledger, keyframes, summary text and experience hits.

    Opens the database without recovering open attempts, so inspecting never changes the ledger.
    Raises ``KeyError`` for an unknown run.
    """
    owns = not isinstance(db_path, MemoryStore)
    store = MemoryStore(db_path) if owns else db_path
    try:
        run = store.get_run(run_id)
        outcomes = attempt_outcomes(store, run_id)
        config = _note(store, run_id, CONFIG_KIND, "config")
        mode = str(config.get("keyframes", "event"))
        keyframes = KeyframeMemory(store, run_id, mode if mode in ("event", "nominated", "none") else "event")
        described = keyframes.describe()
        progress = build_progress(store, run_id)
        hits = _note(store, run_id, EXPERIENCE_KIND, "latest")
        return {
            "run": {"run_id": run_id, "status": run.status, "goal": plain(run.goal), "metadata": plain(run.metadata),
                    "created_at": run.created_at},
            "ledger": {
                "obligations": [{"obligation_id": o.obligation_id, "status": o.status,
                                 "evidence_ids": list(o.evidence_ids)} for o in store.obligations(run_id)],
                "attempts": [{"attempt_id": a.attempt_id, "status": a.status, "obligation_ids": list(a.obligation_ids),
                              "evidence_ids": list(a.evidence_ids), "skill": a.payload.get("skill"),
                              "arm": a.payload.get("arm"), "object_id": a.payload.get("object_id"),
                              "effect": a.payload.get("effect"), "outcome": outcomes.get(a.attempt_id)}
                             for a in store.attempts(run_id)],
                "verified_effects": [e.effect_key for e in store.verified_effects(run_id)],
                "budgets": {name: store.budget(run_id, name)[0] for name in ("attempts", "model_calls")},
            },
            "keyframes": {**described, "contexts": [
                {"context_id": c.context_id, "reason": c.reason, "frame_ids": [f.frame_id for f in c.frames],
                 "missing_frame_ids": list(c.missing_frame_ids)} for c in store.contexts(run_id)[-5:]]},
            "progress": {k: v for k, v in progress.items() if k != "summary"},
            "summary": progress["summary"],
            "experience": {"mode": config.get("experience_mode", "off"),
                           "frozen": bool(config.get("experience_frozen")), "db": config.get("experience_db"), **hits},
            "config": config,
        }
    finally:
        if owns:
            store.close()


def _note(store: MemoryStore, run_id: str, kind: str, note_id: str) -> dict[str, Any]:
    try:
        return plain(store.get_note(run_id, kind, note_id).payload)
    except KeyError:
        return {}


__all__ = ["inspect_run", "SELECTED_KIND"]
