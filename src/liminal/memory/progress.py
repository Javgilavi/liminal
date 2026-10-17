"""Compact run progress record and text summary, built only from verified ledger facts.

MemER found that adding *predicted* subtask text to the planner's memory hurt. So nothing here is a
prediction: the record is derived from the SQLite ledger (obligation status with evidence ids, attempt
status, recorded attempt outcomes) and never from planner output or backend claims. Pending work is
listed as pending, never as done.
"""

from __future__ import annotations

from typing import Any

from .keyframes import plain
from .store import MemoryStore

OUTCOME_EVENT = "attempt_outcome"
MAX_LISTED = 6  # attempts per list; older ones are counted, not listed
FAILED_STATUSES = ("failed", "unknown", "interrupted", "refused", "partial")


def attempt_outcomes(store: MemoryStore, run_id: str) -> dict[str, dict[str, Any]]:
    """Latest recorded outcome (failure kind, reason) by attempt id."""
    found: dict[str, dict[str, Any]] = {}
    for event in store.events(run_id):
        if event.kind == OUTCOME_EVENT and event.attempt_id:
            found[event.attempt_id] = plain(event.payload)
    return found


def build_progress(store: MemoryStore, run_id: str) -> dict[str, Any]:
    """Structured progress: completed / pending / failed, each tied to ledger ids."""
    run = store.get_run(run_id)
    outcomes = attempt_outcomes(store, run_id)
    attempts = store.attempts(run_id)
    verified_by_obligation: dict[str, Any] = {}
    for attempt in attempts:
        if attempt.status == "verified":
            for obligation_id in attempt.obligation_ids:
                verified_by_obligation[obligation_id] = attempt
    completed, pending = [], []
    for obligation in store.obligations(run_id):
        if obligation.status == "completed":
            attempt = verified_by_obligation.get(obligation.obligation_id)
            completed.append({
                "obligation": obligation.obligation_id, "evidence_ids": list(obligation.evidence_ids),
                "attempt_id": attempt.attempt_id if attempt else None,
                "skill": attempt.payload.get("skill") if attempt else None,
                "arm": attempt.payload.get("arm") if attempt else None})
        else:
            pending.append({"obligation": obligation.obligation_id, "status": obligation.status})
    failed_all = []
    for attempt in attempts:
        if attempt.status in FAILED_STATUSES:
            outcome = outcomes.get(attempt.attempt_id, {})
            failed_all.append({
                "attempt_id": attempt.attempt_id, "status": attempt.status,
                "skill": attempt.payload.get("skill"), "arm": attempt.payload.get("arm"),
                "object_id": attempt.payload.get("object_id"), "effect": attempt.payload.get("effect"),
                "failure_kind": outcome.get("failure_kind") or attempt.status,
                "reason": str(outcome.get("reason") or "")[:160]})
    open_attempts = [a.attempt_id for a in attempts if a.status == "started"]
    record = {
        "run_id": run_id,
        "goal": str(run.goal.get("text", "")),
        "source": "verified ledger facts only; no predictions",
        "completed": completed,
        "pending": pending,
        "failed_attempts": failed_all[-MAX_LISTED:],
        "failed_attempts_total": len(failed_all),
        "open_attempts": open_attempts,
        "attempts_total": len(attempts),
    }
    record["summary"] = summarize(record)
    return record


def summarize(record: dict[str, Any]) -> str:
    """Short text of the progress record. Mentions only what the ledger holds."""
    total = len(record["completed"]) + len(record["pending"])
    parts = [f"Ledger progress: {len(record['completed'])} of {total} goal obligations verified by independent "
             f"evidence after {record['attempts_total']} attempt(s)."]
    if record["completed"]:
        parts.append("Verified: " + "; ".join(
            f"{c['obligation']} ({c['attempt_id'] or 'reconciled'}"
            + (f", {c['skill']} on {c['arm']}" if c.get("skill") else "") + ")" for c in record["completed"]) + ".")
    if record["pending"]:
        parts.append("Not done yet: " + ", ".join(p["obligation"] for p in record["pending"]) + ".")
    if record["failed_attempts"]:
        shown = "; ".join(
            f"{f['attempt_id']} {f['skill'] or 'skill'} for {f['effect'] or '?'} ended {f['status']} "
            f"[{f['failure_kind']}]" for f in record["failed_attempts"])
        hidden = record["failed_attempts_total"] - len(record["failed_attempts"])
        parts.append("Failed or unresolved attempts: " + shown + (f" (+{hidden} older)" if hidden > 0 else "") + ".")
    if record["open_attempts"]:
        parts.append("Attempts still open: " + ", ".join(record["open_attempts"]) + ".")
    parts.append("Only the verified list is done; everything else must be re-checked on the current frames.")
    return " ".join(parts)
