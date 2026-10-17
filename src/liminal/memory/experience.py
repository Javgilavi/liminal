"""Cross-run experience memory in the style of Harness VLA: procedural traces plus general failure records.

Two kinds of record outlive a run, in their own SQLite file (separate from any run ledger):

* **procedural traces** - the verified step sequence of a run (skill, object/destination roles, arm,
  retries, duration) with the scenario and capability-registry version it ran under;
* **failure records** - skill, object roles and failure kind (``backend_claimed_success_unverified``,
  ``grasp_failed``, ``stale_plan``, ``unknown_evidence`` ...) for attempts and plans that did not hold.

Retrieval is filtered by compatibility *before* anything is ranked: same registry version and the same
embodiment arms, and at least one overlapping object role. Hits are advisory prompt context only. They
never mark progress, never reach a motor, and every proposal they inspire still passes validation and
fresh-evidence verification. A frozen store (``frozen=True``) is read-only so an evaluation corpus cannot
absorb held-out outcomes; ``corpus_hash`` identifies it.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from .keyframes import plain
from .progress import FAILED_STATUSES, attempt_outcomes
from .store import MemoryStore

EXPERIENCE_MODES = ("off", "read", "readwrite")
FAILURE_KINDS = (
    "backend_claimed_success_unverified", "backend_failed", "grasp_failed", "verification_failed",
    "unknown_evidence", "stale_plan", "effect_not_pending", "emergency_stop", "interrupted",
)
ADVISORY = ("Advisory experience from earlier runs. It is NOT evidence about the present scene and does not "
            "mark any obligation done; re-ground every object on the current frames and plan only registered "
            "steps.")
MAX_STEPS = 24
MAX_DETAIL = 160


class ExperienceStore:
    """SQLite store of procedural traces and failure records. ``writable`` False never creates or alters a file."""

    def __init__(self, path: str | Path, *, writable: bool = True, frozen: bool = False,
                 clock: Callable[[], float] = time.time):
        self.path = str(path)
        self.frozen = frozen
        self.writable = writable and not frozen
        self._clock = clock
        self._db: sqlite3.Connection | None = None
        if self.writable:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
            self._db.row_factory = sqlite3.Row
            self._create_schema()
        elif Path(self.path).exists():
            self._db = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, isolation_level=None,
                                       check_same_thread=False)
            self._db.row_factory = sqlite3.Row

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None

    def __enter__(self) -> ExperienceStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _create_schema(self) -> None:
        assert self._db is not None
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS traces (
                trace_id TEXT PRIMARY KEY, run_id TEXT NOT NULL UNIQUE, created_at REAL NOT NULL,
                scenario TEXT NOT NULL, registry_version TEXT NOT NULL, arms_json TEXT NOT NULL,
                goal_text TEXT NOT NULL, completed INTEGER NOT NULL, steps_json TEXT NOT NULL,
                roles_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS failures (
                failure_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, created_at REAL NOT NULL,
                registry_version TEXT NOT NULL, arms_json TEXT NOT NULL, skill TEXT NOT NULL,
                object_id TEXT NOT NULL, destination_id TEXT NOT NULL, arm TEXT NOT NULL, effect TEXT NOT NULL,
                kind TEXT NOT NULL, detail TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS failure_run ON failures(run_id);
            """
        )

    # -- writes -----------------------------------------------------------------------------

    def record_run(self, run_id: str, *, scenario: str, registry_version: str, arms: Sequence[str],
                   goal_text: str, completed: bool, steps: Sequence[dict[str, Any]],
                   failures: Sequence[dict[str, Any]]) -> bool:
        """Store (or replace, on resume) one run's trace and failures. False when the store is read-only."""
        if not self.writable or self._db is None:
            return False
        arms_json = json.dumps(sorted(arms))
        roles = sorted({str(r) for s in steps for r in (s.get("object_id"), s.get("destination_id")) if r}
                       | {str(r) for f in failures for r in (f.get("object_id"), f.get("destination_id")) if r})
        now = self._clock()
        self._db.execute("BEGIN IMMEDIATE")
        try:
            self._db.execute("DELETE FROM traces WHERE run_id=?", (run_id,))
            self._db.execute("DELETE FROM failures WHERE run_id=?", (run_id,))
            if steps:
                self._db.execute(
                    "INSERT INTO traces VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (f"trace:{run_id}", run_id, now, scenario, registry_version, arms_json, goal_text[:240],
                     int(completed), json.dumps(list(steps)[:MAX_STEPS]), json.dumps(roles)))
            for index, failure in enumerate(failures):
                self._db.execute(
                    "INSERT INTO failures VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (f"fail:{run_id}:{index}", run_id, now, registry_version, arms_json,
                     str(failure.get("skill") or ""), str(failure.get("object_id") or ""),
                     str(failure.get("destination_id") or ""), str(failure.get("arm") or ""),
                     str(failure.get("effect") or ""), str(failure.get("kind") or "unknown"),
                     str(failure.get("detail") or "")[:MAX_DETAIL]))
            self._db.execute("COMMIT")
        except Exception:
            if self._db.in_transaction:
                self._db.execute("ROLLBACK")
            raise
        return True

    # -- reads ------------------------------------------------------------------------------

    def corpus_hash(self) -> str:
        """Stable identity of the stored corpus, to record next to a frozen evaluation."""
        digest = hashlib.sha256()
        if self._db is not None:
            for table, order in (("traces", "trace_id"), ("failures", "failure_id")):
                for row in self._db.execute(f"SELECT * FROM {table} ORDER BY {order}"):
                    row_values = {k: row[k] for k in row.keys() if k != "created_at"}
                    digest.update(json.dumps(row_values, sort_keys=True).encode())
        return digest.hexdigest()[:16]

    def counts(self) -> dict[str, int]:
        if self._db is None:
            return {"traces": 0, "failures": 0}
        return {t: self._db.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in ("traces", "failures")}

    def retrieve(self, *, registry_version: str, arms: Iterable[str], roles: Iterable[str],
                 exclude_run_id: str | None = None, max_procedures: int = 2, max_failures: int = 4) -> dict[str, Any]:
        """Compatible, role-overlapping experience, newest and most relevant first. Advisory only."""
        wanted_arms = sorted(arms)
        wanted_roles = {str(r) for r in roles if r}
        result: dict[str, Any] = {"advisory": ADVISORY, "corpus_hash": self.corpus_hash(), "procedures": [],
                                  "failures": []}
        if self._db is None or not wanted_roles:
            return result
        procedures = []
        for row in self._db.execute("SELECT * FROM traces WHERE registry_version=? AND arms_json=? AND completed=1",
                                    (registry_version, json.dumps(wanted_arms))):
            if row["run_id"] == exclude_run_id:
                continue
            overlap = sorted(wanted_roles & set(json.loads(row["roles_json"])))
            if overlap:
                procedures.append((len(overlap), row["created_at"], {
                    "trace_id": row["trace_id"], "source_run": row["run_id"], "scenario": row["scenario"],
                    "goal": row["goal_text"], "completed": True, "role_overlap": overlap,
                    "steps": json.loads(row["steps_json"])}))
        procedures.sort(key=lambda item: (item[0], item[1]), reverse=True)
        result["procedures"] = [item[2] for item in procedures[:max_procedures]]
        grouped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        for row in self._db.execute("SELECT * FROM failures WHERE registry_version=? AND arms_json=? "
                                    "ORDER BY created_at, failure_id", (registry_version, json.dumps(wanted_arms))):
            if row["run_id"] == exclude_run_id:
                continue
            if not ({row["object_id"], row["destination_id"]} & wanted_roles):
                continue
            key = (row["skill"], row["object_id"], row["destination_id"], row["kind"])
            entry = grouped.setdefault(key, {
                "skill": row["skill"], "object_id": row["object_id"], "destination_id": row["destination_id"],
                "arm": row["arm"], "kind": row["kind"], "count": 0, "runs": [], "detail": "", "_last": 0.0})
            entry["count"] += 1
            if row["run_id"] not in entry["runs"]:
                entry["runs"].append(row["run_id"])
            entry["detail"] = row["detail"] or entry["detail"]
            entry["_last"] = row["created_at"]
        ranked = sorted(grouped.values(), key=lambda e: (e["count"], e["_last"]), reverse=True)[:max_failures]
        for entry in ranked:
            entry.pop("_last")
        result["failures"] = ranked
        return result


def hit_ids(experience: dict[str, Any]) -> list[str]:
    """Stable ids of retrieved items, for logs and inspection."""
    ids = [p["trace_id"] for p in experience.get("procedures", ())]
    ids += [f"failure:{f['skill']}:{f['object_id']}:{f['kind']}" for f in experience.get("failures", ())]
    return ids


def classify_failure(*, decision: str | None, error: str | None) -> str:
    """Name why an attempt did not count, from what happened (never from planner text).

    Verification only runs after the backend reports success, so ``decision`` None means the backend
    itself failed or timed out; a ``false`` or ``unknown`` decision means it claimed success and the
    fresh evidence disagreed or could not tell.
    """
    if error and "EmergencyStop" in error:
        return "emergency_stop"
    if decision is None:
        return "grasp_failed" if error and "grasp" in error.lower() else "backend_failed"
    if decision == "false":
        return "backend_claimed_success_unverified"
    if decision == "unknown":
        return "unknown_evidence"
    return "verification_failed"


def collect_run_experience(store: MemoryStore, run_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    """(verified steps, failure records, run completed) distilled from one run's ledger."""
    outcomes = attempt_outcomes(store, run_id)
    steps: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    tried: dict[str, int] = {}
    for attempt in store.attempts(run_id):
        payload = plain(attempt.payload)
        effect = str(payload.get("effect") or "")
        if attempt.status == "verified":
            steps.append({
                "skill": payload.get("skill"), "arm": payload.get("arm"), "object_id": payload.get("object_id"),
                "destination_id": payload.get("destination_id"), "effect": effect,
                "retries": tried.get(effect, 0), "duration_s": round(attempt.updated_at - attempt.started_at, 3),
                "source_attempt": attempt.attempt_id})
        elif attempt.status in FAILED_STATUSES:
            tried[effect] = tried.get(effect, 0) + 1
            outcome = outcomes.get(attempt.attempt_id, {})
            failures.append({
                "skill": payload.get("skill"), "arm": payload.get("arm"), "object_id": payload.get("object_id"),
                "destination_id": payload.get("destination_id"), "effect": effect,
                "kind": outcome.get("failure_kind") or ("interrupted" if attempt.status == "interrupted"
                                                         else "verification_failed"),
                "detail": outcome.get("reason", "")})
    for event in store.events(run_id):
        if event.kind == "plan_dropped":
            data = plain(event.payload)
            failures.append({
                "skill": data.get("skill"), "arm": data.get("arm"), "object_id": data.get("object_id"),
                "destination_id": data.get("destination_id"), "effect": data.get("effect") or "",
                "kind": "stale_plan" if data.get("reason") == "stale_revision" else str(data.get("reason")),
                "detail": f"plan dropped: {data.get('reason')}"})
    completed = all(o.status == "completed" for o in store.obligations(run_id))
    return steps, failures, completed
