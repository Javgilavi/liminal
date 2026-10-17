"""Dependency-light SQLite ledger and bounded visual frame store."""

from __future__ import annotations

import json
import math
import sqlite3
import time
import uuid
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any


def _json(value: Mapping[str, Any] | None) -> str:
    """Serialize a JSON object into a stable, detached representation."""
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise TypeError("payload must be a JSON object")
    try:
        return json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("payload must contain JSON-compatible values") from exc


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _payload(serialized: str) -> Mapping[str, Any]:
    return _freeze(json.loads(serialized))


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    goal: Mapping[str, Any]
    metadata: Mapping[str, Any]
    status: str
    created_at: float


@dataclass(frozen=True)
class AttemptRecord:
    run_id: str
    attempt_id: str
    status: str
    obligation_ids: tuple[str, ...]
    payload: Mapping[str, Any]
    started_at: float
    updated_at: float
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class ObligationRecord:
    run_id: str
    obligation_id: str
    status: str
    payload: Mapping[str, Any]
    evidence_ids: tuple[str, ...]
    updated_at: float


@dataclass(frozen=True)
class EvidenceRecord:
    run_id: str
    evidence_id: str
    kind: str
    payload: Mapping[str, Any]
    created_at: float


@dataclass(frozen=True)
class EventRecord:
    run_id: str
    event_id: str
    kind: str
    payload: Mapping[str, Any]
    attempt_id: str | None
    effect_key: str | None
    verified: bool
    evidence_ids: tuple[str, ...]
    created_at: float


@dataclass(frozen=True)
class FrameRecord:
    run_id: str
    frame_id: str
    captured_at: float
    camera: str
    event: str
    metadata: Mapping[str, Any]
    data: bytes
    selected: bool = False
    selection_reason: str | None = None


@dataclass(frozen=True)
class FrameSelection:
    run_id: str
    context_id: str
    reason: str
    frames: tuple[FrameRecord, ...]
    created_at: float
    missing_frame_ids: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return not self.missing_frame_ids


@dataclass(frozen=True)
class NoteRecord:
    """Advisory side record (keyframe nominations, summaries, retrieved experience). Never evidence."""

    run_id: str
    kind: str
    note_id: str
    payload: Mapping[str, Any]
    created_at: float
    updated_at: float


@dataclass(frozen=True)
class Reconciliation:
    run_id: str
    interrupted_attempt_ids: tuple[str, ...]
    pending_obligation_ids: tuple[str, ...]
    completed_obligation_ids: tuple[str, ...]
    unresolved_attempt_ids: tuple[str, ...] = ()
    unknown_obligation_ids: tuple[str, ...] = ()
    requires_fresh_observation: bool = True
    auto_resend: bool = False


class MemoryStore:
    """SQLite-backed run ledger with bounded recent and selected frame history.

    Attempts are bookkeeping only. Runtime recovery can opt in to marking attempts
    left ``started`` as ``interrupted``; inspection connections leave them untouched.
    A runtime should hold one writer lock per episode database. This store never
    retries or dispatches work.
    """

    TERMINAL_ATTEMPT_STATUSES = frozenset(
        {"verified", "failed", "unknown", "interrupted", "refused", "partial"}
    )

    def __init__(
        self,
        path: str | Path,
        *,
        clock: Callable[[], float] = time.time,
        max_recent_frames: int = 12,
        max_stored_frames: int = 64,
        max_stored_frame_bytes: int = 16 * 1024 * 1024,
        max_selected_frames: int = 24,
        recover_open_attempts: bool = False,
    ) -> None:
        if (
            max_recent_frames < 0 or max_stored_frames < 1 or max_selected_frames < 1
            or max_stored_frame_bytes < 1
        ):
            raise ValueError("frame limits must be non-negative, with storage limits at least one")
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self.max_recent_frames = max_recent_frames
        self.max_stored_frames = max_stored_frames
        self.max_stored_frame_bytes = max_stored_frame_bytes
        self.max_selected_frames = max_selected_frames
        self.recover_open_attempts = recover_open_attempts
        self._recent: dict[str, deque[str]] = {}
        self._db = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        if self.path != ":memory:":
            self._db.execute("PRAGMA journal_mode = WAL")
        self._create_schema()
        if self.recover_open_attempts:
            self._recover_open_attempts()

    def close(self) -> None:
        self._db.close()

    def __enter__(self) -> MemoryStore:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _create_schema(self) -> None:
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY, goal_json TEXT NOT NULL, metadata_json TEXT NOT NULL,
                status TEXT NOT NULL, created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS obligations (
                run_id TEXT NOT NULL, obligation_id TEXT NOT NULL, status TEXT NOT NULL,
                payload_json TEXT NOT NULL, evidence_json TEXT NOT NULL, updated_at REAL NOT NULL,
                PRIMARY KEY (run_id, obligation_id),
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            );
            CREATE TABLE IF NOT EXISTS evidence (
                run_id TEXT NOT NULL, evidence_id TEXT NOT NULL, kind TEXT NOT NULL,
                payload_json TEXT NOT NULL, created_at REAL NOT NULL,
                PRIMARY KEY (run_id, evidence_id),
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            );
            CREATE TABLE IF NOT EXISTS attempts (
                run_id TEXT NOT NULL, attempt_id TEXT NOT NULL, status TEXT NOT NULL,
                obligation_json TEXT NOT NULL, payload_json TEXT NOT NULL,
                evidence_json TEXT NOT NULL, started_at REAL NOT NULL, updated_at REAL NOT NULL,
                PRIMARY KEY (run_id, attempt_id),
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            );
            CREATE TABLE IF NOT EXISTS events (
                run_id TEXT NOT NULL, event_id TEXT NOT NULL, kind TEXT NOT NULL,
                payload_json TEXT NOT NULL, attempt_id TEXT, effect_key TEXT,
                verified INTEGER NOT NULL, evidence_json TEXT NOT NULL, created_at REAL NOT NULL,
                PRIMARY KEY (run_id, event_id),
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            );
            CREATE UNIQUE INDEX IF NOT EXISTS unique_verified_effect
                ON events(run_id, effect_key) WHERE verified = 1 AND effect_key IS NOT NULL;
            CREATE TABLE IF NOT EXISTS frames (
                run_id TEXT NOT NULL, frame_id TEXT NOT NULL, captured_at REAL NOT NULL,
                camera TEXT NOT NULL, event TEXT NOT NULL, metadata_json TEXT NOT NULL,
                data BLOB NOT NULL, created_at REAL NOT NULL,
                PRIMARY KEY (run_id, frame_id),
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            );
            CREATE INDEX IF NOT EXISTS frame_order ON frames(run_id, captured_at, frame_id);
            CREATE TABLE IF NOT EXISTS selected_frames (
                run_id TEXT NOT NULL, frame_id TEXT NOT NULL, reason TEXT NOT NULL,
                selected_at REAL NOT NULL, priority INTEGER NOT NULL,
                PRIMARY KEY (run_id, frame_id),
                FOREIGN KEY (run_id, frame_id) REFERENCES frames(run_id, frame_id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS context_selections (
                run_id TEXT NOT NULL, context_id TEXT NOT NULL, reason TEXT NOT NULL,
                created_at REAL NOT NULL, PRIMARY KEY (run_id, context_id),
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            );
            CREATE TABLE IF NOT EXISTS context_items (
                run_id TEXT NOT NULL, context_id TEXT NOT NULL, frame_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL, PRIMARY KEY (run_id, context_id, frame_id),
                FOREIGN KEY (run_id, context_id) REFERENCES context_selections(run_id, context_id)
                    ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS run_notes (
                run_id TEXT NOT NULL, kind TEXT NOT NULL, note_id TEXT NOT NULL,
                payload_json TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL,
                PRIMARY KEY (run_id, kind, note_id),
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            );
            CREATE TABLE IF NOT EXISTS budgets (
                run_id TEXT NOT NULL, budget_name TEXT NOT NULL, consumed REAL NOT NULL,
                limit_value REAL, updated_at REAL NOT NULL,
                PRIMARY KEY (run_id, budget_name),
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            );
            """
        )
        self._migrate_context_items()

    def _migrate_context_items(self) -> None:
        """Keep frame IDs in old context snapshots if their bytes are evicted."""
        foreign_keys = self._db.execute("PRAGMA foreign_key_list(context_items)").fetchall()
        if not any(row["table"] == "frames" for row in foreign_keys):
            return
        self._db.execute("BEGIN IMMEDIATE")
        try:
            self._db.execute("ALTER TABLE context_items RENAME TO context_items_old")
            self._db.execute(
                "CREATE TABLE context_items (run_id TEXT NOT NULL, context_id TEXT NOT NULL, frame_id TEXT NOT NULL, "
                "ordinal INTEGER NOT NULL, PRIMARY KEY (run_id, context_id, frame_id), "
                "FOREIGN KEY (run_id, context_id) REFERENCES context_selections(run_id, context_id) ON DELETE CASCADE)"
            )
            self._db.execute(
                "INSERT INTO context_items SELECT run_id, context_id, frame_id, ordinal FROM context_items_old"
            )
            self._db.execute("DROP TABLE context_items_old")
            self._db.execute("COMMIT")
        except Exception:
            if self._db.in_transaction:
                self._db.execute("ROLLBACK")
            raise

    def _recover_open_attempts(self) -> None:
        now = self._clock()
        self._db.execute(
            "UPDATE attempts SET status='interrupted', updated_at=? WHERE status='started'", (now,)
        )

    def _require_run(self, run_id: str) -> None:
        row = self._db.execute("SELECT 1 FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown run: {run_id}")

    def create_run(
        self, run_id: str, goal: Mapping[str, Any], metadata: Mapping[str, Any] | None = None
    ) -> RunRecord:
        goal_json = _json(goal)
        metadata_json = _json(metadata)
        existing = self._db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if existing is not None:
            if existing["goal_json"] != goal_json or existing["metadata_json"] != metadata_json:
                raise ValueError(f"run ID already exists with different goal or metadata: {run_id}")
            return self.get_run(run_id)
        now = self._clock()
        self._db.execute(
            "INSERT INTO runs VALUES (?, ?, ?, 'active', ?)", (run_id, goal_json, metadata_json, now)
        )
        return self.get_run(run_id)

    def set_run_status(self, run_id: str, status: str) -> RunRecord:
        if status not in {"active", "paused", "completed", "failed", "stopped"}:
            raise ValueError(f"invalid run status: {status}")
        self._require_run(run_id)
        self._db.execute("UPDATE runs SET status=? WHERE run_id=?", (status, run_id))
        return self.get_run(run_id)

    def get_run(self, run_id: str) -> RunRecord:
        row = self._db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown run: {run_id}")
        return RunRecord(
            run_id,
            _payload(row["goal_json"]),
            _payload(row["metadata_json"]),
            row["status"],
            row["created_at"],
        )

    def add_obligation(
        self, run_id: str, obligation_id: str, payload: Mapping[str, Any]
    ) -> ObligationRecord:
        self._require_run(run_id)
        now = self._clock()
        self._db.execute(
            "INSERT INTO obligations VALUES (?, ?, 'pending', ?, '[]', ?)",
            (run_id, obligation_id, _json(payload), now),
        )
        return self.get_obligation(run_id, obligation_id)

    def get_obligation(self, run_id: str, obligation_id: str) -> ObligationRecord:
        row = self._db.execute(
            "SELECT * FROM obligations WHERE run_id=? AND obligation_id=?", (run_id, obligation_id)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown obligation: {obligation_id}")
        return ObligationRecord(
            run_id, obligation_id, row["status"], _payload(row["payload_json"]),
            tuple(json.loads(row["evidence_json"])), row["updated_at"]
        )

    def obligations(self, run_id: str, status: str | None = None) -> tuple[ObligationRecord, ...]:
        self._require_run(run_id)
        query = "SELECT obligation_id FROM obligations WHERE run_id=?"
        args: tuple[Any, ...] = (run_id,)
        if status is not None:
            query += " AND status=?"
            args += (status,)
        query += " ORDER BY obligation_id"
        return tuple(self.get_obligation(run_id, row[0]) for row in self._db.execute(query, args))

    def complete_obligation(self, run_id: str, obligation_id: str, evidence_ids: Iterable[str]) -> ObligationRecord:
        return self.set_obligation_status(run_id, obligation_id, "completed", evidence_ids=evidence_ids)

    def set_obligation_status(
        self,
        run_id: str,
        obligation_id: str,
        status: str,
        *,
        evidence_ids: Iterable[str] = (),
    ) -> ObligationRecord:
        """Reconcile current obligation state using evidence from an observer.

        Reopening a previously completed obligation is allowed when fresh evidence
        shows the current world no longer satisfies it. Historical events remain.
        """
        if status not in {"pending", "completed", "unknown"}:
            raise ValueError(f"invalid obligation status: {status}")
        ids = self._validated_evidence_ids(run_id, evidence_ids)
        if status == "completed" and not ids:
            raise ValueError("verified completion requires at least one evidence ID")
        current = self._db.execute(
            "SELECT status, updated_at FROM obligations WHERE run_id=? AND obligation_id=?",
            (run_id, obligation_id),
        ).fetchone()
        if current is None:
            raise KeyError(f"unknown obligation: {obligation_id}")
        if current["status"] == "completed" and status in {"pending", "unknown"}:
            if not ids:
                raise ValueError("revising a completed obligation requires fresh evidence IDs")
            if any(self.get_evidence(run_id, evidence_id).created_at <= current["updated_at"] for evidence_id in ids):
                raise ValueError("revising a completed obligation requires evidence recorded after completion")
        now = self._clock()
        cursor = self._db.execute(
            "UPDATE obligations SET status=?, evidence_json=?, updated_at=? "
            "WHERE run_id=? AND obligation_id=?",
            (status, json.dumps(ids), now, run_id, obligation_id),
        )
        if cursor.rowcount != 1:
            raise KeyError(f"unknown obligation: {obligation_id}")
        return self.get_obligation(run_id, obligation_id)

    def start_attempt(
        self,
        run_id: str,
        attempt_id: str,
        obligation_ids: Iterable[str] = (),
        payload: Mapping[str, Any] | None = None,
    ) -> AttemptRecord:
        self._require_run(run_id)
        ids = tuple(dict.fromkeys(str(value) for value in obligation_ids))
        for obligation_id in ids:
            self.get_obligation(run_id, obligation_id)
        now = self._clock()
        self._db.execute(
            "INSERT INTO attempts VALUES (?, ?, 'started', ?, ?, '[]', ?, ?)",
            (run_id, attempt_id, json.dumps(ids), _json(payload), now, now),
        )
        return self.get_attempt(run_id, attempt_id)

    def finish_attempt(
        self,
        run_id: str,
        attempt_id: str,
        status: str,
        *,
        evidence_ids: Iterable[str] = (),
        verified_obligation_ids: Iterable[str] = (),
        effect_key: str | None = None,
        effect_kind: str = "verified_effect",
        effect_payload: Mapping[str, Any] | None = None,
        reconcile: bool = False,
    ) -> AttemptRecord:
        if status not in self.TERMINAL_ATTEMPT_STATUSES:
            raise ValueError(f"invalid terminal attempt status: {status}")
        ids = self._validated_evidence_ids(run_id, evidence_ids)
        if status == "verified" and not ids:
            raise ValueError("verified attempts require evidence IDs")
        verified_ids = tuple(dict.fromkeys(verified_obligation_ids))
        if verified_ids and (status != "verified" or not ids):
            raise ValueError("verified obligations require a verified attempt and evidence IDs")
        attempt = self.get_attempt(run_id, attempt_id)
        if attempt.status != "started" and not reconcile:
            if attempt.status != status or attempt.evidence_ids != ids:
                raise ValueError(f"attempt is already terminal: {attempt.status}")
            for obligation_id in verified_ids:
                obligation = self.get_obligation(run_id, obligation_id)
                if obligation.status != "completed" or obligation.evidence_ids != ids:
                    raise ValueError("repeated attempt completion differs from stored obligation state")
            if effect_key:
                effect = self._db.execute(
                    "SELECT event_id, kind, payload_json, evidence_json FROM events "
                    "WHERE run_id=? AND effect_key=? AND verified=1",
                    (run_id, effect_key),
                ).fetchone()
                if (
                    effect is None
                    or effect["event_id"] != f"{attempt_id}:effect"
                    or effect["kind"] != effect_kind
                    or effect["payload_json"] != _json(effect_payload)
                    or tuple(json.loads(effect["evidence_json"])) != ids
                ):
                    raise ValueError("repeated attempt effect differs from stored effect")
            return attempt
        if reconcile and attempt.status not in {"interrupted", "unknown", "partial"}:
            raise ValueError(f"attempt does not need reconciliation: {attempt.status}")
        if status == "verified" and any(
            self.get_evidence(run_id, evidence_id).created_at < attempt.started_at for evidence_id in ids
        ):
            raise ValueError("verified attempt evidence must be recorded after the attempt starts")
        if not set(verified_ids).issubset(attempt.obligation_ids):
            raise ValueError("verified obligation is not attached to this attempt")
        if effect_key and (status != "verified" or not ids):
            raise ValueError("verified effects require a verified attempt and evidence IDs")
        now = self._clock()
        try:
            self._db.execute("BEGIN IMMEDIATE")
            allowed_state = (
                "status IN ('started', 'interrupted', 'unknown', 'partial')"
                if reconcile
                else "status='started'"
            )
            cursor = self._db.execute(
                f"UPDATE attempts SET status=?, evidence_json=?, updated_at=? "
                f"WHERE run_id=? AND attempt_id=? AND {allowed_state}",
                (status, json.dumps(ids), now, run_id, attempt_id),
            )
            if cursor.rowcount != 1:
                raise ValueError(f"attempt is already terminal: {attempt.status}")
            for obligation_id in verified_ids:
                cursor = self._db.execute(
                    "UPDATE obligations SET status='completed', evidence_json=?, updated_at=? "
                    "WHERE run_id=? AND obligation_id=?",
                    (json.dumps(ids), now, run_id, obligation_id),
                )
                if cursor.rowcount != 1:
                    raise KeyError(f"unknown obligation: {obligation_id}")
            if effect_key:
                self._db.execute(
                    "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)",
                    (
                        run_id,
                        f"{attempt_id}:effect",
                        effect_kind,
                        _json(effect_payload),
                        attempt_id,
                        effect_key,
                        json.dumps(ids),
                        now,
                    ),
                )
            self._db.execute("COMMIT")
        except Exception:
            if self._db.in_transaction:
                self._db.execute("ROLLBACK")
            raise
        return self.get_attempt(run_id, attempt_id)

    def reconcile_attempt(
        self,
        run_id: str,
        attempt_id: str,
        status: str,
        *,
        evidence_ids: Iterable[str],
        verified_obligation_ids: Iterable[str] = (),
        effect_key: str | None = None,
        effect_kind: str = "verified_effect",
        effect_payload: Mapping[str, Any] | None = None,
    ) -> AttemptRecord:
        """Resolve an interrupted attempt from fresh observations, without dispatch."""
        attempt = self.get_attempt(run_id, attempt_id)
        if attempt.status not in {"interrupted", "unknown", "partial"}:
            raise ValueError(f"attempt does not need reconciliation: {attempt.status}")
        ids = self._validated_evidence_ids(run_id, evidence_ids)
        if not ids:
            raise ValueError("reconciliation requires fresh evidence IDs")
        if any(self.get_evidence(run_id, evidence_id).created_at <= attempt.updated_at for evidence_id in ids):
            raise ValueError("reconciliation evidence must be recorded after the unresolved attempt")
        return self.finish_attempt(
            run_id,
            attempt_id,
            status,
            evidence_ids=evidence_ids,
            verified_obligation_ids=verified_obligation_ids,
            effect_key=effect_key,
            effect_kind=effect_kind,
            effect_payload=effect_payload,
            reconcile=True,
        )

    def consume_budget(
        self,
        run_id: str,
        budget_name: str,
        amount: float = 1,
        *,
        limit: float | None = None,
    ) -> float:
        """Atomically consume a durable per-run budget and return its new total."""
        self._require_run(run_id)
        if isinstance(amount, bool) or not isinstance(amount, (int, float)):
            raise ValueError("budget amount must be positive and limit non-negative")
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, (int, float))):
            raise ValueError("budget amount must be positive and limit non-negative")
        try:
            amount_value = float(amount)
            limit_value = None if limit is None else float(limit)
        except (OverflowError, ValueError) as exc:
            raise ValueError("budget amount and limit must be finite numbers") from exc
        if (
            not math.isfinite(amount_value)
            or amount_value <= 0
            or (limit_value is not None and (not math.isfinite(limit_value) or limit_value < 0))
        ):
            raise ValueError("budget amount must be positive and limit non-negative")
        self._db.execute("BEGIN IMMEDIATE")
        try:
            # Re-read after acquiring the write lock so concurrent consumers cannot
            # both pass the same remaining-budget check.
            row = self._db.execute(
                "SELECT consumed, limit_value FROM budgets WHERE run_id=? AND budget_name=?",
                (run_id, budget_name),
            ).fetchone()
            consumed = float(row["consumed"]) if row else 0.0
            old_limit = row["limit_value"] if row else None
            if old_limit is not None and limit_value is not None and limit_value > float(old_limit):
                raise ValueError(f"budget limit cannot be increased: {budget_name}")
            effective_limit = limit_value if limit_value is not None else old_limit
            new_consumed = consumed + amount_value
            if not math.isfinite(new_consumed):
                raise ValueError(f"budget total is not finite: {budget_name}")
            if effective_limit is not None and new_consumed > float(effective_limit):
                raise ValueError(f"budget exhausted: {budget_name}")
            self._db.execute(
                "INSERT INTO budgets VALUES (?, ?, ?, ?, ?) ON CONFLICT(run_id, budget_name) DO UPDATE SET "
                "consumed=excluded.consumed, limit_value=COALESCE(excluded.limit_value, budgets.limit_value), "
                "updated_at=excluded.updated_at",
                (run_id, budget_name, new_consumed, effective_limit, self._clock()),
            )
            self._db.execute("COMMIT")
        except Exception:
            if self._db.in_transaction:
                self._db.execute("ROLLBACK")
            raise
        return new_consumed

    def budget(self, run_id: str, budget_name: str) -> tuple[float, float | None]:
        self._require_run(run_id)
        row = self._db.execute(
            "SELECT consumed, limit_value FROM budgets WHERE run_id=? AND budget_name=?",
            (run_id, budget_name),
        ).fetchone()
        return (0.0, None) if row is None else (float(row["consumed"]), row["limit_value"])

    def get_attempt(self, run_id: str, attempt_id: str) -> AttemptRecord:
        row = self._db.execute(
            "SELECT * FROM attempts WHERE run_id=? AND attempt_id=?", (run_id, attempt_id)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown attempt: {attempt_id}")
        return AttemptRecord(
            run_id, attempt_id, row["status"], tuple(json.loads(row["obligation_json"])),
            _payload(row["payload_json"]), row["started_at"], row["updated_at"],
            tuple(json.loads(row["evidence_json"]))
        )

    def attempts(self, run_id: str) -> tuple[AttemptRecord, ...]:
        self._require_run(run_id)
        rows = self._db.execute(
            "SELECT attempt_id FROM attempts WHERE run_id=? ORDER BY started_at, attempt_id", (run_id,)
        ).fetchall()
        return tuple(self.get_attempt(run_id, row[0]) for row in rows)

    def record_evidence(
        self, run_id: str, evidence_id: str, kind: str, payload: Mapping[str, Any]
    ) -> EvidenceRecord:
        self._require_run(run_id)
        now = self._clock()
        self._db.execute(
            "INSERT INTO evidence VALUES (?, ?, ?, ?, ?)", (run_id, evidence_id, kind, _json(payload), now)
        )
        return self.get_evidence(run_id, evidence_id)

    def get_evidence(self, run_id: str, evidence_id: str) -> EvidenceRecord:
        row = self._db.execute(
            "SELECT * FROM evidence WHERE run_id=? AND evidence_id=?", (run_id, evidence_id)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown evidence: {evidence_id}")
        return EvidenceRecord(run_id, evidence_id, row["kind"], _payload(row["payload_json"]), row["created_at"])

    def _validated_evidence_ids(self, run_id: str, evidence_ids: Iterable[str]) -> tuple[str, ...]:
        ids = tuple(dict.fromkeys(str(value) for value in evidence_ids))
        for evidence_id in ids:
            self.get_evidence(run_id, evidence_id)
        return ids

    def record_event(
        self,
        run_id: str,
        event_id: str,
        kind: str,
        payload: Mapping[str, Any],
        *,
        attempt_id: str | None = None,
        effect_key: str | None = None,
        verified: bool = False,
        evidence_ids: Iterable[str] = (),
    ) -> EventRecord:
        self._require_run(run_id)
        ids = self._validated_evidence_ids(run_id, evidence_ids)
        if verified and (not effect_key or not ids):
            raise ValueError("verified effects require a unique effect_key and evidence IDs")
        if attempt_id is not None:
            self.get_attempt(run_id, attempt_id)
        now = self._clock()
        try:
            self._db.execute(
                "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, event_id, kind, _json(payload), attempt_id, effect_key, int(verified), json.dumps(ids), now),
            )
        except sqlite3.IntegrityError as exc:
            if verified and effect_key:
                existing = self._db.execute(
                    "SELECT event_id FROM events WHERE run_id=? AND effect_key=? AND verified=1",
                    (run_id, effect_key),
                ).fetchone()
                if existing is not None:
                    return self.get_event(run_id, existing["event_id"])
            raise ValueError(f"duplicate event or verified effect: {event_id}") from exc
        return self.get_event(run_id, event_id)

    def get_event(self, run_id: str, event_id: str) -> EventRecord:
        row = self._db.execute(
            "SELECT * FROM events WHERE run_id=? AND event_id=?", (run_id, event_id)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown event: {event_id}")
        return EventRecord(
            run_id, event_id, row["kind"], _payload(row["payload_json"]), row["attempt_id"],
            row["effect_key"], bool(row["verified"]), tuple(json.loads(row["evidence_json"])), row["created_at"]
        )

    def events(self, run_id: str, *, verified_only: bool = False) -> tuple[EventRecord, ...]:
        self._require_run(run_id)
        query = "SELECT event_id FROM events WHERE run_id=?"
        args: tuple[Any, ...] = (run_id,)
        if verified_only:
            query += " AND verified=1"
        query += " ORDER BY created_at, event_id"
        return tuple(self.get_event(run_id, row[0]) for row in self._db.execute(query, args))

    def verified_effects(self, run_id: str) -> tuple[EventRecord, ...]:
        return self.events(run_id, verified_only=True)

    def reconcile(self, run_id: str) -> Reconciliation:
        self._require_run(run_id)
        unresolved = tuple(
            row[0] for row in self._db.execute(
                "SELECT attempt_id FROM attempts WHERE run_id=? "
                "AND status IN ('started', 'interrupted', 'unknown', 'partial') "
                "ORDER BY started_at, attempt_id",
                (run_id,),
            )
        )
        pending = tuple(item.obligation_id for item in self.obligations(run_id, "pending"))
        completed = tuple(item.obligation_id for item in self.obligations(run_id, "completed"))
        unknown = tuple(item.obligation_id for item in self.obligations(run_id, "unknown"))
        interrupted = tuple(
            row[0] for row in self._db.execute(
                "SELECT attempt_id FROM attempts WHERE run_id=? AND status='interrupted' "
                "ORDER BY started_at, attempt_id",
                (run_id,),
            )
        )
        return Reconciliation(
            run_id, interrupted, pending, completed,
            unresolved_attempt_ids=unresolved, unknown_obligation_ids=unknown
        )

    def add_frame(
        self,
        run_id: str,
        frame_id: str,
        data: bytes | bytearray | memoryview,
        *,
        captured_at: float,
        camera: str,
        event: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> FrameRecord:
        self._require_run(run_id)
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("frame data must be encoded image bytes")
        blob = bytes(data)
        if len(blob) > self.max_stored_frame_bytes:
            raise ValueError("frame exceeds configured byte limit")
        now = self._clock()
        self._db.execute(
            "INSERT INTO frames VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, frame_id, float(captured_at), camera, event, _json(metadata), sqlite3.Binary(blob), now),
        )
        recent = self._recent.setdefault(run_id, deque(maxlen=self.max_recent_frames or None))
        if self.max_recent_frames:
            recent.append(frame_id)
        self._prune_frames(run_id)
        return self.get_frame(run_id, frame_id)

    def _prune_frames(self, run_id: str) -> None:
        rows = self._db.execute(
            "SELECT f.frame_id FROM frames f LEFT JOIN selected_frames s "
            "ON f.run_id=s.run_id AND f.frame_id=s.frame_id "
            "WHERE f.run_id=? AND s.frame_id IS NULL ORDER BY f.captured_at DESC, f.frame_id DESC",
            (run_id,),
        ).fetchall()
        selected = self._db.execute(
            "SELECT count(*), COALESCE(sum(length(data)), 0) FROM frames f "
            "JOIN selected_frames s ON f.run_id=s.run_id AND f.frame_id=s.frame_id WHERE f.run_id=?",
            (run_id,),
        ).fetchone()
        remaining_count = max(0, self.max_stored_frames - selected[0])
        remaining_bytes = max(0, self.max_stored_frame_bytes - selected[1])
        keep: set[str] = set()
        for row in rows:
            frame_id = row[0]
            size = self._db.execute(
                "SELECT length(data) FROM frames WHERE run_id=? AND frame_id=?", (run_id, frame_id)
            ).fetchone()[0]
            if remaining_count > 0 and size <= remaining_bytes:
                keep.add(frame_id)
                remaining_count -= 1
                remaining_bytes -= size
        for row in rows:
            frame_id = row[0]
            if frame_id in keep:
                continue
            frame_id = row[0]
            self._db.execute("DELETE FROM frames WHERE run_id=? AND frame_id=?", (run_id, frame_id))
            recent = self._recent.get(run_id)
            if recent is not None:
                self._recent[run_id] = deque(
                    (item for item in recent if item != frame_id),
                    maxlen=self.max_recent_frames or None,
                )

    def get_frame(self, run_id: str, frame_id: str) -> FrameRecord:
        row = self._db.execute(
            "SELECT f.*, s.reason AS selection_reason FROM frames f LEFT JOIN selected_frames s "
            "ON f.run_id=s.run_id AND f.frame_id=s.frame_id WHERE f.run_id=? AND f.frame_id=?",
            (run_id, frame_id),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown frame: {frame_id}")
        return FrameRecord(
            run_id, frame_id, row["captured_at"], row["camera"], row["event"], _payload(row["metadata_json"]),
            bytes(row["data"]), row["selection_reason"] is not None, row["selection_reason"]
        )

    def select_frames(
        self,
        run_id: str,
        frame_ids: Iterable[str],
        *,
        reason: str,
        priority: int = 0,
    ) -> tuple[FrameRecord, ...]:
        ids = tuple(dict.fromkeys(frame_ids))
        if len(ids) > self.max_selected_frames:
            raise ValueError("selected frame count exceeds configured limit")
        now = self._clock()
        self._require_run(run_id)
        for frame_id in ids:
            self.get_frame(run_id, frame_id)
        if ids:
            placeholders = ",".join("?" for _ in ids)
            total_bytes = self._db.execute(
                f"SELECT COALESCE(sum(length(data)), 0) FROM frames "
                f"WHERE run_id=? AND frame_id IN ({placeholders})",
                (run_id, *ids),
            ).fetchone()[0]
        else:
            total_bytes = 0
        if total_bytes > self.max_stored_frame_bytes:
            raise ValueError("selected frames exceed configured byte limit")
        self._db.execute("BEGIN IMMEDIATE")
        try:
            self._db.execute("DELETE FROM selected_frames WHERE run_id=?", (run_id,))
            self._db.executemany(
                "INSERT INTO selected_frames VALUES (?, ?, ?, ?, ?)",
                [(run_id, frame_id, reason, now, priority) for frame_id in ids],
            )
            self._db.execute("COMMIT")
        except Exception:
            if self._db.in_transaction:
                self._db.execute("ROLLBACK")
            raise
        self._prune_frames(run_id)
        return tuple(self.get_frame(run_id, frame_id) for frame_id in ids)

    def selected_frames(self, run_id: str) -> tuple[FrameRecord, ...]:
        self._require_run(run_id)
        rows = self._db.execute(
            "SELECT frame_id FROM selected_frames WHERE run_id=? ORDER BY priority DESC, selected_at, frame_id",
            (run_id,),
        ).fetchall()
        return tuple(self.get_frame(run_id, row[0]) for row in rows)

    def recent_frames(self, run_id: str) -> tuple[FrameRecord, ...]:
        self._require_run(run_id)
        ids = list(self._recent.get(run_id, ()))
        if not ids and self.max_recent_frames:
            rows = self._db.execute(
                "SELECT frame_id FROM frames WHERE run_id=? ORDER BY captured_at DESC, frame_id DESC LIMIT ?",
                (run_id, self.max_recent_frames),
            ).fetchall()
            ids = [row[0] for row in reversed(rows)]
        return tuple(self.get_frame(run_id, frame_id) for frame_id in ids)

    def select_context(
        self,
        run_id: str,
        *,
        max_frames: int,
        reason: str,
        context_id: str | None = None,
    ) -> FrameSelection:
        """Persist and return a deterministic, bounded selected-plus-recent context."""
        if max_frames < 0:
            raise ValueError("max_frames must be non-negative")
        self._require_run(run_id)
        selected = list(self.selected_frames(run_id))
        recent = list(self.recent_frames(run_id))
        chosen = selected[:max_frames]
        chosen_ids = {frame.frame_id for frame in chosen}
        for frame in reversed(recent):
            if len(chosen) >= max_frames:
                break
            if frame.frame_id not in chosen_ids:
                chosen.append(frame)
                chosen_ids.add(frame.frame_id)
        chosen.sort(key=lambda frame: (frame.captured_at, frame.frame_id))
        now = self._clock()
        context_id = context_id or f"ctx-{uuid.uuid4().hex}"
        self._db.execute(
            "INSERT INTO context_selections VALUES (?, ?, ?, ?)", (run_id, context_id, reason, now)
        )
        self._db.executemany(
            "INSERT INTO context_items VALUES (?, ?, ?, ?)",
            [(run_id, context_id, frame.frame_id, index) for index, frame in enumerate(chosen)],
        )
        return FrameSelection(run_id, context_id, reason, tuple(chosen), now)

    def get_context(self, run_id: str, context_id: str) -> FrameSelection:
        row = self._db.execute(
            "SELECT * FROM context_selections WHERE run_id=? AND context_id=?", (run_id, context_id)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown context selection: {context_id}")
        ids = self._db.execute(
            "SELECT frame_id FROM context_items WHERE run_id=? AND context_id=? ORDER BY ordinal",
            (run_id, context_id),
        ).fetchall()
        frames: list[FrameRecord] = []
        missing: list[str] = []
        for item in ids:
            try:
                frames.append(self.get_frame(run_id, item[0]))
            except KeyError:
                missing.append(item[0])
        return FrameSelection(
            run_id, context_id, row["reason"], tuple(frames), row["created_at"], tuple(missing)
        )

    # -- advisory notes and persisted planner contexts (additive; not part of the verified ledger) --------

    def put_note(self, run_id: str, kind: str, note_id: str, payload: Mapping[str, Any]) -> NoteRecord:
        """Insert or replace one note. Notes keep their first insertion order, and are never evidence."""
        self._require_run(run_id)
        now = self._clock()
        self._db.execute(
            "INSERT INTO run_notes VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(run_id, kind, note_id) DO UPDATE SET "
            "payload_json=excluded.payload_json, updated_at=excluded.updated_at",
            (run_id, kind, note_id, _json(payload), now, now),
        )
        return self.get_note(run_id, kind, note_id)

    def get_note(self, run_id: str, kind: str, note_id: str) -> NoteRecord:
        row = self._db.execute(
            "SELECT * FROM run_notes WHERE run_id=? AND kind=? AND note_id=?", (run_id, kind, note_id)
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown note: {kind}/{note_id}")
        return NoteRecord(run_id, kind, note_id, _payload(row["payload_json"]), row["created_at"], row["updated_at"])

    def notes(self, run_id: str, kind: str | None = None) -> tuple[NoteRecord, ...]:
        self._require_run(run_id)
        query = "SELECT kind, note_id FROM run_notes WHERE run_id=?"
        args: tuple[Any, ...] = (run_id,)
        if kind is not None:
            query += " AND kind=?"
            args += (kind,)
        query += " ORDER BY rowid"
        return tuple(self.get_note(run_id, row[0], row[1]) for row in self._db.execute(query, args))

    def delete_note(self, run_id: str, kind: str, note_id: str) -> None:
        self._require_run(run_id)
        self._db.execute("DELETE FROM run_notes WHERE run_id=? AND kind=? AND note_id=?", (run_id, kind, note_id))

    def persist_context(self, run_id: str, frame_ids: Iterable[str], *, reason: str,
                        context_id: str | None = None) -> FrameSelection:
        """Record an explicitly assembled planner context (same tables as ``select_context``)."""
        self._require_run(run_id)
        frames = tuple(self.get_frame(run_id, frame_id) for frame_id in dict.fromkeys(frame_ids))
        now = self._clock()
        context_id = context_id or f"ctx-{uuid.uuid4().hex}"
        self._db.execute("INSERT INTO context_selections VALUES (?, ?, ?, ?)", (run_id, context_id, reason, now))
        self._db.executemany(
            "INSERT INTO context_items VALUES (?, ?, ?, ?)",
            [(run_id, context_id, frame.frame_id, index) for index, frame in enumerate(frames)],
        )
        return FrameSelection(run_id, context_id, reason, frames, now)

    def contexts(self, run_id: str) -> tuple[FrameSelection, ...]:
        """Persisted context selections, oldest first (bytes of evicted frames are reported missing)."""
        self._require_run(run_id)
        rows = self._db.execute(
            "SELECT context_id FROM context_selections WHERE run_id=? ORDER BY created_at, rowid", (run_id,)
        ).fetchall()
        return tuple(self.get_context(run_id, row[0]) for row in rows)
