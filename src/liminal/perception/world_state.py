"""Shared symbolic world state with an append-only event log.

Perception writes predicates, the orchestrator reads/awaits them. Conditions are
predicate names; prefix with "!" for negation (e.g. "!lid_closed").
"""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from liminal.perception.types import Verdict

Listener = Callable[[str, Any, Any], None]
_MISSING = object()


@dataclass(frozen=True)
class ObservationRecord:
    """Latest perception evidence for one predicate.

    Versions are process-local and monotonically increasing. ``captured_at`` uses
    the monotonic clock by default so callers can also impose an evidence age.
    """

    predicate: str
    value: bool | None
    version: int
    captured_at: float
    source: str
    evidence: str = ""
    confidence: float | None = None


class WorldState:
    def __init__(self, initial: dict[str, Any] | None = None, log_path: str | Path | None = None):
        self._facts: dict[str, Any] = dict(initial or {})
        self._observations: dict[str, ObservationRecord] = {}
        self._next_observation_version = 0
        self._changed = asyncio.Event()
        self._listeners: list[Listener] = []
        self._log_path = Path(log_path) if log_path else None
        if self._log_path:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self.events: list[dict[str, Any]] = []

    # -- reads -------------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        return self._facts.get(key, default)

    def evaluate(self, condition: str) -> bool | None:
        """Evaluate a condition without collapsing unknown into false.

        A missing or explicitly unknown fact satisfies neither ``x`` nor ``!x``.
        """

        key, negated = _condition_parts(condition)
        if key not in self._facts:
            return None
        value = _truth_value(self._facts[key])
        if value is None:
            return None
        return not value if negated else value

    def holds(
        self,
        condition: str,
        *,
        after_version: int | None = None,
        after_time: float | None = None,
        max_age_s: float | None = None,
    ) -> bool:
        if self.evaluate(condition) is not True:
            return False
        if after_version is None and after_time is None and max_age_s is None:
            return True
        record = self.observation(condition)
        if record is None:
            return False
        if after_version is not None and record.version <= after_version:
            return False
        if after_time is not None and record.captured_at < after_time:
            return False
        if max_age_s is not None:
            if max_age_s < 0:
                raise ValueError("maximum observation age cannot be negative")
            if time.monotonic() - record.captured_at > max_age_s:
                return False
        return True

    def known(self, condition: str) -> bool:
        return self.evaluate(condition) is not None

    def observation(self, condition: str) -> ObservationRecord | None:
        key, _ = _condition_parts(condition)
        return self._observations.get(key)

    def observation_version(self, condition: str) -> int:
        record = self.observation(condition)
        return record.version if record else 0

    def snapshot(self) -> dict[str, Any]:
        return dict(self._facts)

    # -- writes ------------------------------------------------------------------------

    def set(self, key: str, value: Any, source: str = "unknown") -> None:
        old = self._facts.get(key, _MISSING)
        invalidated = self._observations.pop(key, None)
        if old == value and invalidated is None:
            return
        self._facts[key] = value
        if old != value:
            previous = None if old is _MISSING else old
            self.log("fact", key=key, value=value, old=previous, source=source)
            for listener in list(self._listeners):
                listener(key, previous, value)
        else:
            self.log("observation_invalidated", key=key, source=source)
        self._notify_changed()

    def observe(
        self,
        key: str,
        value: bool | Verdict | None,
        *,
        source: str,
        captured_at: float | None = None,
        evidence: str = "",
        confidence: float | None = None,
    ) -> ObservationRecord:
        """Record fresh perception evidence, even when its value is unchanged."""

        observed = _observed_value(value)
        timestamp = time.monotonic() if captured_at is None else float(captured_at)
        if not math.isfinite(timestamp):
            raise ValueError("observation timestamp must be finite")
        if confidence is not None and not 0.0 <= confidence <= 1.0:
            raise ValueError("observation confidence must be between zero and one")

        previous_record = self._observations.get(key)
        if previous_record is not None and timestamp < previous_record.captured_at:
            self.log("observation_discarded", key=key, captured_at=timestamp, source=source)
            return previous_record
        old = self._facts.get(key, _MISSING)
        self._next_observation_version += 1
        record = ObservationRecord(
            predicate=key,
            value=observed,
            version=self._next_observation_version,
            captured_at=timestamp,
            source=source,
            evidence=evidence,
            confidence=confidence,
        )
        self._facts[key] = observed
        self._observations[key] = record
        self.log(
            "observation",
            key=key,
            value=observed,
            version=record.version,
            captured_at=timestamp,
            source=source,
            evidence=evidence,
            confidence=confidence,
        )
        if old is _MISSING or old != observed:
            previous = None if old is _MISSING else old
            for listener in list(self._listeners):
                listener(key, previous, observed)
        self._notify_changed()
        return record

    def update(self, facts: dict[str, Any], source: str = "unknown") -> None:
        for key, value in facts.items():
            self.set(key, value, source=source)

    def subscribe(self, listener: Listener) -> None:
        self._listeners.append(listener)

    def _notify_changed(self) -> None:
        # Wake every waiter, then arm a fresh event for the next write.
        self._changed.set()
        self._changed = asyncio.Event()

    def log(self, kind: str, **data: Any) -> None:
        event = {"t": time.time(), "kind": kind, **data}
        self.events.append(event)
        if self._log_path:
            with self._log_path.open("a") as f:
                f.write(json.dumps(event, default=str) + "\n")

    # -- waits -------------------------------------------------------------------------

    async def wait_for(
        self,
        condition: str,
        timeout: float | None = None,
        *,
        after_version: int | None = None,
        after_time: float | None = None,
        max_age_s: float | None = None,
    ) -> bool:
        """Wait until `condition` holds. Returns False on timeout."""
        loop = asyncio.get_running_loop()
        deadline = None if timeout is None else loop.time() + timeout
        while not self.holds(condition, after_version=after_version, after_time=after_time, max_age_s=max_age_s):
            event = self._changed
            remaining = None if deadline is None else deadline - loop.time()
            if remaining is not None and remaining <= 0:
                return False
            try:
                await asyncio.wait_for(event.wait(), remaining)
            except asyncio.TimeoutError:
                return self.holds(condition, after_version=after_version, after_time=after_time, max_age_s=max_age_s)
        return True

    async def wait_until_violated(self, condition: str, *, max_age_s: float | None = None) -> None:
        """Return as soon as `condition` stops holding (used by WHILE monitors)."""
        while self.holds(condition, max_age_s=max_age_s):
            event = self._changed
            record = self.observation(condition)
            remaining = None if max_age_s is None else max(0, record.captured_at + max_age_s - time.monotonic())
            try:
                await asyncio.wait_for(event.wait(), remaining)
            except asyncio.TimeoutError:
                return


def _condition_parts(condition: str) -> tuple[str, bool]:
    return (condition[1:], True) if condition.startswith("!") else (condition, False)


def _truth_value(value: object) -> bool | None:
    if isinstance(value, Verdict):
        return _observed_value(value)
    if value is None:
        return None
    return bool(value)


def _observed_value(value: bool | Verdict | None) -> bool | None:
    if value is Verdict.TRUE:
        return True
    if value is Verdict.FALSE:
        return False
    if value is Verdict.UNKNOWN or value is None:
        return None
    if isinstance(value, bool):
        return value
    raise TypeError("observed predicate values must be true, false, or unknown")
