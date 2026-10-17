"""Shared symbolic state with trusted fixtures and evidence-backed facts."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from liminal.perception.evidence import (
    FactObservation,
    FactRecord,
    PredicateSpec,
    Truth,
    WriterCapability,
    normalize_bindings,
)

Listener = Callable[[str, Any, Any], None]
FactKey = tuple[str, tuple[tuple[str, str], ...]]


class WorldState:
    """Read predicates and await changes; physical observations need authority."""

    def __init__(
        self,
        initial: dict[str, Any] | None = None,
        log_path: str | Path | None = None,
        *,
        strict: bool = True,
        monotonic: Callable[[], float] = time.monotonic,
        wall_time: Callable[[], float] = time.time,
    ):
        # Initial values remain an explicit trusted-fixture path for mock runs.
        self._facts: dict[str, Any] = dict(initial or {})
        self._changed = asyncio.Event()
        self._listeners: list[Listener] = []
        self._log_path = Path(log_path) if log_path else None
        if self._log_path:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self.events: list[dict[str, Any]] = []
        self.strict = strict
        self._monotonic = monotonic
        self._wall_time = wall_time
        self._predicates: dict[str, PredicateSpec] = {}
        self._trusted_fixture_keys: set[str] = set()
        self._capabilities: dict[int, tuple[WriterCapability, frozenset[str]]] = {}
        self._records: dict[FactKey, list[FactRecord]] = {}
        self._sequence = 0
        self._current_run_id: str | None = None
        self._active_attempt_id: str | None = None
        self._completed_attempt_ids: set[tuple[str, str]] = set()
        self._attempt_started_mono_s: float | None = None
        self._frame_sequences: dict[tuple[str, str, str], tuple[int, str]] = {}

    @property
    def sequence(self) -> int:
        """Latest accepted observation sequence, suitable as a freshness watermark."""
        return self._sequence

    @property
    def current_sequence(self) -> int:
        return self.sequence

    @property
    def current_run_id(self) -> str | None:
        return self._current_run_id

    @property
    def active_attempt_id(self) -> str | None:
        """Observers stamp this; with none active, use a fresh observation-only id."""
        return self._active_attempt_id

    def monotonic_now(self) -> float:
        return self._monotonic()

    def register_predicate(self, spec: PredicateSpec) -> None:
        existing = self._predicates.get(spec.name)
        if existing is not None and existing != spec:
            raise ValueError(f"predicate {spec.name!r} already has a different specification")
        self._predicates[spec.name] = spec
        if spec.physical and self.strict:
            # A fixture value written before the registry existed cannot stand in
            # for observer evidence once the predicate is declared physical.
            self._facts.pop(spec.name, None)
            self._trusted_fixture_keys.discard(spec.name)

    def begin_run(self, run_id: str) -> None:
        if not run_id:
            raise ValueError("run_id must be non-empty")
        if self._current_run_id is not None and self._active_attempt_id is not None:
            self._completed_attempt_ids.add((self._current_run_id, self._active_attempt_id))
        self._current_run_id = run_id
        self._active_attempt_id = None
        self._attempt_started_mono_s = None

    def begin_attempt(self, attempt_id: str) -> None:
        if not self._current_run_id:
            raise RuntimeError("begin_run must be called before begin_attempt")
        if not attempt_id:
            raise ValueError("attempt_id must be non-empty")
        if self._active_attempt_id is not None:
            raise RuntimeError("end the current attempt before beginning another")
        if (self._current_run_id, attempt_id) in self._completed_attempt_ids:
            raise ValueError("attempt IDs cannot be reused after completion")
        self._active_attempt_id = attempt_id
        self._attempt_started_mono_s = self._monotonic()

    def end_attempt(self) -> None:
        if self._active_attempt_id is not None:
            assert self._current_run_id is not None
            self._completed_attempt_ids.add((self._current_run_id, self._active_attempt_id))
        self._active_attempt_id = None
        self._attempt_started_mono_s = None

    def register_observer(
        self, observer_id: str, version: str, predicates: set[str] | frozenset[str]
    ) -> WriterCapability:
        if not observer_id or not version:
            raise ValueError("observer_id and version are required")
        allowed = frozenset(predicates)
        missing = allowed.difference(self._predicates)
        if missing:
            raise ValueError(f"observer predicates must be registered first: {sorted(missing)}")
        capability = WriterCapability(observer_id, version, object())
        self._capabilities[id(capability)] = (capability, allowed)
        return capability

    def observe(self, observation: FactObservation, writer_capability: WriterCapability) -> FactRecord:
        registered = self._capabilities.get(id(writer_capability))
        if registered is None or registered[0] is not writer_capability:
            raise PermissionError("unregistered observer capability")
        _, allowed = registered
        if observation.predicate not in allowed:
            raise PermissionError(f"observer cannot write predicate {observation.predicate!r}")
        if (observation.observer_id, observation.observer_version) != (
            writer_capability.observer_id,
            writer_capability.observer_version,
        ):
            raise PermissionError("observation identity does not match writer capability")
        if self._current_run_id is not None and observation.run_id != self._current_run_id:
            self.log("stale_observation", run_id=observation.run_id, attempt_id=observation.attempt_id)
            raise PermissionError("observation belongs to a different run")
        if self._active_attempt_id is not None and observation.attempt_id != self._active_attempt_id:
            self.log("stale_observation", run_id=observation.run_id, attempt_id=observation.attempt_id)
            raise PermissionError("observation belongs to a different or completed attempt")
        if (observation.run_id, observation.attempt_id) in self._completed_attempt_ids:
            self.log("stale_observation", run_id=observation.run_id, attempt_id=observation.attempt_id)
            raise PermissionError("observation belongs to a completed attempt")
        spec = self._predicates[observation.predicate]
        expires = observation.valid_until_mono_s
        if expires is None:
            expires = observation.observed_mono_s + spec.ttl_s
        elif expires > observation.observed_mono_s + spec.ttl_s:
            raise ValueError("observation validity cannot exceed the registered predicate TTL")
        if observation.observed_mono_s > self._monotonic():
            raise ValueError("observation timestamp is in the future")
        if self._attempt_started_mono_s is not None and observation.observed_mono_s < self._attempt_started_mono_s:
            raise ValueError("observation was captured before the active attempt began")
        frame_updates: dict[tuple[str, str, str], tuple[int, str]] = {}
        for evidence in observation.evidence:
            if evidence.capture_mono_s is not None:
                if evidence.capture_mono_s > self._monotonic():
                    raise ValueError("evidence capture timestamp is in the future")
                if evidence.capture_mono_s > observation.observed_mono_s:
                    raise ValueError("evidence capture cannot occur after the observation timestamp")
            if evidence.frame_seq is None or not evidence.camera_id:
                continue
            stream = (observation.observer_id, evidence.camera_id, evidence.stream_epoch)
            previous = frame_updates.get(stream, self._frame_sequences.get(stream))
            if previous is not None and evidence.frame_seq < previous[0]:
                raise ValueError("frame sequence regressed within a camera stream")
            if previous is not None and evidence.frame_seq == previous[0] and evidence.evidence_id != previous[1]:
                raise ValueError("a repeated frame sequence must retain its evidence ID")
            frame_updates[stream] = (evidence.frame_seq, evidence.evidence_id)
        if expires <= self._monotonic():
            raise ValueError("observation is already expired")

        self._frame_sequences.update(frame_updates)
        self._sequence += 1
        record = FactRecord(self._sequence, observation, expires, self._monotonic())
        key = (observation.predicate, observation.object_bindings)
        records = self._records.setdefault(key, [])
        visible_records = [
            item
            for item in records
            if self._current_run_id is None or item.observation.run_id == self._current_run_id
        ]
        old = self._current_truth(visible_records, self._monotonic())
        records.append(record)
        if self._current_run_id is None or observation.run_id == self._current_run_id:
            visible_records.append(record)
        # Keep current truth available through the legacy snapshot/get interface.
        self._facts[observation.predicate] = observation.value is Truth.TRUE
        self._trusted_fixture_keys.discard(observation.predicate)
        self.log(
            "observation",
            sequence=record.sequence,
            predicate=observation.predicate,
            value=observation.value.value,
            evidence_ids=record.evidence_ids,
            observer_id=observation.observer_id,
            observer_version=observation.observer_version,
            observed_utc=observation.observed_utc,
            observed_mono_s=observation.observed_mono_s,
            expires_at_mono_s=record.expires_at_mono_s,
            run_id=observation.run_id,
            attempt_id=observation.attempt_id,
            object_bindings=observation.object_bindings,
            reason=observation.reason,
        )
        new = self._current_truth(visible_records, self._monotonic())
        for listener in list(self._listeners):
            listener(observation.predicate, old, new)
        self._wake_waiters()
        return record

    def evidence_records(
        self, predicate: str, bindings: dict[str, str] | tuple[tuple[str, str], ...] | None = None
    ) -> tuple[FactRecord, ...]:
        return tuple(self._records.get((predicate, normalize_bindings(bindings)), ()))

    def evaluate(
        self,
        key: str,
        *,
        bindings: dict[str, str] | tuple[tuple[str, str], ...] | None = None,
        now_mono_s: float | None = None,
    ) -> Truth:
        """Evaluate a positive or negated fact; absence and expiry stay unknown."""
        negate = key.startswith("!")
        predicate = key[1:] if negate else key
        now = self._monotonic() if now_mono_s is None else now_mono_s
        records = self._records.get((predicate, normalize_bindings(bindings)), ())
        if self._current_run_id is not None:
            records = tuple(r for r in records if r.observation.run_id == self._current_run_id)
        if records:
            result = self._current_truth(records, now)
        elif (
            predicate in self._predicates
            and self._predicates[predicate].physical
            and self.strict
            and not (not normalize_bindings(bindings) and predicate in self._trusted_fixture_keys)
        ):
            result = Truth.UNKNOWN
        elif predicate in self._facts:
            result = Truth.TRUE if bool(self._facts[predicate]) else Truth.FALSE
        else:
            result = Truth.UNKNOWN
        return result.negate() if negate else result

    def _current_truth(self, records: list[FactRecord] | tuple[FactRecord, ...], now: float) -> Truth:
        if not records:
            return Truth.UNKNOWN
        latest_time = max(record.observation.observed_mono_s for record in records)
        latest = [record for record in records if record.observation.observed_mono_s == latest_time]
        if all(record.expires_at_mono_s <= now for record in latest):
            return Truth.UNKNOWN
        values = {record.observation.value for record in latest}
        if len(values) != 1:
            return Truth.UNKNOWN
        return next(iter(values))

    def get(self, key: str, default: Any = None) -> Any:
        if key in self._predicates or any(k[0] == key for k in self._records):
            value = self.evaluate(key)
            return value is Truth.TRUE if value is not Truth.UNKNOWN else default
        return self._facts.get(key, default)

    def holds(
        self, condition: str, *, bindings: dict[str, str] | tuple[tuple[str, str], ...] | None = None
    ) -> bool:
        """Compatibility check: unknown, including negated unknown, never holds."""
        return self.evaluate(condition, bindings=bindings) is Truth.TRUE

    def snapshot(self) -> dict[str, Any]:
        result = dict(self._facts)
        for predicate in self._predicates:
            value = self.evaluate(predicate)
            if value is Truth.UNKNOWN:
                result.pop(predicate, None)
            else:
                result[predicate] = value is Truth.TRUE
        return result

    def set(self, key: str, value: Any, source: str = "unknown") -> None:
        if self.strict and key in self._predicates and self._predicates[key].physical:
            raise PermissionError(f"physical predicate {key!r} requires observer evidence")
        self._set_fixture_value(key, value, source=source)

    def set_fixture(self, key: str, value: Any, source: str = "trusted_fixture") -> None:
        """Explicit mock/test-only path for trusted symbolic fixture state."""
        self._set_fixture_value(key, value, source=source)

    def _set_fixture_value(self, key: str, value: Any, *, source: str) -> None:
        old = self._facts.get(key)
        if old == value:
            self._trusted_fixture_keys.add(key)
            return
        self._facts[key] = value
        self._trusted_fixture_keys.add(key)
        self.log("fact", key=key, value=value, old=old, source=source, trust="fixture_or_symbolic")
        for listener in list(self._listeners):
            listener(key, old, value)
        self._wake_waiters()

    def update(self, facts: dict[str, Any], source: str = "unknown") -> None:
        for key, value in facts.items():
            self.set(key, value, source=source)

    def subscribe(self, listener: Listener) -> None:
        self._listeners.append(listener)

    def log(self, kind: str, **data: Any) -> None:
        event = {"t": self._wall_time(), "kind": kind, **data}
        self.events.append(event)
        if self._log_path:
            with self._log_path.open("a") as f:
                f.write(json.dumps(event, default=str) + "\n")

    def _wake_waiters(self) -> None:
        self._changed.set()
        self._changed = asyncio.Event()

    def notify_clock_advanced(self) -> None:
        """Wake expiry waiters after an injected test clock advances."""
        self._wake_waiters()

    def _next_expiry(
        self,
        condition: str,
        bindings: dict[str, str] | tuple[tuple[str, str], ...] | None = None,
    ) -> float | None:
        predicate = condition[1:] if condition.startswith("!") else condition
        binding_key = normalize_bindings(bindings)
        expiries = [
            record.expires_at_mono_s
            for (name, binding), records in self._records.items()
            if name == predicate and binding == binding_key
            for record in records
            if record.expires_at_mono_s > self._monotonic()
            and (self._current_run_id is None or record.observation.run_id == self._current_run_id)
        ]
        return min(expiries) if expiries else None

    async def wait_for(
        self,
        condition: str,
        timeout: float | None = None,
        *,
        bindings: dict[str, str] | tuple[tuple[str, str], ...] | None = None,
    ) -> bool:
        """Wait until a condition is currently true or its timeout/expiry arrives."""
        loop = asyncio.get_running_loop()
        deadline = None if timeout is None else loop.time() + timeout
        while not self.holds(condition, bindings=bindings):
            event = self._changed
            remaining = None if deadline is None else deadline - loop.time()
            if remaining is not None and remaining <= 0:
                return self.holds(condition, bindings=bindings)
            expiry = self._next_expiry(condition, bindings)
            until_expiry = None if expiry is None else max(0.0, expiry - self._monotonic())
            wait_for = min(x for x in (remaining, until_expiry) if x is not None) if (
                remaining is not None or until_expiry is not None
            ) else None
            try:
                await asyncio.wait_for(event.wait(), wait_for)
            except asyncio.TimeoutError:
                if deadline is not None and loop.time() >= deadline:
                    return self.holds(condition, bindings=bindings)
        return True

    async def wait_until_violated(
        self,
        condition: str,
        *,
        bindings: dict[str, str] | tuple[tuple[str, str], ...] | None = None,
    ) -> None:
        """Return when a condition becomes false or unknown, including at expiry."""
        while self.holds(condition, bindings=bindings):
            event = self._changed
            expiry = self._next_expiry(condition, bindings)
            timeout = None if expiry is None else max(0.0, expiry - self._monotonic())
            try:
                await asyncio.wait_for(event.wait(), timeout)
            except asyncio.TimeoutError:
                pass

    async def wait_for_change(self, timeout: float | None = None) -> bool:
        """Wait for any accepted fact change; returns false if timed out."""
        event = self._changed
        try:
            await asyncio.wait_for(event.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False
