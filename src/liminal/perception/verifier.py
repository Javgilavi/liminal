"""Attempt-scoped verification over observer-authored world evidence."""

from __future__ import annotations

import math
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from liminal.perception.evidence import Bindings, FactRecord, Truth, normalize_bindings
from liminal.perception.world_state import WorldState


@dataclass(frozen=True, slots=True)
class VerificationRequest:
    run_id: str
    attempt_id: str
    expected_effect: str
    object_bindings: Bindings | dict[str, str]
    evidence_after_seq: int
    allowed_observers: tuple[str, ...]
    deadline_mono_s: float
    request_id: str = ""
    evidence_after_mono_s: float | None = None

    def __post_init__(self) -> None:
        if not self.run_id or not self.attempt_id:
            raise ValueError("run_id and attempt_id are required")
        if not self.expected_effect or self.expected_effect.startswith("!"):
            raise ValueError("expected_effect must be a positive predicate name")
        if self.evidence_after_seq < 0:
            raise ValueError("evidence_after_seq must be non-negative")
        if not math.isfinite(self.deadline_mono_s):
            raise ValueError("deadline_mono_s must be finite")
        if self.evidence_after_mono_s is not None and not math.isfinite(self.evidence_after_mono_s):
            raise ValueError("evidence_after_mono_s must be finite")
        if not self.allowed_observers:
            raise ValueError("at least one allowed observer is required")
        object.__setattr__(self, "object_bindings", normalize_bindings(self.object_bindings))
        if not self.request_id:
            object.__setattr__(self, "request_id", uuid.uuid4().hex)


@dataclass(frozen=True, slots=True)
class VerificationResult:
    request_id: str
    decision: Truth
    evidence_ids: tuple[str, ...]
    observer_id: str | None
    observer_version: str | None
    reason: str
    missing_evidence: tuple[str, ...] = ()
    evidence_sequence: int | None = None


class Verifier:
    """Wait for and evaluate evidence newer than the attempt's committed boundary."""

    def __init__(self, *, monotonic: Callable[[], float] | None = None):
        self._monotonic = monotonic

    async def verify(self, request: VerificationRequest, world: WorldState) -> VerificationResult:
        """Return true/false only from fresh matching observation; otherwise unknown."""
        monotonic = self._monotonic or world.monotonic_now
        while True:
            if monotonic() >= request.deadline_mono_s:
                return self._unknown(request, "verification deadline expired")
            fresh = self._matching_records(request, world)
            if fresh:
                latest_time = max(record.observation.observed_mono_s for record in fresh)
                latest = [record for record in fresh if record.observation.observed_mono_s == latest_time]
                values = {record.observation.value for record in latest}
                if len(values) != 1:
                    return self._unknown(request, "conflicting observations at the same capture time")
                chosen = max(latest, key=lambda record: record.sequence)
                value = next(iter(values))
                if value is Truth.UNKNOWN:
                    return self._from_record(request, chosen, Truth.UNKNOWN, "observer abstained")
                return self._from_record(request, chosen, value, chosen.observation.reason or "fresh evidence")

            remaining = request.deadline_mono_s - monotonic()
            if remaining <= 0:
                return self._unknown(request, "no matching fresh evidence before deadline")
            changed = await world.wait_for_change(remaining)
            if not changed and monotonic() >= request.deadline_mono_s:
                return self._unknown(request, "no matching fresh evidence before deadline")

    @staticmethod
    def _matching_records(request: VerificationRequest, world: WorldState) -> list[FactRecord]:
        allowed = set(request.allowed_observers)
        return [
            record
            for record in world.evidence_records(request.expected_effect, request.object_bindings)
            if record.sequence > request.evidence_after_seq
            and record.expires_at_mono_s > world.monotonic_now()
            and record.accepted_at_mono_s <= request.deadline_mono_s
            and record.observation.observed_mono_s <= request.deadline_mono_s
            and record.observation.run_id == request.run_id
            and record.observation.attempt_id == request.attempt_id
            and record.observation.observer_id in allowed
            and (
                request.evidence_after_mono_s is None
                or record.observation.observed_mono_s > request.evidence_after_mono_s
            )
        ]

    @staticmethod
    def _from_record(
        request: VerificationRequest, record: FactRecord, decision: Truth, reason: str
    ) -> VerificationResult:
        observation = record.observation
        return VerificationResult(
            request_id=request.request_id,
            decision=decision,
            evidence_ids=record.evidence_ids,
            observer_id=observation.observer_id,
            observer_version=observation.observer_version,
            reason=reason,
            missing_evidence=() if record.evidence_ids else (request.expected_effect,),
            evidence_sequence=record.sequence,
        )

    @staticmethod
    def _unknown(request: VerificationRequest, reason: str) -> VerificationResult:
        return VerificationResult(
            request_id=request.request_id,
            decision=Truth.UNKNOWN,
            evidence_ids=(),
            observer_id=None,
            observer_version=None,
            reason=reason,
            missing_evidence=(request.expected_effect,),
        )


async def verify(request: VerificationRequest, world: WorldState) -> VerificationResult:
    """Convenience API for callers that do not need a configured verifier."""
    return await Verifier().verify(request, world)
