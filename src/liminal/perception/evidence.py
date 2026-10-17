"""Typed evidence and observation records for physical world state."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias


class Truth(str, Enum):
    """Three-valued truth used by evidence-backed predicates."""

    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"

    def negate(self) -> Truth:
        if self is Truth.TRUE:
            return Truth.FALSE
        if self is Truth.FALSE:
            return Truth.TRUE
        return Truth.UNKNOWN


Bindings: TypeAlias = tuple[tuple[str, str], ...]


def normalize_bindings(bindings: dict[str, str] | Bindings | None) -> Bindings:
    if bindings is None:
        return ()
    items = bindings.items() if isinstance(bindings, dict) else bindings
    normalized = tuple(sorted((str(name), str(value)) for name, value in items))
    if len({name for name, _ in normalized}) != len(normalized):
        raise ValueError("object binding names must be unique")
    return normalized


@dataclass(frozen=True, slots=True)
class EvidenceRef:
    """A reference to the observation artifact supporting a fact."""

    evidence_id: str
    artifact_uri: str = ""
    sha256: str = ""
    camera_id: str = ""
    capture_utc: str = ""
    frame_seq: int | None = None
    manifest_hash: str = ""
    stream_epoch: str = ""
    capture_mono_s: float | None = None

    def __post_init__(self) -> None:
        if not self.evidence_id:
            raise ValueError("evidence_id must be non-empty")
        if self.frame_seq is not None and self.frame_seq < 0:
            raise ValueError("frame_seq must be non-negative")
        if self.capture_mono_s is not None and not math.isfinite(self.capture_mono_s):
            raise ValueError("capture_mono_s must be finite")


@dataclass(frozen=True, slots=True)
class PredicateSpec:
    """Freshness and write-authority policy for a physical predicate."""

    name: str
    ttl_s: float
    physical: bool = True

    def __post_init__(self) -> None:
        if not self.name or self.name.startswith("!"):
            raise ValueError("predicate name must be a non-empty positive key")
        if not math.isfinite(self.ttl_s) or self.ttl_s <= 0:
            raise ValueError("predicate ttl_s must be positive")


@dataclass(frozen=True, slots=True)
class FactObservation:
    """One observer's claim about a predicate and its bound objects."""

    predicate: str
    value: Truth
    evidence: tuple[EvidenceRef, ...]
    observer_id: str
    observer_version: str
    observed_utc: str
    observed_mono_s: float
    run_id: str
    attempt_id: str
    object_bindings: Bindings = ()
    valid_until_mono_s: float | None = None
    reason: str = ""

    def __post_init__(self) -> None:
        if not self.predicate or self.predicate.startswith("!"):
            raise ValueError("predicate must be a non-empty positive key")
        if not self.observer_id or not self.observer_version:
            raise ValueError("observer identity and version are required")
        if not self.observed_utc:
            raise ValueError("observed_utc is required")
        if not isinstance(self.value, Truth):
            raise TypeError("observation value must be a Truth enum member")
        if not self.run_id or not self.attempt_id:
            raise ValueError("run_id and attempt_id are required")
        if not self.evidence:
            raise ValueError("physical observations require at least one evidence reference")
        if not math.isfinite(self.observed_mono_s):
            raise ValueError("observed_mono_s must be finite")
        if self.valid_until_mono_s is not None and (
            not math.isfinite(self.valid_until_mono_s) or self.valid_until_mono_s <= self.observed_mono_s
        ):
            raise ValueError("valid_until_mono_s must be after observed_mono_s")
        object.__setattr__(self, "object_bindings", normalize_bindings(self.object_bindings))


@dataclass(frozen=True, slots=True)
class FactRecord:
    """Accepted observation plus the local sequence used as a watermark."""

    sequence: int
    observation: FactObservation
    expires_at_mono_s: float
    accepted_at_mono_s: float = 0.0

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(ref.evidence_id for ref in self.observation.evidence)


@dataclass(frozen=True, slots=True)
class WriterCapability:
    """Unforgeable-in-process authority issued by a WorldState instance."""

    observer_id: str
    observer_version: str
    _token: object
