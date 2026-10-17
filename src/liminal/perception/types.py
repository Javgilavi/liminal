"""Model-independent contracts for VLM scene inspection.

The rest of the system consumes these types rather than provider-specific model
responses.  Parsing fails closed: missing, malformed, ambiguous, or contradictory
predicates become ``UNKNOWN``.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any


class Verdict(str, Enum):
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"

    @classmethod
    def parse(cls, value: object) -> Verdict:
        if isinstance(value, bool):
            return cls.TRUE if value else cls.FALSE
        if isinstance(value, str):
            try:
                return cls(value.strip().lower())
            except ValueError:
                pass
        return cls.UNKNOWN


@dataclass(frozen=True)
class InspectionRequest:
    """A bounded request for the scene inspector.

    ``exclusive_groups`` describes predicates that cannot simultaneously be true,
    for example ``("access_clear", "access_blocked")``.
    """

    predicates: tuple[str, ...]
    cameras: tuple[str, ...] = ("top",)
    context: str = ""
    definitions: dict[str, str] = field(default_factory=dict)
    timeout_s: float = 5.0
    exclusive_groups: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        if not self.predicates or any(not isinstance(p, str) or not p.strip() for p in self.predicates):
            raise ValueError("an inspection request needs at least one named predicate")
        if len(set(self.predicates)) != len(self.predicates):
            raise ValueError("inspection predicates must be unique")
        if not self.cameras or any(not isinstance(c, str) or not c.strip() for c in self.cameras):
            raise ValueError("an inspection request needs at least one named camera")
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("inspection timeout must be positive")
        requested = set(self.predicates)
        if not set(self.definitions) <= requested:
            raise ValueError("predicate definitions must refer to requested predicates")
        if any(not isinstance(value, str) or not value.strip() for value in self.definitions.values()):
            raise ValueError("predicate definitions must be non-empty strings")
        for group in self.exclusive_groups:
            if len(group) < 2 or not set(group) <= requested:
                raise ValueError("exclusive groups need at least two requested predicates")


@dataclass(frozen=True)
class InspectionProfile:
    """Shared settings used when a running skill asks for scene verification.

    Task-file ``inspect`` nodes may supply a one-off request. FLUX monitoring uses
    this profile so success predicates and active invariants keep one stable meaning
    across every action chunk.
    """

    cameras: tuple[str, ...] = ("top",)
    definitions: dict[str, str] = field(default_factory=dict)
    context: str = ""
    timeout_s: float = 5.0
    retry_delay_s: float = 0.05
    exclusive_groups: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.context, str):
            raise ValueError("inspection context must be a string")
        if not self.cameras or any(
            not isinstance(camera, str) or not camera.strip() for camera in self.cameras
        ):
            raise ValueError("an inspection profile needs at least one named camera")
        if (
            not math.isfinite(self.timeout_s)
            or not math.isfinite(self.retry_delay_s)
            or self.timeout_s <= 0
            or self.retry_delay_s < 0
        ):
            raise ValueError("inspection timeout must be positive and retry delay non-negative")
        if any(
            not isinstance(key, str)
            or not isinstance(value, str)
            or not key.strip()
            or not value.strip()
            for key, value in self.definitions.items()
        ):
            raise ValueError("predicate definitions must use non-empty names and descriptions")
        for group in self.exclusive_groups:
            if len(group) < 2 or any(
                not isinstance(predicate, str) or not predicate.strip() for predicate in group
            ):
                raise ValueError("exclusive groups need at least two named predicates")

    def request(self, conditions: tuple[str, ...], *, context: str = "") -> InspectionRequest:
        predicates = tuple(dict.fromkeys(predicate_name(condition) for condition in conditions))
        requested = set(predicates)
        missing = requested - set(self.definitions)
        if missing:
            raise ValueError(f"missing predicate definitions for {sorted(missing)}")
        return InspectionRequest(
            predicates=predicates,
            cameras=self.cameras,
            context=" ".join(part for part in (self.context.strip(), context.strip()) if part),
            definitions={key: value for key, value in self.definitions.items() if key in requested},
            timeout_s=self.timeout_s,
            exclusive_groups=tuple(group for group in self.exclusive_groups if set(group) <= requested),
        )


@dataclass(frozen=True)
class PredicateObservation:
    predicate: str
    value: Verdict
    evidence: str = ""
    confidence: float | None = None

    @property
    def known(self) -> bool:
        return self.value is not Verdict.UNKNOWN


@dataclass(frozen=True)
class SceneAssessment:
    observations: dict[str, PredicateObservation]
    captured_at: float = field(default_factory=time.monotonic)
    raw_response: str | None = field(default=None, repr=False)
    parse_error: str | None = None

    def observation(self, predicate: str) -> PredicateObservation:
        return self.observations.get(
            predicate,
            PredicateObservation(predicate, Verdict.UNKNOWN, "predicate was not assessed"),
        )

    @classmethod
    def unknown(cls, request: InspectionRequest, reason: str, raw_response: str | None = None) -> SceneAssessment:
        return cls(
            observations={
                predicate: PredicateObservation(predicate, Verdict.UNKNOWN, reason)
                for predicate in request.predicates
            },
            raw_response=raw_response,
            parse_error=reason,
        )

    @classmethod
    def from_model_output(
        cls,
        request: InspectionRequest,
        response: str | Mapping[str, Any],
    ) -> SceneAssessment:
        """Parse the strict scene-assessment schema, falling back to unknown."""

        raw_response = response if isinstance(response, str) else None
        try:
            payload = json.loads(response) if isinstance(response, str) else dict(response)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            return cls.unknown(request, f"invalid JSON response: {exc}", raw_response)
        if not isinstance(payload, dict) or not isinstance(payload.get("predicates"), dict):
            return cls.unknown(request, "response must contain a predicates object", raw_response)

        entries = payload["predicates"]
        ambiguous = _string_set(payload.get("ambiguous", ()))
        observations: dict[str, PredicateObservation] = {}
        for predicate in request.predicates:
            entry = entries.get(predicate)
            if not isinstance(entry, Mapping):
                observations[predicate] = PredicateObservation(
                    predicate, Verdict.UNKNOWN, "predicate is missing or malformed"
                )
                continue

            verdict = Verdict.parse(entry.get("value"))
            evidence = entry.get("evidence", "")
            if not isinstance(evidence, str):
                evidence = ""
                verdict = Verdict.UNKNOWN
            confidence = _confidence(entry.get("confidence"))
            if predicate in ambiguous:
                verdict = Verdict.UNKNOWN
                evidence = evidence or "model marked predicate as ambiguous"
            observations[predicate] = PredicateObservation(predicate, verdict, evidence, confidence)

        for group in request.exclusive_groups:
            if sum(observations[p].value is Verdict.TRUE for p in group) > 1:
                reason = f"contradictory exclusive predicates: {', '.join(group)}"
                for predicate in group:
                    observations[predicate] = replace(
                        observations[predicate], value=Verdict.UNKNOWN, evidence=reason
                    )

        return cls(observations=observations, raw_response=raw_response)


def _string_set(value: object) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {item for item in value if isinstance(item, str)}


def _confidence(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if 0.0 <= number <= 1.0 else None


def predicate_name(condition: str) -> str:
    """Return the observable predicate behind ``x`` or a negated ``!x`` condition."""

    if not isinstance(condition, str):
        raise TypeError("conditions must be strings")
    predicate = condition[1:] if condition.startswith("!") else condition
    if not predicate or predicate.startswith("!") or not predicate.strip():
        raise ValueError(f"invalid predicate condition {condition!r}")
    return predicate
