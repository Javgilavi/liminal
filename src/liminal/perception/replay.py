"""Offline evaluation of a VisionModel against recorded scene frames."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from liminal.perception.inspector import SceneInspector
from liminal.perception.types import InspectionRequest, Verdict
from liminal.perception.vlm import VisionModel
from liminal.perception.world_state import WorldState

ImageLoader = Callable[[Path], Any]


class RecordingError(ValueError):
    """A replay manifest is incomplete or internally inconsistent."""


@dataclass(frozen=True)
class RecordedFrame:
    images: dict[str, Path]
    expected: dict[str, Verdict]
    name: str = ""


@dataclass(frozen=True)
class InspectionRecording:
    name: str
    request: InspectionRequest
    frames: tuple[RecordedFrame, ...]


@dataclass(frozen=True)
class ReplayMismatch:
    frame: str
    predicate: str
    expected: Verdict
    actual: Verdict
    evidence: str = ""
    repeat: int = 1


@dataclass(frozen=True)
class ReplayObservation:
    expected: Verdict
    actual: Verdict
    evidence: str = ""

    @property
    def correct(self) -> bool:
        return self.actual is self.expected

    @property
    def false_positive_transition(self) -> bool:
        return self.actual is Verdict.TRUE and self.expected is not Verdict.TRUE


@dataclass(frozen=True)
class ReplayCall:
    frame: str
    repeat: int
    images: dict[str, str]
    latency_s: float
    observations: dict[str, ReplayObservation]
    raw_response: str | None = None
    parse_error: str | None = None


@dataclass
class PredicateReplayMetrics:
    observations: int = 0
    correct: int = 0
    incorrect: int = 0
    unknown: int = 0
    false_positive_transitions: int = 0
    repeat_groups: int = 0
    consistent_groups: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.observations if self.observations else 0.0

    @property
    def agreement_rate(self) -> float:
        return self.consistent_groups / self.repeat_groups if self.repeat_groups else 0.0


@dataclass
class ReplayReport:
    recording: str
    frames: int
    observations: int
    repeats: int = 1
    correct: int = 0
    incorrect: int = 0
    unknown: int = 0
    false_positive_transitions: int = 0
    repeat_groups: int = 0
    consistent_groups: int = 0
    mismatches: list[ReplayMismatch] = field(default_factory=list)
    calls: list[ReplayCall] = field(default_factory=list)
    per_predicate: dict[str, PredicateReplayMetrics] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return self.correct / self.observations if self.observations else 0.0

    @property
    def passed(self) -> bool:
        return self.incorrect == 0 and self.false_positive_transitions == 0

    @property
    def agreement_rate(self) -> float:
        return self.consistent_groups / self.repeat_groups if self.repeat_groups else 0.0

    @property
    def latency_summary(self) -> dict[str, float | int | None]:
        latencies = [call.latency_s for call in self.calls]
        if not latencies:
            return {"calls": 0, "min_s": None, "median_s": None, "mean_s": None, "max_s": None}
        return {
            "calls": len(latencies),
            "min_s": min(latencies),
            "median_s": statistics.median(latencies),
            "mean_s": statistics.fmean(latencies),
            "max_s": max(latencies),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a stable, JSON-serializable report with inspectable failures."""

        return {
            "recording": self.recording,
            "summary": {
                "passed": self.passed,
                "frames": self.frames,
                "repeats": self.repeats,
                "observations": self.observations,
                "correct": self.correct,
                "incorrect": self.incorrect,
                "unknown": self.unknown,
                "false_positive_transitions": self.false_positive_transitions,
                "accuracy": self.accuracy,
                "agreement_rate": self.agreement_rate,
            },
            "latency": self.latency_summary,
            "per_predicate": {
                predicate: {
                    "observations": metrics.observations,
                    "correct": metrics.correct,
                    "incorrect": metrics.incorrect,
                    "unknown": metrics.unknown,
                    "false_positive_transitions": metrics.false_positive_transitions,
                    "accuracy": metrics.accuracy,
                    "agreement_rate": metrics.agreement_rate,
                }
                for predicate, metrics in self.per_predicate.items()
            },
            "calls": [
                {
                    "frame": call.frame,
                    "repeat": call.repeat,
                    "images": call.images,
                    "latency_s": call.latency_s,
                    "parse_error": call.parse_error,
                    "raw_response": call.raw_response,
                    "predicates": {
                        predicate: {
                            "expected": observation.expected.value,
                            "actual": observation.actual.value,
                            "correct": observation.correct,
                            "false_positive_transition": observation.false_positive_transition,
                            "evidence": observation.evidence,
                        }
                        for predicate, observation in call.observations.items()
                    },
                }
                for call in self.calls
            ],
        }


def load_recording(path: str | Path) -> InspectionRecording:
    """Load and validate a YAML replay manifest; image paths are manifest-relative."""

    manifest_path = Path(path)
    try:
        data = yaml.safe_load(manifest_path.read_text())
    except (OSError, yaml.YAMLError) as exc:
        raise RecordingError(f"cannot load recording {manifest_path}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise RecordingError("recording must be a mapping")

    request_data = data.get("request")
    if not isinstance(request_data, Mapping):
        raise RecordingError("recording.request must be a mapping")
    predicates = _strings(request_data.get("predicates"), "recording.request.predicates")
    cameras = _strings(request_data.get("cameras", ["top"]), "recording.request.cameras")
    definitions = request_data.get("definitions", {})
    if not isinstance(definitions, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in definitions.items()
    ):
        raise RecordingError("recording.request.definitions must be a string-to-string mapping")
    groups_data = request_data.get("exclusive_groups", [])
    if not isinstance(groups_data, list):
        raise RecordingError("recording.request.exclusive_groups must be a list")
    groups = tuple(
        _strings(group, f"recording.request.exclusive_groups[{index}]")
        for index, group in enumerate(groups_data)
    )
    context = request_data.get("context", "")
    if not isinstance(context, str):
        raise RecordingError("recording.request.context must be a string")
    try:
        request = InspectionRequest(
            predicates=predicates,
            cameras=cameras,
            context=context,
            definitions=dict(definitions),
            timeout_s=float(request_data.get("timeout_s", 5.0)),
            exclusive_groups=groups,
        )
    except (TypeError, ValueError) as exc:
        raise RecordingError(f"invalid recording request: {exc}") from exc

    frames_data = data.get("frames")
    if not isinstance(frames_data, list) or not frames_data:
        raise RecordingError("recording.frames must be a non-empty list")
    frames = tuple(
        _recorded_frame(item, index, manifest_path.parent, request)
        for index, item in enumerate(frames_data)
    )
    name = data.get("name", manifest_path.stem)
    if not isinstance(name, str) or not name.strip():
        raise RecordingError("recording.name must be a non-empty string")
    return InspectionRecording(name=name, request=request, frames=frames)


async def evaluate_recording(
    recording: InspectionRecording,
    model: VisionModel,
    *,
    image_loader: ImageLoader | None = None,
    repeats: int = 1,
) -> ReplayReport:
    """Run the production prompt/parser repeatedly on saved frames and score it."""

    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("recording evaluation repeats must be a positive integer")

    cursor = _FrameCursor(recording.frames, image_loader or load_rgb_image)
    cameras = {name: _RecordedCamera(name, cursor) for name in recording.request.cameras}
    inspector = SceneInspector(model, cameras, WorldState(), confirmations=1, source="recording_replay")
    report = ReplayReport(
        recording=recording.name,
        frames=len(recording.frames),
        repeats=repeats,
        observations=len(recording.frames) * len(recording.request.predicates) * repeats,
        per_predicate={predicate: PredicateReplayMetrics() for predicate in recording.request.predicates},
    )
    for index, frame in enumerate(recording.frames):
        cursor.index = index
        frame_name = frame.name or str(index)
        repeated_values: dict[str, list[Verdict]] = {
            predicate: [] for predicate in recording.request.predicates
        }
        for repeat_index in range(1, repeats + 1):
            inspector.reset()
            started_at = time.perf_counter()
            assessment = await inspector.inspect(recording.request)
            latency_s = time.perf_counter() - started_at
            call_observations: dict[str, ReplayObservation] = {}
            for predicate, expected in frame.expected.items():
                actual = assessment.observation(predicate)
                observation = ReplayObservation(expected, actual.value, actual.evidence)
                call_observations[predicate] = observation
                repeated_values[predicate].append(actual.value)
                metrics = report.per_predicate[predicate]
                metrics.observations += 1
                if actual.value is Verdict.UNKNOWN:
                    report.unknown += 1
                    metrics.unknown += 1
                if observation.correct:
                    report.correct += 1
                    metrics.correct += 1
                else:
                    report.incorrect += 1
                    metrics.incorrect += 1
                    report.mismatches.append(
                        ReplayMismatch(
                            frame_name,
                            predicate,
                            expected,
                            actual.value,
                            actual.evidence,
                            repeat_index,
                        )
                    )
                if observation.false_positive_transition:
                    report.false_positive_transitions += 1
                    metrics.false_positive_transitions += 1
            report.calls.append(
                ReplayCall(
                    frame=frame_name,
                    repeat=repeat_index,
                    images={camera: str(path) for camera, path in frame.images.items()},
                    latency_s=latency_s,
                    observations=call_observations,
                    raw_response=assessment.raw_response,
                    parse_error=assessment.parse_error,
                )
            )
        for predicate, values in repeated_values.items():
            metrics = report.per_predicate[predicate]
            report.repeat_groups += 1
            metrics.repeat_groups += 1
            if len(set(values)) == 1:
                report.consistent_groups += 1
                metrics.consistent_groups += 1
    return report


def load_rgb_image(path: Path) -> Any:
    """Load one RGB uint8 image using the optional perception dependency."""

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("recording replay needs the 'perception' extra (opencv-python-headless)") from exc
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"could not decode recorded frame {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class _FrameCursor:
    def __init__(self, frames: tuple[RecordedFrame, ...], loader: ImageLoader):
        self.frames = frames
        self.loader = loader
        self.index = 0


class _RecordedCamera:
    def __init__(self, name: str, cursor: _FrameCursor):
        self.name = name
        self.cursor = cursor

    def read(self) -> Any:
        return self.cursor.loader(self.cursor.frames[self.cursor.index].images[self.name])


def _recorded_frame(
    data: object,
    index: int,
    root: Path,
    request: InspectionRequest,
) -> RecordedFrame:
    path = f"recording.frames[{index}]"
    if not isinstance(data, Mapping):
        raise RecordingError(f"{path} must be a mapping")
    images_data = data.get("images")
    expected_data = data.get("expected")
    if not isinstance(images_data, Mapping) or set(images_data) != set(request.cameras):
        raise RecordingError(f"{path}.images must contain exactly {list(request.cameras)}")
    if not isinstance(expected_data, Mapping) or set(expected_data) != set(request.predicates):
        raise RecordingError(f"{path}.expected must contain exactly {list(request.predicates)}")

    images: dict[str, Path] = {}
    for camera, value in images_data.items():
        if not isinstance(value, str) or not value:
            raise RecordingError(f"{path}.images.{camera} must be a path string")
        image_path = (root / value).resolve()
        if not image_path.is_file():
            raise RecordingError(f"{path}.images.{camera} does not exist: {image_path}")
        images[camera] = image_path
    expected = {
        predicate: _expected_verdict(value, f"{path}.expected.{predicate}")
        for predicate, value in expected_data.items()
    }
    name = data.get("name", "")
    if not isinstance(name, str):
        raise RecordingError(f"{path}.name must be a string")
    return RecordedFrame(images=images, expected=expected, name=name)


def _expected_verdict(value: object, path: str) -> Verdict:
    verdict = Verdict.parse(value)
    valid_unknown = isinstance(value, str) and value.strip().lower() == Verdict.UNKNOWN.value
    if verdict is Verdict.UNKNOWN and not valid_unknown:
        raise RecordingError(f"{path} must be true, false, or unknown")
    return verdict


def _strings(value: object, path: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or any(not isinstance(item, str) or not item for item in value):
        raise RecordingError(f"{path} must be a non-empty list of strings")
    return tuple(value)
