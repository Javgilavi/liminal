"""Offline evaluation of a VisionModel against recorded scene frames."""

from __future__ import annotations

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


@dataclass
class ReplayReport:
    recording: str
    frames: int
    observations: int
    correct: int = 0
    unknown: int = 0
    mismatches: list[ReplayMismatch] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.observations if self.observations else 0.0

    @property
    def passed(self) -> bool:
        return self.correct == self.observations


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
) -> ReplayReport:
    """Run the production prompt/parser on saved frames and score its verdicts."""

    cursor = _FrameCursor(recording.frames, image_loader or load_rgb_image)
    cameras = {name: _RecordedCamera(name, cursor) for name in recording.request.cameras}
    inspector = SceneInspector(model, cameras, WorldState(), confirmations=1, source="recording_replay")
    report = ReplayReport(
        recording=recording.name,
        frames=len(recording.frames),
        observations=len(recording.frames) * len(recording.request.predicates),
    )
    for index, frame in enumerate(recording.frames):
        cursor.index = index
        inspector.reset()
        assessment = await inspector.inspect(recording.request)
        frame_name = frame.name or str(index)
        for predicate, expected in frame.expected.items():
            observation = assessment.observation(predicate)
            if observation.value is Verdict.UNKNOWN:
                report.unknown += 1
            if observation.value is expected:
                report.correct += 1
            else:
                report.mismatches.append(
                    ReplayMismatch(frame_name, predicate, expected, observation.value, observation.evidence)
                )
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
