"""Motors-disabled live camera inspection."""

from __future__ import annotations

import asyncio
import math
import time
from pathlib import Path
from typing import Any

from liminal.perception.inspector import SceneInspector
from liminal.perception.types import InspectionRequest


class SnapshotCamera:
    """Camera decorator that preserves every RGB frame used by the inspector."""

    def __init__(self, camera: Any, output_dir: str | Path):
        self.camera = camera
        self.name = camera.name
        self.output_dir = Path(output_dir)
        self.capture_count = 0
        self.last_capture_path: Path | None = None

    def connect(self) -> None:
        connect = getattr(self.camera, "connect", None)
        if callable(connect):
            connect()

    def disconnect(self) -> None:
        disconnect = getattr(self.camera, "disconnect", None)
        if callable(disconnect):
            disconnect()

    def begin_sample(self) -> None:
        self.last_capture_path = None

    def read(self) -> Any:
        frame = self.camera.read()
        self.capture_count += 1
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"{self.capture_count:03d}_{self.name}.png"
        _save_rgb(frame, path)
        self.last_capture_path = path
        return frame


def _save_rgb(frame: Any, path: Path) -> None:
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("live frame capture requires numpy and Pillow") from exc

    if isinstance(frame, Image.Image):
        frame.convert("RGB").save(path)
        return
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] not in {3, 4}:
        raise ValueError(f"live camera frame must be HxWx3/4, got {array.shape}")
    if array.dtype != np.uint8:
        if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
            raise ValueError("live camera frame must contain finite numeric pixels")
        if array.min() < 0 or array.max() > 255:
            raise ValueError("live camera pixels must be in the range 0..255")
        array = array.astype(np.uint8)
    Image.fromarray(array, mode="RGBA" if array.shape[2] == 4 else "RGB").convert("RGB").save(path)


async def inspect_live(
    inspector: SceneInspector,
    request: InspectionRequest,
    *,
    samples: int = 3,
    interval_s: float = 0.25,
) -> dict[str, Any]:
    """Inspect live views repeatedly and expose every blackboard update."""

    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("live inspection samples must be a positive integer")
    if isinstance(interval_s, bool) or not isinstance(interval_s, int | float) or not math.isfinite(interval_s):
        raise ValueError("live inspection interval must be finite")
    if interval_s < 0:
        raise ValueError("live inspection interval cannot be negative")

    calls = []
    aborted_reason = None
    for index in range(samples):
        for camera_name in request.cameras:
            begin_sample = getattr(inspector.cameras.get(camera_name), "begin_sample", None)
            if callable(begin_sample):
                begin_sample()
        started_at = time.perf_counter()
        assessment = await inspector.inspect(request)
        calls.append({
            "sample": index + 1,
            "latency_s": time.perf_counter() - started_at,
            "captured_at": assessment.captured_at,
            "parse_error": assessment.parse_error,
            "raw_response": assessment.raw_response,
            "images": {
                camera_name: str(path)
                for camera_name in request.cameras
                if (path := getattr(inspector.cameras.get(camera_name), "last_capture_path", None)) is not None
            },
            "predicates": {
                predicate: {
                    "value": assessment.observation(predicate).value.value,
                    "evidence": assessment.observation(predicate).evidence,
                    "observation_version": inspector.world.observation_version(predicate),
                }
                for predicate in request.predicates
            },
        })
        unhealthy = getattr(inspector.model, "unhealthy_reason", None)
        if unhealthy or assessment.parse_error == "vision model timed out":
            aborted_reason = unhealthy or assessment.parse_error
            break
        if index + 1 < samples and interval_s:
            await asyncio.sleep(interval_s)

    final = {}
    for predicate in request.predicates:
        record = inspector.world.observation(predicate)
        final[predicate] = None if record is None else {
            "value": record.value,
            "observation_version": record.version,
            "captured_at": record.captured_at,
            "source": record.source,
            "evidence": record.evidence,
        }
    return {
        "arm_buses_connected": False,
        "arm_commands_issued": False,
        "aborted_reason": aborted_reason,
        "passed": aborted_reason is None and all(
            inspector.world.known(predicate) for predicate in request.predicates
        ),
        "samples": calls,
        "final_blackboard": final,
    }
