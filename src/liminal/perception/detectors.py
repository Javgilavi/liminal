"""Predicate detectors: camera frames -> symbolic facts (owned by WS3).

Start with the cheapest thing that works on the demo table (coloured markers in fixed
zones), and only add a VLM verifier for predicates that are hard to hand-code.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from liminal.core.interfaces import PredicateDetector
from liminal.perception.world_state import WorldState


@dataclass
class ColorZoneRule:
    """`predicate` is true when at least `min_fraction` of pixels inside `roi`
    (x0, y0, x1, y1 in pixels) fall inside the HSV range."""

    predicate: str
    camera: str
    roi: tuple[int, int, int, int]
    hsv_low: tuple[int, int, int]
    hsv_high: tuple[int, int, int]
    min_fraction: float = 0.05
    invert: bool = False


class ColorZoneDetector:
    def __init__(self, rules: Sequence[ColorZoneRule]):
        self.rules = list(rules)

    def detect(self, frames: dict[str, Any]) -> dict[str, bool]:
        import cv2
        import numpy as np

        out: dict[str, bool] = {}
        for r in self.rules:
            frame = frames.get(r.camera)
            if frame is None:
                continue
            x0, y0, x1, y1 = r.roi
            hsv = cv2.cvtColor(np.asarray(frame[y0:y1, x0:x1], dtype=np.uint8), cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(hsv, np.array(r.hsv_low), np.array(r.hsv_high))
            hit = (mask > 0).mean() >= r.min_fraction
            out[r.predicate] = (not hit) if r.invert else bool(hit)
        return out


class VLMVerifier:
    """Ask a VLM yes/no questions about the scene, e.g.
    {"object_inserted": "Is the red block inside the box?"}. Slow: call on demand
    (after a skill) rather than in the perception loop."""

    def __init__(self, model: Any, questions: dict[str, str], camera: str = "scene"):
        self.model = model  # any liminal.orchestration.planner.TextModel
        self.questions = questions
        self.camera = camera

    def detect(self, frames: dict[str, Any], jpeg: bytes | None = None) -> dict[str, bool]:
        if jpeg is None:
            import cv2

            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(frames[self.camera], cv2.COLOR_RGB2BGR))
            jpeg = buf.tobytes() if ok else b""
        out = {}
        for pred, q in self.questions.items():
            answer = self.model.complete("Answer strictly 'yes' or 'no'.", q, [jpeg])
            out[pred] = answer.strip().lower().startswith("yes")
        return out


class PerceptionLoop:
    """Poll cameras, run detectors, push facts into the world state."""

    def __init__(self, cameras: dict[str, Any], detectors: Sequence[PredicateDetector], world: WorldState,
                 rate_hz: float = 10.0):
        self.cameras, self.detectors, self.world, self.rate_hz = cameras, list(detectors), world, rate_hz
        self._task: asyncio.Task[None] | None = None

    async def step(self) -> None:
        frames = {name: cam.read() for name, cam in self.cameras.items()}
        for d in self.detectors:
            self.world.update(d.detect(frames), source=type(d).__name__)

    async def _loop(self) -> None:
        while True:
            await self.step()
            await asyncio.sleep(1.0 / self.rate_hz)

    def start(self) -> None:
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)
