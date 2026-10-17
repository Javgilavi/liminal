"""Provider-neutral vision model interface and deterministic test double."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

ModelResponse = str | Mapping[str, Any]


class VisionModel(Protocol):
    """An async VLM capable of inspecting one or more ordered camera views."""

    async def inspect(self, prompt: str, images: list[Any]) -> ModelResponse: ...


class FixtureVisionModel:
    """Return predefined responses without loading a real model."""

    def __init__(self, responses: Sequence[ModelResponse | Exception]):
        if not responses:
            raise ValueError("fixture vision model needs at least one response")
        self._responses = deque(responses)
        self.calls: list[tuple[str, list[Any]]] = []

    async def inspect(self, prompt: str, images: list[Any]) -> ModelResponse:
        self.calls.append((prompt, images))
        if not self._responses:
            raise RuntimeError("fixture vision model has no responses left")
        response = self._responses.popleft()
        if isinstance(response, Exception):
            raise response
        return response
