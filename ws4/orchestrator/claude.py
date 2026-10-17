"""One structured-output Claude call with cost logging, a budget guard and refusal handling.

Patterns follow the claude-api skill (Python SDK 1.x): ``output_config.format`` with a JSON schema, adaptive
thinking with an explicit ``effort``, ``stop_reason`` checked before reading content, and server-side refusal
fallbacks (``fallbacks: "default"``, beta ``server-side-fallback-2026-07-01``) on the models that support them
(Opus 5.5, Sonnet 5.5; Haiku 5.5 has none). The API key is read from ``ANTHROPIC_API_KEY`` only and never logged.
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
from typing import Any

import numpy as np

from .cost import CostMeter

FALLBACK_BETA = "server-side-fallback-2026-07-01"
FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5"}


class ClaudeRefused(RuntimeError):
    pass


class ClaudeUnavailable(RuntimeError):
    pass


def image_block(img: np.ndarray, max_side: int = 640, quality: int = 90) -> dict[str, Any]:
    """Base64 JPEG image block. Vision costs about (w * h) / 750 tokens: a 640x480 frame is about 410."""
    from PIL import Image

    pil = Image.fromarray(np.asarray(img).astype(np.uint8))
    scale = max_side / max(pil.size)
    if scale < 1:
        pil = pil.resize((round(pil.size[0] * scale), round(pil.size[1] * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    pil.convert("RGB").save(buf, format="JPEG", quality=quality)
    data = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}


def api_key_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


class ClaudeJSON:
    def __init__(self, meter: CostMeter, client: Any = None, timeout_s: float = 90.0, max_retries: int = 2):
        self.meter = meter
        self._client = client
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    @property
    def client(self) -> Any:
        if self._client is None:
            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise ClaudeUnavailable("ANTHROPIC_API_KEY is not set in this shell")
            import anthropic

            self._client = anthropic.Anthropic(api_key=key, timeout=self.timeout_s, max_retries=self.max_retries)
        return self._client

    def call(self, *, purpose: str, model: str, system: str, messages: list[dict], schema: dict,
             effort: str = "low", max_tokens: int = 16000, cache_system: bool = True) -> tuple[dict, Any]:
        """Returns ``(parsed JSON, response)``. Raises ``BudgetExceeded`` before calling when the budget is spent,
        ``ClaudeRefused`` on a refusal (after any fallback), ``RuntimeError`` on truncated or non-JSON output."""
        self.meter.check(purpose)
        system_block: Any = system
        if cache_system:
            # Stable prefix (system prompt + vocabulary): cacheable once it exceeds the model's minimum (512 tokens).
            system_block = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system_block,
            "messages": messages,
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": schema}},
        }
        start = time.perf_counter()
        try:
            if model in FALLBACK_MODELS:
                response = self.client.beta.messages.create(betas=[FALLBACK_BETA], fallbacks="default", **kwargs)
            else:
                response = self.client.messages.create(**kwargs)
        except Exception as exc:
            self.meter.record(purpose, model, None, time.perf_counter() - start, ok=False,
                              note=f"{type(exc).__name__}: {str(exc)[:200]}")
            raise
        latency = time.perf_counter() - start
        served_by = getattr(response, "model", None) or model
        stop = getattr(response, "stop_reason", None)
        self.meter.record(purpose, served_by, getattr(response, "usage", None), latency, ok=stop != "refusal",
                          note=f"stop={stop}")
        if stop == "refusal":
            details = getattr(response, "stop_details", None)
            raise ClaudeRefused(str(getattr(details, "category", None) or "refusal"))
        if stop == "max_tokens":
            raise RuntimeError(f"{purpose}: output truncated at max_tokens={max_tokens}")
        text = next((b.text for b in response.content if getattr(b, "type", None) == "text"), None)
        if text is None:
            raise RuntimeError(f"{purpose}: no text block in the response (stop_reason={stop})")
        try:
            return json.loads(text), response
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{purpose}: response is not JSON: {exc}") from exc
