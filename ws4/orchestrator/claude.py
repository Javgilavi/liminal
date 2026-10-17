"""One structured-output Claude call with cost logging, a budget guard and refusal handling.

Coded against the Anthropic Python SDK ``anthropic==1.15.0`` (pinned in ``requirements.txt``; Python >= 3.10, httpx2
transport). Every API feature used here, with the official page it follows (read 2026-10-17):

- Messages API, ``client.messages.create``: https://platform.claude.com/docs/en/api/messages/create
- Structured outputs, ``output_config.format`` with ``type: "json_schema"`` (closed objects, ``enum``, no numeric
  bounds; the first request with a new schema compiles a grammar, cached 24 h; enum casing may differ):
  https://platform.claude.com/docs/en/build-with-claude/structured-outputs
  Chosen over strict tool use because we want the response itself as JSON, not a tool call (same page, "Strict tool
  use vs JSON outputs"), and forced ``tool_choice`` is a 400 on Opus 5.5 and Sonnet 5.5
  (https://platform.claude.com/docs/en/api/errors, "Forced tool use not supported").
- Adaptive thinking and ``output_config.effort`` (default ``medium`` on Opus 5.5 and Haiku 5.5, ``high`` on Sonnet
  5.5; thinking tokens are billed as output; Opus 5.5 cannot disable thinking, Haiku 5.5 can at effort <= high):
  https://platform.claude.com/docs/en/build-with-claude/thinking-steering-and-cost and .../api/errors
- Refusals and server-side fallback (``fallbacks: "default"`` with beta ``server-side-fallback-2026-07-01``; Opus 5.5
  and Sonnet 5.5 only, Claude API only; Haiku 5.5 has no server-side fallback):
  https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback
- Vision (base64 JPEG/PNG/GIF/WebP, at most 8000x8000 px and 10 MB per image, 32 MB per request; cost is
  ceil(w / 28) * ceil(h / 28) tokens; images before text work best): https://platform.claude.com/docs/en/build-with-claude/vision
- Prompt caching (``cache_control`` on the system block; minimum 512 tokens on all three 5.5 models; changing effort
  or the output format invalidates it): https://platform.claude.com/docs/en/build-with-claude/prompt-caching
- Errors and retries (the SDK retries 408/409/429/5xx and connection errors twice with backoff, honouring
  ``retry-after``; a spend-cap 429 has no ``retry-after``): https://platform.claude.com/docs/en/api/errors,
  https://platform.claude.com/docs/en/api/rate-limits, https://platform.claude.com/docs/en/cli-sdks-libraries/sdks/python

The API key comes from ``ANTHROPIC_API_KEY``, else from the file in ``ANTHROPIC_API_KEY_FILE`` (default
``~/.config/ws4/anthropic.env``, one line ``ANTHROPIC_API_KEY=...``). It is never printed, logged or stored.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from .cost import CostMeter

SDK_VERSION = "1.15.0"
FALLBACK_BETA = "server-side-fallback-2026-07-01"
FALLBACK_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5"}
KEY_FILE = Path("~/.config/ws4/anthropic.env")


class ClaudeRefused(RuntimeError):
    pass


class ClaudeUnavailable(RuntimeError):
    pass


def image_tokens(width: int, height: int) -> int:
    """Visual tokens of an image that is not downscaled (28x28 px patches, vision docs "Resolution and token cost")."""
    return math.ceil(width / 28) * math.ceil(height / 28)


def image_block(img: np.ndarray, max_side: int = 640, quality: int = 90) -> dict[str, Any]:
    """Base64 JPEG image block, downscaled to ``max_side``: a 640x480 frame costs 23 * 18 = 414 tokens. The 5.5 models
    accept up to 2576 px (4784 tokens) natively, so we choose the size for cost and latency, not a limit."""
    from PIL import Image

    pil = Image.fromarray(np.asarray(img).astype(np.uint8))
    scale = max_side / max(pil.size)
    if scale < 1:
        pil = pil.resize((round(pil.size[0] * scale), round(pil.size[1] * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    pil.convert("RGB").save(buf, format="JPEG", quality=quality)
    data = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}


def _key_file() -> Path:
    return Path(os.environ.get("ANTHROPIC_API_KEY_FILE") or KEY_FILE).expanduser()


def resolve_api_key() -> str:
    """The key from the environment, else from the key file; ``ClaudeUnavailable`` (never showing a key) if neither."""
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if key:
        return key
    path = _key_file()
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if line.startswith("ANTHROPIC_API_KEY="):
                value = line.split("=", 1)[1].strip().strip("'\"")
                if value:
                    return value
    except OSError:
        pass
    raise ClaudeUnavailable(f"no Anthropic API key: set ANTHROPIC_API_KEY or put ANTHROPIC_API_KEY=... in {path}")


def api_key_available() -> bool:
    try:
        resolve_api_key()
        return True
    except ClaudeUnavailable:
        return False


def sdk_version() -> str | None:
    try:
        import anthropic
    except ImportError:
        return None
    return anthropic.__version__


class ClaudeJSON:
    def __init__(self, meter: CostMeter, client: Any = None, timeout_s: float = 180.0, max_retries: int = 2):
        self.meter = meter
        self._client = client
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    @property
    def client(self) -> Any:
        if self._client is None:
            key = resolve_api_key()
            import anthropic

            self._client = anthropic.Anthropic(api_key=key, timeout=self.timeout_s, max_retries=self.max_retries)
        return self._client

    def call(self, *, purpose: str, model: str, system: str, messages: list[dict], schema: dict,
             effort: str = "low", max_tokens: int = 16000, cache_system: bool = True,
             thinking: dict | None = None) -> tuple[dict, Any]:
        """Returns ``(parsed JSON, response)``. Raises ``BudgetExceeded`` before calling when the budget is spent,
        ``ClaudeRefused`` on a refusal (after any fallback), ``RuntimeError`` on truncated or non-JSON output."""
        self.meter.check(purpose)
        system_block: Any = system
        if cache_system:
            # Stable prefix (system prompt + vocabulary): cached once it reaches the 512-token minimum.
            system_block = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system_block,
            "messages": messages,
            "thinking": thinking or {"type": "adaptive"},
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
        request_id = getattr(response, "_request_id", None)
        self.meter.record(purpose, served_by, getattr(response, "usage", None), latency, ok=stop != "refusal",
                          note=f"stop={stop}" + (f" request_id={request_id}" if isinstance(request_id, str) else ""))
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
