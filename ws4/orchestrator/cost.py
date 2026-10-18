"""Token usage, estimated cost per Claude call, and a session budget guard.

Prices are USD per million tokens: base input, output, cache read, 5-minute cache write, from the official pricing
page (https://platform.claude.com/docs/en/about-claude/pricing, read 2026-10-17). Cache reads are 0.05x input on Opus
5.5 and Sonnet 5.5 and 0.1x on Haiku 5.5. Haiku 5.5's rate is for prompts up to 100,000 tokens (ours are a few
thousand; above that it is USD 0.50 / 2.50). Thinking tokens are billed as output and are included in
``usage.output_tokens``; ``usage.output_tokens_details.thinking_tokens`` breaks them out
(https://platform.claude.com/docs/en/build-with-claude/thinking-steering-and-cost).
Doc vs skill: the claude-api skill (cached 2026-10-06) lists Sonnet 5.5 cache reads at USD 0.20; the pricing page says
USD 0.10 (0.05x of USD 2). The pricing page wins.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

PRICES: dict[str, tuple[float, float, float, float]] = {
    "claude-opus-5-5": (4.00, 20.00, 0.20, 5.00),
    "claude-sonnet-5-5": (2.00, 10.00, 0.10, 2.50),
    "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125),
}
FALLBACK_PRICE = PRICES["claude-opus-5-5"]  # unknown model (for example a server-side fallback): assume Opus rates


class BudgetExceeded(RuntimeError):
    pass


def _field(obj: Any, key: str) -> Any:
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)


def _get(usage: Any, key: str) -> int:
    return int(_field(usage, key) or 0)


def cost_of(model: str, usage: Any) -> float:
    p_in, p_out, p_read, p_write = PRICES.get(model, FALLBACK_PRICE)
    return (_get(usage, "input_tokens") * p_in + _get(usage, "output_tokens") * p_out
            + _get(usage, "cache_read_input_tokens") * p_read
            + _get(usage, "cache_creation_input_tokens") * p_write) / 1e6


@dataclass
class CallRecord:
    purpose: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    thinking_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    ok: bool = True
    note: str = ""
    time: float = field(default_factory=time.time)


class CostMeter:
    """Thread-safe ledger of Claude calls. ``check`` refuses new calls once the session budget is spent."""

    def __init__(self, budget_usd: float = 5.0, log_path: str | Path | None = None):
        self.budget_usd = float(budget_usd)
        self.log_path = Path(log_path) if log_path else None
        self.calls: list[CallRecord] = []
        self._lock = threading.Lock()

    @property
    def total_usd(self) -> float:
        with self._lock:
            return sum(c.cost_usd for c in self.calls)

    def check(self, purpose: str) -> None:
        spent = self.total_usd
        if spent >= self.budget_usd:
            raise BudgetExceeded(f"session budget USD {self.budget_usd:.2f} spent (USD {spent:.4f}); "
                                 f"refusing {purpose} call")

    def record(self, purpose: str, model: str, usage: Any = None, latency_s: float = 0.0, ok: bool = True,
               note: str = "") -> CallRecord:
        usage = usage or {}
        rec = CallRecord(
            purpose=purpose, model=model,
            input_tokens=_get(usage, "input_tokens"), output_tokens=_get(usage, "output_tokens"),
            cache_read_tokens=_get(usage, "cache_read_input_tokens"),
            cache_write_tokens=_get(usage, "cache_creation_input_tokens"),
            thinking_tokens=_get(_field(usage, "output_tokens_details") or {}, "thinking_tokens"),
            cost_usd=cost_of(model, usage), latency_s=float(latency_s), ok=ok, note=note,
        )
        with self._lock:
            self.calls.append(rec)
            if self.log_path is not None:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a") as f:
                    f.write(json.dumps(asdict(rec)) + "\n")
        return rec

    def summary(self) -> dict[str, Any]:
        with self._lock:
            calls = list(self.calls)
        by: dict[str, dict[str, Any]] = {}
        for c in calls:
            row = by.setdefault(c.purpose, {"calls": 0, "cost_usd": 0.0, "latency_s": [], "input_tokens": 0,
                                            "output_tokens": 0, "failed": 0})
            row["calls"] += 1
            row["cost_usd"] += c.cost_usd
            row["latency_s"].append(c.latency_s)
            row["input_tokens"] += c.input_tokens + c.cache_read_tokens + c.cache_write_tokens
            row["output_tokens"] += c.output_tokens
            row["failed"] += 0 if c.ok else 1
        for row in by.values():
            lat = sorted(row.pop("latency_s"))
            row["median_latency_s"] = lat[len(lat) // 2] if lat else None
            row["max_latency_s"] = lat[-1] if lat else None
            row["cost_usd"] = round(row["cost_usd"], 6)
        total = sum(c.cost_usd for c in calls)
        return {"total_usd": round(total, 6), "budget_usd": self.budget_usd,
                "remaining_usd": round(max(0.0, self.budget_usd - total), 6), "by_purpose": by}
