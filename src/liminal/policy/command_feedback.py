"""Validate committed targets and preserve FLUX joint deltas after intervention."""

from __future__ import annotations

import math

from liminal.core.types import SO101_JOINTS


def committed_target(message: dict, run_id: str, seq: int, side: str) -> list[float]:
    if message.get("run_id") != run_id or type(message.get("seq")) is not int or message["seq"] != seq:
        raise RuntimeError("command acknowledgement has the wrong run or sequence")
    result = message.get("result", {})
    sides = result.get("sides", {})
    if result.get("run_id") != run_id or result.get("seq") != seq or set(sides) != {side}:
        raise RuntimeError("command acknowledgement does not identify the selected arm")
    outcome = sides[side]
    if outcome.get("refused") or outcome.get("dropped"):
        raise RuntimeError(f"Haller did not commit the complete action: {outcome}")
    values = outcome.get("committed", {})
    if outcome.get("arm") != side or set(values) != set(SO101_JOINTS):
        raise RuntimeError("command acknowledgement has missing or unexpected joints")
    command = [float(values[joint]) for joint in SO101_JOINTS]
    if not all(math.isfinite(v) for v in command):
        raise RuntimeError("non-finite committed command")
    return command


def rebase_action(predicted: list[float], previous_prediction: list[float],
                  committed: list[float]) -> list[float]:
    """Apply each arm-joint delta to the actual last command; gripper is absolute.

    Without this, a slew-limited step's unexecuted remainder leaks into every
    later absolute target in the same chunk, even with correct history later.
    """
    if any(len(v) != len(SO101_JOINTS) for v in (predicted, previous_prediction, committed)):
        raise ValueError("expected six action channels")
    if not all(math.isfinite(v) for values in (predicted, previous_prediction, committed) for v in values):
        raise ValueError("non-finite action channel")
    return [actual + target - old for target, old, actual in
            zip(predicted[:-1], previous_prediction[:-1], committed[:-1], strict=True)] + [predicted[-1]]
