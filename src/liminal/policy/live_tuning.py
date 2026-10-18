"""Validated runtime overrides for the released SO-101 inference profile."""

from __future__ import annotations

import math

DEFAULTS = {
    "n_action_steps": 32,
    "num_inference_steps": 4,
    "guidance_scale": 3.0,
    "guidance_scale_action": None,
    "sampler_shift": 6.93,
    "seed": 42,
}


def inference_settings(cfg: dict) -> dict:
    supplied = cfg.get("inference") or {}
    if not isinstance(supplied, dict) or set(supplied) - set(DEFAULTS):
        raise ValueError("unknown or malformed inference settings")
    settings = {**DEFAULTS, **supplied}
    for key in ("n_action_steps", "num_inference_steps", "seed"):
        if isinstance(settings[key], bool) or not isinstance(settings[key], int):
            raise ValueError(f"{key} must be an integer")
    if not 1 <= settings["n_action_steps"] <= 42:
        raise ValueError("n_action_steps must be in [1, 42]")
    if settings["num_inference_steps"] < 1:
        raise ValueError("num_inference_steps must be positive")
    if not 0 <= settings["seed"] < 2**63:
        raise ValueError("seed must be in [0, 2**63)")
    for key in ("guidance_scale", "guidance_scale_action", "sampler_shift"):
        if key == "guidance_scale_action" and settings[key] is None:
            continue
        settings[key] = float(settings[key])
        if not math.isfinite(settings[key]):
            raise ValueError(f"{key} must be finite")
    if settings["sampler_shift"] <= 0:
        raise ValueError("sampler_shift must be positive")
    return settings


def worker_arguments(settings: dict) -> list[str]:
    return [arg for key, value in settings.items() if value is not None
            for arg in ("--" + key.replace("_", "-"), str(value))]


def rollout_duration(cfg: dict) -> float | None:
    value = cfg.get("duration_s")
    if value is None:
        return None
    duration = float(value)
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration_s must be null (unlimited) or finite and positive")
    return duration
