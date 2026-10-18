"""Shared lifecycle for CLI and UI; unwind partial startup and always release hardware."""
from __future__ import annotations

import logging
from contextlib import contextmanager

from liminal.safety import JointLimits


@contextmanager
def connected_cameras(cameras):
    """Connect only cameras; perception preflight must never touch arm buses."""

    resources = []
    try:
        for camera in cameras.values():
            if hasattr(camera, "connect"):
                resources.append(camera)
                camera.connect()
        yield cameras
    finally:
        for camera in reversed(resources):
            try:
                camera.disconnect()
            except Exception:
                logging.getLogger(__name__).exception("shutdown failed for %s", camera.name)


@contextmanager
def connected(ctx):
    resources = []
    try:
        for name, arm in ctx.arms.items():
            if hasattr(arm, "on_fault"):
                arm.on_fault = ctx.safety.estop
            resources.append(arm)
            arm.connect()
            calibrated = getattr(arm, "joint_limits", None)
            if calibrated is not None:
                lower, upper = calibrated
                configured = ctx.safety.limits.get(name)
                if configured:
                    lower = [max(a, b) for a, b in zip(lower, configured.lower, strict=True)]
                    upper = [min(a, b) for a, b in zip(upper, configured.upper, strict=True)]
                if any(lo >= hi for lo, hi in zip(lower, upper, strict=True)):
                    raise ValueError(f"{name}: configured limits do not overlap motor calibration")
                ctx.safety.limits[name] = JointLimits(lower, upper, configured.max_step if configured else 2.0)
                arm.joint_limits = lower, upper
        for camera in ctx.cameras.values():
            if hasattr(camera, "connect"):
                resources.append(camera)
                camera.connect()
        yield ctx
    finally:
        for resource in reversed(resources):
            try:
                resource.disconnect()
            except Exception:
                logging.getLogger(__name__).exception("shutdown failed for %s", resource.name)
