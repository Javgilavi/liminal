"""Opus reviews every FLUX chunk of a run_single_arm attempt before the ws4 guard sees it (``--hybrid-config``).

The decision is liminal's hybrid gate (src/liminal/policy/hybrid.py: RGB, joints, robot-only FK proposals, episode
history; student / edit / eef / stop). It runs in the runner's inference worker thread, so the protocol is
LeRobot's synchronous one: the arm holds its last command while FLUX and then Opus answer, the history does not
advance, the stop keys and STOP file keep working.

Minimum path, because liminal's FK mapping for the Haller is not verified (``kinematics_verified: false``):
- ``student`` (continue): the unchanged FLUX chunk goes on to the runner, every command through SafetyGuard /
  KeepOutGuard (1 deg/tick, keep-out, envelope), the stall and load freeze and the torque cap, as without the gate.
- ``stop``, and ``edit`` / ``eef`` (a correction this mapping cannot execute): HybridStop is raised, which the runner
  turns into its e-stop freeze (``guard.estop()`` + ``hold_here``). Stop never claims success.
- Any model, API or validation error also freezes (fail closed).
The gate never adds motion; the only commands are FLUX's own, through the same guard.
"""

from __future__ import annotations

import asyncio
import base64
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]


class HybridStop(RuntimeError):
    """Opus stopped the attempt, or asked for a correction the unverified mapping cannot execute."""


def _liminal_hybrid():
    try:
        import liminal.policy.hybrid as hybrid
    except ImportError:  # the robot env runs this from a checkout without `pip install -e .`
        sys.path.insert(0, str(REPO / "src"))
        import liminal.policy.hybrid as hybrid
    return hybrid


def jpeg_base64(rgb: np.ndarray) -> str:
    import cv2

    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("hybrid frame JPEG encoding failed")
    return base64.b64encode(encoded.tobytes()).decode()


class GatedBackend:
    """A FluxBackend (flux.backends) whose every chunk Opus reviews first; one instance = one arm's gate."""

    def __init__(self, inner: Any, config_path: str | Path, log_path: str | Path, *, controller: Any = None,
                 encode: Any = jpeg_base64):  # fmt: skip
        import yaml

        hybrid = _liminal_hybrid()
        raw = yaml.safe_load(Path(config_path).read_text())
        config = hybrid.HybridConfig(**raw)
        if not config.enabled:
            raise SystemExit(f"{config_path}: enabled is false")
        if config.kinematics_verified:
            # Corrections would need liminal's local IK with this mapping, which this wrapper does not execute.
            raise SystemExit("the ws4 gate runs continue/stop only; use a kinematics_verified: false config")
        self.inner, self.config, self.encode = inner, config, encode
        self.controller = controller or hybrid.HybridController(config)
        self.contract = inner.contract  # run_single_arm may replace it (--action-steps); the runner reads this one
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._executed: list[list[float]] | None = None
        self._segment: Any = None
        self.decisions: list[dict] = []

    def __getattr__(self, name: str) -> Any:  # fetch_imagination, info, ... of the wrapped backend
        return getattr(self.inner, name)

    def log(self, kind: str, **values: Any) -> None:
        if "images" in values:  # frames stay out of the log
            values["images"] = [image["label"] for image in values["images"]]
        with self.log_path.open("a") as stream:
            stream.write(json.dumps({"t": time.time(), "kind": kind, **values}, default=str) + "\n")

    def new_attempt(self, task: str) -> None:
        """Every attempt is its own episode: no history or frames from the previous one."""
        self.controller.reset()
        self._executed = self._segment = None
        self.log("hybrid_attempt", task=task, model=self.config.model)

    def predict(self, request: Any) -> Any:
        response = self.inner.predict(request)
        if request.target == "warmup":
            return response
        measured = [float(v) for v in np.asarray(request.states, dtype=np.float64)[-1]]
        if self._segment is not None:
            # The previous chunk's commands as handed to the runner (the guard may have clipped them; the request's
            # command history records the clipped ones for FLUX).
            self.controller.record(self._segment, self._executed, measured, self.log)
            self._segment = None
        frames = {role: self.encode(np.asarray(request.images[f"observation.images.{role}"])[-1])
                  for role in ("scene", "wrist")}  # fmt: skip
        commands = np.asarray(response.commands, dtype=np.float64)
        horizon = int(self.contract.n_action_steps)
        try:
            segment = asyncio.run(self.controller.choose(
                goal=request.task, actions=commands.tolist(), observation={"state": measured, **frames},
                max_steps=horizon, log=self.log))  # fmt: skip
        except Exception as exc:
            print(f"OPUS GATE ERROR -> freeze: {type(exc).__name__}: {str(exc)[:160]}", flush=True)
            raise HybridStop(f"gate error {type(exc).__name__}") from exc
        decision = segment.decision
        self.decisions.append(decision)
        status = decision["assessment"]
        print(f"OPUS {decision['mode']} {decision['steps']} ({status['execution_status']}, "
              f"{status['intent_status']}): {decision['reason'][:200]}", flush=True)  # fmt: skip
        if decision["mode"] == "stop":
            raise HybridStop(f"Opus stop: {decision['reason'][:100]}")
        if decision["mode"] != "student":
            raise HybridStop(f"Opus asked for an {decision['mode']} correction; mapping unverified, freezing")
        self._segment, self._executed = segment, commands[:horizon].tolist()
        return response
