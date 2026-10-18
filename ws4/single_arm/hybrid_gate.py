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
- Any model, API or validation error also freezes (fail closed). In the orchestrator the freeze is the executor's
  e-stop ("control error: ..."): every arm holds, Release + Resume retries the attempt.
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
    """A FluxBackend (flux.backends) whose every chunk Opus reviews first: one gate (history, frames) per runner target.

    Used by run_single_arm (``--hybrid-config``) and by the ws4 orchestrator Session (``WS4_HYBRID_CONFIG``), where the
    executor's ``begin`` hook starts a new episode at every step attempt.
    """

    def __init__(self, inner: Any, config_path: str | Path, log_path: str | Path, *, every: int = 1,
                 notify: Any = None, encode: Any = jpeg_base64):  # fmt: skip
        import yaml

        self._hybrid = _liminal_hybrid()
        path = Path(config_path)
        raw = yaml.safe_load((path if path.is_absolute() or path.exists() else REPO / path).read_text())
        model = Path(raw.get("model_path", "sim/single_arm/so101.xml"))
        raw["model_path"] = str(model if model.is_absolute() or model.exists() else REPO / model)
        config = self._hybrid.HybridConfig(**raw)
        if not config.enabled:
            raise SystemExit(f"{config_path}: enabled is false")
        if config.kinematics_verified:
            # Corrections would need liminal's local IK with this mapping, which this wrapper does not execute.
            raise SystemExit("the ws4 gate runs continue/stop only; use a kinematics_verified: false config")
        if type(every) is not int or every < 1:
            raise SystemExit("the gate reviews every N-th chunk: N must be a positive integer")
        # Opus takes seconds per chunk and a chunk is 0.5 s of motion: review the first chunk of every attempt, then
        # every N-th. Unreviewed chunks pass unchanged (as every chunk without the gate), through the same guard.
        self.inner, self.config, self.encode, self.every = inner, config, encode, every
        self.notify = notify or (lambda message: print(message, flush=True))
        self.controllers: dict[str, Any] = {}
        self.contract = inner.contract  # run_single_arm may replace it (--action-steps); the runner reads this one
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._pending: dict[str, tuple[Any, list[list[float]]]] = {}
        self._chunks: dict[str, int] = {}
        self._waited: dict[str, float] = {}  # Opus seconds in the current attempt, per target
        self._reviewing: dict[str, float] = {}  # target -> monotonic start of the review in flight
        self.total_wait_s = 0.0
        self.decisions: list[dict] = []

    def gate_wait_s(self, target: str) -> float:
        """Seconds this attempt spent waiting for Opus (the arm held), including a review in flight: the ws4
        executor leaves them off the attempt's timeout, so a reviewed attempt keeps its motion time."""
        started = self._reviewing.get(target)
        return self._waited.get(target, 0.0) + (time.monotonic() - started if started is not None else 0.0)

    def __getattr__(self, name: str) -> Any:  # fetch_imagination, info, ... of the wrapped backend
        return getattr(self.inner, name)

    def log(self, kind: str, **values: Any) -> None:
        if "images" in values:  # frames stay out of the log
            values["images"] = [image["label"] for image in values["images"]]
        with self.log_path.open("a") as stream:
            stream.write(json.dumps({"t": time.time(), "kind": kind, **values}, default=str) + "\n")

    def new_attempt(self, task: str, target: str | None = None) -> None:
        """Every attempt is its own episode: no history or frames from the previous one."""
        for name in [target] if target is not None else list(self.controllers):
            if name in self.controllers:
                self.controllers[name].reset()
            self._pending.pop(name, None)
            self._chunks.pop(name, None)
            self._waited.pop(name, None)
        self.log("hybrid_attempt", task=task, target=target, model=self.config.model, every=self.every)

    def begin(self, target: str, arm: str, skill: Any) -> None:
        """The ws4 executor's per-attempt hook (executor._start_step)."""
        if hasattr(self.inner, "begin"):
            self.inner.begin(target, arm, skill)
        self.new_attempt(getattr(skill, "prompt", "") or getattr(skill, "name", ""), target)

    def predict(self, request: Any) -> Any:
        response = self.inner.predict(request)
        if request.target == "warmup":
            return response
        target = request.target or "policy"
        if target not in self.controllers:
            self.controllers[target] = self._hybrid.HybridController(self.config)
        controller = self.controllers[target]
        commands = np.asarray(response.commands, dtype=np.float64)
        horizon = int(self.contract.n_action_steps)
        index = self._chunks.get(target, 0)
        self._chunks[target] = index + 1
        if index % self.every:
            if target in self._pending:  # executed since the last review, for that decision's history entry
                self._pending[target][1].extend(commands[:horizon].tolist())
            return response
        measured = [float(v) for v in np.asarray(request.states, dtype=np.float64)[-1]]
        if target in self._pending:
            # Commands as handed to the runner since the last review (the guard may have clipped them; the request's
            # command history records the clipped ones for FLUX), at most 32 of them so the payload stays small.
            segment, executed = self._pending.pop(target)
            step = -(-len(executed) // 32)
            controller.record(segment, executed[::step] + ([executed[-1]] if (len(executed) - 1) % step else []),
                              measured, self.log)  # fmt: skip
        frames = {role: self.encode(np.asarray(request.images[f"observation.images.{role}"])[-1])
                  for role in ("scene", "wrist")}  # fmt: skip
        self._reviewing[target] = time.monotonic()
        try:
            segment = asyncio.run(controller.choose(
                goal=request.task, actions=commands.tolist(), observation={"state": measured, **frames},
                max_steps=horizon, log=self.log))  # fmt: skip
        except Exception as exc:
            self.notify(f"OPUS GATE ERROR -> freeze: {type(exc).__name__}: {str(exc)[:160]}")
            raise HybridStop(f"Opus gate error {type(exc).__name__}") from exc
        finally:
            waited = time.monotonic() - self._reviewing.pop(target)
            self._waited[target] = self._waited.get(target, 0.0) + waited
            self.total_wait_s += waited
        decision = segment.decision
        self.decisions.append(decision)
        status = decision["assessment"]
        self.notify(f"OPUS {decision['mode']} {decision['steps']} ({status['execution_status']}, "
                    f"{status['intent_status']}): {decision['reason'][:200]}")  # fmt: skip
        if decision["mode"] == "stop":
            raise HybridStop(f"Opus stop: {decision['reason'][:100]}")
        if decision["mode"] != "student":
            raise HybridStop(f"Opus asked for an {decision['mode']} correction; mapping unverified, freezing")
        self._pending[target] = (segment, commands[:horizon].tolist())
        return response
