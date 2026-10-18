"""One ws4 orchestrator Session that owns both Haller arms and the cameras for a whole liminal run."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from liminal.ws4_bridge import REPO, import_orchestrator

log = logging.getLogger(__name__)

# The Haller defaults of `python -m orchestrator hmi --env haller` (ws4/orchestrator/__main__.py), headless: the plan
# comes from the behavior tree, not from a planner, and every step is checked by ws4's Haiku verifier.
HALLER_SESSION: dict[str, Any] = {
    "env": "haller",
    "policy": "remote",
    "planner": "stub",
    "verifier": "claude",
    "server": "http://127.0.0.1:8790",
    "head_cam": "/dev/haller_cam_mast",
    "robot_id": "haller_right",
    "max_run_s": 1200.0,
    "task": "liminal_bt",
    "realtime": True,
    # the tree owns recovery (fallback, retry): a ws4 replan would wait for a goal and an approval that never come
    "max_replans": 0,
}
TERMINAL = ("succeeded", "failed", "aborted")  # orchestrator.executor.TERMINAL
_TUPLES = ("hold_arms", "fake_start", "right_fake_start", "image_size")


class Ws4Bridge:
    """Starts the ws4 session on first ``connect`` and closes it on the last ``disconnect`` (arms and cameras share
    it). Reads go through the session's cached observation, so the liminal side never touches a serial port."""

    def __init__(self, settings: Mapping[str, Any] | None = None, session_factory: Callable[[Any], Any] | None = None):
        self.settings = dict(settings or {})
        unknown = set(self.settings) - {"session", "token_cmd_env", "hmi_port", "poll_s"}
        if unknown:
            raise ValueError(f"unknown ws4 bridge settings {sorted(unknown)}")
        self.poll_s = float(self.settings.get("poll_s", 0.1))
        self._factory = session_factory
        self.session: Any = None
        self._hmi: Any = None
        self._users = 0
        self._lock = threading.Lock()
        self._run_lock: asyncio.Lock | None = None
        self.running: str | None = None  # ws4 arm of the step in progress

    # ------------------------------------------------------------------ lifecycle
    def session_config(self) -> dict[str, Any]:
        """HALLER_SESSION, then the robot config's ``session:``, then ``$LIMINAL_WS4_SESSION`` (a JSON object, for the
        settings of one run: ``{"observe": true}`` for a dry run, ``{"hold_arms": ["right"]}`` for a box arm handed
        over holding the lid, ``{"operator": "oscar", "layout": "P3"}`` for the trial log)."""
        try:
            run = json.loads(os.environ.get("LIMINAL_WS4_SESSION", "") or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"LIMINAL_WS4_SESSION is not JSON: {exc}") from exc
        if not isinstance(run, dict):
            raise ValueError("LIMINAL_WS4_SESSION must be a JSON object of ws4 SessionConfig fields")
        cfg = {**HALLER_SESSION, **dict(self.settings.get("session") or {}), **run}
        for key in _TUPLES:
            if cfg.get(key) is not None:
                cfg[key] = tuple(cfg[key])
        for key in ("vocabulary", "runs_dir", "trial_log", "keepout_config"):
            if cfg.get(key) and not Path(cfg[key]).is_absolute():
                cfg[key] = str(REPO / cfg[key])
        if cfg.get("token_file"):
            cfg["token_file"] = str(Path(cfg["token_file"]).expanduser())
        env = self.settings.get("token_cmd_env", "WS4_TOKEN_CMD")
        if not cfg.get("token_file") and not cfg.get("token_cmd") and os.environ.get(env):
            cfg["token_cmd"] = os.environ[env]  # e.g. "ssh -p PORT root@HOST cat /workspace/flux_token"
        return cfg

    def connect(self) -> None:
        with self._lock:
            self._users += 1
            if self.session is not None:
                return
            try:
                self.session = self._build()
            except BaseException:
                self._users -= 1
                raise
        port = int(self.settings.get("hmi_port") or 0)
        if port:
            # the ws4 HMI next to the run: events, cameras, Haiku checks and the e-stop button (spacebar)
            from orchestrator.hmi.server import make_server  # noqa: PLC0415

            self._hmi = make_server(self.session, "127.0.0.1", port)
            threading.Thread(target=self._hmi.serve_forever, daemon=True, name="ws4-hmi").start()
            log.warning("ws4 HMI on http://127.0.0.1:%d/ (e-stop: red button or spacebar)", port)

    def _build(self) -> Any:
        import_orchestrator()
        from orchestrator.session import Session, SessionConfig  # noqa: PLC0415

        cfg = SessionConfig(**self.session_config())
        if self._factory is not None:
            return self._factory(cfg)
        return Session(cfg, threaded=True)

    def disconnect(self) -> None:
        with self._lock:
            self._users = max(0, self._users - 1)
            if self._users or self.session is None:
                return
            session, self.session = self.session, None
        # As the ws4 HMI shuts down: freeze both arms with torque on, then close. Closing keeps torque on for an arm
        # that holds (the box arm) or is away from its launch pose (haller.HallerRig.close).
        session.estop("liminal run finished")
        session.close()
        if self._hmi is not None:
            self._hmi.shutdown()
            self._hmi.server_close()
            self._hmi = None

    def require(self) -> Any:
        if self.session is None:
            raise RuntimeError("the ws4 session is not connected")
        return self.session

    # ------------------------------------------------------------------ reads
    def observation(self) -> dict | None:
        return self.require().observation()

    def joints(self, arm: str) -> list[float]:
        from flux.contract import OBS_STATE, arm_slice  # noqa: PLC0415

        obs = self.observation()
        if obs is None or OBS_STATE not in obs:
            raise RuntimeError("no robot observation yet")
        return [float(v) for v in obs[OBS_STATE][arm_slice(arm)]]

    def image(self, key: str) -> Any:
        obs = self.observation()
        frame = None if obs is None else obs.get(key)
        if frame is None:
            raise RuntimeError(f"no {key} frame yet")
        return frame

    def vocab_skill(self, name: str) -> Any:
        vocab = self.require().vocab
        if name not in vocab.skills:
            raise KeyError(f"ws4 vocabulary has no skill {name!r}; have {vocab.names()}")
        return vocab.skills[name]

    def holding(self, arm: str) -> bool:
        rig = self.require().rig
        return rig is not None and (arm in rig.holding or arm in rig.hold_arms)

    # ------------------------------------------------------------------ motion (all of it inside ws4)
    def hold(self, arm: str) -> None:
        """A WHILE node's support arm: it keeps its last command (no ws4 step drives it) and its torque stays on at
        close. After open_box (``finish: hold``) or ``hold_arms`` it already holds; otherwise it is held where it is."""
        rig = self.require().rig
        if rig is not None and not self.holding(arm):
            rig.mark_holding(arm)
            self.session.events.add("warn", f"liminal: {arm} arm held as a WHILE support arm (it was not handed "
                                            "over by open_box or hold_arms): it keeps its last command")

    def freeze(self, arm: str) -> None:
        """Arm.stop: a step still running on this arm is stopped with the e-stop (freeze, torque on)."""
        if self.session is not None and self.running == arm:
            self.estop(f"liminal stopped the {arm} arm")

    def estop(self, reason: str = "liminal e-stop") -> None:
        """Both arms freeze at their measured pose with torque on, grippers keep their last command (lid held)."""
        if self.session is not None:
            self.session.estop(reason)

    def _halt(self, s: Any, previous: Any, reason: str) -> None:
        """Stop the step just approved, wherever it is: the guard latches first (the next runner tick sends nothing),
        a request the control thread has not picked up yet is dropped, and a run that started gets the e-stop (both
        arms freeze, grips kept) and then the abort."""
        self.estop(reason)
        with s._lock:
            pending = [r for r in s._requests if r[0] == "run"]
            s._requests[:] = [r for r in s._requests if r[0] != "run"]
        if pending:
            return
        deadline = time.monotonic() + 2.0  # picked up: the executor exists within a few ms
        while s.executor is previous and time.monotonic() < deadline:
            time.sleep(0.005)
        self.estop(reason)
        s.abort()
        # return once ws4 processed both: the arms are frozen and the session is idle again
        while time.monotonic() < deadline + 3.0 and s.executor is not None and s.executor.state not in TERMINAL:
            time.sleep(0.005)

    async def run_step(self, skill: str, arm: str, *, label: str, max_attempts: int = 1,
                       should_stop: Callable[[], bool] = lambda: False) -> dict[str, Any]:
        """Run one ws4 step to its end: ``{"status": succeeded | failed | aborted, "failure", "run_dir", ...}``.

        Cancelling the caller (a broken WHILE invariant, a timeout) or ``should_stop`` (the liminal e-stop) latches
        the ws4 e-stop, so both arms freeze holding, then aborts the run. The e-stop stays latched: the next step
        is refused until an operator releases it in the ws4 HMI."""
        s = self.require()
        if self._run_lock is None:
            self._run_lock = asyncio.Lock()
        async with self._run_lock:  # the ws4 session runs one plan at a time
            if s.runner.guard.estopped:
                return {"status": "failed", "failure": "the ws4 e-stop is latched: release it in the ws4 HMI first"}
            step_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")[:40] or "step"
            plan = {"steps": [{"id": step_id, "arm": arm, "skill": skill, "max_attempts": int(max_attempts)}]}
            before, previous = s.last_result, s.executor
            proposal = s.load_plan_json(plan, goal=label)
            if not proposal.result.ok:
                return {"status": "failed", "failure": "ws4 refused the step: " + "; ".join(proposal.result.errors)}
            answer = s.approve(proposal.version)
            if answer != "ok":
                return {"status": "failed", "failure": f"ws4 did not start the step: {answer}"}
            self.running = arm
            try:
                while s.last_result is before:
                    thread = getattr(s, "_thread", None)
                    if thread is not None and not thread.is_alive():
                        self.estop("ws4 control thread stopped")
                        return {"status": "failed", "failure": "the ws4 control thread stopped (see its events)"}
                    if should_stop():
                        self._halt(s, previous, f"liminal e-stop during {label}")
                        return {"status": "aborted", "failure": "liminal e-stop"}
                    await asyncio.sleep(self.poll_s)
            except asyncio.CancelledError:
                self._halt(s, previous, f"liminal cancelled {label}")
                raise
            finally:
                self.running = None
            summary = s.last_result or {}
            result = dict(summary.get("result") or {})
            result["run_dir"] = str(s.run_dir) if s.run_dir else ""
            result["trial"] = summary.get("trial")
            return result
