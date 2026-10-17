"""DAG executor: runs an approved plan on both arms through the (gated) ``DualArmRunner``.

Every control tick: apply operator commands (e-stop, release, resume, abort, overrides, approved replans), start
every pending step whose dependencies are done and whose arm is free, advance the runner one tick, then update the
running steps:

- FLUX skill: verified every ``verify_every_chunks`` chunks (and right away when a scripted stand-in has finished its
  motion); done on a verdict with enough confidence; failed on timeout. Slow verifiers run in a worker thread while
  the arm keeps going.
- primitive: done when the commanded pose is reached.
- hold: no target, the arm keeps its last command; done when every ``hold_until`` step is done.

A failed attempt is retried up to ``max_attempts``; a failed step pauses everything and asks the planner to replan
from the current state (a human approves the new plan), or fails the run when no replans are left.

Time is the active control time: it does not advance while e-stopped, paused or waiting for a replan, so timeouts
only count time the arm was allowed to move. ``clock="ticks"`` (sim) counts 1/fps per tick; ``"wall"`` measures it.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from flux.contract import ARMS, OBS_STATE, arm_slice

from .events import EventLog
from .plan import Plan, Step, concurrent_pairs, levels
from .skills import Skill, Vocabulary
from .verifier import Verdict

TERMINAL = ("succeeded", "failed", "aborted")
MOVING = ("executing", "replanned")


@dataclass
class StepRun:
    step: Step
    skill: Skill
    plan_version: int = 1
    status: str = "pending"  # pending | running | done | failed | interrupted
    attempt: int = 0
    target: str | None = None
    t_start: float = 0.0
    t_end: float | None = None
    chunks: int = 0
    verify_at: int = 0
    verifying: Future | None = None
    final_check: bool = False
    verdicts: list[Verdict] = field(default_factory=list)
    reason: str = ""
    phase: str = ""

    def to_json(self, now: float) -> dict[str, Any]:
        s = self.step
        elapsed = (self.t_end if self.t_end is not None else now) - self.t_start if self.attempt else 0.0
        return {
            "id": s.id, "arm": s.arm, "skill": s.skill, "kind": self.skill.kind, "prompt": self.skill.prompt,
            "depends_on": s.depends_on, "hold_until": s.hold_until, "success_check": s.success_check,
            "max_attempts": s.max_attempts, "status": self.status, "attempt": self.attempt,
            "elapsed_s": round(max(0.0, elapsed), 1), "timeout_s": self.skill.timeout_s, "chunks": self.chunks,
            "reason": self.reason, "phase": self.phase, "plan_version": self.plan_version,
            "verifying": self.verifying is not None,
            "last_verdict": self.verdicts[-1].as_dict() if self.verdicts else None,
        }


@dataclass
class RunResult:
    status: str
    steps_done: int
    steps_total: int
    interventions: int
    failure: str
    duration_s: float
    ticks: int
    replans: int
    concurrency: dict[str, Any]


class Executor:
    def __init__(
        self,
        *,
        runner: Any,
        robot: Any,
        vocab: Vocabulary,
        verifier: Any,
        resolve_target: Callable[[Step, Skill], str],
        backends: dict[str, Any],
        events: EventLog,
        fps: float = 30.0,
        realtime: bool = False,
        clock: str = "ticks",
        max_replans: int = 1,
        on_replan: Callable[[dict], None] | None = None,
        on_frame: Callable[[dict, Any], None] | None = None,
        frame_period_s: float = 0.2,
        max_run_s: float | None = None,
    ):
        self.runner = runner
        self.robot = robot
        self.vocab = vocab
        self.verifier = verifier
        self.resolve_target = resolve_target
        self.backends = backends
        self.events = events
        self.fps = fps
        self.realtime = realtime
        self.clock = clock
        self.max_replans = max_replans
        self.on_replan = on_replan
        self.on_frame = on_frame
        self.frame_period_s = frame_period_s
        self.max_run_s = max_run_s
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="verify")
        self._lock = threading.RLock()
        self._commands: list[tuple[str, Any]] = []
        self.runs: dict[str, StepRun] = {}
        self.done_ids: set[str] = set()
        self.plan: Plan | None = None
        self.state = "idle"
        self.failure = ""
        self.interventions = 0
        self.replans = 0
        self.tick = 0
        self.t = 0.0
        self._last_frame = 0.0
        self._state_before_estop = "running"
        self.parking: dict[str, str] = {}  # arm -> primitive target moving it home to make way
        self._blocked: set[tuple[str, str]] = set()
        self.arm_ticks = {arm: {"moving": 0, "waiting": 0, "idle": 0} for arm in ARMS}
        self.both_moving = 0
        self.active_ticks = 0
        self.latencies: dict[str, list[float]] = {}

    # ------------------------------------------------------------------ operator commands (any thread)
    def _push(self, cmd: str, arg: Any = None) -> None:
        with self._lock:
            self._commands.append((cmd, arg))

    def estop(self, reason: str = "operator") -> None:
        self.runner.guard.estop()  # immediate: the next runner tick sends nothing
        self._push("estop", reason)

    def release(self) -> None:
        self._push("release")

    def resume(self) -> None:
        self._push("resume")

    def abort(self, reason: str = "aborted by operator") -> None:
        self._push("abort", reason)

    def override(self, step_id: str, done: bool) -> None:
        self._push("override", (step_id, done))

    def approve_replan(self, plan: Plan) -> None:
        self._push("replan", plan)

    # ------------------------------------------------------------------ plan bookkeeping
    def load(self, plan: Plan) -> None:
        with self._lock:
            for s in plan.steps:
                self.runs[s.id] = StepRun(s, self.vocab[s.skill], plan.version)
            self.plan = plan

    def _replace_plan(self, plan: Plan) -> None:
        with self._lock:
            kept = {sid: r for sid, r in self.runs.items() if r.status == "done"}
            dropped = [sid for sid in self.runs if sid not in kept]
            self.runs = kept
            self.load(plan)
        self.replans += 1
        self.events.add("info", f"replan v{plan.version} approved: {len(plan.steps)} new steps"
                        + (f", dropped {', '.join(dropped)}" if dropped else ""))

    def progress_state(self, failed: StepRun | None = None) -> dict[str, Any]:
        state: dict[str, Any] = {
            "done": [{"id": r.step.id, "arm": r.step.arm, "skill": r.step.skill,
                      "success_check": r.step.success_check} for r in self.runs.values() if r.status == "done"],
            "not_done": [{"id": r.step.id, "arm": r.step.arm, "skill": r.step.skill, "status": r.status}
                         for r in self.runs.values() if r.status not in ("done",) and r is not failed],
        }
        if failed is not None:
            state["failed"] = {"id": failed.step.id, "arm": failed.step.arm, "skill": failed.step.skill,
                               "attempts": failed.attempt, "reason": failed.reason,
                               "last_verifier_observation": failed.verdicts[-1].observation if failed.verdicts
                               else ""}
        return state

    # ------------------------------------------------------------------ main loop
    def run(self, plan: Plan | None = None) -> RunResult:
        if plan is not None:
            self.load(plan)
        if self.plan is None:
            raise RuntimeError("no plan loaded")
        self.state = "running"
        if hasattr(self.runner, "on_blocked"):
            self.runner.on_blocked = lambda arm, other: self._blocked.add((arm, other))
        self.events.add("info", f"run started: {len(self.runs)} steps, concurrent pairs "
                        f"{concurrent_pairs(self.plan) or 'none'}")
        period = 1.0 / self.fps
        next_tick = time.monotonic()
        last_wall = time.monotonic()
        while self.state not in TERMINAL:
            self._process_commands()
            if self.state == "running":
                self._schedule()
                self._check_stall()
            try:
                report = self.runner.step()
            except Exception as exc:  # backend or transport failure: hold everything, let the operator decide
                self.events.add("error", f"control tick failed: {type(exc).__name__}: {exc}")
                self.estop(f"control error: {exc}")
                report = None
            self.tick += 1
            now_wall = time.monotonic()
            active = self.state == "running" and not self.runner.guard.estopped and report is not None
            if active:
                self.t += period if self.clock == "ticks" else now_wall - last_wall
                self._account(report)
            last_wall = now_wall
            obs = self.robot.get_observation()
            if active and self.state == "running":
                self._update(report, obs)
            if self.on_frame is not None and (now_wall - self._last_frame >= self.frame_period_s
                                              or self.state in TERMINAL):
                self._last_frame = now_wall
                self.on_frame(obs, report)
            if self.max_run_s is not None and self.t > self.max_run_s and self.state == "running":
                self._finish("failed", f"run exceeded {self.max_run_s:.0f} s")
            if self.realtime:
                next_tick += period
                delay = next_tick - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_tick = time.monotonic()
            elif self.tick % 10 == 0:
                time.sleep(0)  # let the HMI threads in
        self._stop_all("run ended")
        return self.result()

    def result(self) -> RunResult:
        countable = [r for r in self.runs.values() if r.skill.name != "hold"]
        done = sum(r.status == "done" for r in countable)
        return RunResult(self.state, done, len(countable), self.interventions, self.failure, round(self.t, 2),
                         self.tick, self.replans, self.concurrency())

    def concurrency(self) -> dict[str, Any]:
        out: dict[str, Any] = {"active_ticks": self.active_ticks, "both_arms_moving_ticks": self.both_moving,
                               "per_arm": {a: dict(v) for a, v in self.arm_ticks.items()}}
        for target, lat in self.latencies.items():
            if lat:
                out.setdefault("latency_s", {})[target] = {"n": len(lat), "median": float(np.median(lat)),
                                                           "max": float(np.max(lat))}
        return out

    # ------------------------------------------------------------------ per tick
    def _account(self, report: Any) -> None:
        self.active_ticks += 1
        moving = 0
        for arm in ARMS:
            target = self.runner._owner.get(arm)  # ws4/flux/dual_arm_runner.py:230, arm -> owning target
            status = report.status.get(target) if target else None
            key = "moving" if status in MOVING else "waiting" if status == "waiting" else "idle"
            self.arm_ticks[arm][key] += 1
            moving += key == "moving"
        self.both_moving += moving == 2
        for target, lat in report.latencies.items():
            if report.status.get(target) == "replanned":
                self.latencies.setdefault(target, []).append(float(lat))

    def _process_commands(self) -> None:
        with self._lock:
            commands, self._commands = self._commands, []
        for cmd, arg in commands:
            if cmd == "estop":
                if self.state in TERMINAL:
                    continue
                if self.state != "estopped":
                    self.interventions += 1
                    self._state_before_estop = "awaiting_replan" if self.state == "awaiting_replan" else "running"
                    self._interrupt_all()
                    self.events.add("error", f"E-STOP ({arg}): both arms hold")
                    self.state = "estopped"
            elif cmd == "release" and self.state == "estopped":
                self.runner.release_estop()
                self.state = "paused"
                self.events.add("warn", "e-stop released; press Resume to continue or Abort")
            elif cmd == "resume" and self.state == "paused":
                for r in self.runs.values():
                    if r.status == "interrupted":
                        r.status = "pending"
                self.state = self._state_before_estop
                self.events.add("info", "resumed: interrupted steps restart"
                                + (" once the new plan is approved" if self.state == "awaiting_replan" else ""))
            elif cmd == "abort" and self.state not in TERMINAL:
                self.interventions += 1
                self._interrupt_all()
                self._finish("aborted", str(arg))
            elif cmd == "override":
                self._override(*arg)
            elif cmd == "replan" and self.state == "awaiting_replan":
                self._replace_plan(arg)
                self.state = "running"

    def _override(self, step_id: str, done: bool) -> None:
        run = self.runs.get(step_id)
        if run is None or self.state in TERMINAL:
            return
        self.interventions += 1
        self.events.add("warn", f"operator marked {step_id} {'done' if done else 'failed'}")
        if done:
            if run.status != "done":
                if run.attempt == 0:
                    run.attempt, run.t_start = 1, self.t
                self._complete(run, "operator override")
        elif run.status in ("pending", "running", "interrupted"):
            run.attempt = run.step.max_attempts
            self._fail_attempt(run, "operator marked it failed")

    def _schedule(self) -> None:
        busy = {r.step.arm for r in self.runs.values() if r.status == "running"} | set(self.parking)
        for run in list(self.runs.values()):
            if self.state != "running":
                return
            if run.status != "pending" or run.step.arm in busy:
                continue
            if all(d in self.done_ids or (d in self.runs and self.runs[d].status == "done")
                   for d in run.step.depends_on):
                self._start(run)
                if run.status == "running":
                    busy.add(run.step.arm)

    def _check_stall(self) -> None:
        """Nothing running and nothing startable: the run can never finish (a dependency failed for good)."""
        if self.state != "running" or self.parking or any(r.status == "running" for r in self.runs.values()):
            return
        if all(r.status == "done" for r in self.runs.values()):
            return self._finish("succeeded", "")
        failed = [r.step.id for r in self.runs.values() if r.status == "failed"]
        blocked = [r.step.id for r in self.runs.values() if r.status == "pending"]
        self._finish("failed", f"stalled: {', '.join(blocked) or 'no steps'} cannot start"
                     + (f" because {', '.join(failed)} failed" if failed else ""))

    def _start(self, run: StepRun) -> None:
        from .sim_backend import SimSkillError

        step, skill = run.step, run.skill
        run.attempt += 1
        run.status, run.t_start, run.t_end = "running", self.t, None
        run.chunks, run.verify_at, run.final_check = 0, skill.verify_every_chunks, False
        run.reason, run.phase = "", ""
        if skill.name == "hold":
            run.target = None
            self.events.add("info", f"{step.id}: {step.arm} arm holds until {', '.join(step.hold_until)} done")
            return
        target = self.resolve_target(step, skill)
        backend = self.backends.get(target)
        try:
            if hasattr(backend, "begin"):
                backend.begin(target, step.arm, skill)
            self.runner.start(target, skill.prompt if skill.is_flux else skill.name)
        except (SimSkillError, RuntimeError) as exc:
            run.target = None
            self._fail_attempt(run, str(exc), retry=False)
            return
        run.target = target
        what = f"FLUX prompt {skill.prompt!r}" if skill.is_flux else "scripted primitive"
        self.events.add("info", f"{step.id}: start on {step.arm} arm, attempt {run.attempt}/{step.max_attempts}, "
                        f"{what} via {target}")

    def _make_way(self, report: Any, state12: np.ndarray) -> None:
        """An idle arm (no step running, not holding anything) that blocks the other arm is parked at home."""
        blocked, self._blocked = self._blocked, set()
        busy = {r.step.arm for r in self.runs.values() if r.status == "running"}
        for arm, other in blocked:
            if other in busy or other in self.parking:
                continue
            target = f"{other}:prim"
            if target not in self.backends:
                continue
            self.runner.start(target, "home")
            self.parking[other] = target
            self.events.add("info", f"make way: the idle {other} arm blocks the {arm} arm; parking it at home")
        for arm, target in list(self.parking.items()):
            if self.backends[target].reached("home", report.command[arm_slice(arm)], state12[arm_slice(arm)]):
                self.runner.stop(target)
                del self.parking[arm]
                self.events.add("info", f"{arm} arm parked")

    def _update(self, report: Any, obs: dict) -> None:
        state12 = np.asarray(obs[OBS_STATE], dtype=float)
        self._make_way(report, state12)
        for run in list(self.runs.values()):
            if run.status != "running" or self.state != "running":
                continue
            step, skill = run.step, run.skill
            if skill.name == "hold":
                status = [self.runs[u].status if u in self.runs else "done" for u in step.hold_until]
                if all(s == "done" for s in status):
                    self._complete(run, "steps it waited for are done")
                elif "failed" in status:
                    self._complete(run, "released: a step it waited for failed")
                continue
            if report.status.get(run.target) == "replanned":
                run.chunks += 1
            backend = self.backends.get(run.target)
            if hasattr(backend, "progress"):
                run.phase = backend.progress(run.target)[2]
            if run.verifying is not None and run.verifying.done():
                fut, run.verifying = run.verifying, None
                self._on_verdict(run, fut.result())
                if run.status != "running":
                    continue
            state6 = state12[arm_slice(step.arm)]
            if skill.kind == "primitive":
                if backend.reached(skill.name, report.command[arm_slice(step.arm)], state6):
                    self._complete(run, "pose reached")
                    continue
            elif run.verifying is None:
                scripted_done = hasattr(backend, "finished") and backend.finished(run.target, state6)
                if scripted_done or run.chunks >= run.verify_at:
                    run.verify_at = run.chunks + skill.verify_every_chunks
                    run.final_check = bool(scripted_done)
                    self._verify(run, obs)
                    if run.status != "running":
                        continue
            if self.t - run.t_start > skill.timeout_s:
                self._fail_attempt(run, f"timeout after {skill.timeout_s:.0f} s without a verified success")

    def _verify(self, run: StepRun, obs: dict) -> None:
        if getattr(self.verifier, "fast", False):
            self._on_verdict(run, self.verifier.check(run.step, run.skill, obs))
        else:
            snapshot = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in obs.items()}
            run.verifying = self._pool.submit(self.verifier.check, run.step, run.skill, snapshot)

    def _on_verdict(self, run: StepRun, verdict: Verdict) -> None:
        run.verdicts.append(verdict)
        ok = verdict.done and verdict.confidence >= run.skill.confidence
        level = "info" if ok else "warn" if verdict.error else "debug"
        self.events.add(level, f"{run.step.id}: check {'PASSED' if ok else 'not yet'} "
                        f"({verdict.source}, conf {verdict.confidence:.2f}): {verdict.observation}"
                        + (f" [{verdict.error}]" if verdict.error else ""), step=run.step.id,
                        verdict=verdict.as_dict())
        if ok:
            self._complete(run, verdict.observation)
        elif run.final_check:
            self._fail_attempt(run, f"motion finished but the check failed: {verdict.observation}")

    def _complete(self, run: StepRun, why: str) -> None:
        self._release(run)
        run.status, run.t_end, run.reason = "done", self.t, why
        self.done_ids.add(run.step.id)
        self.events.add("info", f"{run.step.id}: DONE in {run.t_end - run.t_start:.1f} s ({why})")
        if all(r.status == "done" for r in self.runs.values()):
            self._finish("succeeded", "")

    def _release(self, run: StepRun) -> None:
        if run.verifying is not None:
            run.verifying.cancel()
            run.verifying = None
        if run.target is not None:
            if run.target in self.runner.active_targets():
                self.runner.stop(run.target)
            backend = self.backends.get(run.target)
            if hasattr(backend, "end"):
                backend.end(run.target)

    def _fail_attempt(self, run: StepRun, reason: str, retry: bool = True) -> None:
        self._release(run)
        run.reason, run.t_end = reason, self.t
        if retry and run.attempt < run.step.max_attempts:
            run.status = "pending"
            self.events.add("warn", f"{run.step.id}: attempt {run.attempt} failed ({reason}); retrying")
            return
        run.status = "failed"
        self.events.add("error", f"{run.step.id}: FAILED after {run.attempt} attempt(s): {reason}")
        if self.on_replan is not None and self.replans < self.max_replans:
            self._interrupt_all()
            self.state = "awaiting_replan"
            self.events.add("warn", "asking the planner to replan from the current state")
            self.on_replan(self.progress_state(failed=run))
        else:
            self._finish("failed", f"{run.step.id} ({run.step.skill}): {reason}")

    def _interrupt_all(self) -> None:
        for arm, target in list(self.parking.items()):
            if target in self.runner.active_targets():
                self.runner.stop(target)
            del self.parking[arm]
        for run in self.runs.values():
            if run.status == "running":
                self._release(run)
                run.status = "interrupted"
                run.attempt -= 1  # an interrupted attempt does not count
                run.t_end = self.t

    def _stop_all(self, why: str) -> None:
        for target in list(self.runner.active_targets()):
            self.runner.stop(target)

    def _finish(self, state: str, failure: str) -> None:
        self.state, self.failure = state, failure
        self._stop_all(state)
        level = "info" if state == "succeeded" else "error"
        self.events.add(level, f"run {state.upper()}" + (f": {failure}" if failure else "")
                        + f" after {self.t:.1f} s, {self.tick} ticks")

    # ------------------------------------------------------------------ HMI
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            runs = list(self.runs.values())
            plan = self.plan
        lv = levels(Plan([r.step for r in runs])) if runs else {}
        return {
            "state": self.state, "failure": self.failure, "t": round(self.t, 2), "tick": self.tick,
            "estopped": bool(self.runner.guard.estopped), "interventions": self.interventions,
            "replans": self.replans, "plan_version": plan.version if plan else None,
            "steps": [{**r.to_json(self.t), "level": lv.get(r.step.id, 0)} for r in runs],
            "concurrent_pairs": concurrent_pairs(Plan([r.step for r in runs])) if runs else [],
            "arms": {arm: self.runner._owner.get(arm) for arm in ARMS},
            "concurrency": self.concurrency(),
        }
