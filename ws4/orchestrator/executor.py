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


def _other(arm: str) -> str:
    return "right" if arm == "left" else "left"


def _arms_of(run: StepRun) -> set[str]:
    """The arms a step occupies: its own, and the other one when the skill also poses it (e.g. tucked)."""
    return {run.step.arm} | ({_other(run.step.arm)} if run.skill.other_arm_pose else set())


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
    homing: bool = False  # positioning before the policy starts: the other arm to its pose, then this arm to its own
    prechecking: bool = False  # the skill's precheck (e.g. the lid is still open) runs before anything moves
    precheck_started: bool = False
    moves: list = field(default_factory=list)  # (arm, pose name) ramps still to do before the policy
    move: tuple | None = None  # the (arm, pose name) ramp in progress
    ramp_arm: str | None = None  # the arm whose ramp mode is on
    ramp_on: bool = False  # a scripted ramp (homing, parking or the home step) runs: ramp_mode(True) is in effect
    parking: bool = False  # park_after: the check passed; releasing, ramping home, then checking once more
    park_checked: bool = False
    t_home: float = 0.0
    policy_started: bool = False
    attempt_interventions: int = 0
    returning: bool = False  # finish: home: the check passed, the policy brings the arm back home
    t_return: float = 0.0
    return_checked: bool = False
    ramp_confirmed: bool = False  # the operator resumed after being asked to confirm a ramp from an unknown pose

    def to_json(self, now: float) -> dict[str, Any]:
        s = self.step
        elapsed = (self.t_end if self.t_end is not None else now) - self.t_start if self.attempt else 0.0
        return {
            "id": s.id, "arm": s.arm, "skill": s.skill, "kind": self.skill.kind, "prompt": self.skill.prompt,
            "depends_on": s.depends_on, "hold_until": s.hold_until, "success_check": s.success_check,
            "max_attempts": s.max_attempts, "status": self.status, "attempt": self.attempt,
            "elapsed_s": round(max(0.0, elapsed), 1), "timeout_s": self.skill.timeout_s, "chunks": self.chunks,
            "reason": self.reason, "phase": self.phase, "plan_version": self.plan_version,
            "verifying": self.verifying is not None, "instruction": self.skill.instruction,
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
        far_deg: float = 15.0,
        home_tol_deg: float = 10.0,
        return_timeout_s: float = 20.0,
        max_run_s: float | None = None,
        home_target: Callable[[str], str | None] | None = None,
        monitors: list[Callable[[Any, dict], str | None]] | None = None,
        on_attempt: Callable[[StepRun, bool, str], None] | None = None,
        on_estop: Callable[[str], None] | None = None,
        before_attempt: Callable[[StepRun], str | None] | None = None,
        ramp_mode: Callable[[str, bool, Skill], None] | None = None,
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
        # A ramp from where a policy left an arm (it may be inside the box) farther than far_deg from its target needs
        # the operator's go (e-stop, then Resume); ramps between known poses do not.
        self.far_deg, self.home_tol_deg, self.return_timeout_s = far_deg, home_tol_deg, return_timeout_s
        self.known_pose = dict.fromkeys(ARMS, True)  # the start poses are set up by the operator
        self._state12: np.ndarray | None = None
        self.max_run_s = max_run_s
        self.home_target = home_target  # arm -> primitive target that can run "home" (start_pose: home skills)
        self.monitors = list(monitors or [])  # per-tick checks; a returned reason latches the e-stop
        self.on_attempt = on_attempt  # every FLUX attempt that started: (run, verified, reason)
        self.on_estop = on_estop
        # e.g. refresh the parked arm's pose for the keep-out; a returned reason blocks the attempt (e-stop latched)
        self.before_attempt = before_attempt
        # (arm, True, skill) while that arm ramps (ramp limits, station-only keep-out); (arm, False, skill) when the
        # skill's policy runs or the arm is idle again (policy limits, the skill's keep-out)
        self.ramp_mode = ramp_mode
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

    def confirm(self, step_id: str) -> None:
        """The operator finished an operator step (e.g. reset the pen by hand): planned, not an intervention."""
        self._push("confirm", step_id)

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
            if OBS_STATE in obs:
                self._state12 = np.asarray(obs[OBS_STATE], dtype=float)
            if active and self.state == "running":
                self._update(report, obs)
                for monitor in self.monitors:
                    reason = monitor(report, obs)
                    if reason:
                        self.events.add("error", f"safety monitor: {reason}")
                        self.estop(reason)
                        break
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
                    if self.on_estop is not None:
                        self.on_estop(str(arg))
                    for r in self.runs.values():
                        if r.status == "running":
                            r.attempt_interventions += 1
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
            elif cmd == "confirm" and self.state == "running":
                run = self.runs.get(arg)
                if run is not None and run.status == "running" and run.skill.kind == "operator" and not run.homing:
                    self._complete(run, "operator confirmed")
            elif cmd == "replan" and self.state == "awaiting_replan":
                self._replace_plan(arg)
                self.state = "running"

    def _override(self, step_id: str, done: bool) -> None:
        run = self.runs.get(step_id)
        if run is None or self.state in TERMINAL:
            return
        self.interventions += 1
        run.attempt_interventions += 1
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
        busy = set().union(*(_arms_of(r) for r in self.runs.values() if r.status == "running")) | set(self.parking)
        for run in list(self.runs.values()):
            if self.state != "running":
                return
            if run.status != "pending" or _arms_of(run) & busy:
                continue
            if all(d in self.done_ids or (d in self.runs and self.runs[d].status == "done")
                   for d in run.step.depends_on):
                self._start(run)
                if run.status == "running":
                    busy |= _arms_of(run)

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
        step, skill = run.step, run.skill
        run.attempt += 1
        run.status, run.t_start, run.t_end = "running", self.t, None
        run.chunks, run.verify_at, run.final_check = 0, skill.verify_every_chunks, False
        run.reason, run.phase, run.homing, run.policy_started = "", "", False, False
        run.parking = run.park_checked = False
        run.returning = run.return_checked = False
        run.prechecking = run.precheck_started = False
        run.attempt_interventions = 0
        if skill.name == "hold":
            run.target = None
            self.events.add("info", f"{step.id}: {step.arm} arm holds until {', '.join(step.hold_until)} done")
            return
        if skill.is_flux and self.before_attempt is not None:
            block = self.before_attempt(run)
            if block:
                run.target, run.phase = None, "blocked"
                self.events.add("error", f"{step.id}: not started: {block}")
                self.estop(block)  # the arm holds where it is; Resume retries this attempt
                return
        if skill.precheck:
            if getattr(self.verifier, "name", "") == "manual":
                self.events.add("warn", f"{step.id}: precheck {skill.precheck} skipped (manual verifier)")
            else:
                run.prechecking, run.phase = True, f"checking that {skill.precheck} still holds"
                return  # _update runs the check on the next frame, before anything moves
        self._begin_moves(run)

    def _begin_moves(self, run: StepRun) -> None:
        step, skill = run.step, run.skill
        # Every FLUX step starts where its demonstrations started: the other arm in its pose (e.g. tucked), then this
        # arm in its start pose, one arm moving at a time, each through the ramp guard.
        moves = []
        if self.home_target is not None and skill.kind in ("flux", "operator"):
            other = _other(step.arm)
            if skill.other_arm_pose and self.home_target(other) is not None:
                moves.append((other, skill.other_arm_pose))
            if skill.start_pose and self.home_target(step.arm) is not None:
                moves.append((step.arm, skill.start_pose))
        if moves:
            run.moves, run.homing = moves, True
            self._next_move(run)
            return
        self._start_policy(run)

    def _needs_confirm(self, run: StepRun, arm: str, pose: str) -> bool:
        """True (and e-stop) when the arm would ramp far from where a policy left it: the operator checks the path
        (e.g. that the arm is not inside the box) and resumes, or moves the arm by hand first."""
        if run.ramp_confirmed:
            run.ramp_confirmed = False
            return False
        target = self.home_target(arm) if self.home_target else None
        if (not self.vocab.station.confirm_unknown_ramps or self.known_pose.get(arm, True) or self._state12 is None
                or target is None):
            return False
        cur = self._state12[arm_slice(arm)]
        far = float(np.abs(cur[:5] - self.backends[target].goal(pose, cur)[:5]).max())
        if far <= self.far_deg:
            return False
        run.ramp_confirmed, run.phase = True, "waiting for the operator to confirm a ramp"
        self.estop(f"{run.step.id}: the {arm} arm would ramp {far:.0f} deg to its {pose} pose from where a policy left "
                   "it (possibly inside the box); check the path or move the arm by hand, then Release and Resume")
        return True

    def _next_move(self, run: StepRun) -> None:
        if self._needs_confirm(run, *run.moves[0]):
            return
        arm, pose = run.moves.pop(0)
        target = self.home_target(arm)
        run.move, run.target, run.t_home = (arm, pose), target, self.t
        run.phase = f"moving the {arm} arm to its {pose} pose"
        self._ramp(run, True, arm)
        self.runner.start(target, pose)
        self.events.add("info", f"{run.step.id}: {run.phase} before attempt {run.attempt}/{run.step.max_attempts}")

    def _start_policy(self, run: StepRun) -> None:
        from .sim_backend import SimSkillError

        step, skill = run.step, run.skill
        run.t_start, run.chunks, run.verify_at, run.phase = self.t, 0, skill.verify_every_chunks, ""
        if skill.kind == "operator":  # the arm holds where it is (home) until the operator presses Done
            run.target, run.phase = None, "waiting for the operator"
            self.events.add("warn", f"{step.id}: OPERATOR: {skill.instruction} Then press Done.", step=step.id)
            return
        target = self.resolve_target(step, skill)
        backend = self.backends.get(target)
        if skill.name in ("home", "tuck"):  # scripted ramps too
            if self._needs_confirm(run, step.arm, "tucked" if skill.name == "tuck" else "home"):
                return
            self._ramp(run, True)
        elif skill.is_flux and self.ramp_mode is not None:
            self.ramp_mode(step.arm, False, skill)  # this skill's keep-out (envelope) and the policy limits
        if skill.is_flux:
            self.known_pose[step.arm] = False  # from here on the policy decides where the arm is
        try:
            if hasattr(backend, "begin"):
                backend.begin(target, step.arm, skill)
            self.runner.start(target, skill.prompt if skill.is_flux else skill.name)
        except (SimSkillError, RuntimeError) as exc:
            run.target = None
            self._fail_attempt(run, str(exc), retry=False)
            return
        run.target = target
        run.policy_started = skill.is_flux
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
            backend = self.backends.get(run.target)
            state6 = state12[arm_slice(step.arm)]
            if run.prechecking:
                self._update_precheck(run, obs)
                continue
            if run.returning:
                self._update_return(run, obs, state6)
                continue
            if run.parking:
                self._update_parking(run, backend, report, obs, state6)
                continue
            if run.homing:
                if run.move is None:  # waiting for the operator to confirm the ramp
                    continue
                arm, pose = run.move
                if backend.reached(pose, report.command[arm_slice(arm)], state12[arm_slice(arm)]):
                    self.runner.stop(run.target)
                    self._ramp(run, False)
                    self.known_pose[arm] = True
                    self.events.add("info", f"{step.id}: {arm} arm at its {pose} pose")
                    if run.moves:
                        self._next_move(run)
                    else:
                        run.homing, run.move = False, None
                        self._start_policy(run)
                else:
                    limit = self.vocab["home"].timeout_s if "home" in self.vocab else 30.0
                    if self.t - run.t_home > limit:
                        self._fail_attempt(run, f"the {arm} arm did not reach its {pose} pose within {limit:.0f} s",
                                           retry=False)
                continue
            if skill.kind == "operator":
                if self.t - run.t_start > skill.timeout_s:
                    self._fail_attempt(run, f"the operator did not confirm within {skill.timeout_s:.0f} s",
                                       retry=False)
                continue
            if report.status.get(run.target) == "replanned":
                run.chunks += 1
            if hasattr(backend, "progress"):
                run.phase = backend.progress(run.target)[2]
            if run.verifying is not None and run.verifying.done():
                fut, run.verifying = run.verifying, None
                self._on_verdict(run, fut.result())
                if run.status != "running":
                    continue
            if skill.kind == "primitive":
                if backend.reached(skill.name, report.command[arm_slice(step.arm)], state6):
                    if skill.name in ("home", "tuck"):
                        self.known_pose[step.arm] = True
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
        if ok and run.skill.finish == "home" and not run.returning:
            run.returning, run.t_return, run.return_checked = True, self.t, False
            run.phase, run.final_check = "returning home (policy)", False
            self.events.add("info", f"{run.step.id}: check passed ({verdict.observation}); the policy brings the "
                            f"{run.step.arm} arm back home, then the check runs again")
        elif ok and run.skill.park_after and not run.parking:
            self._park(run, verdict.observation)
        elif ok:
            self._complete(run, verdict.observation)
        elif run.returning:
            self._fail_attempt(run, f"the check failed after the arm returned home: {verdict.observation}")
        elif run.parking:
            self._fail_attempt(run, f"the check failed after parking: {verdict.observation}")
        elif run.final_check:
            self._fail_attempt(run, f"motion finished but the check failed: {verdict.observation}")

    def _update_precheck(self, run: StepRun, obs: dict) -> None:
        """A skill's precheck (e.g. "the lid of the box is open" before a pen skill, while the box arm holds it): pass
        -> the step goes on; fail -> e-stop: every arm holds, nothing else moves; Resume checks again, Abort ends."""
        pre = self.vocab[run.skill.precheck]
        if not run.precheck_started:
            run.precheck_started = True
            check = Step(f"{run.step.id}:precheck", pre.arms[0], pre.name, success_check=pre.success_check)
            if getattr(self.verifier, "fast", False):
                run.verifying = Future()
                run.verifying.set_result(self.verifier.check(check, pre, obs))
            else:
                snapshot = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in obs.items()}
                run.verifying = self._pool.submit(self.verifier.check, check, pre, snapshot)
            return
        if run.verifying is None or not run.verifying.done():
            return
        verdict, run.verifying = run.verifying.result(), None
        run.verdicts.append(verdict)
        ok = verdict.done and verdict.confidence >= pre.confidence
        self.events.add("info" if ok else "error", f"{run.step.id}: precheck {pre.name} "
                        f"{'holds' if ok else 'FAILED'} ({verdict.source}, conf {verdict.confidence:.2f}): "
                        f"{verdict.observation}", step=run.step.id, verdict=verdict.as_dict())
        if ok:
            run.prechecking = False
            self._begin_moves(run)
        else:
            run.phase = "blocked"
            self.estop(f"{pre.name} no longer holds before {run.step.id} ({verdict.observation}); all arms hold, "
                       "Resume checks again, Abort ends the run")

    def _park(self, run: StepRun, why: str) -> None:
        """park_after: the check passed. Stop the policy, open the gripper and ramp home (release first), then the same
        check once more: the step is done only if the result survives the arm leaving (e.g. the lid stays open)."""
        if run.target is not None and run.target in self.runner.active_targets():
            self.runner.stop(run.target)
        backend = self.backends.get(run.target)
        if hasattr(backend, "end"):
            backend.end(run.target)
        run.parking, run.park_checked, run.t_home, run.phase = True, False, self.t, "parking"
        self.events.add("info", f"{run.step.id}: check passed ({why}); parking the {run.step.arm} arm, then checking "
                        "again")
        home = self.home_target(run.step.arm) if self.home_target else None
        run.target = home
        if home is not None:
            self._ramp(run, True)
            self.runner.start(home, "home")

    def _update_return(self, run: StepRun, obs: dict, state6: np.ndarray) -> None:
        """finish: home. The demos end at home, so the policy keeps running until the arm is within home_tol_deg of
        its home pose (a scripted ramp from inside the box could hit it); then the check must pass again. If the policy
        does not get there in return_timeout_s the step still counts (its check passed) and the arm holds where it is;
        the next ramp of that arm then asks the operator first."""
        if run.verifying is not None:
            if run.verifying.done():
                fut, run.verifying = run.verifying, None
                self._on_verdict(run, fut.result())
            return
        if run.return_checked:
            return
        arm = run.step.arm
        target = self.home_target(arm) if self.home_target else None
        near = target is None or float(np.abs(
            state6[:5] - self.backends[target].goal("home", state6)[:5]).max()) <= self.home_tol_deg
        if near or self.t - run.t_return > self.return_timeout_s:
            if run.target is not None and run.target in self.runner.active_targets():
                self.runner.stop(run.target)
        if near:
            self.known_pose[arm] = True
            run.return_checked, run.final_check = True, True
            self.events.add("info", f"{run.step.id}: {arm} arm back home; checking again")
            self._verify(run, obs)
        elif self.t - run.t_return > self.return_timeout_s:
            self.events.add("warn", f"{run.step.id}: the policy did not bring the {arm} arm home within "
                            f"{self.return_timeout_s:.0f} s; it holds where it is (its next ramp asks the operator)")
            self._complete(run, run.verdicts[-1].observation if run.verdicts else "check passed")

    def _update_parking(self, run: StepRun, backend: Any, report: Any, obs: dict, state6: np.ndarray) -> None:
        if run.verifying is not None:
            if run.verifying.done():
                fut, run.verifying = run.verifying, None
                self._on_verdict(run, fut.result())
            return
        if run.park_checked:
            return
        arrived = run.target is None or backend.reached("home", report.command[arm_slice(run.step.arm)], state6)
        if arrived:
            if run.target is not None:
                self.runner.stop(run.target)
                self._ramp(run, False)
                run.target = None
            self.known_pose[run.step.arm] = True
            run.park_checked, run.final_check = True, True
            self.events.add("info", f"{run.step.id}: {run.step.arm} arm parked; checking again")
            self._verify(run, obs)
            return
        limit = self.vocab["home"].timeout_s if "home" in self.vocab else 30.0
        if self.t - run.t_home > limit:
            self._fail_attempt(run, f"did not park within {limit:.0f} s", retry=False)

    def _report_attempt(self, run: StepRun, ok: bool, reason: str) -> None:
        if self.on_attempt is not None and run.skill.is_flux and run.policy_started:
            run.policy_started = False
            try:
                self.on_attempt(run, ok, reason)
            except Exception as exc:  # logging must never stop the arm
                self.events.add("warn", f"attempt log failed: {type(exc).__name__}: {exc}")

    def _complete(self, run: StepRun, why: str) -> None:
        self._report_attempt(run, True, why)
        self._release(run)
        run.status, run.t_end, run.reason = "done", self.t, why
        self.done_ids.add(run.step.id)
        self.events.add("info", f"{run.step.id}: DONE in {run.t_end - run.t_start:.1f} s ({why})")
        if all(r.status == "done" for r in self.runs.values()):
            self._finish("succeeded", "")

    def _ramp(self, run: StepRun, on: bool, arm: str | None = None) -> None:
        arm = arm or run.ramp_arm or run.step.arm
        if self.ramp_mode is not None and run.ramp_on != on:
            self.ramp_mode(arm, on, run.skill)
        run.ramp_on, run.ramp_arm = on, (arm if on else None)

    def _release(self, run: StepRun) -> None:
        self._ramp(run, False)
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
        self._report_attempt(run, False, reason)
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
                self._report_attempt(run, False, f"interrupted ({self.failure or 'e-stop, abort or replan'})")
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
