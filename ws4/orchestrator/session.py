"""Wires one orchestrator session for a backend: robot or sim, policy backends, gate, runner, planner, verifier.

Backends (README.md has the commands):
  (a) env=sim,  policy=scripted, planner=stub    laptop, no GPU, no key
  (b) env=sim,  policy=scripted, planner=claude  same with the real Claude planner (and optionally verifier)
  (c) env=sim or xlerobot, policy=remote         FLUX on the RunPod server through flux.remote.RemoteBackend
  (d) env=xlerobot                               the real robot through flux.run_xlerobot (only when connected)

All robot and MuJoCo access happens on one control thread (the EGL context is bound to it); the HMI talks to the
session from its own threads through queued requests and cached frames.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import io
import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from flux.contract import ARMS, IMG_HEAD, IMG_LEFT_WRIST, IMG_RIGHT_WRIST, SO101_CONTRACT
from flux.dual_arm_runner import per_arm_target
from flux.safety import SafetyConfig, SafetyGuard

from . import PACKAGE_DIR, WS4
from .claude import SDK_VERSION, ClaudeJSON, api_key_available, sdk_version
from .cost import CostMeter
from .events import EventLog
from .executor import TERMINAL, Executor, RunResult
from .gate import CollisionGate, GatedDualArmRunner, TickGuardRobot
from .plan import Plan, Step, concurrent_pairs, fill_defaults, levels, parse_plan, validate_plan
from .planner import ClaudePlanner, PlanResult, StubPlanner, arm_by_side
from .primitives import PrimitiveBackend
from .skills import SIM_VOCABULARY, Skill, Vocabulary, load_vocabulary
from .verifier import ClaudeVerifier, ManualVerifier, ShadowVerifier, SimTruthVerifier

RUNS_DIR = PACKAGE_DIR / "runs"
SIM_TRIAL_LOG = RUNS_DIR / "trials_sim.jsonl"
MOCK_TRIAL_LOG = RUNS_DIR / "trials_mock.jsonl"  # haller with the fake arm or in observe mode: never graded
GRADED_TRIAL_LOG = WS4 / "eval" / "trials.jsonl"
CAMERAS = {"head": IMG_HEAD, "left_wrist": IMG_LEFT_WRIST, "right_wrist": IMG_RIGHT_WRIST}


def load_trial_log():
    """Import ws4/eval/trial_log.py by path (read-only use; stdlib only)."""
    name = "ws4_eval_trial_log"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, WS4 / "eval" / "trial_log.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve the module through sys.modules
    spec.loader.exec_module(module)
    return module


@dataclass
class SessionConfig:
    env: str = "sim"  # sim | haller (Sunday demo: robot-left arm via ws4/single_arm) | xlerobot (both arms)
    policy: str = "scripted"  # scripted (sim only) | fake | remote
    planner: str = "stub"  # stub | claude
    verifier: str = "auto"  # auto | truth | claude | shadow | manual
    planner_model: str = "claude-opus-5-5"
    planner_effort: str = "medium"
    verifier_effort: str = "low"
    verifier_thinking: str = "adaptive"  # Haiku 5.5 only: adaptive | disabled
    verifier_wrist: bool = False
    budget_usd: float = 5.0
    # Policy server (c): ssh -N -L 8765:127.0.0.1:8765 to the pod, see ws4/cloud/RUNPOD.md.
    server: str = "http://127.0.0.1:8765"
    token_file: str | None = None
    package: str = "so101"
    left_package: str | None = None
    right_package: str | None = None
    async_inference: bool | None = None  # default: True for remote
    # Sim
    seed: int | None = 0
    image_size: tuple[int, int] = (240, 320)
    randomize: bool = True
    lid_on_box: bool = False
    # Real robot (d), same flags as flux/run_xlerobot.py
    robot_id: str = "xlerobot"
    left_port: str = ""
    right_port: str = ""
    head_cam: str = ""
    left_wrist_cam: str = ""
    right_wrist_cam: str = ""
    calibration_limits: bool = False
    # Haller (env=haller): run_single_arm.py's robot side, same flags and defaults as its CLI
    robot: str = "so101"  # so101 | fake
    observe: bool = False  # real arm, torque never enabled, nothing sent
    port: str = "/dev/haller_arm_right"
    wrist_cam: str = "/dev/v4l/by-path/pci-0000:00:14.0-usb-0:7.2:1.0-video-index0"
    wrist_rotate: int = 180
    scene_fourcc: str = "none"
    wrist_fourcc: str = "none"
    keepout_config: str = str(WS4 / "single_arm" / "calibration" / "haller_right_workspace.json")
    max_relative_target: float = 20.0
    torque_limit: float = 60.0
    max_step: float = 2.0
    max_gripper_step: float = 5.0
    limit_margin: float = 5.0
    stall_deg: float = 8.0
    stall_ticks: int = 15
    keepout_ticks: int = 45
    home_speed: float = 1.0  # deg per tick when ramping to the start pose (run_single_arm --return-speed)
    allow_torque_off: bool = False
    token_cmd: str = ""  # e.g. "ssh -p PORT root@HOST cat /workspace/flux_token"; output never shown
    # Execution
    gate: bool = True
    realtime: bool = True
    fps: float = 30.0
    max_replans: int = 1
    auto_approve_replans: bool = False
    max_run_s: float | None = 600.0
    # Trial log
    trial_log: str | None = None
    task: str = "pack_box"
    condition: str = "zero_shot"
    variant: str | None = None
    checkpoint: str | None = None
    layout: str | None = None
    operator: str = ""
    runs_dir: str | None = None
    vocabulary: str | None = None
    echo_events: bool = False

    def package_for(self, arm: str) -> str:
        return {"left": self.left_package, "right": self.right_package}[arm] or self.package


@dataclass
class Proposal:
    kind: str  # plan | replan
    version: int
    result: PlanResult
    goal: str
    created: float = field(default_factory=time.time)

    def to_json(self) -> dict[str, Any]:
        plan = self.result.plan
        out: dict[str, Any] = {"kind": self.kind, "version": self.version, "goal": self.goal,
                               "ok": self.result.ok, "errors": self.result.errors, "attempts": self.result.attempts,
                               "source": self.result.source, "latency_s": round(self.result.latency_s, 2),
                               "cost_usd": round(self.result.cost_usd, 5)}
        if plan is not None:
            out["plan"] = plan.to_json()
            out["levels"] = levels(plan)
            out["concurrent_pairs"] = concurrent_pairs(plan)
        return out


class Session:
    """``threaded=True`` (HMI): the env, renderer and robot are created and used only on the control thread, which
    also runs approved plans. ``threaded=False`` (CLI, tests): everything runs on the calling thread."""

    def __init__(self, cfg: SessionConfig | None = None, llm_client: Any = None, threaded: bool = False):
        self.cfg = cfg = cfg or SessionConfig()
        self.vocab: Vocabulary = load_vocabulary(cfg.vocabulary or (SIM_VOCABULARY if cfg.env == "sim" else None))
        self.rig = None  # HallerRig on env=haller
        self.runs_dir = Path(cfg.runs_dir) if cfg.runs_dir else RUNS_DIR
        self.events = EventLog(echo=cfg.echo_events)
        self.meter = CostMeter(cfg.budget_usd)
        self.llm = ClaudeJSON(self.meter, client=llm_client)
        self.sim = None
        self._lock = threading.RLock()
        self._requests: list[tuple[str, Any]] = []
        self._frames: dict[str, bytes] = {}
        self._last_obs: dict | None = None
        self.proposal: Proposal | None = None
        self.goal = ""
        self.executor: Executor | None = None
        self.last_result: dict | None = None
        self.planning = False
        self.run_dir: Path | None = None
        self._plan_version = 0
        self._closing = False
        self._thread: threading.Thread | None = None
        if threaded:
            ready = threading.Event()
            error: list[BaseException] = []

            def main() -> None:
                try:
                    self._build()
                except BaseException as exc:  # surface construction errors to the caller
                    error.append(exc)
                    ready.set()
                    return
                ready.set()
                self._loop()

            self._thread = threading.Thread(target=main, daemon=True, name="control")
            self._thread.start()
            ready.wait()
            if error:
                raise error[0]
        else:
            self._build()

    # ------------------------------------------------------------------ construction
    def _build(self) -> None:
        cfg = self.cfg
        if cfg.env == "sim":
            from .sim_backend import PackBoxSim, SimRobot

            self.sim = PackBoxSim(image_size=tuple(cfg.image_size), randomize=cfg.randomize)
            base_robot: Any = SimRobot(self.sim, cfg.seed, cfg.lid_on_box)
        elif cfg.env == "haller":
            from . import haller

            self.rig = haller.build(cfg)
            base_robot = self.rig.robot
        elif cfg.env == "xlerobot":
            from flux.run_xlerobot import LeRobotBimanualRobot, build_robot

            args = dataclasses.replace(cfg)  # build_robot reads attribute names identical to run_xlerobot's flags
            self._lerobot = build_robot(args)
            base_robot = LeRobotBimanualRobot(self._lerobot)
        else:
            raise ValueError(f"unknown env {cfg.env!r}")
        self.base_robot = base_robot

        self.backends: dict[str, Any] = {}  # runner backend name -> backend
        self.target_backend: dict[str, Any] = {}  # runner target name -> backend object
        targets = []
        arms = self.vocab.station.arms_available if self.rig is not None else ARMS
        home = self.vocab.station.home_pose
        prim = PrimitiveBackend(home=None if home is None else np.array(home),
                                joint_step=cfg.home_speed if self.rig is not None else 2.0)
        for arm in arms:
            name = f"prim_{arm}"
            self.backends[name] = prim
            targets.append(dataclasses.replace(per_arm_target(arm, name), name=f"{arm}:prim"))
            self.target_backend[f"{arm}:prim"] = prim
        if cfg.policy == "scripted":
            if self.sim is None:
                raise ValueError("policy=scripted is the sim stand-in; use policy=remote on the real robot")
            from .sim_backend import ScriptedSimBackend

            for arm in arms:
                be = ScriptedSimBackend(self.sim)
                self.backends[f"sim_{arm}"] = be
                targets.append(dataclasses.replace(per_arm_target(arm, f"sim_{arm}"), name=f"{arm}:sim"))
                self.target_backend[f"{arm}:sim"] = be
        elif cfg.policy in ("fake", "remote"):
            packages = sorted({cfg.package_for(a) for a in arms} | {s.package for s in self.vocab.skills.values()
                                                                    if s.package})
            for pkg in packages:
                self.backends[pkg] = self._policy_backend(pkg)
                for arm in arms:
                    t = f"{arm}:{pkg}"
                    targets.append(dataclasses.replace(per_arm_target(arm, pkg), name=t))
                    self.target_backend[t] = self.backends[pkg]
        else:
            raise ValueError(f"unknown policy {cfg.policy!r}")

        # One moving arm on the Haller: the other is parked and not connected, so an inter-arm check against the
        # phantom slot would be meaningless; the FK keep-out (midline zone included) guards that side instead.
        self.gate = CollisionGate() if cfg.gate and self.rig is None else None
        robot = TickGuardRobot(base_robot, self.gate, self._tick_estop) if self.gate else base_robot
        self.robot = robot
        guard = self.rig.guard if self.rig is not None else SafetyGuard(SafetyConfig())
        if cfg.env == "xlerobot" and cfg.calibration_limits:
            from flux.safety import limits_from_lerobot_calibration
            from lerobot.utils.constants import HF_LEROBOT_CALIBRATION

            for arm in ARMS:
                path = HF_LEROBOT_CALIBRATION / "robots" / "so_follower" / f"{cfg.robot_id}_{arm}.json"
                guard.config = guard.config.with_arm_limits(arm, *limits_from_lerobot_calibration(path))
        async_inf = cfg.async_inference if cfg.async_inference is not None else cfg.policy == "remote"
        self.runner = GatedDualArmRunner(robot, targets, self.backends, guard, async_inference=async_inf,
                                         gate=self.gate, on_event=lambda lvl, msg, data: self.events.add(lvl, msg,
                                                                                                        **data))
        self.verifier = self._make_verifier()
        self.planner = self._make_planner()
        if self.rig is not None and cfg.policy == "remote":
            policy_targets = [t for t in targets if not t.name.endswith(":prim")]
            for t in policy_targets[:1]:
                self.rig.warm_up(self.backends[t.backend], t, self.vocab.flux_prompts())
        self._last_obs = base_robot.get_observation()
        self._encode_frames(self._last_obs)
        self.events.add("info", f"session ready: env={cfg.env} policy={cfg.policy} planner={cfg.planner} "
                        f"verifier={self.verifier.name} gate={'on' if self.gate else 'off'}")
        if cfg.planner == "claude" or self.verifier.name != "sim_truth":
            installed = sdk_version()
            self.events.add("info" if installed == SDK_VERSION else "warn",
                            f"anthropic SDK {installed or 'not installed'} (code checked against {SDK_VERSION}); "
                            f"API key {'found' if api_key_available() else 'NOT found'}")

    def _policy_backend(self, package: str) -> Any:
        cfg = self.cfg
        if cfg.policy == "fake":
            from flux.backends import FakeBackend

            return FakeBackend(SO101_CONTRACT, velocity=0.0, delay_s=0.02)
        from flux.remote import RemoteBackend

        token = None
        if cfg.token_file:
            token = Path(cfg.token_file).expanduser().read_text().strip()
        elif cfg.token_cmd:  # same mechanism as run_single_arm --token-cmd: the token never reaches argv or a log
            import shlex
            import subprocess

            out = subprocess.run(shlex.split(os.path.expanduser(cfg.token_cmd)), capture_output=True, text=True,
                                 stdin=subprocess.DEVNULL, timeout=30)
            if out.returncode != 0 or not out.stdout.strip():
                raise SystemExit(f"--token-cmd failed (exit {out.returncode}): {out.stderr.strip()[:200]}")
            token = out.stdout.strip()
        return RemoteBackend(cfg.server, package, token=token, timeout_s=90.0)

    def _make_verifier(self) -> Any:
        kind = self.cfg.verifier
        if kind == "auto":
            kind = "truth" if self.sim is not None else ("claude" if api_key_available() else "manual")
        if kind == "truth":
            if self.sim is None:
                raise ValueError("verifier=truth needs the sim")
            return SimTruthVerifier(self.sim)
        if kind == "claude":
            return ClaudeVerifier(self.llm, use_wrist=self.cfg.verifier_wrist, effort=self.cfg.verifier_effort,
                                  haiku_thinking=self.cfg.verifier_thinking)
        if kind == "shadow":
            if self.sim is None:
                raise ValueError("verifier=shadow compares Claude with sim ground truth; it needs the sim")

            def report(step, primary, shadow):
                agree = primary.done == (shadow.done and shadow.confidence >= self.vocab[step.skill].confidence)
                self.events.add("info" if agree else "warn",
                                f"shadow verifier {'agrees' if agree else 'DISAGREES'} on {step.id}: truth="
                                f"{primary.done}, {shadow.source}={shadow.done} ({shadow.confidence:.2f}, "
                                f"{shadow.latency_s:.1f} s): {shadow.observation}" + (f" [{shadow.error}]"
                                                                                         if shadow.error else ""))

            return ShadowVerifier(SimTruthVerifier(self.sim),
                                  ClaudeVerifier(self.llm, self.cfg.verifier_wrist, self.cfg.verifier_effort,
                                                 haiku_thinking=self.cfg.verifier_thinking), report)
        if kind == "manual":
            return ManualVerifier()
        raise ValueError(f"unknown verifier {kind!r}")

    def _make_planner(self) -> Any:
        if self.cfg.planner == "claude":
            return ClaudePlanner(self.vocab, self.llm, self.cfg.planner_model, self.cfg.planner_effort)
        hint = arm_by_side(self.sim.poses) if self.sim is not None else None
        return StubPlanner(self.vocab, hint)

    def resolve_target(self, step: Step, skill: Skill) -> str:
        if skill.kind == "primitive":
            return f"{step.arm}:prim"
        if self.cfg.policy == "scripted":
            return f"{step.arm}:sim"
        return f"{step.arm}:{skill.package or self.cfg.package_for(step.arm)}"

    # ------------------------------------------------------------------ frames
    def _encode_frames(self, obs: dict | None) -> None:
        if not obs:
            return
        from PIL import Image

        frames = {}
        for name, key in CAMERAS.items():
            img = obs.get(key)
            if img is None:
                continue
            pil = Image.fromarray(np.asarray(img).astype(np.uint8))
            if pil.size[0] > 480:
                pil = pil.resize((480, round(480 * pil.size[1] / pil.size[0])))
            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=80)
            frames[name] = buf.getvalue()
        with self._lock:
            self._frames.update(frames)
            self._last_obs = obs

    def frame(self, name: str) -> bytes | None:
        with self._lock:
            return self._frames.get(name)

    def head_image(self) -> np.ndarray | None:
        with self._lock:
            obs = self._last_obs
        return None if obs is None else obs.get(IMG_HEAD)

    # ------------------------------------------------------------------ planning
    def plan(self, goal: str, *, kind: str = "plan", state: dict | None = None) -> Proposal:
        goal = goal.strip()
        if not goal:
            raise ValueError("empty goal")
        self.planning = True
        self.events.add("info", f"planning ({self.planner.name}): {goal}" if kind == "plan"
                        else f"replanning ({self.planner.name}) after a failure")
        try:
            # The fake arm's cameras are noise: never show them to Claude as the scene.
            image = None if (self.rig is not None and not self.rig.real) else self.head_image()
            result = self.planner.plan(goal, image=image, state=state)
        except Exception as exc:
            result = PlanResult(None, [f"planner error: {type(exc).__name__}: {exc}"], 1, self.planner.name)
        finally:
            self.planning = False
        with self._lock:
            self._plan_version += 1
            if result.plan is not None:
                result.plan.version = self._plan_version
                result.plan.goal = goal
            self.proposal = Proposal(kind, self._plan_version, result, goal)
            self.goal = goal
        if result.ok:
            pairs = concurrent_pairs(result.plan)
            self.events.add("info", f"plan v{self._plan_version} ready from {result.source} in "
                            f"{result.latency_s:.1f} s ({result.attempts} attempt(s), USD {result.cost_usd:.4f}): "
                            f"{len(result.plan.steps)} steps, concurrent pairs {pairs or 'none'}. "
                            "Waiting for approval.")
        else:
            self.events.add("error", f"planning failed after {result.attempts} attempt(s): "
                            + "; ".join(result.errors))
        return self.proposal

    def load_plan_json(self, data: dict | str, goal: str = "") -> Proposal:
        """A plan from elsewhere (for example WS3's planner): validated like Claude's, approved like any other."""
        plan, errors = parse_plan(data, goal=goal, source="external")
        if plan is not None:
            fill_defaults(plan, self.vocab)
            errors = validate_plan(plan, self.vocab)
        result = PlanResult(None if errors else plan, errors, 1, "external")
        with self._lock:
            self._plan_version += 1
            if result.plan is not None:
                result.plan.version = self._plan_version
            self.proposal = Proposal("plan", self._plan_version, result, goal or (plan.goal if plan else ""))
        return self.proposal

    def _on_replan(self, state: dict) -> None:
        def work() -> None:
            proposal = self.plan(self.goal, kind="replan", state=state)
            if not proposal.result.ok:
                if self.executor is not None:
                    self.executor.abort("replanning failed")
                return
            if self.cfg.auto_approve_replans and self.executor is not None:
                self.events.add("warn", "replan auto-approved (--auto-approve-replans)")
                self.approve(proposal.version)

        threading.Thread(target=work, daemon=True, name="replan").start()

    # ------------------------------------------------------------------ operator actions (any thread)
    def approve(self, version: int) -> str:
        with self._lock:
            p = self.proposal
            if p is None or p.version != version or not p.result.ok:
                return "no valid plan with that version"
            if p.kind == "replan":
                if self.executor is None or self.executor.state != "awaiting_replan":
                    return "no run is waiting for a replan"
                self.executor.approve_replan(p.result.plan)
                self._save_plan(p.result.plan)
                self.proposal = None
                return "ok"
            if self.executor is not None and self.executor.state not in TERMINAL:
                return "a run is in progress"
            if self.runner.guard.estopped:
                return "release the e-stop first"
            self._requests.append(("run", p.result.plan))
            self.proposal = None
        self.events.add("info", f"plan v{version} approved by the operator")
        return "ok"

    def estop(self, reason: str = "operator") -> None:
        self.runner.guard.estop()
        if self.executor is not None and self.executor.state not in TERMINAL:
            self.executor.estop(reason)
        else:
            self.events.add("error", f"E-STOP ({reason}) while idle: no commands will be sent")

    def _tick_estop(self, reason: str) -> None:
        self.events.add("error", reason)
        self.estop(reason)

    def release(self) -> None:
        if self.executor is not None and self.executor.state not in TERMINAL:
            self.executor.release()
        else:
            self.runner.release_estop()
            self.events.add("warn", "e-stop released")

    def resume(self) -> None:
        if self.executor is not None:
            self.executor.resume()

    def abort(self) -> None:
        if self.executor is not None and self.executor.state not in TERMINAL:
            self.executor.abort()

    def override(self, step_id: str, done: bool) -> None:
        if self.executor is not None:
            self.executor.override(step_id, done)

    def reset_scene(self, seed: int | None = None, lid_on_box: bool | None = None) -> str:
        if self.sim is None:
            return "reset is only available in the sim"
        if self.executor is not None and self.executor.state not in TERMINAL:
            return "a run is in progress"
        with self._lock:
            self._requests.append(("reset", (seed, lid_on_box)))
        return "ok"

    # ------------------------------------------------------------------ control thread
    def close(self) -> None:
        self._closing = True
        if self.executor is not None and self.executor.state not in TERMINAL:
            self.executor.abort("session closed")
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.runner.close()
        if self.rig is not None:
            self.events.add("info", f"robot closed: {self.rig.close()}")

    def _loop(self) -> None:
        while not self._closing:
            with self._lock:
                req = self._requests.pop(0) if self._requests else None
            if req is None:
                if self.sim is None or self._last_obs is None:
                    try:
                        self._encode_frames(self.base_robot.get_observation())
                    except Exception as exc:  # camera hiccup while idle
                        self.events.add("warn", f"observation failed: {exc}")
                time.sleep(0.2)
            elif req[0] == "run":
                self.execute(req[1])
            elif req[0] == "reset":
                seed, lid = req[1]
                if seed is not None:
                    self.cfg.seed = seed
                if lid is not None:
                    self.cfg.lid_on_box = lid
                obs = self.base_robot.reset(self.cfg.seed, self.cfg.lid_on_box)
                self.runner.release_estop()
                self._encode_frames(obs)
                self.events.add("info", f"sim scene reset (seed {self.cfg.seed}, lid "
                                f"{'on the box' if self.cfg.lid_on_box else 'on the table'})")

    def execute(self, plan: Plan) -> RunResult:
        """Run an approved plan to the end on the calling thread (the control thread, or the CLI's main thread)."""
        cfg = self.cfg
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = re.sub(r"[^a-z0-9]+", "_", (plan.goal or cfg.task).lower()).strip("_")[:40] or "run"
        self.run_dir = self.runs_dir / f"{stamp}_{slug}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events.set_path(self.run_dir / "events.jsonl")
        self.meter.log_path = self.run_dir / "claude_calls.jsonl"
        self._save_plan(plan)
        for t in list(self.runner.active_targets()):
            self.runner.stop(t)
        self.executor = Executor(
            runner=self.runner, robot=self.robot, vocab=self.vocab, verifier=self.verifier,
            resolve_target=self.resolve_target, backends=self.target_backend, events=self.events, fps=cfg.fps,
            realtime=cfg.realtime, clock="ticks" if self.sim is not None else "wall", max_replans=cfg.max_replans,
            on_replan=self._on_replan, on_frame=lambda obs, report: self._encode_frames(obs),
            max_run_s=cfg.max_run_s,
            home_target=lambda arm: f"{arm}:prim" if f"{arm}:prim" in self.target_backend else None,
            monitors=self.rig.monitors if self.rig is not None else None,
            on_attempt=self._log_attempt,
            on_estop=self.rig.hold if self.rig is not None else None,
        )
        result = self.executor.run(plan)
        self._encode_frames(self.base_robot.get_observation())
        trial = self._log_trial(result, plan)
        summary = {"result": dataclasses.asdict(result), "trial": trial, "cost": self.meter.summary(),
                   "gate": self.gate.stats.as_dict() if self.gate else None, "layout": self.layout_name(),
                   "config": {k: v for k, v in dataclasses.asdict(cfg).items() if "token" not in k}}
        if isinstance(self.verifier, ShadowVerifier):
            summary["shadow_verifier"] = self.verifier.agreement()
        (self.run_dir / "result.json").write_text(json.dumps(summary, indent=2, default=str))
        self.last_result = summary
        return result

    def _save_plan(self, plan: Plan) -> None:
        if self.run_dir is not None:
            (self.run_dir / f"plan_v{plan.version}.json").write_text(plan.dumps())

    def layout_name(self) -> str:
        if self.cfg.layout:
            return self.cfg.layout
        if self.sim is not None:
            return f"seed{self.cfg.seed}" + ("_lidonbox" if self.cfg.lid_on_box else "")
        return ""

    def trial_log_path(self) -> Path:
        if self.cfg.trial_log:
            return Path(self.cfg.trial_log)
        if self.sim is not None:
            return SIM_TRIAL_LOG
        if self.rig is not None and (not self.rig.real or self.rig.observe or self.cfg.policy != "remote"):
            return MOCK_TRIAL_LOG
        return GRADED_TRIAL_LOG

    def _arms(self) -> tuple[str, ...]:
        return self.vocab.station.arms_available if self.rig is not None else ARMS

    def _variant(self) -> str:
        cfg = self.cfg
        if cfg.variant:
            return cfg.variant
        if self.sim is not None:
            return f"{cfg.policy}+{cfg.planner}"
        return "+".join(sorted({cfg.package_for(a) for a in self._arms()})) + f"+{cfg.planner}"

    def _checkpoint(self) -> str:
        cfg = self.cfg
        if cfg.checkpoint:
            return cfg.checkpoint
        if cfg.policy == "scripted":
            return "sim:scripted-stand-in"
        return ", ".join(f"{a}={cfg.package_for(a)}" for a in self._arms())

    def _log_attempt(self, run: Any, ok: bool, reason: str) -> None:
        """One trial_log line per FLUX attempt (task = the skill), so each skill's success rate is graded too."""
        cfg = self.cfg
        tl = load_trial_log()
        verdict = run.verdicts[-1] if run.verdicts else None
        notes = (f"step {run.step.id} attempt {run.attempt}/{run.step.max_attempts} of run "
                 f"{self.run_dir.name if self.run_dir else '-'}; prompt {run.skill.prompt!r}; "
                 f"{(self.executor.t if self.executor else 0) - run.t_start:.1f} s, {run.chunks} chunks; "
                 f"verifier={self.verifier.name}"
                 + (f"; last check {verdict.source} {verdict.done} ({verdict.confidence:.2f}): {verdict.observation}"
                    if verdict else ""))
        trial = tl.Trial(
            task=run.skill.name, condition=cfg.condition, variant=self._variant(), checkpoint=self._checkpoint(),
            steps_total=1, steps_done=1 if ok else 0, success=ok and run.attempt_interventions == 0,
            interventions=run.attempt_interventions, layout=self.layout_name(), failure="" if ok else reason[:120],
            notes=notes[:600], operator=cfg.operator,
        )
        tl.append(self.trial_log_path(), trial)
        self.events.add("info", f"attempt logged: {trial.task} #{trial.attempt} {'verified' if ok else 'failed'}")

    def _log_trial(self, result: RunResult, plan: Plan) -> dict:
        cfg = self.cfg
        tl = load_trial_log()
        sim = self.sim is not None
        variant = self._variant()
        checkpoint = self._checkpoint()
        notes = (f"orchestrator {'SIMULATION, ' if sim else ''}policy={cfg.policy} planner={cfg.planner} "
                 f"verifier={self.verifier.name} replans={result.replans} "
                 f"claude_usd={self.meter.total_usd:.4f} t={result.duration_s:.1f}s run={self.run_dir.name}")
        trial = tl.Trial(
            task=cfg.task, condition=cfg.condition, variant=variant, checkpoint=checkpoint,
            steps_total=max(result.steps_total, 1), steps_done=min(result.steps_done, max(result.steps_total, 1)),
            success=result.status == "succeeded" and result.interventions == 0,
            interventions=result.interventions, layout=self.layout_name(), failure=result.failure[:120],
            notes=notes, operator=cfg.operator,
        )
        path = self.trial_log_path()
        tl.append(path, trial)
        self.events.add("info", f"trial logged to {path}: {trial.task}/{trial.variant} attempt {trial.attempt}, "
                        f"{trial.steps_done}/{trial.steps_total} steps, {'success' if trial.success else 'fail'}")
        return dataclasses.asdict(trial)

    # ------------------------------------------------------------------ HMI state
    def snapshot(self, n_events: int = 150) -> dict[str, Any]:
        with self._lock:
            proposal = self.proposal.to_json() if self.proposal else None
        ex = self.executor.snapshot() if self.executor is not None else None
        return {
            "config": {"env": self.cfg.env, "policy": self.cfg.policy, "planner": self.cfg.planner,
                       "planner_model": self.cfg.planner_model if self.cfg.planner == "claude" else None,
                       "verifier": self.verifier.name, "gate": bool(self.gate), "realtime": self.cfg.realtime,
                       "seed": self.cfg.seed, "lid_on_box": self.cfg.lid_on_box, "task": self.cfg.task,
                       "trial_log": str(self.trial_log_path())},
            "goal": self.goal,
            "planning": self.planning,
            "proposal": proposal,
            "run": ex,
            "estopped": bool(self.runner.guard.estopped),
            "cost": self.meter.summary(),
            "gate": self.gate.stats.as_dict() if self.gate else None,
            "events": self.events.recent(n_events),
            "last_result": self.last_result,
            "vocabulary": [{"name": s.name, "kind": s.kind, "arms": list(s.arms), "prompt": s.prompt}
                           for s in self.vocab.skills.values()],
        }
