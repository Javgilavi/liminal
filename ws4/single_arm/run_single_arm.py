"""Run a FLUX SO-101 package on ONE SO-101 follower (scene + wrist camera), attempt by attempt, with logging.

Built on ``ws4/flux`` (read-only): ``DualArmRunner`` drives one per-arm target. The runner's state is 12-D
``[left | right]``, so the real arm sits in one slot and the other slot is a phantom that is never commanded
(its value stays constant and nothing is sent for it). Everything else is the tested WS4 path: history kept on
the robot side, one POST per chunk to ``flux.remote``, ``SafetyGuard`` before every command.

Safety layers, outermost last:
1. ``SafetyGuard``: joint limits from this arm's LeRobot calibration (minus a margin) and a per-tick step limit.
2. LeRobot ``max_relative_target`` (degrees): every goal is capped relative to the measured position.
3. Stop keys in the terminal (space, q, Enter, Ctrl-C) or ``touch ws4/single_arm/STOP``: the arm freezes on its
   last command and nothing else is sent until you choose what to do next.
4. A hand on the power cable.

Modes: ``--robot fake`` (no hardware), ``--observe`` (real arm, torque never enabled, commands are computed and
logged but not sent) and the default real run. ``--backend fake`` needs no GPU; ``--backend remote`` calls the
policy server through the SSH tunnel.

    python ws4/single_arm/run_single_arm.py --robot fake --backend fake --seconds 5 --yes
    python ws4/single_arm/run_single_arm.py --observe --backend fake --seconds 10
    python ws4/single_arm/run_single_arm.py --backend remote --token-cmd "ssh ... cat /workspace/flux_token" \\
        --task "stack the cups" --layout L1 --variant A0 --attempts 5
"""

from __future__ import annotations

import argparse
import json
import os
import select
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
WS4 = HERE.parent
if str(WS4) not in sys.path:
    sys.path.insert(0, str(WS4))

from flux.backends import FakeBackend, PolicyRequest  # noqa: E402
from flux.cameras import CameraView, prepare_image  # noqa: E402
from flux.contract import (  # noqa: E402
    ARM_DIM,
    ARM_JOINTS,
    IMG_HEAD,
    OBS_STATE,
    SO101_CONTRACT,
    WRIST_KEY,
    arm_slice,
)
from flux.dual_arm_runner import ArmCalibration, DualArmRunner, per_arm_target  # noqa: E402
from flux.history import HistoryBuffer  # noqa: E402
from flux.safety import DEFAULT_MAX_STEP, SafetyConfig, SafetyGuard, limits_from_lerobot_calibration  # noqa: E402

CHECKPOINT = "black-forest-labs/flux-3-action-so101@0713a7a"  # weights identical to c9e13b2 (flux/checkpoint_facts.md)
STOP_FILE = HERE / "STOP"
KEY_FILE = HERE / "KEY"  # one character, read once: lets a headless operator answer the same prompts as the keyboard
RUNS = HERE / "runs"
PHANTOM_STATE = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 50.0])  # the unused slot; constant, never sent
# Steps of "stack the cups" for the trial log: grasp one cup, lift it clear, release it inside the other.
STEPS = ("grasp a cup", "lift it clear of the table", "release it inside the other cup")


# ---------------------------------------------------------------------- robot side
class SingleArmRobot:
    """``RobotInterface`` for one LeRobot ``so_follower`` with cameras named ``scene`` and ``wrist``.

    On the real arm it talks to ``robot.bus`` directly instead of ``SOFollower.get_observation`` /
    ``send_action``, only to add bus retries: one corrupted status packet otherwise aborts the run (seen on this
    arm on 2026-10-17, ``Failed to write 'Lock' ... Incorrect status packet``). The goal cap is LeRobot's own
    ``ensure_safe_goal_position`` with the same ``max_relative_target``, measured once per tick.
    """

    def __init__(self, robot: Any, slot: str = "left", send: bool = True, max_relative_target: float | None = None,
                 retries: int = 3):  # fmt: skip
        self.robot, self.slot, self.send = robot, slot, send
        self.max_relative_target, self.retries = max_relative_target, retries
        self.direct = hasattr(robot, "bus")
        self.last_raw: dict[str, Any] = {}
        self.last_q: np.ndarray | None = None
        self.last_sent: np.ndarray | None = None
        self.capped = 0  # ticks where LeRobot's cap changed the goal

    def joints(self) -> np.ndarray:
        if self.direct:
            pos = self.robot.bus.sync_read("Present_Position", num_retry=self.retries)
            self.last_q = np.array([float(pos[j]) for j in ARM_JOINTS])
        else:
            raw = self.robot.get_observation()
            self.last_q = np.array([float(raw[f"{j}.pos"]) for j in ARM_JOINTS])
        return self.last_q.copy()

    def get_observation(self) -> dict[str, Any]:
        if self.direct:
            q = self.joints()
            raw: dict[str, Any] = {f"{j}.pos": float(v) for j, v in zip(ARM_JOINTS, q, strict=True)}
            for key, cam in self.robot.cameras.items():
                raw[key] = cam.read_latest()
        else:
            raw = self.robot.get_observation()
            self.last_q = np.array([float(raw[f"{j}.pos"]) for j in ARM_JOINTS])
        self.last_raw = raw
        state = np.tile(PHANTOM_STATE, 2)
        state[arm_slice(self.slot)] = self.last_q
        wrist = np.asarray(raw["wrist"])
        scene = np.asarray(raw["scene"]) if "scene" in raw else np.zeros_like(wrist)  # --scene-cam blank
        return {OBS_STATE: state, IMG_HEAD: scene, WRIST_KEY[self.slot]: wrist}

    def send_action(self, action: np.ndarray) -> None:
        arm = np.asarray(action, dtype=np.float64)[arm_slice(self.slot)]
        self.send_joints(arm)

    def send_joints(self, arm: np.ndarray, measure: bool = False) -> None:
        self.last_sent = np.asarray(arm, dtype=np.float64).copy()
        if not self.send:
            return
        goal = {j: float(v) for j, v in zip(ARM_JOINTS, arm, strict=True)}
        if not self.direct:
            self.robot.send_action({f"{j}.pos": v for j, v in goal.items()})
            return
        if self.max_relative_target is not None:
            from lerobot.robots.utils import ensure_safe_goal_position

            present = self.joints() if measure or self.last_q is None else self.last_q
            capped = ensure_safe_goal_position(
                {j: (g, float(present[i])) for i, (j, g) in enumerate(goal.items())}, float(self.max_relative_target)
            )
            self.capped += any(abs(capped[j] - goal[j]) > 1e-9 for j in goal)
            goal = capped
        self.robot.bus.sync_write("Goal_Position", goal, num_retry=self.retries)


class FakeSO101:
    """Stand-in for a connected ``SOFollower``: the arm reaches every goal at once, cameras are noise."""

    def __init__(self, start: np.ndarray | None = None):
        self.q = np.array([0.0, -95.0, 95.0, 70.0, 0.0, 2.0]) if start is None else np.asarray(start, float)
        self.rng = np.random.default_rng(0)

    def get_observation(self) -> dict[str, Any]:
        obs: dict[str, Any] = {f"{j}.pos": float(v) for j, v in zip(ARM_JOINTS, self.q, strict=True)}
        for cam in ("scene", "wrist"):
            obs[cam] = self.rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)
        return obs

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        self.q = np.array([action[f"{j}.pos"] for j in ARM_JOINTS])
        return action

    def disconnect(self) -> None:
        pass


def connect_so101(args: argparse.Namespace) -> Any:
    """Connect without LeRobot's ``connect()``, which always ends with torque enabled.

    Torque stays off in ``--observe``. Otherwise the goal registers are first set to the measured position, so
    enabling torque holds the arm where it is instead of jumping to a stale goal.
    """
    from lerobot.cameras.configs import Cv2Rotation
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.motors.feetech import OperatingMode
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.robots.so_follower.so_follower import SOFollower

    def cam(spec: str, rotate: int) -> OpenCVCameraConfig:
        rotation = {0: Cv2Rotation.NO_ROTATION, 90: Cv2Rotation.ROTATE_90, 180: Cv2Rotation.ROTATE_180,
                    270: Cv2Rotation.ROTATE_270}[rotate]  # fmt: skip
        index = int(spec) if spec.isdigit() else Path(spec)
        # warmup_s=3: the global shutter scene camera needs more than LeRobot's 1 s default for its first frame.
        return OpenCVCameraConfig(index_or_path=index, fps=30, width=640, height=480, rotation=rotation, fourcc="MJPG",
                                  warmup_s=3)  # fmt: skip

    config = SOFollowerRobotConfig(
        id=args.robot_id,
        port=args.port,
        use_degrees=True,
        max_relative_target=float(args.max_relative_target),
        cameras={"wrist": cam(args.wrist_cam, args.wrist_rotate)},
    )
    if args.scene_cam != "blank":
        config.cameras["scene"] = cam(args.scene_cam, args.scene_rotate)
    robot = SOFollower(config)
    if not robot.calibration:
        raise SystemExit(f"no calibration file {robot.calibration_fpath}; run lerobot-calibrate first")
    bus = robot.bus
    bus.connect()
    bus.disable_torque()
    if not bus.is_calibrated:
        print(f"writing {robot.calibration_fpath.name} to the motors")
        bus.write_calibration(robot.calibration)
    for camera in robot.cameras.values():
        camera.connect()
    # Same register setup as SOFollower.configure (lerobot 0.5.1 so_follower.py), minus the final torque enable.
    bus.configure_motors()
    for motor in bus.motors:
        bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
        bus.write("P_Coefficient", motor, 16)
        bus.write("I_Coefficient", motor, 0)
        bus.write("D_Coefficient", motor, 32)
        if motor == "gripper":
            bus.write("Max_Torque_Limit", motor, 500)
            bus.write("Protection_Current", motor, 250)
            bus.write("Overload_Torque", motor, 25)
    if not args.observe:
        # On this arm a Goal_Position write already switches torque on (observed 2026-10-17), so the goal is set
        # to the measured position first: the arm then holds exactly where it rests.
        present = bus.sync_read("Present_Position", normalize=False, num_retry=3)
        bus.sync_write("Goal_Position", present, normalize=False, num_retry=3)
        bus.enable_torque(num_retry=3)
        torque = bus.sync_read("Torque_Enable", normalize=False, num_retry=3)
        if not all(torque.values()):
            bus.disable_torque(num_retry=3)
            raise SystemExit(f"torque did not come on everywhere: {torque}")
    return robot


def disconnect_so101(robot: Any) -> None:
    for camera in robot.cameras.values():
        camera.disconnect()
    robot.bus.disable_torque(num_retry=5)
    robot.bus.disconnect(disable_torque=False)


# ---------------------------------------------------------------------- operator keys
@dataclass
class Keys:
    """Single keys from the terminal (cbreak mode) plus Ctrl-C and the STOP file."""

    pressed: list[str] = field(default_factory=list)
    _old: Any = None

    def __enter__(self) -> Keys:
        if sys.stdin.isatty():
            import termios
            import tty

            self._old = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        signal.signal(signal.SIGINT, lambda *_: self.pressed.append("ctrl-c"))
        STOP_FILE.unlink(missing_ok=True)
        KEY_FILE.unlink(missing_ok=True)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._old is not None:
            import termios

            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old)
        signal.signal(signal.SIGINT, signal.default_int_handler)

    def poll(self) -> str | None:
        if STOP_FILE.exists():
            STOP_FILE.unlink(missing_ok=True)
            return "stop-file"
        if KEY_FILE.exists():
            key = KEY_FILE.read_text()[:1] or "\n"
            KEY_FILE.unlink(missing_ok=True)
            return key
        if self.pressed:
            return self.pressed.pop(0)
        if sys.stdin.isatty() and select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

    def wait(self, allowed: str, prompt: str, auto: str | None = None) -> str:
        """Block until one of ``allowed`` (or a stop) arrives; ``auto`` answers at once when stdin is no TTY."""
        print(prompt, flush=True)
        if not sys.stdin.isatty() and auto is not None:
            return auto
        while True:
            key = self.poll()
            if key is not None and (key in allowed or key in ("ctrl-c", "stop-file")):
                return key
            time.sleep(0.02)


STOP_KEYS = (" ", "q", "\n", "ctrl-c", "stop-file")


# ---------------------------------------------------------------------- recording
class Recorder:
    """Per-attempt trace (npz + json) and a side-by-side scene | wrist video."""

    def __init__(self, out: Path, video: bool):
        self.out = out
        out.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict[str, Any]] = []
        self.writer = None
        self.video = video

    def frame(self, raw: dict[str, Any], text: str) -> None:
        if not self.video or "scene" not in raw:
            return
        import cv2

        tiles = [cv2.resize(np.asarray(raw[k]), (320, 240)) for k in ("scene", "wrist")]
        img = cv2.cvtColor(np.concatenate(tiles, axis=1), cv2.COLOR_RGB2BGR)
        cv2.putText(img, text, (4, 232), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        if self.writer is None:
            self.writer = cv2.VideoWriter(str(self.out / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 240))
        self.writer.write(img)

    def close(self, meta: dict[str, Any]) -> None:
        if self.writer is not None:
            self.writer.release()
        if self.rows:
            np.savez_compressed(
                self.out / "trace.npz",
                t=np.array([r["t"] for r in self.rows]),
                state=np.array([r["state"] for r in self.rows]),
                command=np.array([r["command"] for r in self.rows]),
                guard=np.array([r["guard"] for r in self.rows]),
                status=np.array([r["status"] for r in self.rows]),
            )
        (self.out / "meta.json").write_text(json.dumps(meta, indent=2, default=float))


# ---------------------------------------------------------------------- policy side
def read_token(args: argparse.Namespace) -> str | None:
    """Token from a command's stdout (e.g. ``ssh pod cat /workspace/flux_token``), never printed or put in argv."""
    if args.token_cmd:
        out = subprocess.run(shlex.split(os.path.expanduser(args.token_cmd)), capture_output=True, text=True)
        if out.returncode != 0 or not out.stdout.strip():
            raise SystemExit(f"--token-cmd failed (exit {out.returncode}): {out.stderr.strip()[:200]}")
        return out.stdout.strip()
    return os.environ.get("FLUX_TOKEN")


def build_backend(args: argparse.Namespace) -> Any:
    if args.backend == "fake":
        # Scripted stand-in: a constant per-tick drift on each joint and an optional absolute gripper target.
        gripper = None if args.fake_gripper < 0 else args.fake_gripper
        velocity = np.array(args.fake_velocity)
        return FakeBackend(SO101_CONTRACT, velocity=velocity, absolute=gripper, delay_s=args.fake_delay)
    from flux.remote import RemoteBackend

    return RemoteBackend(args.server, args.package, token=read_token(args), timeout_s=args.timeout)


def warm_up(backend: Any, robot: SingleArmRobot, target: Any, tasks: list[str]) -> None:
    """One throw-away request per prompt, so a prompt the server has not encoded yet (15 to 34 s on its CPU,
    cloud/RUNPOD.md) never stalls an attempt."""
    contract = backend.contract
    obs = robot.get_observation()
    calib = target.calibration[robot.slot]
    hist = HistoryBuffer(contract.n_obs_steps, contract.action_dim)
    images = {key: prepare_image(obs[view.source], view, contract.image_hw) for key, view in target.cameras.items()}
    hist.push(calib.to_model(obs[OBS_STATE][arm_slice(robot.slot)]), images)
    window = hist.window(image_ticks=contract.snapshots)
    for task in tasks:
        start = time.perf_counter()
        response = backend.predict(
            PolicyRequest(task, window.states, window.commands, window.images, contract.snapshots, target="warmup")
        )
        print(f"warm-up {task!r}: {time.perf_counter() - start:.2f} s round trip, "
              f"server {response.info.get('server_latency_s')}", flush=True)  # fmt: skip


# ---------------------------------------------------------------------- one attempt
def move_to(robot: SingleArmRobot, goal: np.ndarray, guard_cfg: SafetyConfig, deg_per_tick: float, keys: Keys) -> bool:
    """Slow joint-space move to ``goal`` (robot units) through the same limits; False if a stop key interrupts."""
    lo, hi = guard_cfg.lower[arm_slice(robot.slot)], guard_cfg.upper[arm_slice(robot.slot)]
    goal = np.clip(goal, lo, hi)
    cmd = robot.last_sent if robot.last_sent is not None else robot.joints()
    step = np.full(ARM_DIM, deg_per_tick)
    step[-1] = 2 * deg_per_tick
    while True:
        if keys.poll() in STOP_KEYS:
            return False
        cmd = cmd + np.clip(goal - cmd, -step, step)
        robot.send_joints(cmd, measure=True)
        if np.allclose(cmd, goal):
            return True
        time.sleep(1 / 30)


def run_attempt(
    args: argparse.Namespace, runner: DualArmRunner, robot: SingleArmRobot, keys: Keys, task: str, out: Path
) -> dict[str, Any]:
    rec = Recorder(out, video=not args.no_video)
    target = robot.slot
    start_q = robot.joints()
    runner.guard.release()
    runner._command = None  # noqa: SLF001 - start every attempt from the measured pose
    runner.start(target, task)
    t0 = time.monotonic()
    end_reason, period, next_tick = "timeout", 1 / SO101_CONTRACT.fps, time.monotonic()
    chunk_latencies: list[float] = []
    server_latencies: list[float] = []
    n_chunks0 = runner._targets[target].n_chunks  # noqa: SLF001
    guard_ticks, capped0 = 0, robot.capped
    print(f"attempt running: {task!r} for up to {args.seconds:.0f} s. space/q/Enter = STOP, s = success, f = fail",
          flush=True)  # fmt: skip
    while time.monotonic() - t0 < args.seconds:
        key = keys.poll()
        if key in STOP_KEYS:
            runner.guard.estop()
            end_reason = f"stop ({key!r})"
            break
        if key in ("s", "f"):
            end_reason = {"s": "operator: success", "f": "operator: fail"}[key]
            break
        try:
            report = runner.step()
        except Exception as exc:  # bus or server failure after retries: freeze on the last command, do not crash
            runner.guard.estop()
            end_reason = f"stop (error {type(exc).__name__}: {str(exc)[:120]})"
            break
        ts = runner._targets[target]  # noqa: SLF001
        if report.status.get(target) == "replanned" and ts.last_response is not None:
            chunk_latencies.append(ts.last_response.latency_s)
            server_latencies.append(ts.last_response.info.get("server_latency_s") or float("nan"))
        guard_ticks += report.guard_intervened
        q = np.array([float(robot.last_raw[f"{j}.pos"]) for j in ARM_JOINTS])
        rec.rows.append({"t": time.monotonic() - t0, "state": q, "command": report.command[arm_slice(target)],
                         "guard": report.guard_intervened, "status": report.status.get(target, "")})  # fmt: skip
        rec.frame(robot.last_raw, f"{task} | {report.status.get(target, '')} | t={time.monotonic() - t0:4.1f}s")
        if report.tick % 30 == 0:
            print(f"t={time.monotonic() - t0:5.1f}s {report.status} q={np.round(q, 1)} guard={guard_ticks}",
                  flush=True)  # fmt: skip
        next_tick += period
        delay = next_tick - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        else:
            next_tick = time.monotonic()
    runner.stop(target)
    duration = time.monotonic() - t0
    print(f"attempt ended: {end_reason} after {duration:.1f} s", flush=True)
    meta = {
        "task": task, "end_reason": end_reason, "duration_s": duration, "start_q": start_q.tolist(),
        "chunks": runner._targets[target].n_chunks - n_chunks0,  # noqa: SLF001
        "chunk_round_trip_s": chunk_latencies, "server_latency_s": server_latencies, "guard_ticks": guard_ticks,
        "lerobot_capped_ticks": robot.capped - capped0, "ticks": len(rec.rows),
        "args": {k: v for k, v in vars(args).items() if k != "token_cmd"},
        "checkpoint": CHECKPOINT if args.backend == "remote" else f"{args.backend} backend",
    }  # fmt: skip
    rec.close(meta)
    return meta


def finish_attempt(args: argparse.Namespace, robot: SingleArmRobot, keys: Keys, guard_cfg: SafetyConfig,
                   rest: np.ndarray, meta: dict[str, Any]) -> None:  # fmt: skip
    """Bring the arm back to the rest pose slowly (or on request after a stop)."""
    if args.observe or args.robot == "fake":
        return
    if meta["end_reason"].startswith("stop"):
        key = keys.wait("rx", "STOPPED, arm holds. r = return slowly to rest, x = keep holding and quit")
        if key != "r":
            raise KeyboardInterrupt
    print("returning to rest pose (slow) ...", flush=True)
    if not move_to(robot, rest, guard_cfg, args.return_speed, keys):
        print("return interrupted; arm holds", flush=True)
        raise KeyboardInterrupt


def ask_result(keys: Keys, meta: dict[str, Any]) -> dict[str, Any] | None:
    if not sys.stdin.isatty():
        return None
    key = keys.wait("0123n", f"steps done 0..{len(STEPS)} ({'; '.join(STEPS)}), n = do not log this attempt")
    if key not in "0123":
        return None
    steps = int(key)
    key = keys.wait("0123", "human interventions during the attempt (0..3)")
    interventions = int(key) if key in "0123" else 0
    return {"steps_done": steps, "interventions": interventions}


def log_trial(args: argparse.Namespace, task: str, meta: dict[str, Any], result: dict[str, Any], out: Path) -> None:
    sys.path.insert(0, str(WS4 / "eval"))
    import trial_log

    failure = "" if result["steps_done"] == len(STEPS) else f"stopped before: {STEPS[result['steps_done']]}"
    trial = trial_log.Trial(
        task=args.trial_task, condition="zero_shot", variant=args.variant, checkpoint=meta["checkpoint"],
        steps_total=len(STEPS), steps_done=result["steps_done"],
        success=result["steps_done"] == len(STEPS) and result["interventions"] == 0,
        interventions=result["interventions"], layout=args.layout, failure=failure,
        notes=f"prompt={task!r}; guidance=server default ({args.guidance_note}); end={meta['end_reason']}; "
        f"chunks={meta['chunks']}; guard_ticks={meta['guard_ticks']}",
        video=str((out / "video.mp4").relative_to(WS4.parent)) if (out / "video.mp4").exists() else "",
        operator=args.operator,
    )  # fmt: skip
    trial = trial_log.append(trial_log.DEFAULT_LOG, trial)
    print(f"logged {args.trial_task}/{args.variant} attempt {trial.attempt}: {trial.steps_done}/{trial.steps_total}",
          flush=True)  # fmt: skip


# ---------------------------------------------------------------------- main
def build_guard(args: argparse.Namespace, slot: str, calib_path: Path | None, rest: np.ndarray) -> SafetyConfig:
    cfg = SafetyConfig()
    if calib_path is not None:
        lower, upper = limits_from_lerobot_calibration(calib_path, margin_deg=args.limit_margin)
        # The folded rest pose sits on the lift and elbow end stops, inside the margin; the clamp must not yank
        # the arm out of it on the first tick, so the limits always include the measured start pose.
        cfg = cfg.with_arm_limits(slot, np.minimum(lower, rest), np.maximum(upper, rest))
    step = np.tile(DEFAULT_MAX_STEP, 2)
    s = arm_slice(slot)
    step[s] = [args.max_step] * (ARM_DIM - 1) + [args.max_gripper_step]
    return SafetyConfig(cfg.lower, cfg.upper, step)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--robot", choices=("so101", "fake"), default="so101")
    p.add_argument("--observe", action="store_true", help="real arm, torque never enabled, nothing sent")
    p.add_argument("--port", default="/dev/ttyACM0")
    p.add_argument("--robot-id", default="red_follower", help="LeRobot calibration id (so_follower/<id>.json)")
    p.add_argument("--scene-cam", default="", help="/dev/videoN of the scene camera; 'blank' = black (plumbing only)")
    p.add_argument("--wrist-cam", default="/dev/video4")
    p.add_argument("--scene-rotate", type=int, choices=(0, 90, 180, 270), default=0)
    p.add_argument("--wrist-rotate", type=int, choices=(0, 90, 180, 270), default=0)
    p.add_argument("--slot", choices=("left", "right"), default="left", help="runner slot; no effect on the model")
    p.add_argument("--backend", choices=("fake", "remote"), default="fake")
    p.add_argument("--fake-delay", type=float, default=0.6, help="fake inference latency, s")
    p.add_argument("--fake-velocity", type=float, nargs=ARM_DIM, default=[0.2, 0.2, -0.2, 0.2, 0.0, 0.0],
                   help="fake policy drift, deg per tick per joint")  # fmt: skip
    p.add_argument("--fake-gripper", type=float, default=40.0, help="fake absolute gripper target; -1 = hold")
    p.add_argument("--server", default="http://127.0.0.1:8765")
    p.add_argument("--package", default="so101")
    p.add_argument("--token-cmd", default="", help="command printing the server token; output is never shown")
    p.add_argument("--timeout", type=float, default=90.0)
    p.add_argument("--guidance-note", default="1.0+compile", help="what the server runs, for the trial notes")
    p.add_argument("--task", action="append", default=[], help="prompt; repeat to alternate prompts per attempt")
    p.add_argument("--seconds", type=float, default=40.0, help="per attempt")
    p.add_argument("--attempts", type=int, default=1)
    p.add_argument("--max-step", type=float, default=2.0, help="guard: deg per tick on joints (60 deg/s at 30 Hz)")
    p.add_argument("--max-gripper-step", type=float, default=5.0, help="guard: percent per tick on the gripper")
    p.add_argument("--max-relative-target", type=float, default=10.0, help="LeRobot cap vs measured pos, deg")
    p.add_argument("--limit-margin", type=float, default=5.0, help="deg inside the calibrated range")
    p.add_argument("--return-speed", type=float, default=1.0, help="deg per tick when returning to rest")
    p.add_argument("--offset", type=float, nargs=ARM_DIM, default=None, help="model = robot + offset (A0m tests)")
    p.add_argument("--variant", default="A0")
    p.add_argument("--layout", default="")
    p.add_argument("--trial-task", default="stack_cups")
    p.add_argument("--operator", default="oscar")
    p.add_argument("--no-log", action="store_true", help="never write to ws4/eval/trials.jsonl")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--yes", action="store_true", help="start attempts without waiting for Enter")
    args = p.parse_args(argv)
    tasks = args.task or ["stack the cups"]
    if args.scene_cam == "blank" and args.backend == "remote" and not args.no_log:
        raise SystemExit("a blank scene camera is out of distribution: such attempts must not be logged (--no-log)")

    calib_path = None
    if args.robot == "so101":
        if not args.scene_cam:
            raise SystemExit("--scene-cam is required on the real arm")
        so101 = connect_so101(args)
        calib_path = so101.calibration_fpath
    else:
        so101 = FakeSO101()
    robot = SingleArmRobot(so101, args.slot, send=not args.observe, max_relative_target=args.max_relative_target)
    rest = robot.joints()
    guard_cfg = build_guard(args, args.slot, calib_path, rest)
    s = arm_slice(args.slot)
    print(f"guard limits {np.round(guard_cfg.lower[s], 1)} .. {np.round(guard_cfg.upper[s], 1)}, "
          f"max step {guard_cfg.max_step[s]}, LeRobot max_relative_target {args.max_relative_target}",
          flush=True)  # fmt: skip
    calibration = ArmCalibration(offset=np.array(args.offset)) if args.offset else ArmCalibration()
    target = per_arm_target(args.slot, "policy", scene=CameraView(IMG_HEAD), calibration=calibration)
    backend = build_backend(args)
    runner = DualArmRunner(robot, [target], {"policy": backend}, SafetyGuard(guard_cfg), async_inference=True)
    print(f"rest pose {np.round(rest, 1)}", flush=True)
    session = RUNS / datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        with Keys() as keys:
            if args.backend == "remote":
                warm_up(backend, robot, target, sorted(set(tasks)))
            for i in range(args.attempts):
                task = tasks[i % len(tasks)]
                if not args.yes:
                    key = keys.wait("\n ", f"[{i + 1}/{args.attempts}] layout {args.layout or '-'}, prompt {task!r}: "
                                    "Enter = start, Ctrl-C = quit", auto="\n")  # fmt: skip
                    if key in ("ctrl-c", "stop-file"):
                        break
                out = session / f"attempt{i + 1:02d}"
                meta = run_attempt(args, runner, robot, keys, task, out)
                finish_attempt(args, robot, keys, guard_cfg, rest, meta)
                result = None if args.no_log or args.backend != "remote" else ask_result(keys, meta)
                if result is not None:
                    meta["result"] = result
                    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=float))
                    log_trial(args, task, meta, result, out)
                print(f"saved {out}", flush=True)
    except KeyboardInterrupt:
        print("quit requested", flush=True)
    finally:
        runner.close()
        if args.robot == "so101":
            at_rest = args.observe or np.allclose(robot.joints()[:5], rest[:5], atol=8.0)
            if at_rest:
                disconnect_so101(so101)
            else:
                # Dropping an extended arm onto the cups is worse than holding it: leave torque on.
                for camera in so101.cameras.values():
                    camera.disconnect()
                so101.bus.disconnect(disable_torque=False)
                print("WARNING: arm is NOT at rest and still holds with torque on. Support it, then switch the "
                      "power off, or rerun and return it to rest.", flush=True)  # fmt: skip
    print(f"session folder {session}")


if __name__ == "__main__":
    main()
