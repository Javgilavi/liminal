"""Drive both XLeRobot arms with FLUX 3 Action sessions.

A *target* is what one policy session controls: one arm (option A: the SO-101 package or a per-arm LoRA, 6-D)
or both arms (option C: a 12-D bimanual package). Each target owns its arms while active, its own history
window and its own action queue; several targets may share one backend (one loaded model, single-flight).

Timing follows LeRobot's synchronous runner, which is what BFL validated on robots: a target that needs a new
chunk holds its arms still until the chunk arrives, and its history does not advance while it waits
(lerobot@146d847:docs/source/flux3.mdx:214-224). With ``async_inference=True`` the wait happens in a worker
thread per backend, so the other arm keeps executing its own queue meanwhile.

``prefetch_ticks=P`` (async only, default 0 = off) requests the next chunk when P commands are left in the queue and
keeps executing meanwhile, with the history advancing every tick. The new chunk replaces the rest of the queue from
the index matching the commands executed since its request (ticks spent waiting with an empty queue do not count:
the arm held still), so the arm does not stop for the round trip when P covers the latency.

Every command goes through ``SafetyGuard.filter`` (step limit and joint limits, robot units) before
``robot.send_action``. Model outputs are unbounded (model card, "Out-of-scope use").

Plug-in points: ``RobotInterface`` (real robot, ws4/sim env via ``GymEnvRobot``), ``FluxBackend`` (local GPU,
remote server, fake), ``ArmCalibration`` (robot units -> SO-101 model units) and ``ActionSpaceAdapter`` (joint
models are the identity; ``ws4/flux/ee50.py`` maps EE50 outputs to joints through ws4/kinematics IK).
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from .backends import FluxBackend, PolicyRequest, PolicyResponse
from .cameras import CameraView, prepare_image
from .contract import (
    ARM_DIM,
    ARMS,
    IMG_HEAD,
    OBS_STATE,
    ROBOT_IMAGE_KEYS,
    SO101_SCENE,
    SO101_WRIST,
    WRIST_KEY,
    ModelContract,
    arm_slice,
)
from .history import HistoryBuffer
from .safety import SafetyGuard


class RobotInterface(Protocol):
    def get_observation(self) -> dict[str, Any]: ...

    def send_action(self, action: np.ndarray) -> None: ...


class GymEnvRobot:
    """Adapts a gym-style env (``reset() -> obs`` or ``(obs, info)``, ``step(a) -> (obs, ...)``) to RobotInterface.

    The ws4/sim env exposes the same observation keys as the robot; actions are the 12-D joint targets.
    """

    def __init__(self, env: Any, seed: int | None = None):
        self.env = env
        out = env.reset(seed=seed) if seed is not None else env.reset()
        self._obs = out[0] if isinstance(out, tuple) else out

    def get_observation(self) -> dict[str, Any]:
        return self._obs

    def send_action(self, action: np.ndarray) -> None:
        out = self.env.step(np.asarray(action, dtype=np.float32))
        self._obs = out[0] if isinstance(out, tuple) else out


@dataclass
class ArmCalibration:
    """``model = robot * scale + offset`` per joint; model units are the SO-101 package's (degrees, gripper %).

    Identity when the arm runs LeRobot ``so_follower`` in degrees mode (``use_degrees=True``, gripper 0..100),
    which is what the package expects (flux-action@830f1ce:docs/setup.md:172-176).
    """

    scale: np.ndarray = field(default_factory=lambda: np.ones(ARM_DIM))
    offset: np.ndarray = field(default_factory=lambda: np.zeros(ARM_DIM))

    def __post_init__(self) -> None:
        self.scale = np.asarray(self.scale, dtype=np.float64).reshape(ARM_DIM)
        self.offset = np.asarray(self.offset, dtype=np.float64).reshape(ARM_DIM)
        if (self.scale == 0).any():
            raise ValueError("scale must be nonzero")

    @classmethod
    def mirrored(cls, dims: tuple[int, ...] = (0, 4)) -> ArmCalibration:
        """Negate shoulder pan and wrist roll: the experiment of presenting a mirrored arm (with a mirrored
        camera view) as an ordinary SO-101. UNVERIFIED: whether the model behaves better this way."""
        scale = np.ones(ARM_DIM)
        scale[list(dims)] = -1.0
        return cls(scale=scale)

    def to_model(self, q: np.ndarray) -> np.ndarray:
        return np.asarray(q, dtype=np.float64) * self.scale + self.offset

    def to_robot(self, q: np.ndarray) -> np.ndarray:
        return (np.asarray(q, dtype=np.float64) - self.offset) / self.scale


class ActionSpaceAdapter(Protocol):
    """Maps SO-101 joint vectors of the target's arms to the model's action space and back."""

    def joints_to_model(self, joints: np.ndarray) -> np.ndarray: ...

    def model_to_joints(self, chunk: np.ndarray, joints_now: np.ndarray) -> np.ndarray: ...


class JointSpace:
    """Joint-space models (SO-101 package, 12-D bimanual heads): the identity."""

    def joints_to_model(self, joints: np.ndarray) -> np.ndarray:
        return np.asarray(joints, dtype=np.float64)

    def model_to_joints(self, chunk: np.ndarray, joints_now: np.ndarray) -> np.ndarray:
        return np.asarray(chunk, dtype=np.float64)


@dataclass
class TargetConfig:
    name: str
    arms: tuple[str, ...]
    backend: str
    cameras: dict[str, CameraView]  # model camera key -> robot view
    calibration: dict[str, ArmCalibration] = field(default_factory=dict)
    adapter: ActionSpaceAdapter = field(default_factory=JointSpace)

    def __post_init__(self) -> None:
        if not self.arms or any(arm not in ARMS for arm in self.arms) or len(set(self.arms)) != len(self.arms):
            raise ValueError(f"invalid arms {self.arms}")
        for arm in self.arms:
            self.calibration.setdefault(arm, ArmCalibration())


def per_arm_target(
    arm: str,
    backend: str = "so101",
    scene: CameraView | None = None,
    calibration: ArmCalibration | None = None,
    mirror: bool = False,
) -> TargetConfig:
    """Option A: the SO-101 package on one arm; scene = head camera, wrist = that arm's gripper camera."""
    scene = scene or CameraView(IMG_HEAD)
    wrist = CameraView(WRIST_KEY[arm])
    if mirror:
        scene = CameraView(scene.source, scene.crop, scene.aspect, mirror=True)
        wrist = CameraView(wrist.source, wrist.crop, wrist.aspect, mirror=True)
        calibration = calibration or ArmCalibration.mirrored()
    return TargetConfig(
        name=arm,
        arms=(arm,),
        backend=backend,
        cameras={SO101_SCENE: scene, SO101_WRIST: wrist},
        calibration={arm: calibration or ArmCalibration()},
    )


def bimanual_target(backend: str = "bimanual", name: str = "both") -> TargetConfig:
    """Option C: one 12-D package on both arms, cameras keep their robot names."""
    return TargetConfig(
        name=name,
        arms=ARMS,
        backend=backend,
        cameras={key: CameraView(key) for key in ROBOT_IMAGE_KEYS},
    )


@dataclass
class _TargetState:
    config: TargetConfig
    contract: ModelContract
    history: HistoryBuffer
    queue: deque = field(default_factory=deque)
    task: str = ""
    active: bool = False
    pending: Future | None = None
    last_response: PolicyResponse | None = None
    n_chunks: int = 0
    executed: int = 0  # commands executed in this episode (prefetch alignment)
    request_executed: int = 0  # ``executed`` when the pending request's observation was taken


@dataclass
class TickReport:
    tick: int
    command: np.ndarray
    status: dict[str, str]
    guard_intervened: bool
    latencies: dict[str, float]


class DualArmRunner:
    def __init__(
        self,
        robot: RobotInterface,
        targets: list[TargetConfig],
        backends: dict[str, FluxBackend],
        guard: SafetyGuard | None = None,
        async_inference: bool = False,
        prefetch_ticks: int = 0,
    ):
        if prefetch_ticks < 0 or (prefetch_ticks and not async_inference):
            raise ValueError("prefetch_ticks needs async_inference and must be >= 0")
        self.robot = robot
        self.backends = backends
        self.guard = guard or SafetyGuard()
        self.async_inference = async_inference
        self.prefetch_ticks = prefetch_ticks
        self._targets: dict[str, _TargetState] = {}
        for cfg in targets:
            if cfg.name in self._targets:
                raise ValueError(f"duplicate target {cfg.name!r}")
            if cfg.backend not in backends:
                raise KeyError(f"target {cfg.name!r} uses unknown backend {cfg.backend!r}")
            contract = backends[cfg.backend].contract
            if contract.output_space == "joints" and contract.action_dim != ARM_DIM * len(cfg.arms):
                raise ValueError(
                    f"target {cfg.name!r} drives {len(cfg.arms)} arm(s) but backend {cfg.backend!r} "
                    f"outputs {contract.action_dim}-D joint actions"
                )
            missing = [k for k in contract.camera_keys if k not in cfg.cameras]
            if missing:
                raise ValueError(f"target {cfg.name!r} maps no robot camera to {missing}")
            self._targets[cfg.name] = _TargetState(
                cfg, contract, HistoryBuffer(contract.n_obs_steps, contract.action_dim)
            )
        # One worker per backend: calls to one model are serialized, different models run in parallel.
        self._executors = {
            name: ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"flux-{name}") for name in backends
        }
        self._owner: dict[str, str] = {}
        self._command: np.ndarray | None = None  # last command sent, robot units
        self._tick = 0

    # ------------------------------------------------------------------ task control
    def start(self, target: str, task: str) -> None:
        """Activate ``target`` with instruction ``task``; claims its arms and starts a fresh episode for it."""
        ts = self._targets[target]
        for arm in ts.config.arms:
            owner = self._owner.get(arm)
            if owner is not None and owner != target:
                raise RuntimeError(f"arm {arm!r} is owned by target {owner!r}; stop it first")
        for arm in ts.config.arms:
            self._owner[arm] = target
        ts.task, ts.active = task, True
        self._reset_target(ts)

    def stop(self, target: str) -> None:
        """Deactivate ``target``; its arms hold their last command."""
        ts = self._targets[target]
        ts.active = False
        self._reset_target(ts)
        for arm in ts.config.arms:
            if self._owner.get(arm) == target:
                del self._owner[arm]

    def _reset_target(self, ts: _TargetState) -> None:
        # Policy, queue and history reset together at task or episode boundaries (flux-action docs/setup.md:360).
        ts.history.reset()
        ts.queue.clear()
        if ts.pending is not None:
            ts.pending.cancel()
        ts.pending = None
        ts.executed = ts.request_executed = 0

    def active_targets(self) -> list[str]:
        return [name for name, ts in self._targets.items() if ts.active]

    def is_busy(self, target: str) -> bool:
        ts = self._targets[target]
        return ts.active and (ts.pending is not None or bool(ts.queue))

    # ------------------------------------------------------------------ observation mapping
    def _model_state(self, ts: _TargetState, state12: np.ndarray) -> np.ndarray:
        joints = np.concatenate(
            [ts.config.calibration[arm].to_model(state12[arm_slice(arm)]) for arm in ts.config.arms]
        )
        return ts.config.adapter.joints_to_model(joints)

    def _model_images(self, ts: _TargetState, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        hw = ts.contract.image_hw
        return {key: prepare_image(obs[view.source], view, hw) for key, view in ts.config.cameras.items()}

    def _request(self, ts: _TargetState) -> PolicyRequest:
        ticks = ts.contract.snapshots
        window = ts.history.window(image_ticks=ticks)
        return PolicyRequest(ts.task, window.states, window.commands, window.images, ticks, target=ts.config.name)

    def _accept(self, ts: _TargetState, response: PolicyResponse, state12: np.ndarray, skip: int = 0,
                replace: bool = False) -> None:  # fmt: skip
        chunk = np.asarray(response.commands, dtype=np.float64)
        if chunk.ndim != 2 or chunk.shape[0] < ts.contract.n_action_steps:
            raise ValueError(f"target {ts.config.name!r}: backend returned {chunk.shape}")
        chunk = chunk[skip : skip + ts.contract.n_action_steps]
        if replace:
            ts.queue.clear()
        ts.last_response = response
        ts.n_chunks += 1
        if not len(chunk):  # the request took longer than the chunk: wait for the next one
            return
        joints_now = np.concatenate(
            [ts.config.calibration[arm].to_model(state12[arm_slice(arm)]) for arm in ts.config.arms]
        )
        joints = ts.config.adapter.model_to_joints(chunk, joints_now)
        if joints.shape != (chunk.shape[0], ARM_DIM * len(ts.config.arms)):
            raise ValueError(f"adapter returned {joints.shape}")
        ts.queue.extend(zip(chunk, joints, strict=True))

    # ------------------------------------------------------------------ control tick
    def release_estop(self) -> None:
        """Clear the e-stop; active targets restart their episode from the current observation."""
        self.guard.release()
        self._command = None
        for ts in self._targets.values():
            if ts.active:
                self._reset_target(ts)

    def _prefetch(self, name: str, ts: _TargetState, obs: dict[str, Any], state12: np.ndarray,
                  status: dict[str, str]) -> bool:  # fmt: skip
        """Prefetch mode for one target and tick: push the observation, take a finished chunk (aligned to the commands
        executed since its request), request the next one early. True when a command is ready to execute."""
        ts.history.push(self._model_state(ts, state12), self._model_images(ts, obs))
        if ts.pending is not None and ts.pending.done():
            future, ts.pending = ts.pending, None
            self._accept(ts, future.result(), state12, skip=ts.executed - ts.request_executed, replace=True)
            status[name] = "replanned"
        if ts.pending is None and len(ts.queue) <= self.prefetch_ticks:
            ts.request_executed = ts.executed
            backend = self.backends[ts.config.backend]
            ts.pending = self._executors[ts.config.backend].submit(backend.predict, self._request(ts))
        if not ts.queue:
            status.setdefault(name, "waiting")
            return False
        return True

    def step(self) -> TickReport:
        if self.guard.estopped:
            # Nothing advances while stopped: no history, no queue, no command.
            self._tick += 1
            status = {name: "estopped" for name in self.active_targets()}
            hold = self._command if self._command is not None else np.zeros(ARM_DIM * len(ARMS))
            return TickReport(self._tick, hold.copy(), status, False, {})
        obs = self.robot.get_observation()
        state12 = np.asarray(obs[OBS_STATE], dtype=np.float64).reshape(-1)
        if state12.shape != (ARM_DIM * len(ARMS),):
            raise ValueError(f"{OBS_STATE} must be 12-D [left | right], got {state12.shape}")
        if self._command is None:
            self._command = state12.copy()
        proposal = self._command.copy()
        status: dict[str, str] = {}
        latencies: dict[str, float] = {}
        executed: dict[str, np.ndarray] = {}
        for name, ts in self._targets.items():
            if not ts.active:
                continue
            if self.prefetch_ticks:
                if not self._prefetch(name, ts, obs, state12, status):
                    continue
            fresh_obs = not self.prefetch_ticks
            if fresh_obs and ts.pending is not None:
                if not ts.pending.done():
                    status[name] = "waiting"
                    continue
                future, ts.pending = ts.pending, None
                self._accept(ts, future.result(), state12)
                fresh_obs = False  # the chunk answers the observation pushed when it was requested
                status[name] = "replanned"
            if fresh_obs:
                ts.history.push(self._model_state(ts, state12), self._model_images(ts, obs))
                if not ts.queue:
                    request = self._request(ts)
                    backend = self.backends[ts.config.backend]
                    if self.async_inference:
                        ts.pending = self._executors[ts.config.backend].submit(backend.predict, request)
                        status[name] = "waiting"
                        continue
                    self._accept(ts, backend.predict(request), state12)
                    status[name] = "replanned"
            model_cmd, joint_cmd = ts.queue.popleft()
            ts.executed += 1
            status.setdefault(name, "executing")
            if ts.last_response is not None:
                latencies[name] = ts.last_response.latency_s
            for i, arm in enumerate(ts.config.arms):
                proposal[arm_slice(arm)] = ts.config.calibration[arm].to_robot(
                    joint_cmd[i * ARM_DIM : (i + 1) * ARM_DIM]
                )
            executed[name] = model_cmd
        command, report = self.guard.filter(proposal, self._command)
        # The history records what was actually commanded (after the guard), in model units.
        for name, model_cmd in executed.items():
            ts = self._targets[name]
            if report.intervened:
                joints = np.concatenate(
                    [ts.config.calibration[arm].to_model(command[arm_slice(arm)]) for arm in ts.config.arms]
                )
                model_cmd = ts.config.adapter.joints_to_model(joints)
            ts.history.record_command(model_cmd)
        self.robot.send_action(command)
        self._command = command
        self._tick += 1
        return TickReport(self._tick, command, status, report.intervened, latencies)

    def run(
        self, seconds: float, fps: float | None = None, on_tick=None, stop_when_idle: bool = False
    ) -> list[TickReport]:
        """Tick at ``fps`` (default: the first target's contract fps) for ``seconds`` of wall time."""
        fps = fps or next(iter(self._targets.values())).contract.fps
        period = 1.0 / fps
        reports = []
        deadline = time.monotonic() + seconds
        next_tick = time.monotonic()
        while time.monotonic() < deadline:
            reports.append(self.step())
            if on_tick is not None:
                on_tick(reports[-1])
            if stop_when_idle and not any(self.is_busy(t) for t in self.active_targets()):
                break
            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()
        return reports

    def close(self) -> None:
        for executor in self._executors.values():
            executor.shutdown(wait=False, cancel_futures=True)


# ---------------------------------------------------------------------- CLI
class _StaticRobot:
    """Dry-run robot: fixed images, state follows the commands exactly."""

    def __init__(self, hw: tuple[int, int] = (480, 640)):
        rng = np.random.default_rng(0)
        self.state = np.zeros(ARM_DIM * len(ARMS))
        self.state[[ARM_DIM - 1, 2 * ARM_DIM - 1]] = 50.0
        self.images = {k: rng.integers(0, 255, (*hw, 3), dtype=np.uint8) for k in ROBOT_IMAGE_KEYS}

    def get_observation(self) -> dict[str, Any]:
        return {OBS_STATE: self.state.copy(), **self.images}

    def send_action(self, action: np.ndarray) -> None:
        self.state = np.asarray(action, dtype=np.float64).copy()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--backend", choices=("fake", "lerobot", "remote"), default="fake")
    parser.add_argument("--checkpoint", default="black-forest-labs/flux-3-action-so101")
    parser.add_argument("--url", default="http://127.0.0.1:8765", help="remote policy server (ws4/flux/remote.py)")
    parser.add_argument("--left-task", default="")
    parser.add_argument("--right-task", default="")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--async-inference", action="store_true")
    args = parser.parse_args(argv)
    from .contract import SO101_CONTRACT

    if args.backend == "fake":
        backend: Any = __import__(f"{__package__}.backends", fromlist=["FakeBackend"]).FakeBackend(
            SO101_CONTRACT, velocity=0.1, delay_s=0.05
        )
    elif args.backend == "lerobot":
        from .backends import LeRobotFlux3Backend

        backend = LeRobotFlux3Backend.from_pretrained(args.checkpoint)
    else:
        from .remote import RemoteBackend

        backend = RemoteBackend(args.url)
    robot = _StaticRobot()
    runner = DualArmRunner(
        robot,
        [per_arm_target("left"), per_arm_target("right")],
        {"so101": backend},
        async_inference=args.async_inference,
    )
    for arm, task in (("left", args.left_task), ("right", args.right_task)):
        if task:
            runner.start(arm, task)
    reports = runner.run(args.seconds, on_tick=lambda r: print(r.tick, r.status, np.round(r.command, 1)))
    runner.close()
    print(f"{len(reports)} ticks")


if __name__ == "__main__":
    main()
