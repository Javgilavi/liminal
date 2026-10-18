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
policy server through the SSH tunnel; ``--backend replay`` feeds one recorded take through the same safety stack
(``replay.py``: a scripted replay, never FLUX; the box skill's fallback).

    python ws4/single_arm/run_single_arm.py --robot fake --backend fake --seconds 5 --yes
    python ws4/single_arm/run_single_arm.py --observe --backend fake --seconds 10
    python ws4/single_arm/run_single_arm.py --backend remote --token-cmd "ssh ... cat /workspace/flux_token" \\
        --task "stack the cups" --layout L1 --variant A0 --attempts 5
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import select
import shlex
import signal
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field, replace
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
from keepout import ZONES, KeepOut, read_arm_joints  # noqa: E402
from placement import PlacementView, composite, load_refs, read_rgb, ref_shift  # noqa: E402
from replay import DEFAULT_TAKES, SLOW_MARGIN, ReplayBackend, load_take, reference_frame, take_source  # noqa: E402

CHECKPOINT = "black-forest-labs/flux-3-action-so101@0713a7a"  # weights identical to c9e13b2 (flux/checkpoint_facts.md)
POLICY_BACKENDS = ("remote", "act")  # learned policies: warmed up, graded and logged
ARRIVE_TOL = np.array([6.0, 8.0, 6.0, 6.0, 6.0])  # move_to: deg per arm joint at which a ramp has arrived
STOP_FILE = HERE / "STOP"
KEY_FILE = HERE / "KEY"  # one character, read once: lets a headless operator answer the same prompts as the keyboard
RUNS = HERE / "runs"
REST_FILE = HERE / "calibration" / "red_follower_rest.json"  # folded pose Oscar set at the first A0 session
PHANTOM_STATE = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 50.0])  # the unused slot; constant, never sent
# Steps of "stack the cups" for the trial log: grasp one cup, lift it clear, release it inside the other.
# Graded steps per trial task (ws4/eval/trial_log.py steps_total / steps_done); --steps overrides.
TASK_STEPS = {
    "stack_cups": ("grasp a cup", "lift it clear of the table", "release it inside the other cup"),
    # Post-training protocol (orchestrator, 17 Oct night): reach = fingers around the pen on the mat, grasp = pen
    # lifted in the gripper, insert = pen in its slot and released (the dataset's keep rule).
    "put_pen_in_slot": ("reach the pen", "grasp the pen", "insert it in its slot"),
}
STEPS = TASK_STEPS["stack_cups"]


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
        self.transforms: dict[str, Any] = {}  # camera key -> frame transform (Haller recorder parity)
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
                frame = cam.read_latest()
                raw[key] = self.transforms[key](frame) if key in self.transforms else frame
        else:
            raw = self.robot.get_observation()
            self.last_q = np.array([float(raw[f"{j}.pos"]) for j in ARM_JOINTS])
        self.last_raw = raw
        state = np.tile(PHANTOM_STATE, 2)
        state[arm_slice(self.slot)] = self.last_q
        wrist = np.asarray(raw["wrist"])
        scene = np.asarray(raw["scene"]) if "scene" in raw else np.zeros_like(wrist)  # --scene-cam blank
        return {OBS_STATE: state, IMG_HEAD: scene, WRIST_KEY[self.slot]: wrist}

    def min_voltage(self) -> float | None:
        """Lowest servo supply voltage (V), sampled about once a second to see whether the brick sags."""
        if not self.direct:
            return None
        try:
            return min(self.robot.bus.sync_read("Present_Voltage", normalize=False, num_retry=1).values()) / 10
        except Exception:  # never let a diagnostic read end an attempt
            return None

    def arm_load(self) -> np.ndarray | None:
        """|Present_Load| of the five arm joints (0.1 % of max torque), read every tick for the load freeze."""
        if not self.direct:
            load = getattr(self.robot, "load", None)
            return None if load is None else np.abs(np.asarray(load, dtype=float))
        try:
            values = self.robot.bus.sync_read("Present_Load", list(ARM_JOINTS[:-1]), normalize=False, num_retry=1)
        except Exception:
            return None
        return np.abs(np.array([float(values[j]) for j in ARM_JOINTS[:-1]]))

    def health(self) -> dict[str, list[int]] | None:
        """Per-joint load (0.1 % of max torque, sign = direction), status error bits and temperature (C)."""
        if not self.direct:
            return None
        out = {}
        try:
            for reg, key in (("Present_Load", "load"), ("Status", "status"), ("Present_Temperature", "temp_c")):
                values = self.robot.bus.sync_read(reg, normalize=False, num_retry=1)
                out[key] = [int(values[j]) for j in ARM_JOINTS]
        except Exception:
            return None
        return out

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
        self.load = None  # tests set arm-joint loads (0.1 % of max torque) here
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

    def cam(spec: str, rotate: int, fourcc: str | None = "MJPG") -> OpenCVCameraConfig:
        rotation = {0: Cv2Rotation.NO_ROTATION, 90: Cv2Rotation.ROTATE_90, 180: Cv2Rotation.ROTATE_180,
                    270: Cv2Rotation.ROTATE_270}[rotate]  # fmt: skip
        index = int(spec) if spec.isdigit() else Path(spec)
        # warmup_s=3: the global shutter scene camera needs more than LeRobot's 1 s default for its first frame.
        return OpenCVCameraConfig(index_or_path=index, fps=30, width=640, height=480, rotation=rotation, fourcc=fourcc,
                                  warmup_s=3)  # fmt: skip

    if args.parity_side:  # the parity transform rotates, so the camera itself must not
        args.scene_rotate = args.wrist_rotate = 0

    config = SOFollowerRobotConfig(
        id=args.robot_id,
        port=args.port,
        use_degrees=True,
        max_relative_target=float(args.max_relative_target),
        cameras={"wrist": cam(args.wrist_cam, args.wrist_rotate,
                              None if args.wrist_fourcc.lower() == "none" else args.wrist_fourcc)},
    )
    if args.scene_cam != "blank":
        fourcc = None if args.scene_fourcc.lower() == "none" else args.scene_fourcc
        config.cameras["scene"] = cam(args.scene_cam, args.scene_rotate, fourcc)
    robot = SOFollower(config)
    if not robot.calibration:
        raise SystemExit(f"no calibration file {robot.calibration_fpath}; run lerobot-calibrate first")
    bus = robot.bus
    bus.connect()
    for camera in robot.cameras.values():
        camera.connect()
    if args.parity_side:
        print("camera parity controls:", station_parity(args)[1](), flush=True)
    robot.ws4_launch_torque = any(bus.sync_read("Torque_Enable", normalize=False, num_retry=3).values())
    if args.observe:
        return robot  # observe: no register writes and no torque change (a holding arm keeps holding)
    # LeRobot's connect() switches torque off to write calibration and its register setup, which drops an arm that
    # holds a pose mid-air (the Haller's task arm did on 2026-10-17). Do that only when something actually differs.
    wanted = {"Operating_Mode": OperatingMode.POSITION.value, "P_Coefficient": 16, "I_Coefficient": 0,
              "D_Coefficient": 32, "Return_Delay_Time": 0, "Acceleration": 254}  # fmt: skip
    gripper_wanted = {"Max_Torque_Limit": 500, "Protection_Current": 250, "Overload_Torque": 25}
    differs = [f"{m}.{r}" for r, v in wanted.items() for m, x in bus.sync_read(r, normalize=False, num_retry=3).items()
               if x != v]  # fmt: skip
    differs += [f"gripper.{r}" for r, v in gripper_wanted.items()
                if bus.read(r, "gripper", normalize=False, num_retry=3) != v]  # fmt: skip
    if not bus.is_calibrated or differs:
        torque_on = any(bus.sync_read("Torque_Enable", normalize=False, num_retry=3).values())
        if torque_on and not args.allow_torque_off:
            raise SystemExit(f"registers or calibration differ ({differs[:4]}...) and the arm holds with torque on: "
                             "put it at rest and rerun with --allow-torque-off")  # fmt: skip
        bus.disable_torque()
        if not bus.is_calibrated:
            print(f"writing {robot.calibration_fpath.name} to the motors")
            bus.write_calibration(robot.calibration)
        # Same register setup as SOFollower.configure (lerobot 0.5.1 so_follower.py), minus the torque enable.
        bus.configure_motors()
        for motor in bus.motors:
            bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
            bus.write("P_Coefficient", motor, 16)
            bus.write("I_Coefficient", motor, 0)
            bus.write("D_Coefficient", motor, 32)
            if motor == "gripper":
                for reg, value in gripper_wanted.items():
                    bus.write(reg, motor, value)
    for motor in bus.motors:
        if motor != "gripper":
            # RAM register (resets at power-off): caps the force of a collision with the table or a cup. The red
            # arm's shoulder needed 6 to 9 % to lift slowly at 12 V (measured 2026-10-17), so 60 % leaves margin.
            bus.write("Torque_Limit", motor, int(args.torque_limit * 10), num_retry=3)
    # On these arms a Goal_Position write already switches torque on (observed 2026-10-17), so the goal is set to
    # the measured position first: the arm then holds exactly where it is.
    present = bus.sync_read("Present_Position", normalize=False, num_retry=3)
    bus.sync_write("Goal_Position", present, normalize=False, num_retry=3)
    bus.enable_torque(num_retry=3)
    torque = bus.sync_read("Torque_Enable", normalize=False, num_retry=3)
    if not all(torque.values()):
        raise SystemExit(f"torque did not come on everywhere: {torque}")
    return robot


def station_parity(args: argparse.Namespace):
    """Haller recorder parity (ws4/teleop/camera_parity.py, read-only use): per-key frame transforms (180 deg rotation
    and colour gains of the wrist camera) and a function writing the recorder's fixed V4L2 controls to OUR device
    paths (the haller_cam_* udev links do not exist on this laptop)."""
    from teleop.camera_parity import StationCameras

    cams = StationCameras.load()
    wrist_key = f"{args.parity_side}_wrist"
    transforms = {"scene": lambda f: cams.transform("head", f), "wrist": lambda f: cams.transform(wrist_key, f)}

    def controls() -> dict:
        sys.path.insert(0, str(Path.home() / "haller_ws/hmi/backend"))
        from haller_hmi import v4l2_controls

        spec = cams.specs[wrist_key]
        return {args.wrist_cam: v4l2_controls.apply(args.wrist_cam, spec.controls)} if spec.controls else {}

    return transforms, controls


class FakeOtherArmHolder:
    """Test stand-in for ``OtherArmHolder``: holds a fixed pose (``--fake-other-q``, default ``FAKE_HOLD``)."""

    FAKE_HOLD = np.array([12.13, -13.8, 12.26, 37.54, 29.93, 0.87])

    def __init__(self, port: str, robot_id: str, torque_limit: float, pose: Any = None):
        self.hold_q = np.asarray(pose, float) if pose is not None else self.FAKE_HOLD.copy()
        self.closed = False

    def read(self) -> np.ndarray:
        return self.hold_q.copy()

    def close(self) -> None:
        self.closed = True


class OtherArmHolder:
    """Takes over the other arm and holds it exactly where it is (the chain: the box arm keeps the lid open while a
    pen skill runs; the HMI held it until now). Order, as on the task arm: cap Torque_Limit, write Goal_Position = its
    present position (raw ticks; on these arms that write already switches torque on), then enable torque and
    verify. It is never commanded anywhere else. ``close`` releases the port with torque left ON, so the arm keeps
    holding the lid after the runner exits: support the lid before switching it off. Hardware path UNVERIFIED
    until the first handover on the Haller (fake-arm tests only)."""

    def __init__(self, port: str, robot_id: str, torque_limit: float):
        from show_joints import JOINTS, open_bus

        self.joints = list(JOINTS)
        self.bus = open_bus(port, robot_id)  # connect + read only
        present = self.bus.sync_read("Present_Position", normalize=False, num_retry=3)
        for motor in self.bus.motors:
            if motor != "gripper":
                self.bus.write("Torque_Limit", motor, int(torque_limit * 10), num_retry=3)
        self.bus.sync_write("Goal_Position", present, normalize=False, num_retry=3)
        self.bus.enable_torque(num_retry=3)
        torque = self.bus.sync_read("Torque_Enable", normalize=False, num_retry=3)
        if not all(torque.values()):
            raise SystemExit(f"other arm: torque did not come on everywhere: {torque}")
        self.hold_q = self.read()
        print(f"other arm ({port}) held at {np.round(self.hold_q, 1)}: torque on, goal = its own pose", flush=True)

    def read(self) -> np.ndarray:
        q = self.bus.sync_read("Present_Position", self.joints, num_retry=3)
        return np.array([q[j] for j in self.joints])

    def close(self) -> None:
        self.bus.port_handler.closePort()  # no torque write: the arm keeps holding
        print("other arm released by the runner with torque ON (still holding): support it before power-off",
              flush=True)  # fmt: skip


OTHER_ARM_READER: Any = None  # set while the runner holds the other arm (its port is ours then)


def read_other(args: argparse.Namespace) -> np.ndarray:
    """The other arm's pose: from the holder when the runner holds it, else read-only from its port."""
    if OTHER_ARM_READER is not None:
        return OTHER_ARM_READER()
    return read_arm_joints(args.other_arm_port, args.other_arm_id)


def disconnect_so101(robot: Any, torque_off: bool = True) -> None:
    for camera in robot.cameras.values():
        camera.disconnect()
    if torque_off:
        robot.bus.disable_torque(num_retry=5)
    robot.bus.disconnect(disable_torque=False)


class KeepOutGuard(SafetyGuard):
    """``SafetyGuard`` plus the FK keep-out (``keepout.py``): a target that goes deeper into the table, base or
    fold-back zone than the measured pose is replaced by the measured pose (arm joints; the gripper keeps its
    command). It is reported as a clamp, so the runner records what was really commanded in the model's history."""

    def __init__(self, config: SafetyConfig, keepout: KeepOut, slot: str, measured):
        super().__init__(config)
        self.keepout, self.slot, self.measured = keepout, slot, measured
        self.counts = dict.fromkeys(ZONES, 0)
        self.last_zone = ""
        # Slow zone (off unless set): while the measured arm is within ``slow_zone`` m of the parked other arm, arm
        # joints move at most ``slow_step`` deg per tick from the previous command; run_attempt also lowers the load
        # freeze there. Used where a skill's demos pass close to the other arm (cutter: lid-holding box arm).
        self.slow_zone: float | None = None
        self.slow_step = 0.5
        self.in_slow_zone = False
        self.slow_ticks = 0

    def filter(self, command: np.ndarray, previous: np.ndarray):
        out, report = super().filter(command, previous)
        self.last_zone = ""
        self.in_slow_zone = False
        q = self.measured()
        if q is None:
            return out, report
        s = arm_slice(self.slot)
        arm = slice(s.start, s.start + ARM_DIM - 1)
        refuse, zone = self.keepout.refuse(out[s], q)
        if refuse:
            out = out.copy()
            out[arm] = q[: ARM_DIM - 1]
            report.clamped = np.asarray(report.clamped).copy()
            report.clamped[arm] = True
            self.counts[zone] += 1
            self.last_zone = zone
        if self.slow_zone is not None and float(self.keepout.other_arm_gap(q)) < self.slow_zone:
            self.in_slow_zone = True
            self.slow_ticks += 1
            if previous is not None:
                slow = np.clip(out[arm], previous[arm] - self.slow_step, previous[arm] + self.slow_step)
                if not np.allclose(slow, out[arm]):
                    report.clamped = np.asarray(report.clamped).copy()
                    report.clamped[arm] |= ~np.isclose(slow, out[arm])
                    out = out.copy()
                    out[arm] = slow
        return out, report


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

    def wait(self, allowed: str, prompt: str, auto: str | None = None, tick: Any = None, tick_s: float = 0.2) -> str:
        """Block until one of ``allowed`` (or a stop) arrives; ``auto`` answers at once when stdin is no TTY.
        ``tick()`` runs every ``tick_s`` while waiting (the placement page); its errors are reported, never raised."""
        print(prompt, flush=True)
        if not sys.stdin.isatty() and auto is not None:
            return auto
        next_tick, failed = 0.0, False
        while True:
            key = self.poll()
            if key is not None and (key in allowed or key in ("ctrl-c", "stop-file")):
                return key
            if tick is not None and time.monotonic() >= next_tick:
                next_tick = time.monotonic() + tick_s
                try:
                    tick()
                except Exception as e:  # noqa: BLE001 (a view must never block the operator)
                    if not failed:
                        print(f"(placement view failed: {e!r})", flush=True)
                    failed = True
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
                keepout=np.array([r.get("keepout", "") for r in self.rows]),
            )
        (self.out / "meta.json").write_text(json.dumps(meta, indent=2, default=float))


# ---------------------------------------------------------------------- policy side
def _env_float(name: str) -> float | None:
    value = os.environ.get(name, "").strip()
    return float(value) if value else None


def _env_int(name: str) -> int | None:
    value = os.environ.get(name, "").strip()
    return int(value) if value else None


def read_token(args: argparse.Namespace) -> str | None:
    """Token from a command's stdout (e.g. ``ssh pod cat /workspace/flux_token``), never printed or put in argv."""
    if args.token_cmd:
        cmd = shlex.split(os.path.expanduser(args.token_cmd))
        out = subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=30)
        if out.returncode != 0 or not out.stdout.strip():
            raise SystemExit(f"--token-cmd failed (exit {out.returncode}): {out.stderr.strip()[:200]}")
        return out.stdout.strip()
    return os.environ.get("FLUX_TOKEN")


def replay_take(args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    """The recorded take for --backend replay / --prime-ticks: --replay-skill's dataset and default take unless
    --replay-root / --replay-episode say otherwise, and the task arm's columns (haller naming, mirrored: the robot-right
    box arm is ``left_*``), which must be the skill's arm."""
    root, episode, columns = take_source(args.replay_skill)
    side_columns = "left" if args.task_side == "right" else "right"
    if side_columns != columns:
        raise SystemExit(f"--replay-skill {args.replay_skill} replays the {columns}_* columns: its task arm is not "
                         f"--task-side {args.task_side}")
    return load_take(args.replay_root or root, episode if args.replay_episode is None else args.replay_episode,
                     columns)  # fmt: skip


def build_backend(args: argparse.Namespace) -> Any:
    if args.backend == "replay":
        return ReplayBackend(*replay_take(args), max_step=args.max_step, max_gripper_step=args.max_gripper_step)
    if args.backend == "fake":
        # Scripted stand-in: a constant per-tick drift on each joint and an optional absolute gripper target.
        gripper = None if args.fake_gripper < 0 else args.fake_gripper
        velocity = np.array(args.fake_velocity)
        return FakeBackend(SO101_CONTRACT, velocity=velocity, absolute=gripper, delay_s=args.fake_delay)
    if args.backend == "act":
        # LeRobot ACT baseline (ws4/act): absolute joint chunks through the same FluxBackend interface, so every
        # command still passes the guard, keep-out, envelope, ramps, stall and load freeze.
        from act.backend import ActBackend

        if not args.act_checkpoint:
            raise SystemExit("--backend act needs --act-checkpoint")
        return ActBackend.from_pretrained(args.act_checkpoint, device=args.act_device, n_action_steps=args.act_steps)
    from flux.remote import RemoteBackend

    if getattr(args, "imagine", None):  # opt-in: the server also keeps FLUX's imagined frames (ws4/imagine/README.md)
        flux = RemoteBackend(args.server, args.package, token=read_token(args), timeout_s=args.timeout,
                             imagine=args.imagine)  # fmt: skip
    else:
        flux = RemoteBackend(args.server, args.package, token=read_token(args), timeout_s=args.timeout)
    if getattr(args, "prime_ticks", None):  # the first ticks from a recorded take, then FLUX (replay.py)
        return ReplayBackend(*replay_take(args), max_step=args.max_step, max_gripper_step=args.max_gripper_step,
                             prime=flux, prime_ticks=args.prime_ticks)  # fmt: skip
    return flux


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
def joint_order(keepout: KeepOut, start: np.ndarray, goal: np.ndarray, step: float = 1.0) -> list[int] | None:
    """First order of the arm joints, moved one at a time from ``start`` to ``goal`` in ``step`` deg steps, whose
    every step the keep-out accepts (simulated on the commands), or None. Pan goes first in the search order."""
    for order in itertools.permutations(range(ARM_DIM - 1)):
        q, ok = np.array(start, dtype=float), True
        for j in order:
            while ok and abs(q[j] - goal[j]) > 1e-6:
                nxt = q.copy()
                nxt[j] += np.clip(goal[j] - q[j], -step, step)
                ok = not keepout.refuse(nxt, q)[0]
                q = nxt
            if not ok:
                break
        if ok:
            return list(order)
    return None


def move_to(robot: SingleArmRobot, goal: np.ndarray, guard_cfg: SafetyConfig, deg_per_tick: float, keys: Keys,
            keepout: KeepOut | None = None, fallback: bool = True) -> bool:  # fmt: skip
    """Slow joint-space move to ``goal`` (robot units) through the same limits and, when given, the FK keep-out on
    every step; False if a stop key, a keep-out refusal or a missed goal interrupts it.

    When the straight ramp is refused (after an insert the gripper sits by the box, and its first straight step went
    deeper into the torso zone: 18 Oct), ``fallback`` moves one joint at a time in an order the keep-out accepts at
    every step (``joint_order``); each of those single-joint ramps is checked live like any other."""
    lo, hi = guard_cfg.lower[arm_slice(robot.slot)], guard_cfg.upper[arm_slice(robot.slot)]
    goal = np.clip(goal, lo, hi)
    # Start from the measured pose: after a stall the last command can be far from the arm (seen 2026-10-17).
    cmd = robot.joints()
    step = np.full(ARM_DIM, deg_per_tick)
    step[-1] = 2 * deg_per_tick
    deadline = None
    while True:
        if keys.poll() in STOP_KEYS:
            return False
        nxt = cmd + np.clip(goal - cmd, -step, step)
        if keepout is not None and robot.last_q is not None:
            refuse, zone = keepout.refuse(nxt, robot.last_q)
            if refuse:
                hold_here(robot)
                print(f"ramp refused by the keep-out ({zone}) at {np.round(nxt, 1)}; arm holds", flush=True)
                if not fallback:
                    return False
                here = robot.joints()
                order = joint_order(keepout, here, goal)
                if order is None:
                    print("no joint-by-joint order clears the keep-out either; arm holds", flush=True)
                    return False
                print("going joint by joint instead: " + ", ".join(ARM_JOINTS[j] for j in order), flush=True)
                waypoint = here.copy()
                for j in order:
                    waypoint[j] = goal[j]
                    if not move_to(robot, waypoint.copy(), guard_cfg, deg_per_tick, keys, keepout, fallback=False):
                        return False
                return move_to(robot, goal, guard_cfg, deg_per_tick, keys, keepout, fallback=False)  # gripper
        cmd = nxt
        robot.send_joints(cmd, measure=True)
        if np.allclose(cmd, goal):
            deadline = deadline or time.monotonic() + 5.0  # let the servos settle on the final goal
            err = np.abs(robot.joints() - goal)[: ARM_DIM - 1]
            # The 7.4 V Haller's lift settles about 4 deg short against gravity at P=16, and 6.1 deg after hours of
            # use (18 Oct): it gets 8 deg, the other joints 6.
            if (err < ARRIVE_TOL).all():
                return True
            if time.monotonic() > deadline:
                print(f"goal not reached, error per joint {np.round(err, 1)}", flush=True)
                return False
        time.sleep(1 / 30)


def load_trip(loads: np.ndarray, cap: float, sustain: float = 0.7, peak: float = 0.9, ticks: int = 9) -> int | None:
    """First tick index at which the load freeze trips for per-tick |loads| (T, joints) in 0.1 % of max torque under
    a torque cap ``cap`` (same units), or None."""
    frac = np.abs(np.asarray(loads, dtype=float)) / cap
    run = np.zeros(frac.shape[1], dtype=int)
    for t, f in enumerate(frac):
        run = np.where(f >= sustain, run + 1, 0)
        if (f >= peak).any() or (run >= ticks).any():
            return t
    return None


def hold_here(robot: SingleArmRobot) -> None:
    """Freeze at the MEASURED pose: a blocked joint stops pushing toward a goal it cannot reach."""
    try:
        robot.send_joints(robot.joints(), measure=True)
    except Exception as exc:  # the bus may be the reason we stopped
        print(f"could not set the hold pose: {exc}", flush=True)


def parked_ok(args: argparse.Namespace, other_q: Any, keepout: KeepOut | None = None) -> bool:
    """The other arm must sit where the skill's demos had it: each arm joint inside ``--other-arm-lo`` ..
    ``--other-arm-hi`` (the min / max over the demos) widened by ``--other-arm-tol`` deg; ``--other-arm-expect`` is
    the one-pose form (a tucked arm). Its capsules in the keep-out use the live pose either way; its place in both
    camera views is part of what was learned (the box arm holding the lid open varies a little per run). The runner
    never moves that arm; a refusal names the joint.

    With ``--other-arm-demo-poses`` (pen-arm chain skills, the box arm holding the lid wherever the box skill ended)
    the joint range is only a loose sanity check and ``gate.check`` decides: refuse when an envelope voxel lies
    inside the box arm or more than ``--other-arm-gate-max-below`` of the demo poses would come closer than the
    skill's other-arm clearance."""
    if not _joint_range_ok(args, other_q):
        return False
    poses = getattr(args, "other_arm_demo_poses", None)
    if poses and keepout is not None and other_q is not None:
        import gate

        ok, d = gate.check(keepout, other_q, gate.load_poses(poses), args.other_arm_gate_max_below)
        print(f"box-arm gate: demo min gap {d['demo_min_gap']:.3f} m, {d['demo_below'] * 100:.1f} % of demo poses "
              f"under {d['clearance']} m, envelope voxels inside the box arm {d['envelope_voxels_inside']} -> "
              f"{'ok' if ok else 'REFUSED'}", flush=True)  # fmt: skip
        if not ok:
            print("REFUSED: the box arm's pose is not compatible with this skill's demonstrated motion. Move it "
                  "(orchestrator ramp) and start again.", flush=True)  # fmt: skip
            return False
    return True


def _joint_range_ok(args: argparse.Namespace, other_q: Any) -> bool:
    expect = getattr(args, "other_arm_expect", None)
    lo, hi = getattr(args, "other_arm_lo", None), getattr(args, "other_arm_hi", None)
    if lo is None or hi is None:
        lo = hi = expect
    if lo is None or other_q is None:
        return True
    q, lo, hi = (np.asarray(v, float)[:5] for v in (other_q, lo, hi))
    err = np.maximum(lo - q, q - hi)  # > 0: outside the demonstrated range by that much
    if err.max() <= args.other_arm_tol:
        return True
    j = int(np.argmax(err))
    print(f"REFUSED: the other arm is at {np.round(other_q, 1)}; the skill's demos had it within {np.round(lo, 1)} .. "
          f"{np.round(hi, 1)} ({ARM_JOINTS[j]} {err[j]:.1f} deg outside, limit {args.other_arm_tol:g}). Move it there "
          f"(orchestrator ramp, or by hand with its torque off) and start again.", flush=True)  # fmt: skip
    return False


class KeepoutStop:
    """Persistent pushing into a zone ends the attempt: ``ticks`` keep-out refusals among the last ``window`` policy
    ticks. Ticks that only hold the pose between chunks ("waiting") neither count nor clear. A consecutive count was
    reset by them and by the first, shallower commands of each new chunk, so FLUX's 32-step chunks never reached 45:
    no Haller attempt ever stopped this way (replays of the recorded traces, README)."""

    def __init__(self, ticks: int, window: int):
        self.ticks, self.window, self.zone = ticks, deque(maxlen=window), ""

    def update(self, zone: str, status: str) -> bool:
        if zone or status != "waiting":
            self.window.append(bool(zone))
            self.zone = zone or self.zone
        return sum(self.window) >= self.ticks

    def reason(self) -> str:
        return f"stop (keep-out: {self.zone} refused for {sum(self.window)} ticks of the last {len(self.window)})"


def run_attempt(
    args: argparse.Namespace, runner: DualArmRunner, robot: SingleArmRobot, keys: Keys, task: str, out: Path
) -> dict[str, Any]:
    rec = Recorder(out, video=not args.no_video)
    irec = None
    if getattr(args, "imagine", None) and hasattr(runner.backends.get("policy"), "fetch_imagination"):
        try:  # opt-in; flag off imports nothing from ws4/imagine
            from imagine.attempt import AttemptImagination

            irec = AttemptImagination(out)
        except Exception as exc:
            print(f"imagination off for this attempt: {type(exc).__name__}: {exc}", flush=True)
    target = robot.slot
    other_q = None
    keepout = getattr(runner.guard, "keepout", None)
    if args.other_arm_port and keepout is not None:
        # Re-read the parked arm before every attempt, in case someone moved it (read-only, torque untouched).
        other_q = read_other(args)
        keepout.other_q = tuple(float(v) for v in other_q)
        print(f"parked other arm at {np.round(other_q, 1)}", flush=True)
        if not parked_ok(args, other_q, keepout):
            raise KeyboardInterrupt
    if args.start_pose is not None and args.robot == "so101" and not args.observe:
        print(f"ramping to the start pose {args.start_pose} ...", flush=True)
        ramp_ko = keepout.without_envelope() if keepout is not None else None  # envelope: policy commands only
        if not move_to(robot, np.array(args.start_pose), getattr(runner, "ramp_config", runner.guard.config),
                       args.return_speed, keys, ramp_ko):  # fmt: skip
            raise KeyboardInterrupt
        time.sleep(0.5)
    start_q = robot.joints()
    runner.guard.release()
    runner._command = None  # noqa: SLF001 - start every attempt from the measured pose
    policy = runner.backends.get("policy")
    replaying = isinstance(policy, ReplayBackend)
    if replaying:
        policy.reset()  # the take starts again from this attempt's start pose
    runner.start(target, task)
    t0 = time.monotonic()
    end_reason, period, next_tick = "timeout", 1 / SO101_CONTRACT.fps, time.monotonic()
    chunk_latencies: list[float] = []
    server_latencies: list[float] = []
    n_chunks0 = runner._targets[target].n_chunks  # noqa: SLF001
    guard_ticks, capped0 = 0, robot.capped
    load_run = np.zeros(ARM_DIM - 1, dtype=int)
    load_cap = 10.0 * (args.torque_limit if args.robot == "so101" else 100.0)  # load units at the torque cap
    keep_stop, keep0 = KeepoutStop(args.keepout_ticks, args.keepout_window), dict(getattr(runner.guard, "counts", {}))
    slow0 = getattr(runner.guard, "slow_ticks", 0)
    stall = np.zeros(ARM_DIM - 1, dtype=int)
    # The 7.4 V Haller's lift trails its command by 8-10 deg under load in the air near the 60 % torque cap (18 Oct,
    # 7 false stall stops): --lift-stall-deg gives it its own threshold; the load freeze still catches a real push.
    stall_deg = np.full(ARM_DIM - 1, args.stall_deg)
    if getattr(args, "lift_stall_deg", None) is not None:
        stall_deg[1] = args.lift_stall_deg
    volts: list[float] = []
    healths: list[dict | None] = []
    print(f"attempt running: {task!r} for up to {args.seconds:.0f} s. space/q/Enter = STOP, s = success, f = fail",
          flush=True)  # fmt: skip
    while time.monotonic() - t0 < args.seconds:
        key = keys.poll()
        if key in STOP_KEYS:
            runner.guard.estop()
            end_reason = f"stop ({key!r})"
            hold_here(robot)
            break
        if key in ("s", "f"):
            end_reason = {"s": "operator: success", "f": "operator: fail"}[key]
            break
        try:
            report = runner.step()
        except Exception as exc:  # bus or server failure after retries: freeze, do not crash
            runner.guard.estop()
            end_reason = f"stop (error {type(exc).__name__}: {str(exc)[:120]})"
            hold_here(robot)
            break
        ts = runner._targets[target]  # noqa: SLF001
        if report.status.get(target) == "replanned" and ts.last_response is not None:
            chunk_latencies.append(ts.last_response.latency_s)
            server_latencies.append(ts.last_response.info.get("server_latency_s") or float("nan"))
        guard_ticks += report.guard_intervened
        q = np.array([float(robot.last_raw[f"{j}.pos"]) for j in ARM_JOINTS])
        zone = getattr(runner.guard, "last_zone", "")
        if replaying and policy.done:  # every replayed command was executed: the arm holds the take's last pose
            end_reason = f"replay finished: {policy.label}, holding where the take ends"
            break
        if keep_stop.update(zone, report.status.get(target, "")):
            runner.guard.estop()
            end_reason = keep_stop.reason()
            hold_here(robot)
            break
        # Stall check on the arm joints (not the gripper, which legitimately squeezes a cup): a joint that lags its
        # command by more than --stall-deg for --stall-ticks is blocked (table, cup, its own base). The elbow was
        # damaged on 2026-10-17 after attempts where the wrist folded into the robot's base.
        lag = np.abs(report.command[arm_slice(target)] - q)[: ARM_DIM - 1]
        stall = np.where(lag > stall_deg, stall + 1, 0)
        if (stall >= args.stall_ticks).any():
            j = int(np.argmax(stall))
            runner.guard.estop()
            end_reason = f"stop (stall: {ARM_JOINTS[j]} lagged {lag[j]:.0f} deg for {stall[j]} ticks)"
            hold_here(robot)
            break
        # Load freeze: a command clamped at a limit can sit at low lag while the servo pushes into an end stop,
        # which the stall check never sees. Thresholds are fractions of the active torque cap (the reading cannot
        # exceed it): sustained >= --load-sustain for --load-ticks, or >= --load-peak on any tick.
        load = robot.arm_load()
        if load is not None:
            frac = load / load_cap
            # In the slow zone (close to the other arm) the freeze triggers earlier: first contact, not a push.
            slow = getattr(runner.guard, "in_slow_zone", False)
            sustain = getattr(args, "slow_load_sustain", args.load_sustain) if slow else args.load_sustain
            ticks = getattr(args, "slow_load_ticks", args.load_ticks) if slow else args.load_ticks
            load_run = np.where(frac >= sustain, load_run + 1, 0)  # same rule as load_trip()
            if (frac >= args.load_peak).any() or (load_run >= ticks).any():
                j = int(np.argmax(frac))
                runner.guard.estop()
                end_reason = f"stop (load: {ARM_JOINTS[j]} at {100 * frac[j]:.0f}% of the torque cap)"
                hold_here(robot)
                break
        rec.rows.append({"t": time.monotonic() - t0, "state": q, "command": report.command[arm_slice(target)],
                         "guard": report.guard_intervened, "status": report.status.get(target, ""),
                         "keepout": zone})  # fmt: skip
        rec.frame(robot.last_raw, f"{task} | {report.status.get(target, '')} | t={time.monotonic() - t0:4.1f}s")
        if irec is not None:  # queues frames only (bounded, drops when full); never raises
            irec.tick(len(rec.rows) - 1, robot.last_raw, report.status.get(target) == "replanned", ts.last_response)
        if report.tick % 30 == 0:
            volts.append(robot.min_voltage())
            healths.append(robot.health())
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
        "start_pose_target": args.start_pose,
        "other_arm_q": None if other_q is None else other_q.tolist(),
        "chunks": runner._targets[target].n_chunks - n_chunks0,  # noqa: SLF001
        "chunk_round_trip_s": chunk_latencies, "server_latency_s": server_latencies, "guard_ticks": guard_ticks,
        "lerobot_capped_ticks": robot.capped - capped0, "ticks": len(rec.rows),
        "min_supply_v_per_s": volts, "min_supply_v": min((v for v in volts if v), default=None),
        "health_per_s": healths,
        "keepout_refusals": {z: n - keep0.get(z, 0) for z, n in getattr(runner.guard, "counts", {}).items()},
        "slow_zone_ticks": getattr(runner.guard, "slow_ticks", 0) - slow0,
        "args": {k: v for k, v in vars(args).items() if k != "token_cmd"},
        "checkpoint": (f"{args.act_checkpoint} (LeRobot ACT)" if args.backend == "act" else args.checkpoint)
        if args.backend in POLICY_BACKENDS else f"{args.backend} backend",
    }  # fmt: skip
    if replaying:  # never mistaken for FLUX: the label, the take and how far it got
        meta["replay"] = {**policy.info, "label": policy.label, "ticks_sent": policy.cursor,
                          "handed_over_to_flux": policy.handed_over}  # fmt: skip
        if args.backend == "replay":
            meta["checkpoint"] = f"{policy.label} ({policy.info['root']}, {policy.info['columns']}_* columns)"
    rec.close(meta)
    if irec is not None:  # after meta.json, before scoring; bounded by --imagine-timeout and never raises
        try:
            from imagine.attempt import finish_with_timeout

            finish_with_timeout(irec, runner.backends["policy"], timeout_s=getattr(args, "imagine_timeout", 30.0),
                                render=getattr(args, "imagine_render", False), task=task)  # fmt: skip
        except Exception as exc:
            print(f"imagination not saved: {type(exc).__name__}: {exc}", flush=True)
    return meta


def finish_attempt(args: argparse.Namespace, robot: SingleArmRobot, keys: Keys, guard_cfg: SafetyConfig,
                   rest: np.ndarray, meta: dict[str, Any], keepout: KeepOut | None = None) -> None:  # fmt: skip
    """Bring the arm back slowly to the start pose (or the rest pose), or on request after a stop."""
    if args.observe or args.robot == "fake":
        return
    if meta["end_reason"].startswith("replay finished"):  # the box take ends holding the lid: no return by default
        key = keys.wait("rx", "REPLAY FINISHED, arm holds where the take ends (the lid). r = return slowly to the "
                        "start pose (support the lid first), x = keep holding and quit", auto="x")  # fmt: skip
        if key != "r":
            raise KeyboardInterrupt
    elif meta["end_reason"].startswith("stop"):
        key = keys.wait("rx", "STOPPED, arm holds. r = return slowly to rest, x = keep holding and quit")
        if key != "r":
            raise KeyboardInterrupt
    goal = np.array(args.start_pose) if args.start_pose is not None else rest
    print(f"returning slowly to {'the start pose' if args.start_pose is not None else 'rest'} ...", flush=True)
    if not move_to(robot, goal, guard_cfg, args.return_speed, keys, keepout):
        print("return interrupted; arm holds", flush=True)
        raise KeyboardInterrupt


def ask_line(keys: Keys, prompt: str, valid: tuple[str, ...]) -> str:
    """Typed answer + Enter, asked again until valid. Single cbreak keys at this prompt went missing on the Haller
    runs (the prompt printed, then "saved" with nothing logged, cause never reproduced), so the terminal is back in
    normal line mode here, keys pressed during the attempt are dropped first, and Ctrl-C cannot skip the question."""
    import termios

    fd = sys.stdin.fileno()
    cbreak = termios.tcgetattr(fd)
    termios.tcflush(fd, termios.TCIFLUSH)
    if keys._old is not None:
        termios.tcsetattr(fd, termios.TCSADRAIN, keys._old)
    try:
        while True:
            try:
                answer = input(f"{prompt}, then Enter: ").strip().lower()
            except EOFError:
                answer = ""
            if answer in valid:
                return answer
            print(f"    not understood: {answer!r}. Type one of {', '.join(valid)} and press Enter.", flush=True)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, cbreak)
        keys.pressed.clear()  # a Ctrl-C typed at the question must not quit the next prompt


def ask_result(keys: Keys, meta: dict[str, Any]) -> dict[str, Any] | None:
    if not sys.stdin.isatty():
        return None
    digits = tuple(str(i) for i in range(len(STEPS) + 1))
    while True:
        answer = ask_line(keys, f">>> steps done 0..{len(STEPS)} ({'; '.join(STEPS)}), or skip", (*digits, "skip"))
        if answer != "skip":
            break
        if ask_line(keys, ">>> really skip logging this attempt (false start only)? yes/no", ("yes", "no")) == "yes":
            print(">>> not logged", flush=True)
            return None
    steps = int(answer)
    answer = ask_line(keys, ">>> human interventions (0..9; your stop key counts as 1)", tuple("0123456789"))
    print(f">>> logging {steps}/{len(STEPS)} steps, {answer} interventions", flush=True)
    return {"steps_done": steps, "interventions": int(answer)}


def log_trial(args: argparse.Namespace, task: str, meta: dict[str, Any], result: dict[str, Any], out: Path) -> None:
    sys.path.insert(0, str(WS4 / "eval"))
    import trial_log

    failure = "" if result["steps_done"] == len(STEPS) else f"stopped before: {STEPS[result['steps_done']]}"
    replay = meta.get("replay")  # never mixed in with the policy rows
    condition, variant = args.condition, args.variant
    if getattr(args, "backend", "") == "replay":  # a recorded take, no policy: its own condition and variant
        condition, variant = "scripted", f"scripted-replay-ep{replay['episode']}"
    elif replay:  # --prime-ticks: a FLUX attempt that started from a recorded take
        variant = f"{args.variant}+primed-ep{replay['episode']}-{args.prime_ticks}"
    trial = trial_log.Trial(
        task=args.trial_task, condition=condition, variant=variant, checkpoint=meta["checkpoint"],
        steps_total=len(STEPS), steps_done=result["steps_done"],
        success=result["steps_done"] == len(STEPS) and result["interventions"] == 0,
        interventions=result["interventions"], layout=meta.get("layout") or args.layout, failure=failure,
        notes=f"{args.note + '; ' if args.note else ''}prompt={task!r}; guidance {args.guidance_note}; "
        f"{replay['label'] + ' (' + replay['root'] + '); ' if replay else ''}"
        f"{f'action horizon {args.action_steps} per chunk; ' if getattr(args, 'action_steps', None) else ''}"
        f"{f'lift stall {args.lift_stall_deg} deg; ' if getattr(args, 'lift_stall_deg', None) else ''}"
        f"{f'chunk prefetch at {args.prefetch_ticks} commands left; ' if getattr(args, 'prefetch_ticks', None) else ''}"
        f"{'start ' + meta['layout_ref'] + '; ' if meta.get('layout_ref') else ''}"
        f"safety: {args.max_step} deg/tick guard, LeRobot cap "
        f"{args.max_relative_target} deg, torque limit {args.torque_limit:.0f}%; "
        f"min supply {meta.get('min_supply_v')} V; keep-out refusals {meta.get('keepout_refusals')}; "
        f"end={meta['end_reason']}; chunks={meta['chunks']}; "
        f"guard_ticks={meta['guard_ticks']}",
        video=str((out / "video.mp4").relative_to(WS4.parent)) if (out / "video.mp4").exists() else "",
        operator=args.operator,
    )  # fmt: skip
    trial = trial_log.append(Path(os.environ.get("WS4_TRIAL_LOG", trial_log.DEFAULT_LOG)), trial)  # env: tests only
    print(f"logged {args.trial_task}/{args.variant} attempt {trial.attempt}: {trial.steps_done}/{trial.steps_total}",
          flush=True)  # fmt: skip


def attempt_plan(args: argparse.Namespace) -> list[str]:
    """Layout of each attempt: ``--layouts A B --per-layout 3`` gives A A A B B B, otherwise ``--layout`` repeated
    ``--attempts`` times."""
    if args.layouts:
        return [layout for layout in args.layouts for _ in range(args.per_layout)]
    return [args.layout] * args.attempts


def start_check(robot: SingleArmRobot, ref_img: np.ndarray, label: str, out: Path) -> dict[str, Any]:
    """The scene as the attempt starts, next to its reference (start_check.jpg), and how far the background sits
    from the reference: a camera bumped since recording shows up as a shift of more than a few px."""
    import cv2

    out.mkdir(parents=True, exist_ok=True)
    live = robot.get_observation()[IMG_HEAD]
    cv2.imwrite(str(out / "start_check.jpg"), cv2.cvtColor(composite(live, ref_img, label), cv2.COLOR_RGB2BGR))
    try:
        dx, dy, rot = ref_shift(live, ref_img)
    except cv2.error:
        print("start check: base plate not aligned (camera moved a lot, or the view is blocked)", flush=True)
        return {}
    print(f"start check: scene background shift {dx:+.1f} px, {dy:+.1f} px, {rot:+.2f} deg vs the reference",
          flush=True)  # fmt: skip
    return {"start_shift_px": [round(dx, 1), round(dy, 1)], "start_rot_deg": round(rot, 2)}


def align_at_home(args: argparse.Namespace, robot: SingleArmRobot, keys: Keys, ref_head: np.ndarray,
                  ref_wrist: np.ndarray, ref_name: str, period_s: float = 1.0) -> bool:  # fmt: skip
    """--align: with the arm held at home, show the runner's own head and wrist frames against the home reference
    (align_station.py's offset, scale, brightness and hint, page on --placement-port) until Enter; False on Ctrl-C/q.
    The station (robot and table together) is moved by hand meanwhile; the arm keeps holding its home goal."""
    import align_station as al
    from placement import PlacementView

    if not sys.stdin.isatty():
        print("align: no terminal, skipped", flush=True)
        return True
    view = PlacementView(args.placement_port)
    print(f"align page: {view.url} vs {ref_name}. Move the station until the wrist row lines up "
          f"(good: |offset| < {al.GOOD['offset_px']:.0f} px, scale {al.GOOD['scale'][0]}..{al.GOOD['scale'][1]}). "
          "Enter = done, go on to the home check; q / Ctrl-C = stop and park", flush=True)  # fmt: skip
    try:
        while True:
            obs = robot.get_observation()
            head, wrist = obs[IMG_HEAD], obs[WRIST_KEY[robot.slot]]
            off = al.view_offset(ref_wrist, wrist)
            bright = float(wrist.mean() / max(float(ref_wrist.mean()), 1.0))
            advice, ok = al.hint(off, bright)
            nums = ("wrist: no overlap with the reference" if off is None else
                    f"wrist: offset {off['dx']:+.0f} px, {off['dy']:+.0f} px, scale {off['scale']:.2f}, "
                    f"{off['inliers']} inliers")  # fmt: skip
            lines = [f"{nums}, brightness {bright:.2f}", ("OK  " if ok else "->  ") + advice,
                     "Enter in the terminal = done; q / Ctrl-C = stop and park"]
            view.show(al.panel(ref_head, head, ref_wrist, wrist, lines))
            print(time.strftime("%H:%M:%S"), lines[0], "|", lines[1], flush=True)
            deadline = time.monotonic() + period_s
            while True:  # keys are polled at least once per frame
                key = keys.poll()
                if key == "\n":
                    return True
                if key in ("q", "ctrl-c", "stop-file"):
                    return False
                if time.monotonic() >= deadline:
                    break
                time.sleep(0.05)
    finally:
        view.close()


def check_at_home(args: argparse.Namespace, runner: DualArmRunner, robot: SingleArmRobot, keys: Keys,
                  session: Path) -> bool:  # fmt: skip
    """Ramp to the start pose (as an attempt does), then the runner's live frames and joints vs the at-home
    references (``home_check.py``). Saves home_check.jpg / .json in the session folder; True = go on."""
    import cv2
    import home_check as hc

    keepout = getattr(runner.guard, "keepout", None)
    if args.other_arm_port and keepout is not None:
        keepout.other_q = tuple(float(v) for v in read_other(args))
        print(f"home check: parked other arm at {np.round(keepout.other_q, 1)}", flush=True)
        if not parked_ok(args, keepout.other_q, keepout):
            return False  # before any motion: no attempts, the arm is parked again below
    if args.start_pose is not None and args.robot == "so101" and not args.observe:
        print(f"home check: ramping to the start pose {args.start_pose} ...", flush=True)
        ramp_ko = keepout.without_envelope() if keepout is not None else None
        if not move_to(robot, np.array(args.start_pose), getattr(runner, "ramp_config", runner.guard.config),
                       args.return_speed, keys, ramp_ko):  # fmt: skip
            return False
        time.sleep(1.0)
    session.mkdir(parents=True, exist_ok=True)
    if getattr(args, "align", False):
        ref_head, ref_wrist, _, ref_name = hc.load_refs(Path(args.home_check), args.home_check_take)
        if not align_at_home(args, robot, keys, ref_head, ref_wrist, ref_name):
            return False
    try:  # the arm holds at home here: a failed analysis is reported, the operator still decides
        ref_head, ref_wrist, ref_state, ref_name = hc.load_refs(Path(args.home_check), args.home_check_take)
        obs = robot.get_observation()
        q = robot.joints()
        ok, lines, img = hc.home_check(obs[IMG_HEAD], obs[WRIST_KEY[robot.slot]], q, ref_head, ref_wrist, ref_state,
                                       side=args.task_side)  # fmt: skip
        cv2.imwrite(str(session / "home_check.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        (session / "home_check.json").write_text(json.dumps({"ok": ok, "reference": ref_name, "lines": lines,
                                                             "q": np.round(q, 2).tolist()}, indent=1))  # fmt: skip
        print(f"home check vs {ref_name}: {'PASS' if ok else 'CHECK'} (image {session / 'home_check.jpg'})",
              flush=True)  # fmt: skip
    except Exception as e:  # noqa: BLE001
        ok, lines = False, [f"home check could not run: {e!r}"]
    for line in lines:
        print(f"  {line}", flush=True)
    if not sys.stdin.isatty():
        return True
    if ok:
        return keys.wait("\n", ">>> home check PASS: Enter = continue to the attempts, Ctrl-C = stop and park") == "\n"
    return ask_line(keys, ">>> home check did NOT pass (above): go = run the attempts anyway, park = stop and park",
                    ("go", "park")) == "go"  # fmt: skip


# ---------------------------------------------------------------------- main
def build_guard(args: argparse.Namespace, slot: str, calib_path: Path | None, rest: np.ndarray,
                ramp: bool = False) -> SafetyConfig:  # fmt: skip
    """Joint limits and per-tick steps in robot units.

    Policy (``ramp=False``): soft limits, the calibrated range shrunk by ``--limit-margin`` (5 deg) at each end, so no
    policy command ever reaches a mechanical end stop. When the policy starts from ``--start-pose`` the limits are
    NOT widened to the rest pose (E's fold drove the Haller to within 1 to 2 deg of its stops otherwise). Without a
    start pose the policy starts at rest, so the limits must include it or the first tick would yank the arm.
    Scripted ramps (``ramp=True``): ``--ramp-margin`` (1 deg) from the stops, widened to include the rest pose.
    """
    cfg = SafetyConfig()
    if calib_path is not None:
        margin = args.ramp_margin if ramp else args.limit_margin
        lower, upper = limits_from_lerobot_calibration(calib_path, margin_deg=margin)
        if ramp or args.start_pose is None:
            lower, upper = np.minimum(lower, rest), np.maximum(upper, rest)
        cfg = cfg.with_arm_limits(slot, lower, upper)
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
    p.add_argument("--scene-fourcc", default="MJPG", help="'none' = OpenCV default (what the haller recorder uses)")
    p.add_argument("--wrist-fourcc", default="MJPG", help="'none' = OpenCV default (what the haller recorder uses)")
    p.add_argument("--parity-side", choices=("left", "right"), default=None,
                   help="old haller_ws recorder transforms (rotation, gains, V4L2); the 2026-10-17 VR recording uses "
                        "none of them, so leave this off for it")  # fmt: skip
    p.add_argument("--other-arm-port", default=None, help="parked other arm, read before each attempt (Haller)")
    p.add_argument("--other-arm-id", default="haller_left", help="calibration id of the parked other arm")
    p.add_argument("--other-arm-expect", type=float, nargs=ARM_DIM, default=None,
                   help="the other arm's (tucked) pose in the skill's demos: refuse to start when it is elsewhere")
    p.add_argument("--other-arm-lo", type=float, nargs=ARM_DIM, default=None,
                   help="per-joint min of the other arm over the skill's demos (with --other-arm-hi)")
    p.add_argument("--other-arm-hi", type=float, nargs=ARM_DIM, default=None, help="per-joint max, see --other-arm-lo")
    p.add_argument("--other-arm-tol", type=float, default=5.0, help="deg allowed outside the pose / range")
    p.add_argument("--hold-other-arm", action="store_true",
                   help="take over the other arm at --other-arm-port and hold it where it is (goal = present, then "
                   "torque on); released with torque still on at the end (the chain: box arm holding the lid)")
    p.add_argument("--fake-other-q", type=float, nargs=ARM_DIM, default=None,
                   help="--robot fake with --hold-other-arm: the pose the fake holder holds")
    p.add_argument("--other-arm-demo-poses", default=None,
                   help="npz of the skill's demo poses: geometric box-arm gate at start (gate.py)")
    p.add_argument("--other-arm-gate-max-below", type=float, default=0.01,
                   help="gate: largest fraction of demo poses allowed under the other-arm clearance")
    p.add_argument("--start-pose", type=float, nargs=ARM_DIM, default=None,
                   help="ramp here before every attempt (Haller demos: 0 0 0 0 0 100)")  # fmt: skip
    p.add_argument("--allow-torque-off", action="store_true",
                   help="allow switching torque off to (re)write calibration or registers; arm must be at rest")
    p.add_argument("--keepout-config", default=None, help="workspace JSON (keepout.py); default red_follower's")
    p.add_argument("--wrist-rotate", type=int, choices=(0, 90, 180, 270), default=0)
    p.add_argument("--slot", choices=("left", "right"), default="left", help="runner slot; no effect on the model")
    p.add_argument("--backend", choices=("fake", *POLICY_BACKENDS, "replay"), default="fake",
                   help="remote = FLUX (or act/serve.py) server; act = LeRobot ACT checkpoint in this process; "
                        "replay = one recorded take as a scripted policy (replay.py), not FLUX; logged with condition "
                        "scripted, variant scripted-replay-ep<N>")
    p.add_argument("--replay-skill", default="box", choices=sorted(DEFAULT_TAKES),
                   help="--backend replay / --prime-ticks: whose takes (skills.py: dataset, arm, default take)")
    p.add_argument("--replay-episode", type=int, default=None,
                   help=f"the take (default per skill: {DEFAULT_TAKES}, see replay.py)")
    p.add_argument("--replay-root", default=None, help="the take's LeRobot dataset (default: the skill's, cached)")
    p.add_argument("--prime-ticks", type=int, default=None,
                   help="--backend remote: replay the first K ticks of --replay-episode (rounded up to the action "
                        "horizon), then FLUX takes over; logged with variant <variant>+primed-ep<N>-<K>")
    p.add_argument("--act-checkpoint", default=None, help="--backend act: the checkpoint's pretrained_model folder")
    p.add_argument("--act-device", default="auto", help="--backend act: cuda, cpu or auto")
    p.add_argument("--act-steps", type=int, default=None,
                   help="--backend act: commands executed per chunk before re-planning (default: the checkpoint's)")
    p.add_argument("--action-steps", type=int, default=_env_int("WS4_ACTION_STEPS"),
                   help="remote/fake: commands executed per chunk before re-planning (default: the package's 32)")
    p.add_argument("--align", action="store_true",
                   help="with --home-check: after the ramp home, a live alignment page vs the home reference until "
                        "Enter (move the station onto the recording's spot)")
    p.add_argument("--prefetch-ticks", type=int, default=_env_int("WS4_PREFETCH_TICKS"),
                   help="request the next chunk when this many commands are left, executing meanwhile (0/unset = "
                        "off: the arm holds for each round trip; env WS4_PREFETCH_TICKS)")
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
    p.add_argument("--max-relative-target", type=float, default=20.0,
                   help="LeRobot cap vs measured pos, deg; 5 starved the shoulder of torque at P=16")  # fmt: skip
    p.add_argument("--limit-margin", type=float, default=5.0, help="policy: deg inside the calibrated range")
    p.add_argument("--ramp-margin", type=float, default=1.0, help="scripted ramps: deg inside the calibrated range")
    p.add_argument("--return-speed", type=float, default=1.0, help="deg per tick when returning to rest")
    p.add_argument("--torque-limit", type=float, default=60.0, help="percent of max servo torque, arm joints")
    p.add_argument("--stall-deg", type=float, default=8.0, help="arm joint lag behind command = stall, deg")
    p.add_argument("--load-sustain", type=float, default=0.7, help="load freeze: fraction of the torque cap ...")
    p.add_argument("--load-ticks", type=int, default=9, help="... held this many ticks (9 = 0.3 s)")
    p.add_argument("--load-peak", type=float, default=0.9, help="load freeze: fraction of the cap on any tick")
    p.add_argument("--slow-zone", type=float, default=None,
                   help="m: while the arm is this close to the parked other arm, step --slow-step and freeze at "
                   "--slow-load-sustain for --slow-load-ticks (off by default)")
    p.add_argument("--slow-step", type=float, default=0.5, help="deg per tick on the arm joints in the slow zone")
    p.add_argument("--slow-load-sustain", type=float, default=0.5, help="load freeze in the slow zone: fraction")
    p.add_argument("--slow-load-ticks", type=int, default=6, help="load freeze in the slow zone: ticks (6 = 0.2 s)")
    p.add_argument("--keepout-ticks", type=int, default=45,
                   help="keep-out refusals within --keepout-window policy ticks that end an attempt")
    p.add_argument("--keepout-window", type=int, default=60, help="policy ticks (holds between chunks excluded)")
    p.add_argument("--no-keepout", action="store_true", help="disable the FK keep-out (fake-arm tests only)")
    p.add_argument("--stall-ticks", type=int, default=15, help="consecutive stalled ticks before freezing (15 = 0.5 s)")
    # Also from the environment, so eval_skill.sh sessions can set them without a script change (18 Oct).
    p.add_argument("--lift-stall-deg", type=float, default=_env_float("WS4_LIFT_STALL_DEG"),
                   help="shoulder-lift stall threshold, deg (default: --stall-deg; env WS4_LIFT_STALL_DEG)")
    p.add_argument("--offset", type=float, nargs=ARM_DIM, default=None, help="model = scale * robot + offset")
    p.add_argument("--scale", type=float, nargs=ARM_DIM, default=None,
                   help="model = scale * robot + offset; must match the package's training data (D: 1 -1 1 1 -1 1)")
    p.add_argument("--variant", default="A0")
    p.add_argument("--checkpoint", default=CHECKPOINT, help="what the served package is, for the trial log "
                   "(a LoRA: its package path and microstep)")
    p.add_argument("--layout", default="")
    p.add_argument("--layouts", nargs="+", default=None, help="layout sequence, e.g. A B C D (overrides --attempts)")
    p.add_argument("--per-layout", type=int, default=1, help="attempts per layout with --layouts")
    p.add_argument("--start-at", type=int, default=1, help="resume: skip to this attempt of the sequence (1-based)")
    p.add_argument("--layout-ref", default=None, help="dir of <layout>_ep<NN>_head.jpg start frames (placement page)")
    p.add_argument("--placement-port", type=int, default=8899)
    p.add_argument("--home-check", default=None, metavar="REF_DIR",
                   help="before the attempts: ramp to --start-pose and compare live frames and joints with these "
                   "at-home references (frames/reference_2026-10-17_night)")
    p.add_argument("--home-check-take", type=int, default=None, help="reference take in --home-check (default last)")
    p.add_argument("--task-side", choices=("left", "right"), default="left",
                   help="robot side of the task arm (left = /dev/haller_arm_right, right = /dev/haller_arm_left)")
    p.add_argument("--condition", choices=("zero_shot", "post_training"), default="zero_shot")
    p.add_argument("--note", default="", help="free text prepended to every logged trial's notes")
    p.add_argument("--trial-task", default="stack_cups", help=f"one of {sorted(TASK_STEPS)} or any with --steps")
    p.add_argument("--steps", nargs="+", default=None, help="graded steps of the task, in order")
    p.add_argument("--operator", default="oscar")
    p.add_argument("--no-log", action="store_true", help="never write to ws4/eval/trials.jsonl")
    p.add_argument("--no-video", action="store_true")
    p.add_argument("--imagine", choices=("lazy", "async", "sync"), default=None,
                   help="also save FLUX's imagined frames per chunk (remote backend; ws4/imagine/README.md). lazy: the "
                   "server decodes them after the attempt while the arm holds. Not for graded eval rows")
    p.add_argument("--imagine-timeout", type=float, default=30.0, help="max seconds to fetch them after an attempt")
    p.add_argument("--imagine-render", action="store_true", help="also write imagination/imagined_vs_actual.mp4")
    p.add_argument("--yes", action="store_true", help="start attempts without waiting for Enter")
    p.add_argument("--home", action="store_true", help="red arm: move slowly to its saved rest pose, then torque off")
    p.add_argument("--park-pose", type=float, nargs=ARM_DIM, default=None,
                   help="only move slowly to this pose (station keep-out on), then torque off (Haller folded rest)")
    args = p.parse_args(argv)
    global STEPS, OTHER_ARM_READER
    STEPS = tuple(args.steps) if args.steps else TASK_STEPS.get(args.trial_task, STEPS)
    tasks = args.task or ["stack the cups"]
    import logging

    logging.getLogger().setLevel(logging.ERROR)  # LeRobot logs every capped goal at WARNING, 30 times a second
    if args.scene_cam == "blank" and args.backend in (*POLICY_BACKENDS, "replay") and not args.no_log:
        raise SystemExit("a blank scene camera is out of distribution: such attempts must not be logged (--no-log)")

    holder = None
    if args.hold_other_arm:
        if not args.other_arm_port:
            raise SystemExit("--hold-other-arm needs --other-arm-port")
        if args.robot == "so101":
            holder = OtherArmHolder(args.other_arm_port, args.other_arm_id, args.torque_limit)
        else:
            holder = FakeOtherArmHolder(args.other_arm_port, args.other_arm_id, args.torque_limit, args.fake_other_q)
        OTHER_ARM_READER = holder.read
    calib_path = None
    if args.robot == "so101":
        if not args.scene_cam:
            raise SystemExit("--scene-cam is required on the real arm")
        so101 = connect_so101(args)
        calib_path = so101.calibration_fpath
    else:
        so101 = FakeSO101()
    robot = SingleArmRobot(so101, args.slot, send=not args.observe, max_relative_target=args.max_relative_target)
    if args.parity_side:
        robot.transforms = station_parity(args)[0]
    rest = robot.joints()
    guard_cfg = build_guard(args, args.slot, calib_path, rest)
    ramp_cfg = build_guard(args, args.slot, calib_path, rest, ramp=True)
    s = arm_slice(args.slot)
    print(f"guard limits {np.round(guard_cfg.lower[s], 1)} .. {np.round(guard_cfg.upper[s], 1)}, "
          f"max step {guard_cfg.max_step[s]}, LeRobot max_relative_target {args.max_relative_target}",
          flush=True)  # fmt: skip
    # model = scale * robot + offset per joint (the LoRA base's convention, e.g. variant D: --scale 1 -1 1 1 -1 1)
    calibration = ArmCalibration(scale=np.array(args.scale) if args.scale else np.ones(ARM_DIM),
                                 offset=np.array(args.offset) if args.offset else np.zeros(ARM_DIM))  # fmt: skip
    target = per_arm_target(args.slot, "policy", scene=CameraView(IMG_HEAD), calibration=calibration)
    backend = build_backend(args)
    if args.backend == "replay" or args.prime_ticks:
        if args.scale or args.offset:
            raise SystemExit("a replay sends the recorded robot-unit commands: no --scale / --offset")
        if args.prefetch_ticks:  # WS4_PREFETCH_TICKS may be set for FLUX rows; prefetching would skip recorded steps
            print(f"chunk prefetch off for the replay (was {args.prefetch_ticks}): it counts whole chunks", flush=True)
            args.prefetch_ticks = 0
        if args.prime_ticks and args.backend != "remote":
            raise SystemExit("--prime-ticks hands over to FLUX: it needs --backend remote")
        info = backend.info
        print(f"backend: {backend.label} ({info['root']}, {info['columns']}_* columns, {info['frames']} frames), "
              f"time-warped to {args.max_step:g} deg/tick: {'NOT FLUX' if args.backend == 'replay' else 'then FLUX'}",
              flush=True)  # fmt: skip
    if args.backend == "replay":
        # The take's own first frames, when the kit has them: the placement overlay and the home check use them.
        # --backend replay only: FLUX runs (also --prime-ticks) keep the last held-out take, so home checks and --align
        # stay comparable across rows (so101-zeroshot).
        own = {d: reference_frame(d, backend.info["episode"]) for d in {args.layout_ref, args.home_check} if d}
        for path in sorted({str(v) for v in own.values() if v}):
            print(f"placement reference: {path} (the replayed take's first head frame)", flush=True)
        if args.home_check and own.get(args.home_check) and args.home_check_take is None:
            args.home_check_take = backend.info["episode"]
    if args.action_steps is not None and args.backend in ("remote", "fake", "replay"):
        # The runner reads the horizon from the backend's contract (DualArmRunner._accept keeps n_action_steps). For a
        # replay it only sets how often the take is asked for: the commands sent are the same.
        backend.contract = replace(backend.contract, n_action_steps=args.action_steps)
        print(f"action horizon: {args.action_steps} of {backend.contract.chunk_size} commands per chunk", flush=True)
    if args.no_keepout and args.robot == "so101" and not args.observe:
        raise SystemExit("the FK keep-out stays on for the real arm")
    keepout = KeepOut.load(Path(args.keepout_config)) if args.keepout_config else KeepOut.load()
    if args.robot == "so101" and not keepout.calibrated:
        print("NOTE: table height not calibrated (keepout.py --touch); using the UNVERIFIED default", flush=True)
    guard = SafetyGuard(guard_cfg) if args.no_keepout else KeepOutGuard(
        guard_cfg, keepout, args.slot, lambda: robot.last_q)
    if args.slow_zone is not None and isinstance(guard, KeepOutGuard):
        guard.slow_zone, guard.slow_step = args.slow_zone, args.slow_step
        if isinstance(backend, ReplayBackend):  # warp at the slow step there, so the guard never clips the path
            zone, slow, fast = args.slow_zone + SLOW_MARGIN, args.slow_step, args.max_step
            backend.step_at = lambda q: slow if float(keepout.other_arm_gap(q)) < zone else fast
    runner = DualArmRunner(robot, [target], {"policy": backend}, guard, async_inference=True,
                           prefetch_ticks=args.prefetch_ticks or 0)  # fmt: skip
    if args.prefetch_ticks:
        print(f"chunk prefetch: next request when {args.prefetch_ticks} commands are left", flush=True)
    runner.ramp_config = ramp_cfg  # scripted ramps (start pose, return, park) use the ramp limits
    if args.home or args.park_pose is not None:
        home = np.array(args.park_pose if args.park_pose is not None else json.loads(REST_FILE.read_text())["rest"])
        print(f"moving slowly from {np.round(rest, 1)} to the saved rest {np.round(home, 1)}", flush=True)
        guard_cfg = build_guard(args, args.slot, calib_path, home, ramp=True)
        ko = keepout.without_envelope() if not args.no_keepout else None
        if ko is not None and args.other_arm_port:
            ko.other_q = tuple(float(v) for v in read_arm_joints(args.other_arm_port, args.other_arm_id))
        with Keys() as keys:
            ok = move_to(robot, home, guard_cfg, args.return_speed, keys, ko)
        print("at rest" if ok else "NOT at rest", flush=True)
        if ok and args.robot == "so101":
            disconnect_so101(so101)
            print("torque off")
        return
    print(f"rest pose {np.round(rest, 1)}", flush=True)
    session = RUNS / datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        with Keys() as keys:
            if args.backend in POLICY_BACKENDS:
                warm_up(backend, robot, target, sorted(set(tasks)))
            layouts = attempt_plan(args)
            if args.home_check and not check_at_home(args, runner, robot, keys, session):
                layouts = []  # Ctrl-C at the home check: no attempts, park below
            refs = load_refs(Path(args.layout_ref)) if args.layout_ref else {}
            view = PlacementView(args.placement_port) if refs and sys.stdin.isatty() else None
            if view is not None:
                print(f"placement page: {view.url} (open it in a browser)", flush=True)
            seen: dict[str, int] = {}
            for i, layout in enumerate(layouts):
                k = seen[layout] = seen.get(layout, -1) + 1  # attempt index within this layout
                if i + 1 < args.start_at:
                    continue
                task = tasks[i % len(tasks)]
                ref = refs[layout][k % len(refs[layout])] if layout in refs else None
                if args.backend == "replay" and args.layout_ref:  # place the objects as at the take's start
                    ref = reference_frame(args.layout_ref, backend.info["episode"]) or ref
                ref_img = read_rgb(ref) if ref is not None else None
                label = f"attempt {i + 1}/{len(layouts)}: layout {layout}" + (f", reference {ref.stem}" if ref else "")
                if not args.yes:
                    tick = None
                    if view is not None and ref_img is not None:
                        def tick(ref_img: np.ndarray = ref_img, label: str = label) -> None:
                            view.update(robot.get_observation()[IMG_HEAD], ref_img, label)
                    key = keys.wait("\n ", f"[{label}] prompt {task!r}: Enter = start, Ctrl-C = quit", auto="\n",
                                    tick=tick)  # fmt: skip
                    if key in ("ctrl-c", "stop-file"):
                        break
                out = session / f"attempt{i + 1:02d}"
                start = start_check(robot, ref_img, label, out) if ref_img is not None else {}
                meta = run_attempt(args, runner, robot, keys, task, out)
                meta.update(layout=layout, layout_ref=ref.stem if ref else "", **start)
                (out / "meta.json").write_text(json.dumps(meta, indent=2, default=float))
                # Score the attempt while the arm still holds where it ended, so a quit ("x") never loses it.
                scored = args.backend in (*POLICY_BACKENDS, "replay")  # a replay: condition "scripted" (log_trial)
                result = None if args.no_log or not scored else ask_result(keys, meta)
                if result is not None:
                    meta["result"] = result
                    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=float))
                    log_trial(args, task, meta, result, out)
                print(f"saved {out}", flush=True)
                ko = getattr(runner.guard, "keepout", None)
                finish_attempt(args, robot, keys, ramp_cfg, rest, meta, ko.without_envelope() if ko else None)
            ko = getattr(runner.guard, "keepout", None)
            if args.start_pose is not None and args.robot == "so101" and not args.observe \
                    and not getattr(so101, "ws4_launch_torque", True):
                # The arm was resting with torque off at launch: park it there again before torque goes off.
                print("parking slowly at the launch pose ...", flush=True)
                move_to(robot, rest, ramp_cfg, args.return_speed, keys, ko.without_envelope() if ko else None)
    except KeyboardInterrupt:
        print("quit requested", flush=True)
    finally:
        runner.close()
        if args.robot == "so101":
            at_rest = np.allclose(robot.joints()[:5], rest[:5], atol=8.0)
            if args.observe:
                disconnect_so101(so101, torque_off=False)  # observe never changes torque
            elif at_rest and not getattr(so101, "ws4_launch_torque", False):
                disconnect_so101(so101)
            else:
                # Dropping an extended arm onto the cups is worse than holding it: leave torque on.
                for camera in so101.cameras.values():
                    camera.disconnect()
                so101.bus.disconnect(disable_torque=False)
                print("arm still holds with torque on (it was holding at launch, or is not at its launch pose). "
                      "Support it before switching the power off.", flush=True)  # fmt: skip
        if holder is not None:
            holder.close()
            OTHER_ARM_READER = None
    print(f"session folder {session}")


if __name__ == "__main__":
    main()
