"""Demo backend: the green Haller's arms through ``ws4/single_arm/run_single_arm.py`` (read-only).

One or both arms, as the vocabulary's ``station.arms_available`` says. The pen-only vocabulary drives the robot-LEFT
arm and only reads the parked robot-right arm for the keep-out; the final chain (skills.yaml) drives both: the right
arm opens the box, then the left arm fills it. Runner slots are robot sides: ``left`` = /dev/haller_arm_right
(calibration haller_right), ``right`` = /dev/haller_arm_left (haller_left).

The orchestrator reuses the single-arm runner's robot side instead of its attempt loop:
- ``connect_so101``: connect without LeRobot's ``connect()`` (no torque drop on an arm that holds a pose), the
  wrist cameras rotated 180 deg and the head unrotated, OpenCV's default fourcc (camera parity with the VR
  recording, single_arm/README.md "Haller"). The head camera belongs to the left arm's connection.
- ``SingleArmRobot`` per arm (bus retries, LeRobot's ``max_relative_target`` cap); ``PairRobot`` merges two of them
  into the runner's 12-D ``[left | right]`` interface.
- ``build_guard`` per arm (policy and ramp limits from each arm's calibration) and ``KeepOut`` per arm, applied by
  ``ArmsKeepOutGuard`` (the same rule as run_single_arm's ``KeepOutGuard``, for every connected arm). With both arms
  connected, each arm's ``other_arm`` zone follows the other arm's LIVE pose every tick.
- on every stop, ``hold_here``'s freeze at the MEASURED pose for the arm joints, but each gripper keeps its last command
  (the box arm holds the lid open by its tab); ``warm_up`` (pre-encode prompts on the server), and the torque-safe
  shutdown of its ``main``.
The per-tick stall, keep-out and load-freeze stops are rebuilt in ``monitors.py`` with the same defaults, and scripted
ramps get the runner's ramp guard (``build_guard(ramp=True)`` limits, ``KeepOut.without_envelope``).

What eval_skill.sh adds per skill comes from the skill's kit (ws4/single_arm/skills/<kit>/, ``keepout:`` in the
vocabulary): its workspace.json while its policy runs, the slow zone of its skill.json ``runner_args``
(``ArmsKeepOutGuard.slow``), and its ``other_arm_gate`` as ``start_gate`` (gate.check with the other arm's live pose).
An arm that holds the lid (``finish: hold``, ``--hold-arm``) keeps its torque at close, as run_single_arm's
OtherArmHolder; a ramp the keep-out refuses goes joint by joint (``ramp_order`` = run_single_arm.joint_order).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from flux.contract import ARM_DIM, ARM_JOINTS, IMG_HEAD, OBS_STATE, WRIST_KEY, arm_slice
from flux.safety import SafetyConfig, SafetyGuard

from . import WS4
from .monitors import KeepOutMonitor, LoadMonitor, StallMonitor
from .skills import resolve_keepout

SINGLE_ARM = WS4 / "single_arm"
CALIBRATION = SINGLE_ARM / "calibration"
SLOT = "left"  # the pen-only task arm is the robot's left arm
OTHER_ARM_PORT = "/dev/haller_arm_left"  # the parked ROBOT-right arm (pen-only vocabulary)
OTHER_ARM_ID = "haller_left"
DEFAULT_PORT = "/dev/haller_arm_right"  # operator-frame name: it is the ROBOT's left arm
DEFAULT_ROBOT_ID = "haller_right"  # LeRobot calibration so_follower/haller_right.json
DEFAULT_HEAD_CAM = "/dev/haller_cam_mast"
DEFAULT_WRIST_CAM = "/dev/v4l/by-path/pci-0000:00:14.0-usb-0:7.2:1.0-video-index0"
# Station zones only (no demonstrated envelope): a skill's envelope comes from its own `keepout:` in skills.yaml.
DEFAULT_KEEPOUT = CALIBRATION / "haller_right_workspace_noenv.json"
RIGHT_PORT = "/dev/haller_arm_left"  # the ROBOT's right arm
RIGHT_ROBOT_ID = "haller_left"
# by USB path: both wrist cams report serial SN0001 (haller_ws hmi/backend/config.flux-real.yaml, wrist_left)
RIGHT_WRIST_CAM = "/dev/v4l/by-path/pci-0000:00:14.0-usb-0:7.3:1.0-video-index0"


@dataclass
class OtherArmPose:
    """Outcome of one refresh. ``block`` set: the attempt must not start (fail closed, so101-zeroshot 2026-10-17)."""

    note: str  # short tag for the attempt's trial_log notes
    message: str
    block: str = ""


def port_users(port: str) -> list[int]:
    """PIDs of OTHER processes that have ``port`` open (scans /proc; a busy bus must never get a second reader).
    Only processes of the same user are visible; another user's process holding the port is not detected."""
    try:
        real = os.path.realpath(port)
    except OSError:
        return []
    users = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == os.getpid():
            continue
        fd_dir = f"/proc/{pid}/fd"
        try:
            for fd in os.listdir(fd_dir):
                if os.path.realpath(os.path.join(fd_dir, fd)) == real:
                    users.append(int(pid))
                    break
        except OSError:  # gone, or not ours to inspect
            continue
    return users


def import_single_arm():
    """``run_single_arm`` imports ``keepout`` as a top-level module, so its folder goes on ``sys.path``."""
    if str(SINGLE_ARM) not in sys.path:
        sys.path.insert(0, str(SINGLE_ARM))
    import run_single_arm  # noqa: PLC0415
    from keepout import KeepOut  # noqa: PLC0415

    return run_single_arm, KeepOut


class ArmsKeepOutGuard(SafetyGuard):
    """``SafetyGuard`` plus the FK keep-out of every connected arm, by run_single_arm's ``KeepOutGuard`` rule: a
    target deeper into a zone than the measured pose is replaced by the measured pose (arm joints; the gripper keeps
    its command) and reported as a clamp. ``other_q`` of each arm's keep-out is the other arm's live measured pose when
    both are connected, else the pose read from the parked arm's bus (``fixed_other_q``)."""

    def __init__(self, config: SafetyConfig, keepouts: dict[str, Any], measured: dict[str, Callable[[], Any]]):
        super().__init__(config)
        self.keepouts = dict(keepouts)  # arm -> KeepOut in force now (station, skill envelope, or ramp copy)
        self.measured = measured  # arm -> its last measured joints (None before the first read)
        self.fixed_other_q: dict[str, tuple[float, ...] | None] = {}
        self.zones = dict.fromkeys(keepouts, "")
        self.counts: dict[str, int] = {}
        self.last_zone = ""
        # Slow zone per arm, run_single_arm's KeepOutGuard rule (set while a skill with --slow-zone in its kit's
        # runner_args runs its policy, the cutter): within slow_zone m of the other arm (measured, capsule surfaces) the
        # arm joints move at most slow_step deg per tick from the previous command, and LoadMonitor freezes earlier.
        self.slow: dict[str, dict] = {}
        self.in_slow_zone = dict.fromkeys(keepouts, False)
        self.slow_ticks = dict.fromkeys(keepouts, 0)

    @property
    def keepout(self) -> Any:  # single-arm view, as run_single_arm's KeepOutGuard
        return next(iter(self.keepouts.values()))

    def slow_load(self, arm: str) -> tuple[float, int] | None:
        """(sustain fraction, ticks) of the load freeze while ``arm`` is in its slow zone, else None."""
        slow = self.slow.get(arm)
        return (slow["slow_load_sustain"], slow["slow_load_ticks"]) if slow and self.in_slow_zone.get(arm) else None

    def filter(self, command: np.ndarray, previous: np.ndarray):
        out, report = super().filter(command, previous)
        self.last_zone = ""
        for arm, ko in self.keepouts.items():
            self.zones[arm], self.in_slow_zone[arm] = "", False
            q = self.measured[arm]()
            if ko is None or q is None:
                continue
            other = "right" if arm == "left" else "left"
            if other in self.measured:
                oq = self.measured[other]()
                ko.other_q = None if oq is None else tuple(float(v) for v in oq)
            else:
                ko.other_q = self.fixed_other_q.get(arm)
            s = arm_slice(arm)
            joints = slice(s.start, s.start + ARM_DIM - 1)
            refuse, zone = ko.refuse(out[s], q)
            if refuse:
                out = out.copy()
                out[joints] = np.asarray(q)[: ARM_DIM - 1]
                report.clamped = np.asarray(report.clamped).copy()
                report.clamped[joints] = True
                self.counts[zone] = self.counts.get(zone, 0) + 1
                self.zones[arm] = self.last_zone = zone
            slow = self.slow.get(arm)
            if slow and float(ko.other_arm_gap(np.asarray(q, dtype=float))) < slow["slow_zone"]:
                self.in_slow_zone[arm] = True
                self.slow_ticks[arm] += 1
                if previous is not None:
                    step = slow["slow_step"]
                    capped = np.clip(out[joints], previous[joints] - step, previous[joints] + step)
                    if not np.allclose(capped, out[joints]):
                        report.clamped = np.asarray(report.clamped).copy()
                        report.clamped[joints] |= ~np.isclose(capped, out[joints])
                        out = out.copy()
                        out[joints] = capped
        return out, report


class PairRobot:
    """``RobotInterface`` over one ``SingleArmRobot`` per connected arm: 12-D state from both, the head camera from the
    arm that carries it, each arm's own wrist camera; each arm is sent only its own slot of the 12-D command."""

    def __init__(self, arms: dict[str, Any], head_arm: str = "left", head_key: str = "",
                 wrist_keys: dict | None = None):
        self.arms, self.head_arm = arms, head_arm
        self.head_key, self.wrist_keys = head_key, wrist_keys or {}

    def get_observation(self) -> dict[str, Any]:
        state, obs = None, {}
        for arm, robot in self.arms.items():
            o = robot.get_observation()
            if state is None:
                state = np.asarray(o[OBS_STATE], dtype=float).copy()
            state[arm_slice(arm)] = np.asarray(o[OBS_STATE], dtype=float)[arm_slice(arm)]
            obs[self.wrist_keys[arm]] = o[self.wrist_keys[arm]]
            if arm == self.head_arm:
                obs[self.head_key] = o[self.head_key]
        obs[OBS_STATE] = state
        return obs

    def send_action(self, action: np.ndarray) -> None:
        for robot in self.arms.values():
            robot.send_action(action)


@dataclass
class Arm:
    name: str  # runner slot = robot side
    so101: Any  # LeRobot SOFollower, or run_single_arm.FakeSO101
    robot: Any  # run_single_arm.SingleArmRobot
    rest: np.ndarray  # measured pose at launch
    policy_limits: SafetyConfig  # build_guard(ramp=False): calibrated range minus --limit-margin, rest NOT included
    ramp_limits: SafetyConfig  # build_guard(ramp=True): --ramp-margin from the stops, widened to include rest
    station_keepout: Any  # station zones and the other arm, no demonstrated envelope


@dataclass
class HallerRig:
    sar: Any  # the run_single_arm module
    arms: dict[str, Arm]
    robot: Any  # what the runner drives: the one SingleArmRobot, or a PairRobot
    guard: ArmsKeepOutGuard
    monitors: list
    real: bool
    observe: bool
    other_arm_port: str = OTHER_ARM_PORT
    other_arm_id: str = OTHER_ARM_ID
    read_other_arm: bool = True
    other_arm_max_age_s: float = 180.0
    clock: Callable[[], float] = time.monotonic
    keepout_cls: Any = None
    _pose_t: float | None = None  # when the parked arm's pose was last read successfully
    _ramping: dict[str, bool] = field(default_factory=dict)
    _skill_keepouts: dict[str, Any] = field(default_factory=dict)  # config path -> KeepOut, loaded once
    # Arms that hold something with torque on and keep holding when the session closes (run_single_arm
    # --hold-other-arm): ``hold_arms`` were handed over holding at launch (--hold-arm; never driven in this session);
    # ``holding`` also has every arm whose last step ended with ``finish: hold`` (the box arm and the lid).
    hold_arms: tuple[str, ...] = ()
    holding: set[str] = field(default_factory=set)
    ramp_step: float = 1.0  # deg per simulated step of the joint-by-joint ramp search (= the ramp speed per tick)
    unguarded: bool = False  # --unguarded: policies run with no keep-out and no slow zone (ramps keep the station's)

    # -------------------------------------------------------------------- single-arm views (pen-only vocabulary)
    @property
    def primary(self) -> Arm:
        return self.arms.get(SLOT) or next(iter(self.arms.values()))

    @property
    def so101(self) -> Any:
        return self.primary.so101

    @property
    def rest(self) -> np.ndarray:
        return self.primary.rest

    @property
    def both(self) -> bool:
        return len(self.arms) == 2

    # -------------------------------------------------------------------- ramp / policy mode per arm
    def skill_keepout(self, arm: str, skill: Any = None) -> Any:
        """The keep-out an arm's policy runs under: the skill's own config (with its demonstrated envelope) or the
        arm's station zones."""
        path = getattr(skill, "keepout", "") if skill is not None and arm in getattr(skill, "arms", ()) else ""
        if not path:
            return self.arms[arm].station_keepout
        full = resolve_keepout(path)  # the skill's kit: ws4/single_arm/skills/<kit>/workspace.json
        if str(full) not in self._skill_keepouts:
            if not full.exists():  # KeepOut.load would fall back to the red arm's defaults
                raise FileNotFoundError(f"keep-out {full} of skill {getattr(skill, 'name', '?')} does not exist")
            self._skill_keepouts[str(full)] = self.keepout_cls.load(full)
        return self._skill_keepouts[str(full)]

    def set_ramp(self, arm: str, on: bool, skill: Any = None) -> None:
        """Scripted ramps (to the start pose, parking, the ``home`` step) run under what run_single_arm gives its
        ramps: the ramp joint limits (the folded rest lies outside the policy's soft limits) and the keep-out without
        the demonstrated envelope (it bounds what the POLICY may command; unfolding from the folded rest leaves it for
        a moment). Otherwise: the policy limits and the skill's keep-out, with the slow zone of its kit's runner_args
        (run_single_arm applies it to policy commands only, not to its ramps). Either way the arm is driven again, so
        it no longer counts as holding."""
        if arm not in self.arms:
            return
        ko = self.skill_keepout(arm, skill)
        self.guard.keepouts[arm] = ko.without_envelope() if on and hasattr(ko, "without_envelope") else ko
        own = skill is not None and arm in getattr(skill, "arms", ())
        self.guard.slow[arm] = dict(getattr(skill, "runner", None) or {}) if own and not on else {}
        if self.unguarded and not on:
            self.guard.keepouts[arm], self.guard.slow[arm] = None, {}
        self._ramping[arm] = on
        self.guard.config = self.limits()
        if arm not in self.hold_arms:
            self.holding.discard(arm)

    def limits(self) -> SafetyConfig:
        """Ramp limits for a ramping arm and for an arm handed over holding (they include its launch pose, so the
        hold command passes unchanged), else the policy limits."""
        return merge_limits({a: (x.ramp_limits if self._ramping.get(a) or a in self.hold_arms else x.policy_limits)
                             for a, x in self.arms.items()})

    def mark_holding(self, arm: str) -> None:
        """``finish: hold`` passed on this arm: it keeps holding (torque on) until a later run drives it again."""
        self.holding.add(arm)

    def ramp_order(self, arm: str, here: np.ndarray, goal: np.ndarray) -> list[int] | None:
        """run_single_arm.move_to's fallback after a keep-out refusal on a ramp: the first order of the arm joints,
        moved one at a time from ``here`` to ``goal``, that the keep-out in force (the ramp copy, with the other arm's
        live capsules) accepts at every simulated step; None if there is none."""
        return self.sar.joint_order(self.guard.keepouts[arm], np.asarray(here, float), np.asarray(goal, float),
                                    step=self.ramp_step)

    # -------------------------------------------------------------------- start gate (pen-arm chain skills)
    def start_gate(self, skill: Any) -> tuple[str, str]:
        """(refusal, verdict) before a skill whose kit records ``other_arm_gate`` (green, cutter, pen_lid), with the
        other arm's LIVE pose, as eval_skill.sh runs run_single_arm.parked_ok: the loose joint range (box-skill end
        poses and the skill's recorded hold, already widened by 5 deg: tolerance 0) and then gate.check (refuse when
        an envelope voxel lies inside the box arm, or more than ``max_below`` of the skill's demo poses would come
        closer than its other-arm clearance). ``refusal`` empty: the skill may start."""
        gate_info = getattr(skill, "start_gate", None)
        if not gate_info:
            return "", ""
        arm = skill.arms[0]
        other = "right" if arm == "left" else "left"
        if other in self.arms:
            q = np.asarray(self.arms[other].robot.joints(), dtype=float)  # fresh read: where the box arm holds now
        else:
            fixed = self.guard.fixed_other_q.get(arm)
            if fixed is None:
                return f"start gate: the {other} arm's pose is unknown", ""
            q = np.asarray(fixed, dtype=float)
        pose = np.round(q[:5], 1).tolist()
        lo, hi = np.asarray(gate_info["joint_lo"], float)[:5], np.asarray(gate_info["joint_hi"], float)[:5]
        err = np.maximum(lo - q[:5], q[:5] - hi)  # > 0: outside the range by that much (run_single_arm rule)
        if err.max() > 0.0:
            j = int(np.argmax(err))
            return (f"start gate REFUSED {skill.name}: the {other} arm holds at {pose}, {ARM_JOINTS[j]} "
                    f"{err[j]:.1f} deg outside the range {np.round(lo, 1).tolist()} .. {np.round(hi, 1).tolist()} "
                    "(box-skill end poses and this skill's recorded hold, +5 deg)"), ""
        import gate  # ws4/single_arm/gate.py (on sys.path since import_single_arm)

        ok, d = gate.check(self.skill_keepout(arm, skill), q, gate.load_poses(gate_info["demo_poses"]),
                           float(gate_info["max_below"]))
        verdict = (f"demo min gap {d['demo_min_gap']:.3f} m, {d['demo_below'] * 100:.1f} % of demo poses under "
                   f"{d['clearance']} m (max {float(gate_info['max_below']) * 100:.0f} %), envelope voxels inside the "
                   f"{other} arm {d['envelope_voxels_inside']}")
        if not ok:
            return (f"start gate REFUSED {skill.name}: the {other} arm holds at {pose}, not compatible with this "
                    f"skill's demonstrated motion ({verdict})"), verdict
        return "", f"start gate ok for {skill.name} with the {other} arm at {pose}: {verdict}"

    # -------------------------------------------------------------------- the parked arm (pen-only vocabulary)
    def refresh_other_arm(self) -> OtherArmPose:
        """Pose of the parked robot-right arm for the keep-out's other_arm zone, read-only (keepout.read_arm_joints:
        no register writes, torque untouched). The midline alone misses some reaches toward that arm, so an unknown
        pose blocks the attempt instead of running without the zone:
        - the port is held by another process: that arm may not be parked at all (someone may be driving it);
        - the read raises: the last pose read in this session is reused if younger than ``other_arm_max_age_s``.
        With both arms connected the zone follows the other arm's live pose and nothing is read here."""
        if self.both:
            return OtherArmPose("live", "both arms connected: the other_arm zone follows the other arm's live pose")
        if not self.real:
            return OtherArmPose("fake arm", "other arm not read (fake arm)")
        arm = self.primary.name
        if not self.read_other_arm:
            self.guard.fixed_other_q[arm] = None
            return OtherArmPose("DISABLED (--no-other-arm)", "other arm not read (--no-other-arm): other_arm zone off")
        users = port_users(self.other_arm_port)
        if users:
            self.guard.fixed_other_q[arm], self._pose_t = None, None
            why = f"{self.other_arm_port} is held by pid {users}: the parked arm may be moving"
            return OtherArmPose("blocked: port held", why, f"other arm: {why}; stop that process, then Release and "
                                "Resume")
        try:
            from keepout import read_arm_joints  # noqa: PLC0415

            q = read_arm_joints(self.other_arm_port, self.other_arm_id)
        except Exception as exc:
            err = f"could not read {self.other_arm_port} ({type(exc).__name__}: {exc})"
            known = self.guard.fixed_other_q.get(arm) is not None
            age = None if self._pose_t is None or not known else self.clock() - self._pose_t
            if age is not None and age <= self.other_arm_max_age_s:
                return OtherArmPose(f"cached {age:.0f} s", f"{err}; reusing its pose read {age:.0f} s ago")
            self.guard.fixed_other_q[arm] = None
            stale = "no earlier pose" if age is None else f"last pose is {age:.0f} s old"
            return OtherArmPose("blocked: read failed", f"{err}, {stale}",
                                f"other arm: {err}, {stale}; check the cable, then Release and Resume")
        self.guard.fixed_other_q[arm], self._pose_t = tuple(float(v) for v in q), self.clock()
        return OtherArmPose("read", f"other arm pose for the keep-out: {np.round(q, 1).tolist()}")

    # -------------------------------------------------------------------- stops, warm-up, shutdown
    def hold(self, reason: str = "") -> None:
        """Freeze every arm's joints at their MEASURED pose, as run_single_arm.hold_here, so a blocked joint stops
        pushing, but keep each gripper's last command: a gripper that holds something keeps holding it (the box arm
        holds the lid open by its tab during the pen skills; the lid closes if it lets go)."""
        if self.real and not self.observe:
            for arm in self.arms.values():
                freeze_keep_grip(arm.robot)

    def warm_up(self, backend: Any, target: Any, prompts: list[str]) -> None:
        arm = target.name.split(":")[0]
        self.sar.warm_up(backend, self.arms.get(arm, self.primary).robot, target, prompts)

    def torque_off_at_close(self, arm: Arm) -> bool:
        """run_single_arm.main's rule: torque off only if it was off at launch and the arm is back at its launch pose.
        Never for an arm that holds something (``holding``, --hold-arm): a box arm handed over holding the lid sits at
        its launch pose with torque off at launch, and switching it off would drop the lid and the arm onto the box
        (run_single_arm's OtherArmHolder releases with torque on for the same reason)."""
        if self.observe or arm.name in self.holding or arm.name in self.hold_arms:
            return False
        at_rest = np.allclose(arm.robot.joints()[:5], arm.rest[:5], atol=8.0)
        return at_rest and not getattr(arm.so101, "ws4_launch_torque", False)

    def close(self) -> str:
        """Same rule as run_single_arm.main, per arm: switch torque off only if it was off at launch and the arm is
        back at its launch pose; otherwise keep holding (dropping an extended arm onto the box is worse)."""
        if not self.real:
            return "fake arm" + ("s" if self.both else "")
        out = []
        for arm in self.arms.values():
            if self.observe:
                self.sar.disconnect_so101(arm.so101, torque_off=False)
                out.append(f"{arm.name}: observe, torque unchanged")
            elif self.torque_off_at_close(arm):
                self.sar.disconnect_so101(arm.so101)
                out.append(f"{arm.name}: at its launch pose, torque off")
            else:
                for camera in arm.so101.cameras.values():
                    camera.disconnect()
                arm.so101.bus.disconnect(disable_torque=False)
                what = "keeps holding (finish: hold or --hold-arm)" if arm.name in self.holding | set(
                    self.hold_arms) else "still holds"
                out.append(f"{arm.name}: {what} with torque on; support it before switching the power off")
        return "; ".join(out)


def freeze_keep_grip(robot: Any) -> None:
    try:
        q = robot.joints()
        if robot.last_sent is not None:
            q[ARM_DIM - 1] = robot.last_sent[ARM_DIM - 1]
        robot.send_joints(q, measure=True)
    except Exception as exc:  # the bus may be the reason we stopped
        print(f"could not set the hold pose of the {robot.slot} arm: {exc}", flush=True)


def merge_limits(per_arm: dict[str, SafetyConfig]) -> SafetyConfig:
    """One 12-D SafetyConfig from per-arm build_guard results (each sets only its own slot)."""
    first = next(iter(per_arm.values()))
    lower, upper = np.array(first.lower, dtype=float), np.array(first.upper, dtype=float)
    step = np.array(first.max_step, dtype=float)
    for arm, cfg in per_arm.items():
        s = arm_slice(arm)
        lower[s], upper[s], step[s] = np.asarray(cfg.lower)[s], np.asarray(cfg.upper)[s], np.asarray(cfg.max_step)[s]
    return SafetyConfig(lower, upper, step)


def arm_settings(cfg: Any, arm: str) -> dict[str, Any]:
    if arm == "left":
        return {"port": cfg.port, "robot_id": cfg.robot_id, "wrist_cam": cfg.wrist_cam, "head_cam": cfg.head_cam,
                "keepout": cfg.keepout_config, "fake_start": cfg.fake_start, "fake_calibration": cfg.fake_calibration}
    wrist = cfg.right_wrist_cam or RIGHT_WRIST_CAM
    return {"port": cfg.right_arm_port, "robot_id": cfg.right_robot_id, "wrist_cam": wrist, "head_cam": "blank",
            "keepout": cfg.right_keepout_config, "fake_start": cfg.right_fake_start,
            "fake_calibration": cfg.right_fake_calibration}


def build(cfg: Any, station: Any = None) -> HallerRig:
    """Connect every arm the station uses (``station.arms_available``; the robot-left arm alone without a station)."""
    sar, keepout_cls = import_single_arm()
    arm_names = tuple(getattr(station, "arms_available", None) or (SLOT,))
    real = cfg.robot == "so101"
    if real and not cfg.head_cam:
        raise SystemExit("--head-cam is required on the real arm")
    arms: dict[str, Arm] = {}
    for name in arm_names:
        hw = arm_settings(cfg, name)
        home = station.home_for(name) if station is not None and hasattr(station, "home_for") else None
        ns = guard_namespace(cfg, home, hw)
        if real:
            so101 = sar.connect_so101(ns)
            calib = so101.calibration_fpath
        else:
            so101, calib = sar.FakeSO101(hw["fake_start"]), hw["fake_calibration"] or None
        robot = sar.SingleArmRobot(so101, name, send=not cfg.observe, max_relative_target=cfg.max_relative_target)
        rest = robot.joints()
        policy_limits, ramp_limits = guard_limits(sar, ns, calib, rest, name)
        if hw["keepout"]:
            station_ko = keepout_cls.load(Path(hw["keepout"]))
        else:
            station_ko = keepout_cls.for_haller(name)
        station_ko = station_ko.without_envelope() if hasattr(station_ko, "without_envelope") else station_ko
        if real and not station_ko.calibrated:
            print(f"NOTE: {name} arm: table height not calibrated (keepout.py --touch); using the UNVERIFIED default",
                  flush=True)
        arms[name] = Arm(name, so101, robot, rest, policy_limits, ramp_limits, station_ko)
    if len(arms) == 1:
        robot_iface = next(iter(arms.values())).robot
    else:
        robot_iface = PairRobot({a: x.robot for a, x in arms.items()}, "left", IMG_HEAD, WRIST_KEY)
    guard = ArmsKeepOutGuard(merge_limits({a: x.policy_limits for a, x in arms.items()}),
                             {a: x.station_keepout for a, x in arms.items()},
                             {a: (lambda r=x.robot: r.last_q) for a, x in arms.items()})
    load_cap = 10.0 * (cfg.torque_limit if real else 100.0)  # Present_Load units at the torque cap (run_attempt)
    monitors: list = []
    lift_stall = getattr(cfg, "lift_stall_deg", None)  # resolved by the session (CLI, WS4_LIFT_STALL_DEG, haller 25)
    unguarded = bool(getattr(cfg, "unguarded", False))
    for a, x in arms.items():
        if unguarded:  # only the ramps' keep-out (joint-by-joint fallback); policies have no keep-out to trip it
            monitors.append(KeepOutMonitor(guard, cfg.keepout_ticks, slot=a, window=cfg.keepout_window))
            continue
        monitors += [StallMonitor(a, cfg.stall_deg, cfg.stall_ticks, lift_stall_deg=lift_stall),
                     KeepOutMonitor(guard, cfg.keepout_ticks, slot=a, window=cfg.keepout_window),
                     LoadMonitor(x.robot.arm_load, load_cap, cfg.load_sustain, cfg.load_peak, cfg.load_ticks, slot=a,
                                 slow=lambda a=a: guard.slow_load(a))]
    hold_arms = tuple(a for a in getattr(cfg, "hold_arms", ()) or () if a in arms)
    for a in hold_arms + (tuple(arms) if unguarded else ()):
        # as run_single_arm's OtherArmHolder: its only command is its own pose, so no keep-out acts on it (one would
        # make the goal follow a sagging arm); the other arm's keep-out still sees it live as capsules. Unguarded: no
        # arm has a keep-out until its first ramp (set_ramp)
        guard.keepouts[a] = None
    rig = HallerRig(sar, arms, robot_iface, guard, monitors, real, cfg.observe, cfg.other_arm_port, cfg.other_arm_id,
                    cfg.read_other_arm, cfg.other_arm_max_age_s, keepout_cls=keepout_cls, hold_arms=hold_arms,
                    holding=set(hold_arms), ramp_step=cfg.home_speed, unguarded=unguarded)
    guard.config = rig.limits()
    return rig


def guard_namespace(cfg: Any, home_pose: Any = None, hw: dict | None = None) -> argparse.Namespace:
    """The run_single_arm flags its connect and build_guard read. ``start_pose`` set: the policy always starts there,
    so its soft limits are not widened to the rest pose (build_guard)."""
    hw = hw or arm_settings(cfg, SLOT)
    return argparse.Namespace(
        port=hw["port"], robot_id=hw["robot_id"], scene_cam=hw["head_cam"], wrist_cam=hw["wrist_cam"], scene_rotate=0,
        wrist_rotate=cfg.wrist_rotate, scene_fourcc=cfg.scene_fourcc, wrist_fourcc=cfg.wrist_fourcc, parity_side=None,
        max_relative_target=cfg.max_relative_target, observe=cfg.observe, allow_torque_off=cfg.allow_torque_off,
        torque_limit=cfg.torque_limit, limit_margin=cfg.limit_margin, ramp_margin=cfg.ramp_margin,
        max_step=cfg.max_step, max_gripper_step=cfg.max_gripper_step,
        start_pose=None if home_pose is None else [float(v) for v in home_pose],
    )


def guard_limits(sar: Any, ns: argparse.Namespace, calib: Any, rest: np.ndarray, arm: str = SLOT) -> tuple[Any, Any]:
    """(policy, ramp) SafetyConfigs as run_single_arm.main builds them."""
    return sar.build_guard(ns, arm, calib, rest), sar.build_guard(ns, arm, calib, rest, ramp=True)
