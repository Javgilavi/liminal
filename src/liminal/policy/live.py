"""Bounded single-arm FLUX rollout through Haller's existing motor owner.

The default is observation-only inference. Execution requires a verified rig
profile and explicit --execute --engage. No serial port is opened here.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import json
import math
import shlex
import sys
import time
import urllib.request
import uuid
from collections import deque
from pathlib import Path
from typing import Any

import yaml

from liminal.core.types import SO101_JOINTS
from liminal.policy.command_feedback import committed_target, rebase_action
from liminal.policy.live_diagnostics import ChunkDiagnostics
from liminal.policy.live_tuning import inference_settings, rollout_duration, worker_arguments
from liminal.safety.stall import StallMonitor
from liminal.safety.supervisor import JointLimits, SafetySupervisor


def vector(value: Any) -> list[float]:
    out = [float(x) for x in value]
    if len(out) != len(SO101_JOINTS) or not all(math.isfinite(x) for x in out):
        raise ValueError("expected six finite joint values")
    return out


def selected_state(message: dict, side: str, *, single_arm: str | None = None) -> list[float]:
    names, values = message["state_names"], message["state"]
    if len(names) != len(values) or len(set(names)) != len(names):
        raise ValueError("ambiguous state names")
    named = dict(zip(names, values, strict=True))
    if single_arm == side and set(named) == set(SO101_JOINTS):
        return vector(named[joint] for joint in SO101_JOINTS)
    return vector(named[f"{side}_{joint}"] for joint in SO101_JOINTS)


def transform(values: list[float], cfg: dict, *, inverse: bool = False) -> list[float]:
    scales, offsets = vector(cfg["joint_scale"]), vector(cfg["joint_offset"])
    if any(abs(x) != 1 for x in scales):
        raise ValueError("joint scales must be +1 or -1; do not rescale degrees")
    return vector((x - b) / a if inverse else a * x + b
                  for x, a, b in zip(vector(values), scales, offsets, strict=True))


def legacy_shoulder_elbow(cfg: dict) -> dict:
    """Opt into the offline-tested legacy convention without changing calibration."""
    if cfg.get("normalization_profile") == "hf_space_native":
        raise ValueError("HF Space normalization requires native coordinates; cannot add the legacy mapping")
    if vector(cfg["joint_scale"]) != [1] * 6 or vector(cfg["joint_offset"]) != [0] * 6:
        raise ValueError("legacy shoulder/elbow mapping requires an identity profile; refusing a second mapping")
    return {**cfg, "coordinate_mapping": "experimental_legacy_shoulder_elbow",
            "joint_scale": [1, -1, 1, 1, 1, 1], "joint_offset": [0, 90, 90, 0, 0, 0]}


def execution_gate(cfg: dict, state: list[float], stats: dict) -> list[str]:
    if cfg.get("camera_verified") is not True or cfg.get("units_verified") is not True:
        raise RuntimeError("motion blocked: verify the selected wrist camera and joint units first")
    model = transform(state, cfg)
    outside = [name for name, x, lo, hi in zip(SO101_JOINTS, model, stats["q01"], stats["q99"], strict=True)
               if not float(lo) - 20 <= x <= float(hi) + 20]
    # Quantiles describe normalization, not physical limits or a required pose.
    # They cannot establish a sign/offset conversion for an unseen robot.
    return outside


def pose_limits(cells: dict, initial: list[float]) -> tuple[list[float], list[float]]:
    lower, upper = [], []
    for joint, q in zip(SO101_JOINTS, vector(initial), strict=True):
        hard_lo, hard_hi = float(cells[joint]["min"]), float(cells[joint]["max"])
        if not (math.isfinite(hard_lo) and math.isfinite(hard_hi) and hard_lo < hard_hi):
            raise ValueError(f"invalid calibrated limits for {joint}")
        lo, hi = hard_lo, hard_hi
        if not lo <= q <= hi:
            raise RuntimeError(f"{joint} is outside allowed limits: {q:.1f}, allowed [{lo:.1f}, {hi:.1f}]")
        lower.append(lo)
        upper.append(hi)
    return lower, upper


def http(route: str, body: dict | None = None) -> dict:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request("http://127.0.0.1:8000" + route, data=data,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=3) as response:
        return json.load(response)


class WireArm:
    """SafetySupervisor transport proxy; only the selected side can be serialized."""

    num_joints = len(SO101_JOINTS)

    def __init__(self, side: str):
        self.name = side
        self.current: list[float] = []
        self.pending: list[float] | None = None

    def read_state(self) -> list[float]:
        return vector(self.current)

    def send_action(self, positions: list[float]) -> None:
        self.pending = vector(positions)

    def stop(self) -> None:
        self.pending = None


class LiveRig:
    def __init__(self, cfg: dict, output: Path):
        self.cfg, self.output = cfg, output
        self.history: deque[dict] = deque(maxlen=8)
        self.latest: dict | None = None
        self.last_observation = 0.0
        self.telemetry: dict = {}
        self.last_telemetry = 0.0
        self.wrist: str | None = None
        self.last_wrist = 0.0
        self.command: list[float] | None = None
        self.command_seq = -1
        self.observed_command_seq = -1
        self.run_id: str | None = None
        self.pending_command: tuple[int, asyncio.Future] | None = None
        self.observation_changed = asyncio.Event()
        self.error: str | None = None
        self.stall = StallMonitor(**cfg.get("stall_detection", {}))

    def event(self, kind: str, **values) -> None:
        with (self.output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps({"time": time.time(), "type": kind, **values}, allow_nan=False) + "\n")

    def diagnostic(self, kind: str, **values) -> None:
        self.event(kind, **values)
        if self.cfg.get("verbose"):
            print(f"[debug:{kind}] " + json.dumps(values, allow_nan=False), flush=True)

    def snapshot(self) -> dict:
        now = time.monotonic()
        arm = self.telemetry.get("arms", {}).get(self.cfg["arm"], {})
        return {"measured": self.latest.get("state") if self.latest else None,
                "committed": self.command, "command_seq": self.command_seq,
                "tracking_error": ([round(c - q, 3) for c, q in zip(self.command, self.latest["state"], strict=True)]
                                   if self.command is not None and self.latest else None),
                "mode": arm.get("mode"), "torque": arm.get("torque"),
                "load": {j: cell.get("effort") for j, cell in arm.get("joints", {}).items()},
                "age_ms": {key: round(1000 * (now - timestamp), 1) if timestamp else None
                           for key, timestamp in (("observation", self.last_observation),
                                                  ("telemetry", self.last_telemetry), ("wrist", self.last_wrist))}}

    async def send_command(self, writer, seq: int, target: list[float]) -> dict:
        if self.pending_command is not None:
            raise RuntimeError("previous command has not been acknowledged")
        future = asyncio.get_running_loop().create_future()
        self.pending_command = (seq, future)
        try:
            writer.write((json.dumps({"type": "policy_action", "seq": seq,
                                     "t_ms": int(time.time() * 1000), "run_id": self.run_id,
                                     "action": {self.cfg["arm"]: dict(zip(SO101_JOINTS, target, strict=True))}})
                          + "\n").encode())
            await writer.drain()
            try:
                return await asyncio.wait_for(future, 0.5)
            except TimeoutError as exc:
                raise RuntimeError(f"Haller did not acknowledge command {seq} within 0.5s") from exc
        finally:
            self.pending_command = None

    async def wait_observed(self, seq: int) -> None:
        try:
            async with asyncio.timeout(0.5):
                while self.observed_command_seq < seq:
                    self.check()
                    self.observation_changed.clear()
                    await self.observation_changed.wait()
        except TimeoutError as exc:
            raise RuntimeError(f"no observation includes committed command {seq}") from exc

    def check(self, *, moving: bool = False) -> None:
        now = time.monotonic()
        if self.error:
            raise RuntimeError(self.error)
        if min(self.last_observation, self.last_telemetry, self.last_wrist) < now - 0.5:
            raise RuntimeError(f"camera or robot telemetry is stale: {self.snapshot()['age_ms']} ms")
        arm = self.telemetry["arms"][self.cfg["arm"]]
        if moving and (arm["mode"] == "stop" or not arm["torque"]):
            raise RuntimeError(
                f"selected arm mode={arm['mode']}, torque={arm['torque']}; refusing to re-engage. "
                "Select MANUAL for this arm in Haller before rerunning; torque Engage alone does not clear STOP."
            )

    async def telemetry_loop(self) -> None:
        from websockets.asyncio.client import connect

        try:
            async with connect("ws://127.0.0.1:8000/ws/telemetry") as ws:
                overloaded = None
                async for raw in ws:
                    self.telemetry = json.loads(raw)
                    self.last_telemetry = time.monotonic()
                    arm = self.telemetry["arms"][self.cfg["arm"]]
                    if arm.get("effort_status") not in ("ok",):
                        raise RuntimeError(f"unhealthy effort telemetry: {arm.get('effort_status')}")
                    joints = arm["joints"]
                    loads = {joint: float(joints[joint]["effort"]) for joint in SO101_JOINTS}
                    if not all(math.isfinite(v) and abs(v) <= 1 for v in loads.values()):
                        raise RuntimeError("invalid motor-load telemetry")
                    if self.command is not None:
                        self.event("telemetry", measured={j: joints[j]["pos"] for j in SO101_JOINTS},
                                   load=loads, committed=self.command, command_seq=self.command_seq)
                    if self.command is not None:
                        # Closing onto an object is intentional gripper contact.
                        # The independent high-load check below still covers it.
                        targets = dict(zip(SO101_JOINTS[:-1], self.command[:-1], strict=True))
                        positions = {joint: float(joints[joint]["pos"]) for joint in targets}
                        self.stall.observe(self.last_telemetry, positions, targets, loads)
                    else:
                        self.stall.reset()
                    if any(abs(load) > 0.7 for load in loads.values()):
                        overloaded = overloaded or self.last_telemetry
                        if self.last_telemetry - overloaded > 0.5:
                            raise RuntimeError("sustained motor load exceeds 70%")
                    else:
                        overloaded = None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.error = f"telemetry failed: {exc}"

    async def observations(self, reader: asyncio.StreamReader) -> None:
        try:
            while raw := await reader.readline():
                message = json.loads(raw)
                if message.get("type") == "policy_action_ack":
                    if self.pending_command is None:
                        raise RuntimeError("unexpected command acknowledgement")
                    seq, future = self.pending_command
                    if abs(time.time() - message["t_ms"] / 1000) > 0.5:
                        raise RuntimeError("stale command acknowledgement")
                    self.command = committed_target(message, self.run_id, seq, self.cfg["arm"])
                    self.command_seq = seq
                    self.event("committed", seq=seq, command=self.command, result=message["result"])
                    future.set_result(message)
                    continue
                if message.get("type") != "policy_observation":
                    raise RuntimeError(message.get("detail", "unexpected Haller wire message"))
                if abs(time.time() - message["t_ms"] / 1000) > 0.5:
                    raise RuntimeError("stale robot observation")
                state = selected_state(message, self.cfg["arm"], single_arm=self.cfg.get("single_arm"))
                if message.get("run_id") != self.run_id:
                    raise RuntimeError("observation belongs to a different run")
                command = selected_state({**message, "state": message["command"]}, self.cfg["arm"],
                                         single_arm=self.cfg.get("single_arm"))
                command_seq = message["command_seq"]
                if type(command_seq) is not int or not self.observed_command_seq <= command_seq <= self.command_seq:
                    raise RuntimeError("observation has an invalid committed-command sequence")
                scene = message["images"][self.cfg["scene_key"]]
                if self.wrist is None or time.monotonic() - self.last_wrist > 0.5:
                    continue
                self.latest = {"state": state, "scene": scene, "wrist": self.wrist}
                self.last_observation = time.monotonic()
                self.history.append({**self.latest, "state": transform(state, self.cfg),
                                     "command_history": transform(command, self.cfg)})
                self.observed_command_seq = command_seq
                self.observation_changed.set()
            raise RuntimeError("Haller policy connection closed")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.error = f"observations failed: {exc}"
            self.observation_changed.set()
            if self.pending_command is not None:
                future = self.pending_command[1]
                if not future.done():
                    future.set_exception(RuntimeError(self.error))

    async def cameras(self) -> None:
        import cv2

        if self.cfg.get("wrist_camera_id"):
            # Haller now owns these USB streams. Reuse its rotated snapshots
            # rather than opening the same V4L device a second time.
            def fetch():
                import numpy as np

                camera = str(self.cfg["wrist_camera_id"])
                with urllib.request.urlopen(f"http://127.0.0.1:8000/cameras/{camera}/snapshot", timeout=3) as r:
                    data = r.read()
                frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None or float(frame.std()) < 2:
                    raise RuntimeError("wrist image has no usable detail; inspect the lens and camera controls")
                return base64.b64encode(data).decode()

            try:
                while True:
                    self.wrist = await asyncio.to_thread(fetch)
                    self.last_wrist = time.monotonic()
                    await asyncio.sleep(1 / 30)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.error = f"camera failed: {exc}"
            return

        cap = cv2.VideoCapture(str(self.cfg["wrist_device"]), cv2.CAP_V4L2)
        try:
            if not cap.isOpened():
                raise RuntimeError("cannot open wrist camera")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            async def read_frame():
                pending = asyncio.create_task(asyncio.to_thread(cap.read))
                try:
                    return await asyncio.shield(pending)
                except asyncio.CancelledError:
                    # Never close the V4L handle while its read runs on a thread.
                    await pending
                    raise

            # V4L starts the stream at the FIRST READ, which resets controls on
            # these cameras. Applying controls after open alone is insufficient.
            ok, _ = await read_frame()
            if not ok:
                raise RuntimeError("cannot start wrist camera stream")
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            cap.set(cv2.CAP_PROP_GAIN, 0)
            cap.set(cv2.CAP_PROP_BACKLIGHT, 55)
            cap.set(cv2.CAP_PROP_AUTO_WB, 1)
            cap.set(cv2.CAP_PROP_SATURATION, 64)
            controls = {name: cap.get(prop) for name, prop in (
                ("auto_exposure", cv2.CAP_PROP_AUTO_EXPOSURE), ("gain", cv2.CAP_PROP_GAIN),
                ("backlight", cv2.CAP_PROP_BACKLIGHT), ("auto_white_balance", cv2.CAP_PROP_AUTO_WB),
                ("saturation", cv2.CAP_PROP_SATURATION),
            )}
            (self.output / "camera-controls.json").write_text(json.dumps(controls, indent=2) + "\n")
            # Let exposure and white balance settle before publishing frames.
            for _ in range(8):
                ok, _ = await read_frame()
                if not ok:
                    raise RuntimeError("wrist camera warmup failed")
            while True:
                ok, frame = await read_frame()
                if not ok:
                    raise RuntimeError("wrist camera read failed")
                if float(frame.std()) < 2:
                    raise RuntimeError("wrist image has no usable detail; inspect the lens and camera controls")
                rotation = self.cfg["wrist_rotation"]
                if rotation == 180:
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                elif rotation != 0:
                    raise ValueError("only 0 or 180 degree wrist rotation is supported")
                ok, jpg = cv2.imencode(".jpg", cv2.resize(frame, (256, 256)))
                if not ok:
                    raise RuntimeError("wrist JPEG encoding failed")
                self.wrist = base64.b64encode(jpg).decode()
                self.last_wrist = time.monotonic()
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.error = f"camera failed: {exc}"
        finally:
            cap.release()


async def remote_worker(cfg: dict, log: Any) -> asyncio.subprocess.Process:
    arguments = [] if cfg.get("official_lerobot") else worker_arguments(inference_settings(cfg))
    if cfg.get("official_lerobot"):
        if cfg.get("checkpoint"):
            arguments += ["--checkpoint", cfg["checkpoint"]]
        if cfg.get("normalization_profile"):
            arguments += ["--normalization-profile", cfg["normalization_profile"]]
    command = ("source /liminal/env.sh; source /liminal/offline_env.sh; exec "
               + shlex.join([cfg["python"], "-u", cfg["worker"], *arguments]))
    return await asyncio.create_subprocess_exec(
        "gcloud", "compute", "ssh", cfg["vm"], "--zone=" + cfg["zone"], "--project=" + cfg["project"],
        "--tunnel-through-iap", "--ssh-flag=-oBatchMode=yes", "--ssh-flag=-T", "--command=" + command,
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=log, limit=4 * 1024 * 1024,
    )


async def worker_reply(process: asyncio.subprocess.Process, timeout: float) -> dict:
    raw = await asyncio.wait_for(process.stdout.readline(), timeout)
    if not raw:
        raise RuntimeError("GPU worker exited; inspect worker.log")
    reply = json.loads(raw)
    if reply.get("type") == "error":
        raise RuntimeError(reply["detail"])
    return reply


async def run(cfg: dict, output: Path, execute: bool) -> None:
    side = cfg["arm"]
    if side not in ("left", "right"):
        raise ValueError("select exactly one arm")
    transform([0] * 6, cfg)  # Validate coordinate configuration before any I/O.
    normalization_profile = cfg.get("normalization_profile", "released")
    if normalization_profile not in ("released", "hf_space_native"):
        raise ValueError("unknown normalization profile")
    if normalization_profile == "hf_space_native":
        if not cfg.get("official_lerobot") or not cfg.get("checkpoint"):
            raise ValueError("HF Space normalization requires an explicit official checkpoint package")
        if vector(cfg["joint_scale"]) != [1] * 6 or vector(cfg["joint_offset"]) != [0] * 6:
            raise ValueError("HF Space normalization requires native coordinates")
    coordinates = {"name": cfg.get("coordinate_mapping", "configured"),
                   "joint_order": list(SO101_JOINTS),
                   "robot_to_model_scale": vector(cfg["joint_scale"]),
                   "robot_to_model_offset": vector(cfg["joint_offset"])}
    duration = rollout_duration(cfg)
    tuning = inference_settings(cfg)
    obsolete = {"max_speed", "max_tracking_error", "max_excursion", "coordinated_slowdown"}.intersection(cfg)
    if obsolete:
        raise ValueError("remove obsolete live bridge caps from the profile: " + ", ".join(sorted(obsolete)))
    if execute and (cfg.get("camera_verified") is not True or cfg.get("units_verified") is not True):
        raise RuntimeError("motion blocked: selected wrist camera or joint units remain unverified in the rig profile")
    inventory = await asyncio.to_thread(http, "/config")
    candidates = [arm for arm in inventory.get("arms", []) if arm.get("id") == side]
    if len(candidates) != 1 or candidates[0].get("source") != "real":
        raise RuntimeError(f"Haller does not identify one real arm named {side!r}")
    if candidates[0].get("port") != cfg["expected_port"]:
        raise RuntimeError("selected controller port changed; reverify the physical arm binding")
    cfg = {**cfg, "single_arm": side if len(inventory["arms"]) == 1 else None}
    preflight = await asyncio.to_thread(http, f"/arm/{side}/preflight")
    if not preflight.get("ok"):
        raise RuntimeError(f"arm preflight failed: {preflight}")
    if execute and preflight.get("mode") == "stop":
        raise RuntimeError(
            "selected arm is in STOP. Select MANUAL for this arm in Haller before rerunning; "
            "torque Engage alone does not clear STOP. No GPU worker or motion was started."
        )
    rig = LiveRig(cfg, output)
    tasks: list[asyncio.Task] = []
    writer = None
    process = None
    engaged = False
    deadline = math.inf
    report: dict = {"arm": side, "prompt": cfg["prompt"], "execute": execute, "status": "starting",
                    "duration_s": duration, "coordinate_mapping": coordinates}
    log = (output / "worker.log").open("w")
    submitted = (output / "submitted-actions.jsonl").open("w")
    plans = (output / "plans.jsonl").open("w")
    try:
        print("Joint coordinate mapping:", json.dumps(coordinates), flush=True)
        if cfg.get("coordinate_mapping") == "experimental_legacy_shoulder_elbow":
            print("Experimental mapping: model shoulder = 90 - robot shoulder; model elbow = robot elbow + 90. "
                  "Other joints unchanged; physical wrist compatibility remains unverified.", flush=True)
        rig.event("coordinate_mapping", **coordinates)
        tasks = [asyncio.create_task(rig.telemetry_loop()), asyncio.create_task(rig.cameras())]
        process = await remote_worker(cfg, log)
        print("Loading the GPU checkpoint; motors unchanged. Logs:", output / "worker.log", flush=True)
        rig.diagnostic("startup", arm=side, port=cfg["expected_port"], prompt=cfg["prompt"],
                       execute=execute, duration_s=duration, preflight=preflight,
                       wrist_camera=cfg.get("wrist_camera_id"), scene_key=cfg.get("scene_key"),
                       joint_order=list(SO101_JOINTS), units="first five: degrees; gripper: percentage points")
        ready = await worker_reply(process, 180)
        if ready.get("type") != "ready":
            raise RuntimeError("GPU worker did not report ready")
        if ready.get("normalization_profile", "released") != normalization_profile:
            raise RuntimeError("GPU worker loaded different normalization statistics than requested")
        report["normalization_profile"] = normalization_profile
        report["normalization_source"] = ready.get("normalization_source")
        report["action_statistics"] = ready.get("action_statistics")
        print("Normalization profile:", normalization_profile, flush=True)
        effective = dict(ready["inference"])
        if cfg.get("official_lerobot"):
            if ready.get("integration") != "lerobot.Flux3Policy + saved pre/post processors":
                raise RuntimeError("expected official LeRobot policy worker")
            effective["control_hz"] = effective["fps"]
            effective["seed"] = effective["inference_seed"]
            report["integration"] = ready["integration"]
            report["source_revision"] = ready["source_revision"]
            print("Official LeRobot policy/processors; Haller owns motors and UI E-stop.", flush=True)
        if effective["chunk_size"] != 42 or effective["control_hz"] != 30:
            raise RuntimeError("GPU worker changed the trained horizon or control rate")
        if any(effective.get(key) != value for key, value in tuning.items()):
            raise RuntimeError("GPU worker did not apply the requested inference settings")
        control_hz = float(effective["control_hz"])
        execute_steps = int(effective["n_action_steps"])
        report["inference"] = effective
        print("Effective inference settings:", json.dumps(effective), flush=True)
        report["checkpoint"] = ready["checkpoint"]
        reader, writer = await asyncio.open_connection("127.0.0.1", 8781, limit=4 * 1024 * 1024)
        run_id = str(uuid.uuid4())
        rig.run_id = run_id
        writer.write((json.dumps({"type": "policy_hello", "run_id": run_id, "unit": "deg",
                                  "command_feedback": 1,
                                  "control_hz_declared": control_hz,
                                  "rig": "solo" if cfg["single_arm"] else "bimanual"}) + "\n").encode())
        await writer.drain()
        ack = json.loads(await asyncio.wait_for(reader.readline(), 5))
        if ack.get("type") != "policy_hello_ack" or not ack.get("ok"):
            raise RuntimeError(ack.get("detail", "Haller refused the policy connection"))
        if ack.get("command_feedback") != 1:
            raise RuntimeError("Haller needs the committed-command feedback update; refusing to execute")
        tasks.append(asyncio.create_task(rig.observations(reader)))
        async with asyncio.timeout(10):
            while len(rig.history) < 8:
                if rig.error:
                    raise RuntimeError(rig.error)
                await asyncio.sleep(0.05)
        rig.check()
        initial = vector(rig.latest["state"])
        report.update(initial_state=initial, model_state=transform(initial, cfg),
                      state_statistics=ready.get("state_statistics"), camera_verified=cfg["camera_verified"],
                      units_verified=cfg["units_verified"], controller=candidates[0])
        model_state = transform(initial, cfg)
        stats = ready.get("state_statistics")
        report["outside_central_training_range"] = [
            joint for joint, q, lo, hi in zip(SO101_JOINTS, model_state, stats["q01"], stats["q99"], strict=True)
            if not lo - 20 <= q <= hi + 20
        ] if stats else []
        for role in ("scene", "wrist"):
            (output / f"{role}.jpg").write_bytes(base64.b64decode(rig.latest[role], validate=True))
        rig.diagnostic("ready", feedback_version=ack["command_feedback"], history_frames=len(rig.history),
                       snapshot=rig.snapshot(), outside_training_range=report["outside_central_training_range"],
                       calibrated_limits={j: {k: cell.get(k) for k in ("min", "max")}
                                          for j, cell in rig.telemetry["arms"][side]["joints"].items()})
        plan_index = 0
        # GPU inference must succeed on actual observations before any commands.
        async def predict() -> dict:
            nonlocal plan_index
            rig.check(moving=engaged)
            await rig.wait_observed(rig.command_seq)
            request = {"history": list(rig.history), "prompt": cfg["prompt"]}
            # Keep the exact visual and numeric conditioning for offline replay.
            # A single startup JPEG cannot reproduce later action chunks.
            request_path = output / f"input-{plan_index:04d}.json"
            request_path.write_text(json.dumps(request, allow_nan=False) + "\n")
            started = time.monotonic()
            process.stdin.write((json.dumps(request, allow_nan=False) + "\n").encode())
            await process.stdin.drain()
            reply = await worker_reply(process, 12)
            reply["rpc_seconds"] = time.monotonic() - started
            reply["command_anchor"] = vector(request["history"][-1]["command_history"])
            rig.check(moving=engaged)
            if reply.get("type") != "actions" or len(reply["actions"]) != 42:
                raise ValueError("invalid action chunk")
            for action in reply["actions"]:
                vector(action)
            reply["robot_actions"] = [transform(action, cfg, inverse=True) for action in reply["actions"]]
            plans.write(json.dumps({"time": time.time(), "inference_seconds": reply["inference_seconds"],
                                    "rpc_seconds": reply["rpc_seconds"], "execute_steps": execute_steps,
                                    "command_anchor": reply["command_anchor"], "actions": reply["actions"],
                                    "actions_frame": "model", "robot_actions": reply["robot_actions"],
                                    "history": [{k: f[k] for k in ("state", "command_history")}
                                                for f in request["history"]]}) + "\n")
            plans.flush()
            reply["plan_index"] = plan_index
            rig.diagnostic("plan", plan=plan_index, inference_s=round(reply["inference_seconds"], 3),
                           rpc_s=round(reply["rpc_seconds"], 3), predicted_steps=len(reply["actions"]),
                           execute_steps=execute_steps, command_anchor=reply["command_anchor"],
                           first_predicted=reply["actions"][0],
                           last_executed_prediction=reply["actions"][execute_steps - 1],
                           first_robot_target=reply["robot_actions"][0],
                           last_robot_target=reply["robot_actions"][execute_steps - 1])
            plan_index += 1
            return reply

        first = await predict()
        report.update(first_inference_seconds=first["inference_seconds"], first_rpc_seconds=first["rpc_seconds"],
                      first_actions=first["actions"], first_robot_actions=first["robot_actions"])
        print(f"Real-camera inference: {first['inference_seconds']:.3f}s for 42 actions; motors unchanged.", flush=True)
        if not execute:
            report["status"] = "observation_only_passed"
            if cfg.get("official_lerobot"):
                from liminal.policy.official_live import check_chunk
                cells = rig.telemetry["arms"][side]["joints"]
                lower, upper = pose_limits(cells, initial)
                try:
                    check_chunk(first["robot_actions"], lower, upper, execute_steps)
                except RuntimeError as exc:
                    report["execution_blocker"] = str(exc)
                    print(str(exc), flush=True)
            print("Observation-only result saved to", output, flush=True)
            return
        outside = execution_gate(cfg, initial, stats) if stats else []
        if outside:
            print("Outside the checkpoint's central training distribution:", ", ".join(outside), flush=True)
        cells = rig.telemetry["arms"][side]["joints"]
        lower, upper = pose_limits(cells, initial)
        proxy = WireArm(side)
        # Enforce calibrated travel without imposing another command-lead cap.
        # The official path rejects invalid chunks before sending any targets.
        full_travel = max(hi - lo for lo, hi in zip(lower, upper, strict=True))
        safety = SafetySupervisor({side: proxy}, {side: JointLimits(lower, upper, full_travel)})
        # The UI must engage first. Never re-arm here: an E-stop during model
        # loading must stay stopped, including a race just before this check.
        rig.check(moving=True)
        engaged = True
        async def watchdog() -> None:
            try:
                while True:
                    rig.check(moving=True)
                    await asyncio.sleep(0.05)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                rig.error = str(exc)
                await asyncio.to_thread(http, f"/arm/{side}/mode", {"mode": "stop"})

        tasks.append(asyncio.create_task(watchdog()))
        seq = 0
        deadline = time.monotonic() + duration if duration is not None else math.inf
        print(f"Executing {execute_steps}/42 actions at {control_hz:g} Hz; "
              + ("no rollout time limit. Ctrl+C stops and releases torque." if duration is None
                 else f"rollout duration {duration:g}s."), flush=True)
        print("Targets use calibrated travel limits; Haller policy slew limiting is disabled. "
              "No startup-relative excursion or command-lead cap.", flush=True)
        async with safety.claim(side, run_id), asyncio.timeout(duration):
            chunk = first
            while True:
                if cfg.get("official_lerobot"):
                    from liminal.policy.official_live import check_chunk
                    check_chunk(chunk["robot_actions"], lower, upper, execute_steps)
                previous_prediction = vector(chunk["command_anchor"])
                diagnostics = ChunkDiagnostics(vector(rig.latest["state"]), time.monotonic())
                for index, predicted in enumerate(chunk["actions"][:execute_steps]):
                    tick = time.monotonic()
                    rig.check(moving=True)
                    proxy.current = vector(rig.latest["state"])
                    anchor = transform(rig.command, cfg) if rig.command is not None else previous_prediction
                    rebased = (vector(predicted) if cfg.get("official_lerobot")
                               else rebase_action(vector(predicted), previous_prediction, anchor))
                    requested = (chunk["robot_actions"][index] if cfg.get("official_lerobot")
                                 else transform(rebased, cfg, inverse=True))
                    target = safety.send(side, requested)
                    submitted.write(json.dumps({"seq": seq, "time": time.time(), "arm": side,
                                                "measured": proxy.current, "predicted": predicted,
                                                "rebased": requested, "submitted": target}) + "\n")
                    submitted.flush()
                    sent_at = time.monotonic()
                    await rig.send_command(writer, seq, target)
                    if cfg.get("official_lerobot") and any(
                        abs(a - b) > 1e-5 for a, b in zip(target, rig.command, strict=True)
                    ):
                        raise RuntimeError("controller altered official target; refusing distorted playback")
                    diagnostics.record(requested, target, rig.command, time.monotonic() - sent_at,
                                       rig.snapshot()["load"])
                    rig.event("command_timing", seq=seq, ack_ms=1000 * (time.monotonic() - sent_at))
                    previous_prediction = vector(predicted)
                    seq += 1
                    await asyncio.sleep(max(0, 1 / control_hz - (time.monotonic() - tick)))
                rig.diagnostic("chunk", plan=chunk["plan_index"],
                               **diagnostics.summary(vector(rig.latest["state"]), time.monotonic()),
                               snapshot=rig.snapshot())
                chunk = await predict()
                submitted.flush()
    except TimeoutError:
        if engaged and time.monotonic() >= deadline:
            report["status"] = "duration_reached_task_success_unverified"
        else:
            report["status"] = "failed"
            report["error"] = "setup or inference timed out"
            raise
    except BaseException as exc:
        report.update(status="stopped" if engaged else "failed", error=str(exc) or type(exc).__name__)
        raise
    finally:
        # Persist the stop reason before cleanup awaits: a second Ctrl+C or
        # failed worker teardown must not erase the only failure report.
        report["last_snapshot"] = rig.snapshot()
        rig.diagnostic("stop", status=report["status"], reason=report.get("error"),
                       snapshot=rig.snapshot(), logs=str(output))
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        cleanup_errors = []
        try:
            if engaged:
                try:
                    await asyncio.to_thread(http, f"/arm/{side}/mode", {"mode": "stop"})
                except Exception as exc:
                    cleanup_errors.append(f"torque release unconfirmed: {exc}")
            if writer:
                try:
                    writer.close()
                    await asyncio.wait_for(writer.wait_closed(), 2)
                except Exception as exc:
                    cleanup_errors.append(f"controller connection close: {exc}")
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if process:
                try:
                    if process.stdin:
                        process.stdin.close()
                    try:
                        await asyncio.wait_for(process.wait(), 10)
                    except TimeoutError:
                        with contextlib.suppress(ProcessLookupError):
                            process.terminate()
                        try:
                            await asyncio.wait_for(process.wait(), 3)
                        except TimeoutError:
                            with contextlib.suppress(ProcessLookupError):
                                process.kill()
                            await asyncio.wait_for(process.wait(), 3)
                except Exception as exc:
                    cleanup_errors.append(f"GPU worker cleanup: {exc}")
        finally:
            log.close()
            submitted.close()
            plans.close()
            if cleanup_errors:
                report["cleanup_errors"] = cleanup_errors
                print("Cleanup needs attention:", "; ".join(cleanup_errors), file=sys.stderr, flush=True)
            (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if cleanup_errors and not report.get("error"):
            raise RuntimeError("cleanup failed: " + "; ".join(cleanup_errors))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/policy/official_haller_right.yaml")
    parser.add_argument("--execute", action="store_true", help="send bounded commands to the selected arm")
    parser.add_argument("--engage", action="store_true", help="acknowledge engaging the selected arm in the UI")
    parser.add_argument("--output", default=None)
    parser.add_argument("--verbose", action="store_true", help="print startup checks and per-chunk diagnostics")
    parser.add_argument("--prompt", help="task instruction for this run")
    parser.add_argument("--legacy-shoulder-elbow", action="store_true",
                        help="experimental software mapping: model shoulder=90-robot shoulder, elbow=robot elbow+90")
    parser.add_argument("--n-action-steps", type=int, help="actions executed per 42-action plan")
    parser.add_argument("--num-inference-steps", type=int, help="Euler denoising steps per plan")
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--guidance-scale-action", type=float)
    parser.add_argument("--sampler-shift", type=float)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.execute != args.engage:
        parser.error("motor execution requires both --execute and --engage")
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.legacy_shoulder_elbow:
        try:
            cfg = legacy_shoulder_elbow(cfg)
        except ValueError as exc:
            parser.error(str(exc))
    cfg["verbose"] = args.verbose
    cfg["inference"] = {**(cfg.get("inference") or {}), **{
        key: getattr(args, key) for key in ("n_action_steps", "num_inference_steps", "guidance_scale",
                                          "guidance_scale_action", "sampler_shift", "seed")
        if getattr(args, key) is not None
    }}
    if args.prompt is not None:
        cfg["prompt"] = args.prompt
    output = Path(args.output or Path.home() / ".local/share/liminal/inference" / time.strftime("%Y%m%d-%H%M%S"))
    output.mkdir(parents=True, exist_ok=False)
    try:
        asyncio.run(run(cfg, output, args.execute))
    except KeyboardInterrupt:
        print(f"Interrupted; selected-arm stop requested if engaged. Logs: {output}")
    except Exception as exc:
        print(f"Inference stopped: {exc}. Logs: {output}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
