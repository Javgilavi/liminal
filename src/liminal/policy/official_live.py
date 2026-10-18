"""Official LeRobot FLUX processors + SO-101 driver, joined by SSH transport.

Observation-only by default. Haller must be stopped. There is no trajectory
reshaping: invalid/out-of-travel chunks stop before execution. Ctrl+C releases
torque. The Haller UI E-stop is unavailable while this process owns the robot.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import json
import math
import shlex
import signal
import sys
import time
from collections import deque
from pathlib import Path

import yaml

from liminal.core.types import SO101_JOINTS
from liminal.hardware.official_so101 import OfficialSO101
from liminal.policy.live import vector, worker_reply
from liminal.safety.stall import StallMonitor
from liminal.safety.supervisor import JointLimits, SafetySupervisor

POD_WORKER = Path(__file__).resolve().parents[3] / "scripts" / "flux_pod_worker.py"


def check_chunk(actions, lower, upper, count):
    if len(actions) != 42:
        raise ValueError("expected 42 predicted actions")
    for index, action in enumerate(actions):
        values = vector(action)
        if index >= count:
            continue
        for joint, value, lo, hi in zip(SO101_JOINTS, values, lower, upper, strict=True):
            if not lo <= value <= hi:
                raise RuntimeError(f"plan refused: step {index}, {joint}={value:.2f}, "
                                   f"calibrated [{lo:.2f}, {hi:.2f}]; targets were not clipped")


async def run(cfg, output, execute):
    import cv2

    from liminal.policy.hybrid import build_hybrid

    hybrid = build_hybrid(cfg, execute=execute)
    arm = OfficialSO101(cfg)
    history = deque(maxlen=8)
    state = {"error": None, "last": 0.0, "command": None}
    observation_changed = asyncio.Event()
    process = None
    sampler = None
    report = {"execute": execute, "arm": cfg["arm"], "prompt": cfg["prompt"], "status": "starting"}
    stall = StallMonitor()
    log = (output / "worker.log").open("w")
    events = (output / "events.jsonl").open("w")

    def event(kind, **values):
        events.write(json.dumps({"time": time.time(), "type": kind, **values}, allow_nan=False) + "\n")
        events.flush()

    def check():
        if state["error"]:
            raise RuntimeError(state["error"])
        if time.monotonic() - state["last"] > 0.5:
            raise RuntimeError("robot/camera observation is stale")

    async def wait_after(timestamp):
        async with asyncio.timeout(0.5):
            while state["last"] <= timestamp:
                check()
                observation_changed.clear()
                await observation_changed.wait()
        check()

    async def sample():
        high_since = None
        try:
            while True:
                started = time.monotonic()
                observation, loads = arm.observe()
                measured = vector(arm.current)
                if not all(math.isfinite(v) and abs(v) <= 1 for v in loads.values()):
                    raise RuntimeError("invalid motor-load telemetry")
                frame = {"state": measured, "command_history": list(state["command"] or measured)}
                for role in ("scene", "wrist"):
                    rgb = observation[role]
                    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    if not ok or float(rgb.std()) < 2:
                        raise RuntimeError(f"unusable {role} camera image")
                    frame[role] = base64.b64encode(encoded).decode()
                history.append(frame)
                state["last"] = time.monotonic()
                observation_changed.set()
                event("observation", measured=measured, command=state["command"], loads=loads)
                if arm.engaged:
                    positions = dict(zip(SO101_JOINTS[:-1], measured[:-1], strict=True))
                    targets = dict(zip(SO101_JOINTS[:-1], state["command"][:-1], strict=True))
                    stall.observe(state["last"], positions, targets, loads)
                    if any(abs(v) > 0.7 for v in loads.values()):
                        high_since = high_since or state["last"]
                        if state["last"] - high_since > 0.5:
                            raise RuntimeError("sustained motor load exceeds 70%")
                    else:
                        high_since = None
                await asyncio.sleep(max(0, 1 / 30 - (time.monotonic() - started)))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            state["error"] = str(exc)
            observation_changed.set()
            try:
                arm.stop()
            except Exception as stop_error:
                state["error"] += f"; {stop_error}"

    try:
        arm.connect_read_only()
        lower, upper = arm.limits()
        # Full-travel max_step prevents Supervisor from reshaping valid targets.
        supervisor = SafetySupervisor({arm.name: arm}, {
            arm.name: JointLimits(lower, upper, max(hi - lo for lo, hi in zip(lower, upper, strict=True)))})
        sampler = asyncio.create_task(sample())
        if cfg.get("transport") == "ws4_pod":
            # the same worker protocol, answered by a served LoRA package on the ws4 policy server
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-u", str(POD_WORKER), "--server", cfg["server"], "--package", cfg["package"],
                "--n-action-steps", str(cfg.get("n_action_steps", 16)),
                "--token-cmd-env", cfg.get("token_cmd_env", "WS4_TOKEN_CMD"),
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=log, limit=4 * 1024 * 1024)
        else:
            command = "source /liminal/offline_env.sh; exec " + shlex.join([cfg["python"], "-u", cfg["worker"]])
            process = await asyncio.create_subprocess_exec(
                "gcloud", "compute", "ssh", cfg["vm"], "--zone=" + cfg["zone"], "--project=" + cfg["project"],
                "--tunnel-through-iap", "--ssh-flag=-oBatchMode=yes", "--ssh-flag=-T", "--command=" + command,
                stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=log, limit=4 * 1024 * 1024)
        print("Loading official LeRobot FLUX on VM; torque OFF. Logs:", output, flush=True)
        ready = await worker_reply(process, 180)
        if ready.get("type") != "ready" or not ready.get("integration", "").startswith("lerobot.Flux3Policy"):
            raise RuntimeError("worker is not the official LeRobot integration")
        report["worker"] = ready
        print(json.dumps(ready), flush=True)
        async with asyncio.timeout(10):
            while len(history) < 8:
                if state["error"]:
                    raise RuntimeError(state["error"])
                await asyncio.sleep(0.05)
        count = ready["inference"]["n_action_steps"]
        if ready["inference"]["fps"] != 30 or not 1 <= count <= 42:
            raise RuntimeError("unexpected execution settings")
        index = 0
        async with supervisor.claim(arm.name, "official-lerobot"):
            while True:
                check()
                request = {"prompt": cfg["prompt"], "history": list(history)}
                (output / f"input-{index:04d}.json").write_text(json.dumps(request) + "\n")
                process.stdin.write((json.dumps(request) + "\n").encode())
                await process.stdin.drain()
                result = await worker_reply(process, 15)
                check()
                (output / f"plan-{index:04d}.json").write_text(json.dumps(result) + "\n")
                if result.get("type") != "actions" or len(result.get("actions", [])) != 42:
                    raise RuntimeError("worker returned no actions")
                print(f"Plan {index}: {result['inference_seconds']:.3f}s, {count}/42 actions", flush=True)
                segment = None
                if hybrid is not None:
                    segment = await hybrid.choose(goal=cfg["prompt"], actions=result["actions"],
                                                  observation=dict(history[-1]), max_steps=count, log=event)
                    check()
                    hybrid.validate_fresh(segment, arm.current)
                    for i in range(segment.steps):
                        hybrid.target(segment, i, arm.current, lower, upper)
                    report["hybrid_decision"] = segment.decision
                if not execute:
                    report["status"] = "observation_only_passed_task_success_unverified"
                    try:
                        if segment is None:
                            check_chunk(result["actions"], lower, upper, count)
                    except RuntimeError as exc:
                        report["execution_blocker"] = str(exc)
                        print(str(exc), flush=True)
                    return
                if segment is not None and segment.stopped:
                    report["status"] = "hybrid_stopped_task_success_unverified"
                    return
                if segment is None:
                    check_chunk(result["actions"], lower, upper, count)
                if not arm.engaged:
                    state["command"] = arm.read_state()
                    arm.engage()
                    print(("EXECUTING hybrid-selected targets. " if hybrid else
                           "EXECUTING unchanged LeRobot targets; no Haller speed limiter. ") +
                          "Ctrl+C releases torque; Haller UI E-stop is NOT active.", flush=True)
                executed = []
                acknowledged_at = None
                for step, target in enumerate(result["actions"][:segment.steps if segment else count]):
                    if segment is not None and acknowledged_at is not None:
                        await wait_after(acknowledged_at)
                    started = time.monotonic()
                    check()
                    if segment is not None:
                        # The synchronous driver returns after send; re-read actual joints
                        # for every local IK update rather than integrating imagined motion.
                        measured = vector(arm.read_state())
                        target = hybrid.target(segment, step, measured, lower, upper)
                    state["command"] = supervisor.send(arm.name, vector(target))
                    acknowledged_at = time.monotonic()
                    executed.append(list(state["command"]))
                    event("command", plan=index, target=state["command"], measured=arm.current)
                    await asyncio.sleep(max(0, 1 / 30 - (time.monotonic() - started)))
                if segment is not None:
                    await wait_after(acknowledged_at)
                    hybrid.record(segment, executed, vector(arm.read_state()), event)
                index += 1
    except BaseException as exc:
        report.update(status="stopped", error=str(exc) or type(exc).__name__)
        raise
    finally:
        errors = []
        try:
            arm.close()
        except Exception as exc:
            errors.append(str(exc))
        if sampler:
            sampler.cancel()
            await asyncio.gather(sampler, return_exceptions=True)
        report["cleanup_errors"] = errors
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        log.close()
        events.close()
        if process:
            if process.stdin:
                process.stdin.close()
            try:
                await asyncio.wait_for(process.wait(), 10)
            except TimeoutError:
                process.terminate()
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(process.wait(), 3)
        if errors:
            print("CLEANUP ERROR:", "; ".join(errors), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/policy/official_right.yaml")
    parser.add_argument("--execute", action="store_true", help="engage and execute after preflight/inference")
    parser.add_argument("--prompt")
    parser.add_argument("--hybrid-config", help="optional action-level Claude gate YAML; disabled by default")
    args = parser.parse_args()
    from liminal.policy.hybrid import load_hybrid_config

    cfg = load_hybrid_config(yaml.safe_load(Path(args.config).read_text()), args.hybrid_config)
    if cfg["arm"] != "right":
        parser.error("this commissioned profile is for software right only")
    if args.prompt:
        cfg["prompt"] = args.prompt
    output = Path.home() / ".local/share/liminal/inference" / (time.strftime("%Y%m%d-%H%M%S") + "-official")
    output.mkdir(parents=True, exist_ok=False)

    async def session():
        task = asyncio.current_task()
        asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, task.cancel)
        await run(cfg, output, args.execute)
    try:
        asyncio.run(session())
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("Interrupted; torque release attempted. Check report:", output)
    except Exception as exc:
        print(f"Stopped: {exc}. Logs: {output}")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
