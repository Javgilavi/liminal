"""Execute one bounded fitting experiment; preserve failures and independent task metrics."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

from fitting import decide, digest, freeze_windows, training_config


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def run_stage(command, log, timeout):
    with log.open("w") as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise RuntimeError(f"FIT_TIMEOUT: {timeout}s limit exceeded; inspect {log}") from exc
    if code:
        raise RuntimeError(f"FIT_DOWNSTREAM: exit={code}; inspect {log}")


def main(args):
    root, output = args.root.resolve(), args.output.resolve()
    if shutil.disk_usage(root).free < 30 * 1024**3:
        raise ValueError("FIT_DISK: require at least 30 GiB free before writing checkpoints")
    lock = (root / ".fixed-window-fitting.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("FIT_LOCK: another fitting experiment owns this workspace") from exc
    output.mkdir(parents=True, exist_ok=False)
    state = {"stage": "preflight", "completed": False, "commands": [], "started_unix": time.time()}
    status = output / "status.json"
    write_json(status, state)
    code = Path(__file__).resolve().parent
    baseline = root / "training/checkpoints/003000/pretrained_model"
    try:
        manifest = json.loads((root / "dataset-manifest.json").read_text())
        frozen = freeze_windows(manifest)
        if {r["seed"] for r in frozen["windows"]} != {100, 10000}:
            raise ValueError("FIT_SEEDS: this protocol requires training 100 and validation 10000")
        windows = output / "windows.json"
        write_json(windows, frozen)
        config = training_config(json.loads((root / "train-config.json").read_text()), baseline,
                                 output / "training", args.steps)
        write_json(output / "config.json", config)
        inputs = [root / "dataset-manifest.json", root / "hf-sim-normalization.json", root / "scene.xml",
                  output / "config.json", windows, *code.glob("*.py"), *baseline.glob("*.json"),
                  *baseline.glob("*.safetensors"),
                  *(Path(r["source"]) for r in frozen["windows"])]
        state["input_sha256"] = {str(p): digest(p) for p in sorted(set(inputs))}
        state["runtime"] = {"python": sys.version, "executable": sys.executable,
                            "lerobot_revision": subprocess.check_output(
                                ["git", "-C", str(args.lerobot), "rev-parse", "HEAD"], text=True).strip()}
        state["known_departures"] = ["20 Hz versus BFL 30 Hz recipe; all current streams agree at 20 Hz",
                                     "Official public LeRobot revision differs from BFL private fork pin"]

        def stage(name, command, timeout=600):
            state["stage"] = name
            state["commands"].append({"stage": name, "argv": command, "timeout_seconds": timeout})
            write_json(status, state)
            print(json.dumps({"stage": name}), flush=True)
            run_stage(command, output / f"{name}.log", timeout)

        prefix = [sys.executable, str(code / "fit_audit.py"), "--root", str(root), "--windows", str(windows)]
        stage("contract-audit", [*prefix, "--checkpoint", str(baseline),
                                 "--output", str(output / "contract-audit.json"), "--audit-only"])
        if args.audit_only:
            state.update(stage="audit-completed", completed=True)
            return
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
        if len(gpu.strip().splitlines()) != 1 or int(gpu.strip()) < 37000:
            raise ValueError(f"FIT_VRAM: require one GPU with >=37000 MiB free, observed {gpu.strip()}")
        state["gpu_free_mib_before"] = int(gpu.strip())
        stage("before", [*prefix, "--checkpoint", str(baseline), "--output", str(output / "before.json")])
        stage("training", [sys.executable, str(code / "fit_train.py"), "--windows", str(windows),
                           "--evidence", str(output / "optimization.json"),
                           f"--config_path={output / 'config.json'}", f"--policy.path={baseline}"], timeout=1800)
        optimization = json.loads((output / "optimization.json").read_text())
        if (not optimization.get("completed") or optimization["microbatches"] != args.steps
                or optimization["optimizer_updates"] != args.steps // 4):
            raise ValueError("FIT_UPDATES: actual optimizer budget differs from the frozen configuration")
        checkpoint = output / f"training/checkpoints/{args.steps:06d}/pretrained_model"
        for name in ["adapter_config.json", "adapter_model.safetensors", "policy_preprocessor.json",
                     "policy_postprocessor.json"]:
            if not (checkpoint / name).is_file():
                raise ValueError(f"FIT_CHECKPOINT: missing {checkpoint / name}")
        state["saved_checkpoint_sha256"] = {p.name: digest(p) for p in checkpoint.glob("*.safetensors")}
        if digest(checkpoint / "adapter_model.safetensors") == digest(baseline / "adapter_model.safetensors"):
            raise ValueError("FIT_SERIALIZATION: saved adapter is identical to warm start")
        stage("after", [*prefix, "--checkpoint", str(checkpoint), "--output", str(output / "after.json")])
        for name, selected in [("before-rollouts", baseline), ("after-rollouts", checkpoint)]:
            stage(name, [sys.executable, str(code / "evaluate.py"), "--root", str(root),
                         "--checkpoint", str(selected), "--output", str(output / name),
                         "--atomic", "--atomic-split", "both", "--episodes", "1", "--skills", "pick_gear"])
        before = json.loads((output / "before.json").read_text())
        after = json.loads((output / "after.json").read_text())
        rollouts = json.loads((output / "after-rollouts/summary.json").read_text())
        result = decide(before, after, rollouts)
        result.update(baseline=str(baseline), checkpoint=str(checkpoint), windows_sha256=digest(windows),
                      limitations=["Eight fitted windows do not cover an entire skill.",
                                   "One train and one validation scene; no success-rate claim.",
                                   "No automatic promotion or second training run."])
        write_json(output / "decision.json", result)
        state.update(stage="completed", completed=True, decision=result)
    except BaseException as exc:
        state.update(failed_stage=state["stage"], stage="failed", error=f"{type(exc).__name__}: {exc}",
                     traceback=traceback.format_exc())
        raise
    finally:
        state["ended_unix"] = time.time()
        write_json(status, state)
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lerobot", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--audit-only", action="store_true")
    main(parser.parse_args())
