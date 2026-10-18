"""Evaluate or serve a LeRobot flux3 LoRA checkpoint without writing a merged 14 GB package.

The adapter (``<run>/checkpoints/<step>/pretrained_model_ema``) is folded into the base package's weights in memory
(``peft.PeftModel.merge_and_unload``, the same fold as ``real_lora.sh merge``), so the volume only holds the 0.45 GB
adapters. The base package supplies the processors (the run trains with them; the checkpoint's copies are equal).

    # held-out action-error eval: ws4/single_arm/offline_diag.py unchanged, variant A = the package's own statistics
    python ws4/cloud/lora_adapter.py eval --package BASE [--adapter CKPT] -- --root DATA --repo-id NAME \\
        --task "put the red pen in its slot in the cardboard box" --variants A --exclude ... --out diag.json
    # serve on the robot protocol (flux.remote), one model per NAME
    python ws4/cloud/lora_adapter.py serve --port 8770 --token-file /workspace/flux_token \\
        --package pen5k=BASE+CKPT --override guidance_scale=1.0 --override guidance_scale_action=1.0
    # swap checkpoints with a short gap: load and fold on the CPU while PID keeps serving, then replace it
    python ws4/cloud/lora_adapter.py serve ... --replace-pid PID
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

WS4 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WS4))


def load_policy(package: str, adapter: str | None = None):
    """Flux3Policy on the CPU, with the adapter folded in when given."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.flux3 import Flux3Policy

    config = PreTrainedConfig.from_pretrained(package)
    config.device = "cpu"  # from_pretrained moves the policy to config.device itself
    policy = Flux3Policy.from_pretrained(package, config=config)
    if adapter:
        from peft import PeftModel

        policy = PeftModel.from_pretrained(policy, str(adapter)).merge_and_unload()
    return policy


def to_backend(policy, package: str, overrides: dict, text_encoder_on_cpu: bool, name: str, device: str = "cuda"):
    """Same end state as ``LeRobotFlux3Backend.from_pretrained``, without its load-time peak (Qwen is detached
    while the trunk moves, as in ``offline_diag.load_backend``)."""
    from flux.backends import LeRobotFlux3Backend
    from lerobot.policies.factory import make_pre_post_processors

    for key, value in overrides.items():
        if not hasattr(policy.config, key):
            raise AttributeError(f"Flux3Config has no field {key!r}")
        setattr(policy.config, key, value)
    policy.config.device = device
    text_encoder = policy.frozen.text_encoder
    if text_encoder_on_cpu:
        object.__setattr__(policy.frozen, "text_encoder", None)  # FrozenComponents is not an nn.Module
    policy.to(device).eval()
    object.__setattr__(policy.frozen, "text_encoder", text_encoder)
    pre, post = make_pre_post_processors(policy.config, pretrained_path=package,
                                         preprocessor_overrides={"device_processor": {"device": device}})  # fmt: skip
    return LeRobotFlux3Backend(policy, pre, post, name=name)


def eval_main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(prog="lora_adapter.py eval")
    ap.add_argument("--package", required=True, help="base package the run trained from")
    ap.add_argument("--adapter", default=None, help="checkpoint adapter dir; omit for the zero-shot base")
    ap.add_argument("--guidance-action", type=float, default=None,
                    help="action guidance when it differs from offline_diag's --guidance (which sets both)")
    ap.add_argument("diag_args", nargs=argparse.REMAINDER, help="-- then offline_diag.py arguments (not --package)")
    args = ap.parse_args(argv)
    rest = args.diag_args[1:] if args.diag_args[:1] == ["--"] else args.diag_args
    sys.path.insert(0, str(WS4 / "single_arm"))
    import offline_diag

    def load_backend(package, overrides, text_encoder_on_cpu, device="cuda"):
        if args.guidance_action is not None:
            overrides = {**overrides, "guidance_scale_action": args.guidance_action}
        label = f"{package}+{args.adapter}" if args.adapter else package
        return to_backend(load_policy(package, args.adapter), package, overrides, text_encoder_on_cpu, label, device)

    offline_diag.load_backend = load_backend  # build_backends looks it up at call time
    sys.argv = ["offline_diag.py", "--package", args.package, *rest]
    offline_diag.main()
    out = Path(rest[rest.index("--out") + 1])
    result = json.loads(out.read_text())
    result["adapter"] = args.adapter
    result["guidance_action"] = args.guidance_action
    out.write_text(json.dumps(result, indent=2, default=float))


def replace(pid: int, timeout_s: float = 60.0) -> None:
    """Stop the server being replaced and wait until it has exited (its GPU memory and port are then free)."""
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + timeout_s
    while _alive(pid) and time.monotonic() < deadline:
        time.sleep(0.5)
    if _alive(pid):
        os.kill(pid, signal.SIGKILL)
        time.sleep(2)
    print(f"replaced server pid {pid}", flush=True)


def _alive(pid: int) -> bool:
    """False once the process is gone or a zombie (a container's init may not reap it)."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except (FileNotFoundError, IndexError):
        return False


def serve_main(argv: list[str]) -> None:
    from flux.remote import make_server

    ap = argparse.ArgumentParser(prog="lora_adapter.py serve")
    ap.add_argument("--package", action="append", required=True, metavar="NAME=BASE[+ADAPTER]")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--token-file", default=None)
    ap.add_argument("--text-encoder-on-cpu", action="store_true")
    ap.add_argument("--override", action="append", default=[], metavar="KEY=JSON")
    ap.add_argument("--replace-pid", type=int, default=None,
                    help="server to stop once the models are loaded and folded on the CPU (short swap gap)")
    args = ap.parse_args(argv)
    overrides = {k: json.loads(v) for k, v in (item.split("=", 1) for item in args.override)}
    loaded = []
    for item in args.package:
        name, spec = item.split("=", 1)
        base, _, adapter = spec.partition("+")
        loaded.append((name, spec, base, load_policy(base, adapter or None)))
        print(f"{name}: {spec} loaded on the CPU", flush=True)
    if args.replace_pid:
        replace(args.replace_pid)
    backends = {}
    for name, spec, base, policy in loaded:
        backends[name] = to_backend(policy, base, overrides, args.text_encoder_on_cpu, spec)
        print(f"{name}: {spec} -> {backends[name].contract}", flush=True)
    token = os.environ.get("FLUX_TOKEN")
    if args.token_file:
        token = Path(args.token_file).read_text().strip()
    server = make_server(backends, args.host, args.port, token)
    print(f"serving {sorted(backends)} on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("eval", "serve"):
        raise SystemExit("usage: lora_adapter.py eval|serve ...")
    {"eval": eval_main, "serve": serve_main}[sys.argv[1]](sys.argv[2:])
