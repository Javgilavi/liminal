"""Measure FLUX 3 Action chunk latency and VRAM on the GPU box, for one and two arms.

Runs the real package through ``LeRobotFlux3Backend`` with synthetic observations (random images, a static
state), so it needs the weights but no robot. Reports per-chunk latency (median, p90), peak VRAM, and the
settings, as JSON. Compare against BFL's published DROID numbers on "RTX 6000 Pro": 772.6 ms BF16 for the
4-step guided base checkpoint (https://bfl.ai/models/flux-3-action, latency table); SO-101 is not published.

    python -m flux.measure_latency --checkpoint black-forest-labs/flux-3-action-so101 --runs 10
    python -m flux.measure_latency ... --override guidance_scale=1.0     # single-pass sampling
    python -m flux.measure_latency ... --two-instances                   # two models on one GPU, in parallel
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import threading
import time

import numpy as np

from .backends import LeRobotFlux3Backend, PolicyRequest
from .contract import ModelContract


def synthetic_request(contract: ModelContract, task: str, seed: int = 0) -> PolicyRequest:
    rng = np.random.default_rng(seed)
    n, d = contract.n_obs_steps, contract.action_dim
    state = np.zeros(d)
    state[list(contract.absolute_dims)] = 50.0
    images = {
        key: rng.integers(0, 255, (len(contract.snapshots), *contract.image_hw, 3), dtype=np.uint8)
        for key in contract.camera_keys
    }
    return PolicyRequest(task, np.tile(state, (n, 1)), np.tile(state, (n, 1)), images, contract.snapshots)


def time_backend(backend: LeRobotFlux3Backend, runs: int, tasks: list[str]) -> list[float]:
    out = []
    for i in range(runs):
        response = backend.predict(synthetic_request(backend.contract, tasks[i % len(tasks)], seed=i))
        out.append(response.latency_s)
    return out


def summary(latencies: list[float]) -> dict[str, float]:
    ordered = sorted(latencies)
    return {
        "median_ms": 1000 * statistics.median(ordered),
        "p90_ms": 1000 * ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
        "min_ms": 1000 * ordered[0],
        "n": len(ordered),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", default="black-forest-labs/flux-3-action-so101")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--override", action="append", default=[], metavar="KEY=JSON")
    parser.add_argument("--text-encoder-on-cpu", action="store_true")
    parser.add_argument("--two-instances", action="store_true", help="load a second copy and run both concurrently")
    args = parser.parse_args(argv)
    import torch

    logging.basicConfig(level=logging.INFO)  # shows the VAE's chosen NATTEN backend on the first prediction
    overrides = {k: json.loads(v) for k, v in (item.split("=", 1) for item in args.override)}
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    backend = LeRobotFlux3Backend.from_pretrained(
        args.checkpoint, device=args.device, overrides=overrides, text_encoder_on_cpu=args.text_encoder_on_cpu
    )
    load_s = time.perf_counter() - t0
    tasks = ["push the box to the side", "pick up the cup", "place the cup on the plate"]
    time_backend(backend, args.warmup, tasks)
    report = {
        "checkpoint": args.checkpoint,
        "overrides": overrides,
        "contract": backend.contract.__dict__,
        "load_s": load_s,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "one_model_sequential": summary(time_backend(backend, args.runs, tasks)),
        "peak_vram_gib_one_model": torch.cuda.max_memory_allocated() / 2**30,
    }
    if args.two_instances:
        second = LeRobotFlux3Backend.from_pretrained(
            args.checkpoint, device=args.device, overrides=overrides, text_encoder_on_cpu=args.text_encoder_on_cpu
        )
        time_backend(second, args.warmup, tasks)
        results: dict[str, list[float]] = {}

        def worker(name: str, b: LeRobotFlux3Backend) -> None:
            results[name] = time_backend(b, args.runs, tasks)

        threads = [threading.Thread(target=worker, args=(n, b)) for n, b in (("a", backend), ("b", second))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        report["two_models_concurrent"] = {k: summary(v) for k, v in results.items()}
        report["peak_vram_gib_two_models"] = torch.cuda.max_memory_allocated() / 2**30
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
