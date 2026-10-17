"""GPU job (optional): put gated TTT layers into the loaded 7B FLUX 3 Action and measure what it costs.

On the real SO-101 package, with one real observation (a re-plan from a recorded episode, or a synthetic one):

1. repeatability: two plain predictions with the fixed inference seed (are GPU kernels deterministic here?);
2. gate 0: attach ``FluxTTTAdapter`` (gated TTT after joint blocks, on the action tokens) with the gate at 0 and
   compare the action chunk with the plain one (expected: bit-identical, or as close as step 1);
3. gate 1e-3 (RoboTTT's init): relative change of the chunk;
4. latency (median over ``--runs`` predictions) and peak VRAM, plain vs with the adapter; adapter size.

Run from ``ws4/`` in the LeRobot env on the GPU (``ws4/cloud/bootstrap.sh``):

    PYTHONPATH=. python -m ttt.flux_scale_check --layers 24 25 26 27 --runs 10 --episode /workspace/ttt/sim/<ep>
    PYTHONPATH=. python -m ttt.flux_scale_check --text-encoder-on-cpu ...     # 32 GB cards

Writes a JSON report next to ``--out`` (default ``ttt_scale_check.json``). ``run_scale_check`` is also exercised on
CPU with a tiny LeRobot policy in ``tests/test_lerobot_integration.py``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _sync():
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed(backend, request, runs: int) -> tuple[np.ndarray, list[float]]:
    out, lat = None, []
    for _ in range(runs):
        t0 = time.perf_counter()
        out = backend.predict(request).commands
        _sync()
        lat.append(time.perf_counter() - t0)
    return out, lat


def run_scale_check(backend, request, layers: list[int], inner_dim: int | None = 512, runs: int = 5) -> dict:
    import torch

    from ttt.flux_hook import FluxTTTAdapter

    dit = backend.policy.dit
    modality = backend.policy.modality
    device = next(dit.parameters()).device
    cuda = device.type == "cuda"
    report: dict = {"layers": list(layers), "inner_dim": inner_dim, "hidden_size": int(dit.hidden_size)}

    if cuda:
        torch.cuda.reset_peak_memory_stats()
    plain_a, plain_lat = _timed(backend, request, runs)
    plain_b = backend.predict(request).commands
    report["plain"] = {"median_s": statistics.median(plain_lat), "runs": plain_lat}
    if cuda:
        report["plain"]["peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
    report["repeatability_max_abs"] = float(np.abs(plain_a - plain_b).max())

    adapter = FluxTTTAdapter(
        dit.hidden_size, layers, (modality, f"{modality}_cond"), gate_init=0.0, inner_dim=inner_dim
    )
    adapter.to(device).eval().attach(dit)
    report["adapter_params"] = int(sum(p.numel() for p in adapter.parameters()))
    try:
        gate0 = backend.predict(request).commands
        report["gate0_max_abs_vs_plain"] = float(np.abs(gate0 - plain_a).max())
        report["gate0_bit_identical"] = bool(np.array_equal(gate0, plain_a))
        with torch.no_grad():
            for layer in adapter.layers.values():
                layer.gate.alpha.fill_(1e-3)
        if cuda:
            torch.cuda.reset_peak_memory_stats()
        gated, gated_lat = _timed(backend, request, runs)
        scale = float(np.abs(plain_a).max()) or 1.0
        report["gate1e-3_max_abs_vs_plain"] = float(np.abs(gated - plain_a).max())
        report["gate1e-3_rel_vs_plain"] = float(np.abs(gated - plain_a).max() / scale)
        report["with_adapter"] = {"median_s": statistics.median(gated_lat), "runs": gated_lat}
        if cuda:
            report["with_adapter"]["peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
        report["added_latency_s"] = report["with_adapter"]["median_s"] - report["plain"]["median_s"]
    finally:
        adapter.detach_hooks()
    return report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", default="black-forest-labs/flux-3-action-so101")
    ap.add_argument("--layers", type=int, nargs="+", default=[24, 25, 26, 27])
    ap.add_argument("--inner-dim", type=int, default=512)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--episode", type=Path, default=None, help="use this episode's middle re-plan as the observation")
    ap.add_argument("--text-encoder-on-cpu", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("ttt_scale_check.json"))
    args = ap.parse_args(argv)

    from flux.backends import LeRobotFlux3Backend

    backend = LeRobotFlux3Backend.from_pretrained(
        args.checkpoint,
        device=args.device,
        overrides={"compile_model": False},
        text_encoder_on_cpu=args.text_encoder_on_cpu,
    )
    if args.episode is not None:
        from ttt.extract_flux_features import episode_requests

        reqs = list(episode_requests(args.episode, backend.contract))
        request = reqs[len(reqs) // 2][1]
    else:
        from flux.measure_latency import synthetic_request

        request = synthetic_request(backend.contract, "Put the apple in the box")
    report = run_scale_check(backend, request, args.layers, args.inner_dim, args.runs)
    report.update(checkpoint=args.checkpoint, observation=str(args.episode or "synthetic"))
    args.out.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k not in ("plain", "with_adapter")}, indent=1))
    print("plain", report["plain"].get("median_s"), "with adapter", report["with_adapter"].get("median_s"))


if __name__ == "__main__":
    main()
