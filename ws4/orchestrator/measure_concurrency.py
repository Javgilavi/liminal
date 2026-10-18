"""What two concurrent FLUX arms look like when they share one model: motion duty cycle, waiting, chunk latency.

Both arms run a FLUX skill at the same time through ``flux.dual_arm_runner.DualArmRunner`` (as the executor does),
on a static robot with 640x480 frames, for ``--seconds`` of wall time at 30 Hz. Per arm it reports the fraction of
ticks the arm moved (executing a chunk) or waited (inference pending), chunks per minute, and chunk latency.

    # Model of the RunPod pod (no GPU needed): 1.46 s per chunk, measured there (ws4/cloud/RUNPOD.md)
    python -m orchestrator.measure_concurrency --backend fake --latency 1.46 --models 1   # one shared model
    python -m orchestrator.measure_concurrency --backend fake --latency 1.46 --models 2   # two instances
    python -m orchestrator.measure_concurrency --backend fake --latency 1.46 --models 1 --sync
    # The real pod, through the SSH tunnel (ws4/cloud/RUNPOD.md); both arms on package so101 = one model:
    python -m orchestrator.measure_concurrency --backend remote --server http://127.0.0.1:8765 --left-package so101 \\
        --right-package so101

Writes a JSON report to ws4/orchestrator/runs/concurrency_<backend>_<models>_<mode>.json.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from . import PACKAGE_DIR  # noqa: F401  (sys.path setup)

PROMPTS = {"left": "Put the apple in the box", "right": "Put the yellow block in the box"}


class StaticRobot:
    def __init__(self, hw=(480, 640)):
        from flux.contract import IMG_HEAD, IMG_LEFT_WRIST, IMG_RIGHT_WRIST, OBS_STATE

        rng = np.random.default_rng(0)
        self.key = OBS_STATE
        self.state = np.array([0, -10, 10, 75, 0, 30] * 2, dtype=float)
        self.images = {k: rng.integers(0, 255, (*hw, 3), dtype=np.uint8)
                       for k in (IMG_HEAD, IMG_LEFT_WRIST, IMG_RIGHT_WRIST)}

    def get_observation(self):
        return {self.key: self.state.copy(), **self.images}

    def send_action(self, action):
        self.state = np.asarray(action, dtype=float).copy()


def measure(args: argparse.Namespace) -> dict:
    from flux.backends import FakeBackend
    from flux.contract import SO101_CONTRACT
    from flux.dual_arm_runner import DualArmRunner, per_arm_target

    packages = {"left": args.left_package, "right": args.right_package}
    if args.backend == "fake":
        shared = FakeBackend(SO101_CONTRACT, delay_s=args.latency)
        backends = {"m0": shared} if args.models == 1 else {
            "m0": FakeBackend(SO101_CONTRACT, delay_s=args.latency),
            "m1": FakeBackend(SO101_CONTRACT, delay_s=args.latency)}
        names = {"left": "m0", "right": "m0" if args.models == 1 else "m1"}
    else:
        from flux.remote import RemoteBackend

        token = Path(args.token_file).read_text().strip() if args.token_file else None
        backends = {p: RemoteBackend(args.server, p, token=token) for p in set(packages.values())}
        names = packages
    targets = [dataclasses.replace(per_arm_target(arm, names[arm]), name=arm) for arm in ("left", "right")]
    runner = DualArmRunner(StaticRobot(), targets, backends, async_inference=not args.sync)
    if args.warmup:
        for arm in ("left", "right"):  # first request per prompt can include a text encode on the pod
            runner.start(arm, PROMPTS[arm])
        runner.run(args.warmup, fps=30)
        for arm in ("left", "right"):
            runner.stop(arm)
    for arm in ("left", "right"):
        runner.start(arm, PROMPTS[arm])
    counts = {arm: {"executing": 0, "replanned": 0, "waiting": 0} for arm in ("left", "right")}
    latencies: dict[str, list[float]] = {"left": [], "right": []}
    tick_times = []
    both = 0

    def on_tick(report):
        nonlocal both
        tick_times.append(time.monotonic())
        moving = 0
        for arm in ("left", "right"):
            st = report.status.get(arm, "waiting")
            counts[arm][st] = counts[arm].get(st, 0) + 1
            moving += st in ("executing", "replanned")
            if st == "replanned" and arm in report.latencies:
                latencies[arm].append(report.latencies[arm])
        both += moving == 2

    start = time.monotonic()
    reports = runner.run(args.seconds, fps=30, on_tick=on_tick)
    wall = time.monotonic() - start
    runner.close()
    n = len(reports)
    slots = wall * 30.0  # 30 Hz control slots: in sync mode the loop stalls during inference and ticks are lost
    out = {"backend": args.backend, "models": args.models if args.backend == "fake" else sorted(set(packages.values())),
           "mode": "sync" if args.sync else "async",
           "latency_model_s": args.latency if args.backend == "fake" else None,
           "seconds": round(wall, 2), "ticks": n, "tick_rate_hz": round(n / wall, 1), "per_arm": {},
           "both_arms_moving": round(both / slots, 3)}
    gaps = np.diff(tick_times) if len(tick_times) > 1 else np.array([0.0])
    out["max_tick_gap_s"] = round(float(gaps.max()), 3)
    for arm in ("left", "right"):
        c = counts[arm]
        moving = c["executing"] + c["replanned"]
        lat = latencies[arm]
        out["per_arm"][arm] = {
            "moving": round(moving / slots, 3), "not_moving": round(1 - moving / slots, 3),
            "chunks": c["replanned"], "chunks_per_min": round(60 * c["replanned"] / wall, 1),
            "backend_latency_median_s": round(float(np.median(lat)), 3) if lat else None,
            "motion_time_per_chunk_s": round(SO101_CONTRACT.n_action_steps / SO101_CONTRACT.fps, 3),
        }
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--backend", choices=("fake", "remote"), default="fake")
    ap.add_argument("--latency", type=float, default=1.46, help="fake: seconds per chunk")
    ap.add_argument("--models", type=int, choices=(1, 2), default=1, help="fake: shared model or one per arm")
    ap.add_argument("--sync", action="store_true", help="synchronous inference (blocks the whole control loop)")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--warmup", type=float, default=0.0, help="seconds of untimed warm-up first")
    ap.add_argument("--server", default="http://127.0.0.1:8765")
    ap.add_argument("--token-file", default=None)
    ap.add_argument("--left-package", default="so101")
    ap.add_argument("--right-package", default="so101")
    args = ap.parse_args(argv)
    result = measure(args)
    print(json.dumps(result, indent=2))
    tag = f"{args.backend}_{args.models if args.backend == 'fake' else 'pod'}_{result['mode']}"
    path = PACKAGE_DIR / "runs" / f"concurrency_{tag}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
