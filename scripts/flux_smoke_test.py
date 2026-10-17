"""WS1 hour-0-6 milestone: load FLUX 3 Action, run inference on a dummy/saved
observation, report latency. No hardware is touched."""

import argparse
import statistics
import time

import yaml

from liminal.core.types import Observation
from liminal.policy.flux import FluxConfig, FluxPolicyBackend

ap = argparse.ArgumentParser()
ap.add_argument("--policy", default="configs/policy/flux3_so101.yaml")
ap.add_argument("--iters", type=int, default=10)
args = ap.parse_args()

cfg = {k: v for k, v in yaml.safe_load(open(args.policy)).items() if k not in ("instances", "devices")}
cfg["camera_roles"] = tuple(cfg.get("camera_roles", ("scene", "wrist")))
backend = FluxPolicyBackend(FluxConfig(**cfg))
backend.load()

import numpy as np  # noqa: E402

# Option A: one arm ("right", 6-D). Option B: the "both" group (12-D, one wrist image per arm).
target = "right" if backend.cfg.action_dim == 6 else "both"
frame = np.zeros((480, 640, 3), dtype=np.uint8)
images = {role: frame for role in backend.cfg.camera_roles}
history = [Observation(target, [0.0] * backend.cfg.action_dim, images) for _ in range(backend.cfg.obs_history)]
lat = []
for _ in range(args.iters):
    t = time.monotonic()
    chunk = backend.predict(history, "Pick up the red block and place it in the bowl", target)
    lat.append(time.monotonic() - t)
print(f"chunk: {len(chunk)} x {len(chunk[0])}")
print(f"latency: median {statistics.median(lat)*1000:.0f} ms, max {max(lat)*1000:.0f} ms")
budget = backend.cfg.execute_steps / backend.cfg.control_hz
print(f"one chunk executes {backend.cfg.execute_steps} steps @ {backend.cfg.control_hz} Hz = {budget:.2f}s")
# FluxPolicyBackend infers then executes, so the arm pauses ~latency between chunks.
# If that pause is visible, prefetch the next chunk during execution (TASK.md WS1 stretch).
print(f"pause between chunks ~{max(lat):.2f}s ({max(lat) / budget:.0%} of chunk time)")
