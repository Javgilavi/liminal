"""Held-out action error of ACT checkpoints, with the protocol and metrics of ``single_arm/offline_diag.py``.

Same frames (``offline_diag.sample_frames``: 200, seed 0, from the episodes not in ``--exclude``), same anchor (the
recorded command before t), same comparison (first 32 predicted absolute commands vs the recorded a[t..t+31]) and
the same metrics and table (``offline_diag.metrics`` / ``render``), so rows line up with the FLUX curve in
``ws4/research/pen_lora.md``. ACT sees what the runner would send it: the current state s[t] and the current
frame of each camera as uint8 (``flux.cameras.prepare_image`` at the checkpoint's resolution), through
``act.backend.ActBackend.predict``. Frames are decoded once and shared by all checkpoints.

    python ws4/act/offline_eval.py --root /workspace/real_lora/pen_in_right_final --repo-id local/pen_in_right_final \\
        --exclude <all but the 12 held-out> --checkpoint act_20k=runs/act_pen84/checkpoints/020000/pretrained_model \\
        --checkpoint act_50k=... --out eval/act_pen84.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

WS4 = Path(__file__).resolve().parents[1]
for path in (WS4, WS4 / "single_arm"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from offline_diag import N_EXEC, metrics, render, sample_frames  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--repo-id", required=True)
    p.add_argument("--root", default=None)
    p.add_argument("--task", default="put the red pen in its slot in the cardboard box")
    p.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PRETRAINED_MODEL_DIR")
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exclude", type=int, nargs="*", default=[], help="every episode except the held-out takes")
    p.add_argument("--device", default="auto")
    p.add_argument("--min-move", type=float, nargs=6, default=[2, 2, 2, 2, 2, 5], help="direction test threshold")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    from flux.backends import PolicyRequest
    from flux.cameras import CameraView, prepare_image
    from flux.package_tools import dataset_arrays
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from act.backend import ActBackend

    arrays = dataset_arrays(args.repo_id, root=args.root)
    states = arrays["observation.state"].astype(float)
    actions = arrays["action"].astype(float)
    ep = arrays["episode_index"]
    frames = sample_frames(ep, args.samples, args.seed, args.exclude)
    episodes = sorted({int(e) for e in ep[frames]})
    print(f"{len(frames)} frames from episodes {episodes}", flush=True)
    backends, paths = {}, {}
    for item in args.checkpoint:
        name, paths[name] = item.split("=", 1)
        backends[name] = ActBackend.from_pretrained(paths[name], device=args.device, name=name)
    contract = next(iter(backends.values())).contract
    for name, b in backends.items():
        if (b.contract.camera_keys, b.contract.image_hw) != (contract.camera_keys, contract.image_hw):
            raise SystemExit(f"{name}: cameras differ from the first checkpoint's")
    ds = LeRobotDataset(args.repo_id, root=args.root)

    def image(idx: int, key: str) -> np.ndarray:
        img = ds[int(idx)][key]  # (3, H, W) float [0, 1], as offline_diag reads it
        rgb = (img.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
        return prepare_image(rgb, CameraView(key), contract.image_hw)

    t0 = time.time()
    requests = []
    for t in frames:
        imgs = {key: image(t, key)[None] for key in contract.camera_keys}
        requests.append(PolicyRequest(args.task, states[t][None], actions[t - 1][None], imgs, contract.snapshots))
    print(f"decoded {len(frames)} frames in {time.time() - t0:.0f} s", flush=True)
    true = np.stack([actions[t : t + N_EXEC] for t in frames])
    anchor = np.stack([actions[t - 1] for t in frames])
    min_move = np.array(args.min_move)
    hold = metrics(np.repeat(anchor[:, None], N_EXEC, axis=1), true, anchor, min_move)
    result = {"args": vars(args), "frames": frames.tolist(), "episodes": episodes, "hold": hold, "variants": {}}
    preds = {}
    for name, backend in backends.items():
        out, lat = [], []
        for req in requests:
            r = backend.predict(req)
            out.append(r.commands[:N_EXEC])
            lat.append(r.latency_s)
        preds[name] = np.stack(out)
        result["variants"][name] = {**metrics(preds[name], true, anchor, min_move),
                                    "latency_s": float(np.median(lat)), "checkpoint": paths[name]}  # fmt: skip
        print(f"{name}: median {np.median(lat) * 1e3:.0f} ms per chunk", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, default=float))
    np.savez_compressed(Path(args.out).with_suffix(".npz"), true=true, anchor=anchor, frames=frames,
                        **{f"pred_{k}": v for k, v in preds.items()})  # fmt: skip
    print(render(result))


if __name__ == "__main__":
    main()
