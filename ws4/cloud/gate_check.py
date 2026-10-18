"""Gates (b) and (c) before robot time: from rest, the first chunk must move, also under live-like perturbations.

The request is built as the runner's first request: the state at the take's frame 0 repeated over the 8-tick history,
commands = that state, both image snapshots = frame 0 (prepared exactly as the runner prepares camera frames). Per
held-out take and per perturbation it reports the summed arm displacement of the first chunk (5 arm joints, step 32
vs the start state). Gate: every take moves at least --min-deg under every perturbation. Perturbations are applied to
the model-input images (256 x 256 after prepare_image): wrist or scene shifted by +-2 px (edge pixels repeated),
brightness x0.5 on the wrist and on both cameras, the knife-edge cases seen live on 18 Oct.

    python ws4/cloud/gate_check.py --package BASE [--adapter ADAPTER] --root DATA --episodes 63 64 65 \\
        --task "put the green pen in the green slot" --out gate.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

SCENE, WRIST = "observation.images.scene", "observation.images.wrist"


def shift(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Translate an (H, W, 3) image by (dx, dy) pixels, repeating the edge pixels."""
    h, w = img.shape[:2]
    padded = np.pad(img, ((abs(dy), abs(dy)), (abs(dx), abs(dx)), (0, 0)), mode="edge")
    top, left = abs(dy) - dy, abs(dx) - dx
    return padded[top : top + h, left : left + w]


def brighten(img: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


PERTURBATIONS = {
    "none": {},
    "wrist +2px x": {WRIST: lambda x: shift(x, 2, 0)},
    "wrist -2px x": {WRIST: lambda x: shift(x, -2, 0)},
    "wrist +2px y": {WRIST: lambda x: shift(x, 0, 2)},
    "wrist -2px y": {WRIST: lambda x: shift(x, 0, -2)},
    "scene +2px x": {SCENE: lambda x: shift(x, 2, 0)},
    "wrist x0.5": {WRIST: lambda x: brighten(x, 0.5)},
    "both x0.5": {WRIST: lambda x: brighten(x, 0.5), SCENE: lambda x: brighten(x, 0.5)},
}


def first_requests(root: str, episodes: list[int], contract) -> dict[int, tuple[np.ndarray, dict[str, np.ndarray]]]:
    """Per take: frame-0 state and the frame-0 model-input image of each camera."""
    from flux.cameras import CameraView, prepare_image
    from flux.package_tools import dataset_arrays
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    arr = dataset_arrays("gate", root=root)
    ep_idx, states = arr["episode_index"], arr["observation.state"].astype(float)
    ds = LeRobotDataset("gate", root=root)
    out = {}
    for e in episodes:
        i0 = int(np.flatnonzero(ep_idx == e)[0])
        frame = ds[i0]
        imgs = {}
        for k in contract.camera_keys:
            rgb = (frame[k].permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
            imgs[k] = prepare_image(rgb, CameraView(k), contract.image_hw)
        out[e] = (states[i0], imgs)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--package", required=True, help="base package")
    ap.add_argument("--adapter", default=None, help="LoRA checkpoint dir; omit for the base alone")
    ap.add_argument("--root", required=True, help="per-arm dataset dir")
    ap.add_argument("--episodes", type=int, nargs="+", required=True, help="held-out takes")
    ap.add_argument("--task", required=True)
    ap.add_argument("--min-deg", type=float, default=20.0)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    from flux.backends import PolicyRequest
    from lora_adapter import load_policy, to_backend

    overrides = {"guidance_scale": args.guidance, "guidance_scale_action": args.guidance}
    backend = to_backend(load_policy(args.package, args.adapter), args.package, overrides, True, "gate")
    c = backend.contract
    starts = first_requests(args.root, args.episodes, c)
    rows = {}
    for name, perturb in PERTURBATIONS.items():
        disp = {}
        for e, (s0, imgs) in starts.items():
            snaps = {k: np.stack([perturb.get(k, lambda x: x)(imgs[k]) for _ in c.snapshots]) for k in c.camera_keys}
            req = PolicyRequest(args.task, np.repeat(s0[None], c.n_obs_steps, 0), np.repeat(s0[None], c.n_obs_steps, 0),
                                snaps, c.snapshots)  # fmt: skip
            pred = np.asarray(backend.predict(req).commands)[: c.n_action_steps]
            disp[e] = float(np.abs(pred[-1, :5] - s0[:5]).sum())
        rows[name] = disp
        print(f"{name:14s} " + "  ".join(f"{e}: {d:5.1f}" for e, d in disp.items())
              + f"   min {min(disp.values()):.1f}", flush=True)  # fmt: skip
    worst = min(min(d.values()) for d in rows.values())
    passed = worst >= args.min_deg
    print(f"GATE {'PASS' if passed else 'FAIL'}: smallest first-chunk arm displacement {worst:.1f} deg "
          f"(threshold {args.min_deg:.0f})")  # fmt: skip
    Path(args.out).write_text(json.dumps({"args": vars(args), "first_chunk_arm_displacement_deg": rows,
                                          "worst_deg": worst, "pass": passed}, indent=2))  # fmt: skip
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
