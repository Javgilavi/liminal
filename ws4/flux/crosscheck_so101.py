"""Cross-check our LeRobot backend against BFL's standalone SO-101 inference on the same observation.

BFL's ``examples/so101/inference.py`` (flux-action env) and ``LeRobotFlux3Backend`` (LeRobot env) are separate
implementations of the same package. Both treat a single observation as a new episode: the history is padded
with that frame and state, and the previous command is the state (flux-action@830f1ce:docs/setup.md:214-217;
lerobot processor_flux3.py:370-383). Same NPZ in, two ``(42, 6)`` absolute command chunks out.

    python -m flux.crosscheck_so101 write --out obs.npz [--scene scene.png --wrist wrist.png] [--state ...]
    # flux-action env:
    uv run --extra encoders python examples/so101/inference.py --weights black-forest-labs/flux-3-action-so101 \
        --observation obs.npz --prompt "pick up the red box" --output bfl.npy
    # LeRobot env:
    python -m flux.crosscheck_so101 predict --observation obs.npz --prompt "pick up the red box" --out ours.npy
    python -m flux.crosscheck_so101 compare bfl.npy ours.npy

If both draw the same sampling noise the chunks agree to rounding; if not, they differ like two samples of
one policy (same direction and scale, not equal). A sign flip, a swapped camera or a units mismatch shows up
as a large, systematic difference either way.
"""

from __future__ import annotations

import argparse

import numpy as np

from .contract import SO101_CONTRACT, SO101_SCENE, SO101_WRIST

NPZ_SCENE, NPZ_WRIST = "images.scene", "images.wrist"  # flux-action@830f1ce:docs/setup.md:172-176


def _image(path: str | None, seed: int) -> np.ndarray:
    hw = SO101_CONTRACT.image_hw
    if path:
        from PIL import Image

        from .cameras import resize_uint8

        return resize_uint8(np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8), hw)
    yy, xx = np.mgrid[0 : hw[0], 0 : hw[1]]
    rng = np.random.default_rng(seed)
    img = np.stack([xx * 255 // hw[1], yy * 255 // hw[0], rng.integers(0, 255, hw)], axis=-1)
    return img.astype(np.uint8)


def write_observation(out: str, scene: str | None, wrist: str | None, state: list[float]) -> None:
    np.savez(
        out,
        **{NPZ_SCENE: _image(scene, 0), NPZ_WRIST: _image(wrist, 1), "state": np.asarray(state, dtype=np.float32)},
    )


def predict(observation: str, prompt: str, out: str, checkpoint: str, device: str) -> np.ndarray:
    from .backends import LeRobotFlux3Backend, PolicyRequest

    with np.load(observation, allow_pickle=False) as data:
        scene, wrist, state = data[NPZ_SCENE], data[NPZ_WRIST], data["state"].astype(np.float64)
    backend = LeRobotFlux3Backend.from_pretrained(checkpoint, device=device)
    c = backend.contract
    window = np.tile(state, (c.n_obs_steps, 1))
    images = {SO101_SCENE: np.stack([scene] * len(c.snapshots)), SO101_WRIST: np.stack([wrist] * len(c.snapshots))}
    response = backend.predict(PolicyRequest(prompt, window, window.copy(), images, c.snapshots))
    np.save(out, response.commands[None].astype(np.float32), allow_pickle=False)
    print(f"saved {response.commands.shape} to {out} in {response.latency_s:.3f} s")
    return response.commands


def compare(a_path: str, b_path: str) -> dict[str, list[float]]:
    a, b = np.load(a_path).reshape(-1, 6), np.load(b_path).reshape(-1, 6)
    if a.shape != b.shape:
        raise ValueError(f"shapes differ: {a.shape} vs {b.shape}")
    diff = b - a
    report = {
        "max_abs_diff": np.abs(diff).max(axis=0).round(3).tolist(),
        "mean_diff": diff.mean(axis=0).round(3).tolist(),
        "net_motion_a": (a[-1] - a[0]).round(3).tolist(),
        "net_motion_b": (b[-1] - b[0]).round(3).tolist(),
    }
    for key, value in report.items():
        print(f"{key:14s} {value}")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write")
    w.add_argument("--out", required=True)
    w.add_argument("--scene")
    w.add_argument("--wrist")
    w.add_argument("--state", default="0,0,0,0,0,50", help="six values: degrees, gripper percent")
    p = sub.add_parser("predict")
    p.add_argument("--observation", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--checkpoint", default="black-forest-labs/flux-3-action-so101")
    p.add_argument("--device", default="cuda")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    args = parser.parse_args(argv)
    if args.cmd == "write":
        write_observation(args.out, args.scene, args.wrist, [float(x) for x in args.state.split(",")])
    elif args.cmd == "predict":
        predict(args.observation, args.prompt, args.out, args.checkpoint, args.device)
    else:
        compare(args.a, args.b)


if __name__ == "__main__":
    main()
