"""Offline data-format diagnostic: which joint convention / normalization lets the released SO-101 package best predict
OUR demonstrations? No robot. Decides the LoRA base.

For sampled frames t with a full history window, the request is built exactly as the runner builds it
(``flux.history`` semantics): states s[t-7..t], the command issued before each observation (recorded action[k-1]),
image snapshots at window ticks 0 and 7 (frames t-7 and t). The 42-step chunk is predicted and its first 32 absolute
commands are compared with the demo's recorded actions a[t..t+31].

Variants (same weights, same guidance and seed):
  A  package statistics, our LeRobot degrees as is.
  B  A0r: history quantiles recomputed on these episodes (``flux.package_tools.compute_quantiles``).
  C  flux-action issue #2 adapter (https://github.com/black-forest-labs/flux-action/issues/2): model = scale * ours
     + offset on states and commands, package statistics, predicted commands mapped back.
  D  sign flips only (shoulder_lift, wrist_roll), quantiles recomputed on the flipped data, predictions flipped back.
  E  package statistics with lift_model = 90 - lift and elbow_model = elbow + 90 (Oscar's mapping), mapped back.

Metrics per variant: per-joint MAE (degrees; gripper in percent) over the 32 executed steps; direction agreement per
joint (sign of predicted vs recorded displacement from the current command to step 32, counted where the recorded
move exceeds --min-move); and the same for a "hold" baseline that repeats the current command.

    python ws4/single_arm/offline_diag.py --root <per-arm dataset> --repo-id <name> --samples 200 \\
        --task "put the red pen in its slot in the cardboard box" --exclude <held-out episodes> --out diag.json
    python ws4/single_arm/offline_diag.py ... --backend fake      # CPU pipeline test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
# flux-action issue #2, experimental adapter for the package's legacy SO-100 convention (that reporter's calibration).
ISSUE2_SCALE = np.array([1, -1, 1, 1, -1, 1], dtype=float) * np.array([4095 / 4096] * 5 + [1.0])
ISSUE2_OFFSET = np.array([7.998, 90.220, 84.111, -4.482, -0.396, 0.0])
FLIP_SCALE = np.array([1, -1, 1, 1, -1, 1], dtype=float)
# Variant E (Oscar, 2026-10-17): lift_model = 90 - lift, elbow_model = elbow + 90, everything else as is.
E_SCALE = np.array([1, -1, 1, 1, 1, 1], dtype=float)
E_OFFSET = np.array([0, 90, 90, 0, 0, 0], dtype=float)
N_OBS, N_EXEC, CHUNK = 8, 32, 42


class Affine:
    """model = scale * ours + offset, per joint; ``inv`` maps model-space commands back to our units."""

    def __init__(self, scale=None, offset=None):
        self.scale = np.ones(6) if scale is None else np.asarray(scale, dtype=float)
        self.offset = np.zeros(6) if offset is None else np.asarray(offset, dtype=float)

    def __call__(self, x):
        return np.asarray(x, dtype=float) * self.scale + self.offset

    def inv(self, y):
        return (np.asarray(y, dtype=float) - self.offset) / self.scale


def sample_frames(ep: np.ndarray, n: int, seed: int, exclude: list[int]) -> np.ndarray:
    """Indices t with t-8 and t+31 inside the same episode (full history incl. the command before s[t-7])."""
    ok = []
    for e in np.unique(ep):
        if int(e) in exclude:
            continue
        idx = np.flatnonzero(ep == e)
        lo, hi = idx[0] + N_OBS, idx[-1] - N_EXEC + 1
        ok.extend(range(lo, hi + 1))
    ok = np.array(ok)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(ok, size=min(n, len(ok)), replace=False))


def window(states, actions, t):
    """Runner history at tick t: states s[t-7..t], commands a[t-8..t-1] (the command issued before each)."""
    return states[t - N_OBS + 1 : t + 1], actions[t - N_OBS : t]


def moving_frames(true: np.ndarray, anchor: np.ndarray, arm_deg: float = 5.0, grip_pct: float = 15.0) -> np.ndarray:
    """Frames whose recorded chunk moves an arm joint by more than ``arm_deg`` or the gripper by more than
    ``grip_pct`` (displacement from the current command to step 32)."""
    disp = np.abs(true[:, -1] - anchor)
    return (disp[:, :5] > arm_deg).any(axis=1) | (disp[:, 5] > grip_pct)


def metrics(pred: np.ndarray, true: np.ndarray, anchor: np.ndarray, min_move: np.ndarray) -> dict:
    """pred/true: (N, 32, 6); anchor: (N, 6) current command. Whole-sample and moving-frame metrics."""
    disp_t, disp_p = true[:, -1] - anchor, pred[:, -1] - anchor
    moved = np.abs(disp_t) > min_move
    agree = np.where(moved, np.sign(disp_t) == np.sign(disp_p), np.nan)
    rel = np.where(moved, np.abs(disp_p - disp_t) / np.maximum(np.abs(disp_t), 1e-6), np.nan)
    mov = moving_frames(true, anchor)
    with np.errstate(all="ignore"):
        out = {
            "mae": np.abs(pred - true).mean(axis=(0, 1)).tolist(),
            "direction_agreement": np.nanmean(agree, axis=0).tolist(),
            "moved_samples": moved.sum(axis=0).tolist(),
            "moving_frames": int(mov.sum()),
            "moving_mae": np.abs(pred[mov] - true[mov]).mean(axis=(0, 1)).tolist() if mov.any() else None,
            "moving_direction_agreement": np.nanmean(agree[mov], axis=0).tolist() if mov.any() else None,
            "moving_joint_samples": moved[mov].sum(axis=0).tolist(),
            "moving_rel_disp_error_median": np.nanmedian(rel[mov], axis=0).tolist() if mov.any() else None,
        }
    return out


def load_backend(package: str, overrides: dict, text_encoder_on_cpu: bool, device: str = "cuda"):
    """``LeRobotFlux3Backend.from_pretrained`` without its load-time VRAM peak: it moves the whole policy (Qwen
    included, 23.6 GiB peak, cloud/RUNPOD.md) to the GPU before moving Qwen back, which overflows a 24 GB RTX 4090.
    Here Qwen is detached while the trunk and VAE move, then re-attached on the CPU: the same end state."""
    from flux.backends import LeRobotFlux3Backend
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.flux3 import Flux3Policy

    config = PreTrainedConfig.from_pretrained(package)
    config.device = "cpu"  # from_pretrained moves the policy to config.device itself (_load_as_safetensor)
    policy = Flux3Policy.from_pretrained(package, config=config)
    for key, value in overrides.items():
        setattr(policy.config, key, value)
    policy.config.device = device
    text_encoder = policy.frozen.text_encoder
    if text_encoder_on_cpu:
        # FrozenComponents is not an nn.Module; the policy's _apply reaches its modules() only (modeling_flux3.py).
        object.__setattr__(policy.frozen, "text_encoder", None)
    policy.to(device).eval()
    object.__setattr__(policy.frozen, "text_encoder", text_encoder)
    pre, post = make_pre_post_processors(policy.config, pretrained_path=package,
                                         preprocessor_overrides={"device_processor": {"device": device}})  # fmt: skip
    return LeRobotFlux3Backend(policy, pre, post, name=package)


def build_backends(args, states, actions, ep, keep_eps):
    """name -> (backend-like with .predict and .contract, Affine)."""
    from flux.backends import FakeBackend
    from flux.contract import SO101_CONTRACT
    from flux.package_tools import compute_quantiles, set_history_quantiles

    sel = np.isin(ep, keep_eps)

    def stats_for(transform: Affine):
        return compute_quantiles(commands=transform(actions[sel]), states=transform(states[sel]),
                                 episode_index=ep[sel], absolute_dims=(-1,))  # fmt: skip

    transforms = {"A": Affine(), "B": Affine(), "C": Affine(ISSUE2_SCALE, ISSUE2_OFFSET), "D": Affine(FLIP_SCALE),
                  "E": Affine(E_SCALE, E_OFFSET)}  # fmt: skip
    renorm = {k: stats_for(transforms[k]) for k in ("B", "D") if k in args.variants}
    out = {}
    if args.backend == "fake":
        for name in args.variants:
            out[name] = (FakeBackend(SO101_CONTRACT, velocity=0.0), transforms[name])
        return out, renorm
    from flux.backends import LeRobotFlux3Backend
    from lerobot.policies.factory import make_pre_post_processors

    overrides = {"guidance_scale": args.guidance, "guidance_scale_action": args.guidance, "compile_model": False}
    base = load_backend(args.package, overrides, args.text_encoder_on_cpu)
    for name in args.variants:
        if name in renorm:
            device = {"device_processor": {"device": "cuda"}}
            pre, post = make_pre_post_processors(base.policy.config, pretrained_path=args.package,
                                                 preprocessor_overrides=device)  # fmt: skip
            set_history_quantiles(pre, post, renorm[name])
            backend = LeRobotFlux3Backend(base.policy, pre, post, name=f"{args.package} [{name}]")
        else:
            backend = base
        out[name] = (backend, transforms[name])
    return out, renorm


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--repo-id", required=True)
    p.add_argument("--root", default=None)
    p.add_argument("--task", required=True)
    p.add_argument("--package", default="black-forest-labs/flux-3-action-so101")
    p.add_argument("--variants", default="A,B,C,D")
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exclude", type=int, nargs="*", default=[], help="held-out / junk episodes")
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--text-encoder-on-cpu", action="store_true")
    p.add_argument("--backend", choices=("lerobot", "fake"), default="lerobot")
    p.add_argument("--min-move", type=float, nargs=6, default=[2, 2, 2, 2, 2, 5], help="direction test threshold")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    args.variants = args.variants.split(",")
    from flux.backends import PolicyRequest
    from flux.cameras import CameraView, prepare_image
    from flux.package_tools import dataset_arrays
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    arrays = dataset_arrays(args.repo_id, root=args.root)
    states = arrays["observation.state"].astype(float)
    actions = arrays["action"].astype(float)
    ep = arrays["episode_index"]
    keep_eps = [int(e) for e in np.unique(ep) if int(e) not in args.exclude]
    frames = sample_frames(ep, args.samples, args.seed, args.exclude)
    print(f"{len(frames)} frames from episodes {keep_eps}", flush=True)
    backends, renorm = build_backends(args, states, actions, ep, keep_eps)
    contract = next(iter(backends.values()))[0].contract
    ds = LeRobotDataset(args.repo_id, root=args.root)
    cams = {k: k for k in ("observation.images.scene", "observation.images.wrist")}  # per-arm dataset keys

    def image(idx: int, key: str) -> np.ndarray:
        img = ds[int(idx)][key]  # (3, H, W) float [0, 1]
        rgb = (img.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
        return prepare_image(rgb, CameraView(key), contract.image_hw)

    preds = {name: [] for name in backends}
    trues, anchors, lat = [], [], {name: [] for name in backends}
    t0 = time.time()
    for n, t in enumerate(frames):
        s_win, c_win = window(states, actions, t)
        imgs = {}
        for model_key, ds_key in cams.items():
            imgs[model_key] = np.stack([image(t - N_OBS + 1 + tick, ds_key) for tick in contract.snapshots])
        trues.append(actions[t : t + N_EXEC])
        anchors.append(actions[t - 1])
        for name, (backend, tf) in backends.items():
            req = PolicyRequest(args.task, tf(s_win), tf(c_win), imgs, contract.snapshots)
            r = backend.predict(req)
            preds[name].append(tf.inv(np.asarray(r.commands)[:N_EXEC]))
            lat[name].append(r.latency_s)
        if n % 20 == 0:
            print(f"{n + 1}/{len(frames)} frames, {time.time() - t0:.0f} s", flush=True)
    true, anchor = np.array(trues), np.array(anchors)
    min_move = np.array(args.min_move)
    result = {
        "args": vars(args), "frames": frames.tolist(), "episodes": keep_eps,
        "hold": metrics(np.repeat(anchor[:, None], N_EXEC, axis=1), true, anchor, min_move),
        "variants": {name: {**metrics(np.array(p), true, anchor, min_move), "latency_s": float(np.median(lat[name]))}
                     for name, p in preds.items()},  # fmt: skip
        "renorm_stats": renorm,
    }  # fmt: skip
    Path(args.out).write_text(json.dumps(result, indent=2, default=float))
    np.savez_compressed(Path(args.out).with_suffix(".npz"), true=true, anchor=anchor, frames=frames,
                        **{f"pred_{k}": np.array(v) for k, v in preds.items()})  # fmt: skip
    print(render(result))


def _pct(values) -> str:
    return ", ".join("-" if v is None or np.isnan(v) else f"{v:.0%}" for v in values)


def render(result: dict) -> str:
    """Moving-frame table first (the selection criterion), then all frames."""
    rows = [f"Moving frames: {result['hold']['moving_frames']} of {len(result['frames'])} "
            "(recorded chunk moves an arm joint > 5 deg or the gripper > 15 %)", "",
            "| variant | MAE pan, lift, elbow, wflex, roll (deg); gripper (%) | direction agreement | "
            "median rel. displacement error |", "|---|---|---|---|"]  # fmt: skip
    entries = [("hold", result["hold"]), *result["variants"].items()]
    for name, m in entries:
        mae = ", ".join(f"{v:.1f}" for v in (m["moving_mae"] or []))
        rows.append(f"| {name} | {mae} | {_pct(m['moving_direction_agreement'] or [])} | "
                    f"{_pct(m['moving_rel_disp_error_median'] or [])} |")  # fmt: skip
    rows += ["", "All frames:", "", "| variant | MAE (deg / %) | direction agreement |", "|---|---|---|"]
    for name, m in entries:
        rows.append(f"| {name} | {', '.join(f'{v:.1f}' for v in m['mae'])} | {_pct(m['direction_agreement'])} |")
    return "\n".join(rows)


if __name__ == "__main__":
    main()
