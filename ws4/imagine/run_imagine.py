"""Command line for imagination runs.

    python -m ws4.imagine.run_imagine fake   --out outputs/imagine/fake          # CPU, no weights
    python -m ws4.imagine.run_imagine replay --dataset <repo|path> --episode 0 ...  # GPU, open loop
    python -m ws4.imagine.run_imagine bench  --decode-device cuda:1              # GPU, latency and VRAM

``replay`` feeds recorded observation windows through the checkpoint's own preprocessor, imagines each chunk
and compares the imagined frames with the frames the dataset actually recorded next (open loop: the
recorded future follows the demonstrator, not the model). ``bench`` measures the cost of imagination on
synthetic observations. Every run writes ``report.json`` next to its GIFs. See GPU_RUNBOOK.md.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from .compose import ActualFrame, contact_sheet, make_canvas, render_run
from .predict_with_video import Imaginer, LatentDecoder, video_vae_of
from .score import frame_distance


def _report(out: Path, payload: dict) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    path = out / "report.json"
    path.write_text(json.dumps(payload, indent=2, default=float))
    return path


# ---------------------------------------------------------------------------------------------
# fake: the whole live path on CPU
# ---------------------------------------------------------------------------------------------
def run_fake(args) -> dict:
    """select_action loop on the fake policy with async decoding, then the side-by-side GIF."""
    from .fakes import FakeConfig, FakeFluxPolicy, fake_batch

    cfg = FakeConfig()
    policy = FakeFluxPolicy(cfg)
    imaginer = Imaginer(policy, decode="async").install()
    policy.reset()
    actual = []
    for step in range(args.steps):
        batch = fake_batch(cfg, seed=step // 8)
        policy.select_action(batch)
        scene = (batch[cfg.camera_keys[0]][0, -1].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        wrist = (batch[cfg.camera_keys[1]][0, -1].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        actual.append(ActualFrame(make_canvas(scene, wrist, cfg.canvas_hw), step=step))
    records = [r.wait(30) for r in imaginer.history]
    imaginer.close()
    out = Path(args.out)
    surprise = render_run(records, actual, out / "side_by_side.gif", task="fake task")
    from PIL import Image

    Image.fromarray(contact_sheet(records[0])).save(out / "contact_sheet.png")
    report = {
        "mode": "fake",
        "chunks": len(records),
        "steps": [r.step for r in records],
        "frames_per_chunk": int(records[0].frames.shape[0]),
        "frame_offsets": list(records[0].frame_offsets),
        "mean_surprise": float(np.nanmean(surprise)),
    }
    _report(out, report)
    return report


# ---------------------------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------------------------
def load_policy(policy_id: str, device: str):
    """Checkpoint and its saved processors (LeRobot main; model card usage)."""
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.flux3 import Flux3Policy

    policy = Flux3Policy.from_pretrained(policy_id)
    policy.to(device).eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=policy_id)
    return policy, pre, post


def make_decoder(policy, decode_device: str | None):
    """Same-GPU decoding reuses the policy's VAE; another device loads a second frozen copy there."""
    if not decode_device or decode_device == str(next(policy.parameters()).device):
        return LatentDecoder(video_vae_of(policy))
    from lerobot.policies.flux3.f3 import load_video_vae

    return LatentDecoder(load_video_vae(policy.config.video_vae_id, device=decode_device), device=decode_device)


def _sync():
    import torch

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)


def _peak_memory() -> dict:
    import torch

    if not torch.cuda.is_available():
        return {}
    return {f"cuda:{i}": torch.cuda.max_memory_allocated(i) / 2**30 for i in range(torch.cuda.device_count())}


# ---------------------------------------------------------------------------------------------
# replay: imagined vs recorded future
# ---------------------------------------------------------------------------------------------
def load_episode(dataset: str, episode: int, keys: dict[str, str], side: int):
    """Episode frames resized to ``side`` x ``side`` uint8, plus state, action and task. UNVERIFIED against
    every LeRobotDataset version: relies only on ``LeRobotDataset(repo_id, root=, episodes=)`` and ``ds[i]``."""
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = dataset if Path(dataset).exists() else None
    ds = LeRobotDataset(dataset if root is None else Path(dataset).name, root=root, episodes=[episode])
    frames = {k: [] for k in keys}
    states, actions, task = [], [], ""
    for i in range(len(ds)):
        item = ds[i]
        if int(item["episode_index"]) != episode:
            continue
        for model_key, ds_key in keys.items():
            img = item[ds_key].float()
            img = torch.nn.functional.interpolate(img[None], size=(side, side), mode="bilinear", antialias=True)[0]
            frames[model_key].append((img.clamp(0, 1) * 255).round().to(torch.uint8))
        states.append(item["observation.state"].float())
        actions.append(item["action"].float())
        task = item.get("task", task)
    return {k: torch.stack(v) for k, v in frames.items()}, torch.stack(states), torch.stack(actions), task


def offline_window(frames, states, actions, t: int, n_obs: int, task: str):
    """Raw observation window for the history preprocessor's offline path (processor_flux3.py:388-394):
    images and states at t-n_obs+1 .. t and the commands sent before each, actions t-n_obs .. t-1 (clamped
    at the episode start)."""
    obs_idx = [max(0, i) for i in range(t - n_obs + 1, t + 1)]
    cmd_idx = [max(0, i) for i in range(t - n_obs, t)]
    batch = {k: (v[obs_idx].float() / 255.0)[None] for k, v in frames.items()}
    batch["observation.state"] = states[obs_idx][None]
    batch["observation.command_history"] = actions[cmd_idx][None]
    batch["task"] = [task]
    return batch


def run_replay(args) -> dict:
    import torch
    from PIL import Image

    policy, pre, _post = load_policy(args.policy, args.device)
    cfg = policy.config
    keys = {cfg.camera_order[0]: args.scene_key, cfg.camera_order[1]: args.wrist_key}
    side = cfg.canvas_hw[0]
    frames, states, actions, task = load_episode(args.dataset, args.episode, keys, side)
    task = args.task or task
    decoder = make_decoder(policy, args.decode_device)
    imaginer = Imaginer(policy, decoder=decoder, decode="sync")
    out = Path(args.out)
    n = len(states)
    starts = list(range(args.start, n - cfg.chunk_size, cfg.n_action_steps))[: args.chunks]
    scene_k, wrist_k = tuple(keys)

    def canvas_at(t: int) -> np.ndarray:
        return make_canvas(frames[scene_k][t].permute(1, 2, 0).numpy(), frames[wrist_k][t].permute(1, 2, 0).numpy(),
                           cfg.canvas_hw)

    out.mkdir(parents=True, exist_ok=True)
    records, per_offset = [], {}
    for k, t in enumerate(starts):
        if hasattr(pre, "reset"):
            pre.reset()
        batch = pre(offline_window(frames, states, actions, t, cfg.n_obs_steps, task))
        _sync()
        (rec,) = imaginer.imagine(batch)
        _sync()
        rec.step, rec.index = t, k
        records.append(rec)
        canvases = {o: canvas_at(t + o) for o in range(1, cfg.chunk_size + 1)}
        for o, img in canvases.items():
            per_offset.setdefault(o, []).append(frame_distance(img, rec.frame_at(o)))
        Image.fromarray(contact_sheet(rec, canvases)).save(out / f"chunk_{k:02d}_t{t}.png")
        rec.save(out / f"chunk_{k:02d}_t{t}.npz")
    actual = [ActualFrame(canvas_at(t), step=t) for t in range(starts[0], starts[-1] + cfg.n_action_steps)]
    render_run(records, actual, out / "side_by_side.mp4", task=task)
    render_run(records, actual, out / "side_by_side.gif", task=task)
    report = {
        "mode": "replay",
        "policy": args.policy,
        "dataset": args.dataset,
        "episode": args.episode,
        "task": task,
        "chunk_starts": starts,
        "sample_s": [r.timings.get("sample_s") for r in records],
        "decode_s": [r.timings.get("decode_s") for r in records],
        "surprise_by_offset": {o: float(np.mean(v)) for o, v in sorted(per_offset.items())},
        "peak_memory_gib": _peak_memory(),
        "torch": torch.__version__,
    }
    _report(out, report)
    return report


# ---------------------------------------------------------------------------------------------
# vaecheck: does the decoded-frame timing derived from the code hold on the real VAE?
# ---------------------------------------------------------------------------------------------
def nearest_sources(decoded: np.ndarray, source: np.ndarray) -> list[int]:
    """For each decoded frame, the index of the closest source frame (mean absolute difference)."""
    src = source.astype(np.float32)
    return [int(np.abs(src - f.astype(np.float32)).mean(axis=(1, 2, 3)).argmin()) for f in decoded]


def run_vaecheck(args) -> dict:
    """Encode a recorded future clip exactly like history training targets (``encode_task`` on
    ``chunk_size`` frames, packing.py:469-473), decode it, and match every decoded frame to its source."""
    import torch
    from lerobot.policies.flux3.f3 import load_video_vae, packing

    from .predict_with_video import VideoSpec, decoded_frame_sources, to_uint8_frames

    policy_cfg = _load_config(args.policy)
    spec = VideoSpec.from_config(policy_cfg)
    keys = {policy_cfg.camera_order[0]: args.scene_key, policy_cfg.camera_order[1]: args.wrist_key}
    frames, _, _, _ = load_episode(args.dataset, args.episode, keys, policy_cfg.canvas_hw[0])
    t0, n = args.start, spec.chunk_size
    cams = torch.stack([frames[k][t0 : t0 + n] for k in keys]).float() / 255.0  # (2, n, 3, H, W)
    video = packing.materialize_video(cams, None, args.device, layout=spec.camera_layout,
                                      canvas_hw=spec.canvas_hw)  # (3, n, Hc, Wc) in [-1, 1]
    vae = load_video_vae(policy_cfg.video_vae_id, device=args.device)
    with torch.inference_mode():
        lat = vae.encode_task(video[None].to(torch.bfloat16))[..., : spec.latent_hw[0], : spec.latent_hw[1]]
        decoded = to_uint8_frames(vae.decode(lat))[0]
    source = to_uint8_frames(video[None].float())[0]
    found = nearest_sources(decoded, source)
    expected = [c for _, c in decoded_frame_sources(n)]
    mse = ((decoded.astype(np.float32) - source[expected].astype(np.float32)) ** 2).mean()
    report = {
        "mode": "vaecheck",
        "latents": list(lat.shape),
        "decoded_frames": int(decoded.shape[0]),
        "expected_sources": expected,
        "nearest_sources": found,
        "agreement": float(np.mean(np.asarray(found) == np.asarray(expected))),
        "psnr_db": float(10 * np.log10(255.0**2 / max(mse, 1e-6))),
    }
    _report(Path(args.out), report)
    return report


def _load_config(policy_id: str):
    from lerobot.configs.policies import PreTrainedConfig

    return PreTrainedConfig.from_pretrained(policy_id)


# ---------------------------------------------------------------------------------------------
# bench: what imagination costs
# ---------------------------------------------------------------------------------------------
def run_bench(args) -> dict:
    import torch

    policy, pre, _post = load_policy(args.policy, args.device)
    cfg = policy.config
    g = torch.Generator().manual_seed(0)
    cam_hw = (cfg.canvas_hw[0], cfg.canvas_hw[1] // 2)
    raw = {k: torch.rand(1, 3, *cam_hw, generator=g) for k in cfg.camera_order}
    raw["observation.state"] = torch.zeros(1, cfg.action_dim)
    raw["task"] = [args.task or "pick up the cube and place it in the bowl"]
    if hasattr(pre, "reset"):
        pre.reset()
    batch = pre(dict(raw))  # synchronous path: the first tick fills the whole history (processor_flux3.py:380)

    def timed(fn, reps):
        times = []
        for _ in range(reps):
            _sync()
            t0 = time.perf_counter()
            fn()
            _sync()
            times.append(time.perf_counter() - t0)
        return {"median_s": statistics.median(times), "min_s": min(times), "n": reps}

    result: dict = {"mode": "bench", "policy": args.policy, "device": args.device}
    for _ in range(args.warmup):
        policy.predict_action_chunk(batch)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    result["stock_predict"] = timed(lambda: policy.predict_action_chunk(batch), args.reps)
    result["peak_memory_stock_gib"] = _peak_memory()
    for name, kwargs in {
        "imagine_no_decode": {"decode": "none"},
        "imagine_sync_same_gpu": {"decode": "sync", "decoder": make_decoder(policy, None)},
    }.items():
        im = Imaginer(policy, **kwargs)
        result[name] = timed(lambda im=im: im.imagine(batch), args.reps)
        im.close()
    if args.decode_device:
        dec = make_decoder(policy, args.decode_device)
        lat = Imaginer(policy, decode="none").imagine(batch)[0]
        result["decode_only_" + args.decode_device] = timed(lambda: dec.decode_record(lat), args.reps)
        im = Imaginer(policy, decode="async", decoder=dec)
        result["imagine_async_" + args.decode_device] = timed(lambda: im.imagine(batch), args.reps)
        im.close()
    for n in args.candidates:
        im = Imaginer(policy, n_candidates=n, decode="none", scorer=lambda clips, **kw: [0.0] * len(clips),
                      decoder=make_decoder(policy, args.decode_device))
        result[f"best_of_{n}_with_decode"] = timed(lambda im=im: im.imagine(batch), max(2, args.reps // 2))
        im.close()
    result["peak_memory_gib"] = _peak_memory()
    result["torch"] = torch.__version__
    _report(Path(args.out), result)
    return result


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fake", help="CPU end-to-end run on the fake policy")
    f.add_argument("--out", default="outputs/imagine/fake")
    f.add_argument("--steps", type=int, default=70)
    for name in ("replay", "bench", "vaecheck"):
        s = sub.add_parser(name)
        s.add_argument("--policy", default="black-forest-labs/flux-3-action-so101")
        s.add_argument("--device", default="cuda:0")
        s.add_argument("--decode-device", default=None, help="e.g. cuda:1 to decode on another GPU")
        s.add_argument("--task", default=None)
        s.add_argument("--out", default=f"outputs/imagine/{name}")
    for name in ("replay", "vaecheck"):
        r = sub.choices[name]
        r.add_argument("--dataset", required=True, help="LeRobot repo id or local dataset root")
        r.add_argument("--episode", type=int, default=0)
        r.add_argument("--start", type=int, default=8)
        r.add_argument("--scene-key", required=True, help="dataset image key used as observation.images.scene")
        r.add_argument("--wrist-key", required=True, help="dataset image key used as observation.images.wrist")
    sub.choices["replay"].add_argument("--chunks", type=int, default=6)
    b = sub.choices["bench"]
    b.add_argument("--reps", type=int, default=10)
    b.add_argument("--warmup", type=int, default=3)
    b.add_argument("--candidates", type=int, nargs="*", default=[2, 4])
    return p


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    return {"fake": run_fake, "replay": run_replay, "bench": run_bench, "vaecheck": run_vaecheck}[args.cmd](args)


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=float))
