"""GPU job: FLUX 3 Action features at every re-plan of recorded episodes, for the P2 progress memory.

For each episode directory (``sim_episodes.py`` format, or any episode with ``episode.json`` + ``ticks.npz`` +
``frames.npz``), every re-plan becomes exactly the ``PolicyRequest`` the dual-arm runner would send for that arm
(``ws4/flux``: per-arm target, history window reset at each step start, snapshots at window ticks 0 and 7, the
runner's image preparation). The real package predicts a chunk, and :class:`flux_hook.FluxFeatureTap` hooks on
joint blocks of the loaded DiT pool the hidden states of two token groups at the last DiT call of the re-plan
(the conditional pass of the final denoising step):

* ``observed``: ``video_cond`` + ``<modality>_cond`` (the snapshots, and the state / past commands);
* ``imagined``: ``video`` + ``<modality>`` (the predicted future video latents and the action chunk).

Output per episode: ``features_flux.npz`` with ``feat`` (R, layers x groups x 3072, float16), ``replan_ticks``
and the settings. Run in the LeRobot env on a GPU with the weights (``ws4/cloud/bootstrap.sh``), from ``ws4/``:

    PYTHONPATH=. python -m ttt.extract_flux_features --data /workspace/ttt/sim --limit 60
    PYTHONPATH=. python -m ttt.extract_flux_features --data ... --override guidance_scale=1.0   # about 2x faster

Nothing here runs on import; ``episode_requests`` is CPU-testable without weights.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_LAYERS = (16, 27)
GROUPS = ("observed", "imagined")


def _flux():
    from flux.backends import PolicyRequest
    from flux.cameras import prepare_image
    from flux.contract import IMG_HEAD, IMG_LEFT_WRIST, IMG_RIGHT_WRIST, arm_slice, snapshot_indices
    from flux.dual_arm_runner import per_arm_target

    return (
        PolicyRequest,
        prepare_image,
        (IMG_HEAD, IMG_LEFT_WRIST, IMG_RIGHT_WRIST),
        arm_slice,
        snapshot_indices,
        per_arm_target,
    )


def episode_requests(ep_dir: Path, contract, mirror_right: bool = False) -> Iterator[tuple[dict, object]]:
    """Yield ``(replan record, PolicyRequest)`` for every re-plan, as ``flux.dual_arm_runner`` would build it.

    History semantics (``flux/history.py``): the window resets at each step start, the first observation of a
    step is repeated to fill it, and before any command the "previous command" is the measured state.
    ``ticks.npz`` stores ``state[i]`` (observed at tick i) and ``command[i]`` (sent right after it).
    """
    from ttt.labels import EpisodeLabels
    from ttt.sim_episodes import decode_jpeg, load_frames

    PolicyRequest, prepare_image, robot_keys, arm_slice, snapshot_indices, per_arm_target = _flux()
    labels = EpisodeLabels.load(ep_dir / "episode.json")
    z = np.load(ep_dir / "ticks.npz")
    state, command = z["state"].astype(np.float64), z["command"].astype(np.float64)
    frame_ticks, frames = load_frames(ep_dir)
    where = {int(t): i for i, t in enumerate(frame_ticks)}
    n = contract.n_obs_steps
    snaps = snapshot_indices(n, contract.history_snapshots)
    for rp in labels.extra["replans"]:
        t, arm = rp["tick"], rp["arm"]
        start = max(s.start_tick for s in labels.segments if s.step_id == rp["step_id"] and s.start_tick <= t)
        cfg = per_arm_target(arm, mirror=mirror_right and arm == "right")
        cal = cfg.calibration[arm]
        win = [max(start, t - (n - 1) + j) for j in range(n)]
        sl = arm_slice(arm)
        states = np.stack([cal.to_model(state[i][sl]) for i in win])
        cmds = np.stack([cal.to_model((state[i] if i == start else command[i - 1])[sl]) for i in win])
        images = {}
        for key, view in cfg.cameras.items():
            cam = {robot_keys[0]: "head", robot_keys[1]: "left_wrist", robot_keys[2]: "right_wrist"}[view.source]
            imgs = []
            for j in snaps:
                tick = win[j]
                # Exact snapshot frame when stored (sim_episodes.py stores FLUX's ticks), else the nearest earlier one.
                near = tick if tick in where else max((t for t in where if t <= tick), default=min(where))
                img = decode_jpeg(frames[cam][where[near]])
                imgs.append(prepare_image(img, view, contract.image_hw))
            images[key] = np.stack(imgs)
        yield rp, PolicyRequest(rp.get("prompt", ""), states, cmds, images, tuple(snaps), target=arm)


def streams_for(group: str, modality: str) -> tuple[str, str]:
    return ("video_cond", f"{modality}_cond") if group == "observed" else ("video", modality)


def extract_episodes(backend, taps, eps, checkpoint: str, layers, groups, overrides: dict) -> dict:
    """Run every re-plan of ``eps`` through ``backend`` and write ``features_flux.npz`` per episode."""
    import torch

    log = {
        "checkpoint": checkpoint,
        "overrides": overrides,
        "layers": list(layers),
        "groups": list(groups),
        "episodes": [],
    }
    for ep in eps:
        e0 = time.time()
        feats, ticks, lat = [], [], []
        for rp, req in episode_requests(ep, backend.contract):
            for _, _, tap in taps:
                tap.features = []
            resp = backend.predict(req)
            lat.append(resp.latency_s)
            feats.append(np.concatenate([tap.pop()[0, 0].cpu().numpy() for _, _, tap in taps]))
            ticks.append(rp["tick"])
        np.savez(
            ep / "features_flux.npz",
            feat=np.stack(feats).astype(np.float16),
            replan_ticks=np.asarray(ticks),
            layers=np.asarray(list(layers)),
            groups=np.asarray(list(groups)),
            checkpoint=np.asarray(checkpoint),
            overrides=np.asarray(json.dumps(overrides)),
        )
        rec = {
            "episode": ep.name,
            "replans": len(ticks),
            "seconds": round(time.time() - e0, 1),
            "median_latency_s": round(float(np.median(lat)), 3),
        }
        if torch.cuda.is_available():
            rec["max_memory_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        log["episodes"].append(rec)
        print(json.dumps(rec), flush=True)
    return log


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, required=True, help="directory of episode directories")
    ap.add_argument("--checkpoint", default="black-forest-labs/flux-3-action-so101")
    ap.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_LAYERS))
    ap.add_argument("--groups", nargs="+", default=list(GROUPS), choices=GROUPS)
    ap.add_argument("--override", action="append", default=[], help="Flux3Config field=value, e.g. guidance_scale=1.0")
    ap.add_argument("--text-encoder-on-cpu", action="store_true", help="for GPUs under 40 GB")
    ap.add_argument("--limit", type=int, default=None, help="at most this many episodes (sorted by name)")
    ap.add_argument("--per-task", type=int, default=None, help="at most this many episodes per task (sorted)")
    ap.add_argument("--skip-done", action="store_true", help="skip episodes that already have features_flux.npz")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    from flux.backends import LeRobotFlux3Backend

    from ttt.flux_hook import FluxFeatureTap

    overrides = {"compile_model": False}  # hooks need the eager DiT
    for kv in args.override:
        k, v = kv.split("=", 1)
        overrides[k] = json.loads(v) if v[:1] in "0123456789-[{tfn" else v
    t0 = time.time()
    backend = LeRobotFlux3Backend.from_pretrained(
        args.checkpoint, device=args.device, overrides=overrides, text_encoder_on_cpu=args.text_encoder_on_cpu
    )
    load_s = time.time() - t0
    dit, modality = backend.policy.dit, backend.policy.modality
    taps = [
        (layer, g, FluxFeatureTap(layer, streams_for(g, modality)).attach(dit))
        for layer in args.layers
        for g in args.groups
    ]
    eps = sorted(p.parent for p in args.data.glob("*/episode.json"))
    if args.per_task:
        by_task: dict[str, list[Path]] = {}
        for e in eps:
            by_task.setdefault(json.loads((e / "episode.json").read_text())["task"], []).append(e)
        eps = sorted(e for group in by_task.values() for e in group[: args.per_task])
    if args.skip_done:
        eps = [e for e in eps if not (e / "features_flux.npz").exists()]
    eps = eps[: args.limit] if args.limit else eps
    print(
        f"loaded {args.checkpoint} in {load_s:.0f}s; {len(eps)} episodes; taps {[(layer, g) for layer, g, _ in taps]}",
        flush=True,
    )
    log = extract_episodes(backend, taps, eps, args.checkpoint, args.layers, args.groups, overrides)
    log["load_s"] = load_s
    log["total_s"] = round(time.time() - t0, 1)
    (args.data / f"extract_flux_log_{int(t0)}.json").write_text(json.dumps(log, indent=1))
    print(f"done in {log['total_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
