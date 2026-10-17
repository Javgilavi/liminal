"""Stage tracking for the single-skill sim task "put the block in the cup": train on scripted demos, score rollouts.

Three coarse stages that both kinds of episode can be labelled with, without FLUX telling us anything:

* ``reach``: the block is still on the table (approach, descend, grasp);
* ``carry``: the block is off the table and not yet in the cup (lift, carry, lower);
* ``in_cup``: the block is in the cup (release, retreat, home); this is also the "done" signal.

Demos carry the scripted phase of every frame (``sim_block_cup.py`` writes ``stages_<arm>.jsonl``); rollouts are
labelled from privileged poses (:func:`stage_from_poses`, the same thresholds as :func:`sim_block_cup.block_in_cup`).
Features are the CPU colour-grid features (``features.FrameFeaturizer``) of the head and task-wrist frames at every
re-plan (every 32 frames), so the same model runs on demos, FLUX rollouts and, after retraining, real recordings.

    python ws4/ttt/cup_stages.py train --data ws4/.cache/ttt/block_cup        # GRU/TTT/none, held-out demos
    python ws4/ttt/cup_stages.py score <rollout episode dirs...>               # stage accuracy on FLUX rollouts
    python ws4/ttt/cup_stages.py train --rollouts <dirs...> --tag _mix          # demos + half the rollouts
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ttt.features import FrameFeaturizer, Sequence  # noqa: E402

WS4 = Path(__file__).resolve().parents[1]
STAGES = ("reach", "carry", "in_cup")
PHASE_TO_STAGE = {"reach": 0, "descend": 0, "grasp": 0, "lift": 1, "carry": 1, "lower": 1, "release": 2,
                  "retreat": 2, "home": 2}  # fmt: skip
STRIDE = 32  # executed actions per FLUX chunk
CUP_INNER_R, CUP_RIM_Z, LIFTED_Z = 0.0325, 0.085, 0.035


def stage_from_poses(block_xyz, cup_xyz) -> np.ndarray:
    """Per-tick stage from block and cup positions ``(T, 3)``."""
    b, c = np.asarray(block_xyz, float), np.asarray(cup_xyz, float)
    r = np.linalg.norm(b[:, :2] - c[:, :2], axis=1)
    dz = b[:, 2] - c[:, 2]
    in_cup = (r < CUP_INNER_R) & (dz > 0) & (dz < CUP_RIM_Z)
    lifted = b[:, 2] > LIFTED_Z
    return np.where(in_cup, 2, np.where(lifted, 1, 0)).astype(np.int64)


def _sequence(name: str, feat: list, stage: np.ndarray, prog: np.ndarray | None = None) -> Sequence:
    stage = np.asarray(stage, np.int64)
    n = len(stage)
    prog = np.arange(n) / max(1, n - 1) if prog is None else np.asarray(prog, float)
    return Sequence(
        name,
        "block_cup",
        np.stack(feat).astype(np.float32),
        stage,
        np.zeros(n, np.int64),
        stage,
        prog,
        prog,
        (stage == 2).astype(np.int64),
        list(STAGES),
    )


def demo_sequences(data: Path, arm: str = "left", featurizer: FrameFeaturizer | None = None) -> list[Sequence]:
    """One sequence per demo: features and stage at frames 0, 32, 64, ... (cached next to the dataset)."""
    cache = data / f"stage_features_{arm}.npz"
    lines = [json.loads(x) for x in (data / f"stages_{arm}.jsonl").read_text().splitlines() if x.strip()]
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return [_sequence(n, list(f), s) for n, f, s in zip(z["names"], z["feats"], z["stages"], strict=True)]
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    featurizer = featurizer or FrameFeaturizer()
    ds = LeRobotDataset(f"local/sim_block_cup_{arm}", root=data / f"sim_block_cup_{arm}")
    starts = np.asarray(ds.meta.episodes["dataset_from_index"])
    ends = np.asarray(ds.meta.episodes["dataset_to_index"])

    def img(t):
        return (t.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)

    names, feats, stages = [], [], []
    for line in lines:
        e = line["episode_index"]
        phase = np.empty(ends[e] - starts[e], dtype=object)
        for seg in line["segments"]:
            phase[seg["start"] : seg["end"]] = seg["phase"]
        f, s = [], []
        for t in range(0, ends[e] - starts[e], STRIDE):
            item, old = ds[int(starts[e] + t)], ds[int(starts[e] + max(0, t - 7))]
            st = np.zeros(12)
            st[(0 if arm == "left" else 6) : (6 if arm == "left" else 12)] = item["observation.state"].numpy()
            wrist = "left_wrist" if arm == "left" else "right_wrist"
            frames = {"head": img(item["observation.images.scene"]), wrist: img(item["observation.images.wrist"])}
            f.append(featurizer(frames, st, img(old["observation.images.scene"])))
            s.append(PHASE_TO_STAGE[phase[t]])
        names.append(f"demo-{e:03d}-seed{line['seed']}")
        feats.append(np.stack(f))
        stages.append(np.asarray(s))
    np.savez(
        cache, names=np.asarray(names), feats=np.asarray(feats, dtype=object), stages=np.asarray(stages, dtype=object)
    )
    return [_sequence(n, list(f), s) for n, f, s in zip(names, feats, stages, strict=True)]


def rollout_sequence(ep_dir: Path, arm: str = "left", featurizer: FrameFeaturizer | None = None) -> Sequence:
    """A closed-loop rollout in the ``sim_episodes.py`` layout with ``block``/``cup`` positions in ``ticks.npz``."""
    from ttt.labels import EpisodeLabels
    from ttt.sim_episodes import decode_jpeg, load_frames

    featurizer = featurizer or FrameFeaturizer()
    lab = EpisodeLabels.load(ep_dir / "episode.json")
    z = np.load(ep_dir / "ticks.npz")
    gt = stage_from_poses(z["block"], z["cup"])
    ft, fr = load_frames(ep_dir)
    where = {int(t): i for i, t in enumerate(ft)}
    wrist = "left_wrist" if arm == "left" else "right_wrist"
    feats, stages = [], []
    for rp in lab.extra["replans"]:
        t = rp["tick"]
        old = rp.get("snapshot_ticks", [t, t])[0]
        frames = {"head": decode_jpeg(fr["head"][where[t]]), wrist: decode_jpeg(fr[wrist][where[t]])}
        prev = decode_jpeg(fr["head"][where[old]]) if old in where else None
        st = np.zeros(12)
        st[(0 if arm == "left" else 6) : (6 if arm == "left" else 12)] = z["state"][t]  # 6-D task arm, as the demos
        feats.append(featurizer(frames, st, prev))
        stages.append(gt[min(t, len(gt) - 1)])
    stages = np.asarray(stages)
    # Progress from the stage, not from time: a failed rollout makes no progress however long it runs.
    return _sequence(f"{ep_dir.parent.name}/{ep_dir.name}", feats, stages, stages / (len(STAGES) - 1))


def _is_test(name: str, k: int = 5) -> bool:
    return int(hashlib.sha1(name.encode()).hexdigest(), 16) % k == 0


def main(argv: list[str] | None = None) -> None:
    from ttt.throttle import throttle

    throttle()  # shares the laptop with 30 Hz robot loops
    import torch

    from ttt.progress_memory import EpisodeProgressMemory
    from ttt.train_progress import evaluate, fit_progress, pad

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["train", "score"])
    ap.add_argument("paths", nargs="*", type=Path)
    ap.add_argument("--data", type=Path, default=WS4 / ".cache" / "ttt" / "block_cup")
    ap.add_argument("--arm", default="left")
    ap.add_argument("--memories", nargs="+", default=["none", "gru", "ttt"])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--models", type=Path, default=WS4 / "ttt" / "models")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--rollouts",
        nargs="*",
        type=Path,
        default=[],
        help="train: FLUX rollout dirs to mix in (half held out by name hash, scored separately)",
    )
    ap.add_argument("--tag", default="", help="model file suffix, e.g. _mix")
    args = ap.parse_args(argv)
    results: dict = {"stages": STAGES}
    if args.cmd == "train":
        seqs = demo_sequences(args.data, args.arm)
        train = [s for s in seqs if not _is_test(s.episode)]
        test = [s for s in seqs if _is_test(s.episode)]
        rolls = [rollout_sequence(p, args.arm) for p in args.rollouts]
        r_train = [s for s in rolls if not _is_test(s.episode, 2)]
        r_test = [s for s in rolls if _is_test(s.episode, 2)]
        train = train + r_train
        feats = torch.cat([torch.from_numpy(s.feat) for s in train])
        results.update(n_train=len(train), n_test=len(test), n_rollouts_train=len(r_train), n_rollouts_test=len(r_test),
                       rollouts_test=[s.episode for s in r_test], models={})  # fmt: skip
        for memory in args.memories:
            torch.manual_seed(0)
            cfg = {"feat_dim": int(feats.shape[1]), "n_subtasks": len(STAGES), "dim": 64, "memory": memory,
                   "n_occurrences": 1, "progress_head": True}  # fmt: skip
            model = EpisodeProgressMemory(**cfg)
            model.set_normalization(feats.mean(0), feats.std(0))
            fit_progress(model, pad(train), steps=args.steps, segment=16)
            ev = evaluate(model, test, list(STAGES))["all"]
            results["models"][memory] = ev
            if r_test:
                ev_r = evaluate(model, r_test, list(STAGES))["all"]
                results["models"][memory] = {"demos": ev, "rollouts": ev_r}
                print(f"{memory:<5} held-out rollouts {len(r_test)}: stage {ev_r['stage_acc']:.3f}", flush=True)
            ck = {"config": cfg, "skills": list(STAGES), "features": "frame", "featurizer": FrameFeaturizer().config(),
                  "state_dict": model.state_dict(), "eval": ev, "max_occurrence": 1}  # fmt: skip
            torch.save(ck, args.models / f"cup_stages_frame{args.tag}_{memory}.pt")
            print(f"{memory:<5} stage {ev['stage_acc']:.3f} done_auroc {ev['done_auroc']:.3f}", flush=True)
    else:
        seqs = [rollout_sequence(p, args.arm) for p in args.paths]
        results["models"] = {}
        for memory in args.memories:
            ck = torch.load(args.models / f"cup_stages_frame{args.tag}_{memory}.pt", weights_only=False)
            model = EpisodeProgressMemory(**ck["config"])
            model.load_state_dict(ck["state_dict"])
            ev = evaluate(model, seqs, list(STAGES))["all"]
            results["models"][memory] = ev
            print(
                f"{memory:<5} rollouts {len(seqs)}: stage {ev['stage_acc']:.3f} done_auroc {ev['done_auroc']:.3f}",
                flush=True,
            )
    if args.out:
        args.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
