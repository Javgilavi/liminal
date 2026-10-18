"""Labelled multi-stage sim episodes for the P2 progress memory.

Runs plans of the demo task's skills (``ws4/orchestrator/skills_sim_pack_box.yaml``: put_apple_in_box, put_block_in_box,
open_lid, close_lid) in the orchestrator's pack-box scene with its scripted stand-ins for FLUX
(``ws4/orchestrator/sim_backend.py``: ``PackBoxSim``, ``ScriptedSimBackend``), all used read-only. The executor
here is deliberately minimal and sequential: one step at a time, one chunk of 42 commands per re-plan of which 32
run, the other arm holding still, a retry when the ground-truth check fails. Each episode directory gets:

* ``episode.json``: :class:`labels.EpisodeLabels` (ground-truth segments, re-plan ticks) plus, in ``extra``,
  one record per re-plan (tick, arm, step, FLUX prompt, snapshot ticks);
* ``ticks.npz``: ``state``/``command`` (T, 12) float32 at 30 Hz, ``stage`` (T,) int;
* ``frames.npz``: per re-plan the two FLUX snapshot ticks (window index 0 and 7, ``flux/contract.py``) as JPEG
  bytes for ``head``, ``left_wrist`` and ``right_wrist`` (240x320, the orchestrator's sim image size).

Task families (``TASKS``): ``pack`` (3 steps, progress visible in the scene), ``pack_lidded`` (6 steps with the
lid opened and closed twice), ``lid_cycles_N`` (open/close the lid N times: identical-looking repeated stages,
where only memory can tell the cycles apart).

    MUJOCO_GL=egl python ws4/ttt/sim_episodes.py --out ws4/.cache/ttt/sim --episodes 40 --task lid_cycles_3
    MUJOCO_GL=egl python ws4/ttt/sim_episodes.py --out ws4/.cache/ttt/sim --mix --episodes 200 --workers 4
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ttt.labels import EpisodeLabels, Segment  # noqa: E402

WS4 = Path(__file__).resolve().parents[1]
CAMERAS = ("head", "left_wrist", "right_wrist")
N_OBS_STEPS, SNAPSHOT_WINDOW = 8, (0, 7)  # SO-101 contract: 8 ticks, snapshots of the oldest and current tick
CHUNK, EXECUTE = 42, 32


def _task_steps(task: str) -> tuple[bool, list[tuple[str, str]]]:
    """``(lid_on_box_at_start, [(skill, arm), ...])``."""
    if task == "pack":
        # The lid starts parked on the right (orchestrator NOMINAL_XY), so the right arm closes it.
        return False, [("put_apple_in_box", "left"), ("put_block_in_box", "right"), ("close_lid", "right")]
    if task == "pack_lidded":
        return True, [
            ("open_lid", "left"),
            ("put_apple_in_box", "left"),
            ("close_lid", "left"),
            ("open_lid", "left"),
            ("put_block_in_box", "right"),
            ("close_lid", "left"),
        ]
    if task.startswith("lid_cycles_"):
        n = int(task.rsplit("_", 1)[1])
        return True, [("open_lid", "left"), ("close_lid", "left")] * n
    raise ValueError(f"unknown task {task!r}")


TASKS = ("pack", "pack_lidded", "lid_cycles_2", "lid_cycles_3", "lid_cycles_4")
MIX = {"pack": 0.2, "pack_lidded": 0.3, "lid_cycles_2": 0.2, "lid_cycles_3": 0.2, "lid_cycles_4": 0.1}


def _jpeg(img: np.ndarray, quality: int = 90) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def decode_jpeg(data: bytes | np.ndarray) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(io.BytesIO(bytes(data))).convert("RGB"))


def _import_orchestrator():
    if str(WS4) not in sys.path:
        sys.path.insert(0, str(WS4))
    import orchestrator  # noqa: F401  (adds the repo root for ws4.sim imports)
    from flux.backends import PolicyRequest
    from flux.contract import arm_slice
    from orchestrator.sim_backend import PackBoxSim, ScriptedSimBackend, SimSkillError
    from orchestrator.skills import SIM_VOCABULARY, load_vocabulary

    def load_sim_vocabulary():  # the pack-box skills (skills.yaml holds the real robot's demo task)
        return load_vocabulary(SIM_VOCABULARY)

    return PolicyRequest, arm_slice, PackBoxSim, ScriptedSimBackend, SimSkillError, load_sim_vocabulary


def generate_episode(task: str, seed: int, out_dir: Path, max_chunks: int = 14, max_attempts: int = 2) -> dict:
    """Run one episode and write it to ``out_dir``. Returns a short summary."""
    PolicyRequest, arm_slice, PackBoxSim, ScriptedSimBackend, SimSkillError, load_vocabulary = _import_orchestrator()
    vocab = load_vocabulary()
    lid_on_box, plan = _task_steps(task)
    rng = np.random.default_rng(seed)
    sim = PackBoxSim(randomize=True, render=False, cameras=CAMERAS)
    obs = sim.reset(seed=seed, lid_on_box=lid_on_box)
    env = sim.env
    backend = ScriptedSimBackend(sim)

    states, commands, stages = [], [], []
    command = np.asarray(obs["observation.state"], dtype=np.float64).copy()
    frame_ticks: dict[int, dict[str, bytes]] = {}

    def snap(tick: int) -> None:
        if tick not in frame_ticks:
            frame_ticks[tick] = {cam: _jpeg(env.render(cam)) for cam in CAMERAS}

    def step_env(cmd: np.ndarray, stage: int) -> np.ndarray:
        nonlocal obs
        states.append(np.asarray(obs["observation.state"], dtype=np.float32))
        commands.append(cmd.astype(np.float32))
        stages.append(stage)
        obs = env.step(cmd.astype(np.float32))[0]
        return np.asarray(obs["observation.state"], dtype=np.float64)

    steps = [{"id": f"s{i + 1}", "skill": s, "arm": a} for i, (s, a) in enumerate(plan)]
    segments, replans, errors = [], [], []
    for k, st in enumerate(steps):
        skill, arm = vocab[st["skill"]], st["arm"]
        for attempt in range(1, max_attempts + 1):
            for _ in range(int(rng.integers(0, 16))):  # hesitation before a step starts (varies stage lengths)
                step_env(command, -1)
            start = len(states)
            try:
                backend.begin(arm, arm, skill)
            except SimSkillError as e:  # the stand-in cannot plan this step here: record a failed, empty attempt
                segments.append(Segment(st["id"], st["skill"], arm, start, start + 1, False, attempt))
                step_env(command, k)
                errors.append({"tick": start, "step_id": st["id"], "error": str(e)})
                break
            step_start = start
            for _ in range(max_chunks):
                t = len(states)
                # The runner resets the history at a step start, so the oldest snapshot never predates it.
                snap_ticks = [max(step_start, t - (N_OBS_STEPS - 1 - SNAPSHOT_WINDOW[0])), t]
                snap(t)  # the current frame is rendered now; the older one was rendered when it happened
                replans.append(
                    {"tick": t, "arm": arm, "step_id": st["id"], "prompt": skill.prompt, "snapshot_ticks": snap_ticks}
                )
                req = PolicyRequest(
                    skill.prompt,
                    np.zeros((N_OBS_STEPS, 6)),
                    np.tile(command[arm_slice(arm)], (N_OBS_STEPS, 1)),
                    {},
                    SNAPSHOT_WINDOW,
                    target=arm,
                )
                chunk = backend.predict(req).commands[:EXECUTE]
                for j, a in enumerate(chunk):
                    if j == EXECUTE - (N_OBS_STEPS - 1):
                        snap(len(states))  # the next re-plan's oldest snapshot
                    command[arm_slice(arm)] = a
                    state = step_env(command, k)
                if backend.finished(arm, state[arm_slice(arm)]):
                    break
            backend.end(arm)
            ok, detail = sim.predicate(skill.sim_check, obs, arm) if skill.sim_check else (True, "")
            segments.append(Segment(st["id"], st["skill"], arm, start, len(states), bool(ok), attempt))
            if ok:
                break
    for _ in range(8):  # a short tail so the last stage is not cut at the very last tick
        step_env(command, -1)

    n_ticks = len(states)
    labels = EpisodeLabels(
        episode_id=out_dir.name,
        source="sim_gt",
        fps=30.0,
        n_ticks=n_ticks,
        task=task,
        steps=steps,
        segments=segments,
        replan_ticks=[r["tick"] for r in replans],
        extra={
            "seed": seed,
            "layout": sim.layout,
            "replans": replans,
            "errors": errors,
            "generator": "ws4/ttt/sim_episodes.py",
        },
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "ticks.npz",
        state=np.asarray(states, np.float32),
        command=np.asarray(commands, np.float32),
        stage=np.asarray(stages, np.int64),
    )
    ticks_sorted = sorted(frame_ticks)
    frames = {"frame_ticks": np.asarray(ticks_sorted, np.int64)}
    for cam in CAMERAS:
        frames[cam] = np.array([np.frombuffer(frame_ticks[t][cam], np.uint8) for t in ticks_sorted], dtype=object)
    np.savez(out_dir / "frames.npz", **frames)
    labels.save(out_dir / "episode.json")  # written last: a directory with episode.json is complete
    env.close()
    return {
        "episode": out_dir.name,
        "task": task,
        "ticks": n_ticks,
        "replans": len(replans),
        "segments": len(segments),
        "success": all(any(g.success for g in segments if g.step_id == st["id"]) for st in steps),
    }


def load_frames(ep_dir: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """``(frame_ticks, {camera: object array of JPEG bytes})``."""
    z = np.load(ep_dir / "frames.npz", allow_pickle=True)
    return z["frame_ticks"], {cam: z[cam] for cam in CAMERAS if cam in z}


def _worker(job: tuple[str, int, str]) -> dict:
    task, seed, out = job
    t0 = time.time()
    summary = generate_episode(task, seed, Path(out))
    summary["seconds"] = round(time.time() - t0, 1)
    print(json.dumps(summary), flush=True)
    return summary


def main(argv: list[str] | None = None) -> None:
    from ttt.throttle import throttle

    max_workers = throttle()  # shares the laptop with 30 Hz robot loops
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, default=WS4 / ".cache" / "ttt" / "sim")
    ap.add_argument("--task", default="pack_lidded", choices=TASKS)
    ap.add_argument("--mix", action="store_true", help="sample tasks from MIX instead of --task")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args(argv)
    args.workers = min(args.workers, max_workers)
    os.environ.setdefault("MUJOCO_GL", "egl")
    rng = np.random.default_rng(args.seed)
    names, probs = zip(*MIX.items(), strict=True)
    jobs = []
    for i in range(args.episodes):
        task = str(rng.choice(names, p=probs)) if args.mix else args.task
        seed = args.seed * 100_000 + i
        jobs.append((task, seed, str(args.out / f"{task}-{seed:06d}")))
    if args.workers > 1:
        from multiprocessing import get_context

        with get_context("spawn").Pool(args.workers) as pool:
            summaries = pool.map(_worker, jobs, chunksize=1)
    else:
        summaries = [_worker(j) for j in jobs]
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "index.jsonl").open("a") as f:
        for s in summaries:
            f.write(json.dumps(s) + "\n")


if __name__ == "__main__":
    main()
