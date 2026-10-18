"""Closed-loop success of one FLUX package on "put the block in the cup" over the held-out seeded sim layouts.

The task arm (left) runs ``flux.dual_arm_runner.DualArmRunner`` with ``per_arm_target`` (head render as ``scene``,
left wrist render as ``wrist``, 480x640 like the demos), the default ``SafetyGuard`` and ``flux.remote.RemoteBackend``
over the SSH tunnel; inference is synchronous (no sim time passes while a chunk is computed). Layouts come from
``results/block_cup_eval_seeds_left.json`` (seeds the scripted expert solves, disjoint from the demo seeds), with
the task arm starting at the haller home like the demos (``sim_block_cup.reset_layout``).

Two success numbers per rollout, from the ground-truth check (``sim_block_cup.block_in_cup``) at chunk boundaries:

* ``success_at_check``: the check passed at some re-plan within the budget; the orchestrator would end the step there
  (FLUX never stops by itself). This is the number to compare with the orchestrator's step success.
* ``success_final``: after the first pass the policy keeps running ``--after-pass`` more chunks (the demos release,
  retreat and go home in about 3), then the block must still be in the cup after a short settle. Without a pass,
  the rollout runs the whole ``--max-chunks`` budget.

Each rollout is written in the ``sim_episodes.py`` layout (``episode.json``, ``ticks.npz`` with ``block``/``cup``
positions, ``frames.npz`` at every re-plan and its oldest snapshot) so ``cup_stages.py score`` can track stages on it.

    MUJOCO_GL=egl python ws4/ttt/eval_block_cup.py --backend fake --n 1 --max-chunks 2     # plumbing, no GPU
    MUJOCO_GL=egl python ws4/ttt/eval_block_cup.py --package so101 --token-file ~/.flux_token --n 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ttt import sim_block_cup as sbc  # noqa: E402
from ttt.labels import EpisodeLabels, Segment  # noqa: E402

WS4 = Path(__file__).resolve().parents[1]
SEEDS = WS4 / "ttt" / "results" / "block_cup_eval_seeds_left.json"
N_OBS_STEPS = 8
SETTLE_TICKS = 15


class _CupRobot:
    """``RobotInterface`` over the cup env: renders head and the task wrist, logs every executed tick."""

    def __init__(self, env, arm: str):
        self.env, self.arm = env, arm
        self.cams = ("head", f"{arm}_wrist")
        self.states, self.commands, self.block, self.cup, self.tcp = [], [], [], [], []
        self.recent: deque = deque(maxlen=N_OBS_STEPS)
        self.on_tick = None
        self._obs: dict = {}

    def reset(self, seed: int) -> dict:
        sbc.reset_layout(self.env, self.arm, seed)  # the demos' start: task arm at the haller home
        self._obs = self._observe()
        return self._obs

    def _observe(self) -> dict:
        obs = self.env.get_obs()
        for cam in self.cams:
            obs[f"observation.images.{cam}"] = self.env.render(cam)
        return obs

    @property
    def tick(self) -> int:
        return len(self.states)

    def get_observation(self) -> dict:
        return self._obs

    def send_action(self, action: np.ndarray) -> None:
        obs, poses = self._obs, self._obs["privileged"]["object_poses"]
        if self.on_tick is not None:
            self.on_tick(self.tick, obs)
        self.recent.append((self.tick, obs))
        self.states.append(np.asarray(obs["observation.state"], np.float32))
        self.commands.append(np.asarray(action, np.float32))
        self.block.append(poses[sbc.BLOCK][:3].astype(np.float32))
        self.cup.append(poses[sbc.CUP][:3].astype(np.float32))
        self.tcp.append(obs["privileged"]["ee_poses"][self.arm][:3].astype(np.float32))
        self.env.step(np.asarray(action, np.float32))
        self._obs = self._observe()

    def obs_at(self, tick: int) -> dict | None:
        return next((o for t, o in self.recent if t == tick), None)


def run_episode(args, env, policy, layout: dict, out_dir: Path, video: bool) -> dict:
    from flux import dual_arm_runner, safety

    from ttt.cup_stages import STAGES, stage_from_poses
    from ttt.flux_sim_closed_loop import _jpeg, _Video

    arm, seed = args.arm, int(layout["seed"])
    rng = np.random.default_rng(seed)
    robot = _CupRobot(env, arm)
    robot.reset(seed)
    target = dual_arm_runner.per_arm_target(arm, "policy")
    runner = dual_arm_runner.DualArmRunner(robot, [target], {"policy": policy}, safety.SafetyGuard(),
                                           async_inference=False)  # fmt: skip
    out_dir.mkdir(parents=True, exist_ok=True)
    frames: dict[int, dict[str, bytes]] = {}
    replans: list[dict] = []
    live = {"check": "", "chunk": 0, "passed_at": None}
    vid = _Video(out_dir / "rollout.mp4", (480, 640), scale=1) if video else None

    def on_tick(tick: int, ob: dict) -> None:
        if vid is None:
            return
        p = ob["privileged"]["object_poses"]
        gt = int(stage_from_poses(p[sbc.BLOCK][None, :3], p[sbc.CUP][None, :3])[0])
        head = f"t {tick / 30:5.1f} s   FLUX ({args.package}): \"{sbc.TASK}\"   chunk {live['chunk']}"
        lines = [(head, (235, 235, 235), True), (f"ground truth: {live['check']}", (200, 200, 200), False),
                 (f"first passed check: chunk {live['passed_at']}" if live["passed_at"] is not None
                  else "check not passed yet", (120, 220, 120) if live["passed_at"] is not None else (200, 200, 200),
                  False)]  # fmt: skip
        vid.frame(ob["observation.images.head"], ob[f"observation.images.{arm}_wrist"], lines,
                  [(s, 1 if i == gt else 0) for i, s in enumerate(STAGES)])  # fmt: skip

    robot.on_tick = on_tick
    for _ in range(int(rng.integers(5, 15))):  # the demos also start with a short hold
        robot.send_action(robot.get_observation()["observation.state"])
    start = robot.tick
    runner.start(target.name, sbc.TASK)
    chunks, passed_at, lat, guards = 0, None, [], []
    stop_at = args.max_chunks
    while True:
        if not runner.is_busy(target.name):  # the next tick re-plans
            ok, detail = sbc.block_in_cup(env)
            live["check"] = f"block in cup {'PASSED' if ok else 'not met'} ({detail})"
            if ok and passed_at is None and chunks > 0:
                passed_at = chunks
                live["passed_at"] = chunks
                stop_at = min(args.max_chunks, chunks + args.after_pass)
            if chunks >= stop_at:
                break
            t = robot.tick
            ob = robot.get_observation()
            oldest = max(start, t - (N_OBS_STEPS - 1))
            for tt, o in ((t, ob), (oldest, ob if oldest == t else robot.obs_at(oldest))):
                if o is not None and tt not in frames:
                    frames[tt] = {cam: _jpeg(o[f"observation.images.{cam}"]) for cam in robot.cams}
            report = runner.step()
            guards.append(bool(report.guard_intervened))
            ts = runner._targets[target.name]
            latency = ts.last_response.latency_s if ts.last_response is not None else None
            lat.append(latency)
            replans.append({"tick": t, "arm": arm, "step_id": "s1", "prompt": sbc.TASK, "snapshot_ticks": [oldest, t],
                            "backend": "flux", "latency_s": latency, "check": bool(ok)})  # fmt: skip
            chunks += 1
            live["chunk"] = chunks
        else:
            guards.append(bool(runner.step().guard_intervened))
    runner.stop(target.name)
    end = robot.tick
    for _ in range(SETTLE_TICKS):
        robot.send_action(robot.get_observation()["observation.state"])
    final_ok, final_detail = sbc.block_in_cup(env)
    if vid is not None:
        vid.close()
    runner.close()

    block, cup, tcp = (np.asarray(x, np.float32) for x in (robot.block, robot.cup, robot.tcp))
    stage = stage_from_poses(block, cup)
    b0 = block[start]
    result = {
        "seed": seed, "package": args.package, "chunks": chunks, "ticks": end - start,
        "success_at_check": passed_at is not None, "passed_at_chunk": passed_at,
        "success_final": bool(final_ok), "final_check": final_detail,
        "max_stage": STAGES[int(stage.max())],
        "block_moved_max_cm": round(float(np.linalg.norm(block - b0, axis=1).max() * 100), 1),
        "tcp_min_to_block_cm": round(float(np.linalg.norm(tcp[start:end] - block[start:end], axis=1).min() * 100), 1),
        "latency_s_median": float(np.median([x for x in lat if x is not None])) if any(lat) else None,
        "guard_intervened_frac": round(float(np.mean(guards)), 3) if guards else 0.0,
    }  # fmt: skip
    labels = EpisodeLabels(
        episode_id=out_dir.name, source="sim_gt", fps=30.0, n_ticks=robot.tick, task="flux_block_cup",
        steps=[{"id": "s1", "skill": "block_cup", "arm": arm}],
        segments=[Segment("s1", "block_cup", arm, start, end, bool(passed_at is not None), 1)],
        replan_ticks=[r["tick"] for r in replans],
        extra={"seed": seed, "layout": layout, "replans": replans, "result": result,
               "policy": {"backend": args.backend, "package": args.package},
               "generator": "ws4/ttt/eval_block_cup.py"},
    )  # fmt: skip
    i0 = 0 if arm == "left" else 6
    np.savez_compressed(out_dir / "ticks.npz", state=np.asarray(robot.states, np.float32)[:, i0 : i0 + 6],
                        command=np.asarray(robot.commands, np.float32)[:, i0 : i0 + 6], block=block, cup=cup, tcp=tcp,
                        stage=stage)  # fmt: skip
    ticks = sorted(frames)
    arrays = {"frame_ticks": np.asarray(ticks, np.int64)}
    for cam in robot.cams:
        arrays[cam] = np.array([np.frombuffer(frames[t][cam], np.uint8) for t in ticks], dtype=object)
    np.savez(out_dir / "frames.npz", **arrays)
    labels.save(out_dir / "episode.json")  # last: a directory with episode.json is complete
    return result


def summarize(results: list[dict]) -> dict:
    n = len(results)

    def rate(key):
        k = sum(bool(r[key]) for r in results)
        return {"success": k, "n": n, "rate": round(k / n, 3) if n else None}

    stages = {s: sum(r["max_stage"] == s for r in results) for s in ("reach", "carry", "in_cup")}
    return {"package": results[0]["package"] if results else None, "at_check": rate("success_at_check"),
            "final": rate("success_final"), "max_stage": stages,
            "latency_s_median": float(np.median([r["latency_s_median"] for r in results if r["latency_s_median"]]))
            if any(r["latency_s_median"] for r in results) else None}  # fmt: skip


def main(argv: list[str] | None = None) -> None:
    from ttt.throttle import throttle

    throttle()  # the laptop also runs 30 Hz robot loops
    os.environ.setdefault("MUJOCO_GL", "egl")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--backend", choices=("remote", "fake"), default="remote")
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--package", default="so101", help="the server's package name (A0, A0r-sim, LoRA)")
    ap.add_argument("--token-file", default=None, help="else $FLUX_TOKEN; never pass the token on argv")
    ap.add_argument("--arm", default="left")
    ap.add_argument("--seeds", type=Path, default=SEEDS)
    ap.add_argument("--n", type=int, default=20, help="first n layouts of --seeds")
    ap.add_argument("--max-chunks", type=int, default=22, help="budget without a pass (expert: about 11 chunks)")
    ap.add_argument("--after-pass", type=int, default=4, help="chunks after the first passed check")
    ap.add_argument("--videos", type=int, default=3, help="write rollout.mp4 for the first k layouts")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    if args.backend == "remote":
        from flux.remote import RemoteBackend

        token = Path(args.token_file).expanduser().read_text().strip() if args.token_file else os.environ.get(
            "FLUX_TOKEN")  # fmt: skip
        policy = RemoteBackend(args.url, args.package, token=token)
    else:
        from flux.backends import FakeBackend
        from flux.contract import SO101_CONTRACT

        policy = FakeBackend(SO101_CONTRACT, velocity=0.0, delay_s=0.0)
        args.package = "fake"
    layouts = json.loads(args.seeds.read_text())
    layouts = (layouts["layouts"] if isinstance(layouts, dict) else layouts)[: args.n]
    env = sbc.make_env(args.arm)
    out = args.out or WS4 / ".cache" / "ttt" / "block_cup_eval" / args.package
    out.mkdir(parents=True, exist_ok=True)
    done = {json.loads(x)["seed"] for x in (out / "index.jsonl").read_text().splitlines()} if (
        out / "index.jsonl").exists() else set()  # fmt: skip
    results = [json.loads(x) for x in (out / "index.jsonl").read_text().splitlines()] if done else []
    for i, layout in enumerate(layouts):
        if int(layout["seed"]) in done:  # resumable: a dropped tunnel costs only the current layout
            continue
        t0 = time.time()
        r = run_episode(args, env, policy, layout, out / f"seed{layout['seed']}", video=i < args.videos)
        r["seconds"] = round(time.time() - t0, 1)
        results.append(r)
        print(json.dumps(r), flush=True)
        with (out / "index.jsonl").open("a") as f:
            f.write(json.dumps(r) + "\n")
    summary = summarize(results)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
