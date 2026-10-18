"""Dry run of ``single_arm/run_single_arm.py`` with an ACT checkpoint on a recorded take: no arm, camera or pod.

``--via remote`` (default) serves the checkpoint on 127.0.0.1 with ``act.serve``'s protocol and runs the runner
unchanged with ``--backend remote``; ``--via act`` uses the in-process ``--backend act``. The runner's fake arm
(``FakeSO101``) is swapped for one that starts at the take's first recorded state, reaches every guarded command at
once and shows the take's recorded scene and wrist frames, one frame per control tick. The fake path has no
calibration file, so the soft limits are taken from the Haller's (``--calibration``); the keep-out (station zones,
parked-arm capsules, demo envelope) is the final-split workspace file. Everything else is the runner's own code:
``DualArmRunner`` with async inference, the guard (``--max-step`` deg per tick), keep-out refusals, the stall
check, the per-attempt trace and meta.json. Frames are open loop (the recording does not react to the arm), so this
checks plumbing and the safety stack on real ACT chunks, not task success.

    python ws4/act/dry_run.py --checkpoint DIR --root ws4/.cache/real_lora/pen_in_right_all --episode 20 \\
        [--seconds 20] [--via remote|act] [-- extra run_single_arm.py arguments]
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path

import numpy as np

WS4 = Path(__file__).resolve().parents[1]
for path in (WS4, WS4 / "single_arm"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_single_arm as rsa  # noqa: E402
from flux.contract import ARM_JOINTS  # noqa: E402

CALIB = WS4 / "single_arm" / "calibration"
TASK = "put the red pen in its slot in the cardboard box"


def load_take(repo_id: str, root: str, episode: int, n: int) -> dict[str, np.ndarray]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id, root=root, episodes=[episode])
    out: dict[str, list] = {"scene": [], "wrist": [], "state": [], "action": []}
    for i in range(min(n, len(ds))):
        item = ds[i]
        if int(item["episode_index"]) != episode or int(item["frame_index"]) != i:
            raise RuntimeError(f"frame {i}: got episode {int(item['episode_index'])} frame {int(item['frame_index'])}")
        for cam in ("scene", "wrist"):
            img = item[f"observation.images.{cam}"]  # (3, H, W) float [0, 1]
            out[cam].append((img.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8))
        out["state"].append(item["observation.state"].numpy())
        out["action"].append(item["action"].numpy())
    return {k: np.stack(v) for k, v in out.items()}


def replay_class(take: dict[str, np.ndarray]):
    class ReplaySO101(rsa.FakeSO101):
        """Recorded frames, one per sent command; joints follow the guarded command (FakeSO101)."""

        def __init__(self, start: np.ndarray | None = None):
            super().__init__(start=take["state"][0].astype(float))
            self.ticks = 0

        def get_observation(self):
            obs = {f"{j}.pos": float(v) for j, v in zip(ARM_JOINTS, self.q, strict=True)}
            i = min(self.ticks, len(take["scene"]) - 1)
            obs["scene"], obs["wrist"] = take["scene"][i], take["wrist"][i]
            return obs

        def send_action(self, action):
            self.ticks += 1
            return super().send_action(action)

    return ReplaySO101


def main() -> None:
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1 :] if "--" in argv else []
    argv = argv[: argv.index("--")] if "--" in argv else argv
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", required=True, help="ACT pretrained_model folder")
    p.add_argument("--root", required=True, help="per-arm LeRobot dataset holding the take")
    p.add_argument("--repo-id", default="local/pen_in_right")
    p.add_argument("--episode", type=int, default=20, help="a held-out take (A 20-22, B 36-38, C 86-88, D 101-103)")
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--act-device", default="auto")
    p.add_argument("--act-steps", type=int, default=None)
    p.add_argument("--calibration", default=str(CALIB / "haller_right.json"), help="soft limits (LeRobot calib)")
    p.add_argument("--keepout-config", default=str(CALIB / "haller_right_workspace_final.json"))
    p.add_argument("--start-pose", type=float, nargs=6, default=[0, 0, 0, 0, 0, 100], help="the arm's home pose")
    p.add_argument("--task", default=TASK, help="prompt (ACT ignores it; it goes into the attempt's meta)")
    p.add_argument("--trial-task", default="put_pen_in_slot")
    p.add_argument("--via", choices=("remote", "act"), default="remote",
                   help="remote: ACT behind act.serve's protocol on 127.0.0.1, runner --backend remote (unchanged "
                        "runner); act: runner --backend act (in process)")  # fmt: skip
    args = p.parse_args(argv)
    take = load_take(args.repo_id, args.root, args.episode, int(args.seconds * 30) + 60)
    print(f"episode {args.episode}: {len(take['state'])} frames, start {np.round(take['state'][0], 1)}", flush=True)
    rsa.FakeSO101 = replay_class(take)
    build_guard = rsa.build_guard
    rsa.build_guard = lambda a, slot, calib_path, rest, ramp=False: build_guard(
        a, slot, calib_path or Path(args.calibration), rest, ramp)  # fmt: skip
    before = set(rsa.RUNS.glob("*")) if rsa.RUNS.exists() else set()
    server = None
    if args.via == "remote":
        from flux.remote import make_server

        from act.backend import ActBackend

        backend = ActBackend.from_pretrained(args.checkpoint, device=args.act_device, n_action_steps=args.act_steps,
                                             name=f"act ({args.checkpoint})")  # fmt: skip
        server = make_server({"act": backend}, port=0)  # loopback only, like act.serve
        threading.Thread(target=server.serve_forever, daemon=True).start()
        policy_argv = ["--backend", "remote", "--server", f"http://127.0.0.1:{server.server_address[1]}",
                       "--package", "act", "--checkpoint", args.checkpoint]  # fmt: skip
    else:
        policy_argv = ["--backend", "act", "--act-checkpoint", args.checkpoint, "--act-device", args.act_device]
        if args.act_steps:
            policy_argv += ["--act-steps", str(args.act_steps)]
    run_argv = ["--robot", "fake", *policy_argv, "--keepout-config", args.keepout_config,
                "--start-pose", *map(str, args.start_pose), "--task", args.task, "--trial-task", args.trial_task,
                "--seconds", str(args.seconds), "--max-step", "1.0", "--max-relative-target", "20",
                "--attempts", "1", "--yes", "--no-log", "--variant", "ACT dry run", *extra]  # fmt: skip
    try:
        rsa.main(run_argv)
    finally:
        if server is not None:
            server.shutdown()
    sessions = sorted(set(rsa.RUNS.glob("*")) - before)
    if not sessions:
        raise SystemExit("the runner wrote no session folder")
    attempt = sessions[-1] / "attempt01"
    meta = json.loads((attempt / "meta.json").read_text())
    trace = np.load(attempt / "trace.npz")
    cmd = trace["command"]
    n = min(len(cmd), len(take["action"]))
    lat = np.array(meta["chunk_round_trip_s"])
    summary = {
        "episode": args.episode, "checkpoint": args.checkpoint, "via": args.via, "end_reason": meta["end_reason"],
        "ticks": meta["ticks"], "chunks": meta["chunks"],
        "chunk_latency_s_median_max": [float(np.median(lat)), float(lat.max())] if lat.size else None,
        "guard_ticks": meta["guard_ticks"], "keepout_refusals": meta["keepout_refusals"],
        "max_step_deg": float(np.abs(np.diff(cmd[:, :5], axis=0)).max()) if len(cmd) > 1 else 0.0,
        "command_range": [cmd.min(axis=0).round(1).tolist(), cmd.max(axis=0).round(1).tolist()],
        "mae_vs_recorded_actions": np.abs(cmd[:n] - take["action"][:n]).mean(axis=0).round(2).tolist(),
        "attempt_dir": str(attempt),
    }  # fmt: skip
    (attempt / "dry_run.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
