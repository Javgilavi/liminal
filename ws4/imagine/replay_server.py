"""Replay a recorded episode through a running ``flux.remote`` server with imagination on, without the robot.

Requests are built exactly as the real-arm runner builds them (ws4/single_arm/offline_diag.py ``window``):
states s[t-7..t], the command issued before each observation a[t-8..t-1], and image snapshots at the model's
snapshot ticks (frames t-7 and t), each prepared with ``flux.cameras.prepare_image``. One chunk every
``n_action_steps`` ticks, open loop: the recorded future follows the demonstrator, not the model.

For every chunk the same request is sent twice, once plain and once with ``imagine``, to measure the added round
trip and to check that the commands are identical. The frames go through ``AttemptImagination``, the recorder the
runner uses, so the output folder has the same layout and renders with ``python -m imagine.attempt``.

On the pod (server already running on PORT; ws4/ on sys.path):

    cd $LIMINAL/ws4 && python -m imagine.replay_server --url http://127.0.0.1:PORT --package so101 \\
        --token-file /workspace/flux_token --root <per-arm dataset> --repo-id <name> --episode 0 \\
        --task "put the red pen in its slot in the cardboard box" --chunks 8 --out /workspace/imagine/replay_ep0
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from .attempt import AttemptImagination, render_attempt


def load_episode(repo_id: str, root: str | None, episode: int, keys: tuple[str, str]):
    """States, actions and the two camera streams (float CHW in [0, 1]) of one episode, indexed 0..n-1."""
    from flux.package_tools import dataset_arrays
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    arrays = dataset_arrays(repo_id, root=root, episodes=[episode])
    if not (arrays["episode_index"] == episode).all():
        raise ValueError("dataset_arrays returned other episodes")
    ds = LeRobotDataset(repo_id, root=root, episodes=[episode])
    if len(ds) != len(arrays["action"]):
        raise ValueError(f"{len(ds)} frames vs {len(arrays['action'])} rows")
    return arrays["observation.state"].astype(float), arrays["action"].astype(float), ds


def rgb(ds, i: int, key: str) -> np.ndarray:
    img = ds[int(i)][key]
    return (img.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--url", required=True)
    p.add_argument("--package", default="so101")
    p.add_argument("--token-file", default=None)
    p.add_argument("--repo-id", required=True)
    p.add_argument("--root", default=None)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--task", required=True)
    p.add_argument("--start", type=int, default=8, help="first chunk tick (>= 8 for a full history)")
    p.add_argument("--chunks", type=int, default=8)
    p.add_argument("--imagine", choices=("lazy", "async", "sync"), default="lazy")
    p.add_argument("--no-stock", action="store_true", help="skip the plain request (no latency or identity check)")
    p.add_argument("--scene-key", default="observation.images.scene")
    p.add_argument("--wrist-key", default="observation.images.wrist")
    p.add_argument("--gif", action="store_true", help="also write a GIF of the first 3 chunks")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    from flux.backends import PolicyRequest
    from flux.cameras import CameraView, prepare_image
    from flux.remote import RemoteBackend

    token = Path(args.token_file).read_text().strip() if args.token_file else None
    plain = RemoteBackend(args.url, args.package, token=token, timeout_s=120)
    imag = RemoteBackend(args.url, args.package, token=token, timeout_s=120, imagine=args.imagine)
    contract = plain.contract
    n_obs, n_exec, chunk = contract.n_obs_steps, contract.n_action_steps, contract.chunk_size
    states, actions, ds = load_episode(args.repo_id, args.root, args.episode, (args.scene_key, args.wrist_key))
    n = len(states)
    starts = [t for t in range(max(args.start, n_obs), n - n_exec, n_exec)][: args.chunks]
    if not starts:
        raise SystemExit(f"episode {args.episode} has {n} frames: too short")
    hw = contract.image_hw
    cams = {contract.camera_keys[0]: args.scene_key, contract.camera_keys[1]: args.wrist_key}

    def model_image(i: int, model_key: str) -> np.ndarray:
        return prepare_image(rgb(ds, i, cams[model_key]), CameraView(model_key), hw)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    irec = AttemptImagination(out)
    rows, last = [], min(n - 1, starts[-1] + chunk)
    responses = {}
    for k, t in enumerate(starts):
        imgs = {key: np.stack([model_image(t - n_obs + 1 + tick, key) for tick in contract.snapshots])
                for key in contract.camera_keys}  # fmt: skip
        req = PolicyRequest(args.task, states[t - n_obs + 1 : t + 1], actions[t - n_obs : t], imgs, contract.snapshots)
        row = {"chunk": k, "tick": t}
        if not args.no_stock:
            stock = plain.predict(req)
            row["stock_round_trip_s"] = stock.latency_s
            row["stock_server_s"] = stock.info.get("server_latency_s")
        t0 = time.perf_counter()
        resp = imag.predict(req)
        row["imagine_round_trip_s"] = time.perf_counter() - t0
        row["imagine_server_s"] = resp.info.get("server_latency_s")
        row["response_bytes_meta"] = resp.imagination["meta"].get("lengths") if resp.imagination else None
        if not args.no_stock:
            row["identical_commands"] = bool(np.array_equal(stock.commands, resp.commands))
            row["max_abs_diff"] = float(np.abs(stock.commands - resp.commands).max())
        responses[t] = resp
        rows.append(row)
        print(json.dumps(row, default=float), flush=True)
    for i in range(starts[0], last + 1):  # the recorder sees the episode as the runner would see the robot
        raw = {"scene": model_image(i, contract.camera_keys[0]), "wrist": model_image(i, contract.camera_keys[1])}
        irec.tick(i, raw, replanned=i in responses, response=responses.get(i))
    summary = irec.finish(imag, task=args.task)
    video = render_attempt(out, task=args.task)
    gif = None
    if args.gif:
        from .attempt import load_attempt
        from .compose import render_run

        records, actual = load_attempt(out)
        keep = records[:3]
        end = keep[-1].step + n_exec
        gif = out / "imagination" / "imagined_vs_actual_first3.gif"
        render_run(keep, [a for a in actual if a.step < end], gif, task=args.task)

    def med(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return statistics.median(vals) if vals else None

    report = {
        "args": {k: v for k, v in vars(args).items() if k != "token_file"},
        "contract": {"n_obs_steps": n_obs, "n_action_steps": n_exec, "chunk_size": chunk, "image_hw": list(hw)},
        "chunks": rows,
        "median_stock_round_trip_s": med("stock_round_trip_s"),
        "median_imagine_round_trip_s": med("imagine_round_trip_s"),
        "median_stock_server_s": med("stock_server_s"),
        "median_imagine_server_s": med("imagine_server_s"),
        "all_identical": all(r.get("identical_commands", True) for r in rows),
        "decode_s": [c.get("decode_s") for c in summary["chunks"]],
        "fetch_s_total": summary.get("fetch_s"),
        "video": str(video),
        "gif": str(gif) if gif else None,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float))
    print(json.dumps({k: v for k, v in report.items() if k != "chunks"}, indent=2, default=float))
    return report


if __name__ == "__main__":
    main()
