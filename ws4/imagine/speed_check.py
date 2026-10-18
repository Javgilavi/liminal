"""Which joints does the policy move, compared with the demonstrations? Model vs demo joint travel on held-out takes.

For each chunk start t (every 32 ticks) of each episode, the request is built exactly as the runner builds it
(``single_arm/offline_diag.py`` ``window``: states s[t-7..t], commands a[t-8..t-1], image snapshots at the model's
snapshot ticks). The model's first 32 commands are compared with the recorded actions a[t..t+31]:
- path: summed absolute per-tick change per joint over the 32 steps (how much each joint moves);
- disp: absolute displacement from the anchor a[t-1] to step 32.
The ratios are model / demo, summed over chunks where the demo moved the arm by more than 5 degrees of total
joint travel. On the GPU box (one model, about 20 GB with the text encoder on the CPU):

    cd $LIMINAL/ws4 && python -m imagine.speed_check --package /workspace/packages/so101_haller_pen_B84 \\
        [--adapter <lora checkpoint>/pretrained_model_ema] --root /workspace/real_lora/pen_in_right_final \\
        --episodes 101 102 103 20 21 22 36 37 38 86 87 88 --out /workspace/imagine/speed_5k
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

TASK = "put the red pen in its slot in the cardboard box"
JOINTS = ("pan", "lift", "elbow", "wrist_flex", "roll", "gripper")


def build_backend(package: str, adapter: str | None, guidance: float, compile_model: bool):
    ws4 = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(ws4), str(ws4 / "cloud"), str(ws4 / "single_arm")]
    overrides = {"guidance_scale": guidance, "guidance_scale_action": guidance, "compile_model": compile_model}
    from lora_adapter import load_policy, to_backend

    return to_backend(load_policy(package, adapter), package, overrides, True, Path(adapter or package).name)


def summarize(rows: list[dict]) -> dict:
    p = {k: np.array([r[k] for r in rows]) for k in ("pred_path", "rec_path", "pred_disp", "rec_disp")}
    moving = p["rec_path"][:, :5].sum(1) > 5
    out: dict = {"chunks": len(rows), "moving_chunks": int(moving.sum()), "joints": list(JOINTS)}
    for k in ("path", "disp"):
        pr, rc = p["pred_" + k][moving], p["rec_" + k][moving]
        out[f"ratio_{k}_model_over_demo"] = np.round(pr.sum(0) / np.maximum(rc.sum(0), 1e-9), 2).tolist()
        out[f"median_{k}_model_deg"] = np.round(np.median(pr, 0), 2).tolist()
        out[f"median_{k}_demo_deg"] = np.round(np.median(rc, 0), 2).tolist()
    travel_p, travel_r = p["pred_path"][moving][:, :5].sum(1), p["rec_path"][moving][:, :5].sum(1)
    out["median_arm_travel_deg"] = {"model": round(float(np.median(travel_p)), 1),
                                    "demo": round(float(np.median(travel_r)), 1)}  # fmt: skip
    out["fraction_model_travels_more"] = round(float(np.mean(travel_p > travel_r)), 2)
    return out


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--package", required=True, help="package path, or the package name on --url")
    ap.add_argument("--adapter", default=None, help="LoRA checkpoint dir (pretrained_model_ema), folded in memory")
    ap.add_argument("--url", default=None, help="query a running flux.remote server instead (no GPU here)")
    ap.add_argument("--token-file", default=None)
    ap.add_argument("--root", required=True, help="per-arm LeRobot dataset root")
    ap.add_argument("--repo-id", default="pen")
    ap.add_argument("--episodes", type=int, nargs="+", required=True)
    ap.add_argument("--task", default=TASK)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.url:
        ws4 = Path(__file__).resolve().parents[1]
        sys.path[:0] = [str(ws4), str(ws4 / "single_arm")]
        from flux.remote import RemoteBackend

        token = Path(args.token_file).read_text().strip() if args.token_file else None
        backend = RemoteBackend(args.url, args.package, token=token, timeout_s=120)
    else:
        backend = build_backend(args.package, args.adapter, args.guidance, args.compile)
    from flux.backends import PolicyRequest
    from flux.cameras import CameraView, prepare_image
    from flux.package_tools import dataset_arrays
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from offline_diag import N_EXEC, N_OBS, window

    c = backend.contract
    rows = []
    for ep in args.episodes:
        arr = dataset_arrays(args.repo_id, root=args.root, episodes=[ep])
        ds = LeRobotDataset(args.repo_id, root=args.root, episodes=[ep])
        states, actions = arr["observation.state"].astype(float), arr["action"].astype(float)

        def img(i: int, key: str, ds=ds) -> np.ndarray:
            rgb = (ds[int(i)][key].permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
            return prepare_image(rgb, CameraView(key), c.image_hw)

        for t in range(N_OBS, len(actions) - N_EXEC, N_EXEC):
            s, cmd = window(states, actions, t)
            ims = {k: np.stack([img(t - N_OBS + 1 + tick, k) for tick in c.snapshots]) for k in c.camera_keys}
            pred = backend.predict(PolicyRequest(args.task, s, cmd, ims, c.snapshots)).commands[:N_EXEC]
            rec, anchor = actions[t : t + N_EXEC], actions[t - 1]
            rows.append({
                "ep": ep, "t": t,
                "pred_path": np.abs(np.diff(np.vstack([anchor, pred]), axis=0)).sum(0).tolist(),
                "rec_path": np.abs(np.diff(np.vstack([anchor, rec]), axis=0)).sum(0).tolist(),
                "pred_disp": np.abs(pred[-1] - anchor).tolist(), "rec_disp": np.abs(rec[-1] - anchor).tolist(),
            })  # fmt: skip
        print(f"episode {ep}: {len(rows)} chunks so far", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"args": {k: v for k, v in vars(args).items() if k != "token_file"}, **summarize(rows)}
    (out / "rows.json").write_text(json.dumps(rows))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    main()
