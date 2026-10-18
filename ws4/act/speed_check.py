"""Joint travel of ACT checkpoints vs the demonstrations, as ``imagine/speed_check.py`` measures FLUX.

Same chunk starts (t = 8, 40, 72, ... of each held-out take), same 32 compared commands, same anchor a[t-1] and the
same summary (``imagine.speed_check.summarize``: path and displacement ratios model / demo over chunks where the
demo moved the arm more than 5 deg). Only the request differs: ACT sees s[t] and the frames at t.

    python ws4/act/speed_check.py --root /workspace/real_lora/pen_in_right_final \\
        --episodes 101 102 103 20 21 22 36 37 38 86 87 88 --checkpoint act_100k=<pretrained_model> --out DIR
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

WS4 = Path(__file__).resolve().parents[1]
for path in (WS4, WS4 / "single_arm"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PRETRAINED_MODEL_DIR")
    p.add_argument("--root", required=True)
    p.add_argument("--repo-id", default="pen")
    p.add_argument("--episodes", type=int, nargs="+", required=True)
    p.add_argument("--task", default="put the red pen in its slot in the cardboard box")
    p.add_argument("--device", default="auto")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    from flux.backends import PolicyRequest
    from flux.cameras import CameraView, prepare_image
    from flux.package_tools import dataset_arrays
    from imagine.speed_check import summarize
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from offline_diag import N_EXEC, N_OBS

    from act.backend import ActBackend

    backends = {}
    for item in args.checkpoint:
        name, path = item.split("=", 1)
        backends[name] = ActBackend.from_pretrained(path, device=args.device, name=name)
    c = next(iter(backends.values())).contract
    rows: dict[str, list[dict]] = {name: [] for name in backends}
    for ep in args.episodes:
        arr = dataset_arrays(args.repo_id, root=args.root, episodes=[ep])
        ds = LeRobotDataset(args.repo_id, root=args.root, episodes=[ep])
        states, actions = arr["observation.state"].astype(float), arr["action"].astype(float)
        for t in range(N_OBS, len(actions) - N_EXEC, N_EXEC):
            item = ds[int(t)]
            ims = {}
            for key in c.camera_keys:
                rgb = (item[key].permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
                ims[key] = prepare_image(rgb, CameraView(key), c.image_hw)[None]
            req = PolicyRequest(args.task, states[t][None], actions[t - 1][None], ims, c.snapshots)
            rec, anchor = actions[t : t + N_EXEC], actions[t - 1]
            for name, backend in backends.items():
                pred = backend.predict(req).commands[:N_EXEC]
                rows[name].append({
                    "ep": ep, "t": t,
                    "pred_path": np.abs(np.diff(np.vstack([anchor, pred]), axis=0)).sum(0).tolist(),
                    "rec_path": np.abs(np.diff(np.vstack([anchor, rec]), axis=0)).sum(0).tolist(),
                    "pred_disp": np.abs(pred[-1] - anchor).tolist(), "rec_disp": np.abs(rec[-1] - anchor).tolist(),
                })  # fmt: skip
        print(f"episode {ep}: {len(rows[name])} chunks so far", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"args": vars(args), **{name: summarize(r) for name, r in rows.items()}}
    (out / "rows.json").write_text(json.dumps(rows))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "args"}, indent=1))


if __name__ == "__main__":
    main()
