"""Would the WHILE (right holds lid_open) survive the pen skills? Head frames every 1 s through each kept pen take,
the exact ws4_haiku lid request on each; the policy is taken as 5x slower than the teleop demos (red-pen inserts took
48-94 s vs 12-18 s takes), so one frame stands for one 5 s WHILE refresh. A refresh breaks the WHILE when its value is
not true; with keep_s, an unclear verdict reuses the last confident one up to keep_s old (seeded true: the lid check
before the skill).

    PYTHONPATH=src python ws4/eval/while_lid_refresh.py <out_dir>     (repo root; Anthropic key; ~USD 0.12, ~2 min)

Reads the recorded datasets in ~/.cache/huggingface/lerobot/ws4 (never while a recorder writes to them).
2026-10-18 09:05: 53 takes, 785 checks, 0 false, 74 unclear; breaks without keep_s 37/53, 15 s 3/53, 30 s 0/53."""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from liminal.ws4_bridge.vision import Ws4HaikuVisionModel

ROOT = Path.home() / ".cache/huggingface/lerobot/ws4"
SETS = {"green": ("haller_put_the_green_pen_in_the_green_slot", 4),
        "cutter": ("haller_put_the_yellow_cutter_in_the_yellow_slot", 4),
        "red_lid_held": ("haller_put_the_red_pen_in_its_slot_lid_held", 1)}
STRETCH, REFRESH = 5.0, 5.0
out_dir = Path(sys.argv[1])


def frames(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    for i in range(0, n, round(fps * REFRESH / STRETCH)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, bgr = cap.read()
        if ok:
            out.append(bgr[:, :, ::-1].copy())
    cap.release()
    return out


episodes = []
for name, (ds, every) in SETS.items():
    root = ROOT / ds
    review = json.loads((root / "review.json").read_text())["episodes"]
    files = sorted((root / "videos/observation.images.head").glob("chunk-*/file-*.mp4"))
    for i, f in enumerate(files):
        ep = int(f.stem.split("-")[1]) + 1000 * int(f.parent.name.split("-")[1])
        if review.get(str(ep), {}).get("status") != "reject" and i % every == 0:
            episodes.append((name, ep, frames(f)))
print(len(episodes), "takes,", sum(len(e[2]) for e in episodes), "refreshes", flush=True)
model = Ws4HaikuVisionModel(budget_usd=0.5)
model.load()
flat = [(k, j, img) for k, (_, _, imgs) in enumerate(episodes) for j, img in enumerate(imgs)]
with ThreadPoolExecutor(6) as pool:
    raw = list(pool.map(lambda x: model._check("lid_open", np.asarray(x[2]))["value"], flat))
seq = {}
for (k, j, _), v in zip(flat, raw, strict=True):
    seq.setdefault(k, {})[j] = v


def breaks(values, keep_s):
    last_true_t = 0.0
    for j, v in enumerate(values):
        t = (j + 1) * REFRESH
        if v == "true":
            last_true_t = t
        elif v == "false" or t - last_true_t > keep_s:
            return j
    return None


report = {}
for keep in (0.0, 15.0, 30.0, 45.0):
    by = {}
    for k, (name, _, _) in enumerate(episodes):
        vals = [seq[k][j] for j in sorted(seq[k])]
        b = breaks(vals, keep)
        by.setdefault(name, [0, 0])
        by[name][0] += b is not None
        by[name][1] += 1
    report[f"keep_{keep:.0f}s"] = {n: f"{b}/{t} takes break" for n, (b, t) in by.items()}
    print(f"keep_s={keep:.0f}:", report[f"keep_{keep:.0f}s"])
counts = {v: raw.count(v) for v in ("true", "false", "unknown")}
print("verdicts", counts, f"USD {model._verifier.llm.meter.total_usd:.4f}")
runs = []
for k in seq:
    vals, cur = [seq[k][j] for j in sorted(seq[k])], 0
    for v in vals:
        cur = cur + 1 if v != "true" else 0
        runs.append(cur)
print("longest run of non-true refreshes:", max(runs), f"(= {max(runs) * REFRESH:.0f} s of policy time)")
(out_dir / "while_refresh_sim.json").write_text(json.dumps({"report": report, "verdicts": counts, "episodes": [
    {"set": n, "ep": ep, "values": [seq[k][j] for j in sorted(seq[k])]} for k, (n, ep, _) in enumerate(episodes)]},
    indent=1))
