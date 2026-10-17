"""Offline accuracy of the Claude success checks on REAL Haller head frames (no robot, no pod).

    cd ws4 && python -m orchestrator.verify_eval --snapshot <dir> --variants current,scene --budget 1.0

``<dir>`` holds read-only copies of the two LeRobot v3 datasets recorded by VR teleop on 2026-10-17
(``haller_put_the_red_pen_in_its_slot_in_the_cardboard_box``, ``haller_take_the_red_pen_out_of_its_slot``): copy
``meta/``, ``review.json`` and ``videos/`` from ``~/.cache/huggingface/lerobot/ws4/`` first, never read the cache
while a recorder writes to it.

Frames: the head camera's first and last frame of every episode (plus the wrist frame at the same instant). Labels:
- put check ("standing in its slot"): put start = no, kept put end = yes, take start = yes, take end = no;
- take check ("lying on the mat, outside the box"): the opposite;
- review.json rejects by their note: failed takes end with the pen on the mat, ep 23 is reversed (starts in the slot,
  ends on the mat), ep 100 ends standing next to the slot (a hard negative for the put check). Takes whose end frame
  is ambiguous by definition (a hand in the box, the pen in the slot but still gripped) are scored apart as edge cases.
Writes ``runs/verify_eval_<time>.json`` with the confusion matrix, latency and cost per variant and check.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from . import PACKAGE_DIR
from .claude import ClaudeJSON, image_block
from .cost import CostMeter, cost_of
from .verifier import VERIFY_SCHEMA, VERIFY_SYSTEM, crop_frac, upscale

PUT_DS = "haller_put_the_red_pen_in_its_slot_in_the_cardboard_box"
TAKE_DS = "haller_take_the_red_pen_out_of_its_slot"
PUT_CHECK = "The red pen is standing in its slot in the cardboard box."
TAKE_CHECK = "The red pen is lying on the mat, outside the box."
HEAD_KEY, WRIST_KEY = "observation.images.head", "observation.images.right_wrist"  # right_wrist = robot-left arm

# Truth overrides (put check, take check) at start / end, from review.json's reject notes and checked by eye on every
# frame (2026-10-17 snapshot); None = edge case, scored apart.
PUT_REJECTS: dict[int, dict[str, tuple[bool | None, bool | None]]] = {
    28: {"end": (False, True)}, 34: {"end": (False, True)},
    # review: "ends with the pen on the mat (head frame at save)", but the episode's last video frame shows the pen
    # standing in its slot with the gripper still at it; the save-time frame is not in the dataset
    15: {"end": (None, None)}, 16: {"end": (None, None)},
    23: {"start": (True, False), "end": (False, True)},
    100: {"end": (False, None)},  # standing to the right of the green marker: not in its slot, not lying either
    43: {"end": (None, None)},  # pen in its slot but a person's hand is in the box
    61: {"end": (None, None)},  # pen in its slot but never released
}
TAKE_OVERRIDES: dict[int, dict[str, tuple[bool | None, bool | None]]] = {
    10: {"start": (False, True)},  # the take starts with the pen already lying on the mat
}


@dataclass
class Sample:
    id: str  # e.g. put_042_end
    head: np.ndarray
    wrist: np.ndarray | None
    truth: dict[str, bool | None]  # check name -> expected done (None: edge case)
    note: str = ""


# ---------------------------------------------------------------------------------------------- frames and labels
def _episodes(root: Path) -> list[dict[str, Any]]:
    import pandas as pd

    files = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.sort_values("episode_index").to_dict("records")


def _frame(root: Path, ep: dict[str, Any], key: str, which: str) -> np.ndarray:
    import cv2

    path = root / "videos" / key / (f"chunk-{int(ep[f'videos/{key}/chunk_index']):03d}/"
                                    f"file-{int(ep[f'videos/{key}/file_index']):03d}.mp4")
    t0, t1 = float(ep[f"videos/{key}/from_timestamp"]), float(ep[f"videos/{key}/to_timestamp"])
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if which == "start":
        idx = round(t0 * fps)
    elif which == "end":
        idx = max(round(t1 * fps) - 1, round(t0 * fps))
    else:  # "m50": that percentage of the episode
        idx = round((t0 + int(which[1:]) / 100 * (t1 - t0)) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot decode frame {idx} of {path}")
    return bgr[:, :, ::-1].copy()  # RGB, as the camera driver delivers it


# Put episodes whose middle frame (50 %) was checked by eye: the pen is in the gripper in the air, the slot empty. These
# are the frames a live check sees while the policy still moves, so a pass there would stop FLUX too early.
MID_EPISODES = tuple(n for n in range(104) if n % 4 in (0, 1))


def load_samples(snapshot: Path, cache: Path | None = None, wrist: bool = True, mid: bool = True) -> list[Sample]:
    cache = cache or snapshot / "_frames"
    cache.mkdir(parents=True, exist_ok=True)
    samples: list[Sample] = []
    for ds, short in ((PUT_DS, "put"), (TAKE_DS, "take")):
        root = snapshot / ds
        review = json.loads((root / "review.json").read_text())["episodes"] if (root / "review.json").exists() else {}
        for ep in _episodes(root):
            n = int(ep["episode_index"])
            moments = ("start", "end") + (("m50",) if mid and short == "put" and n in MID_EPISODES else ())
            for which in moments:
                sid = f"{short}_{n:03d}_{which}"
                f = cache / f"{sid}.npz"
                if f.exists():
                    z = np.load(f)
                    head, wr = z["head"], (z["wrist"] if "wrist" in z else None)
                else:
                    head = _frame(root, ep, HEAD_KEY, which)
                    wr = _frame(root, ep, WRIST_KEY, which) if wrist else None
                    np.savez_compressed(f, head=head, **({"wrist": wr} if wr is not None else {}))
                in_slot = (which == "end") if short == "put" else (which == "start")
                put_t, take_t = in_slot, not in_slot
                note = ""
                if which == "m50":  # carried in the gripper: not in the slot; "on the mat" is not judged here
                    put_t, take_t = False, None
                over = (PUT_REJECTS if short == "put" else TAKE_OVERRIDES).get(n, {})
                if which in over:
                    put_t, take_t = over[which]
                    note = review.get(str(n), {}).get("note", "") or "label from the frame"
                samples.append(Sample(sid, head, wr, {"put": put_t, "take": take_t}, note))
    return samples


# ---------------------------------------------------------------------------------------------- check variants
SCENE_SYSTEM = (
    "You verify one step of a robot manipulation task from camera images. The head camera looks down at a table "
    "from behind the robot. On the table: a black mat, and an open white cardboard box (its lid stands open behind "
    "it). Inside the box are a yellow utility knife lying flat, a green marker standing upright, and, just left of "
    "the green marker, the slot of the red pen: a small white holder into which the red pen is inserted upright. The "
    "red pen has a red body and a black cap. Green robot arms may be in view and may hold or hide the pen; a person "
    "may be in the background. Judge only what is visible now. Answer done=true only if the stated condition clearly "
    "holds; if the pen is hidden or the view is ambiguous, answer done=false with low confidence. Keep the "
    "observation to one sentence about where the red pen is."
)
SCENE_HINTS = {
    "put": ("Done means the red pen stands upright in its white slot next to the green marker, inside the box. "
            "Not done: the pen lies on the mat or anywhere flat, stands somewhere else in the box, or the slot is "
            "empty."),
    "take": ("Done means the red pen lies on the black mat (or the table), outside the box. Not done: the pen is in "
             "its slot or anywhere inside the box, or it is still held up in the air by the gripper."),
}


def _station_checks() -> tuple[str, dict[str, str], tuple[float, float, float, float] | None, int]:
    """The tuned checks as configured in skills.yaml, so "scene2*" measures exactly what the live verifier sends."""
    from .skills import DEFAULT_VOCABULARY, load_vocabulary

    v = load_vocabulary(DEFAULT_VOCABULARY)
    hints = {"put": v["put_pen_in_slot"].verify_hint, "take": v["take_pen_out"].verify_hint}
    return v.station.verify_scene, hints, v.station.verify_crop, v.station.verify_max_side


# v2 (tuned on the dev split): describe the scene, judge by WHERE the pen is, not by how vertical it looks
SCENE2_SYSTEM, SCENE2_HINTS, STATION_CROP, STATION_SIDE = _station_checks()


@dataclass
class Variant:
    name: str
    system: str = VERIFY_SYSTEM
    hints: dict[str, str] = field(default_factory=dict)
    crop: tuple[float, float, float, float] | None = None  # (x0, y0, x1, y1) as fractions of the head frame
    max_side: int = 640
    wrist: bool = False
    model: str = "claude-haiku-5-5"
    effort: str = "low"
    thinking: str = "adaptive"

    def content(self, check: str, sample: Sample) -> list[dict]:
        head = sample.head
        if self.crop is not None:
            head = upscale(crop_frac(head, self.crop), self.max_side)
        blocks: list[dict] = [{"type": "text", "text": "Head camera" + (" (crop of the table):" if self.crop else ":")},
                              image_block(head, self.max_side)]
        if self.wrist and sample.wrist is not None:
            blocks += [{"type": "text", "text": "Gripper camera of the arm doing the task:"},
                       image_block(sample.wrist, 640)]
        text = CHECKS[check]
        if check in self.hints:
            text += " " + self.hints[check]
        blocks.append({"type": "text", "text": f"Condition to check: {text}"})
        return blocks


CHECKS = {"put": PUT_CHECK, "take": TAKE_CHECK}
TABLE_CROP = (0.08, 0.2, 0.92, 0.82)  # mat and box in all four layouts, no person, no ceiling
VARIANTS = {
    "current": Variant("current"),  # exactly what runs today: generic prompt, full 640x480 head frame
    "scene": Variant("scene", SCENE_SYSTEM, SCENE_HINTS),
    "scene_crop": Variant("scene_crop", SCENE_SYSTEM, SCENE_HINTS, crop=TABLE_CROP, max_side=896),
    "scene_wrist": Variant("scene_wrist", SCENE_SYSTEM, SCENE_HINTS, wrist=True),
    "scene_medium": Variant("scene_medium", SCENE_SYSTEM, SCENE_HINTS, effort="medium"),
    "scene_sonnet": Variant("scene_sonnet", SCENE_SYSTEM, SCENE_HINTS, model="claude-sonnet-5-5"),
    "scene2": Variant("scene2", SCENE2_SYSTEM, SCENE2_HINTS),
    "scene2_crop": Variant("scene2_crop", SCENE2_SYSTEM, SCENE2_HINTS, crop=TABLE_CROP, max_side=896),
    "scene2_wrist": Variant("scene2_wrist", SCENE2_SYSTEM, SCENE2_HINTS, wrist=True),
    "scene2_medium": Variant("scene2_medium", SCENE2_SYSTEM, SCENE2_HINTS, effort="medium"),
    "scene2_sonnet": Variant("scene2_sonnet", SCENE2_SYSTEM, SCENE2_HINTS, model="claude-sonnet-5-5"),
    # exactly what the live ClaudeVerifier sends with the shipped skills.yaml
    "live": Variant("live", SCENE2_SYSTEM, SCENE2_HINTS, crop=STATION_CROP,
                    max_side=STATION_SIDE if STATION_CROP else 640),
}


# ---------------------------------------------------------------------------------------------- scoring
def judge(llm: ClaudeJSON, v: Variant, check: str, s: Sample, threshold: float = 0.7) -> dict[str, Any]:
    thinking = {"type": v.thinking} if v.model == "claude-haiku-5-5" else None
    start = time.perf_counter()
    try:
        data, resp = llm.call(purpose=f"verify_eval:{v.name}", model=v.model, system=v.system,
                              messages=[{"role": "user", "content": v.content(check, s)}], schema=VERIFY_SCHEMA,
                              effort=v.effort, max_tokens=4000, cache_system=False, thinking=thinking)
    except Exception as exc:  # scored as an error, never as a verdict
        return {"id": s.id, "check": check, "error": f"{type(exc).__name__}: {str(exc)[:160]}",
                "latency_s": time.perf_counter() - start}
    conf = float(np.clip(float(data.get("confidence", 0.0)), 0.0, 1.0))
    usage = getattr(resp, "usage", None)
    return {"id": s.id, "check": check, "truth": s.truth[check], "done": bool(data.get("done")), "confidence": conf,
            "pass": bool(data.get("done")) and conf >= threshold, "observation": str(data.get("observation", "")),
            "latency_s": time.perf_counter() - start, "cost_usd": cost_of(getattr(resp, "model", v.model), usage),
            "input_tokens": getattr(usage, "input_tokens", 0), "output_tokens": getattr(usage, "output_tokens", 0)}


def score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    graded = [r for r in rows if "pass" in r and r["truth"] is not None]
    tp = sum(r["pass"] and r["truth"] for r in graded)
    fp = sum(r["pass"] and not r["truth"] for r in graded)
    tn = sum(not r["pass"] and not r["truth"] for r in graded)
    fn = sum(not r["pass"] and r["truth"] for r in graded)
    lat = [r["latency_s"] for r in rows if "pass" in r]
    cost = [r["cost_usd"] for r in rows if "pass" in r]
    return {
        "n": len(graded), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": round((tp + tn) / len(graded), 4) if graded else None,
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "recall": round(tp / (tp + fn), 4) if tp + fn else None,
        "errors": sum("error" in r for r in rows),
        "latency_s": {"median": round(statistics.median(lat), 2), "p90": round(float(np.percentile(lat, 90)), 2),
                      "max": round(max(lat), 2)} if lat else None,
        "usd_per_check": round(statistics.mean(cost), 6) if cost else None,
        "wrong": [{k: r[k] for k in ("id", "truth", "done", "confidence", "observation")}
                  for r in graded if r["pass"] != r["truth"]],
        "edge": [{k: r[k] for k in ("id", "done", "confidence", "observation")}
                 for r in rows if "pass" in r and r["truth"] is None],
    }


def sweep(rows: list[dict[str, Any]], thresholds=(0.5, 0.6, 0.7, 0.8)) -> dict[str, dict[str, Any]]:
    """Confusion counts per confidence threshold, overall and for the dev (even) / test (odd) episode split."""
    out: dict[str, dict[str, Any]] = {}
    for split in ("all", "dev", "test"):
        part = [r for r in rows if "done" in r and r["truth"] is not None
                and (split == "all" or (int(r["id"].split("_")[1]) % 2 == 0) == (split == "dev"))]
        for t in thresholds:
            p = [r["done"] and r["confidence"] >= t for r in part]
            tp = sum(x and r["truth"] for x, r in zip(p, part, strict=True))
            fp = sum(x and not r["truth"] for x, r in zip(p, part, strict=True))
            fn = sum((not x) and r["truth"] for x, r in zip(p, part, strict=True))
            tn = len(part) - tp - fp - fn
            out.setdefault(split, {})[f"{t:.1f}"] = {"n": len(part), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                                                     "accuracy": round((tp + tn) / len(part), 4) if part else None}
    return out


def evaluate(samples: list[Sample], variants: list[str], checks: list[str], llm: ClaudeJSON, workers: int = 8,
             threshold: float = 0.7) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in variants:
        v = VARIANTS[name]
        jobs = [(c, s) for c in checks for s in samples if not (s.id.endswith("m50") and s.truth[c] is None)]
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(lambda job, v=v: judge(llm, v, job[0], job[1], threshold), jobs))
        out[name] = {"variant": {k: getattr(v, k) for k in ("model", "effort", "thinking", "crop", "max_side",
                                                             "wrist")} | {"hints": bool(v.hints)},
                     "wall_s": round(time.time() - t0, 1),
                     "checks": {c: score([r for r in rows if r["check"] == c]) for c in checks},
                     "sweep": {c: sweep([r for r in rows if r["check"] == c]) for c in checks},
                     "rows": rows}
        for c in checks:
            m = out[name]["checks"][c]
            print(f"{name:<14} {c:<5} n={m['n']:<4} acc={m['accuracy']}  tp={m['tp']} fp={m['fp']} tn={m['tn']} "
                  f"fn={m['fn']} err={m['errors']}  median {m['latency_s']['median'] if m['latency_s'] else '-'} s  "
                  f"USD {m['usd_per_check']}/check", flush=True)
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--snapshot", required=True, type=Path)
    p.add_argument("--variants", default="current,scene")
    p.add_argument("--checks", default="put,take")
    p.add_argument("--only", default="", help="comma-separated sample ids or prefixes (e.g. put_0,take_)")
    p.add_argument("--every", type=int, default=1, help="keep every n-th episode (plus all rejects)")
    p.add_argument("--split", choices=("all", "dev", "test"), default="all",
                   help="dev = even episode indices (prompt tuning), test = odd ones (held out)")
    p.add_argument("--budget", type=float, default=1.0, help="USD cap for this evaluation")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)
    samples = load_samples(a.snapshot)
    if a.every > 1:
        samples = [s for s in samples if int(s.id.split("_")[1]) % a.every == 0
                   or (s.id.startswith("put") and int(s.id.split("_")[1]) in PUT_REJECTS) or s.id.startswith("take")]
    if a.split != "all":
        samples = [s for s in samples if (int(s.id.split("_")[1]) % 2 == 0) == (a.split == "dev")]
    if a.only:
        keys = a.only.split(",")
        samples = [s for s in samples if any(s.id.startswith(k) for k in keys)]
    meter = CostMeter(a.budget)
    report = {"time": datetime.now().isoformat(timespec="seconds"), "snapshot": str(a.snapshot),
              "samples": len(samples), "threshold": 0.7,
              "results": evaluate(samples, a.variants.split(","), a.checks.split(","), ClaudeJSON(meter),
                                  a.workers)}
    report["total_usd"] = round(meter.total_usd, 4)
    out = a.out or PACKAGE_DIR / "runs" / f"verify_eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"{len(samples)} samples, total USD {report['total_usd']:.4f} -> {out}")


if __name__ == "__main__":
    main()
