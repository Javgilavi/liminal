"""Auto-label stage boundaries of a long episode with one Claude vision call (for data without our own step log).

About 20 head-camera frames spread over the episode, the plan (one line per step), and a structured-output schema:
Claude says, for each frame, which plan step is being executed (or none) and how many steps are already done.
Contiguous runs become ``labels.Segment``s (boundaries at the midpoint between sampled frames) and the result is
saved as ``episode_claude.json`` (``source: "claude_vision"``) next to the episode. On sim episodes, which also
carry ground truth, ``--eval`` reports frame accuracy and boundary error against it, to measure the labeller
before trusting it on public or real data.

    python ws4/ttt/claude_stage_labels.py ws4/.cache/ttt/sim/pack_lidded-100007 --eval
    python ws4/ttt/claude_stage_labels.py <episode dirs...> --frames 20 --model claude-opus-5-5 --effort low

API usage follows the Claude API docs bundled with the claude-api skill (Python SDK ``anthropic`` 1.x):
structured outputs via ``output_config.format`` with a JSON schema, effort via ``output_config.effort``, refusal
handling via ``stop_reason``, and the server-side ``fallbacks: "default"`` opt-in (beta
``server-side-fallback-2026-07-01``) recommended for ``claude-opus-5-5``. The key is read from
``~/.config/ws4/anthropic.env`` (``ANTHROPIC_API_KEY=...``) unless already in the environment; it is never printed.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ttt.labels import EpisodeLabels, Segment, replan_targets  # noqa: E402

KEY_FILE = Path.home() / ".config" / "ws4" / "anthropic.env"
PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50)}

SKILL_TEXT = {
    "open_lid": "take the lid off the box and put it down on the table",
    "close_lid": "put the lid back onto the box",
    "put_apple_in_box": "pick up the red apple and drop it into the open box",
    "put_block_in_box": "pick up the yellow block and drop it into the open box",
}

SCHEMA = {
    "type": "object",
    "properties": {
        "frames": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "frame": {"type": "integer"},
                    "step": {"type": "integer"},
                    "steps_done": {"type": "integer"},
                },
                "required": ["frame", "step", "steps_done"],
                "additionalProperties": False,
            },
        },
        "notes": {"type": "string"},
    },
    "required": ["frames", "notes"],
    "additionalProperties": False,
}


def load_key() -> None:
    if os.environ.get("ANTHROPIC_API_KEY") or not KEY_FILE.exists():
        return
    for line in KEY_FILE.read_text().splitlines():
        k, sep, v = line.strip().partition("=")
        if sep and k.removeprefix("export ").strip() == "ANTHROPIC_API_KEY":
            os.environ["ANTHROPIC_API_KEY"] = v.strip().strip('"').strip("'")


def sample_frames(ep_dir: Path, n: int) -> tuple[list[int], list[bytes]]:
    """``n`` re-plan ticks spread evenly over the episode, with their head-camera JPEG bytes."""
    from ttt.sim_episodes import load_frames

    labels = EpisodeLabels.load(ep_dir / "episode.json")
    ticks = labels.replan_ticks
    idx = np.unique(np.linspace(0, len(ticks) - 1, min(n, len(ticks))).round().astype(int))
    frame_ticks, frames = load_frames(ep_dir)
    where = {int(t): i for i, t in enumerate(frame_ticks)}
    chosen = [ticks[i] for i in idx]
    return chosen, [bytes(frames["head"][where[t]]) for t in chosen]


def prompt_text(steps: list[dict], ticks: list[int], fps: float) -> str:
    plan = "\n".join(
        f"  step {i}: {s['skill']} ({s['arm']} arm): {SKILL_TEXT.get(s['skill'], s['skill'])}"
        for i, s in enumerate(steps)
    )
    when = ", ".join(f"frame {i} at {t / fps:.1f} s" for i, t in enumerate(ticks))
    return (
        "These images are frames from one recording of a two-armed robot, in time order, seen from a camera above "
        "and behind the arms. The robot carries out this plan, one step at a time, in this order:\n"
        f"{plan}\n"
        f"Frame times: {when}.\n"
        "For every frame, give `step`: the index of the plan step the robot is executing in that frame (-1 if it "
        "is not executing any step, for example before the first step or after the last one), and `steps_done`: "
        "how many plan steps are already complete. Steps that repeat look alike; use the order of the frames and "
        "what changed in the scene to tell repetitions apart. Put any doubt in `notes`."
    )


def label_episode(client, ep_dir: Path, n_frames: int, model: str, effort: str) -> dict:
    labels = EpisodeLabels.load(ep_dir / "episode.json")
    ticks, jpegs = sample_frames(ep_dir, n_frames)
    content: list[dict] = []
    for i, data in enumerate(jpegs):
        content.append({"type": "text", "text": f"frame {i}"})
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.standard_b64encode(data).decode(),
                },
            }
        )
    content.append({"type": "text", "text": prompt_text(labels.steps, ticks, labels.fps)})
    kwargs = {}
    if model in ("claude-opus-5-5", "claude-sonnet-5-5"):
        kwargs = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
    create = client.beta.messages.create if kwargs else client.messages.create
    response = create(
        model=model,
        max_tokens=16000,
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": SCHEMA}},
        messages=[{"role": "user", "content": content}],
        **kwargs,
    )
    if response.stop_reason == "refusal":
        raise RuntimeError(f"{ep_dir.name}: refused ({getattr(response, 'stop_details', None)})")
    text = next(b.text for b in response.content if b.type == "text")
    out = json.loads(text)
    pred = {f["frame"]: f for f in out["frames"]}
    steps = [pred.get(i, {"step": -1})["step"] for i in range(len(ticks))]
    usage = response.usage
    pin, pout = PRICES.get(model, (float("nan"), float("nan")))
    cost = (usage.input_tokens * pin + usage.output_tokens * pout) / 1e6
    return {
        "ticks": ticks,
        "steps": steps,
        "raw": out,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cost_usd": cost,
        "model": response.model,
    }


def to_labels(ep_dir: Path, result: dict) -> EpisodeLabels:
    """Contiguous runs of the same predicted step -> segments, boundaries halfway between sampled frames."""
    gt = EpisodeLabels.load(ep_dir / "episode.json")
    ticks, steps = result["ticks"], result["steps"]
    edges = [0] + [(a + b) // 2 for a, b in zip(ticks[:-1], ticks[1:], strict=True)] + [gt.n_ticks]
    segments, i = [], 0
    while i < len(steps):
        j = i
        while j + 1 < len(steps) and steps[j + 1] == steps[i]:
            j += 1
        k = steps[i]
        if 0 <= k < len(gt.steps) and edges[i] < edges[j + 1]:
            st = gt.steps[k]
            if not segments or segments[-1].step_id != st["id"]:
                segments.append(Segment(st["id"], st["skill"], st["arm"], edges[i], edges[j + 1], None, 1))
        i = j + 1
    # A step that reappears later becomes a second attempt; keep only non-overlapping ones per arm.
    clean, last_end = [], {}
    for s in segments:
        if s.start_tick >= last_end.get(s.arm, 0):
            clean.append(s)
            last_end[s.arm] = s.end_tick
    return EpisodeLabels(
        gt.episode_id,
        "claude_vision",
        gt.fps,
        gt.n_ticks,
        gt.task,
        gt.steps,
        clean,
        list(gt.replan_ticks),
        extra={"claude": {k: v for k, v in result.items() if k != "raw"}, "notes": result["raw"].get("notes", "")},
    )


def evaluate(ep_dir: Path, result: dict, auto: EpisodeLabels) -> dict:
    gt = EpisodeLabels.load(ep_dir / "episode.json")
    tg = replan_targets(gt, result["ticks"])
    frame_acc = float(np.mean(np.asarray(result["steps"]) == tg.stage))
    # Re-plan-level agreement of the derived segments with ground truth over the whole episode.
    a, g = replan_targets(auto).stage, replan_targets(gt).stage
    errs = []
    for seg in gt.segments:
        match = [s for s in auto.segments if s.step_id == seg.step_id]
        if match:
            errs.append(abs(match[0].start_tick - seg.start_tick) / gt.fps)
    return {
        "frame_acc": frame_acc,
        "replan_stage_acc": float(np.mean(a == g)),
        "start_error_s_median": float(np.median(errs)) if errs else float("nan"),
        "steps_found": f"{len({s.step_id for s in auto.segments})}/{len(gt.steps)}",
    }


def main(argv: list[str] | None = None) -> None:
    from ttt.throttle import throttle

    throttle()  # shares the laptop with 30 Hz robot loops
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("episodes", type=Path, nargs="+")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--effort", default="low", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--eval", action="store_true", help="compare with the episode's own ground truth (sim)")
    ap.add_argument("--out", type=Path, default=None, help="append one JSON line per episode here")
    args = ap.parse_args(argv)
    import anthropic

    load_key()
    client = anthropic.Anthropic()
    total = 0.0
    for ep in args.episodes:
        result = label_episode(client, ep, args.frames, args.model, args.effort)
        auto = to_labels(ep, result)
        auto.save(ep / "episode_claude.json")
        rec = {
            "episode": ep.name,
            "model": result["model"],
            "frames": len(result["ticks"]),
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "cost_usd": round(result["cost_usd"], 4),
        }
        if args.eval:
            rec.update(evaluate(ep, result, auto))
        total += result["cost_usd"]
        print(json.dumps(rec), flush=True)
        if args.out:
            with args.out.open("a") as f:
                f.write(json.dumps(rec) + "\n")
    print(f"total cost about ${total:.3f}")


if __name__ == "__main__":
    main()
