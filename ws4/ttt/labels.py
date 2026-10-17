"""Shared stage-label format for long-horizon episodes (sim ground truth, orchestrator logs, Claude-vision labels).

One episode = one directory with ``episode.json`` (this schema) plus arrays (``ticks.npz``, ``frames.npz``; see
``sim_episodes.py``). The label part is deliberately small so that every source can produce it:

```json
{
  "schema": "ws4.stage_labels/1",
  "episode_id": "sim-pack_lidded-0007",
  "source": "sim_gt",                  # sim_gt | orchestrator_log | claude_vision | manual
  "fps": 30,
  "n_ticks": 1532,
  "task": "pack_lidded",
  "steps": [                           # the plan, in execution order (orchestrator Step ids / skills / arms)
    {"id": "s1", "skill": "open_lid", "arm": "left"}, ...
  ],
  "segments": [                        # what actually ran: one segment per step attempt, ticks are [start, end)
    {"step_id": "s1", "skill": "open_lid", "arm": "left", "start_tick": 0, "end_tick": 262,
     "success": true, "attempt": 1}, ...
  ],
  "replan_ticks": [0, 32, 64, ...]     # ticks where FLUX (or the stand-in) planned a chunk
}
```

The progress model's sequence unit is one re-plan. :func:`replan_targets` turns segments into per-re-plan targets:
stage index (position of the segment's step in ``steps``), progress inside the stage, overall progress and a
"stage done" flag (the stage ends before the next re-plan).

Orchestrator logs map onto this with :func:`segments_from_events`: one event per step start and end, e.g.
``{"tick": 412, "step_id": "s2", "skill": "put_apple_in_box", "arm": "left", "event": "end", "success": true}``
(``"t"`` in seconds instead of ``"tick"`` also works).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

SCHEMA = "ws4.stage_labels/1"
SOURCES = ("sim_gt", "orchestrator_log", "claude_vision", "manual")


@dataclass
class Segment:
    step_id: str
    skill: str
    arm: str
    start_tick: int
    end_tick: int
    success: bool | None = None
    attempt: int = 1


@dataclass
class EpisodeLabels:
    episode_id: str
    source: str
    fps: float
    n_ticks: int
    task: str
    steps: list[dict]
    segments: list[Segment]
    replan_ticks: list[int] = field(default_factory=list)
    extra: dict = field(default_factory=dict)
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}, got {self.source!r}")
        self.segments = [s if isinstance(s, Segment) else Segment(**s) for s in self.segments]
        ids = [s["id"] for s in self.steps]
        if len(set(ids)) != len(ids):
            raise ValueError("step ids must be unique")
        # Steps on the two arms may run at the same time (orchestrator plans), one step per arm at a time.
        prev_end: dict[str, int] = {}
        for seg in self.segments:
            if seg.step_id not in ids:
                raise ValueError(f"segment refers to unknown step {seg.step_id!r}")
            if not 0 <= seg.start_tick < seg.end_tick <= self.n_ticks:
                raise ValueError(f"bad segment ticks {seg.start_tick}..{seg.end_tick} (n_ticks {self.n_ticks})")
            if seg.start_tick < prev_end.get(seg.arm, 0):
                raise ValueError(f"segments of the {seg.arm} arm must be in time order and must not overlap")
            prev_end[seg.arm] = seg.end_tick

    @property
    def n_stages(self) -> int:
        return len(self.steps)

    def stage_index(self, step_id: str) -> int:
        return [s["id"] for s in self.steps].index(step_id)

    def to_json(self) -> dict:
        d = asdict(self)
        d["segments"] = [asdict(s) for s in self.segments]
        return d

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_json(), indent=1))

    @classmethod
    def load(cls, path: str | Path) -> EpisodeLabels:
        d = json.loads(Path(path).read_text())
        if d.get("schema") != SCHEMA:
            raise ValueError(f"{path}: schema {d.get('schema')!r}, expected {SCHEMA}")
        return cls(**d)


def tick_stage(labels: EpisodeLabels) -> np.ndarray:
    """``(n_ticks,)`` stage index per tick; -1 where no segment runs (gaps between steps)."""
    out = np.full(labels.n_ticks, -1, dtype=np.int64)
    for seg in labels.segments:
        out[seg.start_tick : seg.end_tick] = labels.stage_index(seg.step_id)
    return out


@dataclass
class ReplanTargets:
    ticks: np.ndarray  # (R,) re-plan ticks
    stage: np.ndarray  # (R,) stage index, -1 = between steps (masked in training)
    stage_progress: np.ndarray  # (R,) 0..1 inside the stage
    progress: np.ndarray  # (R,) 0..1 over the whole plan
    done: np.ndarray  # (R,) 1 if the stage ends before the next re-plan
    occurrence: np.ndarray  # (R,) how many earlier segments ran the same skill (0 = first time): aliasing


def replan_targets(
    labels: EpisodeLabels, ticks: list[int] | np.ndarray | None = None, arm: str | None = None
) -> ReplanTargets:
    """Per-re-plan targets. ``arm`` restricts to that arm's segments (one monitor per arm when steps overlap);
    otherwise, where segments of both arms overlap, the one that started last defines the stage."""
    ticks = np.asarray(labels.replan_ticks if ticks is None else ticks, dtype=np.int64)
    segments = [s for s in labels.segments if arm is None or s.arm == arm]
    n = labels.n_stages
    stage = np.full(len(ticks), -1, dtype=np.int64)
    sprog = np.zeros(len(ticks))
    prog = np.zeros(len(ticks))
    done = np.zeros(len(ticks), dtype=np.int64)
    occ = np.zeros(len(ticks), dtype=np.int64)
    seen: dict[str, int] = {}
    seg_occ = []
    for seg in segments:
        seg_occ.append(seen.get(seg.skill, 0))
        seen[seg.skill] = seen.get(seg.skill, 0) + 1
    for i, t in enumerate(ticks):
        nxt = ticks[i + 1] if i + 1 < len(ticks) else labels.n_ticks
        active = [(seg, o) for seg, o in zip(segments, seg_occ, strict=True) if seg.start_tick <= t < seg.end_tick]
        if active:
            seg, o = max(active, key=lambda so: so[0].start_tick)
            k = labels.stage_index(seg.step_id)
            stage[i] = k
            sprog[i] = (t - seg.start_tick) / max(1, seg.end_tick - seg.start_tick)
            prog[i] = (k + sprog[i]) / n
            done[i] = int(seg.end_tick <= nxt)
            occ[i] = o
        else:  # between steps: progress = completed stages so far
            finished = [labels.stage_index(s.step_id) for s in segments if s.end_tick <= t]
            prog[i] = (max(finished) + 1) / n if finished else 0.0
    return ReplanTargets(ticks, stage, sprog, prog, done, occ)


def segments_from_events(events: list[dict], fps: float = 30.0) -> list[Segment]:
    """Orchestrator step log (start/end events per step attempt) -> segments in ticks."""

    def tick(e: dict) -> int:
        return int(e["tick"]) if "tick" in e else int(round(float(e["t"]) * fps))

    open_: dict[str, dict] = {}
    attempts: dict[str, int] = {}
    out: list[Segment] = []
    for e in sorted(events, key=tick):
        sid = e["step_id"]
        if e["event"] == "start":
            open_[sid] = e
        elif e["event"] == "end":
            start = open_.pop(sid, None)
            if start is None:
                raise ValueError(f"end of step {sid!r} without a start")
            attempts[sid] = attempts.get(sid, 0) + 1
            out.append(
                Segment(
                    step_id=sid,
                    skill=e.get("skill", start.get("skill", "")),
                    arm=e.get("arm", start.get("arm", "")),
                    start_tick=tick(start),
                    end_tick=max(tick(e), tick(start) + 1),
                    success=e.get("success"),
                    attempt=attempts[sid],
                )
            )
        else:
            raise ValueError(f"unknown event {e['event']!r}")
    if open_:
        raise ValueError(f"steps never ended: {sorted(open_)}")
    return out
