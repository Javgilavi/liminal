"""Demonstration-collection protocol for task-specific LoRA (owned by WS1).

The point of a protocol is *diversity on purpose*: 50-100 episodes recorded from the
same start pose teach the policy one trajectory. Each episode gets a variation drawn
from a fixed grid so coverage is deliberate and logged.
"""

from __future__ import annotations

import itertools
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class EpisodePlan:
    index: int
    object_zone: str
    object_yaw_deg: int
    distractors: int
    lighting: str
    instruction: str


def make_plan(n_episodes: int, instruction_variants: list[str], zones: list[str],
              yaws: tuple[int, ...] = (0, 45, 90, 135), distractors: tuple[int, ...] = (0, 1, 2),
              lighting: tuple[str, ...] = ("normal", "dim"), seed: int = 0) -> list[EpisodePlan]:
    grid = list(itertools.product(zones, yaws, distractors, lighting))
    rng = random.Random(seed)
    rng.shuffle(grid)
    plans = []
    for i in range(n_episodes):
        z, y, d, light = grid[i % len(grid)]
        plans.append(EpisodePlan(i, z, y, d, light, instruction_variants[i % len(instruction_variants)]))
    return plans


def write_plan(plans: list[EpisodePlan], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(asdict(p)) for p in plans) + "\n")
