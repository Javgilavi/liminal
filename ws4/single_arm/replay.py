"""Scripted replay of one recorded take as the policy (``run_single_arm.py --backend replay``). It is NOT FLUX.

Fallback for the chain's box skill: the FLUX box LoRA went 0/4 on the arm (18 Oct, 08:20: it crawled from rest, the
wrist rolled about 5 deg where the demos roll about 50, and the tcp_env keep-out stopped it; offline from the
recorded frame 0 it barely moves either). The take's actions for the task arm are handed to the runner as policy
chunks, so every command passes exactly what a FLUX command passes: the guard (max step per tick, soft limits), the
keep-out with the skill's demonstrated envelope, the stall, keep-out and load freezes, the torque cap and the
runner's start checks.

The VR takes move up to 4 to 11 deg per frame, more than the 1 deg/tick guard lets through, so the take is
time-warped onto that limit first: every recorded step becomes as many ticks as its largest move needs
(``max_step`` deg on the arm joints, ``max_gripper_step`` on the gripper), linear in joint space, starting from the
arm's pose when the replay starts. The arm follows the recorded path itself, only slower where the operator was
faster; after the last recorded action the chunks repeat it, so the arm holds there (a box take ends holding the lid
open).

Near the other arm (a skill with a slow zone: the cutter by the lid-holding box arm) the guard would clip the arm to
``slow_step`` per tick and the arm would cut corners behind the replayed path; ``step_at(q)`` gives the warp a
smaller step there (the runner and the orchestrator pass the slow step within the slow zone plus ``SLOW_MARGIN`` of
the other arm's live pose), so the path stays the recorded one.

Prime mode (``prime_ticks``, run_single_arm ``--prime-ticks K``): the first K ticks of the replay, then the FLUX
backend takes over from there with the replayed history, for trying a primed FLUX box start.

    python ws4/single_arm/replay.py 59                  # a box take: length, start, end, duration at 1 deg/tick
    python ws4/single_arm/replay.py --skill cutter      # the cutter's default take
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

from flux.backends import PolicyResponse  # noqa: E402
from flux.contract import SO101_CONTRACT, arm_slice  # noqa: E402

DEFAULT_ROOT = Path.home() / ".cache/huggingface/lerobot/ws4/haller_open_the_box"
# Take 59 (P3, the chain's box position; train split; review: keep, "ends with the box arm holding the lid open"):
# of the 32 good P3 takes its first frame is nearest the chain's box home [0, 0.8, 2.0, 0.4, 0, 99.7] (1.6 deg), its
# end pose (last second) has the widest margins in the pen skills' start gate (14 deg inside the joint range; demo
# gaps 4.8 / 1.7 / 8.0 cm for green / cutter / pen_lid), it moves least in its last second (2.1 deg: settled,
# holding), and time-warped from box home it passes the box workspace with its envelope with 0 refusals in 549
# ticks (18 s at 1 deg/tick). Checked offline 18 Oct against skills/box, green, cutter and pen_lid.
DEFAULT_EPISODE = 59
# Per skill (skills.py names): the default take. Cutter 67 (P3, held-out: its first head frame is
# skills/cutter/layouts/P3_ep067_head.jpg, the placement reference): of the three takes with a reference it keeps the
# most room to the lid-holding box arm (2.9 cm min over its path with the box arm where box take 59 ends, 3.2 cm at
# the cutter takes' recorded hold; clearance 1 cm) and spends the fewest ticks in the 4 cm slow zone (33); it starts
# at home with the green pen in its slot (review: "pen arm home -> home, box arm holding the lid, green pen in its
# slot") and ends exactly at home, with 0 refusals by the cutter workspace from home (757 ticks at 1 deg/tick).
DEFAULT_TAKES = {"box": DEFAULT_EPISODE, "cutter": 67}
SLOW_MARGIN = 0.01  # m beyond a slow zone where the warp already slows down (measured lag, live vs recorded poses)


def take_source(skill: str) -> tuple[Path, int, str]:
    """(dataset root in the local cache, default take, the task arm's dataset columns) of a skills.py skill."""
    from skills import ARMS, CACHE, SKILLS

    s = SKILLS[skill]
    return CACHE / s.dataset, DEFAULT_TAKES[skill], ARMS[s.arm].columns


def reference_frame(ref_dir: str | Path, episode: int) -> Path | None:
    """The take's own first head frame in a layout reference folder (``<layout>_ep<NNN>_head.jpg``), if there."""
    hits = sorted(Path(ref_dir).glob(f"*_ep{episode:03d}_head.*"))
    return hits[-1] if hits else None


def label(episode: int, prime_ticks: int | None = None) -> str:
    text = f"scripted replay of recorded demo {episode}"
    return f"{text} for the first {prime_ticks} ticks, then FLUX" if prime_ticks else text


def load_take(root: str | Path, episode: int, columns: str) -> tuple[np.ndarray, dict[str, Any]]:
    """(T, 6) actions of one take for one arm (``columns``: ``left`` / ``right``, the dataset's haller naming: the
    robot-RIGHT box arm is ``left_*``), read straight from the LeRobot v3 parquet files (no videos, no lerobot)."""
    import pyarrow.parquet as pq

    root = Path(root).expanduser()
    files = sorted(root.glob("data/chunk-*/*.parquet"))
    if not files:
        raise SystemExit(f"replay: no parquet files under {root}/data")
    table = pq.ParquetDataset(files, filters=[("episode_index", "=", int(episode))]).read(
        columns=["episode_index", "frame_index", "action"])
    if table.num_rows == 0:
        raise SystemExit(f"replay: episode {episode} is not in {root}")
    order = np.argsort(np.asarray(table["frame_index"]))
    actions = np.stack(table["action"].to_numpy(zero_copy_only=False)).astype(float)[order]
    if actions.shape[1] != 12:
        raise SystemExit(f"replay: {root} has {actions.shape[1]}-D actions, expected the Haller's 12-D [left | right]")
    info = {"episode": int(episode), "root": str(root), "columns": columns, "frames": len(actions)}
    return actions[:, arm_slice(columns)], info


def time_warp(actions: np.ndarray, start: np.ndarray, max_step: float = 1.0, max_gripper_step: float = 5.0,
              step_at: Any = None) -> np.ndarray:  # fmt: skip
    """The path start -> actions[0] -> ... -> actions[-1], linear in joint space, cut into ticks of at most
    ``max_step`` deg per arm joint (``step_at(q)`` at a point, when given: the smaller of a segment's two ends) and
    ``max_gripper_step`` on the gripper."""
    points = np.vstack([np.asarray(start, float)[None], np.asarray(actions, float)])
    steps = np.full(len(points), float(max_step))
    if step_at is not None:
        steps = np.minimum(steps, [float(step_at(q)) for q in points])
    out = []
    for i, (a, b) in enumerate(zip(points[:-1], points[1:], strict=True)):
        step = min(steps[i], steps[i + 1])
        n = int(np.ceil(max(np.abs(b[:5] - a[:5]).max() / step, abs(b[5] - a[5]) / max_gripper_step, 1.0)))
        out.extend(a + (b - a) * k / n for k in range(1, n + 1))
    return np.asarray(out)


class ReplayBackend:
    """``FluxBackend`` interface over one take. ``reset()`` at the start of every attempt (the orchestrator calls
    ``begin``); the sequence is built from the first request's command anchor (the measured pose then). ``done``
    turns True at the first request after every replayed command was executed (the runner only asks again once its
    queue is empty), so the attempt can end there; prefetching would break that count and is refused by the callers.
    With ``prime`` (a FLUX backend) and ``prime_ticks`` K, requests after the first K ticks go to FLUX instead."""

    verify_when_finished = True  # orchestrator: check once the take has been replayed to its end, not mid-take

    def __init__(self, actions: np.ndarray, info: dict[str, Any], max_step: float = 1.0,
                 max_gripper_step: float = 5.0, contract: Any = SO101_CONTRACT, prime: Any = None,
                 prime_ticks: int | None = None):  # fmt: skip
        self.actions, self.info = np.asarray(actions, dtype=float), dict(info)
        self.max_step, self.max_gripper_step = float(max_step), float(max_gripper_step)
        self.contract = prime.contract if prime is not None else contract
        self.prime, self.prime_ticks = prime, prime_ticks
        self.label = label(self.info["episode"], prime_ticks if prime is not None else None)
        self.step_at: Any = None  # q -> deg per tick there (the slow zone), evaluated when the take is warped
        self.requests = 0
        self.reset()

    def reset(self) -> None:
        self.seq: np.ndarray | None = None
        self.cursor, self.done, self.handed_over = 0, False, False

    def begin(self, target: str = "", arm: str = "", skill: Any = None) -> None:
        self.reset()

    def end(self, target: str = "") -> None:
        pass

    def predict(self, request: Any) -> PolicyResponse:
        if request.target == "warmup":  # never advances the replay
            if self.prime is not None:
                return self.prime.predict(request)
            return PolicyResponse(np.repeat(np.asarray(request.commands[-1], float)[None], self.contract.chunk_size, 0),
                                  0.0, {"backend": "replay", "label": self.label})  # fmt: skip
        self.requests += 1
        if self.seq is None:
            self.seq = time_warp(self.actions, request.commands[-1], self.max_step, self.max_gripper_step,
                                 self.step_at)  # fmt: skip
            self.info["ticks"] = len(self.seq)
        if self.prime is not None and self.cursor >= (self.prime_ticks or 0):
            self.handed_over = True
            return self.prime.predict(request)
        if self.cursor >= len(self.seq):
            self.done = True
        n = self.contract.chunk_size
        chunk = self.seq[self.cursor: self.cursor + n]
        if len(chunk) < n:
            chunk = np.vstack([chunk, np.repeat(self.seq[-1:], n - len(chunk), axis=0)])
        index = self.cursor
        self.cursor += self.contract.n_action_steps
        return PolicyResponse(chunk, 0.0, {"backend": "replay", "label": self.label, "replay_tick": index,
                                           "replay_ticks": len(self.seq)})  # fmt: skip

    # orchestrator executor hooks (the sim backend's names)
    def finished(self, target: str = "", state6: Any = None) -> bool:
        return self.done

    def progress(self, target: str = "") -> tuple[int, int, str]:
        n = 0 if self.seq is None else len(self.seq)
        i = min(self.cursor, n)
        if self.handed_over:
            return i, n, f"{self.label}: FLUX since tick {self.prime_ticks}"
        return i, n, f"{self.label}: tick {i}/{n}" + (" (holding at its end)" if self.done else "")


def main(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="summary of a take as the replay would run it")
    p.add_argument("episode", type=int, nargs="?", default=None, help="default: the skill's default take")
    p.add_argument("--skill", default="box", choices=sorted(DEFAULT_TAKES))
    p.add_argument("--root", default=None, help="default: the skill's dataset in the local cache")
    p.add_argument("--columns", default=None, help="dataset columns of the arm (left = the robot-right box arm)")
    p.add_argument("--start", type=float, nargs=6, default=None, help="default: the skill arm's home")
    a = p.parse_args(argv)
    root, episode, columns = take_source(a.skill)
    start = a.start or ([0, 0.8, 2.0, 0.4, 0, 99.7] if a.skill == "box" else [0, 0, 0, 0, 0, 100])
    actions, info = load_take(a.root or root, a.episode if a.episode is not None else episode, a.columns or columns)
    a.start, a.episode = start, info["episode"]
    seq = time_warp(actions, np.array(a.start))
    print(f"{label(a.episode)}: {info['frames']} frames ({info['frames'] / 30:.1f} s recorded), {len(seq)} ticks at "
          f"1 deg/tick ({len(seq) / 30:.1f} s); first {np.round(actions[0], 1).tolist()}, last "
          f"{np.round(actions[-1], 1).tolist()}")  # fmt: skip


if __name__ == "__main__":
    main()
