"""Per-target observation and command history, kept on the robot side.

FLUX 3 Action SO-101 conditions on the last ``n_obs_steps`` measured states, the command issued before each of
them, and image snapshots of a few of those ticks. LeRobot keeps that history inside its saved preprocessor
(lerobot@146d847:src/lerobot/policies/flux3/processor_flux3.py:354-401) and its policy keeps one action queue
(modeling_flux3.py:240-241, 610-623), so one processor pair serves exactly one target. Keeping the history here
instead lets one loaded model serve both arms and lets the model run on a remote GPU: each request carries its
own window and the server calls the policy statelessly with ``observation.command_history`` (offline path,
processor_flux3.py:388-394; documented in lerobot@146d847:docs/source/flux3.mdx:226-230).

Semantics copied from the synchronous processor:
- the first observation is repeated to fill the window (processor_flux3.py:380-383);
- before any command was issued, the "previous command" is the measured state (processor_flux3.py:370-371);
- each tick appends the command issued before that tick's observation (processor_flux3.py:382-383);
- the command anchor for delta integration is the last of those (processor_flux3.py:394, 464-470).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class HistoryWindow:
    states: np.ndarray  # (n, D) float64, model units
    commands: np.ndarray  # (n, D) float64, command issued before each observation
    images: dict[str, np.ndarray]  # model camera key -> (n, H, W, 3) uint8


class HistoryBuffer:
    def __init__(self, n_obs_steps: int, action_dim: int):
        if n_obs_steps < 1 or action_dim < 1:
            raise ValueError("n_obs_steps and action_dim must be positive")
        self.n_obs_steps = n_obs_steps
        self.action_dim = action_dim
        self.reset()

    def reset(self) -> None:
        self._states: deque[np.ndarray] = deque(maxlen=self.n_obs_steps)
        self._commands: deque[np.ndarray] = deque(maxlen=self.n_obs_steps)
        self._images: deque[dict[str, np.ndarray]] = deque(maxlen=self.n_obs_steps)
        self.last_command: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self._states)

    def push(self, state: np.ndarray, images: dict[str, np.ndarray]) -> None:
        state = np.asarray(state, dtype=np.float64).reshape(-1)
        if state.shape != (self.action_dim,):
            raise ValueError(f"expected a {self.action_dim}-D state, got {state.shape}")
        if self.last_command is None:
            self.last_command = state.copy()
        repeats = self.n_obs_steps if not self._states else 1
        for _ in range(repeats):
            self._states.append(state.copy())
            self._commands.append(self.last_command.copy())
            self._images.append(dict(images))

    def record_command(self, command: np.ndarray) -> None:
        command = np.asarray(command, dtype=np.float64).reshape(-1)
        if command.shape != (self.action_dim,):
            raise ValueError(f"expected a {self.action_dim}-D command, got {command.shape}")
        self.last_command = command.copy()

    def window(self, image_ticks: tuple[int, ...] | None = None) -> HistoryWindow:
        """The full window; ``image_ticks`` keeps only those ticks' images (the snapshots the model encodes)."""
        if len(self._states) != self.n_obs_steps:
            raise RuntimeError("push at least one observation before reading the window")
        ticks = range(self.n_obs_steps) if image_ticks is None else image_ticks
        keys = self._images[-1].keys()
        images = {key: np.stack([self._images[t][key] for t in ticks]) for key in keys}
        return HistoryWindow(np.stack(self._states), np.stack(self._commands), images)
