"""Advisory stage and progress estimate for the orchestrator: one ``update`` per FLUX chunk (re-plan).

    import sys; sys.path.insert(0, "ws4")                       # or run from ws4/
    from ttt.progress_api import ProgressMonitor

    monitor = ProgressMonitor.load("ws4/ttt/models/progress_frame_ttt.pt")
    monitor.reset(plan=["open_lid", "put_apple_in_box", "close_lid"])     # skills of the plan steps, in order
    ...
    # every chunk (about 1.07 s), with the images the runner just used:
    est = monitor.update(frames={"head": head, "left_wrist": lw, "right_wrist": rw}, state=state12,
                         prev_head=head_7_ticks_ago)
    est.stage, est.stage_name, est.done, est.progress, est.confidence

``frames``/``state`` use the robot's own keys and units (``observation.images.<cam>`` arrays, ``observation.state``
in LeRobot degrees). ``prev_head`` is the head image of the oldest FLUX snapshot of this re-plan (7 ticks back,
``PolicyRequest.images`` snapshot 0); without it the motion features are zero. A model trained on FLUX features
takes ``update(features=vector)`` instead (``extract_flux_features.py`` layout).

What the numbers mean:

* ``stage``: plan step the model ranks first, combining its skill and "how many times did this skill already
  run" predictions with the plan (``features.stage_posterior``); ``stage_probs`` over all plan steps;
* ``done``: probability that the current step finishes before the next re-plan;
* ``stage_progress``: 0..1 inside the step; ``progress``: 0..1 over the plan;
* ``confidence``: probability of ``stage``.

Advisory only: the orchestrator's vision check still decides. The memory is per episode: call ``reset`` at the
start of every plan run. Trained on the ws4 sim (``train_progress.py``): on the real robot, retrain on recorded
episodes first (labels from the orchestrator step log, ``labels.segments_from_events``).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@dataclass(frozen=True)
class StageEstimate:
    stage: int
    stage_name: str
    progress: float
    stage_progress: float
    done: float
    confidence: float
    stage_probs: tuple[float, ...] = ()
    skill_probs: dict[str, float] = field(default_factory=dict)
    occurrence: int = 0


class ProgressMonitor:
    def __init__(self, model, skills: list[str], features: str, featurizer=None, device: str = "cpu"):
        self.model = model.to(device).eval()
        self.skills = list(skills)
        self.features = features
        self.featurizer = featurizer
        self.device = device
        self.plan: list[str] = []
        self._state = None
        self.history: list[StageEstimate] = []

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> ProgressMonitor:
        import torch

        from ttt.features import FrameFeaturizer
        from ttt.progress_memory import EpisodeProgressMemory

        ck = torch.load(Path(path), map_location=device, weights_only=False)
        model = EpisodeProgressMemory(**ck["config"])
        model.load_state_dict(ck["state_dict"])
        cfg = ck.get("featurizer")
        featurizer = (
            FrameFeaturizer(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg.items()}) if cfg else None
        )
        return cls(model, ck["skills"], ck["features"], featurizer, device)

    def reset(self, plan: list[str] | None = None) -> None:
        """Start a new plan run; ``plan`` lists the skill of every step (orchestrator ``Step.skill``), in order."""
        self.plan = list(plan) if plan else list(self.skills)
        unknown = [s for s in self.plan if s not in self.skills]
        if unknown:
            raise ValueError(f"the model was not trained on skills {unknown}; known: {self.skills}")
        self._state = None
        self.history = []

    def featurize(self, frames: dict, state, prev_head=None) -> np.ndarray:
        if self.featurizer is None:
            raise ValueError("this model takes FLUX features: call update(features=...)")
        frames = {k.removeprefix("observation.images."): v for k, v in frames.items()}
        return self.featurizer(frames, np.asarray(state, dtype=np.float64), prev_head)

    def update(self, features=None, *, frames: dict | None = None, state=None, prev_head=None) -> StageEstimate:
        import torch

        from ttt.features import stage_posterior

        if not self.plan:
            self.reset()
        if features is None:
            if frames is None or state is None:
                raise ValueError("pass features=..., or frames=... and state=...")
            features = self.featurize(frames, state, prev_head)
        x = torch.as_tensor(np.asarray(features, dtype=np.float32), device=self.device).reshape(1, 1, -1)
        with torch.no_grad():
            out, self._state = self.model(x, self._state)
        skill_p = torch.softmax(out["subtask"][0, 0], -1).cpu().numpy()
        occ_p = torch.softmax(out["occurrence"][0, 0], -1).cpu().numpy()
        post = stage_posterior(skill_p, occ_p, self.plan, self.skills)
        k = int(post.argmax())
        sp = float(out["stage_progress"][0, 0]) if "stage_progress" in out else 0.0
        est = StageEstimate(
            stage=k,
            stage_name=self.plan[k],
            progress=float((k + sp) / len(self.plan)),
            stage_progress=sp,
            done=float(torch.sigmoid(out["done"][0, 0])),
            confidence=float(post[k]),
            stage_probs=tuple(float(p) for p in post),
            skill_probs={s: float(p) for s, p in zip(self.skills, skill_p, strict=True)},
            occurrence=int(occ_p.argmax()),
        )
        self.history.append(est)
        return est
