"""Step verifiers. A FLUX skill never reports success itself: a step is done only when its success check passes.

- ``ClaudeVerifier``: one structured-output vision call on the head frame (optionally plus the arm's wrist frame),
  with the model chosen per skill in ``skills.yaml`` (Haiku 5.5 for plain checks, Sonnet 5.5 for subtler states).
- ``SimTruthVerifier``: ground truth from the sim's privileged object poses (tests, laptop demo).
- ``ShadowVerifier``: the primary decides, a second verifier runs on the same frame in the background and its
  agreement is logged (measures the Claude verifier against sim ground truth).
- ``ManualVerifier``: nothing automatic; the operator marks the step done or failed in the HMI.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

from .claude import ClaudeJSON, ClaudeRefused, ClaudeUnavailable, image_block
from .cost import BudgetExceeded
from .plan import Step
from .skills import Skill, Station

HEAD = "observation.images.head"
WRIST = {"left": "observation.images.left_wrist", "right": "observation.images.right_wrist"}


@dataclass
class Verdict:
    done: bool
    confidence: float
    observation: str
    source: str
    latency_s: float = 0.0
    cost_usd: float = 0.0
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"done": self.done, "confidence": round(self.confidence, 2), "observation": self.observation,
                "source": self.source, "latency_s": round(self.latency_s, 2), "cost_usd": round(self.cost_usd, 5),
                "error": self.error}


class SimTruthVerifier:
    name = "sim_truth"
    fast = True

    def __init__(self, sim: Any):
        self.sim = sim

    def check(self, step: Step, skill: Skill, obs: dict) -> Verdict:
        if not skill.sim_check:
            return Verdict(False, 0.0, f"no ground-truth check for {skill.name} in the sim", self.name)
        ok, why = self.sim.predicate(skill.sim_check, obs, step.arm)
        return Verdict(ok, 1.0, why, self.name)


class ManualVerifier:
    name = "manual"
    fast = True

    def check(self, step: Step, skill: Skill, obs: dict) -> Verdict:
        return Verdict(False, 0.0, "waiting for the operator to mark this step done", self.name)


VERIFY_SYSTEM = (
    "You verify one step of a robot manipulation task from camera images. The head camera looks down at a table "
    "from behind two robot arms; the robot's left arm is on the image's left. Judge only what is visible now. "
    "Answer done=true only if the stated condition clearly holds; if the view is occluded or ambiguous, answer "
    "done=false with low confidence. 'Inside' means within the container's walls, not on its rim. Keep the "
    "observation to one sentence about what you see."
)
VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "done": {"type": "boolean"},
        "confidence": {"type": "number"},
        "observation": {"type": "string"},
    },
    "required": ["done", "confidence", "observation"],
    "additionalProperties": False,
}


class ClaudeVerifier:
    name = "claude"
    fast = False

    def __init__(self, llm: ClaudeJSON, use_wrist: bool = False, effort: str = "low", max_side: int = 640,
                 haiku_thinking: str = "adaptive", station: Station | None = None):
        self.llm = llm
        self.station = station or Station()
        self.system = self.station.verify_scene or VERIFY_SYSTEM
        self.use_wrist = use_wrist
        self.effort = effort
        self.max_side = max_side
        # Haiku 5.5 accepts thinking "disabled" at effort <= high (api/errors docs); Sonnet 5.5 and Opus 5.5 do not.
        self.haiku_thinking = haiku_thinking

    def content(self, step: Step, obs: dict, skill: Skill | None = None) -> list[dict]:
        crop = self.station.verify_crop
        if crop is None:
            content: list[dict] = [{"type": "text", "text": "Head camera:"}, image_block(obs[HEAD], self.max_side)]
        else:  # the table region, upscaled: the pen in its slot spans only a few dozen pixels in the full frame
            side = self.station.verify_max_side
            content = [{"type": "text", "text": "Head camera (crop of the table):"},
                       image_block(upscale(crop_frac(obs[HEAD], crop), side), side)]
        if self.use_wrist and WRIST[step.arm] in obs:
            content += [{"type": "text", "text": f"{step.arm} gripper camera:"},
                        image_block(obs[WRIST[step.arm]], self.max_side)]
        hint = f" {skill.verify_hint}" if skill is not None and skill.verify_hint else ""
        content.append({"type": "text", "text": f"Condition to check: {step.success_check}{hint}"})
        return content

    def check(self, step: Step, skill: Skill, obs: dict) -> Verdict:
        model = skill.verifier_model
        start = time.perf_counter()
        spent0 = self.llm.meter.total_usd
        thinking = {"type": self.haiku_thinking} if model == "claude-haiku-5-5" else None
        try:
            data, _ = self.llm.call(purpose="verify", model=model, system=self.system,
                                    messages=[{"role": "user", "content": self.content(step, obs, skill)}],
                                    schema=VERIFY_SCHEMA, effort=self.effort, max_tokens=4000, cache_system=False,
                                    thinking=thinking)
        except (BudgetExceeded, ClaudeRefused, ClaudeUnavailable) as exc:
            return Verdict(False, 0.0, "verifier unavailable", model, time.perf_counter() - start,
                           error=f"{type(exc).__name__}: {exc}")
        except Exception as exc:  # network or API errors must not stop the control loop
            return Verdict(False, 0.0, "verifier call failed", model, time.perf_counter() - start,
                           self.llm.meter.total_usd - spent0, f"{type(exc).__name__}: {str(exc)[:200]}")
        conf = float(np.clip(float(data.get("confidence", 0.0)), 0.0, 1.0))
        return Verdict(bool(data.get("done")), conf, str(data.get("observation", "")), model,
                       time.perf_counter() - start, self.llm.meter.total_usd - spent0)


def crop_frac(img: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    return img[round(y0 * h): round(y1 * h), round(x0 * w): round(x1 * w)]


def upscale(img: np.ndarray, max_side: int) -> np.ndarray:
    from PIL import Image

    pil = Image.fromarray(np.asarray(img).astype(np.uint8))
    scale = max_side / max(pil.size)
    if scale > 1:
        pil = pil.resize((round(pil.size[0] * scale), round(pil.size[1] * scale)), Image.LANCZOS)
    return np.asarray(pil)


class ShadowVerifier:
    """Primary decides; the shadow's verdict on the same frame is reported through ``on_result`` when it arrives."""

    def __init__(self, primary: Any, shadow: Any, on_result=None):
        self.primary = primary
        self.shadow = shadow
        self.name = f"{primary.name}+shadow:{shadow.name}"
        self.fast = primary.fast
        self.on_result = on_result or (lambda step, primary, shadow: None)
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="shadow-verify")
        self.pairs: list[tuple[bool, bool]] = []

    def check(self, step: Step, skill: Skill, obs: dict) -> Verdict:
        verdict = self.primary.check(step, skill, obs)

        def run_shadow() -> None:
            other = self.shadow.check(step, skill, obs)
            if other.error is None:
                self.pairs.append((verdict.done, other.done and other.confidence >= skill.confidence))
            self.on_result(step, verdict, other)

        self._pool.submit(run_shadow)
        return verdict

    def agreement(self) -> dict[str, Any]:
        n = len(self.pairs)
        agree = sum(a == b for a, b in self.pairs)
        false_pos = sum((not a) and b for a, b in self.pairs)
        false_neg = sum(a and not b for a, b in self.pairs)
        return {"checks": n, "agree": agree, "false_positive": false_pos, "false_negative": false_neg}
