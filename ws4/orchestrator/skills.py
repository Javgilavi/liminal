"""The fixed skill vocabulary (``skills.yaml``): what the planner may choose and how each skill is run and checked."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import PACKAGE_DIR

DEFAULT_VOCABULARY = PACKAGE_DIR / "skills.yaml"  # the final demo: 4-skill chain on the Haller, both arms
PEN_VOCABULARY = PACKAGE_DIR / "skills_haller_pen.yaml"  # the red pen alone, robot-left arm only (2026-10-17)
SIM_VOCABULARY = PACKAGE_DIR / "skills_sim_pack_box.yaml"  # the sim pack-box stand-in
ARMS = ("left", "right")
KINDS = ("flux", "primitive", "operator")  # operator: the arm waits while a person does it, then confirms
PRIMITIVES = ("hold", "home", "tuck", "open_gripper", "close_gripper")
VERIFIER_MODELS = ("claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5")


@dataclass(frozen=True)
class Skill:
    name: str
    kind: str
    arms: tuple[str, ...]
    prompt: str = ""
    description: str = ""
    success_check: str = ""
    timeout_s: float = 45.0
    verify_every_chunks: int = 2
    verifier_model: str = "claude-haiku-5-5"
    confidence: float = 0.7
    max_attempts: int = 2
    package: str | None = None
    hold_after: bool = False
    start_pose: str | None = None  # a station pose name ("home"): the executor ramps the arm there before the policy
    other_arm_pose: str = ""  # a station pose name ("tucked"): the other arm is ramped there first and stays
    precheck: str = ""  # a skill whose success check must still hold before this one starts (else: e-stop, hold)
    instruction: str = ""  # operator skills: what the operator does before pressing Done
    verify_hint: str = ""  # appended to the success check for the vision model: what counts as done and not done
    verify_scene: str = ""  # this skill's own verifier system prompt (else station.verify_scene)
    # The check names the state it sees, one of these (the first one = done), instead of answering done yes / no:
    # Haiku described a pen in its slot correctly and still answered done=false (green pen, 2026-10-18).
    verify_states: tuple[str, ...] = ()
    park_after: bool = False  # after a verified success: open the gripper, ramp to home, check again, then done
    keepout: str = ""  # keep-out config with this skill's demonstrated envelope (else the arm's station zones only)
    keywords: tuple[str, ...] = ()
    sim: dict | None = None
    sim_check: dict | None = None

    @property
    def is_flux(self) -> bool:
        return self.kind == "flux"

    def planner_view(self) -> dict[str, Any]:
        """What Claude sees: no sim internals, no package names."""
        view: dict[str, Any] = {"name": self.name, "kind": self.kind, "arms": list(self.arms),
                                "description": self.description}
        if self.is_flux:
            view.update(flux_prompt=self.prompt, default_success_check=self.success_check,
                        timeout_s=self.timeout_s)
            if self.start_pose:
                view["starts_from"] = self.start_pose
        return view


@dataclass
class Station:
    name: str = "xlerobot"
    arms_available: tuple[str, ...] = ARMS
    home_pose: tuple[float, ...] | None = None  # LeRobot degrees + gripper; None = the primitives' default
    notes: str = ""
    # Success checks on this station (tuned on recorded frames, research/orchestrator.md "Success checks on real
    # frames"): the system prompt describing the scene, and the head-frame crop (x0, y0, x1, y1 as fractions) that is
    # upscaled to verify_max_side before it is sent. Empty: the generic prompt and the full frame.
    verify_scene: str = ""
    verify_crop: tuple[float, float, float, float] | None = None
    verify_max_side: int = 640
    home_poses: dict[str, tuple[float, ...]] = field(default_factory=dict)  # per-arm override of home_pose
    poses: dict[str, dict[str, tuple[float, ...]]] = field(default_factory=dict)  # arm -> pose name -> joints
    demo_goal: str = ""  # what the HMI and the CLI propose when the operator has typed nothing

    def home_for(self, arm: str) -> tuple[float, ...] | None:
        return self.poses.get(arm, {}).get("home") or self.home_poses.get(arm, self.home_pose)

    def poses_for(self, arm: str) -> dict[str, tuple[float, ...]]:
        """Named poses of one arm ("home" always, when known)."""
        out = dict(self.poses.get(arm, {}))
        if "home" not in out and self.home_for(arm) is not None:
            out["home"] = self.home_for(arm)
        return out


@dataclass
class Vocabulary:
    skills: dict[str, Skill]
    version: int = 1
    source: str = ""
    raw: dict = field(default_factory=dict)
    station: Station = field(default_factory=Station)

    def __getitem__(self, name: str) -> Skill:
        return self.skills[name]

    def __contains__(self, name: object) -> bool:
        return name in self.skills

    def names(self) -> list[str]:
        return list(self.skills)

    def flux_prompts(self) -> list[str]:
        return [s.prompt for s in self.skills.values() if s.is_flux]

    def planner_view(self) -> list[dict[str, Any]]:
        return [s.planner_view() for s in self.skills.values()]


def load_vocabulary(path: str | Path | None = None) -> Vocabulary:
    import yaml

    path = Path(path) if path else DEFAULT_VOCABULARY
    data = yaml.safe_load(path.read_text()) or {}
    defaults = dict(data.get("defaults") or {})
    skills: dict[str, Skill] = {}
    errors: list[str] = []
    st = data.get("station") or {}
    available = tuple(st.get("arms_available") or ARMS)
    home = st.get("home_pose")
    per_arm = home if isinstance(home, dict) else {}  # home_pose: {left: [...], right: [...]} or one list for all
    home = None if isinstance(home, dict) else home
    if any(a not in ARMS for a in available):
        errors.append(f"station.arms_available must be a subset of {ARMS}")
    for where, pose in [("", home)] + [(f".{a}", p) for a, p in per_arm.items()]:
        if pose is not None and len(pose) != 6:
            errors.append(f"station.home_pose{where} needs 6 values (5 joints in LeRobot degrees + gripper 0..100)")
    if any(a not in ARMS for a in per_arm):
        errors.append(f"station.home_pose keys must be among {ARMS}")
    crop = st.get("verify_crop")
    if crop is not None and not (len(crop) == 4 and 0 <= crop[0] < crop[2] <= 1 and 0 <= crop[1] < crop[3] <= 1):
        errors.append("station.verify_crop must be [x0, y0, x1, y1] fractions with x0 < x1 and y0 < y1")
    station = Station(str(st.get("name", "xlerobot")), available,
                      None if home is None else tuple(float(v) for v in home), str(st.get("notes") or "").strip(),
                      str(st.get("verify_scene") or "").strip(),
                      None if crop is None else tuple(float(v) for v in crop), int(st.get("verify_max_side", 640)))
    station.home_poses = {a: tuple(float(v) for v in p) for a, p in per_arm.items() if len(p) == 6}
    for arm, named in (st.get("poses") or {}).items():
        if arm not in ARMS:
            errors.append(f"station.poses: unknown arm {arm!r}")
            continue
        for pose, q in (named or {}).items():
            if len(q) != 6:
                errors.append(f"station.poses.{arm}.{pose} needs 6 values (5 joints + gripper)")
            else:
                station.poses.setdefault(arm, {})[str(pose)] = tuple(float(v) for v in q)
    station.demo_goal = str(st.get("demo_goal") or "").strip()
    for name, spec in (data.get("skills") or {}).items():
        spec = {**defaults, **(spec or {})}
        kind = spec.get("kind")
        # A skill only runs on arms the station really has (the Haller demo drives its left arm only).
        arms = tuple(a for a in spec.get("arms") or () if a in available)
        if kind not in KINDS:
            errors.append(f"{name}: kind must be one of {KINDS}")
        if not arms or any(a not in ARMS for a in arms):
            errors.append(f"{name}: arms must be a non-empty subset of {ARMS}")
        if kind == "flux" and not str(spec.get("prompt") or "").strip():
            errors.append(f"{name}: a flux skill needs its exact FLUX prompt")
        if kind == "primitive" and name not in PRIMITIVES:
            errors.append(f"{name}: unknown primitive (have {PRIMITIVES})")
        if kind == "operator" and not str(spec.get("instruction") or "").strip():
            errors.append(f"{name}: an operator skill needs an instruction for the operator")
        for key in ("start_pose", "other_arm_pose"):
            pose = spec.get(key)
            if not pose:
                continue
            on = arms if key == "start_pose" else tuple(a for a in available if a not in arms)
            missing = [a for a in on if pose not in station.poses_for(a)]
            if missing:
                errors.append(f"{name}: {key} {pose!r} is not a station pose of the {', '.join(missing)} arm")
        if spec.get("verifier_model") not in VERIFIER_MODELS:
            errors.append(f"{name}: verifier_model must be one of {VERIFIER_MODELS}")
        skills[name] = Skill(
            name=name,
            kind=kind,
            arms=arms,
            prompt=str(spec.get("prompt") or ""),
            description=str(spec.get("description") or ""),
            success_check=str(spec.get("success_check") or ""),
            timeout_s=float(spec.get("timeout_s", 45)),
            verify_every_chunks=max(1, int(spec.get("verify_every_chunks", 2))),
            verifier_model=str(spec.get("verifier_model")),
            confidence=float(spec.get("confidence", 0.7)),
            max_attempts=int(spec.get("max_attempts", 2)),
            package=spec.get("package"),
            hold_after=bool(spec.get("hold_after", False)),
            start_pose=spec.get("start_pose"),
            other_arm_pose=str(spec.get("other_arm_pose") or ""),
            precheck=str(spec.get("precheck") or ""),
            instruction=str(spec.get("instruction") or ""),
            verify_hint=str(spec.get("verify_hint") or "").strip(),
            verify_scene=str(spec.get("verify_scene") or "").strip(),
            verify_states=tuple(str(x) for x in spec.get("verify_states") or ()),
            park_after=bool(spec.get("park_after", False)),
            keepout=str(spec.get("keepout") or ""),
            keywords=tuple(str(k).lower() for k in spec.get("keywords") or ()),
            sim=spec.get("sim"),
            sim_check=spec.get("sim_check"),
        )
    for s in skills.values():
        if s.precheck and (s.precheck not in skills or not skills[s.precheck].success_check):
            errors.append(f"{s.name}: precheck {s.precheck!r} must name a skill with a success_check")
    prompts = [s.prompt for s in skills.values() if s.is_flux]
    if len(prompts) != len(set(prompts)):
        errors.append("two flux skills share one prompt")
    if "hold" not in skills:
        errors.append("the vocabulary must define the hold primitive")
    if errors:
        raise ValueError(f"{path}: " + "; ".join(errors))
    return Vocabulary(skills, int(data.get("version", 1)), str(path), data, station)
