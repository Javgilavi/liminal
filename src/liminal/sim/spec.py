"""The ``sim:`` block of a scenario file: how a symbolic scenario maps onto a MuJoCo scene (WS4).

Pure Python, no MuJoCo import, so a sim scenario loads (and grounds, plans, replays) on any laptop.
The rest of the file is an ordinary replay scenario (goal, objects, obligations, capabilities)::

    sim:
      robot: configs/robot/mujoco_dual.yaml     # arms + the MJCF they live in
      bodies: {red_cube: cube_red}               # movable object id -> MuJoCo body (free joint)
      regions:                                   # destination id -> a box geom on the bench
        place_zone: {geom: place_zone, margin_m: 0.0, max_height_m: 0.05}
      initial_arms: {left: [0, -100, 97, 70, 0, 0]}   # LeRobot units: degrees, gripper 0-100
      jitter_m: 0.0                              # random xy offset of every body at reset
      seed: 0
      cameras: {scene: top, wrist: "{arm}_wrist"}     # policy role -> MuJoCo camera ({arm} = skill arm)
      image_sizes: {scene: [640, 480], wrist: [640, 360]}
      evidence_camera: front                     # the frame stored as evidence for every observation
      max_step_deg: 6.0                          # SafetySupervisor clamp per command (= 180 deg/s at 30 Hz)
      settle_speed_mps: 0.02                     # an object moving faster than this is UNKNOWN, not placed

Every obligation ``{object_id, destination_id}`` must name a body and a region: the observer turns it into
"the object's centre lies over the region and rests at most ``max_height_m`` above it".
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

DEFAULT_ROBOT = "configs/robot/mujoco_dual.yaml"
_FIELDS = {"robot", "bodies", "regions", "initial_arms", "jitter_m", "seed", "cameras", "image_sizes",
           "evidence_camera", "max_step_deg", "settle_speed_mps"}


@dataclass(frozen=True)
class RegionSpec:
    geom: str
    margin_m: float = 0.0  # added to the geom's half-size in x and y (negative = stricter)
    max_height_m: float = 0.05  # object centre at most this far above the region's top face


@dataclass(frozen=True)
class SimSpec:
    bodies: dict[str, str]
    regions: dict[str, RegionSpec]
    robot: str = DEFAULT_ROBOT
    initial_arms: dict[str, tuple[float, ...]] = field(default_factory=dict)
    jitter_m: float = 0.0
    seed: int | None = None
    cameras: dict[str, str] = field(default_factory=lambda: {"scene": "top", "wrist": "{arm}_wrist"})
    image_sizes: dict[str, tuple[int, int]] = field(
        default_factory=lambda: {"scene": (640, 480), "wrist": (640, 360)})
    evidence_camera: str = "front"
    max_step_deg: float = 6.0
    settle_speed_mps: float = 0.02

    def camera_for(self, role: str, arm: str) -> str:
        return self.cameras[role].format(arm=arm)


def parse_sim_spec(raw: Any, objects: Mapping[str, str], obligations: Mapping[str, Mapping[str, str]],
                   available_arms: tuple[str, ...]) -> SimSpec:
    if not isinstance(raw, dict):
        raise ValueError("sim must be an object")
    extra = set(raw) - _FIELDS
    if extra:
        raise ValueError(f"sim: unknown fields {sorted(extra)}")
    bodies = _str_map(raw.get("bodies"), "sim.bodies")
    regions_raw = raw.get("regions")
    if not isinstance(regions_raw, dict) or not regions_raw:
        raise ValueError("sim.regions must map destination ids to {geom, margin_m, max_height_m}")
    regions: dict[str, RegionSpec] = {}
    for key, value in regions_raw.items():
        where = f"sim.regions.{key}"
        if not isinstance(value, dict) or set(value) - {"geom", "margin_m", "max_height_m"} \
                or not isinstance(value.get("geom"), str) or not value["geom"]:
            raise ValueError(f"{where} needs a geom (and optional margin_m, max_height_m)")
        height = _number(value.get("max_height_m", 0.05), f"{where}.max_height_m")
        if height <= 0:
            raise ValueError(f"{where}.max_height_m must be positive")
        regions[str(key)] = RegionSpec(value["geom"], _number(value.get("margin_m", 0.0), f"{where}.margin_m"),
                                       height)
    for name in [*bodies, *regions]:
        if name not in objects:
            raise ValueError(f"sim: {name!r} is not a scenario object")
    if set(bodies) & set(regions):
        raise ValueError("sim: an object cannot be both a body and a region")
    for predicate, binding in obligations.items():
        if binding["object_id"] not in bodies or binding["destination_id"] not in regions:
            raise ValueError(f"sim: obligation {predicate!r} needs its object in sim.bodies and its destination "
                             "in sim.regions")
    arms_raw = raw.get("initial_arms", {})
    if not isinstance(arms_raw, dict):
        raise ValueError("sim.initial_arms must map arms to 6 joint values")
    initial_arms = {}
    for arm, values in arms_raw.items():
        if arm not in available_arms or not isinstance(values, list) or len(values) != 6:
            raise ValueError(f"sim.initial_arms.{arm}: needs an available arm and 6 values (degrees, gripper 0-100)")
        initial_arms[str(arm)] = tuple(_number(v, f"sim.initial_arms.{arm}") for v in values)
    cameras = _str_map(raw.get("cameras", {"scene": "top", "wrist": "{arm}_wrist"}), "sim.cameras")
    sizes_raw = raw.get("image_sizes", {"scene": [640, 480], "wrist": [640, 360]})
    if not isinstance(sizes_raw, dict):
        raise ValueError("sim.image_sizes must map roles to [width, height]")
    sizes = {}
    for role, wh in sizes_raw.items():
        if role not in cameras or not isinstance(wh, list) or len(wh) != 2 \
                or any(isinstance(v, bool) or not isinstance(v, int) or not 16 <= v <= 1920 for v in wh):
            raise ValueError(f"sim.image_sizes.{role}: needs a known camera role and [width, height] pixels")
        sizes[str(role)] = (wh[0], wh[1])
    for role in cameras:
        sizes.setdefault(role, (640, 480))
    seed = raw.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ValueError("sim.seed must be an integer")
    robot = raw.get("robot", DEFAULT_ROBOT)
    evidence_camera = raw.get("evidence_camera", "front")
    if not isinstance(robot, str) or not robot or not isinstance(evidence_camera, str) or not evidence_camera:
        raise ValueError("sim.robot and sim.evidence_camera must be non-empty strings")
    jitter, max_step, settle = (_number(raw.get(k, d), f"sim.{k}") for k, d in
                                (("jitter_m", 0.0), ("max_step_deg", 6.0), ("settle_speed_mps", 0.02)))
    if jitter < 0 or max_step <= 0 or settle <= 0:
        raise ValueError("sim.jitter_m must be >= 0, sim.max_step_deg and sim.settle_speed_mps > 0")
    return SimSpec(bodies, regions, robot, initial_arms, jitter, seed, cameras, sizes, evidence_camera, max_step,
                   settle)


def _str_map(value: Any, where: str) -> dict[str, str]:
    if not isinstance(value, dict) or not value or any(
            not isinstance(k, str) or not k or not isinstance(v, str) or not v for k, v in value.items()):
        raise ValueError(f"{where} must map non-empty names to non-empty names")
    return dict(value)


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{where} must be a finite number")
    return float(value)
