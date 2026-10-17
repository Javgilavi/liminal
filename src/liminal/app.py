"""Build an ExecutionContext from config files. The single place where concrete
hardware/sim/policy implementations are chosen."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from liminal.core.interfaces import Arm, SkillBackend
from liminal.orchestration.context import ExecutionContext
from liminal.orchestration.skills import PrimitiveBackend, SkillRouter
from liminal.perception.inspector import SceneInspector
from liminal.perception.types import InspectionProfile
from liminal.perception.vlm import VisionModel
from liminal.perception.world_state import WorldState
from liminal.safety.supervisor import JointLimits, SafetySupervisor


def _load(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text()) or {}


def build_arms(robot_cfg: dict[str, Any]) -> tuple[dict[str, Arm], dict[str, Any]]:
    arms: dict[str, Arm] = {}
    cameras: dict[str, Any] = {}
    world_sim = None
    if "mujoco" in robot_cfg:
        from liminal.sim.mujoco_arm import MujocoWorld

        world_sim = MujocoWorld(robot_cfg["mujoco"]["xml"])
    for name, a in robot_cfg.get("arms", {}).items():
        kind = a.get("type", "mock")
        if kind == "mock":
            from liminal.hardware.mock_arm import MockArm

            arms[name] = MockArm(name, home=robot_cfg.get("home", {}).get(name))
        elif kind == "so101":
            from liminal.hardware.so101 import SO101Arm

            arms[name] = SO101Arm(name, port=a["port"], robot_id=a.get("id"),
                                  max_relative_target=a.get("max_relative_target"),
                                  max_speed=a.get("max_speed", 20.0),
                                  temperature_limit=a.get("temperature_limit", 55.0))
        elif kind == "mujoco":
            from liminal.sim.mujoco_arm import MujocoArm

            arms[name] = MujocoArm(name, world_sim, a["actuator_prefix"])
        else:
            raise ValueError(f"unknown arm type {kind!r}")
    for name, c in robot_cfg.get("cameras", {}).items():
        kind = c.get("type", "mock")
        if kind == "mock":
            from liminal.hardware.cameras import MockCamera

            cameras[name] = MockCamera(name)
        elif kind == "opencv":
            from liminal.hardware.cameras import OpenCVCamera

            cameras[name] = OpenCVCamera(name, c["index"], c.get("width", 640), c.get("height", 480),
                                         fps=c.get("fps", 30), rotation=c.get("rotation", 0),
                                         label=c.get("label"), fourcc=c.get("fourcc"))
        elif kind == "mujoco":
            from liminal.sim.mujoco_arm import MujocoCamera

            cameras[name] = MujocoCamera(name, world_sim, c["mj_camera"])
    if world_sim is not None:
        world_sim.start()
    return arms, cameras


def build_policy(backend: str, policy_cfg: dict[str, Any] | None = None) -> SkillBackend:
    if backend == "mock":
        from liminal.policy.mock import MockBackend

        return MockBackend()
    if backend == "scripted":
        from liminal.policy.scripted import ScriptedBackend

        return ScriptedBackend.from_dir((policy_cfg or {}).get("waypoints_dir", "data/waypoints"))
    if backend == "flux":
        from liminal.policy.flux import FluxConfig, FluxPolicyBackend

        cfg = {k: v for k, v in (policy_cfg or {}).items() if k not in ("instances", "devices")}
        cfg["camera_roles"] = tuple(cfg.get("camera_roles", ("scene", "wrist")))
        return FluxPolicyBackend(FluxConfig(**cfg))
    raise ValueError(f"unknown backend {backend!r} (mock | scripted | flux)")


def build_context(robot: str | Path | dict[str, Any], backend: str = "mock",
                  policy: str | Path | dict[str, Any] | None = None, log_path: str | Path | None = None,
                  policy_backend: SkillBackend | None = None, vision_model: VisionModel | None = None,
                  inspection: InspectionProfile | None = None,
                  inspection_confirmations: int = 2) -> ExecutionContext:
    robot_cfg = robot if isinstance(robot, dict) else _load(robot)
    policy_cfg = policy if isinstance(policy, dict) or policy is None else _load(policy)
    arms, cameras = build_arms(robot_cfg)
    limits = {k: JointLimits(**v) for k, v in robot_cfg.get("limits", {}).items()}
    p = robot_cfg.get("primitives", {})
    policy_cfg = dict(policy_cfg or {})
    # Option A2 (PLAN.md): one independent policy instance per arm, optionally on its own GPU.
    instances = policy_cfg.pop("instances", "shared")
    devices = policy_cfg.pop("devices", {})
    arm_policies: dict[str, SkillBackend] = {}
    if policy_backend is None and instances == "per_arm":
        default_device = policy_cfg.get("device", "cuda")
        # The shared `policy` below stays lazy (FLUX loads on first use), so it costs no VRAM here.
        arm_policies = {a: build_policy(backend, {**policy_cfg, "device": devices.get(a, default_device)})
                        for a in arms}
    router = SkillRouter(
        policy=policy_backend or build_policy(backend, policy_cfg),
        primitives=PrimitiveBackend(home_poses=robot_cfg.get("home", {}), **p),
        arm_policies=arm_policies,
    )
    groups = {g: list(m) for g, m in robot_cfg.get("arm_groups", {}).items()}
    for g, members in groups.items():
        unknown = set(members) - set(arms)
        if g in arms or unknown:
            raise ValueError(f"arm group {g!r} must not shadow an arm and must list known arms (unknown: {unknown})")
    world = WorldState(log_path=log_path)
    ctx = ExecutionContext(
        arms=arms,
        world=world,
        safety=SafetySupervisor(arms, limits),
        skills=router,
        cameras=cameras,
        inspection=inspection or InspectionProfile(),
        arm_groups=groups,
    )
    if vision_model is not None:
        ctx.inspector = SceneInspector(
            vision_model,
            cameras,
            world,
            confirmations=inspection_confirmations,
        )
    return ctx
