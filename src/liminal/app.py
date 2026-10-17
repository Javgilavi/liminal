"""Build an ExecutionContext from config files. The single place where concrete
hardware/sim/policy implementations are chosen."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from liminal.core.interfaces import Arm, SkillBackend
from liminal.orchestration.context import ExecutionContext
from liminal.orchestration.skills import PrimitiveBackend, SkillRouter
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
                                  max_relative_target=a.get("max_relative_target"))
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

            cameras[name] = OpenCVCamera(name, c["index"], c.get("width", 640), c.get("height", 480))
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

        cfg = dict(policy_cfg or {})
        cfg["camera_roles"] = tuple(cfg.get("camera_roles", ("scene", "wrist")))
        return FluxPolicyBackend(FluxConfig(**cfg))
    raise ValueError(f"unknown backend {backend!r} (mock | scripted | flux)")


def build_context(robot: str | Path | dict[str, Any], backend: str = "mock",
                  policy: str | Path | dict[str, Any] | None = None, log_path: str | Path | None = None,
                  policy_backend: SkillBackend | None = None) -> ExecutionContext:
    robot_cfg = robot if isinstance(robot, dict) else _load(robot)
    policy_cfg = policy if isinstance(policy, dict) or policy is None else _load(policy)
    arms, cameras = build_arms(robot_cfg)
    limits = {k: JointLimits(**v) for k, v in robot_cfg.get("limits", {}).items()}
    p = robot_cfg.get("primitives", {})
    router = SkillRouter(
        policy=policy_backend or build_policy(backend, policy_cfg),
        primitives=PrimitiveBackend(home_poses=robot_cfg.get("home", {}), **p),
    )
    return ExecutionContext(
        arms=arms,
        world=WorldState(log_path=log_path),
        safety=SafetySupervisor(arms, limits),
        skills=router,
        cameras=cameras,
    )
