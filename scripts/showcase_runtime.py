"""Run the real replacement BT and FLUX executor against an explicit symbolic simulator.

No learned weights, GPU or hardware: synthetic checkpoint payloads and deterministic
joint goals exercise orchestration, registry switching, inspection and recovery.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
from pathlib import Path

import yaml
from check_checkpoint_registry import bundle_hash

from liminal.app import build_context, build_policy
from liminal.core.types import SkillResult, Status
from liminal.orchestration import load_task, run_task
from liminal.perception.types import InspectionProfile, SceneAssessment
from liminal.policy.flux import FluxPolicyBackend


class Scene:
    def __init__(self, blocked):
        self.values = dict(lid_open=False, access_clear=not blocked, access_blocked=blocked,
                           faulty_component_in_red_cup=False, replacement_in_slot=False,
                           cable_secured_clear=False, cable_restored=False)
        self.ctx = None

    async def inspect_until_known(self, request, **kwargs):
        values = {**self.values, **{
            predicate: all(abs(q) < 0.001 for q in self.ctx.arms[arm].read_state())
            for predicate, arm in (("main_arm_home", "right"), ("helper_arm_home", "left"))
        }}
        assessment = SceneAssessment.from_model_output(request, {"predicates": {
            key: {"value": values.get(key, False), "evidence": "symbolic simulator sensor"}
            for key in request.predicates
        }})
        for item in assessment.observations.values():
            self.ctx.world.observe(item.predicate, item.value, source="symbolic_simulator",
                                   captured_at=assessment.captured_at)
        return assessment


class SimulatedFlux(FluxPolicyBackend):
    def __init__(self, cfg, scene, fail_once):
        super().__init__(cfg)
        self.scene, self.fail_once = scene, fail_once
        self.calls = []

    def load(self):
        self._checkpoint_identity = self._selected_checkpoint.identity()
        self._model = object()  # Explicit stand-in: these payloads are not learned weights.

    def predict(self, history, instruction, target):
        assert len(history) == self.cfg.obs_history
        return [[0.5] * self.cfg.action_dim] * self.cfg.chunk_size

    def _send(self, target, action, ctx):
        super()._send(target, action, ctx)
        effects = {"open_lid": "lid_open", "clear_cable_to_hold": "cable_secured_clear",
                   "remove_to_red_cup": "faulty_component_in_red_cup",
                   "green_cup_to_slot": "replacement_in_slot", "restore_cable": "cable_restored"}
        self.scene.values[effects[self._checkpoint_name]] = True
        if self._checkpoint_name == "restore_cable":
            self.scene.values["cable_secured_clear"] = False

    async def execute(self, skill, ctx):
        self.calls.append(skill.name)
        if skill.name == "green_cup_to_slot" and self.fail_once:
            self.fail_once = False
            return SkillResult(Status.FAILURE, skill, "simulated placement failure before motion")
        return await super().execute(skill, ctx)


async def demonstrate(output, blocked, recover):
    output.mkdir(parents=True, exist_ok=True)
    (output / "events.jsonl").unlink(missing_ok=True)
    scene = Scene(blocked)
    cfg = yaml.safe_load(Path("configs/policy/showcase_registry.yaml").read_text())
    with tempfile.TemporaryDirectory(prefix="liminal-symbolic-") as temporary:
        cfg.update(midtraining_base="symbolic-base@v1", control_hz=1000, obs_history=2,
                   chunk_size=4, execute_steps=2)
        for name, entry in cfg["checkpoints"].items():
            root = Path(temporary) / name
            root.mkdir()
            entry.update(base=cfg["midtraining_base"], path=str(root), control_hz=1000,
                         obs_history=2, chunk_size=4)
            manifest = {k: v for k, v in entry.items() if k not in ("path", "sha256")}
            (root / "liminal_provenance.json").write_text(json.dumps(manifest))
            (root / "config.json").write_text('{"simulation_only": true}')
            (root / "weights.safetensors").write_text(f"symbolic payload only: {name}")
            entry["sha256"] = bundle_hash(root)
        backend = SimulatedFlux(build_policy("flux", cfg).cfg, scene, recover)
        ctx = build_context("configs/robot/mock_dual.yaml", policy_backend=backend, log_path=output / "events.jsonl")
        ctx.inspection = InspectionProfile(definitions={key: key.replace("_", " ") for key in scene.values})
        scene.ctx, ctx.inspector = ctx, scene
        report = await run_task(load_task("configs/tasks/replace_component.yaml", arms=ctx.arm_names), ctx)
        result = {"evidence": "symbolic simulation; no learned inference or physical grasp validation",
                  "scenario": "blocked" if blocked else "clear", "recovery_injected": recover,
                  "status": report.status.value, "goal_verified": report.goal_verified,
                  "atomic_skills": backend.calls,
                  "checkpoint_sessions": [e["checkpoint"] for e in ctx.world.events if e["kind"] == "policy_session"],
                  "commands": {arm: len(driver.history) for arm, driver in ctx.arms.items()}}
        (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
        if not report.goal_verified:
            raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/showcase"))
    args = parser.parse_args()
    for name, blocked, recover in (("clear", False, False), ("blocked", True, False), ("recovery", True, True)):
        asyncio.run(demonstrate(args.output / name, blocked, recover))


if __name__ == "__main__":
    main()
