"""Real BT/registry/executor boundaries with motor-free engine doubles."""
import asyncio
import hashlib
import json
import threading

import pytest

from liminal.app import build_context
from liminal.core.types import SO101_JOINTS, Skill, Status
from liminal.orchestration.bt import SkillNode
from liminal.policy.flux import FluxConfig, FluxPolicyBackend


def config(tmp_path, **overrides):
    entries = {}
    for name in ("remove", "insert"):
        root = tmp_path / name
        root.mkdir()
        (root / "config.json").write_text("{}")
        (root / "weights.safetensors").write_bytes(name.encode())
        entries[name] = dict(path=str(root), base="base@revision", action_dim=6,
                             camera_roles=["scene", "wrist"], targets=["left", "right"], embodiment="so101",
                             joint_order=list(SO101_JOINTS), state_units="degrees_percent",
                             action_semantics="absolute_position", obs_history=2, chunk_size=4, control_hz=1000)
        manifest = {k: v for k, v in entries[name].items() if k != "path"}
        manifest["normalization_profile"] = "released"
        (root / "liminal_provenance.json").write_text(json.dumps(manifest))
        digest = hashlib.sha256()
        for path in sorted(root.iterdir()):
            digest.update(path.name.encode() + b"\0")
            digest.update(path.read_bytes())
        entries[name]["sha256"] = digest.hexdigest()
    return FluxConfig(checkpoints=entries, skill_checkpoints={"remove": "remove", "insert": "insert"},
                      midtraining_base="base@revision", joint_order=SO101_JOINTS,
                      obs_history=2, chunk_size=4, execute_steps=2,
                      control_hz=1000, camera_map={"right": {"scene": "top", "wrist": "right_wrist"}}, **overrides)


class Engine:
    instances = []

    def __init__(self, checkpoint, cfg, profile):
        self.checkpoint, self.cfg = checkpoint, cfg
        self.calls = []
        self.resets = 0
        self.instances.append(self)

    def reset(self):
        self.resets += 1

    def predict(self, history, commands, prompt):
        assert len(history) == len(commands) == self.cfg.obs_history
        assert history[0].timestamp < history[-1].timestamp
        self.calls.append((history, commands, prompt))
        return [[1.0] * self.cfg.action_dim] * self.cfg.chunk_size


@pytest.fixture
def engine(monkeypatch):
    Engine.instances = []
    monkeypatch.setattr("liminal.policy.lerobot_engine.LeRobotEngine", Engine)
    return Engine


def test_skill_switch_loads_bundle_records_identity_and_resets_history(tmp_path, engine):
    backend = FluxPolicyBackend(config(tmp_path))
    ctx = build_context("configs/robot/mock_dual.yaml", policy_backend=backend)

    async def scenario():
        for name in ("remove", "insert", "insert"):
            assert await SkillNode(Skill("right", name, prompt=name)).run(ctx) is Status.SUCCESS

    asyncio.run(scenario())
    assert len(engine.instances) == 2
    assert [e.resets for e in engine.instances] == [1, 2]
    sessions = [e for e in ctx.world.events if e["kind"] == "policy_session"]
    assert [e["checkpoint"]["name"] for e in sessions] == ["remove", "insert", "insert"]
    assert sessions[0]["checkpoint"]["sha256"] != sessions[1]["checkpoint"]["sha256"]
    windows = [call[0] for e in engine.instances for call in e.calls]
    assert windows[0][-1].timestamp < windows[1][0].timestamp < windows[2][0].timestamp


@pytest.mark.parametrize("failure", [
    "missing", "dimensions", "base", "provenance", "digest", "targets", "units", "load",
])
def test_failed_switch_cannot_reuse_previous_weights_or_move(tmp_path, engine, monkeypatch, failure):
    cfg = config(tmp_path)
    backend = FluxPolicyBackend(cfg)
    ctx = build_context("configs/robot/mock_dual.yaml", policy_backend=backend)

    async def scenario():
        assert await SkillNode(Skill("right", "remove", prompt="remove")).run(ctx) is Status.SUCCESS
        before = len(ctx.arms["right"].history)
        if failure == "missing":
            cfg.skill_checkpoints.pop("insert")
        elif failure == "dimensions":
            cfg.checkpoints["insert"]["action_dim"] = 12
        elif failure == "base":
            cfg.checkpoints["insert"]["base"] = "wrong"
        elif failure == "digest":
            (tmp_path / "insert" / "weights.safetensors").write_bytes(b"changed")
        elif failure == "targets":
            cfg.checkpoints["insert"]["targets"] = ["left"]
        elif failure == "units":
            cfg.checkpoints["insert"]["state_units"] = "radians"
        elif failure == "provenance":
            (tmp_path / "insert" / "liminal_provenance.json").unlink()
        else:
            def fail(*args):
                raise RuntimeError("load failed")
            monkeypatch.setattr("liminal.policy.lerobot_engine.LeRobotEngine", fail)
        assert await SkillNode(Skill("right", "insert", prompt="insert")).run(ctx) is Status.FAILURE
        assert len(ctx.arms["right"].history) == before
        assert ctx.arms["right"].stopped

    asyncio.run(scenario())


def test_cancel_loading_quarantines_and_blocks_queued_skill(tmp_path, engine, monkeypatch):
    entered, finish = threading.Event(), threading.Event()

    class SlowEngine(Engine):
        def __init__(self, *args):
            entered.set()
            finish.wait(2)
            super().__init__(*args)

    monkeypatch.setattr("liminal.policy.lerobot_engine.LeRobotEngine", SlowEngine)
    backend = FluxPolicyBackend(config(tmp_path))
    ctx = build_context("configs/robot/mock_dual.yaml", policy_backend=backend)

    async def scenario():
        first = asyncio.create_task(SkillNode(Skill("right", "remove", prompt="remove")).run(ctx))
        assert await asyncio.to_thread(entered.wait, 1)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        result = await SkillNode(Skill("left", "insert", prompt="insert")).run(ctx)
        finish.set()
        assert result is Status.FAILURE
        assert all(not arm.history for arm in ctx.arms.values())

    asyncio.run(scenario())


def test_inference_timeout_blocks_concurrent_waiter(tmp_path, engine, monkeypatch):
    finish = threading.Event()

    class SlowEngine(Engine):
        def predict(self, *args):
            finish.wait(1)
            return super().predict(*args)

    monkeypatch.setattr("liminal.policy.lerobot_engine.LeRobotEngine", SlowEngine)
    backend = FluxPolicyBackend(config(tmp_path, inference_timeout_s=0.01))
    ctx = build_context("configs/robot/mock_dual.yaml", policy_backend=backend)

    async def scenario():
        results = await asyncio.gather(*[
            SkillNode(Skill(arm, "remove", prompt="remove")).run(ctx) for arm in ("right", "left")
        ])
        finish.set()
        assert results == [Status.TIMEOUT, Status.FAILURE]
        assert all(not arm.history for arm in ctx.arms.values())

    asyncio.run(scenario())
    assert len(engine.instances) == 1


def test_expired_inference_observation_never_reaches_motors(tmp_path, engine, monkeypatch):
    import time

    class DelayedEngine(Engine):
        def predict(self, *args):
            time.sleep(0.02)
            return super().predict(*args)

    monkeypatch.setattr("liminal.policy.lerobot_engine.LeRobotEngine", DelayedEngine)
    backend = FluxPolicyBackend(config(tmp_path, max_observation_age_s=0.005))
    ctx = build_context("configs/robot/mock_dual.yaml", policy_backend=backend)
    assert asyncio.run(SkillNode(Skill("right", "remove", prompt="remove")).run(ctx)) is Status.FAILURE
    assert not ctx.arms["right"].history


@pytest.mark.parametrize("change", ["delta", "reversed", "wrong_units"])
def test_matching_config_and_manifest_still_must_match_driver(tmp_path, engine, change):
    cfg = config(tmp_path)
    entry = cfg.checkpoints["remove"]
    if change == "delta":
        cfg.action_semantics = entry["action_semantics"] = "delta"
    elif change == "reversed":
        cfg.joint_order = tuple(reversed(cfg.joint_order))
        entry["joint_order"] = list(cfg.joint_order)
    else:
        cfg.state_units = entry["state_units"] = "radians"
    backend = FluxPolicyBackend(cfg)
    ctx = build_context("configs/robot/mock_dual.yaml", policy_backend=backend)
    status = asyncio.run(SkillNode(Skill("right", "remove", prompt="remove")).run(ctx))
    assert status is Status.FAILURE
    assert not engine.instances  # rejected by driver contract before loading/hashing
    assert not ctx.arms["right"].history


def test_bimanual_registry_uses_declared_group_joint_order(tmp_path, engine):
    cfg = config(tmp_path)
    cfg.action_dim = 12
    cfg.joint_order = tuple(f"{arm}.{joint}" for arm in ("left", "right") for joint in SO101_JOINTS)
    cfg.camera_roles = ("scene", "left_wrist", "right_wrist")
    cfg.camera_map["both"] = {"scene": "top", "left_wrist": "left_wrist", "right_wrist": "right_wrist"}
    entry = cfg.checkpoints["remove"]
    entry.update(action_dim=12, joint_order=list(cfg.joint_order), targets=["both"],
                 camera_roles=list(cfg.camera_roles))
    root = tmp_path / "remove"
    manifest = {k: v for k, v in entry.items() if k not in ("path", "sha256")}
    manifest["normalization_profile"] = "released"
    (root / "liminal_provenance.json").write_text(json.dumps(manifest))
    digest = hashlib.sha256()
    for path in sorted(root.iterdir()):
        digest.update(path.name.encode() + b"\0")
        digest.update(path.read_bytes())
    entry["sha256"] = digest.hexdigest()
    backend = FluxPolicyBackend(cfg)
    ctx = build_context("configs/robot/mock_dual.yaml", policy_backend=backend)
    assert asyncio.run(SkillNode(Skill("both", "remove", prompt="bimanual")).run(ctx)) is Status.SUCCESS
    assert ctx.arms["left"].history and ctx.arms["right"].history
    history = engine.instances[0].calls[0][0]
    assert len(history[0].state) == 12
