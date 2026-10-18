"""Exercise real engine construction against fake LeRobot modules; no GPU/network."""
import sys
from types import ModuleType, SimpleNamespace

import pytest

from liminal.policy.flux import FluxConfig
from liminal.policy.lerobot_engine import LeRobotEngine


class Tensor:
    def __init__(self, values):
        self.values = values

    def float(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class Pipeline:
    def __init__(self, state):
        self.resets = 0
        self.steps = [SimpleNamespace(state_dict=lambda: state)]

    def reset(self):
        self.resets += 1


@pytest.fixture
def integration(monkeypatch):
    cfg = SimpleNamespace(n_obs_steps=8, chunk_size=42, fps=30,
                          camera_keys=["observation.images.scene", "observation.images.wrist"],
                          output_features={"action": SimpleNamespace(shape=(6,))})
    policy = SimpleNamespace(config=cfg, resets=0)
    policy.to = lambda device: policy
    policy.eval = lambda: policy

    def reset():
        policy.resets += 1

    policy.reset = reset
    values = {f"{stream}.{q}": Tensor([number] * 6)
              for stream in ("state", "action") for q, number in (("q01", -1), ("q99", 1))}
    action = {k: v for k, v in values.items() if k.startswith("action")}
    pre, post = Pipeline(values), Pipeline(action)
    pre.steps.append(SimpleNamespace(state_dict=lambda: action))
    factory = ModuleType("lerobot.policies.factory")
    factory.make_pre_post_processors = lambda config, pretrained_path: (pre, post)
    modeling = ModuleType("lerobot.policies.flux3.modeling_flux3")
    modeling.Flux3Policy = SimpleNamespace(from_pretrained=lambda checkpoint: policy)
    for name in ("lerobot", "lerobot.policies", "lerobot.policies.flux3"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    monkeypatch.setitem(sys.modules, factory.__name__, factory)
    monkeypatch.setitem(sys.modules, modeling.__name__, modeling)
    return cfg, policy, pre, post


def test_real_engine_loads_saved_processors_and_resets_all_state(integration, tmp_path):
    _, policy, pre, post = integration
    engine = LeRobotEngine(str(tmp_path), FluxConfig())
    engine.reset()
    assert [policy.resets, pre.resets, post.resets] == [2, 2, 2]
    assert engine.statistics["action_statistics"]["q01"] == [-1] * 6


@pytest.mark.parametrize("field,value", [
    ("n_obs_steps", 1), ("chunk_size", 10), ("fps", 60), ("camera_keys", ["observation.images.other"]),
    ("output_features", {"action": SimpleNamespace(shape=(12,))}),
])
def test_real_engine_refuses_incompatible_saved_configuration(integration, tmp_path, field, value):
    cfg, _, _, _ = integration
    setattr(cfg, field, value)
    with pytest.raises(ValueError):
        LeRobotEngine(str(tmp_path), FluxConfig())
