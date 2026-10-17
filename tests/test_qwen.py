import asyncio

import pytest

from liminal.perception import load_perception_config
from liminal.perception.qwen import QwenInferenceMetrics, QwenVisionConfig, QwenVisionModel, _EngineResult
from liminal.perception.vlm import build_vision_model, warm_vision_model


def test_shipped_qwen_config_is_lazy_and_complete():
    runtime = load_perception_config("configs/perception/qwen3_vl_4b.yaml")
    assert isinstance(runtime.model, QwenVisionModel)
    assert not runtime.model.loaded
    assert runtime.model.config.checkpoint == "Qwen/Qwen3-VL-4B-Instruct"
    assert runtime.model.config.device == "cuda:2"
    assert runtime.confirmations == 2
    assert set(runtime.inspection.definitions) == {
        "lid_open",
        "access_clear",
        "access_blocked",
        "cable_secured_clear",
        "faulty_component_in_red_cup",
        "replacement_in_slot",
        "cable_restored",
        "main_arm_home",
    }


def test_qwen_model_loads_once_and_records_each_inference(monkeypatch):
    metrics = QwenInferenceMetrics(0.1, 0.2, 0.3, 0.4, 100.0, 120.0)

    class Engine:
        gpu_memory_allocated_mb = 90.0

        def __init__(self):
            self.calls = []

        def inspect(self, prompt, images):
            self.calls.append((prompt, images))
            return _EngineResult('{"predicates": {}}', metrics)

    engine = Engine()
    loads = 0

    def factory(config):
        nonlocal loads
        loads += 1
        return engine

    async def inline(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", inline)
    model = QwenVisionModel(QwenVisionConfig(device="cpu"), engine_factory=factory)
    warm_vision_model(model)
    warm_vision_model(model)

    async def scenario():
        return await model.inspect("first", ["image"]), await model.inspect("second", ["image"])

    assert asyncio.run(scenario()) == ('{"predicates": {}}', '{"predicates": {}}')
    assert loads == 1 and len(engine.calls) == 2
    assert model.loaded and model.load_metrics.gpu_memory_allocated_mb == 90.0
    assert model.inference_metrics == [metrics, metrics]


def test_unknown_vision_provider_is_rejected():
    with pytest.raises(ValueError, match="unsupported vision model provider"):
        build_vision_model({"provider": "missing"})
