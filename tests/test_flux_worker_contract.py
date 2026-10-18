"""Processor provenance checks without torch, model weights, network or hardware."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "flux_lerobot_worker", Path(__file__).resolve().parents[1] / "scripts/flux_lerobot_worker.py"
)
WORKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKER)


class Value:
    def __init__(self, values):
        self.values = values

    def float(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values


def pipelines():
    values = {f"{s}.{q}": [n] * 6 for s in ("state", "action") for q, n in (("q01", -1), ("q99", 1))}

    def step(name, streams):
        return type(name, (), {"state_dict": lambda self: {
            k: Value(v) for k, v in values.items() if k.split(".")[0] in streams
        }})()

    pre = SimpleNamespace(steps=[step("Observation", {"state", "action"}), step("Target", {"action"})])
    post = SimpleNamespace(steps=[step("Output", {"action"})])
    return pre, post


def test_released_processor_contract_reports_loaded_values(tmp_path):
    contract = WORKER.processor_contract(*pipelines(), str(tmp_path))
    assert contract["normalization_profile"] == "released"
    assert contract["state_statistics"] == contract["action_statistics"] == {"q01": [-1] * 6, "q99": [1] * 6}


def test_inconsistent_output_statistics_are_refused(tmp_path):
    pre, post = pipelines()
    post.steps[0].state_dict = lambda: {"action.q01": Value([-2] * 6), "action.q99": Value([1] * 6)}
    with pytest.raises(ValueError, match="disagree"):
        WORKER.processor_contract(pre, post, str(tmp_path))


def test_hf_profile_provenance_must_match_actual_tensors(tmp_path):
    metadata = {"profile": "hf_space_native", "normalization": {}}
    path = tmp_path / "normalization_source.json"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="differs"):
        WORKER.processor_contract(*pipelines(), str(tmp_path))
    metadata["normalization"] = {s: {"q01": [-1] * 6, "q99": [1] * 6} for s in ("state", "action")}
    path.write_text(json.dumps(metadata))
    assert WORKER.processor_contract(*pipelines(), str(tmp_path))["normalization_profile"] == "hf_space_native"


def test_dataset_native_profile_requires_its_own_saved_quantiles(tmp_path):
    metadata = {"profile": "dataset_native", "normalization": {}}
    path = tmp_path / "normalization_source.json"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="differs"):
        WORKER.processor_contract(*pipelines(), str(tmp_path))
    metadata["normalization"] = {s: {"q01": [-1] * 6, "q99": [1] * 6} for s in ("state", "action")}
    path.write_text(json.dumps(metadata))
    assert WORKER.processor_contract(*pipelines(), str(tmp_path))["normalization_profile"] == "dataset_native"
