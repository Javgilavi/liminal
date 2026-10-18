"""Fixed-window experiments must keep holdout data and failed gates out of training."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SPEC = importlib.util.spec_from_file_location(
    "fitting", Path(__file__).resolve().parents[1] / "scripts/sim_flux_assembly/fitting.py"
)
fit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fit)
sys.modules["fitting"] = fit


def load_script(name):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parents[1] / f"scripts/sim_flux_assembly/{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def manifest():
    return {"fps": 20, "episodes": [
        {"episode_index": 0, "split": "train", "skill": "pick_gear", "frames": 274,
         "start": 0, "end": 274, "source": "train.npz", "seed": 100},
        {"episode_index": 1, "split": "validation", "skill": "pick_gear", "frames": 260,
         "start": 10, "end": 270, "source": "validation.npz", "seed": 10000},
    ]}


def test_frozen_windows_respect_full_history_future_and_holdout():
    frozen = fit.freeze_windows(manifest())
    assert frozen == fit.freeze_windows(manifest())
    assert len(frozen["windows"]) == 16
    assert fit.fit_indices(frozen) == [8, 39, 71, 103, 135, 167, 199, 231]
    for row in frozen["windows"]:
        assert row["source_anchor"] - 8 >= row["start"]
        assert row["source_anchor"] + 42 < row["end"]
    assert set(fit.fit_indices(frozen)).isdisjoint(
        r["dataset_index"] for r in frozen["windows"] if r["split"] == "validation"
    )


@pytest.mark.parametrize("field,value", [("source", "train.npz"), ("seed", 100)])
def test_overlapping_holdout_is_rejected(field, value):
    data = manifest()
    data["episodes"][1][field] = value
    with pytest.raises(ValueError, match="FIT_LEAKAGE"):
        fit.freeze_windows(data)


def test_short_episode_and_missing_split_fail():
    data = manifest()
    data["episodes"][0]["frames"] = 50
    with pytest.raises(ValueError, match="FIT_WINDOW_BOUNDS"):
        fit.freeze_windows(data)
    with pytest.raises(ValueError, match="FIT_SPLIT"):
        fit.freeze_windows({"fps": 20, "episodes": []})


def test_training_config_is_bounded_and_leaves_template_unchanged():
    template = {"dataset": {}, "policy": {"caption_dropout": 0.1},
                "accelerator": {"gradient_accumulation": {"steps": 8}}}
    cfg = fit.training_config(template, "checkpoint", "new-output")
    assert cfg["dataset"] == {"episodes": [0], "eval_split": 0.0}
    assert cfg["steps"] == 1000 and cfg["policy"]["caption_dropout"] == 0
    assert template["policy"]["caption_dropout"] == 0.1
    for steps in [0, 7, 1004]:
        with pytest.raises(ValueError, match="FIT_BUDGET"):
            fit.training_config(template, "checkpoint", "output", steps)


def metrics(error, grip=1):
    return {"groups": [{"split": "train", "joint_rmse": {"32": error}, "gripper_mae": {"32": grip}}]}


@pytest.mark.parametrize("after,picked,heldout,action", [
    (8, False, False, "inspect_optimization_and_serialization"),
    (1, False, False, "inspect_startup_history_and_closed_loop_drift"),
    (1, True, False, "collect_targeted_recovery_development_data"),
    (1, True, True, "evaluate_new_scenes_before_expanding_skills"),
])
def test_fitting_and_real_pickup_are_independent_gates(after, picked, heldout, action):
    rolls = [{"seed": seed, "task_report": {"goal_verified": ok}}
             for seed, ok in [(100, picked), (10000, heldout)]]
    decision = fit.decide(metrics(10), metrics(after), rolls)
    assert decision["next_action"] == action
    assert decision["promoted"] is False


def test_gripper_error_and_missing_rollout_prevent_acceptance():
    rolls = [{"seed": seed, "task_report": {"goal_verified": True}} for seed in [100, 10000]]
    assert not fit.decide(metrics(10), metrics(1, 20), rolls)["window_fit_passed"]
    with pytest.raises(ValueError, match="FIT_ROLLOUTS"):
        fit.decide(metrics(10), metrics(1), rolls[:1])


def test_official_trainer_passes_accelerator_by_keyword():
    module = load_script("fit_train")
    accelerator = object()
    assert module.update_accelerator((0, 1, 2, 3, 4), {"accelerator": accelerator}) is accelerator, (
        "[BUG-1 REGRESSION] official trainer passes accelerator as a keyword"
    )
    assert module.update_accelerator((0, 1, 2, 3, 4, accelerator), {}) is accelerator


def test_fixed_sampler_repeats_exact_anchors():
    sampler = load_script("fit_train").FixedSampler([8, 100, 167, 231])
    assert len(sampler) == 4
    assert list(sampler) == list(sampler) == [8, 100, 167, 231]


def test_failed_subprocess_retains_log_and_raises(tmp_path):
    module = load_script("run_fitting")
    log = tmp_path / "failed.log"
    with pytest.raises(RuntimeError, match="FIT_DOWNSTREAM: exit=7"):
        module.run_stage([sys.executable, "-c", "print('intentional diagnostic failure');raise SystemExit(7)"], log, 5)
    assert "intentional diagnostic failure" in log.read_text()


def test_subprocess_timeout_is_a_failure(tmp_path):
    module = load_script("run_fitting")
    with pytest.raises(RuntimeError, match="FIT_TIMEOUT"):
        module.run_stage([sys.executable, "-c", "import time;time.sleep(30)"], tmp_path / "timeout.log", 0.1)


def test_artifact_hash_is_exact(tmp_path):
    path = tmp_path / "artifact"
    path.write_bytes(b"abc")
    assert fit.digest(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_pipeline_stops_after_failed_contract_and_keeps_failure_evidence(tmp_path, monkeypatch):
    module = load_script("run_fitting")
    baseline = tmp_path / "training/checkpoints/003000/pretrained_model"
    baseline.mkdir(parents=True)
    data = manifest()
    for row in data["episodes"]:
        source = tmp_path / row["source"]
        source.write_bytes(b"immutable input fixture")
        row["source"] = str(source)
    (tmp_path / "dataset-manifest.json").write_text(json.dumps(data))
    (tmp_path / "hf-sim-normalization.json").write_text("{}")
    (tmp_path / "scene.xml").write_text("<mujoco/>")
    monkeypatch.setattr(module, "checkpoint_inputs", lambda *a, **kw: [])
    monkeypatch.setattr(module.importlib.metadata, "version", lambda _: "fixture-version")
    (tmp_path / "train-config.json").write_text(json.dumps({
        "dataset": {}, "policy": {}, "accelerator": {"gradient_accumulation": {"steps": 4}}
    }))
    monkeypatch.setattr(module.shutil, "disk_usage", lambda _: SimpleNamespace(free=50 * 1024**3))
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **kw: "test-revision\n")
    commands = []

    def fail(command, log, timeout):
        commands.append(command)
        raise RuntimeError("FIT_DOWNSTREAM: intentional contract failure")

    monkeypatch.setattr(module, "run_stage", fail)
    output = tmp_path / "failed-run"
    args = SimpleNamespace(root=tmp_path, output=output, lerobot=tmp_path, steps=1000, audit_only=False)
    with pytest.raises(RuntimeError, match="intentional contract failure"):
        module.main(args)
    saved = json.loads((output / "status.json").read_text())
    assert saved["completed"] is False and saved["failed_stage"] == "contract-audit"
    assert len(commands) == 1 and "--audit-only" in commands[0]
    assert not (output / "training").exists()


def test_checkpoint_provenance_includes_base_and_both_encoders(tmp_path):
    adapter, base, text = [tmp_path / name for name in ["adapter", "base", "text"]]
    for directory in [adapter, base, text]:
        directory.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": str(base)}))
    vae = tmp_path / "vae.safetensors"
    vae.write_bytes(b"vae")
    (base / "model.safetensors").write_bytes(b"base")
    (text / "model.safetensors").write_bytes(b"text")
    (adapter / "config.json").write_text(json.dumps({"video_vae_id": str(vae), "text_encoder_id": str(text)}))
    assert set(fit.checkpoint_inputs(adapter)) == {vae, base / "model.safetensors", text / "model.safetensors"}
    vae.unlink()
    with pytest.raises(ValueError, match="FIT_ENCODER"):
        fit.checkpoint_inputs(adapter)


def test_pinned_hub_encoder_references_resolve_only_from_offline_cache(tmp_path, monkeypatch):
    module = load_script("run_fitting")
    calls = []

    def snapshot(**kw):
        calls.append(kw)
        return str(tmp_path)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot))
    revision = "a" * 40
    path = module.resolve_cached_encoder(f"org/model:text_encoder@{revision}")
    assert path == tmp_path / "text_encoder", "[BUG-2 REGRESSION] saved checkpoints use pinned Hub references"
    assert calls == [{"repo_id": "org/model", "revision": revision, "local_files_only": True,
                      "allow_patterns": ["text_encoder", "text_encoder/**"]}]
    with pytest.raises(ValueError, match="FIT_ENCODER_PIN"):
        module.resolve_cached_encoder("org/model:text_encoder@main")
