"""Check the pinned pen task contract without model downloads or hardware."""
from pathlib import Path

import yaml

from liminal.app import build_policy
from liminal.orchestration import load_task
from liminal.policy.checkpoints import Checkpoint


def test_pen_profile_preserves_trained_timing_and_dataset_normalization():
    cfg = build_policy("flux", yaml.safe_load(Path("configs/policy/pen_place_registry.yaml").read_text())).cfg
    entry = Checkpoint.parse("red_pen_raw_10000", cfg.checkpoints["red_pen_raw_10000"])
    entry.validate(cfg, cfg.midtraining_base, "left")
    assert entry.normalization_profile == "dataset_native"
    assert (cfg.control_hz, cfg.obs_history, cfg.chunk_size, cfg.execute_steps) == (30, 8, 42, 16)
    task = load_task("configs/tasks/pen_place.yaml")
    skill = task.tree.children[0].skill
    assert skill.prompt == "put the red pen in its slot in the cardboard box"
    assert cfg.skill_checkpoints[skill.name] == entry.name
    assert task.goal == skill.success == "red_pen_in_slot"
