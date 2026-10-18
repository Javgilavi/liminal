"""``flux.remote`` that also serves LoRA checkpoints: an adapter directory is merged into its base at load time.

A merged package would be a second full copy of the 7B weights, which the 100 GB sim volume cannot hold. LeRobot
saves each LoRA checkpoint (``pretrained_model``, ``pretrained_model_ema``) as a PEFT adapter whose
``adapter_config.json`` names the base package, next to the processors the training used. Here
``Flux3Policy.from_pretrained`` on such a directory loads that base, applies the adapter and merges it in memory
(``merge_and_unload``); ``LeRobotFlux3Backend`` then reads the processors from the adapter directory as usual, so
the served contract is the trained one. Plain packages load unchanged.

    python ws4/ttt/serve_lora.py --package cuplora=<checkpoint>/pretrained_model --token-file ... (flux.remote args)
"""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

# Run as a script, this file's directory (ws4/ttt: labels.py, features.py, ...) would shadow top-level imports.
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]

from lerobot.policies.flux3 import Flux3Policy  # noqa: E402

_from_pretrained = Flux3Policy.from_pretrained.__func__


def _from_pretrained_or_adapter(cls, pretrained_name_or_path, *args, **kwargs):
    adapter = Path(str(pretrained_name_or_path))
    config = adapter / "adapter_config.json"
    if not config.is_file():
        return _from_pretrained(cls, pretrained_name_or_path, *args, **kwargs)
    from peft import PeftModel

    base = json.loads(config.read_text())["base_model_name_or_path"]
    policy = _from_pretrained(cls, base, *args, **kwargs)
    merged = PeftModel.from_pretrained(policy, str(adapter)).merge_and_unload()
    print(f"serve_lora: merged {adapter} into {base}", flush=True)
    return merged


Flux3Policy.from_pretrained = classmethod(_from_pretrained_or_adapter)

if __name__ == "__main__":
    sys.argv[0] = "flux.remote"
    runpy.run_module("flux.remote", run_name="__main__", alter_sys=True)
