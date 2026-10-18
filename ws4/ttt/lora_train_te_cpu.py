"""``lerobot_train`` with FLUX 3 Action's frozen Qwen3-VL text encoder kept on the CPU (8.3 GiB less on the GPU).

On the 32 GB RTX PRO 4500 the LoRA preset runs out of memory in the video VAE encode after a few steps (base load
25.6 GiB, of which 8.3 GiB is the text encoder). The policy caches each caption's context on the GPU
(``Flux3Policy._context``) and a single-task dataset has two captions (the task and the caption-dropout blank),
so the encoder runs twice on the CPU and never again. LeRobot's docs give the same move for inference
(``policy.frozen.text_encoder.cpu()``); here it also has to survive accelerate's ``.to(device)``, which reaches the
frozen components through ``FrozenComponents._apply``, so the encoder is left out of that walk.

    python ws4/ttt/lora_train_te_cpu.py --config_path=examples/flux3/lora.json ...   # same arguments as lerobot_train
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

# Run as a script, this file's directory (ws4/ttt: labels.py, features.py, ...) would shadow top-level imports.
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != Path(__file__).resolve().parent]

from lerobot.policies.flux3 import modeling_flux3 as _mf  # noqa: E402

_modules = _mf.FrozenComponents.modules
_init = _mf.Flux3Policy.__init__


def _modules_without_text_encoder(self):
    for module in _modules(self):
        if not (getattr(self, "text_encoder_on_cpu", False) and module is self.text_encoder):
            yield module


def _init_text_encoder_on_cpu(self, *args, **kwargs):
    _init(self, *args, **kwargs)
    self.frozen.text_encoder.cpu()
    object.__setattr__(self.frozen, "text_encoder_on_cpu", True)


_mf.FrozenComponents.modules = _modules_without_text_encoder
_mf.Flux3Policy.__init__ = _init_text_encoder_on_cpu

if __name__ == "__main__":
    sys.argv[0] = "lerobot_train"
    runpy.run_module("lerobot.scripts.lerobot_train", run_name="__main__", alter_sys=True)
