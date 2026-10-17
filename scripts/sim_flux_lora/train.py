"""Launch the unchanged official LeRobot trainer with a process memory limit."""

from __future__ import annotations

import json
import os
import runpy
import time
from pathlib import Path


def main():
    import torch

    torch.set_num_threads(8)
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("SIM_LORA_MEMORY_FRACTION", ".42")))
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        runpy.run_module("lerobot.scripts.lerobot_train", run_name="__main__")
    finally:
        if destination := os.environ.get("SIM_LORA_METRICS_FILE"):
            metrics = {
                "wall_seconds": time.perf_counter() - start,
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
            Path(destination).write_text(json.dumps(metrics, indent=2) + "\n")


if __name__ == "__main__":
    main()
