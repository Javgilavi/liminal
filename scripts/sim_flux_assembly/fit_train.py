"""Restrict the official trainer to audited anchors; record actual optimization evidence."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from fitting import fit_indices


class FixedSampler:
    def __init__(self, indices):
        self.indices = list(indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def update_accelerator(positional, keywords):
    return keywords["accelerator"] if "accelerator" in keywords else positional[5]


def main(args, trainer_args):
    import torch
    from lerobot.scripts import lerobot_train as trainer

    frozen = json.loads(args.windows.read_text())
    indices = fit_indices(frozen)
    original_sampler, original_update = trainer.EpisodeAwareSampler, trainer.update_policy
    evidence = {"microbatches": 0, "optimizer_updates": 0, "indices": indices, "gradient_samples": {}}
    initial, probes, handles = {}, {}, []

    def sampler(*a, **kw):
        official = original_sampler(*a, **kw)
        if not set(indices).issubset(official.indices):
            raise ValueError("FIT_SAMPLER: frozen anchors fall outside the official valid training windows")
        return FixedSampler(indices)

    def hook(name):
        def check(gradient):
            if not torch.isfinite(gradient).all():
                raise ValueError(f"FIT_GRADIENT: nonfinite gradient in {name}")
            value = float(gradient.detach().abs().max())
            prior = evidence["gradient_samples"].setdefault(name, {"calls": 0, "max_abs": 0.0})
            prior["calls"] += 1
            prior["max_abs"] = max(prior["max_abs"], value)
        return check

    def update(*a, **kw):
        policy, optimizer = a[1], a[3]
        if not initial:
            params = dict(policy.named_parameters())
            active = {id(p) for p in params.values() if p.requires_grad}
            optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
            if not active or not active.issubset(optimized):
                raise ValueError("FIT_OPTIMIZER: trainable parameters absent from optimizer")
            for label, text in [("adapter", "lora_B"), ("head", "modules_to_save.default")]:
                matches = [(n, p) for n, p in params.items() if p.requires_grad and text in n]
                if not matches:
                    raise ValueError(f"FIT_TRAINABLE: no fully trainable {label} parameters")
                name, parameter = matches[0]
                probes[name] = parameter
                initial[name] = parameter.detach().float().flatten()[:256].cpu().clone()
                handles.append(parameter.register_hook(hook(name)))
            evidence["trainable_parameters"] = sum(p.numel() for p in params.values() if p.requires_grad)
            evidence["optimizer_covers_all_trainable"] = True
            evidence["trainable_names"] = [n for n, p in params.items() if p.requires_grad]
        result = original_update(*a, **kw)
        evidence["microbatches"] += 1
        accelerator = update_accelerator(a, kw)
        if accelerator.sync_gradients and not accelerator.optimizer_step_was_skipped:
            evidence["optimizer_updates"] += 1
        if not torch.isfinite(torch.tensor(float(result[0].loss.avg))):
            raise ValueError("FIT_LOSS: nonfinite training loss")
        return result

    torch.set_num_threads(8)
    torch.cuda.set_per_process_memory_fraction(0.42)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    trainer.EpisodeAwareSampler, trainer.update_policy = sampler, update
    sys.argv = ["lerobot-train", *trainer_args]
    try:
        trainer.main()
        changes = {n: float((p.detach().float().flatten()[:256].cpu() - initial[n]).abs().max())
                   for n, p in probes.items()}
        evidence["parameter_probe_max_changes"] = changes
        if not changes or not all(v > 0 for v in changes.values()):
            raise ValueError("FIT_WEIGHTS: adapter/head probes did not both change")
        if not all(v["max_abs"] > 0 for v in evidence["gradient_samples"].values()):
            raise ValueError("FIT_GRADIENT: adapter/head probes did not receive nonzero gradients")
        evidence["completed"] = True
    finally:
        trainer.EpisodeAwareSampler, trainer.update_policy = original_sampler, original_update
        for handle in handles:
            handle.remove()
        evidence.update(wall_seconds=time.monotonic() - started,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved())
        args.evidence.write_text(json.dumps(evidence, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    arguments, remaining = parser.parse_known_args()
    if os.environ.get("WORLD_SIZE", "1") != "1":
        raise ValueError("FIT_TOPOLOGY: this bounded diagnostic supports one process only")
    main(arguments, remaining)
