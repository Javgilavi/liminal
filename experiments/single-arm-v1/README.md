# Single-arm validation v1

Read [the validation report](../../docs/validation/single-arm-v1.md) first.

| File | Purpose |
|---|---|
| `manifest.json` | Revisions, hashes, scope and retained raw-artifact locations |
| `simulation-results.json` | Original corrected native/HF/20 Hz results, checks and per-seed outcomes |
| `real-robot-results.json` | Derived official-policy trial index with source-log hashes; no task-success inference |
| `legacy-run-audit.json` | Early standalone-run chunk diagnostics and source hashes; superseded controller behavior |
| `train.json` | Recorded rank-32/1,000-microbatch recipe with portable output paths |

Reproduce using the [simulation guide](../../docs/simulation-validation.md) or
[real-robot guide](../../docs/official-inference.md). `prepare_experiment.py` resolves
training dataset/output paths and the prepared base for a new run. The JSON
results retain original machine paths as provenance; they are not portable input
paths or hosted download links.

Raw datasets, weights, optimizer states, complete logs and videos are deliberately
outside Git. Their recorded locations require access to the experiment machines.
An artifact bundle can be published separately; none was uploaded as part of this
commit group. Rebuilding the real trial index is read-only:

```bash
python scripts/summarize_real_runs.py \
  --source ~/.local/share/liminal/inference \
  --output outputs/real-robot-results.json
```

Keep this validation snapshot unchanged when collecting later trials. Use a new
validation directory for new evidence and claim only independently checked task
outcomes.
