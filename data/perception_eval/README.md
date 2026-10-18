# Perception evaluation data

Keep synchronized camera frames under state-oriented directories such as:

```text
lid_open/
lid_closed/
cable_blocking/
cable_held_clear/
component_in_red_cup/
replacement_installed/
ambiguous/
```

Each scene should normally provide `top`, `left_wrist`, and `right_wrist` images
captured at the same time. Copy `labels.example.yaml` to `labels.yaml`, add one
frame entry per scene, and label every requested predicate as `true`, `false`, or
`unknown`. Use `unknown` when the ground-truth image set does not visibly establish
the predicate; this explicitly tests whether the model fails closed.

Run the evaluator with:

```bash
uv run liminal evaluate-perception data/perception_eval/labels.yaml \
  --perception configs/perception/qwen3_vl_4b.yaml \
  --repeats 3
```

The evaluator intentionally scores independent calls with consensus disabled.
Repeat agreement is reported separately, so unstable predictions are visible
instead of being hidden by the production confirmation mechanism.
