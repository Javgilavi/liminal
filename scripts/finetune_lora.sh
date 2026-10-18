#!/usr/bin/env bash
# Historical placeholder, not the current training entrypoint. See
# docs/simulation-validation.md and docs/pen-checkpoint.md for implemented workflows.
# WS1: task-specific LoRA on top of the FLUX 3 Action SO-101 checkpoint.
#
# TODO(WS1): replace the command below with the official BFL LoRA recipe
# (planning notes: ~10k training microsteps in the published recipe, ~200 demos in
# BFL's own adaptation example - we have 50-100, so expect to trade steps vs. overfit
# and hold out ~10% of episodes for eval).
set -euo pipefail

DATASET="${DATASET:?e.g. your_hf_user/liminal_so101_pick_place_v1}"
OUT="${OUT:-outputs/lora/pick_place_v1}"
STEPS="${STEPS:-10000}"

echo "dataset=$DATASET out=$OUT steps=$STEPS"
echo "Not wired yet: fill in the BFL fine-tuning entrypoint here." >&2
exit 2
