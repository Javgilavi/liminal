#!/usr/bin/env bash
# Explicit simulation-only reproduction. Collection, training and evaluation can take hours.
set -euo pipefail
if [ "$#" -ne 3 ]; then
  echo "Usage: $0 OUTPUT_ROOT RELEASED_SO101_PACKAGE MENAGERIE_CHECKOUT" >&2
  exit 2
fi
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
experiment_root=$(realpath -m "$1")
base_package=$(realpath "$2")
menagerie_checkout=$(realpath "$3")
sim_python=${SIM_PYTHON:-python}
sim_scripts="$PWD/scripts/sim_flux_lora"
export MUJOCO_GL=${MUJOCO_GL:-egl}
"$sim_python" "$sim_scripts/prepare_experiment.py" --base "$base_package" --root "$experiment_root"
trap 'code=$?; printf "%s\n" "$code" > "$experiment_root/pipeline.exit"' EXIT
"$sim_python" "$sim_scripts/prepare_scene.py" --menagerie "$menagerie_checkout" --output "$experiment_root/scene"
export LIMINAL_SIM_ROOT="$experiment_root/scene"
for split in train validation; do
  count=64
  seed=1000
  if [ "$split" = validation ]; then count=16; seed=10000; fi
  echo "Collecting $count successful $split demonstrations"
  "$sim_python" -u "$sim_scripts/generate.py" --output "$experiment_root/raw/$split" \
    --episodes "$count" --start-seed "$seed" --render > "$experiment_root/collection-$split.log" 2>&1
done
"$sim_python" "$sim_scripts/validate_dataset.py" --raw "$experiment_root/raw" \
  --output "$experiment_root/collection-inference-audit.json"
"$sim_python" -u "$sim_scripts/export_dataset.py" --train "$experiment_root/raw/train" \
  --validation "$experiment_root/raw/validation" --output "$experiment_root/dataset-native-v3" \
  --repo-id local/liminal_so101_native_hf_pick_place --frame native --fps 20 \
  --normalization-description 'HF simulator statistics saved before training' > "$experiment_root/export.log" 2>&1
"$sim_python" "$sim_scripts/validate_native_export.py" --root "$experiment_root"
echo "Training fresh rank-32 LoRA for 1,000 microbatches"
SIM_LORA_METRICS_FILE="$experiment_root/training-metrics.json" "$sim_python" -u "$sim_scripts/train.py" \
  --config_path="$experiment_root/train-config.json" --policy.path="$experiment_root/base-native-hf" \
  > "$experiment_root/training.log" 2>&1
for kind in zeroshot raw ema; do
  checkpoint="$experiment_root/base-native-hf"
  if [ "$kind" = raw ]; then checkpoint="$experiment_root/training-1000/checkpoints/001000/pretrained_model"; fi
  if [ "$kind" = ema ]; then checkpoint="$experiment_root/training-1000/checkpoints/001000/pretrained_model_ema"; fi
  echo "Evaluating $kind on five held-out scenes"
  "$sim_python" -u "$sim_scripts/evaluate.py" --checkpoint "$checkpoint" \
    --dataset "$experiment_root/dataset-native-v3" --repo-id local/liminal_so101_native_hf_pick_place \
    --output "$experiment_root/evaluation-$kind" --episodes 5 --ticks 600 --frame native \
    > "$experiment_root/evaluation-$kind.log" 2>&1
done
"$sim_python" "$sim_scripts/summarize_consistent.py" --root "$experiment_root"
echo "Results: $experiment_root/RESULTS.md"
