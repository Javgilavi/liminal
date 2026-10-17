#!/usr/bin/env bash
# Real track on real-train-6000 (RTX PRO 6000 96 GB): LoRA on the Haller "put the red pen in its slot in the cardboard
# box" VR demos, robot-left task arm, as a per-arm FLUX SO-101 dataset.
#
#   bash /workspace/liminal/ws4/cloud/real_lora.sh package STATS.json   # base package with A0r quantiles (minutes)
#   bash /workspace/liminal/ws4/cloud/real_lora.sh probe                # 10 updates: startup, step time, peak memory
#   STEPS=20000 setsid nohup bash /workspace/liminal/ws4/cloud/real_lora.sh train > /dev/null 2>&1 < /dev/null &
#   CKPT=.../checkpoints/010000/pretrained_model_ema bash .../real_lora.sh merge   # adapter folded into a package
#
# STATS.json: ws4/single_arm/a0r_stats.py output (its state / action keys), computed on the training split. Variant D
# (shoulder_lift and wrist_roll negated) needs the training data negated the same way, and the runner too.
# Paths (override with env): DATA=/workspace/real_lora/pen_in_right, REPO=local/pen_in_right,
# PKG=/workspace/packages/so101_haller_pen, OUT=/workspace/real_lora/lora_pen, MERGED=$PKG_lora.
# LoRA settings: LeRobot examples/flux3/lora.json (rank/alpha 32, effective batch 8, lr 1e-4, heads x5, EMA 0.999,
# checkpoints every 500 microsteps with raw and EMA weights). LeRobot counts steps, save_freq and eval_steps in
# microsteps (updates = steps / ACCUM). On 96 GB the text encoder stays on the GPU (TE_CPU=0; the 32 GB card needed
# ws4/ttt/lora_train_te_cpu.py). The flux3 history normalizers come from the package's saved processor files:
# lerobot_train's dataset-stats overrides only reach the generic normalizer steps.
# Split: TRAIN_EPS and EVAL_EPS hold space-separated episode indices (dropped takes in neither). LeRobot keeps the
# order of --dataset.episodes and holds out the last ceil(n * eval_split) episodes per task, so the eval episodes go
# last with eval_split = (n_eval - 0.5) / n (n_eval / n can round up one episode: 30 * 0.1 = 3.0000000000000004).
# The eval loss then runs on the held-out takes, which are never trained on. Without TRAIN_EPS: every episode,
# EVAL_SPLIT as given.
set -euo pipefail
[ -f /etc/rp_environment ] && source /etc/rp_environment
WS4_ROOT="${WS4_ROOT:-/workspace/ws4}"
LIMINAL="${LIMINAL:-/workspace/liminal}"
DATA="${DATA:-/workspace/real_lora/pen_in_right}"
REPO="${REPO:-local/pen_in_right}"
PKG="${PKG:-/workspace/packages/so101_haller_pen}"
OUT="${OUT:-/workspace/real_lora/lora_pen}"
MERGED="${MERGED:-${PKG}_lora}"
BASE="${BASE:-black-forest-labs/flux-3-action-so101}"
BATCH="${BATCH:-2}"
ACCUM="${ACCUM:-4}"
WORKERS="${WORKERS:-4}"
MP_CONTEXT="${MP_CONTEXT:-spawn}"
STEPS="${STEPS:-10000}"
SAVE="${SAVE:-500}"
EVAL_STEPS="${EVAL_STEPS:-500}"
EVAL_SPLIT="${EVAL_SPLIT:-0.1}"
MAX_EVAL="${MAX_EVAL:-256}"   # eval-loss samples per eval pass (0 = the whole held-out set)
EMA="${EMA:-0.999}"
source "$WS4_ROOT/lerobot/.venv/bin/activate"
export PYTHONPATH="$LIMINAL/ws4"
mkdir -p "$(dirname "$OUT")" "$(dirname "$PKG")" /workspace/logs

case "${1:-}" in
package)
  STATS="${2:?usage: $0 package STATS.json}"
  HF_HUB_OFFLINE=1 python - "$BASE" "$PKG" "$STATS" <<'EOF'
import json
import sys

from flux.package_tools import describe_stats, renormalize_package

base, dst, path = sys.argv[1:]
data = json.load(open(path))
stats = {k: data[k] for k in ("state", "action")}
print(describe_stats(stats, ["pan", "lift", "elbow", "wflex", "roll", "grip"]))
print(renormalize_package(base, dst, stats))
EOF
  ;;
probe|train)
  steps="$STEPS"; out="$OUT"
  if [ "$1" = probe ]; then steps=$((10 * ACCUM)); out="${OUT}_probe_b${BATCH}"; rm -rf "$out"; fi
  # GPU memory every second, for the peak (training logs only report the allocator's view).
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 1 > "$out.gpu.csv" &
  smi=$!
  # A checkpoint is 0.45 GB raw + 0.45 GB EMA + 1.4 GB training state. Keep both weight sets of every checkpoint
  # (for picking the best one later) but the training state only on the newest (enough to resume).
  pruner=""
  if [ "$1" = train ]; then
    ( while sleep 300; do
        find "$out/checkpoints" -mindepth 1 -maxdepth 1 -type d -name '[0-9]*' 2>/dev/null | sort | head -n -1 |
          while read -r d; do rm -rf "$d/training_state"; done
      done ) &
    pruner=$!
  fi
  trap 'kill $smi $pruner 2>/dev/null' EXIT
  if [ "${TE_CPU:-0}" = 1 ]; then train=("$LIMINAL/ws4/ttt/lora_train_te_cpu.py"); else train=(-m lerobot.scripts.lerobot_train); fi
  ds_args=(--dataset.eval_split="$EVAL_SPLIT")
  if [ -n "${TRAIN_EPS:-}" ]; then
    read -r -a tr <<< "$TRAIN_EPS"
    read -r -a ev <<< "${EVAL_EPS:-}"
    all=("${tr[@]}" "${ev[@]}")
    split=$(python -c "print((${#ev[@]} - 0.5) / ${#all[@]} if ${#ev[@]} else 0.0)")
    ds_args=(--dataset.episodes="[$(IFS=,; echo "${all[*]}")]" --dataset.eval_split="$split")
  fi
  cd "$WS4_ROOT/lerobot"
  echo "start $(date -u +%FT%TZ) steps=$steps batch=$BATCH accum=$ACCUM workers=$WORKERS ($MP_CONTEXT) pkg=$PKG data=$DATA ${ds_args[*]}"
  python "${train[@]}" --config_path=examples/flux3/lora.json \
    --policy.path="$PKG" --policy.device=cuda \
    --dataset.repo_id="$REPO" --dataset.root="$DATA" "${ds_args[@]}" \
    --batch_size="$BATCH" --accelerator.gradient_accumulation.steps="$ACCUM" \
    --num_workers="$WORKERS" --dataloader_multiprocessing_context="$MP_CONTEXT" \
    --steps="$steps" --save_freq="$SAVE" --eval_steps="$EVAL_STEPS" --max_eval_samples="$MAX_EVAL" --log_freq="$ACCUM" --ema.decay="$EMA" \
    --output_dir="$out" 2>&1 | tee "$out.log"
  echo "end $(date -u +%FT%TZ); peak GPU memory $(cut -d, -f2 "$out.gpu.csv" | sort -n | tail -1)" | tee -a "$out.log"
  ;;
merge)
  CKPT="${CKPT:-$OUT/checkpoints/last/pretrained_model_ema}"
  python - "$CKPT" "$PKG" "$MERGED" <<'EOF'
import shutil
import sys
from pathlib import Path

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.flux3 import Flux3Policy
from peft import PeftModel

ckpt, base, dst = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
cfg = PreTrainedConfig.from_pretrained(base)
cfg.device = "cpu"
merged = PeftModel.from_pretrained(Flux3Policy.from_pretrained(base, config=cfg), str(ckpt)).merge_and_unload()
merged.save_pretrained(dst)
for f in [*ckpt.glob("policy_*"), Path(base) / "normalization_source.json"]:  # the processors training used
    if f.exists():
        shutil.copy(f, dst / f.name)
print("merged", ckpt, "->", dst)
EOF
  ;;
*)
  echo "usage: $0 package STATS.json | probe | train | merge" >&2; exit 2;;
esac
