#!/usr/bin/env bash
# Sim track on the 4500 pod: A0r-sim package (sim quantiles) + LoRA on the scripted "put the block in the cup" demos.
#
#   bash /workspace/liminal/ws4/ttt/pod_sim_lora.sh package          # quantiles + renormalized package (minutes)
#   bash /workspace/liminal/ws4/ttt/pod_sim_lora.sh probe            # 10 updates: step time and peak memory
#   STEPS=3200 bash /workspace/liminal/ws4/ttt/pod_sim_lora.sh train # the run (nohup it; log in $OUT.log)
#   bash /workspace/liminal/ws4/ttt/pod_sim_lora.sh merge            # adapter folded into a servable package
#
# Paths (override with env): DATA=/workspace/ttt/block_cup/sim_block_cup_left, PKG=/workspace/ttt/packages/so101sim_cup_left,
# OUT=/workspace/ttt/lora_sim_block_cup, MERGED=/workspace/ttt/packages/so101sim_cup_lora. LoRA settings: LeRobot
# examples/flux3/lora.json (rank/alpha 32, effective batch 8, lr 1e-4, heads x5). Changed here: the batch split (below),
# the budget, and the EMA decay: lora.json's 0.999 suits its 2,500 updates, while at 800 updates the shadow would still
# hold about 45% of the initial weights, so EMA=0.995 (openpi uses 0.99). Needs the GPU to itself: stop the policy
# server first (it holds about 25 GiB of the 32). flux.remote serves plain Flux3Policy packages, hence `merge`.
set -euo pipefail
[ -f /etc/rp_environment ] && source /etc/rp_environment
WS4_ROOT="${WS4_ROOT:-/workspace/ws4}"
LIMINAL="${LIMINAL:-/workspace/liminal}"
DATA="${DATA:-/workspace/ttt/block_cup/sim_block_cup_left}"
PKG="${PKG:-/workspace/ttt/packages/so101sim_cup_left}"
OUT="${OUT:-/workspace/ttt/lora_sim_block_cup}"
# LeRobot counts steps, save_freq and eval_steps in microsteps (optimizer updates = steps / ACCUM); keep them
# divisible by ACCUM. With the text encoder on the GPU the 32 GB card runs out of memory in the video VAE encode at
# batch 2 and at batch 1 (31 GiB in use), hence TE_CPU below. Two DataLoader workers: each spawn re-imports torch from
# the network volume (about 3.5 min, one after the other), and two keep up with about one sample per second.
BATCH="${BATCH:-2}"
ACCUM="${ACCUM:-4}"
WORKERS="${WORKERS:-2}"
STEPS="${STEPS:-3200}"
SAVE="${SAVE:-800}"
EVAL_STEPS="${EVAL_STEPS:-800}"
EMA="${EMA:-0.995}"
MERGED="${MERGED:-/workspace/ttt/packages/so101sim_cup_lora}"
source "$WS4_ROOT/lerobot/.venv/bin/activate"
export PYTHONPATH="$LIMINAL/ws4"
mkdir -p "$(dirname "$OUT")" /workspace/logs

case "${1:-}" in
package)
  python - <<EOF
from flux.package_tools import compute_quantiles, dataset_arrays, describe_stats, renormalize_package
a = dataset_arrays("local/sim_block_cup_left", root="$DATA")
stats = compute_quantiles(a["action"], a["observation.state"], a["episode_index"], absolute_dims=(-1,))
print(describe_stats(stats, ["pan", "lift", "elbow", "wflex", "roll", "grip"]))
print(renormalize_package("black-forest-labs/flux-3-action-so101", "$PKG", stats))
EOF
  ;;
probe|train)
  steps="$STEPS"; out="$OUT"
  if [ "$1" = probe ]; then steps=$((10 * ACCUM)); out="${OUT}_probe"; rm -rf "$out"; fi
  cd "$WS4_ROOT/lerobot"
  # TE_CPU=1 (default): ttt/lora_train_te_cpu.py, i.e. lerobot_train with the text encoder on the CPU.
  if [ "${TE_CPU:-1}" = 1 ]; then train=("$LIMINAL/ws4/ttt/lora_train_te_cpu.py"); else train=(-m lerobot.scripts.lerobot_train); fi
  python "${train[@]}" --config_path=examples/flux3/lora.json \
    --policy.path="$PKG" --policy.device=cuda \
    --dataset.repo_id=local/sim_block_cup_left --dataset.root="$DATA" \
    --batch_size="$BATCH" --accelerator.gradient_accumulation.steps="$ACCUM" --num_workers="$WORKERS" \
    --steps="$steps" --save_freq="$SAVE" --eval_steps="$EVAL_STEPS" --log_freq="$ACCUM" --ema.decay="$EMA" \
    --output_dir="$out" 2>&1 | tee "$out.log"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader
  ;;
merge)
  # CKPT: .../pretrained_model (raw adapter) or .../pretrained_model_ema; LeRobot saves both at every checkpoint.
  CKPT="${CKPT:-$OUT/checkpoints/last/pretrained_model_ema}"
  python - "$CKPT" "$PKG" "$MERGED" "${MERGE_DEVICE:-cuda}" <<'EOF'
import shutil
import sys
from pathlib import Path

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.flux3 import Flux3Policy
from peft import PeftModel

ckpt, base, dst = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
cfg = PreTrainedConfig.from_pretrained(base)
cfg.device = sys.argv[4]  # cuda once training has ended: the 62 GB pod RAM would hold the 7B weights twice
merged = PeftModel.from_pretrained(Flux3Policy.from_pretrained(base, config=cfg), str(ckpt)).merge_and_unload()
merged.save_pretrained(dst)
for f in [*ckpt.glob("policy_*"), Path(base) / "normalization_source.json"]:  # the processors training used
    if f.exists():
        shutil.copy(f, dst / f.name)
print("merged", ckpt, "->", dst)
EOF
  ;;
*)
  echo "usage: $0 package|probe|train|merge" >&2; exit 2;;
esac
