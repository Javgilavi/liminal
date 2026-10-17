#!/usr/bin/env bash
# LeRobot ACT baseline on the Haller pen demos (84 train takes of ws4_split.json "final"), on a pod with the
# real-train-is volume at /workspace. Code copy and outputs live under /workspace/act/ only.
#
#   bash ws4/act/train_act.sh view                 # train-only stats view of pen_in_right_final (train_view.py)
#   bash ws4/act/train_act.sh probe NAME STEPS     # short run for the step time
#   bash ws4/act/train_act.sh train NAME STEPS     # the run; checkpoints every SAVE_FREQ (10k)
#
# Env: BATCH (8), WORKERS (12), SAVE_FREQ (10000), LOG_FREQ (200). LeRobot ACT defaults otherwise: ResNet18
# ImageNet backbone, chunk_size 100, n_action_steps 100, lr 1e-5 (backbone 1e-5), KL weight 10, VAE on.
set -euo pipefail
source /etc/rp_environment
source "$WS4_ROOT/lerobot/.venv/bin/activate"
source /workspace/real_lora/split_final.env  # TRAIN_EPS, EVAL_EPS, EXCLUDE
export HF_HUB_OFFLINE=1 PYTHONPATH=/workspace/act/ws4
VIEW=/workspace/act/data/pen84
mode="${1:?view|probe|train}"

if [ "$mode" = view ]; then
  python /workspace/act/ws4/act/train_view.py --src /workspace/real_lora/pen_in_right_final --dst "$VIEW" \
    --episodes $TRAIN_EPS
  exit 0
fi

name="${2:?run name}"
steps="${3:?steps}"
episodes="[$(echo $TRAIN_EPS | tr ' ' ',')]"
[ "$(echo $TRAIN_EPS | wc -w)" = 84 ] || { echo "expected 84 train episodes"; exit 1; }
out=/workspace/act/runs/$name
save_freq="${SAVE_FREQ:-10000}"
[ "$mode" = probe ] && save_freq=$steps
# Module path (not the lerobot-train entry point) so pod_job_watchdog.sh's "lerobot_train" pattern matches.
python -m lerobot.scripts.lerobot_train \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --dataset.repo_id=local/pen84 --dataset.root="$VIEW" --dataset.episodes="$episodes" \
  --batch_size="${BATCH:-8}" --num_workers="${WORKERS:-12}" --steps="$steps" \
  --save_freq="$save_freq" --log_freq="${LOG_FREQ:-200}" --save_checkpoint=true \
  --wandb.enable=false --seed=1000 --job_name="$name" --output_dir="$out"
