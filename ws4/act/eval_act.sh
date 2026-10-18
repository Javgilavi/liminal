#!/usr/bin/env bash
# Held-out action error of ACT checkpoints on the pod (offline_eval.py = offline_diag.py's protocol on the 12
# held-out takes: 200 frames, seed 0, first 32 commands).
#
#   bash ws4/act/eval_act.sh RUN LABEL STEP [STEP ...]    # e.g. act_pen84 curve 020000 050000 080000
#
# Writes /workspace/act/eval/RUN_LABEL.json (+ .npz). Pen defaults; skill_act.sh passes ROOT and EXCLUDE.
set -euo pipefail
source /etc/rp_environment
source "$WS4_ROOT/lerobot/.venv/bin/activate"
export HF_HUB_OFFLINE=1 PYTHONPATH=/workspace/act/ws4
run="${1:?run}" label="${2:?label}"
shift 2
root="${ROOT:-/workspace/real_lora/pen_in_right_final}"
if [ -z "${EXCLUDE:-}" ]; then
  source /workspace/real_lora/split_final.env  # EVAL_EPS
  EXCLUDE=$(python -c "import sys; held = set(map(int, sys.argv[1].split())); print(*[e for e in range(104) if e not in held])" "$EVAL_EPS")
fi
ckpts=()
for step in "$@"; do
  ckpts+=(--checkpoint "act_$((10#$step / 1000))k=/workspace/act/runs/$run/checkpoints/$step/pretrained_model")
done
mkdir -p /workspace/act/eval
python /workspace/act/ws4/act/offline_eval.py --root "$root" --repo-id "local/$(basename "$root")" \
  --task "${TASK:-put the red pen in its slot in the cardboard box}" --samples 200 --seed 0 --exclude $EXCLUDE "${ckpts[@]}" \
  --out "/workspace/act/eval/${run}_${label}.json"
echo "ACT_EVAL_DONE ${run}_${label}"
