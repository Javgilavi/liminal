#!/usr/bin/env bash
# Pod side of the per-skill FLUX LoRA pipeline (18 Oct): runs ON a GPU pod whose volume has the bootstrap envs and
# weights (ws4/cloud/bootstrap.sh) and a ws4/ sync at $LIMINAL/ws4. Works on any RunPod account.
#
#   bash skill_pod.sh hub   SKILL REPO REV SIDE   # dataset from the Hub -> /workspace/skills/SKILL/{raw,skill.env}
#   bash skill_pod.sh data  SKILL                 # per-arm conversion, B stats on the train takes, base package
#   bash skill_pod.sh train SKILL                 # free-space check, 30 min job watchdog, LoRA run in the background
#   bash skill_pod.sh gates SKILL [STEP]          # after training: (a) RAW + EMA held-out eval, (b)+(c) gate_check.py
#   bash skill_pod.sh all   SKILL                 # data, train, then gates when training has exited (detached)
#
# Recipe (18 Oct evidence: RAW 5k was best on the arm; the live failure was a wrist-view knife edge):
#   - base = released flux-3-action-so101 + B quantiles of the skill's train takes (BASE_PKG overrides it, e.g.
#     /workspace/packages/so101_haller_pen_B84 for the red pen);
#   - LeRobot examples/flux3/lora.json, STEPS=5000, SAVE=1000, AUGMENT=1 (flux3 95 % crop + colour jitter);
#   - output /workspace/skills/SKILL/RUN (RUN default lora_aug).
# Gates per skill, before any robot time: (a) offline RAW and EMA at STEP on the held-out takes; (b) from rest, the
# first chunk moves >= 20 deg (summed arm joints) on every held-out take; (c) the same under +-2 px shifts and 0.5x
# brightness. The winner is the variant that passes (b) and (c) with the better arm MAE.
set -uo pipefail
# shellcheck disable=SC1091
[ -f /etc/rp_environment ] && source /etc/rp_environment
WS4_ROOT="${WS4_ROOT:-/workspace/ws4}"; LIMINAL="${LIMINAL:-/workspace/liminal}"
# shellcheck disable=SC1091
source "$WS4_ROOT/lerobot/.venv/bin/activate" 2>/dev/null || { echo "no venv at $WS4_ROOT/lerobot (run bootstrap.sh)"; exit 1; }
export PYTHONPATH="$LIMINAL/ws4"
W=/workspace/skills
cmd="${1:?command}"; skill="${2:?skill}"
D="$W/$skill"; RUN="${RUN:-lora_aug}"; STEPS="${STEPS:-5000}"
BASE_PKG="${BASE_PKG:-/workspace/packages/so101_haller_${skill}_B}"
log() { printf '%s skill_pod %s: %s\n' "$(date -u +%H:%M:%S)" "$skill" "$*"; }
envfile() { [ -s "$D/skill.env" ] || { log "no $D/skill.env"; exit 1; }; source "$D/skill.env"; }

case "$cmd" in
hub)
  repo="${3:?REPO}"; rev="${4:?REVISION}"; side="${5:?SIDE left|right}"
  mkdir -p "$D"
  python - "$repo" "$rev" "$D/raw" <<'EOF'
import sys
from huggingface_hub import snapshot_download
repo, rev, dst = sys.argv[1:]
print(snapshot_download(repo, repo_type="dataset", revision=rev, local_dir=dst))
EOF
  python "$LIMINAL/ws4/cloud/skill_env.py" "$D/raw" "$side" "$skill" > "$D/skill.env" || exit 1
  cp "$D/raw/ws4_split.json" "$D/ws4_split_final.json"
  log "hub data ready: $(grep -c . "$D/skill.env") env lines, $(du -sh "$D/raw" | cut -f1)"
  ;;
data)
  envfile
  export HF_HUB_OFFLINE=1
  cd "$LIMINAL/ws4"
  [ -e "$D/data" ] || python data/convert_bimanual.py "$D/raw" "$D/data" --to per_arm --side "$SIDE" \
    --scene observation.images.head --wrist "observation.images.${SIDE}_wrist" --src-units degrees --task-suffix "" \
    | tail -3
  if [ -e "$BASE_PKG/config.json" ]; then
    log "base exists: $BASE_PKG"
  else
    # shellcheck disable=SC2086
    [ -s "$D/a0r_B_$skill.json" ] || python single_arm/a0r_stats.py --root "$D/data" --repo-id "local/$skill" \
      --exclude $STATS_EXCLUDE --out "$D/a0r_B_$skill.json" | tail -8
    PKG="$BASE_PKG" bash cloud/real_lora.sh package "$D/a0r_B_$skill.json" 2>&1 | grep -v -i warn | tail -2
  fi
  ;;
train)
  envfile
  [ -e "$D/$RUN" ] && { log "$D/$RUN exists: not starting twice"; exit 1; }
  used=$(du -s --block-size=1G /workspace 2>/dev/null | cut -f1)
  free=$(( ${QUOTA_GB:-150} - used ))
  log "volume: $used GB used of ${QUOTA_GB:-150}, $free GB free; a 5k run needs about 7 GB"
  [ $(( free - 7 )) -ge "${MIN_FREE_GB:-40}" ] || { log "NOT STARTING: would leave under ${MIN_FREE_GB:-40} GB free"; exit 1; }
  pgrep -f "^bash $LIMINAL/ws4/ttt/pod_job_watchdog.sh" > /dev/null || setsid nohup bash \
    "$LIMINAL/ws4/ttt/pod_job_watchdog.sh" 30 >> "/workspace/logs/job_watchdog_$skill.log" 2>&1 < /dev/null &
  mkdir -p /workspace/logs
  cd "$LIMINAL/ws4/cloud"
  PKG="$BASE_PKG" DATA="$D/data" REPO="local/$skill" OUT="$D/$RUN" STEPS="$STEPS" SAVE="${SAVE:-1000}" \
    AUGMENT="${AUGMENT:-1}" TRAIN_EPS="$TRAIN_EPS" EVAL_EPS="$EVAL_EPS" HF_HUB_OFFLINE=1 \
    setsid nohup bash real_lora.sh train > "/workspace/logs/train_${skill}_$RUN.out" 2>&1 < /dev/null &
  log "training started: $D/$RUN, $STEPS microsteps, augment ${AUGMENT:-1}, base $BASE_PKG"
  ;;
gates)
  envfile
  step="${3:-$STEPS}"; s6=$(printf %06d "$step")
  export HF_HUB_OFFLINE=1
  ckpt="$D/$RUN/checkpoints/$s6"
  until [ -f "$ckpt/pretrained_model_ema/adapter_model.safetensors" ]; do sleep 60; done
  while pgrep -f "^python -m lerobot.scripts.lerobot_train" > /dev/null; do sleep 60; done
  # The eval base: BASE_PKG itself, or the base an adapter was trained on when BASE_PKG is an adapter (resume).
  base="$BASE_PKG"
  [ -f "$BASE_PKG/adapter_config.json" ] && base=$(python -c "import json, sys; print(json.load(open(sys.argv[1]))['base_model_name_or_path'])" "$BASE_PKG/adapter_config.json")
  E="$D/${RUN}_gates"; mkdir -p "$E"
  cd "$LIMINAL/ws4"
  for v in raw ema; do
    a="$ckpt/pretrained_model"; [ "$v" = ema ] && a="$ckpt/pretrained_model_ema"
    log "(a) offline eval $v $step"
    # shellcheck disable=SC2086
    python cloud/lora_adapter.py eval --package "$base" --adapter "$a" -- --root "$D/data" --repo-id "local/$skill" \
      --task "$TASK" --variants A --samples 200 --seed 0 --text-encoder-on-cpu --exclude $EVAL_EXCLUDE \
      --out "$E/${v}_$step.json" > "$E/${v}_$step.log" 2>&1
    grep -A3 "^| variant" "$E/${v}_$step.log" | tail -1
    log "(b)+(c) from-rest gates $v $step"
    # shellcheck disable=SC2086
    python cloud/gate_check.py --package "$base" --adapter "$a" --root "$D/data" --episodes $EVAL_EPS --task "$TASK" \
      --out "$E/gate_${v}_$step.json" > "$E/gate_${v}_$step.log" 2>&1
    tail -1 "$E/gate_${v}_$step.log"
  done
  log "GATES DONE"
  ;;
all)
  bash "$0" data "$skill" || exit 1
  bash "$0" train "$skill" || exit 1
  setsid nohup bash "$0" gates "$skill" > "$D/${RUN}_gates.out" 2>&1 < /dev/null &
  log "gates armed: $D/${RUN}_gates.out"
  ;;
*)
  sed -n 2,20p "$0"; exit 2;;
esac
