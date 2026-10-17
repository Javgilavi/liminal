#!/usr/bin/env bash
# Stop this pod once neither the GPU nor a ws4/ttt job has been busy for IDLE_MIN minutes (default 30).
#
#   source /etc/rp_environment; setsid nohup bash /workspace/liminal/ws4/ttt/pod_job_watchdog.sh 30 \
#       >> /workspace/logs/job_watchdog.log 2>&1 < /dev/null &
#
# ws4/cloud/idle_stop.sh samples GPU utilization only, every 30 s. A LoRA run spends about 15 min loading the model
# and spawning DataLoader workers at 0% (torch imports from the network volume), and a short probe's burst can fall
# between two samples, so it stopped the pod twice on 17 Oct. Here a running lerobot_train, eval_block_cup.py,
# cup_stages.py, pod_sim_lora.sh merge or ws4/cloud/bootstrap.sh (a fresh volume's CPU-only setup, about 45 min)
# also counts as busy; WATCH_JOBS replaces the list. Same stop call as idle_stop.sh (runpodctl 1.x).
set -uo pipefail
IDLE_SEC=$(( ${1:-30} * 60 ))
POD_ID="${RUNPOD_POD_ID:-}"
JOBS="${WATCH_JOBS:-lerobot_train|lora_train_te_cpu|eval_block_cup|cup_stages|pod_sim_lora.sh merge|bootstrap.sh}"
log() { printf '%s job_watchdog: %s\n' "$(date +%H:%M:%S)" "$*"; }
[ -n "$POD_ID" ] || { log "RUNPOD_POD_ID is not set, nothing to stop"; exit 1; }
last_busy=$(date +%s)
log "stopping pod $POD_ID after ${IDLE_SEC}s without GPU use above 5% or a job matching: $JOBS"
while sleep 15; do
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
  if { [ -n "$util" ] && [ "$util" -gt 5 ]; } || pgrep -f "$JOBS" | grep -qv "^$$\$"; then
    last_busy=$(date +%s)
  fi
  if [ $(( $(date +%s) - last_busy )) -ge "$IDLE_SEC" ]; then
    log "idle for $(( $(date +%s) - last_busy ))s, stopping pod $POD_ID"
    if runpodctl stop pod --help >/dev/null 2>&1; then runpodctl stop pod "$POD_ID"; else runpodctl pod stop "$POD_ID"; fi
    exit 0
  fi
done
