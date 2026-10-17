#!/usr/bin/env bash
# Stop this RunPod Pod after its GPU has been idle for a while, so an inference pod costs nothing between
# robot sessions (a stopped Pod bills only storage: docs.runpod.io/pods/manage-pods).
#
#   bash ws4/cloud/idle_stop.sh [IDLE_MIN]        # default 20; started by runpod_serve.sh
#
# "Busy" = GPU utilization above BUSY_PCT (default 5) in any sample. The policy server runs one inference per
# action chunk (about one per second while the robot moves), so a robot session keeps the GPU busy and an idle
# server reads 0%. The clock starts at launch, which gives the robot IDLE_MIN minutes to connect.
# Needs the pod's own RUNPOD_API_KEY in the environment (source /etc/rp_environment in non-interactive shells).
# Test knobs: IDLE_SEC overrides IDLE_MIN, POLL_SEC (default 30), NVIDIA_SMI / RUNPODCTL point at fakes.
set -uo pipefail

IDLE_SEC="${IDLE_SEC:-$(( ${1:-20} * 60 ))}"
POLL_SEC="${POLL_SEC:-30}"
BUSY_PCT="${BUSY_PCT:-5}"
NVIDIA_SMI="${NVIDIA_SMI:-nvidia-smi}"
RUNPODCTL="${RUNPODCTL:-runpodctl}"
POD_ID="${RUNPOD_POD_ID:-}"

log() { printf '%s idle_stop: %s\n' "$(date +%H:%M:%S)" "$*"; }

# runpodctl 1.x (what RunPod images ship, e.g. 1.14.15) has `stop pod <id>`; the newer CLI in docs.runpod.io
# has `pod stop <id>`. On 1.14.15 the pod's own RUNPOD_API_KEY passes authorization for `stop pod` (probed with a
# non-existent id: "Attempted to stop pod that does not exist"), while `get pod` and the REST API return 403.
stop_pod() {
  if "$RUNPODCTL" stop pod --help >/dev/null 2>&1; then
    "$RUNPODCTL" stop pod "$POD_ID"
  else
    "$RUNPODCTL" pod stop "$POD_ID"
  fi
}

[ -n "$POD_ID" ] || { log "RUNPOD_POD_ID is not set: not on a RunPod pod, nothing to stop"; exit 1; }
last_busy=$(date +%s)
log "stopping pod $POD_ID after ${IDLE_SEC}s with GPU utilization <= ${BUSY_PCT}%"
while true; do
  util=$("$NVIDIA_SMI" --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
  now=$(date +%s)
  if [ -n "$util" ] && [ "$util" -gt "$BUSY_PCT" ]; then
    last_busy=$now
  fi
  if [ $(( now - last_busy )) -ge "$IDLE_SEC" ]; then
    log "idle for $(( now - last_busy ))s, stopping pod $POD_ID"
    if stop_pod; then
      exit 0
    fi
    log "WARNING: stopping the pod failed; stop it from the console or the RunPod MCP"
    last_busy=$now   # retry after another idle period instead of hammering the API
  fi
  sleep "$POLL_SEC"
done
