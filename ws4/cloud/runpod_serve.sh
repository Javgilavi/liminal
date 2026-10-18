#!/usr/bin/env bash
# Serve the FLUX policy on a RunPod inference pod and stop the pod once it has been idle.
# Run once per pod start, after ws4/cloud/bootstrap.sh has finished once on the network volume
# (see ws4/cloud/RUNPOD.md). From the laptop:
#
#   ssh root@<pod-ip> -p <port> 'nohup bash /workspace/liminal/ws4/cloud/runpod_serve.sh >/workspace/logs/serve.out 2>&1 &'
#   ssh root@<pod-ip> -p <port> cat /workspace/flux_token          # the token the robot runner needs
#   ssh -N -L 8765:127.0.0.1:8765 root@<pod-ip> -p <port> &      # then the robot talks to http://127.0.0.1:8765
#
# Env (the pod template sets WS4_ROOT and LIMINAL; the rest is optional):
#   PACKAGES   space-separated NAME=CHECKPOINT, default "so101=black-forest-labs/flux-3-action-so101". LoRA
#              adapters from the teammate's box: push them to a private HF repo readable by this pod's HF_TOKEN
#              and add e.g. "B=ORG/xlerobot-lora-b".
#   IDLE_MIN   minutes of idle GPU before the pod stops itself (default 20; 0 disables the watchdog)
#   PROMPTS    file of skill prompts to pre-encode at startup, one per line (default /workspace/prompts.txt,
#              seeded from ws4/cloud/warm_prompts.py DEFAULT_PROMPTS). Must match the runner's strings exactly.
#   EXTRA_ARGS extra flux.remote flags. On a GPU under 40 GB the text encoder goes to the CPU automatically
#              (Qwen3-VL is 8.3 GiB; the SO-101 policy peaks at 23.6 GiB without it on the 32 GB pod).
set -euo pipefail

# Non-interactive SSH sessions do not get the pod's environment variables (HF_TOKEN, HF_HOME, ...) otherwise.
# shellcheck disable=SC1091
[ -f /etc/rp_environment ] && source /etc/rp_environment

WS4_ROOT="${WS4_ROOT:-/workspace/ws4}"
LIMINAL="${LIMINAL:-/workspace/liminal}"
PACKAGES="${PACKAGES:-so101=black-forest-labs/flux-3-action-so101}"
IDLE_MIN="${IDLE_MIN:-20}"
LOGS=/workspace/logs
mkdir -p "$LOGS"

if pgrep -f "^python -m flux.remote" >/dev/null; then  # anchored: never match a shell whose command line mentions it
  echo "policy server already running"
  exit 0
fi

# Token for the robot runner, kept on the volume so it survives pod restarts (readable only by root).
if [ ! -s /workspace/flux_token ]; then
  (umask 077 && openssl rand -hex 16 > /workspace/flux_token)
fi

# shellcheck disable=SC1091
source "$WS4_ROOT/lerobot/.venv/bin/activate"
export PYTHONPATH="$LIMINAL/ws4"
args=()
for p in $PACKAGES; do args+=(--package "$p"); done

mem_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -1)
if [ "${mem_mib:-0}" -lt 40000 ]; then
  args+=(--text-encoder-on-cpu)
fi

PROMPTS="${PROMPTS:-/workspace/prompts.txt}"
if [ ! -s "$PROMPTS" ]; then
  python -c "import sys; sys.path.insert(0, '$LIMINAL/ws4/cloud'); import warm_prompts; print(*warm_prompts.DEFAULT_PROMPTS, sep='\n')" > "$PROMPTS"
fi

if [ "$IDLE_MIN" != 0 ]; then
  nohup bash "$LIMINAL/ws4/cloud/idle_stop.sh" "$IDLE_MIN" >> "$LOGS/idle_stop.log" 2>&1 &
fi
echo "starting flux.remote with ${args[*]} (log: $LOGS/flux_remote.log)"
# shellcheck disable=SC2086
python -m flux.remote --host 127.0.0.1 --port 8765 --token-file /workspace/flux_token "${args[@]}" ${EXTRA_ARGS:-} \
  >> "$LOGS/flux_remote.log" 2>&1 &
server=$!

# Encode every skill prompt once, before the robot connects (a first-time CPU encode takes about 16 s).
python "$LIMINAL/ws4/cloud/warm_prompts.py" --prompts-file "$PROMPTS" --token-file /workspace/flux_token \
  >> "$LOGS/warm_prompts.log" 2>&1 && echo "prompts warmed (log: $LOGS/warm_prompts.log)" \
  || echo "WARNING: prompt warm-up failed, see $LOGS/warm_prompts.log"
echo "READY: policy server on 127.0.0.1:8765"
wait "$server"
