#!/usr/bin/env bash
# Sim track on the 4500 pod: serve one package, run the cup-task closed loop against it on the pod (the token stays
# on the pod), stop serving. One 25 GiB model fits on the 32 GiB card, so packages are served one at a time.
#
#   bash /workspace/liminal/ws4/ttt/pod_sim_eval.sh stop                                  # free the GPU
#   bash /workspace/liminal/ws4/ttt/pod_sim_eval.sh serve so101simcup /workspace/ttt/packages/so101sim_cup_left
#   bash /workspace/liminal/ws4/ttt/pod_sim_eval.sh eval so101simcup                      # 20 layouts, log in /workspace/logs
#
# `serve` reuses ws4/cloud/runpod_serve.sh with IDLE_MIN=0 (the pod already runs its idle watchdog) and warms only
# the cup prompt. `eval` needs /workspace/venv-simeval (MuJoCo 3.8.1, as on the laptop) and the EGL loader (libegl1).
set -euo pipefail
LIMINAL="${LIMINAL:-/workspace/liminal}"
OUT="${OUT:-/workspace/ttt/block_cup_eval}"
mkdir -p /workspace/logs "$OUT"

case "${1:-}" in
stop)
  pkill -f "^bash $LIMINAL/ws4/cloud/runpod_serve.sh" || true
  pkill -f "^python -m flux.remote" || true
  pkill -f "^python $LIMINAL/ws4/ttt/serve_lora.py" || true
  for _ in $(seq 60); do pgrep -f "^python (-m flux.remote|$LIMINAL/ws4/ttt/serve_lora.py)" >/dev/null || break; sleep 2; done
  nvidia-smi --query-gpu=memory.used --format=csv,noheader
  ;;
serve)
  name="$2"; path="$3"
  echo "put the block in the cup" > /workspace/prompts_cup.txt
  extra="${EXTRA_ARGS:---override guidance_scale=1.0 --override guidance_scale_action=1.0 --override compile_model=true}"
  if [ -f "$path/adapter_config.json" ]; then
    # A LoRA checkpoint: ttt/serve_lora.py merges it into its base in memory (runpod_serve.sh runs plain flux.remote).
    [ -f /etc/rp_environment ] && source /etc/rp_environment
    source "${WS4_ROOT:-/workspace/ws4}/lerobot/.venv/bin/activate"
    export PYTHONPATH="$LIMINAL/ws4"
    # shellcheck disable=SC2086
    setsid nohup python "$LIMINAL/ws4/ttt/serve_lora.py" --host 127.0.0.1 --port 8765 --token-file /workspace/flux_token \
      --package "$name=$path" --text-encoder-on-cpu $extra >> /workspace/logs/flux_remote_lora.log 2>&1 < /dev/null &
    { python "$LIMINAL/ws4/cloud/warm_prompts.py" --prompts-file /workspace/prompts_cup.txt --token-file /workspace/flux_token \
        >> /workspace/logs/warm_prompts.log 2>&1 && echo "READY: $name"; } > "/workspace/logs/serve_$name.out" 2>&1
    tail -2 "/workspace/logs/serve_$name.out"
    exit 0
  fi
  # The same inference settings as every closed-loop run on this pod (A0 and the lid A0r-sim): no CFG, compiled.
  EXTRA_ARGS="${EXTRA_ARGS:---override guidance_scale=1.0 --override guidance_scale_action=1.0 --override compile_model=true}" \
  PACKAGES="$name=$path" IDLE_MIN=0 PROMPTS=/workspace/prompts_cup.txt \
    setsid nohup bash "$LIMINAL/ws4/cloud/runpod_serve.sh" > "/workspace/logs/serve_$name.out" 2>&1 < /dev/null &
  for _ in $(seq 180); do grep -q READY "/workspace/logs/serve_$name.out" 2>/dev/null && break; sleep 5; done
  tail -2 "/workspace/logs/serve_$name.out"
  ;;
eval)
  name="$2"; shift 2
  # The container disk is wiped when the pod stops; the EGL loader lives there.
  ldconfig -p | grep -q "libEGL.so.1 " || (apt-get update -qq && DEBIAN_FRONTEND=noninteractive \
    apt-get install -y -qq libegl1 libgl1 libglvnd0 > /dev/null)
  cd "$LIMINAL/ws4"
  WS4_TTT_NICE=0 MUJOCO_GL=egl /workspace/venv-simeval/bin/python ttt/eval_block_cup.py --package "$name" \
    --token-file /workspace/flux_token --out "$OUT/$name" "$@" > "/workspace/logs/eval_cup_$name.log" 2>&1
  tail -20 "/workspace/logs/eval_cup_$name.log"
  ;;
*)
  echo "usage: $0 stop | serve NAME PATH | eval NAME [eval_block_cup.py args]" >&2; exit 2;;
esac
