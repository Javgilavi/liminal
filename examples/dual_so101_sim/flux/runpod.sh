#!/usr/bin/env bash
# Repeat the verified single-arm Runpod + local MuJoCo workflow.
set -euo pipefail
flux_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
flux_host="${RUNPOD_SSH_HOST:-154.54.102.27}"
flux_ssh_port="${RUNPOD_SSH_PORT:-11752}"
flux_key="${RUNPOD_SSH_KEY:-$HOME/.ssh/id_ed25519}"
flux_local_port="${FLUX_LOCAL_PORT:-18100}"
flux_socket="$flux_dir/.runpod/tunnel.sock"
flux_known_hosts="$flux_dir/.runpod/known_hosts"
flux_ssh=(ssh -i "$flux_key" -p "$flux_ssh_port" -o BatchMode=yes
  -o StrictHostKeyChecking=yes -o UserKnownHostsFile="$flux_known_hosts"
  -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=3)

case "${1:-help}" in
  tunnel|tunnel-foreground)
    if "${flux_ssh[@]}" -S "$flux_socket" -O check "root@$flux_host" >/dev/null 2>&1; then
      printf 'Runpod tunnel already running.\n'
    else
      # A disconnected SSH process can leave its control socket behind.
      rm -f -- "$flux_socket"
      flux_detach=(-f)
      if [[ "$1" == tunnel-foreground ]]; then
        flux_detach=()
        printf 'FLUX endpoint: http://127.0.0.1:%s (keep this terminal open)\n' "$flux_local_port"
      fi
      "${flux_ssh[@]}" -M -S "$flux_socket" "${flux_detach[@]}" -N -o ExitOnForwardFailure=yes \
        -L "127.0.0.1:$flux_local_port:127.0.0.1:8100" "root@$flux_host"
    fi
    printf 'FLUX endpoint: http://127.0.0.1:%s\n' "$flux_local_port"
    ;;
  start)
    "${flux_ssh[@]}" "root@$flux_host" 'set -e
      flux_root=/workspace/flux3-action
      if test -f "$flux_root/server.pid" && kill -0 "$(cat "$flux_root/server.pid")" 2>/dev/null; then
        echo "Model server already running; check status or logs."
        exit 0
      fi
      test -x /opt/flux3-action/.venv/bin/python
      flux_model_cache="$flux_root/hf-cache"
      if test -f /opt/flux3-action/hf-cache/.complete; then
        flux_model_cache=/opt/flux3-action/hf-cache
      fi
      env HF_HOME="$flux_model_cache" HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 \
        setsid /opt/flux3-action/.venv/bin/python "$flux_root/server.py" \
        --host 127.0.0.1 --port 8100 --device cuda > "$flux_root/server.log" 2>&1 < /dev/null &
      echo $! > "$flux_root/server.pid"
      echo "Started model loading. Use runpod.sh logs and runpod.sh status."
    '
    ;;
  status)
    "${flux_ssh[@]}" "root@$flux_host" \
      'nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv; curl --max-time 5 -fsS http://127.0.0.1:8100/health'
    ;;
  logs)
    "${flux_ssh[@]}" "root@$flux_host" 'tail -n 60 /workspace/flux3-action/server.log'
    ;;
  ui)
    shift
    flux_sim_dir="$(cd -- "$flux_dir/.." && pwd)"
    flux_python="${FLUX_SIM_PYTHON:-$flux_sim_dir/.venv/bin/python}"
    if ! "${flux_ssh[@]}" -S "$flux_socket" -O check "root@$flux_host" >/dev/null 2>&1; then
      printf 'Runpod tunnel is not running. Run: bash runpod.sh tunnel\n' >&2
      exit 1
    fi
    "$flux_python" - "http://127.0.0.1:$flux_local_port" <<'PY'
import json
import sys
import urllib.request
try:
    with urllib.request.urlopen(sys.argv[1] + '/health', timeout=10) as response:
        health = json.load(response)
except OSError as error:
    raise SystemExit(f'Cloud FLUX is not ready: {error}. Check: bash runpod.sh status') from error
if (health.get('fake') is not False or health.get('device') != 'cuda'
        or health.get('model') != 'black-forest-labs/flux-3-action-so101'):
    raise SystemExit('Expected real FLUX on the cloud CUDA GPU; refusing to launch.')
print('Verified real FLUX on Runpod CUDA. This laptop runs only the simulator/UI.')
PY
    export MUJOCO_GL="${MUJOCO_GL:-egl}"
    exec "$flux_python" "$flux_dir/app.py" --policy "http://127.0.0.1:$flux_local_port" "$@"
    ;;
  stop-tunnel)
    "${flux_ssh[@]}" -S "$flux_socket" -O exit "root@$flux_host"
    ;;
  stop-server)
    "${flux_ssh[@]}" "root@$flux_host" 'python3 -c '\''from pathlib import Path
import os, signal
p=Path("/workspace/flux3-action/server.pid")
pid=int(p.read_text())
cmd=Path(f"/proc/{pid}/cmdline").read_bytes()
assert b"/workspace/flux3-action/server.py" in cmd, "PID does not belong to this model server"
os.kill(pid, signal.SIGTERM)
p.unlink()
print("Stopped this FLUX server. The Pod itself is still running.")'\'
    ;;
  *)
    printf 'Usage: bash runpod.sh {start|logs|status|tunnel|tunnel-foreground|ui [--port 8000]|stop-tunnel|stop-server}\n'
    ;;
esac
