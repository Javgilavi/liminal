#!/usr/bin/env bash
# Run on the existing GPU Pod after copying server.py and the pinned LeRobot source.
# HF_TOKEN is supplied through the SSH channel/environment, never as a command argument.
set -euo pipefail

flux_root=/workspace/flux3-action
flux_runtime=/opt/flux3-action
flux_python="$flux_runtime/.venv/bin/python"
export HF_HOME="$flux_root/hf-cache"
export UV_CACHE_DIR="$flux_runtime/uv-cache"
export UV_LINK_MODE=copy
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONUNBUFFERED=1

mkdir -p "$flux_root" "$flux_runtime"
test -f "$flux_root/lerobot/pyproject.toml"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# The template ships torch 2.8, whereas this FLUX integration needs torch 2.11
# and its matching NATTEN wheel. Keep the template's interpreter unchanged.
# The global volume rejects chmod/utime, so executable environments use /opt;
# source, logs and the large model cache stay on the mounted volume.
if [ ! -x "$flux_python" ]; then
  uv venv --python /usr/bin/python3 "$flux_runtime/.venv"
fi
# Editable builds create temporary metadata beside the source. Those operations
# are rejected by this Pod's global-volume filesystem, so build from a local copy.
"$flux_python" - <<'PY'
from pathlib import Path
import shutil
shutil.copytree('/workspace/flux3-action/lerobot', '/opt/flux3-action/lerobot',
                dirs_exist_ok=True, ignore=shutil.ignore_patterns('*.egg-info', '__pycache__'))
print('Copied the pinned LeRobot source to the container runtime directory.', flush=True)
PY
uv pip install --python "$flux_python" 'torch==2.11.0' 'torchvision==0.26.0' \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python "$flux_python" -e "$flux_runtime/lerobot[flux3]" \
  'fastapi==0.119.0' 'uvicorn==0.37.0' 'pillow>=10,<13'
uv pip install --python "$flux_python" 'natten==0.21.6+torch2110cu128' \
  --find-links https://whl.natten.org/

"$flux_python" -c 'import torch, natten; print("torch", torch.__version__, "CUDA available", torch.cuda.is_available(), "natten", natten.__version__); assert torch.cuda.is_available()'
"$flux_python" "$flux_root/download_models.py"
"$flux_python" "$flux_root/stage_models.py"
uv pip freeze --python "$flux_python" > "$flux_root/runtime-packages.txt"
printf '\nFLUX_SETUP_COMPLETE\n'
