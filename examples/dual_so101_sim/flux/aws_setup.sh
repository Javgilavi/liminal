#!/usr/bin/env bash
# One-time setup of a cloud GPU box (e.g. AWS g6e.xlarge, 1x L40S 48 GB) for flux/server.py.
# Run ON the instance, from the folder you copied server.py into:
#     bash aws_setup.sh
# Then start the server with:
#     ~/lerobot/.venv/bin/python server.py --host 127.0.0.1 --port 8100
# Follows docs.bfl.ai/flux_3/flux3_action_so101 (Python 3.12, torch 2.11 + CUDA 12.8, LeRobot main, NATTEN).
set -euo pipefail

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
# torch cu128 wheels need an NVIDIA driver >= 570.
driver_major=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
if [ "$driver_major" -lt 570 ]; then
  echo "NVIDIA driver $driver_major is too old for CUDA 12.8; use a newer Deep Learning AMI." >&2
  exit 1
fi

# Loading stages ~25 GB of weights in RAM before moving them to the GPU; g6e.xlarge has 32 GiB.
ram_gb=$(free -g | awk '/^Mem:/ {print $2}')
if [ "$ram_gb" -lt 48 ] && ! swapon --show | grep -q /swapfile; then
  sudo fallocate -l 16G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
fi

if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

if [ ! -d "$HOME/lerobot" ]; then
  git clone --depth 1 https://github.com/huggingface/lerobot.git "$HOME/lerobot"
fi
cd "$HOME/lerobot"
[ -d .venv ] || uv venv --python 3.12 .venv
PY=.venv/bin/python
uv pip install --python $PY 'torch==2.11.0' 'torchvision==0.26.0' --index-url https://download.pytorch.org/whl/cu128
uv pip install --python $PY -e '.[flux3]'
uv pip install --python $PY 'natten==0.21.6+torch2110cu128' --find-links https://whl.natten.org/
uv pip install --python $PY fastapi uvicorn pillow

# Gated weights: accept the license on both model pages with this HF account first.
if ! .venv/bin/hf auth whoami >/dev/null 2>&1; then
  .venv/bin/hf auth login
fi
.venv/bin/hf download black-forest-labs/flux-3-action-so101 \
  --revision c9e13b2aca6a0a472b3ea03fd90cff32d3e85849
.venv/bin/hf download black-forest-labs/flux-3-action-base \
  --revision 62878e2925e59b7a89ec14463ce89932624c490d \
  --include 'video_vae.safetensors' --include 'text_encoder/*'

$PY -c "import torch, natten; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| natten', natten.__version__)"
echo "Setup done. Start: ~/lerobot/.venv/bin/python server.py --host 127.0.0.1 --port 8100"
