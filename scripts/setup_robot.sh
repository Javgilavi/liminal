#!/usr/bin/env bash
# Install control dependencies only; no serial access and no system-wide installs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
command -v uv >/dev/null || { echo "Install uv: https://docs.astral.sh/uv/getting-started/installation/" >&2; exit 1; }
command -v fuser >/dev/null || { echo "Install psmisc (provides fuser) with your system package manager." >&2; exit 1; }
if [ ! -d .venv-robot ]; then uv venv --python 3.12 .venv-robot; fi
env -u PYTHONPATH PYTHONNOUSERSITE=1 uv pip install --python .venv-robot/bin/python \
  --torch-backend=cpu -e '.[hardware,ui,live,dev]'
echo "Ready. Start: ./scripts/control_robot.sh"
