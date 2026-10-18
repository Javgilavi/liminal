#!/usr/bin/env bash
# Companion only: Haller remains the motor and camera owner.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
research_python="${LIMINAL_UI_PYTHON:-.venv/bin/python}"
exec env -u PYTHONPATH PYTHONNOUSERSITE=1 "$research_python" -m liminal research \
  --site-url https://javgilavi.github.io/liminal/ "$@"
