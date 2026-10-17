#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec env -u PYTHONPATH PYTHONNOUSERSITE=1 .venv-robot/bin/python -m liminal.policy.live --config configs/policy/official_haller_right.yaml "$@"
