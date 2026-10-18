#!/usr/bin/env bash
# Exclusive controller; refuses busy serial ports. Starts with torque OFF.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [ ! -x .venv-robot/bin/liminal ]; then
  echo "Run ./scripts/setup_robot.sh first." >&2
  exit 1
fi
echo "Operator UI: http://127.0.0.1:8080 (torque off until Engage)"
exec env -u PYTHONPATH PYTHONNOUSERSITE=1 .venv-robot/bin/liminal serve \
  --robot configs/robot/xlerobot.yaml --backend scripted --port 8080 "$@"
