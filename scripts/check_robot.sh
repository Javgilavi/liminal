#!/usr/bin/env bash
# Keep an inherited ROS PYTHONPATH out of the isolated control environment.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
env -u PYTHONPATH PYTHONNOUSERSITE=1 .venv-robot/bin/python -m pytest "$@"
env -u PYTHONPATH PYTHONNOUSERSITE=1 .venv-robot/bin/ruff check src tests scripts
