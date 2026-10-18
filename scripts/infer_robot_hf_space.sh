#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# Compatibility alias: keep both everyday commands on the same default profile.
exec ./scripts/infer_robot.sh "$@"
