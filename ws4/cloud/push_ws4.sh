#!/usr/bin/env bash
# Copy the uncommitted ws4/ work from this laptop to a cloud GPU box (ws4/ is not pushed to GitHub yet).
#   bash ws4/cloud/push_ws4.sh <ssh-host> [remote liminal dir, default ~/liminal]
set -euo pipefail
HOST="${1:?usage: push_ws4.sh <ssh-host> [remote dir]}"
REMOTE_DIR="${2:-liminal}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ssh "$HOST" "mkdir -p $REMOTE_DIR"
rsync -az --info=stats1 \
  --exclude '.cache/' --exclude '__pycache__/' --exclude '*.pyc' --exclude 'outputs/' \
  "$HERE/ws4" "$HOST:$REMOTE_DIR/"
echo "ws4/ synced to $HOST:$REMOTE_DIR/ws4"
