#!/usr/bin/env bash
# Download the public XLeRobot-style datasets on the GPU box and turn them into training-ready LeRobot datasets.
# See ws4/research/datasets.md sections 4 to 7 for why these sets and what the conversion does.
#
#   source $WS4_ROOT/lerobot/.venv/bin/activate      # LeRobot main env from ws4/cloud/bootstrap.sh
#   bash ws4/data/fetch_public.sh                    # core set: zonglin11 + cortx fruit sorting (9.5 GB)
#   EXTRA=1 bash ws4/data/fetch_public.sh            # also rk000000 + bigse0u1 (+6.5 GB, kept as separate sets)
#
# Output under $DATA_ROOT (default ~/data):
#   raw/<owner>__<name>/            the Hub datasets at pinned revisions
#   per_arm/<name>_{left,right}/    FLUX SO-101 contract: 6-D degrees, cameras scene + wrist, task "... with the left arm"
#   xle12/<name>/                   12-D [left | right], cameras head, left_wrist, right_wrist
#   public_per_arm/                 merge of every compatible per-arm set: train LoRA P on it, or merge it with
#                                   our own per-arm data for LoRA B (python ws4/data/merge_datasets.py, same rule)
# Conversion rewrites metadata only and hard-links the videos, so it takes minutes. Idempotent.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${DATA_ROOT:-$HOME/data}"
EXTRA="${EXTRA:-0}"
ROBOT_TYPE_ARM="${ROBOT_TYPE_ARM:-so_follower}"      # must equal robot_type in our own per-arm recordings
ROBOT_TYPE_BI="${ROBOT_TYPE_BI:-bi_so_follower}"     # must equal robot_type in our own bimanual recordings
PY="${PY:-python}"
run() { timeout 1800 "$PY" -I "$@"; }   # guards against a hang at interpreter exit

# id  revision  extra converter flags   (revisions pinned 2026-10-17, see ws4/data/candidates.yaml)
CORE=(
  "zonglin11/xlerobot_clean_table 709834f02f777cd62ac7c7512fdcf98e8435ab3b"
  "cortx-labs/bimanual-fruit-sorting b04733fefae964eb15dec287fda354823d2ebc3b"
)
OPTIONAL=(
  "rk000000/xlerobot-handkerchief-quarters-mix-20260918-480 e2195d3e8ae0af48e2b75c97d7139bf8ffdd8a2e"   # h264: not mergeable with the AV1 sets
  "bigse0u1/xlerobot_scrub_7tool 27f60d6804fcda83080cdb1daf540808e4a09aa6 --flip-action-gripper"        # 240x320 wrists: not mergeable either
)

sets=("${CORE[@]}")
[ "$EXTRA" = 1 ] && sets+=("${OPTIONAL[@]}")
mkdir -p "$DATA_ROOT"/{raw,per_arm,xle12}
df -BG --output=avail "$DATA_ROOT" | tail -1 | xargs echo "free space in $DATA_ROOT:"

mergeable=()
for entry in "${sets[@]}"; do
  read -r id rev flags <<<"$entry"
  name="${id##*/}"
  raw="$DATA_ROOT/raw/${id/\//__}"
  echo "=== $id @ $rev"
  hf download "$id" --repo-type dataset --revision "$rev" --local-dir "$raw" >/dev/null
  run "$HERE/inspect_dataset.py" "$raw" | sed -n '1,4p'
  for side in left right; do
    out="$DATA_ROOT/per_arm/${name}_${side}"
    [ -d "$out" ] || run "$HERE/convert_bimanual.py" "$raw" "$out" --to per_arm --side "$side" \
      --robot-type "$ROBOT_TYPE_ARM" ${flags:-} >/dev/null
    case " ${CORE[*]} " in *" $id "*) mergeable+=("$out") ;; esac
  done
  out="$DATA_ROOT/xle12/$name"
  [ -d "$out" ] || run "$HERE/convert_bimanual.py" "$raw" "$out" --to xle12 \
    --robot-type "$ROBOT_TYPE_BI" ${flags:-} >/dev/null
done

echo "=== merge the compatible per-arm sets into $DATA_ROOT/public_per_arm"
if [ ! -d "$DATA_ROOT/public_per_arm" ]; then
  run "$HERE/merge_datasets.py" "$DATA_ROOT/public_per_arm" "${mergeable[@]}" --repo-id local/public_per_arm
fi
run "$HERE/inspect_dataset.py" "$DATA_ROOT/public_per_arm" --contracts flux3_so101 | sed -n '1,12p'
du -sh "$DATA_ROOT"/* 2>/dev/null
