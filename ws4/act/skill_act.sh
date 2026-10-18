#!/usr/bin/env bash
# Per-skill ACT fallback (box, green, cutter; pen has its own run), on a pod with the real-train-is volume at
# /workspace. Dataset, task arm and wrist camera come from single_arm/skills.py; the split is build_skill.py's
# (act/skill_split.py). Outputs under /workspace/act/ only.
#
#   rsync the 12-D dataset from the laptop cache (with its ws4_split.json) to /workspace/act/raw/<dataset name>
#   bash ws4/act/skill_act.sh prep SKILL           # per-arm conversion (videos hard-linked) + train-only stats view
#   bash ws4/act/skill_act.sh train SKILL STEPS    # LeRobot ACT defaults, checkpoints every SAVE_FREQ (10k)
#   bash ws4/act/skill_act.sh watch SKILL STEP...  # offline eval of each checkpoint as it lands (curve_watch.sh)
#   bash ws4/act/skill_act.sh eval SKILL STEP...   # offline eval of finished checkpoints
#
# prep refuses a provisional split (no ws4_split.json yet) unless ALLOW_PROVISIONAL=1.
set -euo pipefail
source /etc/rp_environment
source "$WS4_ROOT/lerobot/.venv/bin/activate"
export HF_HUB_OFFLINE=1 PYTHONPATH=/workspace/act/ws4
A=/workspace/act
W=$A/ws4
mode="${1:?prep|train|watch|eval}"
skill="${2:?skill (box, green, cutter)}"
shift 2
read -r dataset columns wrist < <(cd "$W/single_arm" && python -c "
from skills import ARMS, SKILLS
s = SKILLS['$skill']; a = ARMS[s.arm]; print(s.dataset, a.columns, a.wrist_key)")
raw=$A/raw/$(basename "$dataset")
arm=$A/data/${skill}_arm
view=$A/data/${skill}_view
run=act_$skill
[ -d "$raw" ] || { echo "upload $dataset to $raw first"; exit 1; }
eval "$(python "$W/act/skill_split.py" "$raw")"
echo "$skill: $dataset, columns $columns, wrist $wrist; train $(echo $TRAIN_EPS | wc -w) takes, held-out $EVAL_EPS ($SPLIT_SOURCE)"

case "$mode" in
  prep)
    if [ "$PROVISIONAL" != 0 ] && [ -z "${ALLOW_PROVISIONAL:-}" ]; then
      echo "provisional split: wait for ws4_split.json or set ALLOW_PROVISIONAL=1"; exit 1
    fi
    rm -rf "$arm" "$view"
    python -I "$W/data/convert_bimanual.py" "$raw" "$arm" --to per_arm --side "$columns" \
      --scene observation.images.head --wrist "$wrist" --src-units degrees --task-suffix ""
    [ -f "$raw/ws4_split.json" ] && cp "$raw/ws4_split.json" "$arm/"
    python "$W/act/train_view.py" --src "$arm" --dst "$view" --episodes $TRAIN_EPS
    echo "$HELDOUT_JSON" > "$view/meta/heldout.json"
    ;;
  train)
    VIEW="$view" TRAIN_EPS="$TRAIN_EPS" EXPECT_TRAIN="$(echo $TRAIN_EPS | wc -w)" \
      bash "$W/act/train_act.sh" train "$run" "${1:?steps}"
    ;;
  watch | eval)
    task=$(python -c "import pandas as pd; t = pd.read_parquet('$arm/meta/tasks.parquet')
print(t['task'].iloc[0] if 'task' in t.columns else t.index[0])")  # ACT ignores it; recorded in the JSON
    export ROOT="$arm" EXCLUDE TASK="$task"
    if [ "$mode" = watch ]; then
      bash "$W/act/curve_watch.sh" "$run" "$@"
    else
      bash "$W/act/eval_act.sh" "$run" "${LABEL:-final}" $(printf '%06d ' "$@")
    fi
    ;;
  *) echo "mode must be prep, train, watch or eval"; exit 2 ;;
esac
