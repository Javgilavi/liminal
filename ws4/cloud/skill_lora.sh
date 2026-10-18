#!/usr/bin/env bash
# Per-skill LoRA pipeline for the Haller demo chain (17 to 18 Oct): the pen recipe (ws4/cloud/real_lora.sh) for each
# new VR dataset, on its own pod that mounts the real-train-is volume (envs and weights already there, no bootstrap).
#
#   bash ws4/cloud/skill_lora.sh prep  SKILL SRC SIDE      # laptop, low CPU: split check, raw dataset to the volume
#   bash ws4/cloud/skill_lora.sh train SKILL HOST PORT     # on SKILL's pod: per-arm data, B stats, base, LoRA (5k)
#   bash ws4/cloud/skill_lora.sh eval  SKILL HOST PORT [STEP]   # zero-shot B, RAW and EMA on the held-out takes
#   bash ws4/cloud/skill_lora.sh report SKILL HOST PORT [STEP]  # the eval table and the pick
#
# prep needs SRC/ws4_split.json with status "final" (ws4-vr2: layouts = box positions, last 3 good takes per position
# held out, review.json rejects dropped). SIDE is the dataset's operator-frame side: left = robot RIGHT arm
# (left_* columns, left_wrist camera), right = robot LEFT arm. The task string is read verbatim from tasks.parquet.
# The laptop records VR at 30 Hz meanwhile, so prep only reads the split and rsyncs the closed raw dataset under nice -n 19
# and idle IO priority; conversion and stats run on the pod (orchestrator, 18 Oct 00:40 CEST).
# Volume layout: /workspace/skills/SKILL/{raw, data, a0r_B_SKILL.json, skill.env, lora, eval}; base package
# /workspace/packages/so101_haller_SKILL_B. Settings as lora_pen84 except STEPS (default 5000; 5k RAW was the pen's
# best) and SAVE (checkpoint every 1000, the pruner keeps training_state on the newest only; the volume must keep
# MIN_FREE_GB=40 free of QUOTA_GB=250, checked with du before a run starts).
# Test knobs: NO_UPLOAD=1 stops prep before the upload; NO_TRAIN=1 makes train stop after data, stats and
# package (no watchdog, no run); DRY=1 makes eval only write and print its script.
set -euo pipefail
WS4="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KEY="${KEY:-$HOME/.ssh/id_ed25519_osrdvs}"
UPLOAD_HOST="${UPLOAD_HOST:-root@157.157.221.177}"
UPLOAD_PORT="${UPLOAD_PORT:-16301}"  # any pod on real-train-is; default real-train-6000
cmd="${1:-}"; skill="${2:-}"
[ -n "$cmd" ] && [ -n "$skill" ] || { sed -n 2,16p "$0"; exit 2; }
remote() {  # host port script: run on a pod; setsid jobs in the script can keep ssh open, hence the timeout
  timeout "${REMOTE_TIMEOUT:-120}" ssh -n -o ConnectTimeout=20 -p "$2" -i "$KEY" "$1" "$3" 2>&1 | grep -v setlocale || true
}

case "$cmd" in
prep)
  src="${3:?SRC dataset dir}"; side="${4:?SIDE left|right}"
  out="$WS4/.cache/skills/$skill"
  mkdir -p "$out"
  OMP_NUM_THREADS=2 nice -n 19 python3 - "$src" "$side" "$skill" > "$out/skill.env" <<'EOF'
import json
import shlex
import sys
from pathlib import Path

import pandas as pd

src, side, skill = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
split = json.loads((src / "ws4_split.json").read_text())
assert split["status"] == "final", f"split status is {split['status']}"
info = json.loads((src / "meta/info.json").read_text())
n = info["total_episodes"]
review = json.loads((src / "review.json").read_text()) if (src / "review.json").exists() else {"episodes": {}}
rejects = sorted(int(k) for k, v in review["episodes"].items() if v.get("status") == "reject")
train = sorted(split["train"])
held = sorted(e for v in split["heldout"].values() for e in v)
assert not set(train) & set(held), "train and held-out overlap"
assert not (set(train) | set(held)) & set(rejects), "a review.json reject is in train or held-out"
assert max(train + held) < n, f"split names episode {max(train + held)}, dataset has {n}"
tasks = list(pd.read_parquet(src / "meta/tasks.parquet").index)
assert len(tasks) == 1, f"expected one task, got {tasks}"
env = {"SKILL": skill, "SIDE": side, "TASK": tasks[0], "N_EPISODES": str(n),
       "TRAIN_EPS": " ".join(map(str, train)), "EVAL_EPS": " ".join(map(str, held)),
       "STATS_EXCLUDE": " ".join(str(e) for e in range(n) if e not in set(train)),
       "EVAL_EXCLUDE": " ".join(str(e) for e in range(n) if e not in set(held))}
for k, v in env.items():
    print(f"{k}={shlex.quote(v)}")
print(f"# train {len(train)}, held-out {len(held)} ({split['heldout']}), rejects {rejects}, {n} episodes", file=sys.stderr)
EOF
  # shellcheck disable=SC1091
  source "$out/skill.env"
  cp "$src/ws4_split.json" "$out/ws4_split_final.json"
  [ -f "$src/review.json" ] && cp "$src/review.json" "$out/review_final.json"
  echo "split OK (task: $TASK; train $(wc -w <<< "$TRAIN_EPS"), held-out $(wc -w <<< "$EVAL_EPS"), $N_EPISODES episodes)"
  [ "${NO_UPLOAD:-0}" = 1 ] && { echo "NO_UPLOAD=1: not uploaded"; exit 0; }
  remote "$UPLOAD_HOST" "$UPLOAD_PORT" "mkdir -p /workspace/skills/$skill"
  nice -n 19 ionice -c3 rsync -rltz -e "ssh -o ConnectTimeout=20 -p $UPLOAD_PORT -i $KEY" "$src/" \
    "$UPLOAD_HOST:/workspace/skills/$skill/raw/" 2>&1 | grep -v setlocale || true
  rsync -rltz -e "ssh -o ConnectTimeout=20 -p $UPLOAD_PORT -i $KEY" "$out/skill.env" "$out/ws4_split_final.json" \
    "$UPLOAD_HOST:/workspace/skills/$skill/" 2>&1 | grep -v setlocale || true
  remote "$UPLOAD_HOST" "$UPLOAD_PORT" "python3 -c \"import json; n = json.load(open('/workspace/skills/$skill/raw/meta/info.json'))['total_episodes']; print('uploaded raw:', n, 'episodes', 'OK' if n == $N_EPISODES else 'MISMATCH with the split')\"; du -sh /workspace/skills/$skill/raw"
  ;;
train)
  host="${3:?HOST e.g. root@1.2.3.4}"; port="${4:?PORT}"
  remote "$host" "$port" "set -e; source /etc/rp_environment; source /workspace/skills/$skill/skill.env
    [ -e /workspace/skills/$skill/lora ] && { echo 'lora output exists: not starting twice'; exit 1; }
    used=\$(du -s --block-size=1G /workspace 2>/dev/null | cut -f1)
    free=\$(( ${QUOTA_GB:-250} - used ))
    echo \"volume: \$used GB used of ${QUOTA_GB:-250}, \$free GB free; a 5k run needs about 7 GB\"
    [ \$(( free - 7 )) -ge ${MIN_FREE_GB:-40} ] || { echo \"NOT STARTING: would leave under ${MIN_FREE_GB:-40} GB free\"; exit 1; }
    [ \"${NO_TRAIN:-0}\" = 1 ] || setsid nohup bash /workspace/liminal/ws4/ttt/pod_job_watchdog.sh 30 >> /workspace/logs/job_watchdog_$skill.log 2>&1 < /dev/null &
    source \$WS4_ROOT/lerobot/.venv/bin/activate; export PYTHONPATH=/workspace/liminal/ws4 HF_HUB_OFFLINE=1
    cd /workspace/liminal/ws4
    [ -e /workspace/skills/$skill/data ] || python data/convert_bimanual.py /workspace/skills/$skill/raw \
      /workspace/skills/$skill/data --to per_arm --side \$SIDE --scene observation.images.head \
      --wrist observation.images.\${SIDE}_wrist --src-units degrees --task-suffix '' 2>&1 | tail -4
    [ -s /workspace/skills/$skill/a0r_B_$skill.json ] || python single_arm/a0r_stats.py --root /workspace/skills/$skill/data \
      --repo-id local/$skill --exclude \$STATS_EXCLUDE --out /workspace/skills/$skill/a0r_B_$skill.json 2>&1 | tail -8
    cd /workspace/liminal/ws4/cloud
    [ -e /workspace/packages/so101_haller_${skill}_B ] || PKG=/workspace/packages/so101_haller_${skill}_B \
      bash real_lora.sh package /workspace/skills/$skill/a0r_B_$skill.json 2>&1 | grep -v -i warn | tail -2
    [ \"${NO_TRAIN:-0}\" = 1 ] && { echo 'NO_TRAIN=1: data, stats and package only'; ls /workspace/skills/$skill; exit 0; }
    PKG=/workspace/packages/so101_haller_${skill}_B DATA=/workspace/skills/$skill/data REPO=local/$skill \
      OUT=/workspace/skills/$skill/lora STEPS=${STEPS:-5000} SAVE=${SAVE:-1000} TRAIN_EPS=\"\$TRAIN_EPS\" EVAL_EPS=\"\$EVAL_EPS\" \
      setsid nohup bash real_lora.sh train > /workspace/logs/train_$skill.out 2>&1 < /dev/null &
    sleep 5; head -c 300 /workspace/logs/train_$skill.out; echo; tail -1 /workspace/logs/job_watchdog_$skill.log"
  ;;
eval)
  host="${3:?HOST}"; port="${4:?PORT}"; step="${5:-5000}"
  s6=$(printf %06d "$step")
  remote "$host" "$port" "source /etc/rp_environment; source /workspace/skills/$skill/skill.env
    mkdir -p /workspace/skills/$skill/eval
    cat > /workspace/skills/$skill/eval/run_$step.sh <<EOF
set -uo pipefail
source /etc/rp_environment; source \\\$WS4_ROOT/lerobot/.venv/bin/activate
export PYTHONPATH=/workspace/liminal/ws4 HF_HUB_OFFLINE=1; cd /workspace/liminal/ws4
C=/workspace/skills/$skill/lora/checkpoints/$s6
for v in zeroshot raw ema; do
  case \\\$v in zeroshot) a=(); out=zeroshot ;; raw) a=(--adapter \\\$C/pretrained_model); out=raw_$step ;;
    ema) a=(--adapter \\\$C/pretrained_model_ema); out=ema_$step ;; esac
  [ -s /workspace/skills/$skill/eval/\\\$out.json ] && continue
  python cloud/lora_adapter.py eval --package /workspace/packages/so101_haller_${skill}_B \"\\\${a[@]}\" -- \\\\
    --root /workspace/skills/$skill/data --repo-id local/$skill --task \"\$TASK\" --variants A --samples 200 --seed 0 \\\\
    --text-encoder-on-cpu --exclude \$EVAL_EXCLUDE --out /workspace/skills/$skill/eval/\\\$out.json \\\\
    > /workspace/skills/$skill/eval/\\\$out.log 2>&1
done
echo EVAL_DONE
EOF
    if [ \"${DRY:-0}\" = 1 ]; then bash -n /workspace/skills/$skill/eval/run_$step.sh && cat /workspace/skills/$skill/eval/run_$step.sh; exit 0; fi
    setsid nohup bash /workspace/skills/$skill/eval/run_$step.sh > /workspace/skills/$skill/eval/run_$step.out 2>&1 < /dev/null &
    echo started evals for $skill at $step"
  ;;
report)
  host="${3:?HOST}"; port="${4:?PORT}"; step="${5:-5000}"
  remote "$host" "$port" "cat /workspace/skills/$skill/eval/run_$step.out 2>/dev/null; python3 - <<'EOF'
import json
from pathlib import Path
d = Path('/workspace/skills/$skill/eval')
rows = {}
for v in ('zeroshot', 'raw_$step', 'ema_$step'):
    f = d / f'{v}.json'
    if not f.exists():
        print(v, 'missing'); continue
    a = json.loads(f.read_text())['variants']['A']
    dirs = [100 * x for x in a['moving_direction_agreement']]
    rows[v] = (sum(a['moving_mae'][:5]) / 5, sum(dirs) / 6)
    print(f\"{v}: arm MAE {', '.join(f'{x:.1f}' for x in a['moving_mae'][:5])} (mean {rows[v][0]:.2f}); gripper {a['moving_mae'][5]:.1f} %; \"
          f\"direction {', '.join(f'{x:.0f}' for x in dirs)} % (mean {rows[v][1]:.1f}); {json.loads(f.read_text())['hold']['moving_frames']} moving frames\")
c = [v for v in ('raw_$step', 'ema_$step') if v in rows]
if c:
    best = min(c, key=lambda v: (rows[v][0], -rows[v][1]))
    dominates = all(rows[best][0] <= rows[o][0] and rows[best][1] >= rows[o][1] for o in c)
    print('pick', best, '(dominates)' if dominates else '(lower arm MAE; direction trade-off, check the table)')
EOF"
  ;;
*)
  sed -n 2,16p "$0"; exit 2;;
esac
