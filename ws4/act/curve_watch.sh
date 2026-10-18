#!/usr/bin/env bash
# Evaluate each ACT checkpoint of RUN as it lands (eval_act.sh, one JSON per checkpoint), on the training pod.
#
#   setsid nohup bash ws4/act/curve_watch.sh act_pen84 10000 20000 ... > /workspace/act/logs/curve_watch.log 2>&1 &
#
# A checkpoint counts as written once LeRobot points checkpoints/last at it (it updates the link after saving).
set -uo pipefail
run="${1:?run}"
shift
ckdir=/workspace/act/runs/$run/checkpoints
for n in "$@"; do
  step=$(printf %06d "$n")
  until [ "$(readlink "$ckdir/last" 2>/dev/null | xargs -r basename)" = "$step" ] || \
        [ -e "$ckdir/$(printf %06d $((n + 10000)))" ]; do
    sleep 30
  done
  echo "$(date +%H:%M:%S) evaluating $step"
  bash /workspace/act/ws4/act/eval_act.sh "$run" "$step" "$step" > "/workspace/act/logs/eval_${run}_$step.log" 2>&1
  echo "$(date +%H:%M:%S) $step rc=$? $(tail -1 "/workspace/act/logs/eval_${run}_$step.log")"
done
echo "CURVE_WATCH_DONE"
