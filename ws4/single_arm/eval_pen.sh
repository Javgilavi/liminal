#!/usr/bin/env bash
# The pen rows of eval_skill.sh (approved order B84, median, LoRA; see eval_skill.sh for every option):
#   POD_HOST=157.157.221.177 POD_PORT=15377 LORA=lora_5000_raw bash ws4/single_arm/eval_pen.sh session
SKILL=pen exec bash "$(dirname "$0")/eval_skill.sh" "$@"
