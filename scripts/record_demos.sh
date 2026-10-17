#!/usr/bin/env bash
# WS1: collect 50-100 teleoperated demos for task-specific LoRA (SO-101 leader -> follower).
# Flags follow LeRobot's `lerobot-record`; they change between releases, so check
# `lerobot-record --help` on the installed version first.
#
# Before recording: generate the variation plan and follow it episode by episode:
#   python scripts/make_demo_plan.py --episodes 60 --out datasets/plans/pick_place_v1.jsonl
set -euo pipefail

HF_USER="${HF_USER:?set HF_USER to your Hugging Face username}"
TASK="${TASK:-Pick up the red block and place it in the bowl}"
REPO="${REPO:-${HF_USER}/liminal_so101_pick_place_v1}"
EPISODES="${EPISODES:-60}"
FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/ttyACM0}"
LEADER_PORT="${LEADER_PORT:-/dev/ttyACM1}"

# Camera names must match what FLUX expects (scene + wrist) and stay fixed at eval time.
CAMERAS='{ top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30} }'

lerobot-record \
  --robot.type=so101_follower --robot.port="$FOLLOWER_PORT" --robot.id=right_follower \
  --robot.cameras="$CAMERAS" \
  --teleop.type=so101_leader --teleop.port="$LEADER_PORT" --teleop.id=right_leader \
  --display_data=true \
  --dataset.repo_id="$REPO" \
  --dataset.num_episodes="$EPISODES" \
  --dataset.single_task="$TASK" \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=15
