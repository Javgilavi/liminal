# WS1: Single-arm FLUX 3 Action baseline, demos, task LoRA

## Questions to answer

1. What does the out-of-the-box SO-101 checkpoint do on our table, cameras and objects?
2. Where does it fail: perception (wrong object), grasp, placement, drift, timing?
3. How much does a LoRA on 50–100 of our demos improve the same eval?
4. Latency budget: how long is the pause between chunks, and does it matter?

## Eval protocol (fix it before collecting data)

- Task: `configs/tasks/single_pick_place.yaml` (prompt fixed verbatim).
- Layouts: 3 start layouts × 10 trials = 30 trials per model. Same camera poses, lighting noted.
- Success = `block_in_bowl` true within 30 s, verified by perception *and* by a human.
- Log per trial: success, time, failure mode, `runs/*.jsonl` file.
- Compare zero-shot vs LoRA on identical layouts.

## Demo collection

- `scripts/make_demo_plan.py --episodes 60` → variation plan (zone × yaw × distractors × lighting).
- `scripts/record_demos.sh` (LeRobot `lerobot-record`, leader → follower). Check `--help` first.
- Keep camera names/poses identical to inference. A moved scene camera invalidates data.
- Paraphrase instructions a little (`--instruction` repeatable) only if eval uses paraphrases.
- Hold out ~10% of episodes; never train on eval layouts.
- Quality beats quantity: discard hesitant or failed demos, keep motion speed consistent.

## Fine-tuning

- Planning notes: the published SO-101 LoRA recipe uses ~10k training microsteps; BFL's
  example adaptation used ~200 teleop demos. With 50–100, watch for overfitting:
  checkpoint often, eval the early ones.
- Fill in `scripts/finetune_lora.sh` from the official recipe, and set `lora_adapter` in
  `configs/policy/flux3_so101.yaml`.

## To verify first (`VERIFY` markers)

checkpoint id · processors · input image size · action units vs LeRobot calibration ·
chunk/execute/Hz · history length · VRAM · LoRA entrypoint.

## References

- FLUX 3 Action SO-101 model card and integration docs (BFL / Hugging Face). Links are in the planning notes; verify them.
- LeRobot (Hugging Face): `lerobot-record`, `lerobot-calibrate`, dataset format.
