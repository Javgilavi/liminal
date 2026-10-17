# Pen-in-slot LoRA on FLUX 3 Action SO-101 (real track, 17 to 18 Oct)

LoRA fine-tune of the released `black-forest-labs/flux-3-action-so101` (revision `0713a7af`) on the Haller VR demos of
"put the red pen in its slot in the cardboard box" (robot-left task arm, per-arm FLUX SO-101 format). Pod, costs and
commands: `ws4/cloud/RUNPOD.md` "Real-track LoRA pod real-train-6000".

## Setup

| What | Value |
|---|---|
| Data | `haller_put_the_red_pen_in_its_slot_in_the_cardboard_box`, frozen at 104 episodes (46,372 frames, 0.43 h), layouts A to D; `convert_bimanual --to per_arm --side right --src-units degrees --task-suffix ""` |
| Split | ws4-vr2's `ws4_split.json` (final, 19:43 UTC): 84 train, 12 held-out (last 3 good takes of each layout: A 20 to 22, B 36 to 38, C 86 to 88, D 101 to 103), 8 dropped (review.json rejects) |
| Base | variant B: history quantiles recomputed on the 84 train takes (38,686 frames), no joint transforms, gripper absolute as recorded (`ws4/single_arm/a0r_stats.py`, `flux.package_tools.renormalize_package`) |
| LoRA | LeRobot `examples/flux3/lora.json`: rank/alpha 32, lr 1e-4 (heads x5), EMA 0.999, batch 2 x accumulation 4 (effective 8), text encoder on the GPU; 20,000 microsteps = 5,000 updates (about 1 epoch of the train frames) |
| Checkpoints | every 500 microsteps, raw and EMA adapters (0.45 GB each); eval loss on 256 held-out samples at each |
| Run | `/workspace/real_lora/lora_pen84` on real-train-6000, started 19:50:20 UTC, about 1.0 s per microstep |

## Held-out action-error curve

`ws4/cloud/lora_adapter.py eval` runs so101-zeroshot's `ws4/single_arm/offline_diag.py` unchanged on the 12
held-out takes, with the EMA adapter folded into the weights in memory:
- the same 200 frames for every row (seed 0), guidance 1.0, compile off;
- requests built as the runner builds them (8-tick state and command history, snapshots at ticks 0 and 7);
- the first 32 of the 42 predicted absolute commands compared with the recorded actions;
- "moving" frames: the recorded chunk moves an arm joint by more than 5 deg or the gripper by more than 15 %.

Moving frames: 181 of 200.

| Model | Microsteps | MAE pan, lift, elbow, wflex, roll (deg); gripper (%) | Direction agreement pan .. gripper | Median rel. displacement error | Eval loss |
|---|---|---|---|---|---|
| hold (repeat the current command) | - | 3.8, 7.6, 5.9, 7.6, 3.2; 8.5 | - | 100 % | - |
| zero-shot B (base, no LoRA) | 0 | 4.6, 6.3, 4.9, 8.0, 3.7; 11.3 | 65, 71, 68, 54, 75, 66 % | 152, 88, 81, 171, 70, 93 % | - |
| zero-shot median (multimodalart 09edebb stats) | 0 | 2.7, 6.7, 5.4, 5.3, 3.8; 6.8 | 74, 64, 72, 77, 42, 87 % | 69, 93, 84, 71, 120, 64 % | - |
| LoRA EMA | 2,500 | 3.0, 5.1, 4.7, 6.0, 2.9; 9.9 | 72, 77, 78, 64, 75, 78 % | 71, 68, 71, 96, 63, 90 % | 0.233 |

Raw results: `/workspace/real_lora/eval/*.json` (and `.npz` with the predictions) on the `real-train-is` volume.
offline_diag.py labels every row "A" (its variant name for "the package's own statistics"); the package path in each
JSON says which model it is.

Training eval loss (LeRobot flow-matching loss, 256 samples of the 12 held-out takes, live weights): 0.261 at 500,
0.249 at 1,000, 0.231 at 1,500, 0.231 at 2,000, 0.233 at 2,500 microsteps.

- At 2,500 microsteps (625 updates) the LoRA beats both zero-shot variants on direction (mean 74 % vs 66.5 % for B84
  and 69 % for median) and on arm MAE (mean 4.3 vs 5.5 and 4.8 deg).
- Its gripper MAE (9.9 %) is still worse than hold (8.5 %).
- ws4-act's ACT baseline at 10k steps (section at the end) is the bar to clear: arm MAE mean 4.3 deg, direction
  81 to 90 %.

## Zero-shot normalization variants (same weights, same protocol)

M = the median new-calibration SO-101 statistics from multimodalart's Space
(`multimodalart/flux-3-action-so101-sim@09edebb`, app.py NORMALIZATION). Its gripper range is 0..40, ours is 0..100
(open = 100), so its gripper numbers are listed separately. Moving frames: 181 of 200.

| Zero-shot | Arm MAE pan, lift, elbow, wflex, roll (deg), mean | Arm direction agreement, mean | Gripper MAE / direction | Overall direction (6 joints) |
|---|---|---|---|---|
| hold | 3.8, 7.6, 5.9, 7.6, 3.2 (5.6) | - | 8.5 % / - | - |
| B84 | 4.6, 6.3, 4.9, 8.0, 3.7 (5.5) | 65, 71, 68, 54, 75 % (67) | 11.3 % / 66 % | 66.5 % |
| zero-shot median (M) | 2.7, 6.7, 5.4, 5.3, 3.8 (4.8) | 74, 64, 72, 77, 42 % (66) | 6.8 % / 87 % | 69.3 % |

Speed: ws4-imagine's `speed_check.py` on all 127 moving held-out chunks; model path over demo path, summed per joint. My B84
rerun reproduces their numbers exactly.

| Zero-shot | Path ratio pan, lift, elbow, wflex, roll, grip | Displacement ratio |
|---|---|---|
| B84 | 1.49, 0.32, 0.55, 1.06, 0.99, 1.00 | 1.83, 0.30, 0.53, 1.37, 1.40, 0.78 |
| zero-shot median (M) | 0.62, 0.65, 0.83, 0.89, 0.57, 0.87 | 0.57, 0.70, 0.83, 1.02, 0.72, 0.73 |

- M fixes B84's pan over-swing and doubles lift motion (0.65x vs 0.32x).
- Its arm MAE is lower (4.8 vs 5.5), but direction agreement is no better overall and roll falls below chance (42 %).
- Against the bar for a median-base LoRA (overall direction of at least 80 % and MAE no worse than B84 on every
  joint) it is NO-GO, confirmed by the orchestrator. No median LoRA. The staged base `so101_haller_pen_M84g` and
  `launch_penM.sh` on the volume are dropped and unused.
- Median is the better-balanced zero-shot (twice B84's lift motion, no pan over-swing), so it is served as `median`
  for graded real-arm attempts from the 5k swap on.
