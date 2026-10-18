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
| LoRA EMA (on the Hub as `step-4000`) | 4,000 | 2.4, 4.5, 4.3, 5.4, 2.5; 8.4 | 76, 82, 80, 71, 82, 81 % | 53, 53, 63, 83, 64, 87 % | 0.219 |
| LoRA EMA (on the Hub as `step-5000`, served as `lora_5000`) | 5,000 | 2.1, 4.4, 4.2, 5.0, 2.4; 6.7 | 79, 84, 83, 75, 89, 81 % | 46, 48, 60, 75, 64, 79 % | 0.210 |
| LoRA RAW | 5,000 | 1.9, 4.0, 4.0, 4.0, 2.3; 4.7 | 89, 88, 87, 85, 89, 62 % | 47, 44, 55, 46, 53, 72 % | 0.210 |
| LoRA EMA | 10,000 | 1.7, 4.0, 3.9, 4.2, 2.2; 4.3 | 91, 87, 88, 84, 89, 84 % | 42, 40, 45, 53, 58, 78 % | 0.213 |
| LoRA RAW (staged as `step10000-raw`, served as `lora_10000_raw`) | 10,000 | 1.7, 4.0, 3.8, 4.3, 2.1; 4.5 | 93, 85, 88, 84, 89, 92 % | 45, 41, 49, 50, 53, 55 % | 0.213 |

Raw results: `/workspace/real_lora/eval/*.json` (and `.npz` with the predictions) on the `real-train-is` volume.
offline_diag.py labels every row "A" (its variant name for "the package's own statistics"); the package path in each
JSON says which model it is.

Training eval loss (LeRobot flow-matching loss, 256 samples of the 12 held-out takes, live weights): 0.261 at 500,
0.249 at 1,000, 0.231 at 1,500, 0.231 at 2,000, 0.233 at 2,500, 0.223 at 3,000, 0.226 at 3,500, 0.219 at 4,000,
0.226 at 4,500, 0.210 at 5,000 microsteps.

## Raw vs EMA and guidance at 5k (BFL so101-lora.md and LeRobot flux3.mdx recommendations)

The same protocol, 5k checkpoint. Latency is the in-process predict time from the eval itself: compile off, on the
shared training GPU, so only relative. Served with compile on, guidance 1/1 takes 0.36 s on the server.

| 5k variant | Arm MAE pan, lift, elbow, wflex, roll (deg), mean | Direction pan .. roll; gripper | Arm direction mean | Gripper MAE | Rel. displacement error | s per chunk |
|---|---|---|---|---|---|---|
| EMA, guidance 1/1 (served, on the Hub) | 2.1, 4.4, 4.2, 5.0, 2.4 (3.62) | 79, 84, 83, 75, 89; 81 % | 82 % | 6.7 % | 46, 48, 60, 75, 64, 79 % | 0.77 |
| RAW, guidance 1/1 | 1.9, 4.0, 4.0, 4.0, 2.3 (3.24) | 89, 88, 87, 85, 89; 62 % | 87.5 % | 4.7 % | 47, 44, 55, 46, 53, 72 % | 0.76 |
| EMA, guidance 3/3 | 2.2, 5.1, 4.6, 4.8, 2.4 (3.83) | 81, 84, 72, 79, 88; 84 % | 81 % | 7.5 % | 56, 58, 69, 67, 62, 54 % | 1.48 |
| EMA, guidance 4/1 (video 4, action 1) | 2.3, 4.6, 4.2, 5.0, 2.3 (3.67) | 81, 84, 79, 76, 89; 84 % | 82 % | 7.0 % | 57, 49, 64, 68, 58, 48 % | 1.44 |

- **Raw beats EMA at 5k on every arm joint.** At 1,250 updates a 0.999 EMA still carries about 29 % of the initial
  adapter, as the orchestrator noted.
  - Arm MAE 3.24 vs 3.62 deg, arm direction 87.5 vs 82 %, wrist flex direction 85 vs 75 %, gripper MAE 4.7 vs 6.7 %.
  - Only the gripper's direction is worse (62 vs 81 %).
  - On the arm, raw 5k is level with ACT (final 100k) on direction (87.5 vs 85 %) with lower MAE (3.24 vs 4.3 deg).
- **Higher guidance does not help.** 3/3 and 4/1 are no better than 1/1 and double the per-chunk compute: two DiT passes
  per solver step for classifier-free guidance. Served with compile, 3/3 would take about 0.7 s on the server and
  about 1.0 s round trip, at the arm's budget.
- **Selection mismatch.** LeRobot's eval loss, used to pick the Hub checkpoint, is computed on the live (raw) weights,
  while the Hub and 8790 hold the EMA adapter.

## Lid-held scene: offline check of RAW 10k (18 Oct about 02:20 UTC)

RAW 10k scored 0/5 live in the lid-held red-pen scene, returning hold chunks. The same model offline on the
lid-held dataset (22 takes, `/workspace/skills/red_pen_lid`), same protocol:

| Takes | Moving frames | Arm MAE (mean, deg) | Hold arm MAE | Direction (6-joint mean) | Path ratio pan .. grip |
|---|---|---|---|---|---|
| old held-out (12, layouts A to D) | 181 | 3.15 | 5.61 | 88.6 % | 0.75, 0.86, 0.85, 0.63, 0.61, 1.12 |
| lid-held held-out 19 to 21 | 199 | 5.57 | 9.24 | 80.2 % | 0.52, 0.64, 0.62, 0.51, 0.40, 1.00 |
| lid-held all 20 good | 184 | 5.24 | 7.79 | 76.6 % | 0.53, 0.73, 0.78, 0.51, 0.43, 1.49 |

- **From rest, as a live first request:** state at frame 0 repeated 8 ticks, commands = state, both snapshots =
  frame 0. The model moves 32 to 61 deg (summed arm displacement over the chunk) on all 20 lid-held takes and 34
  to 66 deg on the 12 old ones; never hold-like.
- So the scene costs accuracy and about half the motion, but it does not explain hold chunks. The live inputs
  likely differ from the recorded ones (camera mapping or rotation, state units or calibration, command history,
  task string or package). The continuation fine-tune on the 126-episode dataset targets the scene shift only.

## Public checkpoint on the Hub

[Oskrt/flux-3-action-so101-haller-pen-lora](https://huggingface.co/Oskrt/flux-3-action-so101-haller-pen-lora) holds
only the latest best EMA adapter.

| Tag | Step | Held-out eval loss | Uploaded by |
|---|---|---|---|
| `step-4000` | 4,000 | 0.2187 | orchestrator, 17 Oct about 21:25 UTC, on Oscar's request |
| `step-5000` | 5,000 | 0.2096 | `hub_watch.py`, 17 Oct 21:43:53 UTC (Oscar approved automatic replacement for the night at 21:30) |
| `step-5000-raw` (staged, not uploaded) | 5,000 RAW | 0.2096 (the eval loss is the raw weights') | not uploaded: the orchestrator chose raw 5k, the permission check blocked the public upload, and Oscar (about 22:25 UTC): "its fine for now", so the Hub keeps EMA 5k |

- A later checkpoint replaces it only if its held-out eval loss is lower and its offline action-error eval does not
  regress against the Hub checkpoint's: mean direction (6 joints) at most 1 point lower, mean arm MAE at most
  0.1 deg higher. `/workspace/real_lora/hub_watch.py` on real-train-6000 applies this and uploads, running its own
  offline eval for checkpoints between the curve targets.
- About 00:45 UTC on 18 Oct the orchestrator reported that the RAW 10k red-pen LoRA and the three skill datasets are
  now public on the Hub (uploaded on its side).
- From 22:15 UTC (orchestrator): eval loss alone no longer triggers a replacement. At 10k, 15k and 20k,
  `select_watch.py` evaluates RAW and EMA and picks the variant that beats the current best (RAW 5k) on both arm MAE
  and 6-joint direction. It stages the pick and never uploads; the Hub stays at EMA 5k until Oscar says otherwise.
  `hub_watch.py` is stopped.
- 5k against 4k: direction 81.7 vs 78.4 %, arm MAE 3.62 vs 3.83 deg, eval loss 0.2096 vs 0.2187. The Hub adapter's
  sha256 matches the local 5000 EMA file.
- Staging: `/workspace/serve/stage_hub.sh <step>` on real-serve-6000 rebuilds the `step4000` folder for that step
  and regenerates only the card's checkpoint paragraph.

- At 2,500 microsteps (625 updates) the LoRA beats both zero-shot variants on direction (mean 74 % vs 66.5 % for B84
  and 69 % for median) and on arm MAE (mean 4.3 vs 5.5 and 4.8 deg).
- Its gripper MAE (9.9 %) is still worse than hold (8.5 %).
- The ACT baseline (section "ACT baseline" at the end) is flat from 10k to 100k steps. Final, ACT 100k:
  - arm MAE 2.7, 5.4, 5.1, 5.1, 3.0 deg (mean 4.3), gripper 4.7 %;
  - direction 86, 85, 88, 78, 87, 94 % (arm mean 85 %, 6-joint mean 86 %).
- The LoRA beats it on arm MAE from 4k on. RAW 5k is level on arm direction (87.5 vs 85 %) and ties on gripper MAE
  (4.7 %); EMA 5k trails on the gripper (6.7 %).

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
| LoRA EMA 5k (`lora_5000` over HTTP on 8790, ws4-imagine, 21:45 to 21:55 UTC) | 0.76, 0.78, 0.64, 0.60, 0.55, 1.02 | 0.77, 0.86, 0.70, 0.64, 0.64, 0.58 |

- The 5k LoRA moves lift at 0.78x demo speed (B84 0.32x) and no longer over-swings pan (0.76x vs 1.49x).
- Every arm joint still moves slower than the demos (0.55 to 0.78x path).
- On the served model with compile on, lazy imagination gave the same commands as plain requests on all 23 replayed
  chunks, decoding at 0.12 s per chunk (ws4-imagine, `/workspace/imagine/lora5000_ep10{1,2,3}`).

- M fixes B84's pan over-swing and doubles lift motion (0.65x vs 0.32x).
- Its arm MAE is lower (4.8 vs 5.5), but direction agreement is no better overall and roll falls below chance (42 %).
- Against the bar for a median-base LoRA (overall direction of at least 80 % and MAE no worse than B84 on every
  joint) it is NO-GO, confirmed by the orchestrator. No median LoRA. The staged base `so101_haller_pen_M84g` and
  `launch_penM.sh` on the volume are dropped and unused.
- Median is the better-balanced zero-shot (twice B84's lift motion, no pan over-swing), so it is served as `median`
  for graded real-arm attempts from the 5k swap on.

## ACT baseline (ws4-act, LeRobot ACT on the same 84 train takes)

Baseline and fallback for the LoRA: LeRobot ACT trained from an ImageNet backbone on the same demos, scored with the
same frames and metrics as the curve above. Code and runbook: `ws4/act/` (README there). No real-arm attempts yet.

| What | Value |
|---|---|
| Data | the 84 train takes of `ws4_split.json` "final" (38,686 frames) from `pen_in_right_final`; scene and wrist cameras at 480x640, 6-D state, 6-D absolute actions (LeRobot degrees, gripper 0..100) |
| Normalization | state and action mean/std of the 84 train takes only (`ws4/act/train_view.py`; `--dataset.episodes` alone would normalize with all 104 takes), ImageNet statistics for the images |
| Policy | LeRobot ACT defaults at lerobot@146d847: ResNet18 (ImageNet, frozen BatchNorm), dim 512, 4 encoder / 1 decoder layers, CVAE (latent 32, KL weight 10), chunk 100, n_action_steps 100, AdamW lr 1e-5 (backbone 1e-5), weight decay 1e-4, batch 8, 52 M parameters, fp32, seed 1000 |
| Run | `/workspace/act/runs/act_pen84`: 100,000 steps (800k samples, about 20.7 epochs), checkpoints every 10k, wandb off; pod `act-train-4500` (RTX PRO 4500 32 GB, EUR-IS-1, $0.72/h, same volume), 0.089 s per step, 20:35 to 22:57 UTC |
| Eval | `ws4/act/offline_eval.py`: offline_diag.py's frame sampler, metrics and table (its hold row reproduces the one above exactly: 181 of 200 moving frames). ACT gets s[t] and the current frame of each camera (it has no history input); the first 32 of its 100 absolute commands are compared with a[t..t+31]. Raw: `/workspace/act/eval/act_pen84_<step>.json` (+ `.npz`) |

Moving frames: 181 of 200. MAE: arm joints (mean in brackets); gripper.

| Model | Steps | MAE pan, lift, elbow, wflex, roll (deg) (mean); gripper (%) | Direction agreement pan .. gripper | Median rel. displacement error | Train L1 at that step |
|---|---|---|---|---|---|
| hold (repeat the current command) | - | 3.8, 7.6, 5.9, 7.6, 3.2 (5.6); 8.5 | - | 100 % | - |
| zero-shot B (from the table above) | 0 | 4.6, 6.3, 4.9, 8.0, 3.7 (5.5); 11.3 | 65, 71, 68, 54, 75, 66 % | 152, 88, 81, 171, 70, 93 % | - |
| ACT | 10k | 2.8, 5.5, 4.9, 5.1, 3.2 (4.3); 5.6 | 88, 90, 82, 81, 84, 88 % | 40, 50, 55, 50, 54, 24 % | 0.156 |
| ACT | 20k | 2.7, 5.5, 5.1, 5.0, 3.2 (4.3); 5.2 | 89, 86, 83, 84, 83, 87 % | 38, 46, 55, 57, 53, 14 % | 0.116 |
| ACT | 30k | 2.6, 5.2, 4.7, 5.0, 3.1 (4.1); 4.8 | 86, 87, 85, 83, 84, 88 % | 37, 45, 52, 55, 50, 12 % | 0.096 |
| ACT | 40k | 2.5, 5.2, 5.0, 5.2, 3.1 (4.2); 5.0 | 87, 88, 83, 79, 82, 90 % | 36, 47, 55, 53, 53, 9 % | 0.084 |
| ACT | 50k | 2.7, 5.3, 5.4, 5.2, 3.1 (4.3); 4.7 | 86, 85, 81, 84, 84, 91 % | 37, 43, 57, 52, 45, 10 % | 0.075 |
| ACT | 60k | 2.7, 5.1, 5.1, 5.0, 3.0 (4.2); 4.9 | 88, 87, 86, 81, 82, 91 % | 40, 48, 53, 50, 51, 9 % | 0.069 |
| ACT | 70k | 2.8, 5.2, 5.1, 5.1, 3.1 (4.3); 4.6 | 86, 87, 86, 80, 85, 91 % | 42, 48, 52, 52, 50, 7 % | 0.064 |
| ACT | 80k | 2.5, 5.2, 5.1, 5.1, 2.9 (4.2); 4.7 | 86, 87, 88, 82, 86, 92 % | 37, 49, 48, 48, 48, 8 % | 0.060 |
| ACT | 90k | 2.7, 5.1, 4.9, 5.1, 3.1 (4.2); 4.6 | 86, 85, 88, 79, 86, 91 % | 42, 46, 48, 53, 51, 5 % | 0.056 |
| ACT | 100k | 2.7, 5.4, 5.1, 5.1, 3.0 (4.3); 4.7 | 86, 85, 88, 78, 87, 94 % | 45, 53, 50, 54, 53, 8 % | 0.054 |

Speed (`ws4/act/speed_check.py`: ws4-imagine's chunk starts and summary, so the same 127 moving held-out chunks as the
zero-shot speed table above; model path or displacement over the demo's, summed per joint):

| ACT | Path ratio pan, lift, elbow, wflex, roll, grip | Displacement ratio |
|---|---|---|
| 10k | 1.47, 1.39, 1.42, 1.12, 1.47, 1.10 | 0.99, 1.01, 1.04, 0.90, 1.09, 0.92 |
| 50k | 1.17, 1.22, 1.20, 1.03, 1.09, 0.99 | 1.00, 1.03, 1.01, 0.93, 0.96, 0.96 |
| 100k | 1.11, 1.13, 1.06, 0.99, 1.08, 0.95 | 1.00, 1.01, 0.96, 0.94, 1.04, 0.93 |

Per held-out layout (`ws4/act/by_layout.py`, moving frames A 46, B 56, C 38, D 41; the FLUX rows are recomputed from
ws4-datasets' `.npz` files on the volume): arm MAE mean (deg); gripper MAE (%).

| Model | A | B | C | D |
|---|---|---|---|---|
| zero-shot B84 | 6.1; 6.9 | 5.6; 12.5 | 5.3; 13.4 | 4.7; 12.3 |
| LoRA EMA 5k | 3.9; 5.5 | 3.6; 6.8 | 3.6; 8.8 | 3.4; 6.0 |
| ACT 100k | 4.9; 4.8 | 4.4; 5.5 | 4.2; 4.6 | 3.3; 3.6 |

- **ACT is flat from 10k steps on.**
  - Every checkpoint from 10k to 100k lands within 0.2 deg of arm MAE mean (4.1 to 4.3), and the gripper stays at
    4.6 to 5.6 %.
  - Direction agreement stays at 78 to 94 % per joint.
  - Meanwhile its train L1 falls from 0.156 to 0.054: it keeps fitting the 84 train takes, but gains nothing more on
    held-out takes after about 2 epochs (10k steps). More steps neither help nor hurt on these metrics.
  - It moves the right distance (displacement ratios 0.93 to 1.04 on every joint; B84: 0.30 to 1.83). The extra
    path (1.47 at 10k, 1.13 at 100k) is chunk-to-chunk jitter, which shrinks with training.
- **FLUX LoRA vs ACT on the same frames.**
  - From the 5k LoRA rows above: the LoRA has the lower arm MAE (EMA 3.62, raw 3.24 vs ACT 4.3 deg mean) and is
    ahead on layouts A, B and C. On D the two tie (3.4 vs 3.3).
  - ACT keeps the better gripper (4.7 % vs EMA 6.7 %, raw ties at 4.7 %) and gripper direction (94 vs 81 %, raw 62 %).
  - On arm direction, raw 5k is level with ACT (87.5 vs 85 %, the mean of ACT 100k's five arm joints).
  - So the LoRA passes the ACT bar on arm MAE from 4k microsteps (3.8 deg) on, after 8k training samples (micro-batch
    2), where ACT's first checkpoint had seen 80k. It does not pass it on the gripper yet.
- **Caveats.**
  - This is offline action error on held-out takes of the same four layouts, not task success.
  - The inputs differ: ACT gets s[t] and one frame per camera at 480x640; FLUX gets 8 ticks of state and command
    history plus 2 snapshots at 256x256.
  - At deployment ACT executes its whole 100-command chunk open loop by default (3.3 s), while FLUX re-plans every 32.

ACT checkpoint for the arm, if Oscar gives it arm time: `/workspace/act/runs/act_pen84/checkpoints/100000/pretrained_model`
(final step; every 10k checkpoint is kept on the volume, 200 MB each). Laptop copy: `ws4/.cache/act/act_pen84_100k`.

Runner (no arm runs so far):
- `run_single_arm.py --backend act --act-checkpoint <dir>` runs ACT in process, with LeRobot 0.5.1 on the laptop.
  - `ws4/act/backend.py` loads the pod's 146d847 checkpoints and needs no download.
  - The T550 GPU takes 69 ms per chunk (81 max), its CPU 273 ms; the pod GPU took 9 ms.
- `ws4/act/serve.py` serves the same backend with `flux.remote`'s protocol, for `--backend remote` (101 ms per chunk
  on 127.0.0.1).
- Every command goes through the runner's guard, keep-out, demo envelope, soft limits, stall and load freeze and torque
  cap. Variant and condition names are the orchestrator's call; ACT-100k / post_training is the suggestion.
- Dry runs (`ws4/act/dry_run.py`): the runner with the fake arm replaying a held-out take's frames, Haller soft limits,
  the final-split keep-out and eval_pen.sh's `--max-step 1.0`.
  - Takes 20, 36, 86 and 101 with the 100k checkpoint: all 16 s, 0 keep-out or envelope refusals, at most 1.0 deg per
    tick.
  - The guard clamped 145 to 235 of 480 ticks. Demos exceed 1 deg/tick on 13.6 % of ticks (p99 about 2), so under
    `--max-step 1.0` the arm lags ACT's plan and every new chunk starts with a ramp.
- `ws4/act/tests` (10 CPU tests) includes a fake-robot run where ACT asks for pan 120 deg: the envelope refuses it, pan
  never passes 57.6 deg, and the attempt stops after 45 refused ticks.

Compute: pod `act-train-4500` (id zhkilyhem41c2f, RTX PRO 4500, EUR-IS-1, $0.72/h, on this volume) ran from 20:08:46
to 23:04 UTC (2.9 h, about $2.10) and is terminated. Everything of ws4-act's lives under `/workspace/act/`.

### ACT on the real arm (18 Oct, about 04:20 CEST)

Five graded attempts on the Haller in the original pen84 scene (layouts A B C D A from the held-out start frames, lid
hinged open, box arm folded at the tuck), Oscar grading, `SKILL=pen bash ws4/act/eval_act_arm.sh`:
- Checkpoint ACT pen84 100k (`ws4/.cache/act/act_pen84_100k`), in process on the laptop GPU (70 ms median per chunk,
  135 ms max), re-planning every 32 commands, 40 s per attempt.
- Every eval-kit guard: final-split keep-out with the demo envelope, soft limits, 1 deg per tick, ramps to haller
  home, home check, stall and load freeze, 60 % torque cap.
- One override (Oscar's, in every trial note): box-arm start tolerance 15 deg instead of 5, because its wrist flex
  sat at 66 deg vs 78 in the pen84 takes.

| Attempt | Layout | Steps (reach, grasp, insert) | What ACT did |
|---|---|---|---|
| 1 | A | 0/3 | dipped and opened/closed the gripper in place (pan near -2 deg) for 40 s, never left the start area |
| 2 | B | 2/3 | grasped the pen at 13 s, carried it over the box, hovered above the slot gripper closed until timeout |
| 3 | C | 2/3 | grasped at 7 s, over the box at 12 s, wrist tipping down toward the slot, no release |
| 4 | D | 2/3 | grasped at 7 s, over the box at 12 s, wrist tipping down toward the slot, no release |
| 5 | A | 2/3 | grasped at 7 s, over the box at 12 s, hovered above the slot, no release |

- **8 of 15 steps; grasp and carry in 4 of 5; no insert.** For comparison, FLUX raw 10k scored 0/5 holding still in the
  lid-held scene, and held still on all 24 requests of so101-zeroshot's camera check in this rebuilt pen84 scene.
- Nothing held ACT back: 0 keep-out or envelope refusals, no stall or load freeze, 0 interventions, every command at
  most 1.0 deg per tick. The guard count is only the per-tick step limit smoothing fast commands.
- The failure is the last step: from about 12 s on, ACT creeps over the slot with the gripper closed and never lowers
  and opens within the 40 s.
- Session folder `ws4/single_arm/runs/20261018_041957`; trial lines in `ws4/eval/trials.jsonl` (put_pen_in_slot,
  post_training, ACT-pen84-100k).
