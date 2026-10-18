# Chain skills: ACT fallback per skill (18 Oct night)

The final demo is a 4-skill chain on the Haller: box (robot RIGHT arm opens the box), green pen, yellow cutter, red
pen (robot LEFT arm). FLUX 3 Action LoRAs per skill are the main path (ws4-datasets); LeRobot ACT is the per-skill
fallback, trained on the same splits and scored with the same offline protocol. The red pen is in
`ws4/research/pen_lora.md` ("ACT baseline"). No real-arm attempts.

| What | Value |
|---|---|
| Data | each skill's 12-D VR dataset from the laptop cache, converted to the task arm (`ws4/data/convert_bimanual.py --to per_arm`, parquet only, videos linked): box = haller `left` columns + `left_wrist`; green and cutter = `right` + `right_wrist`; scene = head |
| Split | the dataset's `ws4_split.json` (ws4-vr2: last 3 good takes of each box position held out, review rejects dropped), read through `single_arm/build_skill.split_of` (`ws4/act/skill_split.py`), so the same takes as the eval kit's `skills/<skill>/skill.json` |
| Normalization | state and action mean/std of the train takes only (`ws4/act/train_view.py`), ImageNet statistics for the images |
| Policy | LeRobot ACT defaults (as for the pen): ResNet18, chunk 100, n_action_steps 100, lr 1e-5, KL 10, batch 8, both cameras at 480x640 |
| Steps | 40k (the pen was flat from 10k on), checkpoints every 10k |
| Eval | `ws4/act/offline_eval.py` (offline_diag's frame sampler, metrics and table: 200 frames, seed 0, first 32 of the 100 commands vs the recorded actions) |
| Commands | `bash ws4/act/skill_act.sh prep|train|watch|eval SKILL` on a pod with the real-train-is volume; outputs in `/workspace/act/` |

## Box (open the box by pulling its handle; robot right arm)

Dataset `ws4/haller_open_the_box` (snapshot 01:27 CEST): 75 takes, 31,678 frames. Split (ws4_split.json 01:25 CEST): 59
train, held out P2 36-38 and P3 72-74 (P1 has only 3 good takes, none held out), 10 dropped. Run
`/workspace/act/runs/act_box`, 23:30 to 00:34 UTC.

Moving frames: 199 of 200. MAE: arm joints (mean in brackets); gripper.

| Model | Steps | MAE pan, lift, elbow, wflex, roll (deg) (mean); gripper (%) | Direction agreement pan .. gripper | Median rel. displacement error | Train L1 |
|---|---|---|---|---|---|
| hold (repeat the current command) | - | 6.2, 9.4, 6.8, 6.9, 4.3 (6.7); 7.3 | - | 100 % | - |
| ACT | 10k | 3.4, 7.5, 6.8, 5.7, 4.3 (5.5); 6.4 | 94, 81, 71, 81, 73, 96 % | 30, 65, 74, 58, 69, 5 % | 0.153 |
| ACT | 20k | 3.1, 7.4, 7.1, 6.0, 4.6 (5.6); 5.4 | 94, 81, 73, 81, 73, 93 % | 28, 60, 79, 60, 72, 2 % | 0.110 |
| ACT | 30k | 3.2, 8.2, 7.7, 6.1, 4.7 (6.0); 5.0 | 93, 79, 71, 79, 73, 93 % | 25, 64, 85, 56, 61, 1 % | 0.089 |
| ACT | 40k | 3.1, 8.4, 7.9, 6.4, 5.2 (6.2); 5.0 | 92, 82, 71, 80, 73, 93 % | 30, 66, 92, 63, 76, 1 % | 0.077 |

- Unlike the pen, the box overfits: held-out arm MAE rises from 5.5 deg at 10k to 6.2 at 40k while train L1 keeps
  falling. Elbow and roll at 10k are only at hold level; pan, lift and wrist flex beat hold clearly.
- Pick: 20k (`/workspace/act/runs/act_box/checkpoints/020000/pretrained_model`): arm MAE within 0.1 deg of the best
  (10k) and a better gripper (5.4 vs 6.4 %).
  Laptop copy (ws4-datasets, 18 Oct 02:05 CEST): `~/haller_rescue/act/act_box_020000/pretrained_model`; run it with
  `run_single_arm.py --backend act --act-checkpoint <that dir>` and the box skill's arm flags.
- Dry runs (`ws4/act/dry_run.py`, fake arm replaying held-out takes 36 and 72, Haller left calibration,
  `skills/box/workspace.json` of the 01:46 CEST build with validation ok, `--max-step 1.0`): both 16 s, 0 keep-out or envelope refusals, at most 1.0 deg per
  tick, 33 to 38 ms per chunk on the pod GPU. The guard clamped 349 and 380 of 480 ticks: the pull is fast for
  1 deg/tick.
- Not done (pod stopped for budget at 00:41 UTC): speed check and per-position breakdown.

## Green pen, yellow cutter: stopped for budget

Both splits were final (green 01:50 CEST: 58 train, held out P3 63-65; cutter 02:35 CEST: 59 train, held out P3
67-69) and both datasets are converted on the volume (`/workspace/act/data/{green,cutter}_{arm,view}`). Training
started, but the orchestrator stopped the ACT pods for Oscar's RunPod budget (green at 00:22 UTC, then again at
00:41 on the box pod; cutter at 00:41) before any green or cutter checkpoint was written. They resume only if Oscar
wants them, on another account. To resume on a pod with this volume: `bash ws4/act/skill_act.sh train green 40000` and `watch green 10000 20000 30000 40000`, the same for
cutter (about 1 h each on an RTX PRO 4500).

## FLUX 3 Action LoRA per skill (ws4-datasets)

The pen recipe per skill (`ws4/cloud/skill_lora.sh`):
- base = released `flux-3-action-so101` with B quantiles of the skill's train takes, no joint transforms;
- LoRA rank/alpha 32, lr 1e-4, batch 2 x 4, 5,000 microsteps, EMA 0.999, text encoder on the GPU;
- one RTX PRO 6000 per skill.

Same held-out protocol as the pen curve (offline_diag, 200 frames, seed 0, guidance 1, moving frames). The zero-shot rows
were skipped for budget; the pen's are in `pen_lora.md`. Winner = better arm MAE and 6-joint direction, served as the
skill's package. Pod and timing details are in `ws4/cloud/RUNPOD.md`.

### Green pen (58 train, held out 63 to 65)

Moving frames: 196 of 200. Run `/workspace/skills/green_pen/lora` on skill-green-pen-6000, 23:57 to 01:40 UTC.
Eval loss (held-out, live weights): 0.222 at 500, 0.191 at 4,000, 0.186 at 5,000.

| Model | Microsteps | MAE pan, lift, elbow, wflex, roll (deg) (mean); gripper (%) | Direction agreement pan .. gripper (mean) |
|---|---|---|---|
| LoRA RAW (winner) | 5,000 | 2.6, 4.7, 4.3, 4.8, 3.4 (3.95); 5.3 | 96, 87, 86, 83, 87, 94 % (89.1) |
| LoRA EMA | 5,000 | 3.0, 5.7, 6.8, 7.4, 4.2 (5.40); 6.7 | 92, 87, 79, 69, 76, 89 % (82.1) |

- RAW dominates EMA on every joint. As on the pen at 5k, a 0.999 EMA still carries much of the initial adapter.
- The winner is `skill_green_pen` in `/workspace/serve/demo_packages.txt`: base `so101_haller_green_pen_B` + RAW 5k.
  The laptop copy is in `~/haller_rescue/green_pen/`.

### Box (59 train, held out P2 36 to 38, P3 72 to 74)

Moving frames: 199 of 200. Run `/workspace/skills/open_box/lora` on skill-open-box-6000, 23:30 to 01:40 UTC.
Eval loss: 0.342 at 500, 0.286 at 4,000, 0.276 at 5,000.

| Model | Microsteps | MAE pan, lift, elbow, wflex, roll (deg) (mean); gripper (%) | Direction agreement pan .. gripper (mean) |
|---|---|---|---|
| LoRA RAW (winner) | 5,000 | 3.3, 7.1, 5.8, 5.5, 3.9 (5.11); 5.7 | 89, 65, 69, 72, 70, 71 % (72.8; arm 73) |
| LoRA EMA | 5,000 | 3.7, 7.5, 6.2, 6.9, 4.8 (5.82); 8.6 | 84, 64, 68, 72, 62, 96 % (74.6; arm 70) |

- RAW is better on arm MAE, arm direction and gripper MAE; EMA only on gripper direction (96 vs 71 %).
- Against ACT box 20k (above): lower arm MAE (5.11 vs 5.6 deg), weaker arm direction (73 vs 80 %).
- The box is the hardest skill for both, and it overfit under ACT.
- Winner `skill_open_box` = `so101_haller_open_box_B` + RAW 5k. Laptop copy in `~/haller_rescue/open_box/`.

### Yellow cutter

No non-augmented run (budget hold); the augmented run below is the only cutter LoRA.

### Augmented runs and pre-robot gates (18 Oct morning)

Same recipe with `--policy.augment=true` (flux3 95 % crop + colour jitter, one draw per window), RUN `lora_aug`, from
each skill's B base (`ws4/cloud/skill_pod.sh all SKILL`). Gates before any robot time (`ws4/cloud/gate_check.py`):
(b) from rest (frame-0 state over the 8-tick history, frame-0 images) the first chunk moves >= 20 deg summed over the
5 arm joints on every held-out take; (c) the same under +-2 px wrist/scene shifts and 0.5x brightness (wrist, both).
The box and cutter demos move 39 to 150 deg in that window. Gate column = smallest first-chunk move over all takes and perturbations.

| Skill | Model | MAE pan, lift, elbow, wflex, roll (deg); gripper (%) | Gate (deg) | Eval loss 500 / 5k |
|---|---|---|---|---|
| Cutter (held out 67 to 69) | aug RAW (winner) | 2.5, 4.7, 7.1, 5.2, 6.8; 11.3 | 44.2 pass | 0.243 / 0.189 |
| | aug EMA | 3.1, 4.1, 6.4, 6.7, 7.3; 14.9 | 53.1 pass | |
| Box | aug RAW | 3.2, 7.3, 6.0, 5.7, 4.1; 5.8 | 5.3 fail | 0.381 / 0.294 |
| | aug EMA | 3.7, 7.6, 6.1, 6.9, 4.9; 8.5 | 5.5 fail | |
| | non-aug RAW (served) | 3.3, 7.1, 5.8, 5.5, 3.9; 5.7 | 4.9 fail | |
| Green pen | aug RAW | 2.7, 5.2, 4.9, 5.3, 3.6; 5.4 | 73.3 pass | 0.233 / 0.195 |
| | aug EMA | 3.0, 5.8, 7.0, 7.5, 4.2; 9.2 | 12.6 fail | |
| | non-aug RAW (served) | 2.6, 4.7, 4.3, 4.8, 3.4; 5.3 | not run | |

- Augmentation left the offline MAE about where it was (box and green within 0.2 to 0.6 deg per joint of non-aug).
- The box's from-rest knife edge is not from augmentation: aug and non-aug both barely move on takes 38 and 73
  (6 to 8 deg unperturbed vs 104 and 71 in the demos), and a dim wrist view drops every take to 5 to 13 deg.
- EMA fails or nearly fails under a dim wrist on green; RAW keeps its margin. RAW again wins wherever it passes.
- Cutter winner: `skill_yellow_cutter` = `so101_haller_yellow_cutter_B` + `lora_aug` RAW 5k. Laptop copies of all
  three runs (raw, EMA, gates, logs) are in `~/haller_rescue/{open_box,green_pen,yellow_cutter}_lora_aug/`.

## Compute

act-skills-4500 (jklmwivzh1h2i4) 23:26 to 00:41 UTC, act-skills-4500b (93244np4opnv1w) 23:52 to 00:22,
act-skills-4500c (1ayazmsx18wm41) 00:39 to 00:41; RTX PRO 4500, EUR-IS-1, $0.72/h each: about $1.10 in total. All
three are terminated.
