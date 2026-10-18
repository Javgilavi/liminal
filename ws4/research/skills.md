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

## Compute

act-skills-4500 (jklmwivzh1h2i4) 23:26 to 00:41 UTC, act-skills-4500b (93244np4opnv1w) 23:52 to 00:22,
act-skills-4500c (1ayazmsx18wm41) 00:39 to 00:41; RTX PRO 4500, EUR-IS-1, $0.72/h each: about $1.10 in total. All
three are terminated.
