# Open datasets for adapting FLUX 3 Action to the XLeRobot station

Owner: ws4-datasets agent, 2026-10-17. Machine-readable list: `ws4/data/candidates.yaml`. Tools: `ws4/data/`.
Long notes: `ws4/data/notes/human_handheld_datasets.md` (UMI, EgoDex, HOT3D and others, every number sourced),
`ws4/data/notes/flux_action_data_path.md` (what the flux-action training code accepts, with file:line citations).

## Status

**Done**
- Surveyed the Hub: 859 search hits (xlerobot, bi_so*, bimanual, so101) plus all 1,755 sub-datasets of
  `lerobot/community_dataset_v3`, by reading their `meta/info.json`. Inspected 30 shortlisted datasets
  (stats, tasks, sizes, licences). Downloaded trajectories only (parquet, ~250 MB, no video) for 12 of them and
  looked at sampled frames from 10 (`ws4/.cache/frames/*.jpg`).
- Confirmed, from BFL's own corpus file, what the SO-101 checkpoint was trained on, and from the flux-action
  source what a fine-tune can and cannot consume (video-only samples, EE50, mixing).
- Code, tested on CPU (33 tests, `pytest ws4/data/tests`, ruff clean):
  - `inspect_dataset.py`: prints a dataset's contract and diffs it against the FLUX SO-101 and 12-D contracts.
  - `convert_bimanual.py`: bimanual to per-arm FLUX SO-101 format, or to our 12-D format. It rewrites
    metadata only and does no video re-encode. Tested on real `zonglin11` and `bigse0u1` data.
  - `ee_convert.py`: UMI-style EE datasets or robot joints to a common EE format, plus EE50 z-score stats. It
    reuses ws4-ik's `kinematics/` read-only. Tested on real `axiboai` (UMI), `roco` (UMI) and `zonglin11` (FK).
  - Supporting tools: `survey_hub.py`, `hub_sizes.py`, `sample_frames.py`, `build_candidates.py`.

**Verified**
- Joint units per dataset, from stats plus the raw trajectories (state against action fit).
- Camera roles and viewpoints, from frames.
- Repo sizes, revisions and licences, from the Hub API.

**Not verified**
- Video decoding of the converted datasets inside LeRobot or flux-action (no videos downloaded yet).
- Every EE50 convention (BFL never released them).
- Whether any human dataset is already in FLUX's midtraining (BFL names none).

**Next 3 steps**
1. The GPU-box teammate runs `ws4/data/fetch_public.sh` (9.5 GB core set) as in `ws4/HANDOFF_GPU.md` step 3.
2. Convert the core set per arm, then run LoRA A (own data only) and LoRA B (own data plus public) in parallel
   on two GPUs, for the time-split evidence in §8.
3. Request access to Pantheon DiverseUMI now (manual gate). Use it only if the EE50 gate in §5 passes.

## 1. Headline findings

1. **FLUX SO-101 never saw a bimanual robot or an XLeRobot.** `configs/so101/community_corpus.json`
   (flux-action@830f1ce) lists 1,021 parts. There are 709 from the `allenai/MolmoAct2-SO100_101-Dataset`
   manifest and 312 from `community_dataset_v3`, 631,391 training windows (about 146 h, my estimate from stride
   25 at 30 fps).
   - The selection rules include "30 fps datasets only" and "parts whose action dimension is not 6 contribute
     no windows". Both 12-D parts (`tfoldi/so100_glass_1`, `woohyunwoo/put_in_black_box`) got zero windows.
   - No part and no `community_dataset_v3` sub-dataset has an `xle` robot type or name.
   - Only 15 parts (11,229 windows) are a single arm of a bimanual rig, used unmirrored.
   - Only 11 of the 1,021 chosen scene cameras are `head` or `overhead`. The rest are mostly `image`, `phone`
     and `laptop`.
   - So the most valuable public data is exactly what is missing: an XLeRobot-style head camera on a mast, with
     two arms in view.
2. **About 900 public XLeRobot / bimanual SO-101 episodes match our station well.** They come from four labs:
   head camera behind and above the arms, two wrist cameras, 30 fps, joints in degrees, Apache-2.0.
   - Two of them (`zonglin11`, `saivishwak`) even appear to use white slatted fin-ray style fingers (from frames).
   - Core set that merges for LoRA training: `zonglin11` + `cortx`, 3.3 h, 9.5 GB of video (§6).
3. **EE50 cannot be trained with the released code.** flux-action@830f1ce has no EE50 encoder, no stats, no
   frame convention and no midtraining loader. Its head wiring drops every `action_prediction*` head that is
   not the run's own modality (`src/flux_action/models/wiring.py:118-143`).
   - The base checkpoint carries "ee50/gaming action heads" (HF card), but it is gated. Reusing them means
     writing that plumbing ourselves and guessing the conventions.
   - On the data side, Simone's plan step 1 is doable (`ee_convert.py`), but the plan is gated on code, not on
     data.
4. **Video-only samples are only half supported.** A window can carry `action_mask = zeros(D)`, which makes its
   action loss exactly 0 (`src/flux_action/processing/packing.py:545-554`). But:
   - State and action tensors are still required, and dummy tokens still enter attention.
   - Masked windows dilute the action loss denominator (`packing.py:564-569`).
   - There is no state mask. Clean action-free training needs about six code changes, listed in the notes file.
   - Not worth it for this hackathon.
5. **Units are the biggest trap.** FLUX SO-101 wants arm joints in degrees and the gripper in 0..100
   (https://docs.bfl.ai/flux_3/flux3_action_so101). There are three different setups in the wild:
   - LeRobot 0.5.1 defaults to degrees (`robots/so_follower/config_so_follower.py:42`).
   - LeRobot 0.3 to 0.4 defaulted to RANGE_M100_100 (ws4-flux-dual notes).
   - The XLeRobot plugin defaults to RANGE_M100_100 (ws4-flux-dual notes).

   `info.json` never says which. `inspect_dataset.py` decides per arm: any value beyond ±100.5 proves degrees,
   and two channels stopping at exactly ±100 is the RANGE_M100_100 signature. Results:
   - All picks below are degrees.
   - `siyulw2025`, `yihao-brain-bot`, `nbirukov`, `saivishwak/desk-marker` and others are ±100. Converting
     those needs the recorder's calibration spans: about ±3% for SO-101 joints, and unknown for XLeRobot
     wrist_roll, which is calibrated by a manual sweep.

## 2. Contracts

| | FLUX SO-101 package (use 1) | Our 12-D (use 3) |
|---|---|---|
| fps | 30 | 30 |
| state / action | 6 measured / 6 absolute commanded, gripper last | `[left 6 \| right 6]`, gripper at 5 and 11 |
| joint order | shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper | same per arm |
| units | arm degrees (LeRobot DEGREES: 0 = calibration midpoint), gripper 0..100 | same |
| cameras | `observation.images.scene` (left tile), `observation.images.wrist` (right tile), 256x256 each | `head`, `left_wrist`, `right_wrist` |
| format | LeRobot v3.0 | LeRobot v3.0 |
| head config (flux-action) | `joint_delta`, `absolute_action_dims [5]`, weights `[1,1,1,1,1,2]` | `joint_delta`, **`absolute_action_dims [5, 11]`** (the indexer default `-1` would delta-encode the left gripper), weights `[1,1,1,1,1,2]x2` |

Sources: docs.bfl.ai SO-101 page, `configs/so101/train.json:10-39`, `src/flux_action/cli.py:125-130`.
The bimanual head config is assembled from those constraints and has not been run.

## 3. What is already in FLUX and in `community_dataset_v3`

- **`community_dataset_v3` today:** 1,755 sub-datasets, 365 h, all v3.0 (current revision, not the 791 its
  README says). It is 87% so100/so101 single arm, with 13.7 h of 12-D bimanual SO-100. It contains no XLeRobot.
  Its camera keys were renamed by a curation pass (`camera_curation_collection.json`, 1,468 renamed).
- **Base midtraining:** the category mix only. Egocentric hands 13.54%, handheld 14.03%, teleop 15.93% over
  14 embodiments (https://bfl.ai/models/flux-3-action). No dataset is named anywhere: research page, HF card,
  docs or repo.
  - Plausibly already in it: EgoDex, the UMI zarrs, FastUMI-100K, GenRobot, AgiBot.
  - New to it: Pantheon DiverseUMI recordings from 2026-09-28 onward (after the 23 Sep release). See the
    human notes file, §1 and §3.1.

## 4. Ranked candidates

Abbreviations:
- deg = degrees.
- Viewpoint: ++ = XLeRobot head camera behind the arms (like ours), + = high camera with both arms in view,
  0 = top-down or other, − = front camera facing the robot.

All rows are 30 fps, LeRobot v3.0, Apache-2.0, unless noted. "Traj" = trajectory parquet size.

### Use 1: per-arm task LoRA on `flux-3-action-so101`

Each bimanual episode becomes two 6-D episodes: scene = head, wrist = that arm's wrist camera, task text
suffixed " with the left arm" / " with the right arm".

| # | dataset | eps / h | view | units | video GB / traj MB | why, flags |
|---|---|---|---|---|---|---|
| 1 | `zonglin11/xlerobot_clean_table` | 50 / 1.13 | ++ | deg | 3.8 / 5 | closest rig: XLeRobot head view, fin-ray-like fingers; drop `head_motor_*` (one is a constant 1500) |
| 2 | `cortx-labs/bimanual-fruit-sorting` | 50 / 2.13 | + | deg | 5.7 / 10 | sorting with many sub-steps (153 s episodes), good for long horizon |
| 3 | `bigse0u1/xlerobot_scrub_7tool` | 700 / 2.29 | ++ | deg | 4.4 / 8 | **action gripper inverted** (`--flip-action-gripper`, verified: slope −0.96 → +0.96); wrist cameras 240x320 |
| 4 | `rk000000/xlerobot-handkerchief-quarters-mix-20260918-480` | 74 / 0.76 | ++ | deg | 2.1 / 8 | head tilt moves (up to 61); cloth folding |
| 5 | `MrC4t/bi_so_bag` | 100 / 1.43 | 0 | deg | 8.1 / 8 | clean bi_so_follower contract, but "head" is a close top-down camera |
| 6 | `saivishwak/xlerobot-desk-marker-phase1` | 150 / 0.68 | ++ | ±100 | 3.6 / 5 | fin-ray-like fingers; needs approximate degree conversion |
| 7 | `cortx-labs/bimanual-trash-and-plates-sorting` | 51 / 1.22 | + | deg | 3.6 / 6 | same rig as #2 |
| 8 | `Elvinky/bi-so101-insert-screw-562ep` | 562 / 19.2 | + | deg | 21.5 / 80 | big, single task; **wrist_flex action/state slope 0.5** (leader/follower calibration mismatch) |
| 9 | `saivishwak/xlerobot-vr-teleop-medicine-bowl` | 100 / 0.68 | ++ | deg | 3.3 / 5 | VR teleop via IK |
| 10 | `Odog16/xlerobot_generalist_v2` | 236 / 2.30 | + | deg | 4.9 / 21 | mobile base moves in 23% of frames; floor picks; 360x640 |

Not recommended:
- `nitred/bimanual-so-stack-cube_100`: front camera facing the robots.
- `armnet/armnetbench_v01_lerobot_bimanual_so101`: 20 fps, 43 GB.
- `siyulw2025/xlerobot-20251026-alldata-v3.0`: ±100 units, mobile, 10 GB. It does have 22 tasks with explicit
  left/right arm language, the only such set.
- `shuaishuaicdp/so101-bimanual-v3-all-trajectories`: 46.5 h and 132 GB, a canonicalised merge without stats
  and without per-source provenance in the metadata. Its 59 tasks may be worth a later look.

### Use 2: EE50 fine-tune from `flux-3-action-base` (Simone's plan)

Blocked first by code (§1, finding 3). If the gate in §5 passes, the data ranking is:

| # | source | type | size | why |
|---|---|---|---|---|
| 1 | our own XLeRobot demos via FK | robot | n/a | the only data with our exact kinematics; FK error is ours to control |
| 2 | `cortx-labs/bimanual-fruit-sorting`, `zonglin11`, `MrC4t` via FK | robot | traj 25 MB | degrees, so FK works; FK needs LeRobot-zero = URDF-zero (ws4-ik to confirm for the SO-101 "new calib" URDF) |
| 3 | `axiboai/umi_stack_sohaib_no_roll` | bimanual UMI, wrist cams | 0.24 GB, 100 eps, 0.38 h | already xyz + rot6d + gripper, 30 fps; converted and tested here. rot6d row/column convention undocumented |
| 4 | `rocochallenge2025/roco_iros2026_umi_dataset` | bimanual handheld (ISuit-V3) | 69 h, 4,928 eps; traj 0.96 GB, video 249 GB | xyz + quaternion + gripper per hand, head stereo (`fpv_*`) + 2 cams per hand, 30 fps. **No licence stated**; tasks are industrial assembly. Converted and tested (40 eps) |
| 5 | Pantheon DiverseUMI sample25 | bimanual UMI, top + 2 wrist | about 300 pick-place eps ≈ 4.5 GB (est.) | best content match and partly new to FLUX; **manual HF gate**: request now |
| 6 | HiFi-UMI-2K | bimanual handheld + head stereo | smallest slice with video ≈ 7.3 GB | rot6d is already "first two rows", shared gravity-aligned frame, CC BY 4.0, 25 fps |
| 7 | FastUMI-100K `dual_arm` | bimanual UMI, wrist cams | about 250 eps in 5 GB | open, but 20 Hz, Euler angles, undefined relative frame, likely already in FLUX |

Human-hand sets (EgoDex, PH2D, HOT3D, ARCTIC) have no gripper (a thumb-index proxy is needed) and are likely
already in FLUX's 13.5% egocentric share. Skip them for a 24 h project.

### Use 3: native 12-D joint fine-tune (`[left 6 | right 6]`, 3 cameras)

Same data as use 1 but kept bimanual (`convert_bimanual.py --to xle12`). Ranking:
1. `zonglin11`
2. `saivishwak/desk-marker` (if converted)
3. `bigse0u1` (flip gripper)
4. `rk000000`
5. `MrC4t`
6. `saivishwak/medicine-bowl`
7. `cortx` fruit sorting (high camera with both arms in view, but not an XLeRobot body)

Caveat from the flux-action source: a 3-camera layout is either `grid` (2x2 with a black cell, "no checkpoint
was trained on it") or `droid` (trained, but needs 360x640 frames and puts the first key full resolution on
top). Owner: ws4-flux-dual.

### Use 4: video-only / world-model co-training

Possible only through a custom dataset plug with zero state/action and `action_mask = 0`, with the loss
dilution described above. Best video, if pursued:
- `cortx-labs/bimanual-fruit-sorting`: long, structured, sorting.
- The XLeRobot head-view sets.

No extra download is worth it tonight.

## 5. The EE50 gate (what must be true before anyone spends GPU time on Simone's plan)

1. **Head weights.** Someone with accepted access (the model repos are gated; this laptop is not logged in)
   lists the tensor names in the header of `flux-3-action-base.safetensors`. This needs only an HTTP range
   read of the first few MB, not the 14 GB file. Check that an EE50 head (`emb_in.*` / `final_layer.*`, 50
   channels) exists, and under which modality name.
2. **Conventions (all UNVERIFIED).**
   - The relative frame. ws4-ik's default is anchor frame `T_a^-1 T_t`; with it, each source's world frame
     cancels.
   - The gripper slot (index 9 assumed).
   - The z-score floor.
   - The tool-axis convention. `ee_convert.apply_tool_fix` lets every source be rotated to one convention.
3. **IK.** ws4-ik shows that 5-DoF IK tracks EE50 decodes of our own FK'd demos within tolerance.

If 1 fails, EE50 offers nothing over a fresh joint head, and BFL's own DROID result favours joint heads at
larger batch (40.13% vs 37.88%).

## 6. What to download tonight, in order

Run on the GPU VM, not on the laptop. Pin revisions (`candidates.yaml: revision`). Every item over 2 GB needs
Oscar's OK.

LeRobot main cannot train on several datasets at once (multi-dataset is disabled in `datasets/factory.py`), so
public data must be merged into one dataset. Its merge rule (`datasets/feature_utils.py:143-171`) ignores encoder
settings but **requires the same codec and frame size**. That splits the candidates:

| order | dataset | GB | for | status |
|---|---|---|---|---|
| 0 | trajectories + meta of 12 shortlisted sets | 0.25 | analysis, ws4-ik FK tests | done, `ws4/.cache/datasets/` |
| 1 | `zonglin11/xlerobot_clean_table` | 3.8 | uses 1, 3 | core (AV1, 480x640) |
| 2 | `cortx-labs/bimanual-fruit-sorting` | 5.7 | uses 1, 4 | core (AV1, 480x640) |
| | **core total** | **9.5** | 3.3 h, 100 episodes, ≈ 6.5 arm-hours after the per-arm split; merge-checked together | `fetch_public.sh` |
| 3 | `rk000000/xlerobot-handkerchief-quarters-mix-20260918-480` | 2.1 | uses 1, 3 | optional: H.264, so it cannot merge with the AV1 sets without a re-encode |
| 4 | `bigse0u1/xlerobot_scrub_7tool` | 4.4 | uses 1, 3 | optional: 240x320 wrist cameras, same problem |
| 5 | `axiboai/umi_stack_sohaib_no_roll` | 0.24 | use 2 smoke test | only if the EE50 track restarts |
| 6 (optional) | `MrC4t/bi_so_bag`, `saivishwak/xlerobot-desk-marker-phase1`, `cortx-labs/bimanual-trash-and-plates-sorting` | 15.3 | more variety | only if LoRA B helps; check codecs first |
| 7 (only if the EE50 gate passes) | Pantheon DiverseUMI subset (request access now), roco trajectories (0.96 GB) | ≈ 5.5 | use 2 | blocked |

Everything for the box is scripted in `ws4/data/fetch_public.sh` (core set; `EXTRA=1` adds 3 and 4). It
downloads at pinned revisions, converts per arm and to 12-D, and merges the compatible per-arm sets. The
step-by-step for the teammate who runs the GPU box is `ws4/HANDOFF_GPU.md`. Our own recordings must use AV1
at 640x480 (LeRobot main's default codec, UNVERIFIED on the recording PC) to merge with the core set for LoRA B.

## 7. Conversion plans

- **Bimanual LeRobot to per-arm** (`convert_bimanual.py SRC DST --to per_arm --side left|right`):
  - Joints are selected by name (`canonical_joint` handles `left_`, `left_arm_`, `.pos`, `main_`) and
    reordered to the contract.
  - Extra channels are dropped: head motors, base velocities, `gantry.height_mm`, effort.
  - Cameras: scene = head/top, wrist = that side's wrist. Video files are hard-linked under the new keys, and
    the episode metadata columns are renamed to match.
  - Stats are sliced exactly per dimension (no recompute needed).
  - Units: `--src-units auto` refuses ambiguous data. `m100` converts with
    `deg = v · span/200 · 360/4095` using our own calibration spans: exact zero, scale error about ±3% on
    other robots.
  - `--flip-action-gripper` for `bigse0u1`. The task suffix names the arm. No left/right mirroring: FLUX's
    corpus already trained unmirrored single arms of bimanual rigs (15 `left_`/`right_` 6-D parts, 11,229 windows).
  - Requires 30 fps source and v3.0 (v2.1: run lerobot's `convert_dataset_v21_to_v30` first).
  - Merging with lerobot's `merge_datasets` needs identical features. `bigse0u1` wrist videos are 240x320 vs
    480x640 elsewhere, so either keep it as a separate dataset or re-encode its wrists.
- **Bimanual / XLeRobot to 12-D** (`--to xle12`): same as above, both arms in `[left | right]` order, whatever
  the source order (`ArthurWangSawau` stores right first). Drop or skip windows where the base moves
  (`Odog16` 23%, `siyulw2025` 10%), since the head camera moves with the base.
- **UMI / handheld to EE50** (`ee_convert.py SRC --preset roco|axibo|umi_zarr --out X.npz --stats-out S.npz`):
  1. Each source goes to the common `ee20` per frame: per arm, position in metres, rotation as the first two
     rows of R, gripper open fraction 0..1. Quaternions (xyzw), rotation vectors and 6D (rows or columns) are
     supported.
  2. Grippers are normalised per dataset (1st..99th percentile → 0..1; which end is open is UNVERIFIED per
     source; UMI widths / 0.08 m).
  3. EE50 windows and per-offset z-scores then come from ws4-ik's `kinematics/ee50.py`.
  - fps: roco and axibo are 30. UMI zarr (about 60 Hz) and FastUMI (20 Hz) need resampling, which is not
    implemented.
  - Measured 1-second motion std (offset +30): UMI axibo 2 to 8 cm, roco 1 to 9 cm, `zonglin11` FK 2 to 3.5 cm.
    The scales are compatible, but human-held grippers move faster: consider time-scaling or per-embodiment
    z-scores (which EE50 does anyway).
- **Robot joints to EE50** (`--preset so101_joints`): FK through ws4-ik `Station`. Valid only for degree
  datasets, and only if the LeRobot calibration midpoint equals the URDF zero (open question to ws4-ik, sent).

## 8. How much of our own data, and the time split (judging rule: "justify your time split")

**Own data is required. Public data cannot replace it:**
- The target task and objects.
- Our black TPU fin-ray grippers.
- Our RealSense mast pose and intrinsics.
- Our calibration. The SO-101 LoRA keeps the package's normalization fixed, so the actions must be in our
  degrees.
- BFL's recipe asks for "normal starts, varied starts, disturbances, and recovery behavior" (BFL fine-tune
  docs), which only we can record on our setup.

**How much:** the shipped LoRA recipe is 10,000 microsteps at batch 2, i.e. 20,000 training windows
(`docs/so101-lora.md:166-167`). One hour of our own demos at 30 fps is about 100k start positions, already 5x
what one LoRA run samples. Data volume is not the bottleneck; coverage is. Recommendation:
- 40 to 60 demos per skill for the 3 to 4 skills of the demo chain, 10 to 30 s each.
- About 45 to 75 min of teleop in total, plus 10 held-out episodes per skill for evaluation.
- With leader arms and two people, about 2 to 3 h of wall-clock.
- Record with `use_degrees=True` and one fixed calibration file per arm.

**Why still add public data:**
- It adds what FLUX lacks and we cannot record in 24 h: about 900 episodes of XLeRobot-style head views with two
  arms in frame, from four labs, many objects and backgrounds.
- It costs almost no human time. The survey and converters are done, the download is about 16 GB on the VM, and
  conversion is minutes (no video re-encode).
- It also gives us an offline test set of our body layout before we touch hardware.

**Proposed split:**
- Human time: about 85% own data (record, clean, evaluate on hardware), about 15% public (already spent today).
- Training: public data at no more than 30 to 50% of LoRA windows.

**Evidence for the judges:** run LoRA A (own only) and LoRA B (own + 30% public per-arm XLeRobot) in parallel on
two of the three GPUs, with identical steps. Report:
- offline action error on held-out own episodes;
- hardware success over N trials.

Keep public data only if B wins.

## 9. Method notes

- **Hub survey:** `survey_hub.py --search ...` and `--collection lerobot/community_dataset_v3`. The caches
  (`ws4/.cache/hub_meta*`, gitignored) hold every `info.json` read. Anonymous Hub access hit HTTP 429 rate
  limits, and the FLUX model repos return 401. Logging in (`hf auth login` by Oscar) fixes both, and is needed
  anyway for the gated weights and the EE50 header check.
- **Units check on trajectories:** the state against action fit is about 1.0 slope in all 10 downloaded sets.
  Exceptions:
  - `bigse0u1` gripper: action inverted.
  - `Elvinky` wrist_flex: slope 0.5.
  - `Odog16` and `Elvinky` grippers: slope 0.6 to 0.9 (commanded closed, measured stopped by the object;
    expected for grasps, not a units problem).
- **Viewpoint:** judged from 2 episodes × all cameras per dataset (`sample_frames.py`, a few MB per dataset
  over HTTP range requests). Contact sheets are in `ws4/.cache/frames/`.
