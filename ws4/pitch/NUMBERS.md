# Pitch numbers ledger (single source for slides, video captions and the README)

Drafted Sat 17 Oct 2026, about 21:45 CEST, by the pitch-kit session. Every number in `ws4/pitch/**` must appear
here with its source. Rules:

- **MEASURED**: read from the cited file, which records a measurement or a logged attempt.
- **UNVERIFIED**: an estimate, a derived count, or a fact nobody has checked on the real system. Say so on the slide.
- **PENDING**: arrives overnight or Sunday morning. Every PENDING value has a placeholder `[[NUM:...]]` in the
  pitch files. Fill this table first, then replace the placeholders.

Find everything still open (numbers, team facts, clips):

```bash
grep -rn "\[\[" ws4/pitch
```

Refresh the trial table right before the slides freeze:

```bash
python ws4/eval/trial_log.py summary --markdown
```

## 0. Final demo, Sun 18 Oct: the 4-skill chain (read this section first)

Updated Sun 18 Oct about 09:30 CEST. Sections 1 to 13 below are the Saturday history; where they disagree, this
section wins.

### 0.1 The chain

Claude Opus plans, Claude Haiku checks each step from the head camera, the runner's guards bound every command.
**The box step is a scripted replay, not FLUX** (see below); the three left-arm steps are FLUX 3 Action, one LoRA per
skill on one shared base.

| # | Arm | Prompt (exact) | Dataset on the Hub (takes) | Haiku check on held-out frames, final scene |
|---|---|---|---|---|
| 1 | robot RIGHT (box arm) | "open the box by pulling its handle" | [Oskrt/haller_open_the_box](https://huggingface.co/datasets/Oskrt/haller_open_the_box) (75) | 63/63, no false pass |
| 2 | robot LEFT (pen arm) | "put the green pen in the green slot" | [Oskrt/haller_put_the_green_pen_in_the_green_slot](https://huggingface.co/datasets/Oskrt/haller_put_the_green_pen_in_the_green_slot) (66) | 114/119 (96 %), 2 false passes in 82 not-done frames |
| 3 | robot LEFT | "put the yellow cutter in the yellow slot" | [Oskrt/haller_put_the_yellow_cutter_in_the_yellow_slot](https://huggingface.co/datasets/Oskrt/haller_put_the_yellow_cutter_in_the_yellow_slot) (70) | 91/91, no false pass |
| 4 | robot LEFT | "put the red pen in its slot in the cardboard box" | [Oskrt/haller_put_the_red_pen_in_its_slot_in_the_cardboard_box](https://huggingface.co/datasets/Oskrt/haller_put_the_red_pen_in_its_slot_in_the_cardboard_box) (126 = 104 + 22 with the lid held) | 65/66, 1 false pass |

Sources: Hub `meta/info.json` of each dataset (read 18 Oct 07:20: 75 / 31,678 frames, 66 / 28,832, 70 / 30,514,
126 / 55,422; take-out 11 / 11,385); `ws4/research/orchestrator.md` check table (lines 166 to 169). All recorded by VR
teleop (box, green, cutter overnight, about 00:25 to 03:00 CEST at "tonight's spot", notes `final_demo_runbook.md`).
Planner: Claude plans "right arm first, then the three left-arm steps", checked live in 6.4 s for USD 0.02
(`ws4/orchestrator/README.md`). Chain result: 2 of 3 runs 4/4 (section 0.1b).

**Box = scripted replay, never call it FLUX.** The FLUX box LoRA went 0/3 live in the runner plus 0/1 in the HMI
(orchestrator, 09:00; the trial log has 20 `open_box` FLUX rows, all 0, all "stopped before: reach the handle"). It
barely moves from rest: the from-rest gate (first chunk must move >= 20 deg on every held-out take) failed for every
box variant, smallest move 4.9 deg (non-aug RAW), 5.3 (aug RAW), 5.5 (aug EMA) (`ws4/research/skills.md`, augmented
runs table). In the demo the box step replays a recorded demo (take 59 by default, take 72 in the final chain runs) through the same safety stack (1 deg/tick, the box
kit's keep-out with its envelope, stall freeze, torque cap; the start gate then checks the held lid for the pen
steps), labelled "SCRIPTED REPLAY OF RECORDED DEMO 59, NOT FLUX" in the HMI, events and trial log (condition
`scripted`, variant `scripted-replay-ep59`; `ws4/orchestrator/README.md`, `skills.yaml`). Say: "the planner mixes
learned skills and scripted primitives under the same safety layer". A cutter replay fallback (take 67) also exists.

Executed by, in the demo: box = scripted replay (take 72 in the final runs); green pen = FLUX non-augmented RAW 5k; yellow cutter = FLUX
augmented RAW 5k; red pen = FLUX RAW 5k on B84. All four FLUX adapters are public, root = the served adapter,
`checkpoints/<run>/<step>/{raw,ema}` for the rest: `Oskrt/flux-3-action-so101-haller-{pen,open-box,green-pen,yellow-cutter}-lora`
(Hub API lists all four, 18 Oct 09:20).

Each skill is its own LoRA adapter, about 0.45 GB, on one shared FLUX base (`ws4/research/pen_lora.md` setup:
"raw and EMA adapters (0.45 GB each)"). Hot-swapping adapters on one loaded base per step is future work.

### 0.1b The full chain on the real Haller (final, 18 Oct 11:46 to 12:10)

Source: `ws4/eval/trials.jsonl` (task `haller_chain`, variant `lora_chain_unguarded`, plus one line per step).
Unguarded demo mode (commit 18ece73, approved by Oscar for the demo; see 0.5).

| Run | Result | Time | Replans | Claude cost | Steps |
|---|---|---|---|---|---|
| 11:46 | 1/3 steps, operator abort | 185.9 s | 1 | USD 0.05 | box replay ok; green 2 attempts failed; cutter timeout |
| **11:52** | **4/4, no human touch** | 208.6 s | 0 | USD 0.02 | box replay 17.6 s; green FLUX 57.7 s; cutter FLUX 112.1 s; red pen check passed in 2.8 s |
| **12:05** | **4/4, no human touch** | 302.2 s | 1 (5.2 s) | USD 0.05 | box replay missed twice, opened on the replanned third try (17.1 s); green FLUX 55.1 s; cutter FLUX timeout, then inserted in 45.6 s; red pen check passed in 3.8 s |

- **2 of 3 runs completed 4/4 with no human touch during the run.**
- **The red pen was not placed by the red-pen LoRA in either run:** the green-pen policy slotted it during its
  return-home window, and the red step's check confirmed it in 2.8 s and 3.8 s. Say: "all four objects ended in place
  without human touch; the box by scripted replay, green and cutter by their FLUX LoRAs, the red pen placed by the
  green-pen LoRA". The red-pen LoRA's own numbers stay those of the single-skill runs (3/5, 2/3).
- The box step replays recorded take **72** in these runs (trial notes: "NOT FLUX: open_box: scripted replay of
  recorded demo 72"); earlier documentation names take 59 as the default.

### 0.2 First real-arm successes (red pen, robot left arm, pen84 table, 120 s attempts)

Source: `ws4/single_arm/benchmarks/red_pen_2026-10-18.md` (05:00 to 05:35 CEST, scored by Oscar after each
attempt) and `ws4/eval/trials.jsonl`.

| Block | Success | Steps (reach, grasp, insert) | How it ended |
|---|---|---|---|
| **FLUX RAW 5k, horizon 16, lift stall 25** | **3/5** full inserts (A 94 s, B 91 s, C 48 s) | 9/15 | D and A stopped by Oscar at 26 s and 43 s without a reach |
| FLUX RAW 5k, horizon 32 | 0/5 | 8/15 | stall freeze 4, keep-out stop 1 |
| FLUX EMA 10k, horizon 32 | 0/5 | 4/15 | stall freeze 4 |
| FLUX RAW 10k, horizon 32 | 0/5 | 3/15 | keep-out stop 3 |
| ACT pen84 100k | 0/5 | 8/15 (grasp and carry in 4) | timeout 5, never inserted |
| FLUX RAW 10k, lid-held scene | 0/5 | 0/15 | held still: the scene the LoRA never saw |
| FLUX RAW 5k, horizon 8, lift stall 25 | 0/2 | 0/6 | darkest room, arm still 70 % of the time |

- **Trial log total right after the benchmark:** RAW 5k at horizon 16 = **5/8** (3 more attempts at 05:35 to
  05:40: D success, A 2 steps, D success); all RAW 5k rows 5/16 (`trial_log.py summary --markdown`, 18 Oct 07:20).
  Quote 3/5 with the benchmark file, or 5/8 with the trial log; say which.
- **Caveats (say them):** small n; horizon 16 came together with the 25 deg lift-stall threshold, two changes at
  once; the successes ended with Oscar's stop key after the insert; lighting fell during the session.
- Why horizon 16 helps (benchmark notes): replanning every 16 commands (about 0.5 s); the arm still waits about
  0.6 s for each chunk. At horizon 32 the stall freeze ended 9 of 15 attempts (shoulder lift trailing at about two
  thirds of the 60 % torque cap, no contact).
- Zero-shot on the arm stays as in section 1: 0/16 graded attempts on Saturday. Zero-shot B84 and median were
  measured offline only (sections 4 and 0.4); no arm attempts were run (none in the trial log at 12:20).

### 0.2b This morning, per skill, live (trial log, `trial_log.py summary --markdown` at about 09:20)

| Skill | Model | Result | Notes |
|---|---|---|---|
| red pen | FLUX RAW 5k, horizon 16 | graded 2/3 (A, C inserts; B 0 steps) | runner, pen84 spot, 07:xx |
| red pen in the HMI | FLUX RAW 5k | 1 insert, 33.1 s, confirmed by Haiku ("stands upright ... immediately left of the green marker") | run `20261018_074841`; the next step passed in 4.2 s (2 chunks), likely the pen still in place: not counted; the third timed out twice; chain 4/6 steps |
| red pen, all RAW 5k rows so far | FLUX RAW 5k | 9/23 | trial log total, all horizons and sessions |
| green pen | FLUX non-aug RAW 5k | 1/2 (one full insert, 3/3 steps, 08:33; one no-reach) | the orchestrator's 1/1 predates the second attempt |
| yellow cutter | FLUX aug RAW 5k | 1/6 inserts in the trial log | orchestrator 09:00: "reached 5/5, grasped 4/5, 0/5 inserts"; the trial log scores those five as 0 steps and has a sixth attempt with a full insert (3/3). Reconcile before quoting |
| box | FLUX RAW 5k (all variants) | 0 (0/3 runner + 0/1 HMI per the orchestrator; 20 logged rows, all 0) | replaced by the scripted replay in the demo |
| red pen, lid held | FLUX RAW 10k | 0/5 (benchmark, held still) | no separate RAW 5k check was logged; in the chain runs the pen was already slotted by the green-pen policy |

### 0.3 The knife edge: FLUX's decision hangs on the wrist camera's view of the room

Source: notes `single_arm_handover.md` updates 06:50 to 08:00 (18 Oct, offline on saved live frames) and the
benchmark's "What each change did", item 1.

- Live, after the robot had been moved, RAW 10k returned hold chunks at home: 0/15 steps in the lid-held scene.
- A field-by-field diff of the live request against a recorded one found them identical in task bytes, image keys
  and order, shapes, dtypes, units and history; only the wrist image differed. Recorded images with live state and
  commands gave a reach (elbow +26.7 deg, gripper -96.5 %); live images with recorded state gave a hold.
- The live wrist frame flips hold and reach on a 1 px shift or a 1 deg tilt; recorded frames reach under the same
  shifts. Sweep: 0 of 27 wrist shifts and 0 of 10 seeds reach on the live room view; averaging never rescues it.
- Fixes: `ws4/single_arm/align_station.py` (live alignment against a recording's home view: wrist room offset, scale,
  brightness; the benchmark ran at offset -11 px, scale 0.97) and image augmentation (crop + colour jitter) in the
  new LoRAs. Pre-robot gates (`ws4/research/skills.md`, augmented runs): from rest the first chunk must move >= 20 deg
  on every held-out take, also under +-2 px shifts and 0.5x brightness. Cutter aug RAW 44.2 deg (pass, served);
  green aug RAW 73.3 deg (pass; the served green is the non-aug RAW 5k); box 4.9 to 5.5 deg for every variant (fail,
  hence the replay). Augmentation left offline MAE about where it was (within 0.2 to 0.6 deg per joint).
- Brightness also matters: wrist gain x1.6 to 2.0 made the gripper close (notes 08:00). Runbook target: brightness
  0.85 to 1.2 of the recording.

### 0.4 Offline, held out (same protocol for every row: 200 frames, moving frames only, guidance 1/1)

| Skill / model | Arm MAE (deg, mean) | Direction | Source |
|---|---|---|---|
| hold still (red pen) | 5.6 | - | `pen_lora.md` |
| red pen, zero-shot B84 / median | 5.5 / 4.8 | 66.5 % / 69 % (6 joints) | `pen_lora.md` |
| red pen, ACT 100k | 4.3 | 85 % arm (86 % 6 joints) | `pen_lora.md` "ACT baseline" |
| **red pen, LoRA RAW 5k** (the 3/5 model) | **3.24** | 87.5 % arm | `pen_lora.md` "Raw vs EMA and guidance at 5k" |
| **red pen, LoRA RAW 10k** (public on the Hub) | **3.15** | 88.6 % (6 joints) | `pen_lora.md` lid-held section, row "old held-out" |
| red pen RAW 10k on the lid-held takes | 5.57 (hold 9.24) | 80.2 % | same |
| box, LoRA RAW 5k | 5.11 (hold 6.7) | 73 % arm | `ws4/research/skills.md` |
| box, ACT 20k | 5.6 | 80 % arm | same |
| green pen, LoRA RAW 5k | 3.95 | 89.1 % (6 joints) | same |
| yellow cutter, LoRA aug RAW 5k (served) | 5.26 (2.5, 4.7, 7.1, 5.2, 6.8; gripper 11.3 %) | not reported | `skills.md` augmented runs |
| box / green, LoRA aug RAW 5k | 5.26 / 4.34 | not reported | same |

- RAW beat EMA in every comparison (pen 5k: 3.24 vs 3.62 deg, 87.5 vs 82 % arm direction; green and box likewise):
  at 1,250 updates a 0.999 EMA still carries much of the initial adapter.
- Guidance 3/3 and 4/1 gained nothing over 1/1 (arm MAE 3.83 and 3.67 vs 3.24) at twice the compute per chunk.
- Public adapter: https://huggingface.co/Oskrt/flux-3-action-so101-haller-pen-lora (RAW 10k per the orchestrator,
  18 Oct; the Hub table in `pen_lora.md` lists the uploads). How-to for others: `ws4/LORA_HOWTO.md`.

### 0.5 Safety per skill (chain)

**The final chain runs were unguarded** (commit 18ece73 "--unguarded demo mode", approved by Oscar for the demo):
policies ran without keep-out zones and without the stall, keep-out and load stops; the start gate, the lid precheck
and the ramp confirmation are off too (commit message). Joint limits, 1 deg per tick, the 60 % torque cap and the
e-stop stay on. Why (orchestrator): 17 of 20 keep-out stops in the morning came from the demonstrated-envelope zones,
not from collision zones, so the guards were stopping good motion. Say this plainly; the guarded numbers below are the
single-skill runs.


Source: `ws4/orchestrator/README.md` "What each step runs with", `ws4/single_arm/README.md` skills section.

- Each skill's workspace is built from its own demos (station zones plus the demonstrated envelope), with recorded
  overrides: box midline +0.17 m, green -0.23 m with a 2 cm trim near the box arm, cutter -0.21 m with 1 cm box-arm
  clearance, red pen (lid-held) -0.15 m with a 1.4 cm torso margin.
- Cutter: a 4 cm slow zone near the lid-holding arm (0.5 deg/tick, load freeze at 50 % for 0.2 s).
- Start gate before every pen-arm step, before anything moves: the box arm's LIVE pose against the skill's recorded
  hold plus a geometric check; a refusal ends the chain with the reason shown, both arms keep holding (the lid stays
  held).
- Box-arm takeover (`--hold-arm right`): the HMI takes over an arm already holding the lid without moving it.
- On the real arm this morning the keep-out refused ticks in 15 of 33 benchmark attempts and ended 7 of them
  (`red_pen_2026-10-18.json` per-attempt refusals; benchmark "How attempts ended"). No collision or damage was
  reported. Do NOT say "0 keep-out refusals": the dry runs had 0 (box ACT, `skills.md`), the real runs did not.

### 0.6b Team integration and media

- The teammates' behaviour-tree harness drove the real Haller through the ws4 bridge (branch
  `claude/harness-bridge-11f8b6`). Best run: the green pen placed. Runs then stopped on the harness's own perception
  consensus and lid-predicate timeouts, and later on the ws4 start gate (the box arm sagging under the lid). The Opus
  5.5 hybrid gate is wired in and verified in observation (continue for 15 steps); gated motion runs were refused by
  the start gate before moving (orchestrator, 18 Oct).
- Media for slides and the video: Google Drive folder "liminal", subfolder `00_highlights`: red-pen insert clips
  (`redpen_insert_layoutA_94s_flux_raw5k_h16.mp4`, `..._layoutB_91s_...`, `..._layoutC_48s_...`,
  `redpen_insert_morning_A_69s.mp4`, `redpen_insert_morning_C_36s.mp4`), imagination clips
  (`imagination_replay_imagined_vs_actual_ep101.mp4`, `imagination_lora5k_imagined_vs_actual_ep101.mp4`,
  `imagination_b84_vs_lora5k_ep101.png`), `knife_edge_wrist_view_training_vs_now.jpg`, `zeroshot_failure_strip.png`,
  `flux_in_sim_closed_loop_overlay.mp4` (listed 18 Oct 09:25).

### 0.6 Sim track (final, honest)

0/20 for every policy on "put the block in the cup" in MuJoCo: released A0, A0r-sim, LoRA EMA 1.6k, LoRA raw 2.4k.
The raw LoRA grasped and lifted the block in 2 of 20 but never released it in the cup
(`ws4/research/robottt_flux.md` 11.12, status 18 Oct 01:20).

## 1. Graded attempts on real hardware (zero-shot, as of 17 Oct 21:15)

Source: `ws4/eval/trials.jsonl` via `python ws4/eval/trial_log.py summary --markdown` (16 lines, run 17 Oct
21:40). Checkpoint for all rows: `black-forest-labs/flux-3-action-so101@0713a7a`.

| task | variant | success | steps done (mean) | human help | status |
|---|---|---|---|---|---|
| stack_cups (red SO-101) | A0 | 0/3 | 0.0/3 | 3/3 | MEASURED |
| stack_cups (red SO-101) | A0-5V (12 V servos on a 5 V brick, not a FLUX result) | 0/2 | 0.0/3 | 0/2 | MEASURED |
| put_pen_in_slot (Haller) | A0-E | 0/3 | 0.0/3 | 2/3 | MEASURED |
| grasp_cup (Haller) | A0-E | 0/3 | 0.0/2 | 3/3 | MEASURED |
| grasp_cup (Haller) | A0r-B | 0/3 | 0.0/2 | 1/3 | MEASURED |
| grasp_cup (Haller) | A0r-D | 0/2 | 0.0/2 | 1/2 | MEASURED |
| **total** | | **0/16** | | | MEASURED |

Variant key (`ws4/single_arm/README.md`): A0 = released package unchanged. A0r = same weights, normalization
quantiles recomputed from our own demos (B: as recorded; D: shoulder lift and wrist roll negated). A0-E = released
package statistics plus Oscar's joint mapping (lift = 90 - lift, elbow = elbow + 90).

**Sunday's real-arm session (plan, orchestrator 17 Oct):** 5 attempts each of zero-shot `A0r-B84` (the LoRA's exact
base: our quantiles from the 84 train takes), zero-shot `A0r-M` (median statistics of current-calibration SO-101
community datasets) and the LoRA at 5k microsteps, same layouts and budget (`ws4/single_arm/eval_pen.sh`,
`ws4/single_arm/README.md`). The zero-shot pen row becomes two rows: B84 and median. Until they run, the only
zero-shot pen row is A0-E (0/3). A teammate's own script with the median statistics "basically worked" on the Haller
in one run: anecdotal, not graded, do not quote as a result.

## 2. Post-training on real hardware (filled 18 Oct 07:30; see section 0.2)

| What | Value | Source |
|---|---|---|
| LoRA pen task success | RAW 5k at horizon 16: 3/5 (benchmark block), 5/8 including the three attempts right after | `ws4/single_arm/benchmarks/red_pen_2026-10-18.md`; `ws4/eval/trials.jsonl` |
| Steps done (reach, grasp, insert) | 9/15 in the benchmark block | same |
| Human interventions (as scored) | 0 in every block; the successes ended with Oscar's stop key after the insert | same, caveats |
| Top failure | no reach (D and A stopped at 26 s and 43 s); at horizon 32 the stall freeze (shoulder lift torque-limited) | same |
| Checkpoint used | RAW 5k = `lora_pen84` checkpoint 005000 on base `so101_haller_pen_B84` | same |
| Held-out eval loss of that checkpoint | 0.2096 | `ws4/research/pen_lora.md` |
| Zero-shot B84 / median on the arm | not run (offline only) | trial log, 12:20 |
| Full chain (Claude-planned) success, steps | 2 of 3 runs 4/4 (unguarded mode; section 0.1b) | trial log |
| Pen-out skill | not in the final chain (dataset of 11 takes is public) | orchestrator, 18 Oct |

## 3. The diagnosis: why zero-shot failed (MEASURED unless marked)

| Number | Value | Source |
|---|---|---|
| Released package normalization (state q01..q99) | shoulder lift 84.8 to 184.2 deg, elbow 61.4 to 178.9 deg; one community dataset (`VoicAndrei/so100_kitchen`), pre-2025 SO-100 degree convention | `ws4/flux/checkpoint_facts.md:35-47` |
| Our folded red arm in today's LeRobot degrees | lift -100, elbow 96 (outside the package range) | `ws4/single_arm/README.md`, "What happened" |
| A0 on the red arm | 191 deg of commanded lift travel in each of the first two attempts, both prompts nearly the same trajectory (6 to 20 deg mean per-joint difference) | same |
| One observation, A0 vs A0r (offline) | lift moves -102 to +1.6 in one chunk (A0) vs -102 to -87.7 (A0r); fastest step 4.6 vs 1.5 deg/tick | same, "A0 vs A0r offline" (one frame, not a hardware result) |
| Offline diagnostic: 200 frames of Oscar's pen demos (151 moving), released weights, 5 normalizations | see the A to E table below | `ws4/single_arm/README.md`, "Offline data-format diagnostic" |
| Robustness, seed 1 (168 moving frames) | B direction 86, 83, 70, 62, 83, 74 %; D 84, 81, 70, 60, 88, 70 % | same |
| Decision | LoRA base = B (no transform needed at train or deploy time) | same |

The A to E table (copy exactly; joints pan, lift, elbow, wrist flex, roll; gripper last):

| variant | MAE (deg; gripper %) | direction agreement | median rel. displacement error |
|---|---|---|---|
| hold (repeat current command) | 1.9, 4.4, 4.7, 2.9, 2.0; 4.4 | - | 100 % |
| A package stats | 4.3, **19.9**, 10.1, 11.2, 3.7; 6.6 | 42, 57, 53, 58, 34, 78 % | 195, 498, 242, 291, 177, 56 % |
| B our quantiles (A0r) | 1.9, 3.2, 3.8, 3.7, 1.8; 8.4 | 80, 79, 68, 60, 73, 72 % | 46, 73, 86, 100, 68, 102 % |
| C issue-2 adapter | 4.4, **16.8**, 6.0, 10.1, 1.9; 6.5 | 41, 43, 70, 64, 77, 82 % | 208, 476, 127, 283, 56, 57 % |
| D flips + our quantiles | 1.9, 3.2, 3.9, 3.5, 1.6; 7.6 | **80, 83, 70, 62, 81, 80 %** | 52, 76, 87, 99, 58, 100 % |
| E Oscar's mapping + package stats | 4.3, 16.7, 5.9, 9.6, 3.2; 6.4 | 41, 43, 69, 64, 33, 82 % | 205, 483, 134, 286, 158, 54 % |

Issue link: https://github.com/black-forest-labs/flux-action/issues/2 (the experimental adapter tested as C).

What remains open after the fix (MEASURED on the Haller, `ws4/single_arm/README.md`, real-arm zero-shot):
wrist flex sat at its 93.5 deg soft limit for 1,192 and 1,177 of 1,800 ticks (A0r-B #2, #3) and 1,372 of 1,800
(A0r-D), beyond the demos' 78.5 deg maximum. Convention problem or zero-shot ignorance: **UNVERIFIED**.

## 4. Data and training

| Number | Value | Status and source |
|---|---|---|
| VR teleop takes recorded (pen into slot) | 104 takes, 46,372 frames, 30 fps | MEASURED, `ws4/.cache/real_lora/review_final.json` and `pen_in_right_final/meta/info.json` (gitignored, laptop only) |
| Takes rejected on review | 8 (pen on the mat, reversed, hand in the box, never released) | MEASURED, same file |
| Takes kept | 96 | MEASURED, dataset `ws4_split.json` "final" (21:43 CEST, `~/.cache/huggingface/lerobot/ws4/haller_put_the_red_pen_in_its_slot_in_the_cardboard_box/ws4_split.json`) |
| LoRA split | 84 train, 12 held out (last 3 good takes of each layout: A 20 to 22, B 36 to 38, C 86 to 88, D 101 to 103), 8 dropped | MEASURED, same file; LeRobot reports 84 train / 12 eval; `ws4/research/pen_lora.md` |
| Layouts | A, B, C, D (4 layouts; 24, 15, 50 and 15 takes) | MEASURED, same file |
| Data volume | 46,372 frames, 0.43 h | `ws4/research/pen_lora.md` setup table |
| LoRA base `so101_haller_pen_B84` | variant B quantiles from the 84 train takes (38,686 frames), no joint transforms | `ws4/research/pen_lora.md` |
| Public datasets | https://huggingface.co/datasets/Oskrt/haller_put_the_red_pen_in_its_slot_in_the_cardboard_box and https://huggingface.co/datasets/Oskrt/haller_take_the_red_pen_out_of_its_slot | orchestrator, 17 Oct 22:50; both answer publicly (HF API 200, checked 17 Oct) |
| Teleop rig | Meta Quest Pro, haller_ws HMI in solo mode on the task arm, h264 live then AV1 offline | `ws4/research/vr_teleop.md`, `ws4/PLAN.md` "Data" |
| LoRA recipe | LeRobot `examples/flux3/lora.json`: rank/alpha 32, effective batch 8 (2 x 4), lr 1e-4, heads x5, EMA 0.999 | MEASURED config, `ws4/cloud/real_lora.sh` header |
| LoRA pod | `real-train-6000`, 1x RTX PRO 6000 96 GB, EUR-IS-1, $2.49/h | `ws4/cloud/RUNPOD.md`, real-track LoRA section |
| Step time and memory | 1.02 s per microstep, 34.4 GiB peak, about 3,000 microsteps per hour with eval and checkpoints | MEASURED, same |
| Overnight run `lora_pen84` | 20,000 microsteps = 5,000 updates (about 1 epoch of the train frames), started 19:50:20 UTC on `real-train-6000`, about 1.0 s per microstep, ETA about 02:35 UTC | MEASURED start, `ws4/research/pen_lora.md`; ETA is an estimate |
| Cost of that run | about $17 at $2.49/h | UNVERIFIED estimate, `ws4/cloud/RUNPOD.md` |
| Pipeline test (provisional 28-take run) | checkpoint 500: eval loss 0.285 on held-out takes; merged package 13.9 GB; serves at 0.80 s per chunk while sharing the GPU | MEASURED, same |
| Checkpoints evaluated | to 10k microsteps (RAW 10k eval loss 0.213); later checkpoints not reported in the sources | MEASURED, `ws4/research/pen_lora.md` |

### Held-out action error: zero-shot B vs LoRA (offline, no robot)

Source: `ws4/research/pen_lora.md` "Held-out action-error curve" (`ws4/cloud/lora_adapter.py eval` running
`offline_diag.py` unchanged on the 12 held-out takes; the same 200 frames per row, 181 moving; guidance 1.0; EMA
adapter folded in). Raw results on the `real-train-is` volume, `/workspace/real_lora/eval/`.

| Model | Microsteps | MAE pan, lift, elbow, wflex, roll (deg); gripper (%) | Direction agreement | Median rel. displacement error |
|---|---|---|---|---|
| hold (repeat the current command) | - | 3.8, 7.6, 5.9, 7.6, 3.2; 8.5 | - | 100 % |
| zero-shot B (base, no LoRA) | 0 | 4.6, 6.3, 4.9, 8.0, 3.7; 11.3 | 65, 71, 68, 54, 75, 66 % | 152, 88, 81, 171, 70, 93 % |
| LoRA EMA | 2,500 | 3.0, 5.1, 4.7, 6.0, 2.9; 9.9 (arm mean 4.3) | 72, 77, 78, 64, 75, 78 % (mean 74) | 71, 68, 71, 96, 63, 90 % |
| LoRA EMA | 4,000 | 2.4, 4.5, 4.3, 5.4, 2.5; 8.4 (arm mean 3.8) | 76, 82, 80, 71, 82, 81 % | 53, 53, 63, 83, 64, 87 % |
| **LoRA EMA (Hub tag `step-5000`, served as `lora_5000`)** | 5,000 | **2.1, 4.4, 4.2, 5.0, 2.4; 6.7 (arm mean 3.6)** | **79, 84, 83, 75, 89, 81 % (mean 82; arm 82)** | 46, 48, 60, 75, 64, 79 % |
| LoRA RAW | 5,000 | 1.9, 4.0, 4.0, 4.0, 2.3; 4.7 (arm mean 3.24) | 89, 88, 87, 85, 89, 62 % (arm 87.5) | 47, 44, 55, 46, 53, 72 % |
| LoRA EMA | 10,000 | 1.7, 4.0, 3.9, 4.2, 2.2; 4.3 | 91, 87, 88, 84, 89, 84 % | 42, 40, 45, 53, 58, 78 % |
| LoRA RAW (public) | 10,000 | 1.7, 4.0, 3.8, 4.3, 2.1; 4.5 | 93, 85, 88, 84, 89, 92 % | 45, 41, 49, 50, 53, 55 % |
| ACT 100k (final) | - | 2.7, 5.4, 5.1, 5.1, 3.0; 4.7 (arm mean 4.3) | 86, 85, 88, 78, 87, 94 % (arm 85) | - |
| ACT baseline (ws4-act), 10k steps | - | arm MAE mean 4.3 | 81 to 90 % | - |

### Zero-shot with median community statistics (variant M), same 12 held-out takes

Source: `ws4/research/pen_lora.md` "Zero-shot normalization variants" (MEASURED by ws4-datasets; 181 of 200 frames
moving). M = stock weights with the median statistics of current-calibration SO-101 community datasets from
multimodalart's Space (`huggingface.co/spaces/multimodalart/flux-3-action-so101-sim`, commit `09edebb`, app.py
NORMALIZATION). Its gripper range is 0..40 where ours is 0..100, so its gripper numbers are not like for like.

| Zero-shot | Arm MAE pan, lift, elbow, wflex, roll (deg), mean | Arm direction, mean | Gripper MAE / direction | Direction, 6 joints |
|---|---|---|---|---|
| hold | 3.8, 7.6, 5.9, 7.6, 3.2 (5.6) | - | 8.5 % / - | - |
| B84 | 4.6, 6.3, 4.9, 8.0, 3.7 (5.5) | 65, 71, 68, 54, 75 % (67) | 11.3 % / 66 % | 66.5 % |
| M (median) | 2.7, 6.7, 5.4, 5.3, 3.8 (4.8) | 74, 64, 72, 77, 42 % (66) | 6.8 % / 87 % | 69.3 % |

Path speed vs the demos (model path over demo path, per joint, all 127 moving held-out chunks, ws4-imagine's
`speed_check.py`, reproduced by ws4-datasets):

| Zero-shot | Path ratio pan, lift, elbow, wflex, roll, gripper | Displacement ratio |
|---|---|---|
| B84 | 1.49, 0.32, 0.55, 1.06, 0.99, 1.00 | 1.83, 0.30, 0.53, 1.37, 1.40, 0.78 |
| M (median) | 0.62, 0.65, 0.83, 0.89, 0.57, 0.87 | 0.57, 0.70, 0.83, 1.02, 0.72, 0.73 |

Honest reading: M has the lowest arm MAE (4.8 vs 5.5 for B84 and 5.6 for holding still), fixes B84's pan over-swing
(0.62x vs 1.49x) and doubles lift motion (0.65x vs 0.32x), so every joint moves at 0.57 to 0.89 of the demo path:
the most balanced motion. But its overall direction agreement is no better (69 vs 67 %) and wrist roll falls below
chance (42 %). As a LoRA base it missed ws4-datasets' GO bar (direction of at least 80 %), so the LoRA stays on B84.
Story line: "the released statistics fail; two ways of fixing them, with no training, give sensible predictions;
median gives the most balanced motion."

Held-out eval loss (LeRobot flow-matching loss, 256 samples): 0.261 at 500, 0.233 at 2,500, 0.219 at 4,000, 0.210 at
5,000 microsteps (MEASURED, `ws4/research/pen_lora.md`). Public checkpoint:
https://huggingface.co/Oskrt/flux-3-action-so101-haller-pen-lora (tag `step-5000`, eval loss 0.2096; it holds only the
latest best EMA adapter).

Speed, model path over demo path (127 moving held-out chunks, `pen_lora.md`): B84 pan, lift, elbow, wflex, roll 1.49,
0.32, 0.55, 1.06, 0.99; LoRA 5k 0.76, 0.78, 0.64, 0.60, 0.55 (gripper 1.02). The LoRA learned to reach down (lift
0.32x to 0.78x) and stopped over-swinging pan (1.49x to 0.76x), but every arm joint still moves slower than the demos
(0.55 to 0.78x). The ACT row is cited in `pen_lora.md` from ws4-act; its own report section is
not in the repo yet: UNVERIFIED until it lands.

Reading at 5,000 microsteps (1,250 updates): arm MAE 3.6 deg (B84 5.5, median 4.8, hold 5.6, ACT 4.3), direction 82 %
(B84 66.5 %, ACT 81 to 90 %), gripper MAE 6.7 %, now better than holding still (8.5 %). Say: "at a quarter of the
run, the LoRA has a third less arm error than zero-shot, matches ACT on direction and has lower arm error than ACT".

Earlier reading at 2,500 microsteps (625 updates): the LoRA beats both zero-shot variants on arm MAE (4.3 vs 5.5 for B84, 4.8
for median, 5.6 for holding still) and on direction (74 % vs 66.5 % and 69 %); its gripper MAE (9.9 %) is still worse
than holding still (8.5 %). ACT is ahead on direction (81 to 90 %) at the same arm MAE. Keep that comparison on the
slide: FLUX LoRA vs ACT is a real result either way; final numbers come from the 20k checkpoint and the arm.

Reading (B84): zero-shot B is about as good as holding still in magnitude (better on lift and elbow, worse on pan, wrist
flex, roll and gripper) and better than chance on direction. The slide claim to make, if the curve supports it:
"from about as good as holding still to clearly better than holding still, on takes it never trained on".
| FLUX SO-101 training corpus | 1,021 parts, about 146 h, zero XLeRobot data, 11 of 1,021 scene cameras are head or overhead | `ws4/research/datasets.md:42-50` (146 h is that report's estimate: UNVERIFIED) |

## 5. Serving latency

| Setting | Value | Source |
|---|---|---|
| RTX PRO 4500, default guidance 3 | 1.46 s per chunk on the server, 1.78 s laptop round trip | MEASURED, `ws4/PLAN.md` "Risks" table |
| RTX PRO 4500, guidance 1.0 + compile | 0.63 s per chunk on the server | MEASURED server time; round trip about 0.94 s is an estimate |
| RTX PRO 6000, guidance 1.0 + compile (red arm runs) | 0.45 to 0.6 s per chunk round trip | MEASURED, `ws4/single_arm/README.md` |
| RTX 4090, guidance 1.0 (Haller runs) | 0.92 to 0.94 s per chunk round trip | MEASURED, same |
| One chunk | 42 actions predicted, 32 executed at 30 Hz (1.07 s of motion) | `ws4/CONTEXT.md` (package config) |
| Two arms on one shared model | arms alternate, 30 % and 28 % motion each, never simultaneous; 1.76 s round trip | MEASURED, `ws4/research/orchestrator.md` |
| Sunday demo server | real-serve-6000b (RTX PRO 6000), guidance 1, compile, chunk 42 at horizon 16; the arm waits about 0.6 s per chunk | `ws4/single_arm/benchmarks/red_pen_2026-10-18.md`; notes `final_demo_runbook.md` |

## 6. Imagination (FLUX's predicted video)

| Number | Value | Source |
|---|---|---|
| Frames per chunk | 11 predicted latent frames decode to 41 frames, 256 x 512 (scene and wrist side by side), control offsets +2 to +42 (0.07 to 1.4 s ahead) | `ws4/research/imagination.md` status report and section 1.7 |
| Actions with imagination on vs off | **bit-identical**, with `compile_model` on and off: max abs diff 0.0 over fresh processes (lora_500, guidance 1.0, seeds 0 to 4), and on 23 chunks of held-out takes 101 to 103 with B84 (compile off). The 0.255 command-unit difference ws4-datasets saw was between two separately compiled server processes, not caused by imagination (compile on vs off alone differ by up to 0.70). Say: "Decoding FLUX's imagined video leaves its actions unchanged (bit-identical, compile on and off)" | MEASURED by ws4-imagine on `real-serve-6000`, evidence `/workspace/imagine/compile_check/`; `ws4/imagine/README.md` status |
| Decode cost | lazy mode adds no round trip (median 0.39 s with imagination vs 0.39 to 0.43 s plain); decoding the 41 frames takes 0.11 to 0.12 s on the RTX PRO 6000, after the take. Earlier sync / async decode on the RTX PRO 4500: 0.23 s / about 0.2 s, 17.6 GiB peak | MEASURED, `ws4/imagine/README.md`; `ws4/PLAN.md` "Risks" |
| Prior public demos of decoded FLUX imagination | none found; BFL docs say the API "returns actions only" | `ws4/research/imagination.md` 1.1 ("we would be first" is our claim: phrase as "we found none") |
| Imagined vs actual clips | replay of held-out takes 101, 102, 103 (layout D, never trained on) through FLUX with B84 real weights: `ws4/imagine/outputs/replay_ep10{1,2,3}/imagination/imagined_vs_actual.mp4` (H.264, 1048x340, 8 to 10 s, gitignored), contact sheets `replay_ep101/contact_chunk1.png`, `contact_chunk4.png`. A clip from a live Sunday attempt is optional: `[[CLIP:IMAGINED_VS_ACTUAL]]` | HAVE (replay), `ws4/imagine/README.md` "First clips" |
| Imagination accuracy (3 held-out takes) | scene camera: the imagined frame is closer to reality than "nothing moves" on 80 % of (chunk, offset) pairs; wrist camera only 43 % (frames blur within about 0.5 s); under about 0.4 s ahead it looks near-identical | MEASURED, `ws4/imagine/README.md` accuracy table (mean abs RGB at 64x64 vs the VAE's held still frame) |
| Fine-tuning improves the imagination | wrist-camera error of imagined frames, B84 zero-shot vs LoRA 5k: +2..+8 (0.07 to 0.27 s) 0.064 to 0.050 (-22 %), +9..+20 0.138 to 0.113, +21..+32 0.205 to 0.173, +33..+42 (1.1 to 1.4 s) 0.249 to 0.206 (-17 %); scene camera about -5 %. Zero-shot imagines a smeared, broken pen by +1 s; the 5k LoRA imagines it held cleanly, as it was. Caveat: the 5k replays ran with compile on, the others with compile off | MEASURED, `ws4/imagine/README.md` (table "Wrist-camera error by training stage"), `outputs/offset_errors_b84_to_lora5000.json`; slide image `ws4/imagine/outputs/b84_vs_lora5000_ep101_chunk1.png`; clips `outputs/lora5000_ep10{1,2,3}/imagination/imagined_vs_actual.mp4` |
| Production config (live arm server, `lora_5000`, compile on) | commands identical with lazy imagination on all 23 chunks; round trip 0.34 s with imagination vs 0.34 to 0.38 s without; decode 0.12 s | MEASURED, `ws4/imagine/README.md` (21:45 UTC) |

## 7. Safety stack

| Number | Value | Source |
|---|---|---|
| Per-tick step limit | 1 deg per tick (30 deg/s) on arm joints, 5 % per tick gripper | `ws4/single_arm/README.md`, "Safety layers" |
| Soft limits | 5 deg inside the calibrated stops | same, Haller section |
| Torque cap | 60 % on arm joints (servo RAM register) | same |
| Stall freeze | any arm joint 8 deg behind its command for 0.5 s ends the attempt, hold at the measured pose | same |
| Load freeze | 70 % of the torque cap for 9 ticks, or 90 % once | `single_arm_handover.md` (notes folder) and commit 4396989 |
| Keep-out | FK zones: table, torso box, mast, midline, parked other arm as link capsules (4 cm clearance), demonstrated joint and TCP envelope | `ws4/single_arm/README.md`, Haller table |
| Keep-out vs demos | 0 refusals on 24 good pen-in takes (15,833 steps) and 9 pen-out takes | MEASURED, same |
| Keep-out vs the damaging red-arm attempts (replay) | refused at 0.97 s and 1.27 s (elbow load passed 50 % at 7 s and 8 s) | MEASURED, same |
| Peak load on the Haller during FLUX runs | 29 % of max torque | MEASURED, same |
| Red arm before the guards existed | 92 % load with the overload bit (elbow servo damaged) | MEASURED, same |
| Parked arm pose unknown | e-stop latches before FLUX starts (fails closed) | `orchestrator_handover.md`, commit 5519d47 |

## 8. Claude orchestrator

| Number | Value | Source |
|---|---|---|
| Planner | Claude Opus 5.5, structured output, validated plan (schema, vocabulary, DAG, deadlocks), errors fed back at most twice | `ws4/research/orchestrator.md` report |
| Planner latency | 12.1 s cold, 5.9 s with the cached system prompt (sim head frame) | MEASURED live, same |
| Step checks | Claude Haiku 5.5, 1.2 to 2.1 s, 4 of 4 correct on sim frames; Sonnet 5.5 lid check 1.7 s, correct | MEASURED live (sim renders only), same |
| Live check cost | USD 0.04 in total | MEASURED, same |
| Sim end to end | 4 random layouts, 3/3 steps each (scripted IK stand-ins behind the FLUX interface, not FLUX) | MEASURED, same |
| Tests | 58 CPU tests | same |
| Claude checks on recorded Haller frames (offline) | put 97.1 %, take 98.2 %, no false pass on 278 negatives (52 with the pen still in the gripper); held-out odd episodes 97.8 % and 99.1 %; median 1.3 s and 1.2 s; about USD 0.00018 per check; the generic prompt before tuning passed 2 of 107 true put frames | MEASURED, `ws4/research/orchestrator.md` "Success checks on real frames" (282 head-camera frames, Haiku 5.5, tuned station prompt). Live accuracy: not measured |

## 9. TTT progress memory (research result, sim)

| Number | Value | Source |
|---|---|---|
| Gated TTT layers after FLUX joint blocks 24 to 27 | 17.9 M params | MEASURED on the pod, `ws4/PLAN.md` decisions table |
| Gate 0 | bit-identical to stock FLUX on a real observation | same |
| Cost | +32 ms per chunk (+4.2 %, 0.758 to 0.790 s at guidance 1.0), +0.08 GiB VRAM | same |
| Stage accuracy, held-out sim episodes, 3 seeds (repeated stages in brackets) | frame features: none 0.40 (0.39), GRU 0.72 (0.62), TTT 0.75 (0.70); **FLUX features: none 0.54 (0.15), GRU 0.87 (0.74), TTT 0.90 (0.80)** | same; `ws4/research/robottt_flux.md` section 11.9 |
| Caveat | on failed A0 rollouts, monitors trained only on scripted demos invented progress: "in cup" on 43 % (GRU) and 37 % (TTT) of re-plans; adding 12 failed rollouts to training fixed it (0.98 to 0.99) | `ws4/research/robottt_flux.md` 11.12 |
| How to say it | "a memory over FLUX features tracks the task stage at 0.90 vs 0.54 without memory (GRU 0.87, TTT 0.90), sim only". Never imply TTT alone makes the difference. TTT's case is constant-size state over long contexts; our episodes are too short to show it. The live chain uses Claude Haiku checks, not this monitor | orchestrator, 17 Oct |

## 10. Sim track

| Number | Value | Source |
|---|---|---|
| Scripted demos, "put the block in the cup" | 60 episodes, 15,578 frames (60 of 137 layouts; 65 out of reach, 12 missed) | `ws4/research/robottt_flux.md` 11.12 |
| A0 in sim, 20 held-out layouts | 0/20; cup knocked over in 8, block pushed in 2; closest approach median 12 cm | MEASURED, same |
| A0r-sim and LoRA-sim, same 20 layouts | A0r-sim 0/20; LoRA EMA 1.6k 0/20; LoRA raw 2.4k 0/20 (grasped and lifted in 2, never released) | MEASURED, `ws4/research/robottt_flux.md` 11.12 (18 Oct 01:20) |

## 11. Kinematics (EE50 feasibility, research)

| Number | Value | Source |
|---|---|---|
| FK/IK round trip on real robot motion | 25,168 frames, 100 % success, p99 error 1e-7 mm, 1 to 3 ms per arm | `ws4/PLAN.md` decisions table, `ws4/research/ee50_ik.md` |
| Human handheld chunks position-feasible | 98 %, median unexecutable rotation 0.5 deg | same |

## 12. Cost (fill in the morning)

| Item | Placeholder |
|---|---|
| GPU spend, all pods | `[[NUM:GPU_SPEND]]` (RunPod billing; keep invoices) |
| Claude API spend | `[[NUM:CLAUDE_SPEND]]` (the HMI shows live spend) |

## 13. Second skill dataset

| Number | Value | Source |
|---|---|---|
| "take the red pen out of its slot" VR takes | 11 takes, 11,385 frames, 30 fps | MEASURED, `~/.cache/huggingface/lerobot/ws4/haller_take_the_red_pen_out_of_its_slot/meta/info.json` |

## 14. Placeholder index (who fills what, and when)

Find what is still open: `grep -rn "\[\[" ws4/pitch`. Times are CEST, Sun 18 Oct.

| Placeholder | Meaning | Owner | Expected |
|---|---|---|---|
| `[[CLIP:HMI_CHAIN]]` | screen recording of the chain in the HMI | demo-rehearsal | with the chain run |
| `[[CLIP:IMAGINED_VS_ACTUAL]]` | optional live imagination clip (replay clips exist) | demo-rehearsal / ws4-imagine | with the chain run |
| `[[NUM:GPU_SPEND]]`, `[[NUM:CLAUDE_SPEND]]` | spend for the cost line | Oscar (RunPod billing, HMI) | 13:00 |
| `[[NUM:RECORD_HOURS]]`, `[[NUM:TRAIN_HOURS]]` | time split, if wanted beyond section 0.1 | Oscar | 13:00 |
| `[[TEAM:NAME]]`, `[[TEAM:MEMBERS]]`, `[[TEAM:WORKSTREAMS]]`, `[[TEAM:WS1]]` to `[[TEAM:WS5]]`, `[[TEAM:WS1_WHO]]` to `[[TEAM:WS5_WHO]]`, `[[TEAM:CREDITS]]` | team facts | Oscar / team | as early as possible |
| `[[TEAM:VIDEO_URL]]`, `[[TEAM:SLIDES_URL]]` | links for the README and close slide | Oscar | 13:30 |
| `[[TEAM:DEMO_LAYOUT]]` | live-demo layout | Oscar | 12:30 |
