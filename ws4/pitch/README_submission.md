<!--
Draft submission README, Sat 17 Oct 2026 about 22:30 CEST. Written for the ROOT of Javgilavi/liminal on main (links
are repo-root relative). In the morning: fill every [[...]] placeholder (grep -rn "\[\[" ws4/pitch), re-run
`python ws4/eval/trial_log.py summary --markdown` and paste it under "Results", then copy this file to README.md at
the repo root (or link it from there). Numbers and their sources: ws4/pitch/NUMBERS.md. Delete this comment.
-->

# liminal: a robot that imagines, then acts

**ERH Physical AI Hackathon 2026, track "Imagine, Then Act" (Black Forest Labs FLUX 3 Action).**
Team `[[TEAM:NAME]]`: `[[TEAM:MEMBERS]]`.

Demo video: `[[TEAM:VIDEO_URL]]` · Slides: `[[TEAM:SLIDES_URL]]` · Data:
[pen into slot](https://huggingface.co/datasets/Oskrt/haller_put_the_red_pen_in_its_slot_in_the_cardboard_box),
[pen out of slot](https://huggingface.co/datasets/Oskrt/haller_take_the_red_pen_out_of_its_slot)

Our green XLeRobot "Haller" (two SO-101-style arms on a torso, a RealSense D455 head camera, wrist cameras, soft
fin-ray fingers) puts a red pen into its slot in a cardboard box with **FLUX 3 Action in the control loop**. A Claude
orchestrator chains the skill and checks each step from the camera, a safety stack bounds every command, and an
imagination panel shows the future video FLUX predicts next to what actually happens.

| | Zero-shot (released checkpoint) | Post-training (our LoRA) |
|---|---|---|
| Pen into slot, Haller | our quantiles (B84) `[[NUM:B84_PEN_SUCCESS]]`, median statistics `[[NUM:MEDIAN_PEN_SUCCESS]]`, package statistics + mapping (A0-E) 0/3 | `[[NUM:LORA_PEN_SUCCESS]]` |
| All graded real-arm attempts so far | 0/16 | `[[NUM:LORA_ALL]]` |
| Sim, "put the block in the cup", 20 held-out layouts | 0/20 | `[[NUM:SIM_LORA]]` |

## What we found: zero-shot failed on a calibration contract, not on skill

The released `flux-3-action-so101` package normalizes joints with quantiles from one community dataset recorded in
the pre-2025 SO-100 degree convention (shoulder lift 85 to 184 deg, elbow 61 to 179 deg;
[checkpoint_facts.md](ws4/flux/checkpoint_facts.md)). With today's LeRobot calibration our folded arm reads lift
-100 deg: the model sees an impossible robot and drives the shoulder to its limit whatever the prompt
(191 deg of commanded lift per attempt on our first arm).

We measured five normalizations offline on 200 frames of our own demonstrations, released weights unchanged, before
touching the robot again ([offline_diag.py](ws4/single_arm/offline_diag.py),
[single_arm/README.md](ws4/single_arm/README.md#offline-data-format-diagnostic-which-normalization-for-the-lora-base-2026-10-17-no-robot)):

| variant | MAE pan, lift, elbow, wrist flex, roll (deg); gripper (%) | direction agreement | median rel. displacement error |
|---|---|---|---|
| hold (repeat current command) | 1.9, 4.4, 4.7, 2.9, 2.0; 4.4 | - | 100 % |
| A package statistics | 4.3, **19.9**, 10.1, 11.2, 3.7; 6.6 | 42, 57, 53, 58, 34, 78 % | 195, 498, 242, 291, 177, 56 % |
| B our quantiles (A0r) | 1.9, 3.2, 3.8, 3.7, 1.8; 8.4 | 80, 79, 68, 60, 73, 72 % | 46, 73, 86, 100, 68, 102 % |
| C [flux-action issue #2](https://github.com/black-forest-labs/flux-action/issues/2) adapter | 4.4, **16.8**, 6.0, 10.1, 1.9; 6.5 | 41, 43, 70, 64, 77, 82 % | 208, 476, 127, 283, 56, 57 % |
| D sign flips + our quantiles | 1.9, 3.2, 3.9, 3.5, 1.6; 7.6 | 80, 83, 70, 62, 81, 80 % | 52, 76, 87, 99, 58, 100 % |
| E manual offset mapping + package statistics | 4.3, 16.7, 5.9, 9.6, 3.2; 6.4 | 41, 43, 69, 64, 33, 82 % | 205, 483, 134, 286, 158, 54 % |

Re-normalizing (B) needs no training: the lift error drops from 19.9 to 3.2 deg and FLUX anticipates the
demonstrated direction on every joint (60 to 83 %, chance is about 50 %). Magnitudes stay off (46 to 102 %
displacement error) and on the robot the wrist flex still runs to its soft limit, so we fine-tune from base B.
Lesson for anyone deploying a released VLA on their own arm: check the normalization contract offline first.

## Results

### Graded attempts (paste `python ws4/eval/trial_log.py summary --markdown` here on Sunday)

As of Sat 17 Oct 21:15 ([trials.jsonl](ws4/eval/trials.jsonl)):

| task | condition | variant | success | steps done (mean) | human help | top failures | checkpoint |
|---|---|---|---|---|---|---|---|
| grasp_cup | zero_shot | A0-E | 0/3 (0%) | 0.0/2 | 3/3 | never approached the cup x2; folded to the lift/elbow end stops and swung outward x1 | black-forest-labs/flux-3-action-so101@0713a7a |
| grasp_cup | zero_shot | A0r-B | 0/3 (0%) | 0.0/2 | 1/3 | wrist flex ran into its soft limit and the gripper curled back toward the robot body x2; reached the cup but did not close on it, then folded down and in toward the robot body x1 | black-forest-labs/flux-3-action-so101@0713a7a + A0r-B quantiles (20 pen demos) |
| grasp_cup | zero_shot | A0r-D | 0/2 (0%) | 0.0/2 | 1/2 | stopped before: grasp the cup x2 | black-forest-labs/flux-3-action-so101@0713a7a |
| put_pen_in_slot | zero_shot | A0-E | 0/3 (0%) | 0.0/3 | 2/3 | approached the pen but never closed the gripper; operator stop x2; closed beside the pen, then aligned over it but timed out before closing x1 | black-forest-labs/flux-3-action-so101@0713a7a |
| stack_cups | zero_shot | A0 | 0/3 (0%) | 0.0/3 | 3/3 | moved away from the cups; operator stop to avoid collision x3 | black-forest-labs/flux-3-action-so101@0713a7a |
| stack_cups | zero_shot | A0-5V | 0/2 (0%) | 0.0/3 | 0/2 | stopped before: grasp a cup x2 | black-forest-labs/flux-3-action-so101@0713a7a |

Variants: A0 = released package unchanged; A0r = same weights with quantiles from our demos (B as recorded, D with
lift and roll negated); A0-E = package statistics with a manual joint mapping; A0-5V = our first arm's 12 V servos on
a 5 V supply (not a FLUX result). `stack_cups` ran on a red SO-101 that was retired after its elbow servo was damaged
(see Safety). Post-training rows: `[[NUM:LORA_TABLE]]`.

### Post-training setup

- Data: 96 VR-teleoperated demonstrations of "put the red pen in its slot in the cardboard box"
  (104 recorded, 8 rejected on review; 46,372 frames, 0.43 h), 4 layouts, 30 fps, head + wrist camera at 640x480,
  LeRobot degrees, recorded with a Meta Quest Pro through the haller_ws HMI
  ([vr_teleop.md](ws4/research/vr_teleop.md)). Split: 84 train, 12 held out (the last 3 good takes of each layout),
  never trained on. Public on Hugging Face:
  [Oskrt/haller_put_the_red_pen_in_its_slot_in_the_cardboard_box](https://huggingface.co/datasets/Oskrt/haller_put_the_red_pen_in_its_slot_in_the_cardboard_box)
  and the second skill,
  [Oskrt/haller_take_the_red_pen_out_of_its_slot](https://huggingface.co/datasets/Oskrt/haller_take_the_red_pen_out_of_its_slot)
  (11 takes).
- Time split: `[[NUM:RECORD_HOURS]]` recording, `[[NUM:TRAIN_HOURS]]` training. Own data only: the released corpus
  has no XLeRobot data and 11 of its 1,021 parts use a head or overhead camera
  ([datasets.md](ws4/research/datasets.md)).
- LoRA: LeRobot `examples/flux3/lora.json` (rank/alpha 32, effective batch 8, lr 1e-4, EMA 0.999) from the
  re-normalized base B (quantiles of the 84 train takes), 20,000 microsteps (5,000 updates, about one epoch) planned
  on one RTX PRO 6000 96 GB, about 1.0 s per microstep; `[[NUM:LORA_STEPS_DONE]]` done, checkpoint
  `[[NUM:LORA_CKPT]]`, held-out eval loss `[[NUM:LORA_EVAL_LOSS]]`
  ([pen_lora.md](ws4/research/pen_lora.md), [real_lora.sh](ws4/cloud/real_lora.sh), [RUNPOD.md](ws4/cloud/RUNPOD.md)).
- Evaluation protocol: same layouts, prompts, start pose (`[0, 0, 0, 0, 0, 100]`), time budget and safety stack for
  zero-shot and post-training; every attempt logged with video and per-tick trace.

### Held-out action error, before the robot (offline)

The same offline diagnostic, run on the 12 held-out takes (200 frames, 181 moving; EMA adapter folded in;
[pen_lora.md](ws4/research/pen_lora.md)):

| Model | Microsteps | MAE pan, lift, elbow, wrist flex, roll (deg); gripper (%) | direction agreement | median rel. displacement error |
|---|---|---|---|---|
| hold (repeat the current command) | - | 3.8, 7.6, 5.9, 7.6, 3.2; 8.5 | - | 100 % |
| zero-shot B84 (the LoRA's base) | 0 | 4.6, 6.3, 4.9, 8.0, 3.7; 11.3 | 65, 71, 68, 54, 75, 66 % | 152, 88, 81, 171, 70, 93 % |
| zero-shot median statistics (M) | 0 | 2.7, 6.7, 5.4, 5.3, 3.8; 6.8 (gripper range 0..40, not like for like) | 74, 64, 72, 77, 42, 87 % | not reported (path ratios in the text below) |
| LoRA from B84 | 2,500 | 3.0, 5.1, 4.7, 6.0, 2.9; 9.9 | 72, 77, 78, 64, 75, 78 % | 71, 68, 71, 96, 63, 90 % |
| LoRA from B84 | 5,000 to 20,000 | `[[NUM:LORA_CURVE]]` | | |
| ACT baseline (same demos, 10k steps) | - | arm mean 4.3 | 81 to 90 % | - |

The released statistics fail; two ways of fixing them, with no training, give sensible predictions. B84 (our own
quantiles) is about as good as holding still in magnitude but better than chance on direction. Median statistics of
current-calibration SO-101 community datasets (from
[multimodalart's Space](https://huggingface.co/spaces/multimodalart/flux-3-action-so101-sim), commit `09edebb`) give
the lowest arm error (4.8 vs 5.5 deg mean) and the most balanced motion (every joint at 0.57 to 0.89 of the demo
path, where B84 over-swings pan at 1.49x and barely lifts at 0.32x), but no better direction overall and wrist roll
below chance (42 %). The LoRA trains from B84. After 2,500 of 20,000 microsteps it beats both on arm error (4.3 deg mean)
and direction (74 %), though its gripper error (9.9 %) is still worse than holding still (8.5 %). An ACT policy
trained on the same demos is ahead on direction (81 to 90 %) at the same arm error; the final comparison uses the 20k
checkpoint and the real-arm attempts.

### Imagination: what FLUX expects to happen

FLUX 3 Action denoises future video latents together with every action chunk; LeRobot and flux-action both discard
them. Our wrapper ([predict_with_video.py](ws4/imagine/predict_with_video.py)) decodes them: 41 frames per chunk,
scene and wrist, 0.07 to 1.4 s ahead. Decoding FLUX's imagined video leaves its actions unchanged (bit-identical, with
`compile_model` on and off, measured on the RTX PRO 6000), adds no round trip in lazy mode and takes 0.11 s per
chunk after the take. On held-out takes the imagined scene frame is closer to reality than "nothing moves" on 80 % of
offsets (43 % on the fast-moving wrist view; [imagine/README.md](ws4/imagine/README.md)). We did not find any earlier public demo of decoded FLUX imagination.
Report: [imagination.md](ws4/research/imagination.md). Clip: held-out take 101 replayed through FLUX, actual left and
imagined right (`ws4/imagine/outputs/replay_ep101/imagination/imagined_vs_actual.mp4`, in the demo video and slides).

### Claude orchestrator

Claude Opus 5.5 turns a goal into a validated plan over a fixed skill list (schema, vocabulary, dependencies,
deadlocks; errors fed back at most twice); Claude Haiku 5.5 checks each step from the head camera: on 282 recorded Haller frames, 97.1 % for "put" and 98.2 %
for "take" with no false pass (median 1.3 s, about USD 0.0002 per check; live accuracy not yet measured). An operator HMI shows the plan, live
status, camera thumbnails, Claude spend and an e-stop. No LLM output reaches a motor. Report:
[orchestrator.md](ws4/research/orchestrator.md).

### Research results (simulation and offline)

- **Memory over FLUX features (sim only)**: gated test-time-training layers after joint blocks 24 to 27 of the 7B
  model (17.9 M params) are bit-identical to stock at gate 0 and cost +32 ms (+4.2 %) and +0.08 GiB. A memory over
  FLUX's own features tracks the task stage at 0.90 on held-out sim episodes, against 0.54 without memory (GRU 0.87,
  TTT 0.90: the memory matters, not TTT itself). Trained on demos alone, these monitors invented progress on failed
  FLUX rollouts until 12 such rollouts joined training ([robottt_flux.md](ws4/research/robottt_flux.md) sections
  11.9 and 11.12). The live chain uses the Claude checks instead.
- **Simulation of the station** (MuJoCo, XLeRobot 0.4.0 CAD, our fin-ray fingers, D455 head): the same FLUX contract
  in sim. Released checkpoint 0/20 on "put the block in the cup"; re-normalized `[[NUM:SIM_A0R]]`; LoRA on 60
  scripted demos `[[NUM:SIM_LORA]]` ([robottt_flux.md](ws4/research/robottt_flux.md) section 11.12,
  [sim/README.md](ws4/sim/README.md)).
- **EE50 feasibility for the bimanual heads**: FK/IK round-trips 25,168 frames of real robot motion at p99 1e-7 mm;
  98 % of human handheld chunks are position-feasible on the 5-DoF arm ([ee50_ik.md](ws4/research/ee50_ik.md)).

## Safety

On Saturday afternoon, during zero-shot attempts, our first arm's elbow pushed against an obstruction (92 % load with
the overload bit set) while the wrist folded toward the robot's base, and the elbow servo was damaged. Since then
every command passes, in order:

1. soft joint limits 5 deg inside the calibrated stops and a 1 deg per tick (30 deg/s) step limit;
2. LeRobot's relative-target cap;
3. a forward-kinematics keep-out: table, torso, mast, midline, the parked arm as link capsules (4 cm clearance), and
   the demonstrated joint and gripper-tip envelope;
4. a stall freeze (8 deg behind for 0.5 s) and a load freeze (70 % of the cap for 9 ticks, or 90 % once);
5. a 60 % servo torque cap;
6. operator e-stop (HMI button, spacebar, STOP file); if the parked arm's pose cannot be read, FLUX does not start.

Replayed, the two damaging attempts are refused within 0.97 s and 1.27 s; all 24 good pen demos (15,833 steps) pass
with zero refusals; on the Haller the load never exceeded 29 % during FLUX runs
([single_arm/README.md](ws4/single_arm/README.md)).

## Architecture

```
Quest Pro VR teleop ──> LeRobot dataset (FLUX contract: 30 fps, head + wrist, degrees)
                              │
                              ├──> A0r quantiles (offline diagnostic picks base B)
                              v
                     LoRA on RTX PRO 6000 ──> merged package
                                                   │
Operator HMI ──goal──> Claude Opus planner         v
   │  ^                    │ validated plan   FLUX 3 Action server (GPU pod, SSH tunnel)
   │  └── Haiku checks <───┤                       │ 42-step action chunk + imagined video
   │                       v                       v
   └──── imagination panel <── runner: guards, keep-out, stall/load freeze ──> Haller arm (STS3215)
```

## Repository map

| Path | What | Tests |
|---|---|---|
| [ws4/single_arm/](ws4/single_arm/README.md) | Real-arm runner, calibration, FK keep-out, offline diagnostic, A0r statistics | 20 (CPU) |
| [ws4/flux/](ws4/flux/) | FLUX serving (`flux.remote`), per-arm runner, history, safety guard, package tools | 24 (LeRobot-main env) |
| [ws4/orchestrator/](ws4/orchestrator/README.md) | Claude planner and verifier, DAG executor, HMI, Haller wiring | 58 (CPU) |
| [ws4/imagine/](ws4/imagine/) | Decoded imagination, imagined-vs-actual rendering, surprise and goal scorers | 157 + 5 against real LeRobot code |
| [ws4/ttt/](ws4/ttt/) | TTT and GRU progress memory, FLUX feature tap, sim demos and closed-loop eval | 30 |
| [ws4/sim/](ws4/sim/README.md) | MuJoCo model of the Haller station | 26 (headless EGL) |
| [ws4/kinematics/](ws4/kinematics/) | FK/IK, collision capsules, EE50 feasibility | 31 |
| [ws4/data/](ws4/data/), [ws4/teleop/](ws4/teleop/) | Dataset conversion and merging, VR teleop launcher and checks | 34, 32 |
| [ws4/eval/trial_log.py](ws4/eval/trial_log.py) | One JSONL line per graded attempt, markdown summary | 5 |
| [ws4/cloud/](ws4/cloud/RUNPOD.md) | GPU bootstrap, serving, LoRA scripts, idle watchdogs | |
| `[[TEAM:WORKSTREAMS]]` | WS1, WS2, WS3, WS5 parts (`src/liminal/`, `configs/`, `scripts/`) | |

Test counts are the last counts recorded in each report; re-run before submitting.

## Reproduce

Laptop (CPU, no robot): the unit tests and every fake-backend path.

```bash
git clone https://github.com/Javgilavi/liminal && cd liminal/ws4
python -m pytest single_arm/tests orchestrator/tests imagine/tests ttt/tests eval/tests -q   # UNVERIFIED as one line
python -m orchestrator hmi --open --env haller --robot fake --policy fake --planner claude --verifier manual
```

GPU (one RTX PRO 6000 or a 32 GB card with the text encoder on the CPU), with a Hugging Face token that has accepted
the gated `flux-3-action-so101` and `flux-3-action-base` repos:

```bash
bash ws4/cloud/bootstrap.sh                                   # envs + weights, about 11 min on a fresh volume
python ws4/single_arm/a0r_stats.py ...                        # A0r quantiles from your demos
bash ws4/cloud/real_lora.sh package STATS.json                # re-normalized base package
STEPS=20000 bash ws4/cloud/real_lora.sh train                 # LoRA
CKPT=.../pretrained_model_ema bash ws4/cloud/real_lora.sh merge
bash ws4/cloud/runpod_serve.sh                                # flux.remote policy server
```

Robot: calibrate with `lerobot-calibrate`, check the bus with `ws4/single_arm/probe_bus.py`, dry-run in observe mode,
then run `python -m orchestrator hmi --open --env haller --planner claude --package <package> ...` (full commands:
[orchestrator/README.md](ws4/orchestrator/README.md), "Sunday demo"; bring-up recipe:
[single_arm/README.md](ws4/single_arm/README.md), "Recipe for the teammate").

## What is not verified

- Guidance 1.0 (used for latency) vs the package default 3.0 on motion quality.
- Why the wrist flex runs to its limit zero-shot even with our quantiles (convention or skill).
- Claude's step checks on live camera frames (measured offline on recorded frames only).
- The table height in the keep-out (`keepout.py --touch` not run).
- Both arms moving at once: one shared model makes them take turns (measured); two instances are untested.

## Credits

FLUX 3 Action by Black Forest Labs (weights under their licence, see the model cards). LeRobot by Hugging Face.
XLeRobot by Vector-Wangel. Claude by Anthropic. The Haller VR HMI (haller_ws) by Oscar Devos. `[[TEAM:CREDITS]]`
