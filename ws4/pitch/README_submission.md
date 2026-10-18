<!--
Submission README draft, re-centred on the 4-skill chain, Sun 18 Oct about 08:00 CEST. Written for the ROOT of
Javgilavi/liminal on main (links are repo-root relative). Before 14:00: fill every [[...]] (grep -rn "\[\[" ws4/pitch;
owners and times in ws4/pitch/NUMBERS.md section 14), paste the fresh `python ws4/eval/trial_log.py summary --markdown`
under "Results", then copy this file to README.md at the repo root. Numbers and their sources:
ws4/pitch/NUMBERS.md (section 0 first). Delete this comment.
-->

# liminal: a robot that imagines, then acts

**ERH Physical AI Hackathon 2026, track "Imagine, Then Act" (Black Forest Labs FLUX 3 Action).**
Team **Liminal**: Oscar Devos, Javier Gil, Tomas Leggaet, Simone Utili, Stanislaw Sufin.

Repository: [github.com/Javgilavi/liminal](https://github.com/Javgilavi/liminal) · Project page:
[javgilavi.github.io/liminal](https://javgilavi.github.io/liminal/) · Demo video:
[liminal_2min_v4.mp4](https://drive.google.com/file/d/1et5ffQpXLLh4a1qOvociNB1weNlaCGPU/view) · Slides: [hosted deck](https://claude.ai/artifact/FsptG2MfRYqkmaAqL4EDQm) (PDF: `liminal_slides.pdf` next to the video) ·
Adapters and data: [huggingface.co/Oskrt](https://huggingface.co/Oskrt) ·
How to train your own: [ws4/LORA_HOWTO.md](ws4/LORA_HOWTO.md)

Our green XLeRobot "Haller" (two SO-101-style arms on a torso, a RealSense D455 head camera, a wrist camera per arm,
soft fin-ray fingers) runs a **4-skill chain with FLUX 3 Action in the control loop**. Claude Opus plans it and
Claude Haiku checks each step from the head camera:

1. robot right arm: "open the box by pulling its handle", then keep holding the lid. **This step is a scripted replay
   of a recorded demo (take 72 in the final runs), not FLUX** (our FLUX box LoRA would not start from rest; see Results), run through the same
   safety stack and labelled as such in the HMI and the logs;
2. robot left arm, FLUX: "put the green pen in the green slot";
3. FLUX: "put the yellow cutter in the yellow slot";
4. FLUX: "put the red pen in its slot in the cardboard box".

The planner mixes learned skills and scripted primitives under one safety layer. One FLUX base, one LoRA adapter per
skill (about 0.45 GB each, all public), a safety stack per skill, and an imagination panel that shows the future video
FLUX predicts next to what actually happens.

## Results

| | Zero-shot (released checkpoint) | Post-training (our LoRAs) |
|---|---|---|
| Red pen on the arm | 0/3 (and 0/16 over all Saturday attempts) | **3/5 full inserts** (RAW 5k, action horizon 16); 5/8 including the next three attempts |
| Red pen, 12 held-out takes, offline arm error / direction | 5.5 deg / 66.5 % (our quantiles), hold still 5.6 deg | **3.24 deg / 87.5 %** (RAW 5k); RAW 10k 3.15 deg / 88.6 % |
| Green pen / yellow cutter on the arm | - | 1/2 / 1 insert (trial log 1/6) |
| Open the box | - | FLUX 0 (0/3 runner, 0/1 HMI): replaced by a scripted replay |
| **Full 4-skill chain on the arm** | - | **2 of 3 runs completed 4/4 with no human touch** (208.6 s and 302.2 s) |
| Sim, "put the block in the cup", 20 held-out layouts | 0/20 | 0/20 (grasped and lifted in 2) |

### On the arm (red pen, robot left arm, 120 s per attempt)

From [red_pen_2026-10-18.md](ws4/single_arm/benchmarks/red_pen_2026-10-18.md) (scored by the operator after each
attempt, one `trials.jsonl` line each):

| Block | Success | Steps (reach / grasp / insert) | How attempts ended |
|---|---|---|---|
| **FLUX RAW 5k, horizon 16, lift stall 25 deg** | **3/5** (94 s, 91 s, 48 s) | 9/15 | two stopped without a reach |
| FLUX RAW 5k, horizon 32 | 0/5 | 8/15 | stall freeze 4, keep-out 1 |
| FLUX EMA 10k, horizon 32 | 0/5 | 4/15 | stall freeze 4 |
| FLUX RAW 10k, horizon 32 | 0/5 | 3/15 | keep-out 3 |
| ACT (same data, 100k steps) | 0/5 | 8/15 (grasp and carry in 4) | timeout 5, never inserted |
| FLUX RAW 10k, lid held up (a scene it never saw) | 0/5 | 0/15 | held still |

Caveats: small n; horizon 16 came together with a 25 deg shoulder-lift stall threshold (two changes at once); the
successes ended with the operator's stop key after the insert; lighting fell during the session. At horizon 32 the
stall freeze ended 9 of 15 attempts: the shoulder lift trails its command at about two thirds of the 60 % torque cap.
At horizon 16 FLUX replans every 16 commands (about 0.5 s) instead of 32.

This morning, per skill (trial log, about 09:20; paste the final `trial_log.py summary --markdown` here):

| Skill | Model | Result |
|---|---|---|
| red pen | FLUX RAW 5k, horizon 16 | 2/3 graded; in the HMI, one insert in 33 s confirmed by Claude Haiku; 9/23 over all RAW 5k attempts |
| green pen | FLUX RAW 5k | 1/2 (one full insert) |
| yellow cutter | FLUX augmented RAW 5k | 1 full insert (1/6 in the trial log; the orchestrator reports it reached 5/5 and grasped 4/5 in the first five) |
| open the box | FLUX RAW 5k, every variant | 0 (0/3 in the runner, 0/1 in the HMI): its first chunk from rest moves 4.9 to 5.5 deg on held-out takes, demos move 39 to 150, so the demo replays recorded take 59 |
| red pen with the lid held | FLUX RAW 10k | 0/5 (held still: a scene the LoRA never saw); not rerun with RAW 5k |
| full chain (unguarded mode) | box replay + FLUX skills | 2 of 3 runs 4/4 (see below) |

### The full chain

Three runs on the real Haller, Claude Opus planning and Claude Haiku checking every step, in an unguarded demo mode
(see Safety):

| Run | Result | Time | Replans | Claude cost |
|---|---|---|---|---|
| 11:46 | 1/3 steps, operator abort | 186 s | 1 | USD 0.05 |
| 11:52 | **4/4, no human touch** | 209 s | 0 | USD 0.02 |
| 12:05 | **4/4, no human touch** | 302 s | 1 (the box replay missed twice; Claude replanned it) | USD 0.05 |

In both completed runs all four objects ended in place without human touch: the box by scripted replay (recorded take
72), the green pen and the cutter by their FLUX LoRAs (green 58 s and 55 s; cutter 112 s, and 46 s on a second
attempt), and the red pen **placed by the green-pen LoRA** during its return-home motion; the red-pen step's check
then confirmed it in 3 to 4 s. The red-pen LoRA's own results are the single-skill runs above.

### Offline, on held-out takes (same protocol for every row)

200 frames per row, moving frames only, FLUX's next 32 commands vs the recorded ones, guidance 1/1
([pen_lora.md](ws4/research/pen_lora.md), [skills.md](ws4/research/skills.md)):

| Skill, model | Arm MAE (deg, mean of 5 joints) | Direction agreement |
|---|---|---|
| red pen, hold still | 5.6 | - |
| red pen, zero-shot, our quantiles (B84) / community median statistics | 5.5 / 4.8 | 66.5 % / 69 % |
| red pen, ACT 100k | 4.3 | 85 % (arm) |
| **red pen, FLUX LoRA RAW 5k** | **3.24** | **87.5 %** (arm) |
| red pen, FLUX LoRA RAW 10k (public) | 3.15 | 88.6 % |
| box, FLUX LoRA RAW 5k / ACT 20k | 5.11 / 5.6 | 73 % / 80 % (arm) |
| green pen, FLUX LoRA RAW 5k | 3.95 | 89.1 % |
| yellow cutter, FLUX LoRA augmented RAW 5k | 5.26 | not reported |

RAW adapters beat EMA every time (pen 5k: 3.24 vs 3.62 deg); classifier-free guidance 3/3 or 4/1 gave nothing over
1/1 at twice the compute per chunk.

## What we found

### 1. Zero-shot failed on a calibration contract, not on skill

The released `flux-3-action-so101` package normalizes joints with quantiles from one community dataset recorded in
the pre-2025 SO-100 degree convention (shoulder lift 85 to 184 deg; [checkpoint_facts.md](ws4/flux/checkpoint_facts.md)).
With today's LeRobot calibration our folded arm reads lift -100 deg: the model sees an impossible robot and drives the
shoulder to its limit whatever the prompt. Five normalizations, measured offline on our own demos before touching the
robot again ([offline_diag.py](ws4/single_arm/offline_diag.py)):

| variant | MAE pan, lift, elbow, wrist flex, roll (deg); gripper (%) | direction agreement |
|---|---|---|
| hold (repeat current command) | 1.9, 4.4, 4.7, 2.9, 2.0; 4.4 | - |
| A package statistics | 4.3, **19.9**, 10.1, 11.2, 3.7; 6.6 | 42, 57, 53, 58, 34, 78 % |
| B our quantiles (A0r) | 1.9, 3.2, 3.8, 3.7, 1.8; 8.4 | 80, 79, 68, 60, 73, 72 % |
| C [flux-action issue #2](https://github.com/black-forest-labs/flux-action/issues/2) adapter | 4.4, **16.8**, 6.0, 10.1, 1.9; 6.5 | 41, 43, 70, 64, 77, 82 % |
| D sign flips + our quantiles | 1.9, 3.2, 3.9, 3.5, 1.6; 7.6 | 80, 83, 70, 62, 81, 80 % |
| E manual offset mapping + package statistics | 4.3, 16.7, 5.9, 9.6, 3.2; 6.4 | 41, 43, 69, 64, 33, 82 % |

Every LoRA trains from base B: the released weights with quantiles from the skill's own training takes.

### 2. FLUX's decision hangs on the wrist camera's view of the room

After the robot was moved, the red-pen LoRA held still at home on live frames, while the same model reached on the
recorded ones. A field-by-field diff of a live request against a recorded one found them identical in task string,
image keys and order, shapes, dtypes, units and history: only the wrist image differed. Recorded images with live
state gave a reach; live images with recorded state gave a hold. The live wrist frame flips hold and reach on a
**1 px shift or a 1 deg tilt**; recorded frames reach under the same shifts; no seed or averaging rescued the live
view. Fixes: [align_station.py](ws4/single_arm/align_station.py) aligns the station to a recording's home view
(wrist offset, scale, brightness) before every run, and the newest LoRAs train with crop and colour augmentation,
checked before robot time by a gate: from rest, the first chunk must move at least 20 deg on every held-out take, also
under 2 px shifts and half brightness. The cutter passes (44.2 deg) and green (73.3); every box variant fails (4.9 to
5.5), which is why the box step is replayed ([skills.md](ws4/research/skills.md)).

### 3. FLUX imagines, and fine-tuning makes it imagine better

FLUX 3 Action denoises future video latents with every action chunk; LeRobot and flux-action discard them. Our
wrapper ([predict_with_video.py](ws4/imagine/predict_with_video.py)) decodes 41 frames per chunk, 0.07 to 1.4 s
ahead. Decoding leaves the actions unchanged (bit-identical, with `compile_model` on and off, also on the live arm
server) and adds no round trip in lazy mode (0.12 s to decode after the take). The red-pen LoRA cut the wrist-camera
imagination error by 22 % at 0.07 to 0.27 s ahead and 17 % at 1.1 to 1.4 s: zero-shot imagines a smeared, broken pen
by +1 s, the LoRA imagines it held, as it was ([imagine/README.md](ws4/imagine/README.md)).

## Claude orchestrator

Claude Opus 5.5 turns the goal into a validated plan over a fixed skill list (right arm first, then the three
left-arm steps; 6.4 s and USD 0.02 when checked live). Claude Haiku 5.5 checks each step from the head camera; on
held-out frames of the final scene: lid open 63/63, green pen 114/119 (2 false passes), cutter 91/91, red pen 65/66
(1 false pass), about 1.2 to 1.3 s per check. An operator HMI shows the plan lanes, live status, camera thumbnails,
the imagination panel, Claude spend and an e-stop. No LLM output reaches a motor
([orchestrator.md](ws4/research/orchestrator.md), [orchestrator/README.md](ws4/orchestrator/README.md)).

## Safety

**The final chain runs used an unguarded demo mode** (commit 18ece73, approved by the operator for the demo): the
policies ran without keep-out zones and without the stall, keep-out and load stops, start gate or lid precheck; joint
limits, 1 deg per tick, the 60 % torque cap and the e-stop stayed on. The reason: 17 of 20 keep-out stops in the
morning came from the demonstrated-envelope zones, not from collision zones, so the guards were stopping good motion.

In the single-skill runs every command passes soft joint limits, a 1 deg per tick step limit, LeRobot's relative cap,
a forward-kinematics keep-out, a stall and load freeze and a 60 % servo torque cap, with an e-stop on the HMI and the
spacebar. Per skill:

- each skill's workspace is built from its own demos (station zones plus the demonstrated envelope), with recorded
  overrides (box midline +0.17 m, green -0.23 m, cutter -0.21 m, red pen with the lid held -0.15 m);
- the cutter has a 4 cm slow zone next to the lid-holding arm (0.5 deg per tick, load freeze at 50 %);
- before every pen-arm step a start gate checks the box arm's live pose against the skill's recorded hold and the
  geometry; a refusal ends the chain with both arms holding (the lid stays held);
- an arm already holding the lid can be taken over without moving it.

On the arm this morning the keep-out refused ticks in 15 of 33 benchmark attempts and ended 7 of them; nothing
collided or broke. It exists because on Saturday, under zero-shot FLUX, our first arm's elbow pushed against an
obstruction at 92 % load until its servo broke ([single_arm/README.md](ws4/single_arm/README.md)).

## Data and adapters (all public)

All recorded by VR teleoperation (Meta Quest Pro through the haller_ws HMI), LeRobot v3, 30 fps, head + wrist camera:

| Skill | Dataset | Takes |
|---|---|---|
| open the box | [Oskrt/haller_open_the_box](https://huggingface.co/datasets/Oskrt/haller_open_the_box) | 75 |
| green pen | [Oskrt/haller_put_the_green_pen_in_the_green_slot](https://huggingface.co/datasets/Oskrt/haller_put_the_green_pen_in_the_green_slot) | 66 |
| yellow cutter | [Oskrt/haller_put_the_yellow_cutter_in_the_yellow_slot](https://huggingface.co/datasets/Oskrt/haller_put_the_yellow_cutter_in_the_yellow_slot) | 70 |
| red pen | [Oskrt/haller_put_the_red_pen_in_its_slot_in_the_cardboard_box](https://huggingface.co/datasets/Oskrt/haller_put_the_red_pen_in_its_slot_in_the_cardboard_box) | 126 (104 + 22 with the lid held) |
| take the red pen out | [Oskrt/haller_take_the_red_pen_out_of_its_slot](https://huggingface.co/datasets/Oskrt/haller_take_the_red_pen_out_of_its_slot) | 11 |

Held out: the last 3 good takes of each layout, never trained on. LoRA recipe: LeRobot `examples/flux3/lora.json`
(rank/alpha 32, lr 1e-4, effective batch 8), 5,000 microsteps, about two hours per skill on one RTX PRO 6000.
Adapters, all public (root = the served adapter, `checkpoints/<run>/<step>/{raw,ema}` for every checkpoint):
[pen](https://huggingface.co/Oskrt/flux-3-action-so101-haller-pen-lora),
[open box](https://huggingface.co/Oskrt/flux-3-action-so101-haller-open-box-lora),
[green pen](https://huggingface.co/Oskrt/flux-3-action-so101-haller-green-pen-lora),
[yellow cutter](https://huggingface.co/Oskrt/flux-3-action-so101-haller-yellow-cutter-lora).
[ws4/LORA_HOWTO.md](ws4/LORA_HOWTO.md) is the short version for anyone with an SO-101.

## Research (simulation and offline)

- **Simulation of the station** (MuJoCo, XLeRobot CAD, our fin-ray fingers, D455 head): 0/20 for every policy on
  "put the block in the cup", released, re-normalized and two LoRAs; the raw LoRA grasped and lifted in 2
  ([robottt_flux.md](ws4/research/robottt_flux.md) 11.12).
- **Memory over FLUX features (sim only)**: gated test-time-training layers after joint blocks 24 to 27 of the 7B
  model are bit-identical at gate 0 and cost +4.2 %; a memory over FLUX features tracks the task stage at 0.90 vs
  0.54 without memory (GRU 0.87: the memory matters, not TTT itself).
- **EE50 feasibility**: FK/IK round-trips 25,168 frames of real robot motion at p99 1e-7 mm
  ([ee50_ik.md](ws4/research/ee50_ik.md)).

## Architecture

```
Quest Pro VR teleop ──> LeRobot dataset per skill (FLUX contract: 30 fps, head + wrist, degrees) ──> Hub
                              │
                              ├──> base B: our quantiles (offline diagnostic)
                              v
                     one LoRA per skill (RTX PRO 6000, about 2 h) ──> packages on one server
                                                   │
Operator HMI ──goal──> Claude Opus planner         v
   │  ^                    │ validated plan   FLUX 3 Action server (GPU pod, SSH tunnel)
   │  └── Haiku checks <───┤                       │ action chunk + imagined video
   │                       v                       v
   └──── imagination panel <── runner per arm: start gate, keep-out, slow zone, freezes ──> Haller arms
```

## Repository map

| Path | What |
|---|---|
| [ws4/single_arm/](ws4/single_arm/README.md) | Real-arm runner, per-skill kits (`skills/`), keep-out, start gate, station alignment, offline diagnostic, benchmarks |
| [ws4/orchestrator/](ws4/orchestrator/README.md) | Claude planner and verifier, DAG executor, HMI, chain wiring |
| [ws4/imagine/](ws4/imagine/README.md) | Decoded imagination, imagined-vs-actual rendering, scorers |
| [ws4/flux/](ws4/flux/) | FLUX serving (`flux.remote`), runner, history, safety guard, package tools |
| [ws4/cloud/](ws4/cloud/RUNPOD.md) | GPU bootstrap, per-skill LoRA (`skill_lora.sh`), adapter serving, watchdogs |
| [ws4/act/](ws4/act/README.md) | ACT baseline on the same data |
| [ws4/eval/trial_log.py](ws4/eval/trial_log.py) | One JSONL line per graded attempt, markdown summary |
| [ws4/ttt/](ws4/ttt/), [ws4/sim/](ws4/sim/README.md), [ws4/kinematics/](ws4/kinematics/) | Research: memory, simulation, kinematics |
| [ws4/data/](ws4/data/), [ws4/teleop/](ws4/teleop/) | Dataset conversion, VR teleop launcher |
| `src/liminal/`, `configs/`, `scripts/` | The behaviour-tree harness (Stanislaw Sufin, Simone Utili): task YAML to tree, VLM perception, per-skill checkpoint registry, coherent skill runtime; it drove the real Haller through this stack (open the box by replay, the green pen placed live) |
| `site/`, `docs/research-ui.md` | The project page, the full-screen live lab and the Haller HMI companion with the MuJoCo digital twin (Javier Gil) |

## Reproduce

Train your own skill: [ws4/LORA_HOWTO.md](ws4/LORA_HOWTO.md) (record, normalize, train with `ws4/cloud/skill_lora.sh`,
serve with `ws4/cloud/lora_adapter.py`). Laptop rehearsal without hardware:

```bash
git clone https://github.com/Javgilavi/liminal && cd liminal/ws4
python -m orchestrator hmi --open --env haller --robot fake --policy fake --planner claude --verifier manual
```

On the robot: [orchestrator/README.md](ws4/orchestrator/README.md) (chain) and
[single_arm/README.md](ws4/single_arm/README.md) (bring-up, alignment, per-skill runs).

## What is not verified

- The chain's reliability beyond 3 runs (2 complete), and the chain with the guards on; the real-arm numbers are
  small blocks.
- A FLUX box skill: every box LoRA failed the from-rest gate; the demo replays a recorded take.
- The team's behaviour-tree harness end to end on the arm: it drove the Haller through the ws4 bridge and placed the
  green pen in its best run, then stopped on its own perception and lid-predicate timeouts and on the start gate.
- Claude's checks on live frames (measured on recorded held-out frames).
- Both arms moving at the same time (the chain moves one arm at a time).

## Credits

FLUX 3 Action by Black Forest Labs (weights under their licence, see the model cards). LeRobot by Hugging Face.
XLeRobot by Vector-Wangel. Claude by Anthropic. Median SO-101 statistics from multimodalart's
`flux-3-action-so101-sim` Space. The Haller VR HMI (haller_ws) by Oscar Devos. The project page, live lab, HMI companion with the MuJoCo twin and media curation by Javier Gil; the behaviour-tree harness by Stanislaw Sufin and Simone Utili; the FLUX LoRA pipeline, data collection, safety runner and Claude chain by Oscar Devos. All on main via PR #3 and #4.
