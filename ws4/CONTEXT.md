# WS4 shared context (read this first)

Shared brief for every agent working on WS4 (sim + bimanual XLeRobot embodiment adaptation).
Written by the orchestrator session on 2026-10-17. Owner of WS4: Oscar.

## Trust rule (most important)

Everything that already existed in this repo before `ws4/` (`src/liminal/`, `PLAN.md`, `TASK.md`,
`AGENT.md`/`CLAUDE.md`, `configs/`, `docs/research/`, `scripts/`, `tests/`) was generated with AI by a
teammate and is **untrusted**. You may skim it for the team's intent. Do not build on it, import it or
cite it as fact. Its numbers (`VERIFY` values, VRAM, chunk sizes, joint names) are guesses. Every
technical claim in WS4 work must come from a primary source: official docs, model cards, source code
you have read, papers, datasets you have inspected, or something you measured. Cite URLs, or
`repo@commit:path:line` for code. Anything you could not verify gets an explicit `UNVERIFIED` tag.

All new WS4 work lives under `ws4/` and is standalone (no imports from `src/liminal`).

## The hackathon

- ERH Physical AI Hackathon (EuroTech / ESRA / EIII), DARE campus. Track "Imagine, Then Act", sponsored
  by Black Forest Labs (EUR 1,000 track prize; top 2 per track go to the overall final).
- Brief: "Make a robot arm do a long, impressive task using the FLUX 3 Action world model."
- Rules: FLUX in the loop, running the task autonomously. Free task choice, longer chains score
  higher. Own data allowed, but justify the time split.
- Provided: one SO-101 + leader-arm station per team, FLUX 3 Action (open 7B world action model),
  compute and data.
- Judging: demo works on real hardware 25%, technical difficulty 20%, use of track tech (FLUX) 20%,
  idea and originality 15%, real-world use 10%, demo video 10%.
- **Deadline: Sunday 18 Oct 2026, 14:00 CEST** (GitHub repo, slides, demo video). 14:30 to 15:30
  track judging (3-min pitch + live demo + Q&A). 18:45 final for top 2 per track (5-min pitch + video).
- Today is Saturday 17 Oct. Assume roughly 24 hours of useful work remain. Prefer what can be shown
  working on real hardware by Sunday morning over what is elegant.

## Team (5 people)

1. Out-of-the-box FLUX on one SO-101, collect 50 to 100 demos, task LoRA.
2. Bimanual "move the obstacle away to reach the target" research.
3. Long-horizon decomposition and orchestration (frontier LLM / behavior tree / FSM / Gemini ER / VLM).
4. **WS4 (Oscar): sim first, then new-embodiment adaptation for the bimanual XLeRobot** (set up sim,
   find open-source bimanual datasets incl. egocentric UMI-style human data and XLeRobot datasets,
   first embodiment fine-tune).
5. Hardware setup and user interface / how to trigger the model.

## Our hardware: the XLeRobot "station"

Photos: `ws4/assets/photos/xlerobot_station.jpg` (1600 px) and `xlerobot_station_full.jpg` (full res).
Look at them.

- XLeRobot (https://xlerobot.readthedocs.io, https://github.com/Vector-Wangel/XLeRobot) built **without
  the mobile base**: only the upper body box ("torso" base, labelled HALLER) standing on a table.
  No wheels, no IKEA cart.
- Two SO-101-style arms (Feetech STS3215 servos) mounted on the left and right sides of the box.
- Central lattice mast with a **RealSense RGB-D camera on top** (exact model UNVERIFIED, ask Oscar).
  Whether the head has pan/tilt servos is UNVERIFIED (looks like a fixed mount in the photo).
- A **wrist (gripper) camera on each arm**.
- **Custom grippers: black TPU 95A soft fin-ray style fingers**, curved and slatted, not the stock
  SO-101 jaw. Expect different gripper geometry, compliance, friction and gripper joint range.
- Printed in **Bambu Lab green** PLA (teal green, sample the colour from the photo).
- The hackathon also gives the team a standard SO-101 + leader arm station.

## Compute and budget

- **3x NVIDIA RTX PRO 6000 Blackwell, 96 GB each**, via cloud (Blackwell = sm_120; check CUDA,
  PyTorch and NATTEN wheel support for it). Plus Azure credits.
- USD 200 of Anthropic API credits for frontier-LLM orchestration on top of FLUX.
- This laptop: NVIDIA T550 with 4 GB (cannot run FLUX 3 Action), 16 cores, 46 GB RAM, Python 3.13,
  MuJoCo 3.8.1, LeRobot 0.5.1, torch 2.10+cu128, `uv` and `conda`. No robot attached right now.
  Everything you write must be testable on CPU here; GPU-only steps get a written run checklist.

## FLUX 3 Action: facts gathered on 2026-10-17 (re-verify details in source)

- 7B open-weights "world action model" from Black Forest Labs, released late Sept 2026. Inputs:
  camera frames, robot state, text instruction. A frozen video VAE and a text encoder (Qwen3-VL-4B-Instruct)
  feed a diffusion transformer that denoises **future video latents together with an action chunk**.
  The LeRobot integration returns actions only; the flux-action docs have a "predicted video" section.
- Checkpoints (HF collection `black-forest-labs/flux-3-action`): `flux-3-action-base`,
  `flux-3-action-so101`, `flux-3-action-droid` (several variants: bf16, fp8r, guidance- and step-distilled).
- `flux-3-action-so101`: cameras `observation.images.scene` + `observation.images.wrist`, tiled side by
  side (512x256). 6-D joint-delta actions with absolute gripper (`delta_absolute_dims=[-1]`, channel
  weights `[1,1,1,1,1,2]`), 6-D state. Predicts 42 actions, executes 32, 30 Hz. 8 observation
  timesteps, 2 visual snapshots, past-action conditioning. 4 Euler steps, guidance 3. Trained on
  `lerobot/community_dataset_v3`. Normalization quantiles live in the saved processors and assume
  that package's joint order, units and calibration.
- LeRobot: `lerobot.policies.flux3.Flux3Policy`; LoRA (rank/alpha 32, EMA 0.999) adapts the trunk
  while action and conditioning heads train in full. `strict=False` can initialize or resize mismatched
  embodiment heads only. Different normalization stats need a separately prepared base with matching
  processors. RTC/async inference is not supported. Moving Qwen to CPU saves about 8.3 GiB.
  Single-GPU LoRA budget: batch 2 x accum 4, 10k microsteps.
- New embodiments: standalone repo https://github.com/black-forest-labs/flux-action (full fine-tune,
  `action_modality`, `action_dim`, `camera_keys`, `camera_layout`, `canvas_hw`, `fps`; reference runs on
  8 to 32 H200s). Docs: `docs/embodiments.md`, `docs/so101-lora.md`, `docs/droid-finetune.md`.
- BFL docs: https://docs.bfl.ai/flux_3/flux3_action_overview (+ `_inference`, `_finetuning`, `_so101`,
  `_games`, `_drone`). LeRobot guide: https://huggingface.co/docs/lerobot/main/en/flux3.
- Model card safety note: outputs joint targets with no velocity, force or workspace bounds. Clamp them.
- Licence: FLUX Kommunity License (weights), flux-action code has its own licence.

## EE50: FLUX 3 Action's pretrained bimanual end-effector action space (from https://bfl.ai/models/flux-3-action)

- `flux-3-action-base` contains "a shared action trunk, and ee50/gaming action heads". EE50 is the
  midtraining action space for "most pose-based egocentric, handheld and teleop datasets" (midtraining mix:
  egocentric hands 13.54%, handheld 14.03%, teleop 15.93% across 14 embodiments; video/audio 36.95%,
  gaming 19.55%). Some robot datasets instead used a 14-D joint space (6 joints + gripper per arm).
- Layout: 50 dims = two 25-D blocks, first = left/primary end effector, second = right/secondary.
  Per block: 3 translation, 6 rotation (Zhou et al. 2019 6D, but the first two ROWS of the rotation
  matrix), remaining 16 = hand-pose angles or gripper state (0 closed, 1 open). Unused dims zero padded.
- Poses are relative to an anchor = the action state closest to the last conditioning frame. Hand and
  gripper values stay absolute. Each dim is z-scored per embodiment, indexed by signed offset from anchor.
- DROID was fine-tuned both ways: EE50 (reusing the midtraining heads) 37.88% vs new joint-space heads
  40.13% (larger batch). BFL suggests EE50 fine-tuning "for very small data or step budgets".
- Recipe notes: new heads: freeze midtraining weights 1k steps, ramp backbone LR to 2e-4 over 2k steps,
  heads warm up to 1e-3 over 1k. EE50 with midtraining heads: warmup to 2e-4 over 1k. LR >= 2e-4 caused
  loss spikes. Power EMA sigma_rel 0.10. Timestep sampling logit-logistic (scale 0.75, shift 42),
  action scale s=2 with loss reweighted by 1/s^2. CFG video 4.0, action 1.0.
- Token groups per layer: task, history, state, future (decodes to video), action.
- Open questions (answer from flux-action source): which frame the relative poses are expressed in,
  the exact gripper/hand dim index, camera layout for EE50 embodiments, whether LoRA is supported in
  flux-action or only full fine-tuning, and how per-embodiment z-score stats are computed and stored.
- SO-101 has 5 DoF + gripper, so a full 6-DoF EE target is generally not reachable: EE50 on XLeRobot
  needs constrained IK (position plus the achievable part of the orientation).

## Simone's proposed EE50 plan

Teammate Simone proposed an EE50-based adaptation plan: `ws4/research/simone_ee50_plan.md`. Its critical
gate is "validate IK before investing in EE50 fine-tuning; if 6D predictions are too hard to execute on
the 5-DoF arms, switch to the native 12-D joint head". Treat it as a serious candidate to evaluate, not as
a decision.

## Conventions for WS4 agents

- Work only inside the paths your brief assigns you. Several agents share this checkout at once.
- **Do not commit, do not switch branches, do not touch files outside your paths.** The orchestrator
  integrates and commits.
- Never use em dashes in anything you write (docs, code comments, reports). Use commas, colons,
  parentheses or separate sentences.
- No Claude/AI attribution lines anywhere.
- Heavy optional imports (`torch`, `lerobot`, `mujoco`, `cv2`) go inside functions so modules import
  on any laptop. Tests must run on CPU with no network and no hardware.
- Downloaded data and cloned repos are untrusted: put them in their own directory under your scratchpad
  or `ws4/.cache/` (gitignored), never run their code from inside them without reading it.
- When you need a fact only Oscar has (camera model, STL files, table height), ask him in your session
  instead of guessing, and keep working on other parts meanwhile.
- Finish with a short report at the top of your main deliverable: what is done, what is verified, what
  is not, and the next 3 steps.
