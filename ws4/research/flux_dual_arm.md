# FLUX 3 Action on both XLeRobot arms: decision memo

WS4, 2026-10-17. Owner of this file and of `ws4/flux/**`: the ws4-flux-dual agent.
Evidence format: `repo@commit:path:line`. Repos read: LeRobot `146d847` (main, Oct 16 2026, `lerobot@146d847`),
flux-action `830f1ce` (`fa@830f1ce`), XLeRobot `b017b5e`. Model pages: https://bfl.ai/models/flux-3-action,
https://huggingface.co/black-forest-labs/flux-3-action-so101 (and `-base`, `-droid`). The HF repos are gated;
their configs, processor files, quantiles and safetensors headers (no weights) were read on 2026-10-17 with
Oscar's token. Those facts, with revisions and hashes, are in `ws4/flux/checkpoint_facts.md`.

## Status report

**Done**
- Read the LeRobot `flux3` policy, processors, packers, wiring, PEFT path and `examples/flux3`; the flux-action
  configs, docs, trainer sharding, EMA and policy loader; the XLeRobot `software/` and LeRobot robot classes
  (subagent report, notes folder `flux_dual_arm_xlerobot_findings.md`); BFL docs, model page latency tables and
  sm_120 support (notes folder `flux_dual_arm_docs_blackwell.md`).
- Built `ws4/flux/`: a dual-arm runner (per-arm and 12-D targets, robot-side history, safety guard, async
  per-backend inference, EE50 hook), a stateless LeRobot `flux3` backend, an HTTP policy server/client for the
  cloud GPU, a `bi_so_follower` adapter and launch script, package tools (per-arm re-normalization), the option C
  bimanual base builder, latency/VRAM and checkpoint-inspection scripts, and `GPU_RUNBOOK.md`.

**Verified (run on CPU)**
- 26 tests pass (20 numpy-only with the system Python; all 26 in a LeRobot-main CPU venv, two of them on the
  real SO-101 package files from the HF cache).
- On LeRobot's real `flux3` code with a tiny random DiT: our runner (history kept on the robot side, one
  stateless call per chunk) produces **the same commands as LeRobot's documented synchronous loop**
  (`pre -> select_action -> post` every tick) over three replans, to 1e-3.
- One loaded model drives both arms with separate histories, prompts and cameras.
- `make_bimanual_base` turns an SO-101-style package into a 12-D package: trunk and action branch bit-identical,
  only the four embodiment-head tensors resized and re-initialized (zero output layer), grid camera layout,
  `delta_absolute_dims=[5, 11]`, and the result runs end to end in the runner.
- Re-normalizing a package swaps the quantiles and keeps weights bit-identical (symlinked).
- Against the real SO-101 package (config, processor JSONs and quantile files from the HF cache, no weights):
  its config equals our `SO101_CONTRACT`, and our runner matches LeRobot's synchronous loop with the real
  config and real processors around a tiny random DiT (8 ticks, 2 snapshots, 42 / 32 actions, two replans).
- The package's normalization is one community dataset's (`VoicAndrei/so100_kitchen`, legacy degrees), and the
  base ships EE50 heads `action_prediction_ee50` [50 / 50] and gaming heads [64], no 14-D joint head (A.4, B).

**Not verified**
- Anything on a GPU or with the real 7B weights: latency, VRAM, NATTEN on sm_120, LoRA step time.
- Zero-shot quality on our arms, and whether package quantiles (A0) or our own (A0r) work better.
- LeRobot PEFT training of fresh 12-D heads (code path exists, BFL says it is not validated for new embodiments).

**Next 3 steps**
1. GPU box, after `ws4/cloud/bootstrap.sh`: run runbook sections 0 to 2 (session setup, package inspection,
   `measure_latency`, cross-check against BFL's standalone SO-101 inference), and paste the latency JSON into
   this memo.
2. Robot: calibrate both arms in LeRobot degrees mode, serve `so101` from GPU 0, run `flux.run_xlerobot` with
   one arm at a time on a simple prompt, first with re-normalized packages (A0r), then the package as is (A0).
3. Record 50+ single-arm demos per skill of the demo task tonight and start the per-arm task LoRA on GPU 1;
   start the 12-D LoRA on GPU 2 only if two leader arms and 50+ bimanual demos exist.

## Recommendation

**(i) Sunday demo: option A + D.** The released SO-101 checkpoint (with a per-arm task LoRA from tonight's
demos) runs on each XLeRobot arm as its own FLUX session with its own prompt, history and action queue, and a
behaviour tree (WS3, optionally with Claude for decomposition) sequences arm skills: for example "left arm:
push the box out of the way", perception check, "right arm: pick up the cup", hold, place. Coordination
(who moves, holds, mutual exclusion, retries) is scripted; manipulation inside each skill is FLUX. Fallback:
zero-shot SO-101 checkpoint per arm, then the team's single SO-101 station.

**(ii) Tonight on the 3 GPUs:**

| GPU | Job | Why |
|---|---|---|
| 0 | Policy server for robot testing all night and for the demo (`flux.remote`): re-normalized `left`/`right` packages (A0r), `so101` as is (A0), later the LoRAs | The robot needs a GPU in the loop to test anything; 2 to 3 instances fit in 96 GB |
| 1 | Per-arm task LoRA on our XLeRobot single-arm demos (LeRobot `lora.json`, base = the A0r package) | Highest demo value per GPU-hour; existing heads, existing processors |
| 2 | Stretch: 12-D bimanual LoRA from the SO-101 checkpoint (`make_bimanual_base`, option C) if 50+ bimanual demos exist; otherwise a second option A variant (other normalization, or the other arm's adapter) | Only path to learned coordination by Sunday; high risk |

No EE50 training and no flux-action full fine-tuning tonight (sections B and C.3).

## Comparison

| | A0 / A0r: SO-101 zero-shot per arm (package / own quantiles) | A1: per-arm task LoRA | B: EE50 (Simone) | C1: 12-D heads, LeRobot PEFT from SO-101 | C2: 12-D heads, flux-action full FT from base | D: harness on A |
|---|---|---|---|---|---|---|
| Data | none (5 to 10 teleop episodes for a quantile check) | 50 to 100 single-arm demos per skill, one leader arm | FK-converted robot data + UMI converted to unpublished conventions | 50 to 100+ bimanual 12-D demos, two leader arms | hundreds+ bimanual demos | none beyond A |
| Compute | inference only | 1 GPU, roughly 3 to 6 h (estimate) | full FT or new LeRobot processors, multi-GPU | 1 to 2 GPUs, roughly 4 to 8 h (estimate) | 3 GPUs FSDP over PCIe, about 10 h per 1k updates at batch 64 (estimate) | LLM API, cents per plan |
| Time to first result | 1 to 2 h | tonight | days (code missing) | Sunday morning at best | after the hackathon | hours (WS3) |
| Risk | viewpoint, soft gripper, prompt wording; A0 normalizes with a legacy-convention dataset's bounds (A.4) | low to medium | very high | high (unvalidated path, fresh heads, little data) | very high | low |
| Learned coordination | no | no | yes in principle | yes | yes | scripted |
| Judging value | FLUX in the loop, weak reliability | reliable FLUX demo on real hardware (25% + 20%) | originality, but will not run | technical difficulty | none by Sunday | long chains score higher |

## How the SO-101 policy works (what matters for two arms)

- **Contract**: two cameras `observation.images.scene` (left tile) and `observation.images.wrist` (right tile),
  each 256x256 on a 256x512 canvas; 6-D state and actions (shoulder pan, shoulder lift, elbow flex, wrist flex,
  wrist roll, gripper; arm joints in degrees, gripper in percentage points); command deltas with an absolute
  gripper (`delta_absolute_dims=[-1]`), channel weights `[1,1,1,1,1,2]`; 8 observation ticks, 2 encoded image
  snapshots, past-command conditioning; 42 predicted, 32 executed at 30 Hz; Euler 4 steps, shift 6.93, guidance
  3 (video and action), seed 42, BF16 (`fa@830f1ce:tests/fixtures/so101_history/config.json:4-110`,
  `fa@830f1ce:docs/setup.md:172-176`, `fa@830f1ce:docs/so101-lora.md:40-64,137-139`, model card).
- **State lives in two places.** The policy holds only the action queue
  (`lerobot@146d847:src/lerobot/policies/flux3/modeling_flux3.py:240-241,610-623`). The saved preprocessor holds
  the observation, image and command history and the delta anchor
  (`processor_flux3.py:334-338,354-401`); the postprocessor integrates deltas from that anchor and writes back the
  last command (`processor_flux3.py:455-475`), linked by `connect_history_processors` (`:478-508`). So one
  processor pair plus one policy queue serve exactly one arm.
- **Stateless path exists.** With `observation.command_history` in the batch, the preprocessor uses the given
  window instead of its deques (`processor_flux3.py:388-394`; documented in `lerobot@146d847:docs/source/
  flux3.mdx:226-230`). Only the snapshot ticks are encoded: ticks 0 and 7 of 8
  (`f3/packing.py:457-466`). `ws4/flux` keeps the window per arm on the robot side and sends 8 states, 8
  commands and the 2 snapshots per camera per chunk; the parity test shows this equals the synchronous loop.
- **Batching gives no speed-up.** `predict_action_chunk` loops over batch elements and samples each separately
  (`modeling_flux3.py:594-608`); each sample runs two DiT passes per step when any guidance is not 1
  (`modeling_flux3.py:542-547`, `f3/sampling.py:43-66`). One forward with left and right "at batch 2" is two
  sequential samples. It also forces both arms to replan on the same tick.
- **Synchronous only.** `select_action` refuses RTC arguments (`modeling_flux3.py:617-618`); LeRobot's remote
  inference lists FLUX.3 as unsupported because of history and previous-command conditioning
  (`lerobot@146d847:src/lerobot/remote_inference/README.md:161`). Hence our own small server (`flux/remote.py`).

## A. Per-arm, no retraining (and per-arm LoRA)

### A.1 One model for both arms, or two

- One loaded `Flux3Policy` can serve both arms: weights are stateless; give each arm its own history and queue
  (done in `flux/dual_arm_runner.py`) and serialize calls behind one lock (the history step stores the delta
  anchor between pre and post, `processor_flux3.py:394,464-470`). The caption cache is shared and harmless
  (`modeling_flux3.py:399-409`).
- Cost: when both arms need a chunk at once, the second waits one inference. With the runner's
  `async_inference=True`, each backend has one worker thread, so the waiting arm holds while the other keeps
  executing its queue.
- Two instances: about 30 GB each (BFL: about 32 GB BF16 on H200 for DROID, notes folder report item 3), so two
  or three fit on one 96 GB card, but concurrent calls on one GPU contend for compute. For truly parallel
  replans put the second instance on another GPU. `measure_latency --two-instances` measures it.
- Moving Qwen3-VL to CPU after loading saves about 8.3 GiB; captions are cached (`flux3.mdx:234-242`).

### A.2 Latency (estimates until measured)

- Published, "RTX 6000 Pro", DROID workload, median per chunk: 772.6 ms BF16 4-step guided, 426 ms guidance
  distilled, 118.8 ms step distilled; FP8 553 / 309 / 92 ms (https://bfl.ai/models/flux-3-action, latency tab;
  extracted in `ws4/.cache/bfl_docs/latency_tables_from_js.txt`). About 1.95x slower than H200 on every row.
- Those fast paths are DROID only: the prepared BF16/FP8 backend installs only for 8-D, 32-action
  `action_prediction_droid` configs (`fa@830f1ce:src/flux_action/policy.py:55-66,263-283`); there are no
  distilled or FP8 SO-101 packages, and LeRobot `flux3` has no FP8 path.
- SO-101 tokens per DiT pass: 2 snapshots x 128 + 11 future latent frames x 128 + 42 actions + 8 history tokens +
  320 text = about 2,030 (canvas 256x512 -> 8x16 latents, `f3/packing.py:65-79`; `text_fixed_length` 320), versus
  about 3,100 plus text for DROID. Both run 8 passes (4 steps x 2 for guidance). Estimate for SO-101 on our GPU
  in LeRobot's eager path: **0.6 to 1.0 s per chunk** (UNVERIFIED). One chunk executes 32 actions = 1.07 s, so
  a synchronous arm moves roughly half the time. Levers, in order: guidance 1.0 on both streams (one pass per
  step, about 2x), `compile_model=true`, a second GPU for the second arm.

### A.3 Cameras

- Scene = the RealSense head view, wrist = that arm's gripper camera. The model stretches each view to 256x256
  without cropping (`f3/packing.py:162-171`) and BFL built its corpus the same way (`camera_tile_hw` 256x256 in
  `fa@830f1ce:configs/so101/community_corpus.json`). Community cameras are 640x480 (4:3), so stream the
  RealSense at 640x480 (or center-crop 16:9 to 4:3 with `CameraView(aspect=4/3)`) to keep the same squash.
- What the SO-101 model saw as "scene": a VLM-selected top-down or high-angle view per dataset; side,
  wrist-only and human-heavy datasets were dropped (`community_corpus.json`, `selection`). Camera keys in
  `lerobot/community_dataset_v3`: `top` 1,256, `wrist` 848, `side` 847, `front_side` 437, `left_wrist` /
  `right_wrist` 40 each; 640x480 AV1 at 30 fps (notes folder XLeRobot report, item 4). Our mast camera looking
  down at the table is a high-angle view, so it is in distribution as a viewpoint type; seeing two arms is not.
- Options to test zero-shot (runner supports all, `flux/cameras.py`): full head view; the half of the head view
  in front of the arm (`--scene-crop half`, which side is which is UNVERIFIED); mirrored left arm (flip images
  and negate pan and wrist roll, `per_arm_target(mirror=True)`). A mirrored left arm is not a problem in itself:
  the corpus has arms seen from many sides and mountings, and each dataset was normalized separately.
  XLeRobot's two arms use identical chains with the right mount rotated 180 degrees, so lift, elbow and wrist
  flex mirror naturally but pan and roll do not (XLeRobot `b017b5e`, `simulation/xlerobot/xlerobot.urdf:1004`
  vs `:1112`, from the notes report).

### A.4 Units, calibration, normalization

- BFL documents the package's units as degrees and gripper percentage points (`fa@830f1ce:docs/setup.md:172-176`).
  Today's LeRobot degrees mode puts 0 at the middle of each joint's calibrated range
  (`(raw - mid) * 360 / 4095`, LeRobot `motors_bus.py` `_normalize`), gripper `RANGE_0_100`. LeRobot main and 0.5.1
  default `use_degrees=True` (`lerobot@146d847:src/lerobot/robots/so_follower/config_so_follower.py:42`); the
  XLeRobot plugin defaults to `False` (+-100 ranges) and refuses to connect without the base wheel motors, so
  drive the torso with `bi_so_follower` (`config_bi_so_follower.py:25-36`), keys `left_shoulder_pan.pos` ...
  `right_gripper.pos`, cameras `head`, `left_wrist`, `right_wrist` (`flux/run_xlerobot.py`).
- The package's quantiles "assume its joint order, units and calibration; they are not universal SO-101
  calibration values"; different statistics need "a separate base with matching processors"
  (`flux3.mdx:180-184`). **Read from the package** (`ws4/flux/checkpoint_facts.md`): state q01..q99 are pan
  -92..54, lift 85..184, elbow 61..179, wrist flex -24..81, roll -125..60, gripper 0.5..54; command deltas
  about +-3 to 5 deg per tick, gripper 0..54. These are exactly the bounds of one corpus part,
  `VoicAndrei/so100_kitchen` (`fa@830f1ce:configs/so101/community_corpus.json:316`), recorded in the pre-2025
  SO-100 degree convention (zero at a folded pose; ws4/data `contracts.py` calls it `degrees_legacy`).
- Consequence for our arms in today's degrees mode: at a joint's mid-range zero, lift normalizes to -2.7 and
  elbow to -2.0; over Oscar's calibrated ranges (`haller_follower.json`: lift +-104, elbow +-97 degrees) lift spans
  -4.8..-0.6 and elbow -3.7..-0.4, while the package's own data spans -1..1. A fully open gripper (100) maps to
  2.75. So **A0 feeds shifted states** on the two joints that matter most for reaching.
- The corpus mixes units: about 795 parts in degrees and 226 within +-100 (notes report, from
  `community_corpus.json` quantiles), and BFL's full fine-tune normalized each part with its own q01/q99
  (`community_corpus.json` `builder.normalization`). The trunk therefore learned per-dataset normalized joint
  spaces. That supports **A0r: re-normalize per arm** with q01/q99 from our own teleop data, a cheap package
  edit (`flux.package_tools.renormalize_package`: same weights, new processor state, CPU-tested). Use A0r for
  zero-shot and as the LoRA base. BFL keeps the package statistics fixed for its own task LoRAs and calls them
  "a choice for this adaptation workflow" (`fa@830f1ce:docs/so101-lora.md:32-38`), so a LoRA on A0 should also
  adapt, at the cost of starting from shifted inputs. Mapping our degrees into the legacy convention with
  `ArmCalibration` offsets would need the legacy zero pose and signs (UNVERIFIED); not recommended.
- Command-delta scale: one normalized unit of the package is about 3 to 5 degrees per tick (90 to 160 deg/s);
  A0r rescales to our own teleop speed. Keep the guard's per-tick step limit on either way.
- Soft fin-ray gripper: still one channel, 0..100 between the calibrated closed and open ends (0 closed, 100
  open per ws4/kinematics). Re-calibrate the gripper range with the fin-ray fingers mounted. Compliance means a
  "closed" command on an object stalls against it, which is how the stock gripper is used as well. It looks
  different in the wrist view; expect worse zero-shot grasps than on a stock SO-101 (UNVERIFIED).

### A.5 Expected quality

- Zero-shot (A0/A0r): no published success rate on a new SO-101 setup. BFL's SO-101 examples are task LoRAs
  (color selection, screwdriver, box on cup; `fa@830f1ce:docs/so101-lora.md:14-20`). Expect reaching and
  pushing roughly toward the prompted object, unreliable grasps. Good for a "FLUX in the loop" demo of simple
  verbs, not for a judged chain.
- Per-arm task LoRA (50 to 200 demos): this is the workflow BFL documents and ships (`lora.json`: rank 32, EMA
  0.999, batch 2 x accum 4, 10k microsteps; `lerobot@146d847:examples/flux3/lora.json`), with the package's
  heads and processors, so it is the lowest-risk way to make specific skills reliable on our arms, camera and
  gripper. Budget: measure step time and cut `--steps` to fit the night (runbook section 6).
- Data collection is cheaper too: a single leader arm teleoperates one follower; record per-arm datasets
  directly with `scene`/`wrist` keys (runbook section 5). Bimanual 12-D demos need two leaders.

## B. EE50 path (Simone's plan)

### What the source says

- **No EE50 code is released.** `grep -rniE "ee50"` over flux-action `src/`, `docs/`, `configs/` and LeRobot
  `src/` returns nothing. What exists: the base checkpoint "has 457 tensors, with video and text streams, a
  shared action trunk, and ee50/gaming action heads" (base model card), and the conventions described on the
  BFL model page (CONTEXT.md, EE50 section).
- **Head reuse is mechanically supported.** The base header (`ws4/flux/checkpoint_facts.md`) has
  `emb_in.action_prediction_ee50` [3072, 50], `emb_in.action_prediction_ee50_cond` [3072, 50] and the matching
  final layers, gaming heads of width 64, and **no 14-D joint head** (the header metadata says the release
  removed "joint14 heads" from the step-724k EMA trunk). The conditioning width 50 means the EE50
  midtraining conditioned on state only, without past commands (`f3/packing.py:603-604`). The wiring keeps a head
  whose name equals the configured `action_modality` and drops other `action_prediction*` heads
  (`f3/wiring.py:131-156`), so `action_modality="action_prediction_ee50"`, `action_dim=50`,
  `condition_on_past_actions=false` loads these heads in LeRobot (`export_base.py` with the base trunk) or
  flux-action (only the bare `action_prediction` is reserved, `fa@830f1ce:src/flux_action/config.py:177-183`).
- **LoRA exists only in LeRobot.** flux-action has no LoRA or PEFT (no match in `src/`); it does full
  fine-tuning with FSDP2/HSDP (`fa@830f1ce:src/flux_action/training/distributed.py:117-128,155-187`). LeRobot PEFT
  needs a saved complete policy as base (`modeling_flux3.py:251-257`) whose processors match the data.
- **Normalization does not fit.** EE50 is z-scored per embodiment and per signed offset from the anchor
  (BFL model page). Both codebases only implement per-channel q01/q99 range normalization with absolute or
  consecutive-delta actions (`processor_flux3.py:223-290`, `fa@830f1ce:src/flux_action/processing/
  normalization.py:14-80`); LeRobot's state-relative mode cannot be deployed (`modeling_flux3.py:612-616`) and is
  incompatible with history conditioning (`configuration_flux3.py:231-234`). BFL did not release the
  midtraining statistics or the anchor/frame code. EE50 needs a new processor (LeRobot) or a dataset plug plus
  inference code (flux-action `TrainConfig.dataset`, `fa@830f1ce:docs/embodiments.md:61`), all written blind.
- **Camera layout for EE50 is unknown.** No EE50 embodiment ships a layout; egocentric/UMI data has no head
  view like ours.
- **BFL's own result:** DROID EE50 37.88% vs new joint heads 40.13% (larger batch); EE50 suggested "for very
  small data or step budgets" (model page). For a joint-space robot with its own demos there is no evidence
  that EE50 wins.

### Review of Simone's five steps

1. **Dataset preparation.** Right: data is the bottleneck, and FK conversion of our own joint data is easy
   (ws4/kinematics has an EE50 codec and SO-101 FK). Risky: "convert UMI to BFL's EE50 conventions" targets
   conventions that are not published (relative frame, gripper slot, per-offset z-scores; ws4/kinematics
   marks them UNVERIFIED), UMI has wrist fisheye views only, and human wrist trajectories include
   orientations a 5-DoF arm cannot reach. Missing: our own XLeRobot demos come first; public XLeRobot data is
   scarce (ws4-datasets report).
2. **Validate EE50 to XLeRobot.** Right to gate on IK. But for data generated by FK from our joints every
   target is reachable by construction, so IK only fails on model error or on UMI data. The real question is
   what the IK drops: with 5 joints only position plus two orientation components are controllable, so the
   constrained IK must choose which rotation to ignore (ws4-ik owns this). Missing: a joint-space baseline to
   compare against, which A gives for free.
3. **Embodiment fine-tuning with LoRA from the EE50 checkpoint.** Not possible as stated: flux-action has no
   LoRA. LeRobot can build a base that keeps the EE50 heads (head names confirmed, see above), but its
   processors cannot express EE50's anchor-relative, per-offset z-scored representation, so a new EE50 processor
   (normalization and anchor handling above) has to be written and validated first. Full fine-tuning on our 3
   PCIe GPUs is slow (section C.3). Missing: the inference side (EE50 outputs -> IK -> joints every tick;
   `flux/ee50_hook.py` is the hook, without any EE50 backend behind it).
4. **Task LoRA, 50 to 100 demos.** Right number. A second LoRA on top of the first needs merging the first
   adapter into a new complete base package first (LeRobot adapters reference one base,
   `flux3.mdx:186-188`); doable, one extra step.
5. **Evaluation.** Right metrics. Add offline held-out evaluation (`flux-action evaluate`, or LeRobot eval
   loss) before robot time, and sim rollouts via the runner (`GymEnvRobot`).
- **Critical decision.** Agree with the spirit; the evidence already decides it for this weekend: no released
  EE50 pipeline, no LoRA path, unpublished statistics and conventions (only the head names are known), and BFL's
  DROID result favoring joint heads. **Order:** A0r / A0 zero-shot now, A1 task LoRA tonight, C1 as stretch,
  EE50 as a post-hackathon research track (keep ws4/kinematics' codec and IK; they also help analysis and the
  "imagine" visualization).

## C. New 12-D joint heads

### C.1 What changes (both routes)

- `action_dim` 12 (`[left 6 | right 6]`); state is the same 12 (state dim must equal action dim,
  `configuration_flux3.py:328-332`).
- Conditioning head width 24 with past commands (`f3/packing.py:603-604`).
- `delta_absolute_dims=[5, 11]` (both grippers absolute), `action_channel_weights=[1,1,1,1,1,2]*2`.
- Cameras: three (`head`, `left_wrist`, `right_wrist`). `side_by_side` takes exactly two and `single` one
  (`configuration_flux3.py:310-315`); `grid` takes any number on a near-square grid
  (`lerobot@146d847:src/lerobot/policies/flux3/utils.py:52-72`), "but no checkpoint was trained on it"
  (`configuration_flux3.py:69-71`). We use a 2x2 grid on a 384x512 canvas: 192x256 cells keep 4:3, the fourth
  cell is black, 12x16 = 192 latent tokens per frame (1.5x SO-101).
- New processors with 12-D q01/q99 of command deltas and states from our data (`export_base.py:38-78`).
- Data: 30 Hz LeRobot v3, episodes of at least 50 frames for the 8 + 42 window (`fa@830f1ce:docs/so101-lora.md:213-219`).

### C.2 Via LeRobot PEFT (C1, recommended C route)

- `Flux3Policy.from_pretrained(so101, config=cfg12, strict=False)` keeps every trunk and action-branch tensor
  and re-initializes only shape-mismatched head tensors with BFL's reference init (xavier input, zero output),
  `modeling_flux3.py:259-317`, `f3/wiring.py:159-181`; trunk tensors stay mandatory and shape-checked
  (`flux3.mdx:259-263`). `flux.make_bimanual_base` does this plus processors and a training config; CPU test
  confirms the result. The alternative source is the action-pretrained base trunk exactly like
  `examples/flux3/export_base.py` (`--source-kind base`).
- Training: `lerobot-train --config_path=<base>/lora_train.json`. LoRA on all attention and MLP projections,
  heads trained in full as `modules_to_save` at 5x LR (`modeling_flux3.py:224-249`). We add BFL's new-head
  schedule scaled to a short run: trunk frozen 200 updates, warmup 600, heads warmup 200
  (`fa@830f1ce:docs/embodiments.md:63-64`; LeRobot counts microsteps, `flux3.mdx:115-119`).
- Caveats: BFL says new embodiments should use flux-action (`flux3.mdx:244-253`) and that PEFT quality on a new
  embodiment is not established (`flux3.mdx:265-269`). Fresh heads plus a new canvas layout plus little data is
  the riskiest combination we can afford; treat C1 as a stretch with an offline loss check before robot time.
- Data volume (judgement, not measured): the SO-101 package needed no head learning for task LoRAs; here the
  heads start at zero. Plan for 100+ bimanual demos of the target task; 50 is the minimum worth a GPU-night.

### C.3 Via flux-action full fine-tuning (C2)

- Config fields: `action_dim`, `action_modality`, `camera_layout`, `camera_keys`, `canvas_hw`, `fps`,
  `absolute_action_dims`, `camera_dropout`, history fields (`fa@830f1ce:src/flux_action/config.py:44-136`);
  data via `flux-action index-lerobot --camera stream=feature ... --absolute-action-dims 5,11`
  (`fa@830f1ce:src/flux_action/cli.py:101-146`) or a custom `dataset` plug (`docs/embodiments.md:49-72`).
  Runbook section 7 has the full command.
- Memory on 3x96 GB, FSDP `shard_size=3` (must divide world size, `distributed.py:117-128`): fp32 masters
  (`configs/so101/train.json:57`), fp32 grads and AdamW states (about 112 GB for 7B), plus two fp32 power-EMA
  shadows (`training/ema.py:77`, `ema_sigma_rels [0.1, 0.05]`, about 56 GB): about 56 GB per GPU, plus the frozen
  VAE and Qwen (about 11 GB) and activations. Fits with small `windows_per_rank` and one EMA (estimate,
  UNVERIFIED); run `--override steps=4` first.
- Time: BFL's game run did 126 windows per update in 9 to 10 s on 7 H200 (notes report), about 1.9
  windows/s per H200 at roughly 1k tokens per window. Our grid windows are about 3k tokens, the GPU about 2x
  slower and the interconnect PCIe: roughly 0.3 windows/s per GPU, so about 10 h per 1,000 updates at batch 64.
  Not a hackathon-night job.

## D. Harness or hybrid

- Per-arm FLUX sessions do not know about each other: the left arm's policy sees the right arm only as part
  of the scene. Collision avoidance between arms, holds, handovers and ordering must be **scripted**: one arm
  moves while the other holds a safe pose, or workspaces are split (the runner enforces arm ownership; WS3's
  tree decides who owns what).
- **Learned** by FLUX inside a skill: approach, push, grasp, place, recovery from small errors. **Scripted**:
  sequencing, safe poses and resets (waypoints), success checks (perception), retries and timeouts, prompt
  selection.
- An LLM (Claude, USD 200 credits) fits above the tree: turn a goal into a sequence of `(arm, prompt)` steps
  from a fixed vocabulary of prompts the LoRAs were trained with (FLUX conditions on the caption; wording
  outside the training captions is a distribution shift). LLM output never reaches motors directly.
- Runner API for WS3: `start(target, task)`, `stop(target)`, `is_busy(target)`, `step()` / `run()`;
  targets `left`, `right` (6-D packages) or `both` (12-D package), with ownership checks.

## E. Blackwell (sm_120) notes

From the notes report `flux_dual_arm_docs_blackwell.md`, with the source checks done there:
- PyTorch: cu128 wheels of torch 2.10/2.11 include sm_120; flux-action locks torch 2.10.0+cu128; driver at
  least 570.x.
- NATTEN is required by the video VAE for any GPU use (`fa@830f1ce:src/flux_action/models/video_vae.py:26`,
  `lerobot@146d847:docs/source/flux3.mdx:22-29`); wheels `0.21.6+torch2100cu128` and `+torch2110cu128` exist. The
  VAE probes Blackwell (SM100) and Hopper fused kernels first, then generic CUTLASS, then flex attention
  (`f3/video_vae.py:47-76`); on sm_120 expect `cutlass-fna`. Override with `F3_NATTEN_BACKEND=flex-fna` if needed.
- Attention: keep `attn_mode=torch` (PyTorch SDPA covers sm_120). FA3 is Hopper only, FA2 has no sm_120 build
  path; LeRobot's `flash` mode imports FA3 and `cudnn` has no fallback.
- FP8: flux-action's FP8r uses `torch._scaled_mm` and accepts compute capability (12, 0), but only for the
  released DROID FP8 packages; nothing to use for SO-101 or our heads.
- 3 GPUs on PCIe without NVLink: data-parallel LoRA is fine; FSDP full fine-tuning is communication-bound.

## Risks

1. Latency: about 1 s pauses between 1 s chunks make motion look hesitant; mitigations in A.2.
2. Zero-shot transfer to our viewpoint (two arms in view), fin-ray gripper and calibration may be poor; the
   per-arm LoRA is the fix, and needs demos tonight.
3. Network between venue and cloud: the runner sends about 80 KB per chunk (JPEG snapshots); an SSH tunnel
   must survive the demo. Keep a local fallback (team SO-101 station).
4. HF gate access and 40 GB of downloads on the box before anything runs.
5. Unvalidated code paths: C1 (PEFT with fresh heads) and our stateless call path on real weights (parity is
   verified only on a tiny model).
6. Safety: model outputs are unbounded (model card); the runner's guard limits per-tick steps (default 4
   deg/tick, UNVERIFIED tuning) and joint limits (from LeRobot calibration with `--calibration-limits`). Keep a
   hand on the e-stop; LeRobot's `max_relative_target` is a second layer.

## Code (`ws4/flux/`)

| File | What |
|---|---|
| `contract.py` | Robot keys, 12-D split/merge, model contracts (SO-101, bimanual grid), snapshot rule |
| `history.py` | Per-target history with the synchronous processor's semantics |
| `cameras.py` | Robot camera -> model camera: crop, 4:3 aspect, mirror, resize to 256x256 |
| `safety.py` | Step and joint-limit guard, e-stop latch, limits from LeRobot calibration |
| `backends.py` | `FluxBackend` protocol, `FakeBackend`, `LeRobotFlux3Backend` (stateless, single-flight) |
| `dual_arm_runner.py` | The runner: per-arm or 12-D targets, ownership, async inference, calibration, adapters |
| `ee50_hook.py` | EE50 chunks -> joints via ws4/kinematics FK/IK (hook only) |
| `remote.py` | HTTP policy server for the GPU box and `RemoteBackend` client (token, JPEG) |
| `run_xlerobot.py` | LeRobot `bi_so_follower` adapter and launch script for the real robot |
| `package_tools.py` | Quantiles (`export_base` rule), per-arm re-normalized packages |
| `make_bimanual_base.py` | Option C1/C base builder and LoRA training config |
| `measure_latency.py`, `inspect_checkpoint.py` | GPU latency/VRAM; gated header and quantile inspection |
| `checkpoint_facts.md` | What the gated so101, base and droid packages contain (configs, quantiles, heads, revisions) |
| `crosscheck_so101.py` | Same observation through BFL's standalone SO-101 inference and our backend |
| `GPU_RUNBOOK.md` | After `ws4/cloud/bootstrap.sh`: inspection, smoke test, serving, recording, training commands |

Import as `flux` with `ws4/` on `sys.path`. The ws4/sim env plugs in through `GymEnvRobot` (gym-style `reset`
and `step` returning the same observation keys).

## Questions for Oscar

1. RealSense model and the resolution it streams; does the head have pan/tilt motors?
2. Is there a second leader arm (for 12-D bimanual demos)? The calibration folder has `haller_leader.json`.
3. Which HF account has (or can accept) the FLUX gates, and how will the robot PC reach the cloud box (SSH)?
4. Which demo task do we commit to, so tonight's demos and prompts match it?
