# Pen LoRA training and deployment audit

Audit date: 18 October 2026. Scope: Oskrt's red pen insertion adapter, all episodes in its current source dataset, the `ws4-handoff` training pipeline, and the available local robot trials.

The strongest evidence points to several training to deployment mismatches, compounded by an early checkpoint and limited recovery coverage. The optimizer recipe is substantially consistent with the official recipe. The B84 normalization calculation is correct. There is no basis for concluding that a visible, stationary second arm makes a single-arm policy invalid.

The five available B84 robot trials used EMA step 4,000 and all started with an almost closed gripper, unlike 82 of the 84 training starts. Four ended on predicted wrist-limit violations. One additional trial changed the adapter's normalization at inference. The current Hub contains raw step 10,000, which was not used in those trials. These trials do not establish the capability of the current adapter under matched conditions.

## Evidence and reproducibility

| Artifact | Pinned identity |
| --- | --- |
| Remote `ws4-handoff` | `1a0d5e38b6a312e9797d47bb9bb7ffba90289cb3`; checked against the remote |
| Current Hub adapter | `Oskrt/flux-3-action-so101-haller-pen-lora@e85fc5831ae9d28a27ef175aa1c0196b6bc5ec16` |
| Adapter SHA256 | `997284e4e0335baccf3849cf6824611a2f8de1feaf70373917d1c49ceb98d76e` |
| Adapter used in local B84 trials | `492e06e50f94fdb26286bf88b90e8faf657c4bb2`, EMA 4,000 |
| Dataset | `Oskrt/haller_put_the_red_pen_in_its_slot_in_the_cardboard_box@96307c0fcf7e379515b03ccb602de152c9f53954` |
| Training base recorded by branch | SO101 `0713a7af6768562e618e81d534372c6795d41b2e` with B84 processors |
| Branch bootstrap LeRobot pin | `146d847`; official source inspected at `146d8474932c8b566717eb2738ba459337ab71c4` |

Independent checks in this audit cover all 55,422 numerical rows, all 126 episodes, all four MP4 files, all 110,844 decoded camera frames, all current adapter tensors, and ten local pen run reports. Visual review covers four temporal samples per camera per episode, 1,008 images. This is not a frame-by-frame semantic annotation of grasp success. Existing offline prediction scores below are attributed to the branch/model card; GPU inference was not rerun in this audit.

Evidence files, scripts, and the visual index are in [the audit directory](../../outputs/audits/pen-deep-20261018/). The downloaded source data and checkpoint are cached under `/tmp/pen-audit-20261018`. Production code, configurations, motors, training jobs, and public repositories were not changed.

## Ranked findings

| Priority | Finding | Evidence and causal status |
| --- | --- | --- |
| 1 | Robot starts differ substantially from demonstrations | Confirmed in all five B84 run logs. Strong candidate for failure to initiate grasping; a matched-start comparison is still required. |
| 1 | Old EMA checkpoint used for live trials | Confirmed revision in reports. Raw 10,000 improves published offline metrics; its real success is unmeasured. |
| 1 | Normalization changed for one adapter trial | Confirmed HF Space profile. This changes both input and output coordinates seen by the learned adapter and invalidates a matched evaluation. |
| 2 | Camera and control clocks are not equivalent | Confirmed wrist history duplicates, asynchronous sampling, and long pauses between chunks. Exact sensor capture synchronization is unrecorded. |
| 2 | Final lid-held scene absent from this adapter's training | Confirmed episode IDs. Relevant when deploying in layout E; it cannot explain earlier trials that preceded that layout. |
| 2 | Offline score does not test closed-loop task completion | Confirmed evaluation construction. It supplies expert histories and measures joint error, not grasp, insertion, release, or recovery. |
| 3 | Limited training exposure and narrow demonstrations | Confirmed short run and four layouts. More training may help, but is not a demonstrated solution by itself. |
| 3 | Gripper and release quality need targeted review | Weak-release episodes were retained; substantial command/state gripper lag exists. Some lag is normal actuator response, not automatically bad labels. |

## Critical frequencies and synchronization

**A 30 fps MP4 and 30 Hz dataset rows do not establish 30 fresh, synchronized images per second.** Capture, sampling, history assembly, inference, and execution use different clocks.

| Stage | Intended setting | Observed or verified result |
| --- | --- | --- |
| Training recorder | 30 Hz | Training wall-clock intervals imply about 29.995 Hz; p99 interval 34.582 ms; four gaps over 100 ms; maximum 163.881 ms. |
| Training video encoding | 30 fps | `ffprobe` reports 30/1 for all four files. Frame counts match episode metadata and all frames decode. This is an encoding property. |
| Camera hardware capture | Configured 30 fps | No per-frame exposure timestamp or frame counter is stored in the dataset or model request. Actual historical capture rate remains unproven. |
| Live scene observations | Declared 30 Hz | Server schedules observation messages at that rate, with reads/encoding on the same loop. Exact arrival cadence is not retained in the saved histories. |
| Live wrist snapshots | Nominal 30 Hz loop | HTTP fetch completes, then the loop sleeps 1/30 s. Fetch period is request duration plus 33.33 ms, so sustained fetch rate is below 30 Hz. |
| Model history | Eight state/command ticks | Nominal span is 7/30 = 233.3 ms. Saved items lack timestamps, so actual span and jitter cannot be reconstructed exactly. |
| Visual conditioning | Two snapshots per view | The checkpoint selects history indices 0 and 7, not eight independently encoded visual snapshots. |
| GPU prediction | Four Euler steps per chunk | Median model time around 0.386 s in B84 trials; this is not the whole request latency. |
| Request and response | One per chunk | Median RPC 0.583–0.711 s across B84 runs; requests include serialization and transport. |
| Replanning | After 32 executed actions | Median plan intervals 1.680–1.782 s, about 0.56–0.60 Hz. Model inference is not running at 30 Hz. |
| Within-chunk commands | 30 Hz | Mean 29.34–29.36 Hz in B84 runs; median intervals about 34.03–34.05 ms. |
| Between-chunk commands | Hold during inference | Median boundary gaps 624–726 ms; observed maximum 2.547 s. |
| Overall command throughput | Often described as 30 Hz | Only 16.27–17.88 Hz across the five B84 runs once holds are included. This is burst-and-pause playback, not uniform 17 Hz playback. |
| Command acknowledgment | Per command | Median about 1.1–1.2 ms, p99 about 5.5–6.2 ms. It is not the main source of the long pauses. |
| UI telemetry | Configured 20 Hz | Logged telemetry events about 19.54–19.63 Hz. Policy observation code independently reads arm state, so 20 Hz telemetry alone does not prove a 20 Hz policy-state rate. |
| Video positional clock | `video_position_fps=24` | Saved checkpoint convention, distinct from dataset/control `fps=30`. Do not “fix” it to 30 in an existing adapter without compatibility testing. |
| Servo internal loop | Hardware dependent | Not measured by these logs. Command submission and acknowledgment do not prove actuator settling or internal servo rate. |

The five B84 runs contain 134 saved histories, including one request without a completed plan. Their wrist images have 146 identical adjacent pairs out of 938, **15.6%**; per-run fractions are 14.3–16.9%. Scene images have one identical adjacent pair out of 938. Median unique images per eight-tick history: scene 8, wrist 7. All sampled history endpoints differed for both cameras.

That last distinction matters: because this model encodes the first and last visual snapshots, a repeated middle wrist frame does not by itself prove corrupted visual conditioning. It does prove that “each history tick contains a newly fetched wrist image” is false. Endpoint acquisition times and scene/wrist skew still need measurement.

The training decoder check found very few exactly identical resized adjacent frames, but H264 is lossy: independently decoded images can differ even if the recorder sampled the same source exposure. Those counts cannot certify capture FPS. Camera wrappers use `read_latest(max_age_ms=500)`, which returns the latest buffered frame without requiring a new frame ID. “Fresh” currently means less than 500 ms old, not captured for this tick. Live wrist freshness is additionally stamped at HTTP completion, not exposure time. The scene and wrist are fetched through different paths, without a common capture timestamp.

Relevant code: [live wrist loop](../../src/liminal/policy/live.py), Haller `cameras.py:150–211`, `policy_ingest.py:411–441`, `policy_bridge.py:329–414`, and LeRobot `camera_opencv.py:434–568` in the inspected local checkouts. Detailed measurements are in [frequency-audit.json](../../outputs/audits/pen-deep-20261018/frequency-audit.json).

Haller's localhost API was offline during this audit. Therefore hardware capture FPS, current USB scheduling, exposure duration, and exact live history cadence remain unmeasured. A live frequency audit should record monotonic capture timestamp and increasing frame ID for each camera, state-read timestamp, committed-action sequence/time, observation send/receive time, selected history timestamps, inference start/end, and motor command/ack time. Report p50/p95/p99 interval, age, skew, repeat fraction, and missed deadlines for each stage. Polling the same JPEG repeatedly is not a substitute for sensor counters.

Proposed implementation: carry capture metadata with both images in the same observation message; assemble history against a 30 Hz timestamp grid; pace snapshot acquisition to absolute deadlines rather than sleeping after work. Preserve the existing synchronous semantics until a separately tested latency-aware execution strategy exists. Starting predictions early without accounting for commands executed during inference creates a new anchoring error.

## Training data and arm identity

The source has 12 state and command channels. The task arm is the physical robot-left arm, named `right_*` in Haller's operator-frame dataset. The converter selects those six channels and maps head to scene and right wrist to wrist. The adapter's saved joint order is shoulder pan, shoulder lift, elbow flex, wrist flex, wrist roll, gripper. Five joints use degrees; gripper uses 0–100, with 100 open. Raw dataset actions are absolute commands. The processor converts the five joints into consecutive command deltas; gripper remains absolute.

The unused arm's five-joint measured range across the 84 training episodes is only approximately `[0.264, 0.264, 0, 0.176, 0.176]` degrees. Its command range is similarly small. This supports that the correct moving arm was selected and the other arm really was stationary in this training set. A six-action policy need not output commands for every object or arm visible in its scene camera.

Current Hub data contain 126 episodes; training used the earlier 104-episode snapshot: 84 train, 12 held out, eight excluded. The original rejected IDs are 15, 16, 23, 28, 34, 43, 61, 100. The saved training configuration excludes them. Held-out IDs are 20–22, 36–38, 86–88, 101–103. The split implementation holds out the final `ceil(n * eval_split)` episodes per task; the deliberately ordered 96 IDs and split 0.1197916667 produce the intended 84/12 division.

Layout E, episodes 104–125, was added later. The current split has 101 training episodes and 15 held-out episodes, but **that newer split was not used to train this adapter**. Visual samples show a changed second-arm pose and a lid-held arrangement. Adding data to the Hub does not update existing weights. A fixed second arm in the original pose is supported by the demonstrations; a newly positioned arm, changed occlusion, or moving box/lid is a different condition. Masking or cropping that arm only at deployment would introduce another visual shift.

Training frames are unevenly distributed: A 13,259; B 5,997; C 16,297; D 3,133; E zero. Uniform frame sampling gives layout D only 8.1% of samples and A+C about 76.4%. Many poses and paths repeat, so frame count overstates scene diversity. The holdout uses new takes of known layouts, not unseen scene configurations.

## Demonstration quality

All numerical state/action values are finite, frame indices are contiguous within episodes, and every episode has sufficient length for the model's history and prediction window. Video lengths agree with metadata; the head and wrist may roll into new files at different episode positions, which is valid because their offsets are stored separately.

The visual sheets show coherent scene and moving-wrist roles, visible pen approaches and transfers, and the transition to layout E. They also show people in the background, occasional close human presence, and variable release/end poses. These are useful review targets, not proof of a universal label failure. The existing rejection list addresses several failed/reversed/intervened demonstrations. Retained episodes 58, 85, and 99 have weak releases according to the review metadata; episode 14 was retained despite an unverified head-frame outcome. Recheck those dense video segments before a new training run. The four-sample sheets cannot independently certify every successful release.

Measured training starts have median state `[0, 2.20, 0.79, 0.22, 0, 99.35]`. Eighty-two of 84 start with gripper above 90. Episodes 5 and 77 are exceptions, starting with the gripper nearly closed and non-home joints. This does not provide broad recovery coverage from arbitrary closed-gripper states.

Mean command/state differences during training are `[0.59, 1.80, 1.45, 0.82, 0.41]` degrees and 5.12 gripper percentage points. Gripper error exceeds 50 points in 1,379 training rows. Abrupt open/close commands and physical response delay can produce this; inspect lag and contact phases before relabeling commands as measurements. Replacing command targets with observed state would change the learning problem.

About 21.1% of training rows have a preceding commanded step exceeding one degree in at least one arm joint. A blanket one-degree-per-tick deployment clamp would distort these demonstrations. This is a compatibility measurement, not a reason to bypass safety constraints. The current Liminal trial path uses calibrated limits and checks command acknowledgments; the branch's older runner has additional caps, so comparisons must identify the runner version.

## Normalization and checkpoint integrity

Independently recomputed B84 quantiles match the current adapter's stored tensors: maximum discrepancy below 1e-6 for state and below 6e-8 for action. Episode-start joint deltas were explicitly zeroed, with gripper absolute. All three processor states agree on action statistics. There is no sign of held-out contamination in these quantiles or a mistaken absolute-action statistic being used as a delta statistic.

Choosing task-specific quantiles before adaptation is a deliberate departure from the prepared-base default, not intrinsically a bug. Keeping them fixed during adaptation and inference is the key contract. Replacing B84 with HF Space quantiles after training changes normalized observations, historical deltas, and output delta magnitudes. The gripper range changes especially strongly. The trial `20261018-000058` used this substituted profile; it cannot be interpreted as a clean evaluation of the trained B84 policy. See [BFL's normalization guidance](https://docs.bfl.ai/flux_3/flux3_action_so101).

The downloaded current adapter has 642 tensors and 136,851,456 elements: 318 LoRA A tensors, 318 LoRA B tensors, and six head tensors. All are finite; all LoRA B elements are nonzero, excluding the simple failure mode of an untouched zero-initialized adapter. Its PEFT configuration targets attention/MLP projections and includes the action and conditioning heads. Tensor validity does not establish correct deployment merge or successful behavior. The deployed merged model's complete tensor equivalence to the intended base plus adapter was not revalidated on the GPU host.

Calibration remains a physical convention, not something quantiles prove. The source and local trial profiles agree on native units/order and wrist rotation, but a hardware recalibration or mounting change must be checked against recorded physical poses. The [upstream calibration issue](https://github.com/black-forest-labs/flux-action/issues/2) is a useful diagnostic reference, not a universal sign/offset correction to apply to an already trained native-coordinate adapter.

## Optimizer recipe and training exposure

| Setting | Saved training configuration | Assessment |
| --- | --- | --- |
| LoRA rank and alpha | 32 / 32 | Matches official preset |
| Adapter learning rate | 1e-4, constant | Matches preset |
| Head learning rate | 5e-4 | Matches preset |
| Batch and accumulation | 2 × 4 = 8 | Matches preset |
| Precision | BF16, gradient checkpointing | Matches preset |
| AdamW | betas 0.9/0.999, epsilon 1e-8, decay 0, clip 1 | Matches preset |
| Warmup and decay | None | Preset choice, not an accidental omission |
| EMA | Constant 0.999 | Matches preset; requires comparing raw and EMA |
| Caption dropout | 0.1 | Matches preset |
| Image/model augmentation | Disabled | Matches preset; limited visual robustness remains a concern |
| Loss weights | Video 1, action 0.5; gripper channel 2 | Saved model convention |
| History and horizon | Eight ticks, two visual snapshots, 42 predictions, 32 executed | Saved model convention |
| Guidance in trials | Video/action 1/1 | Deliberate evaluated override from saved default 3; branch reports no benefit from higher guidance |

The [official LeRobot preset](https://github.com/huggingface/lerobot/blob/146d8474932c8b566717eb2738ba459337ab71c4/examples/flux3/lora.json) is a baseline, not a convergence guarantee. The guide's pinned old collaboration-repository links returned 404 during this audit; the public LeRobot source at the branch's bootstrap pin was available. Record the actual executed commit and installed dependency versions in the next checkpoint manifest, rather than relying only on bootstrap defaults.

Steps mean microbatches. EMA 4,000 represents 1,000 optimizer updates and 8,000 sampled windows. Raw 10,000 represents 2,500 updates and 20,000 windows. The run reportedly stopped at 12,145 of the planned 20,000 microsteps. The inspected sampler excludes eight initial and 42 final positions per episode, leaving 34,486 valid training window starts. Thus 4,000 and 10,000 microsteps amount to approximately 0.23 and 0.58 passes over valid window starts. Each window contains many correlated frames; these are not independent new demonstrations. This supports testing further training, but does not prove the task merely needs a larger step count.

Under a constant 0.999 EMA, the retained initial contribution is approximately 37% after 1,000 updates, 29% after 1,250, and 8% after 2,500. This is the ordinary exponential coefficient; exact initialization behavior can slightly change it. Early EMA can lag a rapidly adapting raw model substantially.

## Checkpoint selection and offline evaluation

The [training loop](https://github.com/huggingface/lerobot/blob/146d8474932c8b566717eb2738ba459337ab71c4/src/lerobot/scripts/lerobot_train.py) computes dataset validation loss on live/raw weights. EMA is saved separately; its use for environment evaluation does not change that validation-loss path. Earlier selection paired a raw-weight loss score with an EMA artifact. Raw/EMA identity must be part of every score and checkpoint selection.

The [pinned model card](https://huggingface.co/Oskrt/flux-3-action-so101-haller-pen-lora/blob/e85fc5831ae9d28a27ef175aa1c0196b6bc5ec16/README.md) reports raw 10,000 at mean arm MAE 3.15 degrees and six-channel direction agreement 89%, compared with raw 5,000 at 3.24 degrees and 83%. Branch results report EMA 4,000 arm MAE around 3.83 degrees. These are existing offline measurements, not new reproductions.

The evaluator supplies recorded state/command history and images, predicts once, and compares the next 32 commands against the expert. It does not feed its own errors back into the world. It samples 200 windows from 12 held-out episodes; overlapping windows are correlated, and only 181 meet its moving criterion. A positive direction score can coexist with a grasp miss, insufficient displacement, or mistimed release. Several degrees of arm error are consequential for a thin pen and narrow slot; exact Cartesian tolerance requires measured kinematics and geometry.

The branch reports EMA 5,000 path lengths only 0.55–0.78 times demonstration motion for the five arm joints. That could contribute to slow progress, but extrapolating it to raw 10,000 without measuring is unjustified. Check phase-specific displacement, gripper timing, and actual task success alongside MAE. Compare hold and ACT baselines using matched control settings; the branch's ACT scores do not constitute a live-robot success baseline either.

## What happened in the local robot trials

| Run | Start gripper | Start wrist flex | Overall command Hz | Termination |
| --- | --- | --- | --- | --- |
| 233652 | 1.57 | 73.10 degrees | 17.10 | Predicted wrist beyond limit |
| 234139 | 2.23 | 45.49 degrees | 17.24 | Predicted wrist beyond limit |
| 234453 | 1.57 | 67.12 degrees | 17.88 | Predicted wrist beyond limit |
| 234630 | 3.01 | 14.81 degrees | 16.27 | Cancelled |
| 235010 | 3.47 | 62.20 degrees | 17.55 | Predicted wrist beyond limit |

These 17 October runs all used B84 EMA 4,000. The wrist-limit refusals occur near the calibrated +98.48-degree limit; they show the guard prevented invalid targets. Do not remove that limit to make the rollout continue. The important preceding question is why the model repeatedly drives wrist flex toward it from these unusual starting states.

Four other pen launches failed before any command due to worker exit; a checked worker log shows CUDA out-of-memory. Those are infrastructure failures, not failed grasps. The single HF-normalized adapter trial also began closed and was cancelled. No independent success labels are present in these logs.

## Experiments that distinguish the causes

1. **Establish a matched baseline before retraining.** Use the current raw 10,000 adapter with its original base/encoders and B84 processors, exact task text, correct task arm, and training-matched scene and wrist orientation. Reset to a measured demonstrated home pose with the gripper open using the existing supervised hardware workflow. Do not command a nominal pose without checking the real calibration and scene.
2. **Measure all clocks in an observation-only run.** Record frame IDs and capture timestamps, state/command timestamps, history spacing, endpoint ages, and camera skew. The evidence above makes this a required diagnostic before attributing failures to model capacity. Fix wrist polling and lost timing provenance, then compare identical saved requests through the training/evaluation processor and deployment processor.
3. **Compare checkpoint variants under identical conditions.** EMA 4,000, raw 10,000, and EMA 10,000 if available. Keep B84 and 1/1 guidance fixed. Use predeclared starts and object positions, reset between trials, and label approach, grasp, lift, placement, release, recovery, timeout, and guard refusal separately.
4. **Separate scene and start-pose effects.** Compare original A–D scene with layout E; separately compare demonstrated home with modest documented perturbations. Do not change checkpoint, normalization, lid pose, start pose, and timing together.
5. **Audit action timing on saved windows.** Match the official state history `s[t-7:t]`, command history `a[t-8:t-1]`, and target `a[t:t+41]`. Haller's recorder publishes a pre-read state with the committed target for that tick, which is compatible in intent; camera capture phase and the exact recorded software revision remain to be proven. Compare ±1-tick shifts offline rather than guessing a correction.
6. **Train for the deployment scene and recovery behavior.** Include reviewed layout E demonstrations if the other arm must hold the lid. Add successful recoveries from missed grasps, closed-gripper starts, and small pen displacement. Balance layouts and phases; reserve genuinely unseen configurations for final evaluation. Keep processor statistics fixed when continuing this adapter, or start a clearly separate normalization experiment.
7. **Extend training only with controlled selection.** Preserve optimizer/EMA state when resuming. Compare raw and EMA using the same deterministic held-out sample set and phase metrics. Evaluate further checkpoints rather than assuming 20,000 microsteps is sufficient. Log actual valid samples, optimizer updates, loss components, clipping, and camera/source revisions.
8. **Evaluate latency changes separately.** A shorter executed chunk offers more opportunities for correction but increases the fraction of wall time spent waiting under synchronous inference. Local inference or a proven latency-aware strategy may help; naive overlapping inference risks stale anchors. Retain safety supervision and truthful command feedback throughout.

A practical first experiment is a matched raw-10,000/B84/home-pose trial series with timestamped observations. If it still cannot approach and grasp the pen in the original training scene, the next investigation should focus on processor equivalence, action/camera alignment, and phase-specific prediction errors. If it succeeds there but fails with the lid held or a changed start, the evidence supports collecting the missing conditions rather than changing normalization or blaming the visible second arm.
