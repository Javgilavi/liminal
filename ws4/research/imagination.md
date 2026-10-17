# Imagination: seeing and using what FLUX 3 Action predicts

Owner: ws4-imagine. Code: `ws4/imagine/`. GPU steps: `ws4/imagine/GPU_RUNBOOK.md`. Written 2026-10-17.

## Status report

**Done**
- Found where the imagined video lives and why nobody sees it. Both LeRobot and flux-action denoise future
  video latents jointly with every action chunk and throw them away at the last line of `_sample`. BFL's
  inference docs confirm the API "returns actions only".
- `ws4/imagine/predict_with_video.py`: a thin wrapper (`Imaginer`) that returns, per chunk, the actions plus
  the decoded imagined frames, their control-step timing and an optional score. It does not edit either library:
  it hooks `policy._sample` on the instance and records the solver output through a thread-local hook. It
  supports sync, async and other-GPU decoding, best-of-N over seeds, imagining alternative instructions, and
  a `replan()` for failure reflexes.
- `compose.py` (ACTUAL | IMAGINED video/GIF with timestamps, chunk boundaries, progress bar and surprise
  trace, plus contact sheets), `score.py` (pixel, colour-zone, latent and embedding goal scorers, a
  `SurpriseMonitor`, and a `ClaudeJudge` on `claude-opus-5-5` with structured JSON output),
  `run_imagine.py` (`fake`, `replay`, `vaecheck`, `bench` commands), `fakes.py`, and 162 tests.

**Verified**
- On CPU against the real LeRobot `Flux3Policy` source (lerobot@146d847, tiny random DiT, SO-101 history
  schema: 8 observations, 2 snapshots, past commands, predict 42 / execute 32). With imagination on, actions
  are bit-identical to stock. The capture returns 11 latent frames (96 x h x w), which decode to 41 frames.
  Best-of-N executes exactly the chosen seed, and instruction alternatives match single runs.
  (`ws4/imagine/tests/test_imagine_lerobot_real.py`, 5 tests, run in a LeRobot-main venv.)
- 157 CPU tests on this laptop's default env (the real-LeRobot file skips there because LeRobot 0.5.1 has no
  flux3). `ruff` is clean.
- The frame timing assumptions hold against the **released** `flux-3-action-so101` config (repo sha
  `0713a7af`), which I parsed with LeRobot's real `Flux3Config` (section 1.7). It gives 11 predicted latent
  frames on an 8 x 16 grid, 41 decoded frames at offsets +2 .. +42, a 256 x 512 canvas, and 2 snapshots of 8
  observations.
- The released video VAE (`flux-3-action-base/video_vae.safetensors`, header only) matches LeRobot's VAE module
  exactly: 547 tensors, no name or shape differences. It has 96 latent channels, /32 spatial, and 2 temporal
  merge plus 2 temporal expansion layers (section 1.7).

**Not verified (needs the cloud GPUs, see the runbook)**
- Real-weight frame quality, decode latency and VRAM on RTX PRO 6000 (sm_120, NATTEN `cutlass-fna`
  expected).
- That the trained VAE decoder reproduces each slot's frame in order (the architecture matches; `vaecheck`
  measures the learned behaviour).
- The flux-action standalone path (same hook point, not run), DROID frame-conditioning on real code, and
  the gd/sd/fp8 variants.
- Claude judge latency and usefulness.

**Next 3 steps**
1. On the cloud box, after `ws4/cloud/bootstrap.sh`: runbook steps 0 to 5 (shell setup, tests, `vaecheck`,
   `bench`, `replay`) on one recorded episode. That takes under an hour and yields the first
   imagined-vs-actual clip for the video.
2. With ws4-flux-dual / WS5: `Imaginer(policy, decode="async").install()` in the live runner, plus the
   panel in the operator UI. Log `ActualFrame`s and record every run.
3. With WS3: pick 1 to 2 decision points in the long task for "imagine alternatives, Claude picks". Calibrate
   the `SurpriseMonitor` threshold on the replays to drive retries.

## 1. Where the imagined frames are, exactly

Pinned sources: LeRobot `146d847` (`src/lerobot/policies/flux3/`, below "LR"), flux-action `830f1ce`
(below "FA"), HF model cards, https://docs.bfl.ai/flux_3/flux3_action_inference,
https://bfl.ai/models/flux-3-action.

### 1.1 The tensors

- `Flux3Policy._sample(cond, caption, seed)` (LR `modeling_flux3.py:513-578`) builds two streams from seeded
  noise: `x_video` (`n_pred` latent frames, line 518-524) and `x_action` (line 525-527). It runs one solver over
  the dict: `sampling.cosmos_unipc_order2` or `sampling.euler` (lines 572-577; both return the final dict
  of every stream, `f3/sampling.py:69-83, 134-220`). It then returns only `out[ak][0] / action_scale`
  (line 578). The module docstring says it outright: "Video is never decoded here" (line 18). flux-action is
  identical in structure (FA `policy.py:786-879`). Its optimized DROID DiTs also return the video stream
  (`DroidPrediction.as_dict -> {"x_video", ...}`, FA `models/transformer_inf_bf16.py:100-105`,
  `transformer_inf_fp8r.py:161-171, 983-987`). FA `docs/setup.md:393`: "Video tokens are sampled jointly with
  actions but are not decoded."
- BFL inference docs, "Predicted video": the sampler predicts video latents, but the API returns actions only,
  does not decode, and has no video flag. The overview's camera views are "a scripted illustration". **So far
  nobody shows real FLUX imagination; we would be first.**
- Format at the end of the solver: `x_video` is `(1, n_pred * h * w, 96)` fp32 (the solver state stays fp32,
  LR `modeling_flux3.py:528-530`). The token order is `(t h w)` with 96 channels (`f3/positional.py:38-51`;
  the VAE has 96 latent channels, /32 spatial, /4 temporal, `f3/packing.py:52-54`). There is no further
  patchify, so `tokens.reshape(n, h, w, 96)` recovers the latents (`tokens_to_latents`).
- The conditioning tokens `x_video_cond` arrive in `cond` and are captured too. For SO-101 they are 2
  independently encoded snapshots (`packing.py:462-467`), which decode to the "what the model saw" frames.

### 1.2 How many frames, which resolution, which time

`n_pred` comes from the packer (LR `packing.py:556-557, 607-608`): `latent_frames(chunk)` for history
conditioning, `latent_frames(chunk + 1) - 1` for frame conditioning, with `latent_frames(T) = 1 + (T-1)//4`.
The decoded frame count and timing follow from the VAE's two `TemporalMerging` layers (encoder, LR
`video_vae.py:194-224`: an odd sequence duplicates its first element, then pairs merge) and two
`TemporalExpansion` layers (decoder, `video_vae.py:249-270`: each latent becomes 2 frames and the first output
frame is dropped, so `n` latents give `4n - 3` frames). `encoder_slots` / `decoded_slots` in
`predict_with_video.py` reproduce these rules and are tested for every clip length up to 99.

| Checkpoint | Conditioning | Future clip | Latents | Decoded frames | Control offsets shown | Seconds ahead | Canvas |
|---|---|---|---|---|---|---|---|
| SO-101 (`flux-3-action-so101`) | history, chunk 42, 30 Hz | 42 frames at +1..+42 (LR `configuration_flux3.py:382-387`), encoded alone (`packing.py:469-473`) | 11 | 41 | **+2 .. +42** | 0.067 .. 1.40 s | 256 x 512 (scene 256x256 left, wrist right) |
| DROID (`flux-3-action-droid`) | frame, chunk 32, 15 Hz | window of 33 frames at 0..+32, encoded as one clip (`packing.py:570-576`) | 8 (+1 condition) | 32 after dropping the condition frame | +1 .. +32 | 0.067 .. 2.13 s | 544 x 640 (wrist on top, two exteriors below) |
| LeRobot default history config | history, chunk 32 | 32 frames | 8 | 29 | +4 .. +32 | 0.13 .. 1.07 s | config |

Relation to the actions: in history mode action `k` sits at `k / fps` (LR `packing.py:620-621`), so the frame
at offset `+m` shows the result of actions `0 .. m-1`. SO-101 executes 32 actions and then replans.
- Decoded frame 30 (offset +32) is the imagined view at the next replan. Compare it with reality for
  success checks.
- Frames 31 to 40 (+33 .. +42) imagine 10 steps beyond the replan point that are never executed. That is
  free look-ahead.
- The 42-frame future clip has an even length. Its first latent therefore merges frames +1 and +2, and the
  first decoded frame shows +2. Offset +1 has no frame of its own. `vaecheck` tests this on the real VAE
  (**UNVERIFIED** until then).

Other facts:
- Position ids use `video_position_fps = 24` (`packing.py:611-617`). They are RoPE positions only and do not
  change the 30 Hz timing.
- Imagined resolution equals the canvas (`latent_hw * 32`). For SO-101, each camera is imagined at 256 x 256.

### 1.3 Decoding

`VideoVAE.decode(latents)` takes normalized latents `(B, 96, n, h, w)` in bf16 and returns
`(B, 3, 4n-3, 32h, 32w)` in [-1, 1] (LR `video_vae.py:881-895, 983-985`; it denormalizes internally).
Two rules follow from how each packer encoded its targets:

- **History (SO-101):** decode the predicted latents alone.
- **Frame (DROID):** decode `cat([condition_latent, predicted])`, then drop the first decoded frame.
  Decoding the predictions alone would misread latent 1 as a causal first latent.

Both are implemented in `LatentDecoder.decode_record` and tested on the fake VAE.

The VAE needs NATTEN, which is already a hard requirement because the policy encodes its camera inputs with
the same VAE. On sm_120 the backend probe (LR `video_vae.py:58-75`) skips `blackwell-fna` (SM100 only) and
should pick `cutlass-fna`. The NATTEN cu128 wheels include sm_120 (sibling notes
`flux_dual_arm_docs_blackwell.md` section 5.2). Override it with `F3_NATTEN_BACKEND`.

### 1.4 Extra latency and VRAM

- **Capture:** about zero. The video is computed in every chunk anyway. The wrapper copies 1408 x 96 floats
  (540 KB) to CPU right after the solver; the action chunk is synchronized at the same point anyway.
- **Decode memory:** about 5 GB peak on the decode device. Estimate from the VAE's own calibration
  constants, not measured: `1.35e6 B x (11*8*16) + 2 GiB`, times the 1.25 safety factor
  (`video_vae.py:106-112, 876-879`). That is far below the threshold where chunked decoding kicks in on a
  96 GB card. A second VAE copy on another GPU costs 2.34 GiB of weights (`video_vae.safetensors`, HF API
  size).
- **Decode time:** **UNVERIFIED**, measure it with `bench`.
  - Back-of-envelope (est.): about 7 TFLOP of decoder matmuls for 11 latents at 256x512 (4 Swin stages,
    dims 2048 to 256, depths 8/8/4/1), plus NATTEN windows.
  - That suggests tens to a few hundred ms on an RTX PRO 6000, which is well under the 1.07 s execution
    window.
- **Asynchronous decoding never slows control.** In LeRobot's synchronous loop the GPU is idle while the arm
  executes the 32 queued actions (RTC/async inference is not supported, per LR docs `flux3.mdx` and the BFL
  inference docs). A background decode on the same GPU therefore uses idle time. `AsyncDecoder` copies
  latents to CPU in the control thread, decodes on a worker thread, drops chunks rather than block if its
  queue fills, and can target another device (`decode_device="cuda:1"`).

### 1.5 Which checkpoints

- **SO-101 BF16 via LeRobot:** works. Verified against the real code with a tiny model; real weights
  UNVERIFIED.
- **DROID root BF16 via LeRobot:** same hook, frame-packer decode rule. Tested on the fake only.
- **flux-action standalone (all six DROID packages):**
  - `_sample(cond, caption, seed)` is the same hook point (FA `policy.py:786`). It calls `sampling.*` at call
    time through `from .inference import sampling` (FA `policy.py:36`), so the capture hook applies to the
    eager and the prepared BF16/FP8r paths.
  - `VideoSpec.from_config` reads FA's `inference_profile`.
  - Not run. **UNVERIFIED**.
- **Distilled variants:**
  - gd (guidance-distilled) and sd (1-step) exist only for DROID. HF `flux-3-action-so101` ships a single
    package.
  - In sd, the video is the one-step x0 estimate. Whether distillation kept video quality is not documented
    and is **UNVERIFIED**.
  - FP8r packages require flux-action (DROID card).
  - For the team's SO-101 and XLeRobot work only the BF16 eager path matters. The prepared fast path is
    DROID-geometry only (FA `policy.py:55-66`).
- **XLeRobot fine-tunes (ws4-flux-dual, flux-action):** the trunk, VAE and `_sample` stay the same, so
  imagination keeps working if the video loss stays on. The LeRobot default is `video_loss_weight=1.0`
  (LR `configuration_flux3.py:165`). **Ask fine-tuners not to set it to 0.** A grid or 3-camera layout
  decodes to one canvas; `Imagination.views()` then returns the whole canvas.

### 1.6 Does guidance change frame quality?

- Each stream has its own scale: `guidance = {"x_video": guidance_scale, "x_action": guidance_scale_action or
  guidance_scale}` (LR `modeling_flux3.py:542-545`). The combination is `uc + s (c - uc)` per stream
  (`f3/sampling.py:43-66`).
- SO-101 ships 3 / 3. Model card: "four Euler steps, shift 6.93, guidance 3". LR `flux3.mdx` notes 4/1 as an
  opt-in. DROID ships 4 / 1.
- Any scale other than 1 doubles the DiT passes, and the unconditional pass runs once for all streams
  (`modeling_flux3.py:547`). Changing the video scale therefore does not change the cost.
- Video CFG is not a free "prettier frames" knob. The streams are denoised jointly, so it changes the
  actions too. BFL's sweep on DROID: video CFG 4 / action 1 gave 42.00% success, video CFG 6 / action 1
  dropped to 39.67% (bfl.ai model page).
- Keep the checkpoint's settings on the control path. Then the frames shown are the frames the policy
  actually acted on, which is also the honest story for judges.
- Higher video CFG for offline, visualization-only imagination (e.g. counterfactuals in the video) may sharpen
  frames: **UNVERIFIED**.

### 1.7 Checked against the released checkpoint config (2026-10-17)

I downloaded only `config.json`, `policy_preprocessor.json`, `policy_postprocessor.json` and `lora.json` of
`black-forest-labs/flux-3-action-so101` (repo sha `0713a7af`), no weights. The config is identical to the
schema the real-code tests use (flux-action `tests/fixtures/so101_history/config.json`), apart from the pinned
encoder ids.

| Setting | Released value | What the Imaginer derives | Match |
|---|---|---|---|
| `conditioning`, `history_snapshots`, `n_obs_steps` | history, 2, 8 | 2 conditioning snapshots, 128 tokens each | yes |
| `condition_on_past_actions`, `action_representation`, `delta_absolute_dims` | true, delta, [-1] | (does not touch actions; postprocessor integrates) | n/a |
| `chunk_size`, `n_action_steps`, `fps` | 42, 32, 30.0 | `latent_frames(42)` = 11 predicted latents | yes |
| `canvas_hw`, `camera_layout`, `camera_keys` | [256, 512], side_by_side, [scene, wrist] | latent grid 8 x 16, decoded 256 x 512, scene left | yes |
| decoded frames and offsets | n/a | 41 frames at +2 .. +42; offset +32 is index 30 | n/a |
| `video_position_fps` | 24.0 | RoPE positions only: future latent i at id `(2 + i) * 4 / 24 s` (33, 50, .. 200); the actions sit at k / 30 s | yes |
| `sampler`, `num_inference_steps`, `sampler_shift`, `guidance_scale`, `guidance_scale_action` | euler, 4, 6.93, 3.0, null (= 3.0) | CFG on both streams: two DiT passes per step | yes |
| `inference_seed` | 42 | best-of-N candidates use seeds 42, 43, .. (candidate 0 = stock) | yes |
| `video_loss_weight` | 1.0 | the video stream was trained, so imagination is meaningful | yes |
| `video_vae_id`, `text_encoder_id` | `black-forest-labs/flux-3-action-base:...@62878e29` | the same frozen VAE the policy uses decodes the frames | n/a |

Method: `PreTrainedConfig.from_pretrained(<downloaded dir>)` in the LeRobot `146d847` venv returns a
`Flux3Config`. `packing.build_packer(cfg).predicted_latent_frames(cfg)` returns 11, and
`VideoSpec.from_config(cfg)` reproduces the offsets above.

The pre/post-processor JSONs confirm the processor chain the wrapper assumes:
- **Preprocessor:** rename, batch, device, history normalizer (`n_obs_steps` 8, `chunk_size` 42, past
  actions), action-target normalizer, camera resize (a no-op for side_by_side).
- **Postprocessor:** history unnormalizer (delta integration), then CPU.

**VAE (checked 2026-10-17, no weights downloaded).** `flux-3-action-base` is accessible now.

The repo has no VAE config file:
- At the revision the SO-101 config pins (`62878e29`, 2026-09-22) it holds only `flux-3-action-base.safetensors`,
  `video_vae.safetensors` and `text_encoder/`.
- Main (`99bc185b`) adds a 333-byte `config.json`. It is a manifest only: `type flux3_action_base`, weight
  file names, 457 tensors, bf16, streams video/text/action.
- `video_vae.safetensors` is byte-identical in both revisions (2,508,132,617 bytes, LFS sha256 `db1c3211...`).

So the VAE architecture lives in code, LeRobot's `ViTNormInferenceParams` (LR `video_vae.py:79-103`). I checked
it against the checkpoint itself by reading only the safetensors header (`HfApi.parse_safetensors_file_metadata`,
a range request).

| Check | Result |
|---|---|
| Tensors in the checkpoint vs `ViTNormInference(ViTNormInferenceParams())` built on the meta device (LeRobot `146d847`) | 547 vs 547; **no missing, no extra, no shape mismatch** |
| dtype, parameter count | 546 bf16 + 1 bool (`z_normalizer.initialized`); 1.254 B parameters |
| Latent channels | `decoder.proj_in` [2048, 96], `encoder.proj` [192, 2048] (mean + logvar of 96), `z_normalizer.running_mean/var` [96]: **96 channels** |
| Spatial patch | `decoder.proj_out` [48, 256] = 1 x 4 x 4 x 3; with 3 PatchMerging stages: **/32 spatial** |
| Temporal layers | exactly **2 `TemporalMerging`** in the encoder (features 7 and 10) and **2 `TemporalExpansion`** in the decoder (features 1 and 5): **/4 temporal**, the structure `encoder_slots` / `decoded_slots` mirror |
| Causality | `enc_causal=True`, `dec_causal=False` (code defaults, consistent with the module the weights load into) |

The duplicate-first-frame rule for odd lengths (`TemporalMerging(dim, dim)` uses the default `zero_pad=False`,
`video_vae.py:601, 212-215`) and the drop-first rule of `TemporalExpansion` are code, not weights, so this
header check confirms the structure the frame timing rests on. `load_state_dict` is strict
(`video_vae.py:1022, 1026`), so a mismatched file would fail to load rather than decode wrong.

One check remains, on the GPU box: `vaecheck` (runbook step 3). It confirms that the trained decoder reproduces
each slot's frame in order (decoded frame d shows control step d + 2 for the 42-frame SO-101 clip).

### 1.8 Were the pen demos too fast? (measured 2026-10-17 22:30 CEST)

No. They are slower than FLUX's training data.

**Speed tails.** Per-tick command deltas at the 1st..99th percentile (deg per tick at 30 Hz), stock SO-101 package
(community data) vs `so101_haller_pen_B84` (our 84 train takes):

| | pan | lift | elbow | wrist flex | roll |
|---|---|---|---|---|---|
| community | -3.6..3.3 | -2.6..4.1 | -3.4..4.1 | -5.1..5.4 | -3.3..3.2 |
| ours | -0.7..1.4 | -1.9..1.8 | -1.7..1.9 | -1.3..2.4 | -0.8..1.3 |
| spread ratio, ours / community | 0.31 | 0.54 | 0.47 | 0.34 | 0.33 |

These are tails, not typical speeds. Variant B renormalizes with our quantiles, so it rescales speed anyway.

**What the model does instead** (`outputs/speed/`). B84, guidance 1.0, zero-shot, 127 chunks of 12 held-out takes
(101 to 103, 20 to 22, 36 to 38, 86 to 88), requests built like the runner's. Joint travel over the 32 executed
steps, model / demo:

| | pan | lift | elbow | wrist flex | roll | gripper |
|---|---|---|---|---|---|---|
| ratio | 1.49 | **0.32** | **0.55** | 1.06 | 0.99 | 1.00 |

Median total arm travel per chunk is 40 deg for the model vs 53 deg for the demos, and the model moves more in only
39 % of chunks.

Same script on the overnight LoRA (`imagine/speed_check.py`; B84 rerun through it gives identical ratios):

| model / demo travel | pan | lift | elbow | wrist flex | roll | gripper | median arm travel (demo 53.5 deg) |
|---|---|---|---|---|---|---|---|
| B84 zero-shot | 1.49 | 0.32 | 0.55 | 1.06 | 0.99 | 1.00 | 40.2 deg |
| LoRA 3000 steps | **0.86** | **0.52** | 0.56 | 0.64 | 0.56 | 0.96 | 30.1 deg |

At 3000 steps the LoRA stops over-swinging the pan and uses the lift more, but it moves more cautiously
overall than the demos. 5000 steps: pending.

The mismatch is which joints move, not speed. The model barely uses shoulder lift and elbow (the reach toward the
pen) and swings the pan more. This fits the lift findings in `single_arm/README.md`. It is what the pen LoRA has
to learn; compare its checkpoints with the same script.

## 2. Uses, ranked for Sunday

The ranking weighs judging criteria (use of track tech 20%, originality 15%, demo video 10%), risk to the
live demo, and effort.

| Rank | Use | Value | Risk to control | Effort | API |
|---|---|---|---|---|---|
| 1 | **Live IMAGINED / ACTUAL panel** in the operator UI plus a recorded side-by-side for the video | Highest: the track is literally "Imagine, Then Act", and BFL's own page only shows a scripted illustration | None (async decode in idle GPU time) | Low | `Imaginer(policy, decode="async").install()`, `compose.Renderer`, `render_run` |
| 2 | **Surprise reflex**: actual vs imagined frame for the same control step; spike, then replan or report | High: visible failure anticipation (drop the cube on purpose, and the graph spikes and the robot replans) | Low (one extra chunk when it fires) | Low to medium (calibrate the threshold on replays) | `score.SurpriseMonitor`, `Imaginer.replan()` |
| 3 | **Imagine alternatives, then act** at 1 to 3 decision points: one observation, N candidate instructions, N imagined futures, Claude picks | Very high originality; a direct copy of tau0-VLA's world-model-guided search, using FLUX itself as the world model | Robot pauses N chunks + 1 API call (a few seconds) | Medium (WS3 supplies the candidates) | `alternatives_batch`, `ClaudeJudge.rank` |
| 4 | **Success anticipation at subtask boundaries**: is the imagined end frame a success (colour-zone predicate, goal image, or Claude `anticipate`)? After execution, `compare` imagined vs actual and return continue / retry / recover | Medium to high: a second, model-internal success signal next to WS3's perception checks | None if async | Medium | `ColorFractionScorer.predicate`, `GoalImageScorer`, `ClaudeJudge.anticipate/compare` |
| 5 | **Seed best-of-N** inside `select_action` (N seeds, score imagined futures, execute the best) | Uncertain: seed diversity with 4 steps and CFG may be small (**UNVERIFIED**, measure the candidate spread in `bench`/`replay`) | N x inference pause per chunk | Low (already implemented) | `Imaginer(n_candidates=N, scorer=...)` |

### Notes per use

**(a) Panel.**
- Use `same_time` alignment for the "does reality follow imagination" view. It shows imagined and actual
  frames of the same moment and computes surprise.
- Use `lookahead` for "what it is about to do". The imagined frame k steps ahead appears next to the live
  camera.
- Mark chunk boundaries (yellow border) so viewers see the replanning rhythm. The panel also shows the
  executed vs imagined-only part of the chunk.

**(b) Anticipation.** Three signals, from cheap to expensive:
1. `SurpriseMonitor` (pixel distance on downsampled frames, per step, windowed).
2. Colour or goal predicates on the imagined end frame (offset +32), before the chunk runs.
3. `ClaudeJudge.compare(imagined_end, actual_end)` at subtask boundaries.

Feed them to whatever orchestrator WS3 runs as plain booleans/scores. The imagination never commands motors
and never marks a goal done by itself. The orchestrator decides.

**(c) Best-of-N cost.**
- Candidates run sequentially through the stock `_sample`, each a full solver run with CFG.
- Latency is about N x the stock chunk time, plus N decodes for pixel scorers (none for `LatentGoalScorer`).
- Batching seeds on one GPU would save less than N-fold at about 2,000 tokens per window (est.: already
  compute-bound).
- Three policy replicas on 3 GPUs (about 24 GiB of weights each) would run 3 candidates at 1x latency. That
  conflicts with training, so use it only for the final demo if at all.
- Against 30 Hz control with 32 executed steps (1.07 s of motion), best-of-N only makes sense at decision
  points, not every chunk.

**(d) Other 2026 work worth copying.**
- **tau0-VLA** (arXiv 2608.16885, section IV-B / VI-D):
  - Searches over **language subtasks**, not action chunks. A world model predicts each candidate's terminal
    head-camera image, and a VLM value model scores it (5 ordinal levels). They run a beam search, then a
    reflective final pick.
  - Open-loop subtask accuracy, OOD book organization: 74.0% vs 50.0% plan-once vs 57.5% best-of-N.
  - Closed loop: 5/10 to 7/10, 6/10 to 9/10, 5/10 to 7/10. No latency reported.
  - Our version (use 3) needs no extra world model: FLUX's own imagination is the predicted terminal image,
    and Claude is the value model.
- **NVIDIA, "Pretrained to Imagine, Fine-Tuned to Act"** (developer.nvidia.com blog):
  - Cortex 2.0 scores candidate rollouts for "progress, risk, and efficiency" and conditions execution on the
    best. Our `RANK_SCHEMA` uses the same three axes (progress, risk, plausibility).
  - Fast-WAM reports 590 to 800 ms per chunk with full video generation, vs about 190 ms for pi0.5.
  - A Veo 3.1 probe showed artifacts such as a gripper changing shape. That is why the judge prompt
    penalizes implausible imagination.
- **Language-grounding demo:** the same observation imagined under two instructions ("pick the red cube" vs
  "pick the blue cube"), side by side. It shows that the model's imagination follows language. It is cheap,
  zero-risk, good for Q&A, and is the same `alternatives_batch` call.
- **Not for Sunday:**
  - Visual subgoals (pi-0.7 style) would need training.
  - Using imagination as an inverse-dynamics plan (UniPi) would duplicate what FLUX does jointly.

## 3. Integration design

```
observation -> saved preprocessor -> policy.select_action(batch)
                                         |  (queue empty)
                                         v
                      policy.predict_action_chunk(batch)          <- Imaginer._predict_hook (records per item)
                                         |  per item
                                         v
                      policy._sample(cond, caption, seed)          <- Imaginer._sample_hook
                         for seed in seeds:  with capture(sampling):  original _sample -> action, out["x_video"]
                         optional: decode candidates, scorer(...) -> chosen
                                         |
                       chosen action chunk returned unchanged -> queue -> postprocessor -> robot
                                         |
                       Imagination(record) -> decode (sync | async thread | other GPU) -> on_frames(record)
```

- `Imagination` fields:
  - `action` (chunk, normalized model units) and `frames` (T, H, W, 3) uint8.
  - `frame_offsets` (control steps) and `cond_frames` (VAE view of the conditioning snapshots).
  - `latents` (96, n, h, w) and `step` (select_action tick).
  - `candidates` (seed, score, time), `chosen`, `score`, and `timings` (sample_s, decode_s, score_s).
  - Helpers: `frame_at(offset)`, `executed_end_frame()`, `final_frame()`, `views()` (scene / wrist split),
    `save()` / `load_imagination()`.
- **One-shot API for any orchestrator:** `Imaginer(policy, scorer=...).imagine(preprocessed_batch,
  n_candidates=N)` returns a list of `Imagination` (one per batch item) with actions, frames and score.
  `alternatives_batch(batch, instructions)` turns one observation into N instruction candidates.
- **Live API:**
  - `install()` makes `select_action` imagine on every chunk. Records go to `on_imagination` /
    `on_frames` / `imaginer.history`.
  - `last_step` tags camera frames for alignment, and `replan()` flushes the queue.
  - `uninstall()` restores the policy exactly. The wrappers are instance attributes; deleting them reveals the
    class methods.
- **Contracts kept:**
  - Stock actions are bit-identical with `n_candidates=1` (tested on real code).
  - Inference stays single-flight per model (lock).
  - The solver hook records only in the capturing thread, so two policies in two threads (one per arm) do not
    mix.
  - The wrapper never touches processors, normalization or the robot. Processors are stateful: never run the
    preprocessor twice per tick to get more candidates. The wrapper samples candidates inside one
    `predict_action_chunk` call instead.
- **For sibling workstreams:**
  - ws4-flux-dual: install it on each policy instance in the runner.
  - ws4-ttt: `Imagination.latents` / `cond_latents` are available as features.
  - WS5 UI: call `compose.Renderer.render(row)` per tick, or push `record.frames` to the browser.
  - WS3: `ClaudeJudge`, `SurpriseMonitor`, predicates.
- **Absolute commands:** `to_robot_actions(postprocessor, chunk)` uses the saved postprocessor. For history
  checkpoints, a 3-D chunk integrates deltas from the current anchor without updating the live
  `last_command` (LR `processor_flux3.py:455-475`).

## 4. Latency budget (SO-101 geometry, 30 Hz, 42 predicted / 32 executed)

| Item | Value | Source |
|---|---|---|
| Robot motion per chunk | 32 / 30 Hz = **1.067 s** | model card, BFL inference docs |
| Stock chunk inference, RTX PRO 6000, BF16, 4 steps, CFG 3/3 | **UNVERIFIED.** DROID geometry on the prepared path measured **772.6 ms** (FP8 553.1, gd 426.0, sd 118.8). SO-101 has about 2,034 tokens per window vs about 3,100 for DROID, but runs the eager reference path. Expect 0.5 to 1.0 s (est.) | bfl.ai Table 4 via sibling notes `flux_dual_arm_docs_blackwell.md`; token count `ws4/research/robottt_flux.md`; FA `policy.py:55-66` |
| Control pause per chunk | = stock inference (synchronous loop, no RTC) | LR `flux3.mdx` |
| Capture of imagined latents | about 0 (540 KB device-to-host copy) | code |
| Decode, async (same GPU idle during motion, or cuda:1) | **0 on the control path** | design |
| Decode itself | **Measured 0.11 to 0.12 s** for 41 frames at 256 x 512 (RTX PRO 6000 Blackwell Server, 23 chunks of held-out pen takes, decoded after each take) | `imagine/README.md`, `/workspace/imagine/replay_ep10{1,2,3}/report.json` |
| Round trip through `flux.remote`, B84 package, guidance 1.0, on the pod | **Measured median 0.39 s with imagination (`lazy`) vs 0.39 to 0.43 s plain; commands bit-identical on all 23 chunks** | same |
| Decode VRAM | about 5 GB peak (est. from code constants) + 2.34 GiB if a second VAE copy is used | `video_vae.py:106-112`, HF API |
| Best-of-N (seeds) | about N x stock inference + N x decode | code |
| Alternatives (N instructions) | about N x stock inference + N x decode + 1 Claude call | code |
| Claude judge call | **UNVERIFIED** latency (seconds). Cost est. about USD 0.02 to 0.04 per rank call (about 10 images x about 170 tokens + text at USD 4 / 20 per MTok for Opus 5.5). USD 200 is thousands of calls | claude-api skill pricing; image tokens about 1 per 28x28 patch |

Rule of thumb: imagination for display and monitoring is free. Deliberation (best-of-N, alternatives, judge)
costs seconds, so spend it at most at a few decision points per task and show it on screen as "imagining".

## 5. Recommended demo moments

1. **Cold open (video 0:00 to 0:15, live pitch first 20 s):**
   - Full-screen split ACTUAL | IMAGINED while the robot starts the task. Caption: "FLUX 3 Action imagines
     the next 1.4 s, then acts."
   - The imagined gripper reaches the object about a second before the real one.
2. **Throughout the long task:**
   - Keep the panel on a monitor next to the robot during the live demo, with chunk boundaries and the
     surprise trace visible.
   - Judges see that every action chunk comes with a prediction of its consequences.
3. **The decision point (the most original beat):**
   - At the obstacle, the robot pauses. The screen shows a 2 to 3 tile grid of imagined futures for
     candidate instructions ("push it left", "lift it"). Claude's pick is highlighted with its one-line
     reason, and the robot executes it.
   - Reality then matches the chosen imagination (contact sheet in the video).
4. **Anticipated failure:**
   - A teammate moves or knocks the target mid-chunk. The surprise trace spikes, the robot replans
     immediately (`replan()`), and the new imagination shows the recovery.
   - In the video, freeze-frame the spike.
5. **Closing slide:**
   - A contact sheet of one chunk: imagined row over actual row at +0.13 .. +1.4 s.
   - One line: "first public decode of FLUX 3 Action's imagination, used to choose and to check actions".

For Q&A, have the counterfactual pair ready (same scene, two instructions, two imagined futures) and the
`bench` numbers.

## 6. Risks

- **Imagination quality out of distribution:** our XLeRobot scene, TPU fin-ray grippers and cameras are
  unlike the community SO-101 data. Frames may be blurry or wrong. Mitigations:
  - Show replays from the team's own recorded episodes first. LoRA on 50 to 100 own demos also trains the
    video stream.
  - In the pitch, wrong imagination is a feature: it is exactly what the surprise reflex catches.
- **Decode speed on sm_120 (`cutlass-fna`):** if decode exceeds about 1 s, decode every second chunk (the
  async queue already drops rather than blocks) or decode only for recording.
- **Upstream API drift:** pin LeRobot `146d847`. The wrapper raises a clear error if `x_video` disappears from
  the solver output.
- **Judge latency, refusals or outages:** the judge returns the policy's own seed on any error, and
  server-side refusal fallback is on (`fallbacks="default"`). It never blocks the control loop unless used at
  a deliberate decision point.

## 7. Evidence index

- LeRobot main `146d847` (cloned sparse to `ws4/.cache/imagine-lerobot`):
  - `policies/flux3/modeling_flux3.py:18, 147, 240-241, 399-409, 513-578, 580-608, 610-623`
  - `configuration_flux3.py:153-158, 165, 279-281, 382-387`
  - `f3/packing.py:52-55, 65-79, 162-171, 443-474, 556-582, 607-621`
  - `f3/positional.py:38-51`
  - `f3/sampling.py:43-83, 134-220`
  - `f3/video_vae.py:58-75, 106-112, 194-270, 556-601, 834-895, 946-985`
  - `processor_flux3.py:354-401, 455-475`
  - `docs/source/flux3.mdx` (install, guidance 3/3 vs 4/1, offline prediction with `command_history`,
    Qwen offload)
- flux-action `830f1ce` (`ws4/.cache/flux-action`):
  - `src/flux_action/policy.py:36, 55-66, 786-945`
  - `models/transformer_inf_bf16.py:100-105`, `models/transformer_inf_fp8r.py:161-171, 983-987`
  - `docs/setup.md:393`, `docs/so101-lora.md:194`
  - `tests/fixtures/so101_history/config.json` (SO-101 schema), `configs/so101/community_corpus.json`
- HF model cards (`flux-3-action-so101`, `-droid`, `-base`, repo sha `0713a7af` for so101), file sizes from
  the HF API.
- https://docs.bfl.ai/flux_3/flux3_action_inference ("Predicted video", loop, plan length).
- https://bfl.ai/models/flux-3-action (token groups, CFG sweep, latency tables, distillation).
- arXiv 2608.16885 (tau0-VLA).
- https://developer.nvidia.com/blog/pretrained-to-imagine-fine-tuned-to-act-the-rise-of-world-action-models/
- Sibling notes in the session notes folder: `flux_dual_arm_docs_blackwell.md` (RTX 6000 Pro latency table,
  sm_120 env, NATTEN), `robottt_flux_findings.md` (token count).
