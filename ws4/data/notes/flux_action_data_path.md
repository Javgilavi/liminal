# FLUX 3 Action training repo: answers for a bimanual adaptation

Source: read-only clone at `ws4/.cache/flux-action`, commit `830f1ce178d7aea52f18fde31d8e8e36ce6bae21`
(history: `e2dd1d8` initial import, `1af43c2` README, `830f1ce` merge; one branch, `main`). Nothing was
executed. Citations use `flux-action@830f1ce:path:LINE`. Anything marked **estimate** is my own arithmetic,
not a statement of the repo (treat as `VERIFY`).

Headline: this repo is a **joint-space** fine-tuning codebase (DROID 8-D absolute joints, SO-101 6-D joint
deltas, a 4-D game head). The terms "EE50", 25-D per-arm blocks, 6D rotation, z-score stats "indexed by
signed offset from anchor", and the midtraining dataset list (UMI, EgoDex, HOT3D, Ego4D, ...) do **not
appear anywhere in the repo**. They must come from another source (paper, blog, HF model card) and should
be verified there.

---

## 1. Video-only / action-free samples, and actions without state

**Direct answer.** There is no `has_action` flag, modality flag or per-sample loss weight. The only
per-sample knob is an optional `action_mask (D,)` per window; an all-zero mask makes that window's action
loss exactly zero, so a video(+text)-only sample is *expressible* through a custom dataset plug by
zero-filling `state` and `action` and passing `action_mask = zeros(D)`, but the dummy state and noised
dummy action tokens still enter the window's attention and still dilute the action-loss denominator.
"Actions but no state" is not supported either: `state` is mandatory with width `action_dim`; the only
option is a constant or zero state (or, as the games example does, a substitute such as the previous action).

### Evidence

Window contract (all keys mandatory except the mask):
- `flux-action@830f1ce:docs/embodiments.md:12-14`: each sample has `images.<camera>`, `state (D,)`,
  `action (chunk_size, D)`, `task`, `window_seed`, "and optionally `action_mask (D,)`".
- `flux-action@830f1ce:docs/embodiments.md:23-24`: the mask is meant "only if embodiments with different
  action widths share one head. Zero-pad the narrower action to `D` and set the mask per window; the loss
  averages over unmasked dims."

Mask handling in the loss (`flux-action@830f1ce:src/flux_action/processing/packing.py:545-554`):
```python
    sq = (pred[ak].float() - targets[ak].float()) ** 2  # (B, Na, D)
    if action_mask is None:
        action = sq.mean(-1)  # (B, Na)
    else:
        n = action_mask.shape[0]
        m = action_mask.float()  # (n, D)
        ...
        action = (sq * m).sum(-1) / m.sum(-1).clamp_min(1.0)
```
`clamp_min(1.0)` means an all-zero row gives 0 loss, not NaN. But the reductions still count those tokens
(`flux-action@830f1ce:src/flux_action/processing/packing.py:564-569`):
```python
    if reduction == "modalities":
        total = video_weight * video.mean() + action_weight * weighted_action.mean()
    elif reduction == "joint_tokens":
        total = (video_weight * video.sum() + action_weight * weighted_action.sum()) / (
            video.numel() + action.numel()
        )
```
So the effective action weight shrinks with the fraction of action-free windows in a micro-batch.

The mask is read per window in `prepare` (`flux-action@830f1ce:src/flux_action/policy.py:624-629`), stored in
`PreparedWindows.action_mask` (`flux-action@830f1ce:src/flux_action/policy.py:156`), and passed to
`flow_loss` (`flux-action@830f1ce:src/flux_action/policy.py:681-690`). Tests:
`flux-action@830f1ce:tests/training/test_embodiment_plug.py:147-180` and
`flux-action@830f1ce:tests/test_history_training.py:119-137`.

State and action are hard requirements in `prepare`:
- `flux-action@830f1ce:src/flux_action/policy.py:436-442`: `_state` asserts `state.shape[1] == action_dim`.
- `flux-action@830f1ce:src/flux_action/policy.py:589-594`: `actions` must be `(B, chunk_size, action_dim)`;
  under `joint_delta` it also needs `action_prev` (`flux-action@830f1ce:src/flux_action/policy.py:449-456`),
  and the history profile needs `command_history` (`flux-action@830f1ce:src/flux_action/policy.py:577-582`).
- Index-based data (DROID/LeRobot) cannot express missing values at all: rows must be finite
  (`flux-action@830f1ce:src/flux_action/training/data.py:295-297`) and the trainer requires
  `state_dim == action_dim` (`flux-action@830f1ce:src/flux_action/training/trainer.py:325-331`).
- The only per-window "skip" is `*_is_pad` flags, which drop the whole window, video included
  (`flux-action@830f1ce:src/flux_action/policy.py:481-494`).
- The default collate takes the key set of the first item (`flux-action@830f1ce:src/flux_action/training/data.py:338-345`),
  so every window of a run must carry `action_mask` once any does.

What the DiT itself allows: a forward call may omit the action stream entirely. Missing streams become two
dummy tokens that never enter joint attention (`flux-action@830f1ce:src/flux_action/models/transformer.py:758-799`,
`flux-action@830f1ce:src/flux_action/models/transformer.py:851-853`, and the comment at
`flux-action@830f1ce:src/flux_action/config.py:23-25`). This is per call (per packed micro-batch), not per
sample, and `FluxActionPolicy.forward` always builds the action stream
(`flux-action@830f1ce:src/flux_action/policy.py:650-674`).

Games example as precedent for "no real state": "state: the last executed action, same padding"
(`flux-action@830f1ce:examples/games/dataset.py:23`), zeros at frame 0
(`flux-action@830f1ce:examples/games/dataset.py:214`). Note that under `joint_delta` the state is also the
integration anchor at inference (`flux-action@830f1ce:src/flux_action/policy.py:1078-1085`), so "no state"
is only coherent with `absolute` actions.

Midtraining with video/image/audio: the only traces are the trunk's extra content streams
`("video", "video_cond", "image", "image_cond", "audio", "audio_cond")`
(`flux-action@830f1ce:src/flux_action/config.py:26`), their legacy channel widths
(`flux-action@830f1ce:src/flux_action/models/wiring.py:70`), the "co-trained `action_prediction`" branch
(`flux-action@830f1ce:src/flux_action/models/wiring.py:16-24`), and that fine-tuning freezes image/audio
(`flux-action@830f1ce:src/flux_action/training/trainer.py:137-138`,
`flux-action@830f1ce:docs/droid-finetune.md:104`). The base is "Action-pretrained 724k EMA"
(`flux-action@830f1ce:docs/droid-finetune.md:61`). No pretraining/midtraining data loader, mix config or
video-only loss code: **not found in repo**.

### Code paths to change for true action-free (and state-free) samples
1. Dataset plug: emit a per-window `has_action` (or reuse `action_mask` all zeros) plus zero `state`/`action`
   (and `action_prev`/`command_history` if `joint_delta`/history). Keep keys identical for every window.
2. `FluxActionPolicy.prepare` (`src/flux_action/policy.py:577-629`): accept missing `state`/`action`,
   build placeholders, carry a per-window flag in `PreparedWindows` (`policy.py:143-157`).
3. `FluxActionPolicy.forward` (`policy.py:650-674`): for action-free windows set the action timestep to 1
   (pure noise, so the zeros carry no signal) via `action_timesteps` in `packing.build_forward_kwargs`
   (`packing.py:428-475`), and optionally zero or mask the state token; or, when a whole micro-batch is
   action-free, drop `x_<modality>`/`x_<modality>_cond` from the call (the DiT already supports this).
4. `packing.flow_loss` (`packing.py:519-572`): divide the action term by the number of action-bearing
   windows instead of all windows.
5. `Trainer._check_gradients` (`src/flux_action/training/trainer.py:698-710`) runs on update 0 and raises if
   any trainable parameter got no gradient; an all-video first micro-batch with the action stream dropped
   would trip it.
6. State-free with actions: add a state mask/dropout to `PolicyConfig` (no such field exists; fields are at
   `src/flux_action/config.py:44-136`) and apply it in `prepare` (`policy.py:584`) or in
   `history.conditioning_values` (`src/flux_action/processing/history.py:120-134`).

---

## 2. Dataset input format, prepare step, embodiment config, fps, camera tiling

**Direct answer.** Training streams directly from source files: either Cosmos3-DROID (compact LeRobot v3)
via `index-droid`, or any LeRobot **v2.1 or v3.0** dataset (parquet + mp4 video features) via
`index-lerobot`, or anything else through a Python dataset plug. Indexing writes only `manifest.json`,
`rows.f32.npy` (state+action per frame) and, for LeRobot, `statistics.json` (q01/q99); no VAE latents or text
embeddings are precomputed on disk, both encoders run every step (text contexts are cached in memory per
caption). There is no fps resampling: windows are consecutive recorded frames, and `policy.fps` must be set
by hand to the dataset fps (the trainer does not cross-check it).

### Evidence: on-disk formats
- `flux-action@830f1ce:src/flux_action/data/lerobot/index.py:16-24`: reads v2.1 (`meta/episodes.jsonl`, one
  parquet and mp4 per episode and camera) and v3.0 (`meta/episodes/chunk-*/file-*.parquet`, concatenated
  files addressed by `dataset_from_index` and `from_timestamp`).
- `flux-action@830f1ce:src/flux_action/data/lerobot/index.py:96-103`: `codebase_version` must be v2.x/v3.x,
  `fps` "must be a positive whole number".
- `flux-action@830f1ce:src/flux_action/data/lerobot/index.py:143-146`: cameras must be `dtype: video`
  ("only video streams are read"); image-dtype features are rejected.
- `flux-action@830f1ce:src/flux_action/data/lerobot/index.py:296`: state and action are 1-D vectors from
  `--state-key` (default `observation.state`) and `--action-key` (default `action`)
  (`flux-action@830f1ce:src/flux_action/cli.py:117-118`).
- `flux-action@830f1ce:docs/prepare.md:120`: "Training does not convert episodes."
- `flux-action@830f1ce:docs/prepare.md:134-139` and `flux-action@830f1ce:src/flux_action/data/lerobot/index.py:26-30`:
  index = manifest + `rows.f32.npy`, "A training window is then one video seek per camera and one slice of
  the rows array".
- Custom plug: `TrainConfig.dataset = "pkg.module:callable"`
  (`flux-action@830f1ce:src/flux_action/training/trainer.py:92-96`, `flux-action@830f1ce:docs/embodiments.md:18`).

### Evidence: what "prepare" produces
`prepare-droid` / `prepare` write decoded RGB, not latents (`flux-action@830f1ce:docs/prepare.md:46-53`):
`episode.npz` with uint8 `(N, 360, 640, 3)` per camera, 15 Hz timestamps, `state`/`action (N, 8)`, plus
`metadata.json`, `manifest.json`, `source.json`. It is "meant for a few episodes and for inference
observations. Training reads the whole dataset directly from the compact files"
(`flux-action@830f1ce:docs/prepare.md:115-116`).

Encoders run inside the training step: `prepare` does "caption dropout, augmentation, canvas composition,
video-VAE and text encoding" (`flux-action@830f1ce:src/flux_action/policy.py:559-568`), overlapped with the
backward pass on a side stream (`flux-action@830f1ce:src/flux_action/training/trainer.py:28-32`). Text
contexts are cached in memory, bounded at 256 captions (`flux-action@830f1ce:src/flux_action/policy.py:504-516`).
On-disk latent or text-embedding caches: **not found in repo**.

Disk/compute per hour of video: **not stated in repo**. Related figures: the two-episode DROID sample download
is about 800 MB (`flux-action@830f1ce:docs/prepare.md:31`); uint8 frames are "2.2 GB per 32-window batch"
in host memory (`flux-action@830f1ce:src/flux_action/training/data.py:39-40`); a checkpoint "is 4x the fp32
model size" (`flux-action@830f1ce:src/flux_action/training/trainer.py:141`). **Estimate**: the rows array
for 1 h at 30 Hz with 14-D state + 14-D action is 108,000 x 28 x 4 B, about 12 MB; videos stay as the
source mp4 files.

### Evidence: fields that define an embodiment
`PolicyConfig` (`flux-action@830f1ce:src/flux_action/config.py:44-80`): `action_dim`, `action_modality`,
`camera_layout`, `camera_keys`, `canvas_hw`, `chunk_size`, `n_action_steps`, `fps`, `action_scale`,
`gripper_flip_dims`, `action_parameterization`, `absolute_action_dims`, `action_normalization`,
`state_normalization`, `normalization_clip`, `camera_dropout`; history fields `inference_profile`,
`n_obs_steps`, `history_snapshots`, `condition_on_past_actions`, `video_position_fps`, `text_fixed_length`
(`flux-action@830f1ce:src/flux_action/config.py:100-107`). The resume contract lists the same set
(`flux-action@830f1ce:src/flux_action/training/trainer.py:398-430`). Index-side: `--camera STREAM=FEATURE`
in layout order, `gray` for a missing camera (`flux-action@830f1ce:src/flux_action/cli.py:109-116`,
`flux-action@830f1ce:src/flux_action/data/lerobot/index.py:279-282`), `TrainConfig.frame_hw` is the size
frames are rescaled to before tiling (`flux-action@830f1ce:src/flux_action/training/data.py:181-187`).
The trainer reconciles chunk size, widths and normalization with the index
(`flux-action@830f1ce:src/flux_action/training/trainer.py:312-385`).

### Evidence: fps
- Decoding copies consecutive frames with `-fps_mode passthrough`
  (`flux-action@830f1ce:src/flux_action/data/droid/video.py:79-87`); resampling: **not found in repo**.
- `policy.fps` only sets token time ids on a 10 ms clock: `times_to_ids` is `time * 1000 // 10`
  (`flux-action@830f1ce:src/flux_action/models/positional.py:121-122`); video latent `i` sits at
  `i * 4 / fps` (`flux-action@830f1ce:src/flux_action/processing/packing.py:333-335`), action `k` at
  `(k + 1) / fps` (`flux-action@830f1ce:src/flux_action/processing/packing.py:375-377`).
- `_reconcile_with_index` checks `chunk_size` and widths but not fps
  (`flux-action@830f1ce:src/flux_action/training/trainer.py:320-331`), so a mismatch is silent.
- DROID is pinned to 15 Hz (`flux-action@830f1ce:src/flux_action/data/schema.py:44-45`); SO-101 needs 30 Hz
  (`flux-action@830f1ce:docs/so101-lora.md:197`); history profiles place video at `video_position_fps`
  (24 in the SO-101 config) (`flux-action@830f1ce:src/flux_action/processing/history.py:169-183`).

### Evidence: camera layout on the canvas
`flux-action@830f1ce:src/flux_action/processing/packing.py:193-222`:
```python
    if layout == "droid":   # 3 cams of 360x640: wrist full-res on top, left|right half-res below
        ...
        composite = torch.cat([wrist, torch.cat([left, right], dim=-1)], dim=-2)  # (T, 3, 540, 640)
        return pad_composite(composite, canvas_hw)   # reflect-pad to 544x736
    elif layout == "single":  # one camera resized to the canvas
    elif layout == "side_by_side":  # two cameras, each (H, W/2)
    elif layout == "grid":    # compose_grid
```
- `grid`: near-square, row-major, cells `canvas // (rows, cols)`, unused cells black; 3 or 4 cameras give 2x2
  (`flux-action@830f1ce:src/flux_action/processing/packing.py:160-181`). Mixed resolutions are first
  resized to the largest (`flux-action@830f1ce:src/flux_action/policy.py:418-423`). "no checkpoint was
  trained on it" (`flux-action@830f1ce:src/flux_action/config.py:48-53`).
- Canvas sides must be multiples of 32 and at least 32; `droid` needs exactly 3 keys and canvas 544x736;
  `side_by_side` exactly 2 (`flux-action@830f1ce:src/flux_action/config.py:193-204`); `droid` frames must be
  360x640 (`flux-action@830f1ce:src/flux_action/policy.py:416`).
- VAE: 96 latent channels, 32x spatial, 4x temporal, 45-frame chunks
  (`flux-action@830f1ce:src/flux_action/processing/packing.py:57-60`); DROID composite becomes a 17x20
  latent and 2720 video tokens per 33-frame window (`flux-action@830f1ce:src/flux_action/processing/packing.py:56`,
  `flux-action@830f1ce:src/flux_action/processing/packing.py:67-71`).
- SO-101: each view 256x256, scene left, wrist right, 256x512 canvas
  (`flux-action@830f1ce:docs/so101-lora.md:51-58`, `flux-action@830f1ce:configs/so101/train.json:12-14`,
  `flux-action@830f1ce:configs/so101/train.json:55`). Supply separate streams, do not pre-tile
  (`flux-action@830f1ce:docs/so101-lora.md:66`).

---

## 3. EE50 action space

**Direct answer.** EE50 is **not found in repo**: no definition, no per-arm 25-D layout, no 6D-rotation
convention, no anchor/camera frame, no z-score statistics, no joint-to-EE or UMI converter, and no
kinematics code of any kind. The repo's only normalization is per-channel 1st/99th-percentile range
normalization stored in `statistics.json`; padding is zero-fill plus `action_mask`. No midtraining dataset
list exists; the only dataset names are DROID (Cosmos3-DROID), the SO-100/101 community corpus, and the two
browser games.

### Evidence
- Searches over the whole tree (excluding `uv.lock`) for `ee50`, `end_effector`, `6d`, `rot6`, `quaternion`,
  `rotation`, `anchor`, `signed offset`, `zscore`, `umi`, `egodex`, `hot3d`, `ego4d`, `midtrain`,
  `kinematic`, `urdf` return nothing relevant. The only "anchor" is the inference integration anchor for
  joint deltas (`flux-action@830f1ce:src/flux_action/policy.py:1081`).
- The only "aloha" hit is `Ashton3/lerobot-aloha`, a 6-D SO-100 single-arm part of the community corpus
  (`flux-action@830f1ce:configs/so101/community_corpus.json:101`), not the ALOHA bimanual dataset.
- Statistics actually used (`flux-action@830f1ce:src/flux_action/data/lerobot/index.py:636-659`): JSON with
  `format_version`, `normalization: "range"`, `clip`, and per block `q01`/`q99` lists from
  `numpy.percentile(1/99)`, actions after the delta transform, state raw:
```python
    def bounds(values: np.ndarray) -> dict[str, list[float]]:
        return {
            "q01": [float(v) for v in np.percentile(values, 1, axis=0)],
            "q99": [float(v) for v in np.percentile(values, 99, axis=0)],
        }
```
  Formula `x_norm = clip(2 (x - q01) / span - 1, -clip, clip)`, span 1 when degenerate
  (`flux-action@830f1ce:src/flux_action/processing/normalization.py:14-20`,
  `flux-action@830f1ce:src/flux_action/processing/normalization.py:40-45`). One set of bounds per run
  (`flux-action@830f1ce:src/flux_action/config.py:69-76`).
- Gripper conventions that do exist: DROID stores a closed fraction in `[0, 1]` and the policy flips
  `x -> 1 - x` on `gripper_flip_dims` (default `(-1,)`) so the model sees openness
  (`flux-action@830f1ce:docs/prepare.md:55-57`, `flux-action@830f1ce:src/flux_action/config.py:61`,
  `flux-action@830f1ce:src/flux_action/policy.py:384-391`); SO-101 uses percentage points, no flip, gripper
  last (`flux-action@830f1ce:docs/setup.md:172-176`, `flux-action@830f1ce:configs/so101/train.json:25`).
- Padding: zero-pad plus mask (`flux-action@830f1ce:examples/games/dataset.py:110-115`,
  `flux-action@830f1ce:examples/games/dataset.py:215-216`).

---

## 4. Multi-dataset / multi-embodiment mixing

**Direct answer.** One run has exactly one `action_modality` head with one `action_dim`, one normalization,
one parameterization, one camera layout and one fps. Several datasets or embodiments can be mixed only by
writing a dataset plug that zero-pads every embodiment to the widest `D`, sets `action_mask`, and lets the
caption identify the embodiment; the built-in `index` path reads a single manifest. There are no mixing
weights: the plug decides the sampling (the games example mixes in proportion to episode counts).

### Evidence
- One head per run: `action_dim`/`action_modality` scalars (`flux-action@830f1ce:src/flux_action/config.py:46-47`);
  the DiT gets one `<modality>` and `<modality>_cond` pair (`flux-action@830f1ce:src/flux_action/models/wiring.py:93-111`);
  packed training requires "one stream per modality"
  (`flux-action@830f1ce:src/flux_action/models/transformer.py:771-775`).
- The trunk shares one action backbone across "dataset-specific action modalities" and drops other heads
  (`flux-action@830f1ce:src/flux_action/models/wiring.py:16-26`,
  `flux-action@830f1ce:src/flux_action/models/wiring.py:118-143`).
- Index path is a single manifest (`flux-action@830f1ce:src/flux_action/training/trainer.py:222-229`);
  the SO-101 guide says it "trains one dataset per index; it does not reconstruct the historical merged
  corpus" (`flux-action@830f1ce:docs/so101-lora.md:249-251`).
- Mixing through the plug, proportional to episode counts
  (`flux-action@830f1ce:examples/games/dataset.py:28-30`, pooling roots at
  `flux-action@830f1ce:examples/games/dataset.py:65-67`):
```python
    modality "game", action_dim 4:
        GRUNT   [move, strafe, turn, fire]
        VECTOR  [steer, throttle, nitro, 0]     4th dim padded, masked from the loss
    caption: the game's own line, the only thing that says which game a window is from.
```
  (`flux-action@830f1ce:examples/games/dataset.py:19-22`).
- Historical SO-101 corpus (built outside this repo): per-dataset normalization, "merge share cap 5% of
  training windows per part", and "parts whose action dimension is not 6 contribute no windows"
  (`flux-action@830f1ce:configs/so101/community_corpus.json:22-29`,
  `flux-action@830f1ce:configs/so101/community_corpus.json:53-55`). To reproduce per-dataset normalization
  with this trainer, normalize inside the plug and leave `action_normalization`/`state_normalization` at
  `None` (identity, `flux-action@830f1ce:src/flux_action/processing/normalization.py:40-45`).
- EE human-hand plus joint-space robot in one run: mechanically possible only as one padded head with
  disjoint masked dims, a single `action_parameterization` (`flux-action@830f1ce:src/flux_action/config.py:67-68`)
  and a single `fps`; separate per-embodiment heads in one run: **not found in repo**.

---

## 5. LoRA vs full fine-tuning; GPU memory

**Direct answer.** The standalone `flux-action` trainer does **full fine-tuning only**; LoRA exists only in the
separate pinned LeRobot fork (`collab-lerobot-nv-bfl-flux3@c027ee36`) and only as a task adapter on the
prepared SO-101 checkpoint. No minimum GPU memory is stated; the hints are an H200 reference environment,
a ~8.9 GB text encoder, fp32 master weights plus two fp32 EMAs, and FSDP2/HSDP over 8 or more GPUs.

### Evidence
- `flux-action@830f1ce:docs/so101-lora.md:5-6`: "This workflow uses LeRobot's trainer and adapter loader;
  standalone `flux-action` provides full fine-tuning and base-policy inference." Pin at
  `flux-action@830f1ce:docs/so101-lora.md:8-9`. No `peft` dependency in
  `flux-action@830f1ce:pyproject.toml:10-21`; searching `lora`/`peft` in `src/` finds nothing.
- LeRobot preset: "rank/alpha 32, constant adapter/head LRs 1e-4/5e-4, BF16, activation checkpointing and EMA
  0.999. Heads train in full" (`flux-action@830f1ce:docs/so101-lora.md:161-162`); "one GPU, batch 2 and
  accumulation 4" (`flux-action@830f1ce:docs/so101-lora.md:166`). GPU model and memory for it: not stated.
- Hints: H200 example environment (`flux-action@830f1ce:docs/setup.md:13`); 7B model
  (`flux-action@830f1ce:README.md:5`); text encoder "Saves its 8.9 GB of GPU memory" when offloaded
  (`flux-action@830f1ce:src/flux_action/policy.py:327`); default `param_dtype: float32`
  (`flux-action@830f1ce:configs/droid/train.json:23`); EMA shadows in FP32 on the parameter's device
  (`flux-action@830f1ce:src/flux_action/training/ema.py:25-26`); 8-GPU DROID command with `shard_size=8`
  (`flux-action@830f1ce:docs/droid-finetune.md:15-19`); reference "replicate 16 x shard 4 on GB200"
  (`flux-action@830f1ce:src/flux_action/training/distributed.py:120`); peak memory is logged as
  `peak_mem_gb` (`flux-action@830f1ce:src/flux_action/training/trainer.py:683`).
- **Estimate**: fp32 master (4 B) + grad (4 B) + AdamW moments (8 B) + two fp32 EMAs (8 B) is about 24 B per
  trainable parameter, roughly 170 GB of state for 7B before activations and encoders, so full fine-tuning
  realistically needs FSDP across several 80+ GB GPUs. `frozen_steps` etc. must shrink for short runs
  (`flux-action@830f1ce:docs/embodiments.md:20-21`).

---

## 6. Joint-space action modalities (14-D or 12-D bimanual)

**Direct answer.** Joint heads are configured with `action_dim`, `action_parameterization` (`absolute` or
`joint_delta`), `absolute_action_dims` (the dims kept absolute under deltas; there is no field called
`delta_absolute_dims`), `gripper_flip_dims` and `action_channel_weights`. DROID is 8-D absolute and SO-101 is
6-D `joint_delta` with the gripper absolute and weighted 2x; **no bimanual config exists in the repo**. For a
bimanual head you must list *both* gripper indices as absolute, because the indexer's default keeps only the
last dim absolute (the corpus's 12-D SO-100 bimanual parts show exactly that mistake).

### Evidence
SO-101 head (`flux-action@830f1ce:configs/so101/train.json:10`, `flux-action@830f1ce:configs/so101/train.json:25-27`,
`flux-action@830f1ce:configs/so101/train.json:36-39`):
```json
    "action_dim": 6,
    "gripper_flip_dims": [],
    "action_parameterization": "joint_delta",
    "absolute_action_dims": [5],
    "loss_reduction": "modalities",
    "action_channel_weights": [1, 1, 1, 1, 1, 2],
    "action_loss_weight": 0.5,
    "video_loss_weight": 1.0,
```
- DROID defaults: 8-D (7 joints + gripper), absolute, flip `(-1,)`, scale 2, `joint_tokens` loss with
  action weight 50 (`flux-action@830f1ce:src/flux_action/config.py:46-61`,
  `flux-action@830f1ce:src/flux_action/config.py:122-125`, `flux-action@830f1ce:docs/droid-finetune.md:62`,
  `flux-action@830f1ce:docs/droid-finetune.md:70`).
- `joint_delta` semantics: `a[t] - a[t-1]` with `absolute_action_dims` untouched; inference integrates onto
  the observed state (`flux-action@830f1ce:src/flux_action/config.py:62-68`,
  `flux-action@830f1ce:src/flux_action/processing/normalization.py:56-80`).
- Constraints: under `joint_delta` every flipped gripper dim must be absolute; history profile forbids flips;
  `chunk_size % 4 == 0` outside history; channel weights need `action_dim` entries, squared and
  mean-normalized (`flux-action@830f1ce:src/flux_action/config.py:149-150`,
  `flux-action@830f1ce:src/flux_action/config.py:161-170`, `flux-action@830f1ce:src/flux_action/config.py:205-207`,
  `flux-action@830f1ce:src/flux_action/config.py:227-231`, `flux-action@830f1ce:src/flux_action/processing/packing.py:556-563`).
- Conditioning width is `action_dim`, doubled with past-command history
  (`flux-action@830f1ce:src/flux_action/config.py:276-279`); heads are fresh, zero-initialized outputs
  (`flux-action@830f1ce:src/flux_action/models/wiring.py:146-169`).
- Indexer default: `--absolute-action-dims` defaults to `-1` (`flux-action@830f1ce:src/flux_action/cli.py:125-130`).
  The 12-D bimanual SO-100 parts in the corpus have names `d_left_gripper ... right_gripper`, i.e. the left
  gripper was delta-encoded and only the right one kept absolute, and they contributed 0 training windows
  (`flux-action@830f1ce:configs/so101/community_corpus.json:695`,
  `flux-action@830f1ce:configs/so101/community_corpus.json:709`).

**Not from repo, assembled from the constraints above**: a 12-D `[left 6 | right 6]` SO-101 bimanual head
would be `action_dim: 12`, `action_parameterization: "joint_delta"`, `absolute_action_dims: [5, 11]`,
`gripper_flip_dims: []`, `action_channel_weights: [1,1,1,1,1,2, 1,1,1,1,1,2]`, indexed with
`--absolute-action-dims 5,11`; a 14-D (6 joints + gripper per arm) head uses `absolute_action_dims: [6, 13]`
and 14 weights. For three cameras, `grid` (2x2, one black cell) is untrained, while `droid` is the trained
3-camera layout but requires 360x640 frames and puts the first key full-res on top.
