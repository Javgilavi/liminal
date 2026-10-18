# FLUX 3 Action checkpoint facts (read 2026-10-17, no weights downloaded)

Read with Oscar's HF token from the gated repos: small files with `snapshot_download` (configs, processor JSONs,
quantile safetensors), weight files only through their safetensors header (HTTP range request,
`python -m flux.inspect_checkpoint ... --keys`). Revisions read: so101 `0713a7af6768562e618e81d534372c6795d41b2e`,
base `99bc185b6ea58c496ffae14c0fce9cc29be1a045` (main), droid `d84c80adc0721780245aabb75e45d4164ead7b97` (main).

## Revisions

- so101 `main` (`0713a7a`, Oct 14) differs from `c9e13b2` (Sep 22, pinned in flux-action `docs/so101-lora.md`) only
  in README/LICENSE: `config.json`, `model.safetensors`, both processor JSONs, the three quantile files,
  `dataset_statistics.json` and `lora.json` have identical hashes.
- The so101 config pins its encoders to base `62878e2925e59b7a89ec14463ce89932624c490d`
  (`video_vae_id` / `text_encoder_id` end in `@62878e2...`). At that revision and at base `main` the files
  `video_vae.safetensors` (sha256 `db1c3211548371cf...`), `text_encoder/model-0000{1,2}-of-00002.safetensors` and
  `flux-3-action-base.safetensors` (sha256 `eade59cbc25f2fdc...`) are identical, so prefetching base `main` into the
  HF cache covers what the so101 package resolves.

## flux-3-action-so101

- `config.json`: identical to the flux-action test fixture schema (`tests/fixtures/so101_history/config.json`)
  with real encoder ids: `conditioning=history`, `n_obs_steps=8`, `history_snapshots=2`,
  `condition_on_past_actions=true`, `action_representation=delta`, `delta_absolute_dims=[-1]`,
  `action_channel_weights=[1,1,1,1,1,2]`, `chunk_size=42`, `n_action_steps=32`, `fps=30`,
  `camera_layout=side_by_side`, `camera_keys=[observation.images.scene, observation.images.wrist]` (3x256x256
  each), `canvas_hw=[256, 512]`, `action_modality=action`, `action_scale=2.0`, `text_fixed_length=320`,
  `video_position_fps=24`, `sampler=euler`, `num_inference_steps=4`, `guidance_scale=3.0`,
  `guidance_scale_action=null` (so 3.0 on actions too), `sampler_shift=6.93`, `inference_seed=42`,
  `dtype=bfloat16`, `attn_mode=torch`, `compile_model=false`. `ws4/flux/contract.py:SO101_CONTRACT` equals it
  (test `test_real_so101_package_contract_and_quantiles`).
- Processor JSONs: identical to the fixture (rename, to_batch, device cuda, history normalizer, action target
  normalizer, camera resize / history unnormalizer, device cpu).
- **Quantiles** (all three state files agree, and equal `dataset_statistics.json`):

  | channel | state q01 | state q99 | action q01 | action q99 |
  |---|---|---|---|---|
  | shoulder_pan | -92.021 | 54.141 | -3.604 | 3.340 |
  | shoulder_lift | 84.814 | 184.219 | -2.637 | 4.131 |
  | elbow_flex | 61.436 | 178.857 | -3.428 | 4.131 |
  | wrist_flex | -23.994 | 80.596 | -5.098 | 5.449 |
  | wrist_roll | -124.541 | 60.205 | -3.252 | 3.164 |
  | gripper | 0.460 | 53.575 | -0.096 | 54.163 |

  Joint actions are per-tick command deltas in degrees; the gripper action is absolute percent. `clip` 6.0.
- **These are one community dataset's bounds**: exactly the `VoicAndrei/so100_kitchen` part of BFL's SO-101
  corpus (flux-action@830f1ce:`configs/so101/community_corpus.json:316`, MolmoAct origin, 16,366 clips). Its
  joints are in the pre-2025 LeRobot SO-100 degree convention (lift 85..184, elbow 61..179; ws4/data
  `contracts.py` calls it `degrees_legacy`), not today's degrees mode with zero at mid-range.
- Header: 451 tensors, 6.947 B parameters, bf16, keys prefixed `dit.`; hidden size 3072; 5 content mode blocks
  per stream, 28 single blocks, 5 text mode blocks; streams `video`, `video_cond`, `action`, `action_cond` (no
  image/audio). Heads: `emb_in.action` [3072, 6], `emb_in.action_cond` [3072, 12] (state + past command),
  `final_layer.action.linear` [6, 3072], `final_layer.action_cond.linear` [12, 3072].

## flux-3-action-base

- `config.json`: `type=flux3_action_base`, 457 tensors, bfloat16, streams video / text / action; weights
  `flux-3-action-base.safetensors`, `video_vae.safetensors`, `text_encoder/`.
- Header: 457 tensors, 6.986 B parameters, keys **without** the `dit.` prefix (the trunk format
  `load_action_checkpoint` expects). Shared action branch named `action_prediction` / `action_prediction_cond`
  (content mode blocks and stream modulations), remapped to the configured modality by
  lerobot `f3/wiring.py:131-156`.
- **EE50 heads**: `emb_in.action_prediction_ee50.weight` [3072, 50],
  `emb_in.action_prediction_ee50_cond.weight` [3072, 50], `final_layer.action_prediction_ee50.linear.weight`
  [50, 3072], `final_layer.action_prediction_ee50_cond.linear.weight` [50, 3072] (plus their
  `adaLN_modulation.1.weight` [6144, 3072]). The conditioning width equals the action width, so the EE50
  midtraining conditioned on state only, without past commands (`conditioning_channels = action_dim x 2` only
  with past-action conditioning, lerobot `f3/packing.py:603-604`).
- **Gaming heads**: `action_prediction_gaming` and `_cond`, width 64.
- **No 14-D joint head** (and no other action head) ships in the base. The safetensors metadata says why:
  `{'step': '724000', 'note': 'FLUX Action DiT base for embodiment finetuning: video + text + shared action
  streams, ee50 and gaming action heads. Removed: image and audio streams, joint14 heads, repa projectors. BF16.',
  'model': 'flux-action-dit', 'ema': 'classic, decay 0.999'}`. So the midtraining had joint14 heads (the 6 + 1
  per arm joint space on the BFL model page); they were removed from the release. The trunk is the step-724k EMA
  (matches "Action-pretrained 724k EMA base" in `fa@830f1ce:docs/droid-finetune.md`). The so101 and droid
  headers carry no metadata beyond `format: pt`.
- Complete list of head tensors (identical at the pinned `62878e2` and at `main`, same file hash):

  | tensor | shape |
  |---|---|
  | `emb_in.action_prediction_ee50.weight` | [3072, 50] |
  | `emb_in.action_prediction_ee50_cond.weight` | [3072, 50] |
  | `final_layer.action_prediction_ee50.linear.weight` | [50, 3072] |
  | `final_layer.action_prediction_ee50.adaLN_modulation.1.weight` | [6144, 3072] |
  | `final_layer.action_prediction_ee50_cond.linear.weight` | [50, 3072] |
  | `final_layer.action_prediction_ee50_cond.adaLN_modulation.1.weight` | [6144, 3072] |
  | `emb_in.action_prediction_gaming.weight` | [3072, 64] |
  | `emb_in.action_prediction_gaming_cond.weight` | [3072, 64] |
  | `final_layer.action_prediction_gaming.linear.weight` | [64, 3072] |
  | `final_layer.action_prediction_gaming.adaLN_modulation.1.weight` | [6144, 3072] |
  | `final_layer.action_prediction_gaming_cond.linear.weight` | [64, 3072] |
  | `final_layer.action_prediction_gaming_cond.adaLN_modulation.1.weight` | [6144, 3072] |
  | `emb_in.video.weight`, `emb_in.video_cond.weight` | [3072, 96] |
  | `final_layer.video.linear.weight`, `final_layer.video_cond.linear.weight` | [96, 3072] |
  | `final_layer.video.adaLN_modulation.1.weight`, `final_layer.video_cond.adaLN_modulation.1.weight` | [6144, 3072] |
- Reuse: an `action_modality` equal to the head name keeps it (wiring keeps heads whose name is the modality and
  drops other `action_prediction*` heads). `action_prediction_ee50` is allowed in both LeRobot and flux-action
  (flux-action only reserves the bare `action_prediction`, `config.py:177-183`). A LeRobot or flux-action config
  with `action_modality="action_prediction_ee50"`, `action_dim=50`, `condition_on_past_actions=false` loads these
  heads instead of fresh ones. Still missing for EE50: the data representation (anchor-relative poses),
  per-offset z-score statistics and the frame conventions (ws4/kinematics `ee50.py` lists them as UNVERIFIED).

## flux-3-action-droid (for comparison)

- Header: 451 tensors, 6.947 B parameters, `dit.` prefix; heads `action_prediction_droid` [3072, 8] and
  `_cond` [3072, 8] (state only, frame conditioning); same trunk shape as so101.
