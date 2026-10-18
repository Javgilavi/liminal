# Imagination GPU runbook (cloud box: 3x RTX PRO 6000 Blackwell, 96 GB, sm_120)

Goal: see FLUX 3 Action's imagined frames on real weights, measure what they cost, then wire them into the
live loop and the demo video. Every step writes a `report.json`. Paste the numbers into
`ws4/research/imagination.md` (section "Latency budget") and remove the `UNVERIFIED` tags you confirmed.

Imagination adds **no new GPU dependency**. The video VAE that decodes the frames is the same frozen VAE the
policy already uses to encode its camera inputs, so if the policy runs, decoding runs. Everything below uses
the **LeRobot env** that `ws4/cloud/bootstrap.sh` builds; the flux-action env is not needed here.

Suggested GPU split (agree with the team first): `cuda:0` robot policy (control), `cuda:1` decode, judge-side
work and evaluation, `cuda:2` training (LoRA, embodiment fine-tunes).

## 0. Starting point: after `ws4/cloud/bootstrap.sh` finished

Bootstrap has done the following:
- Built `$WS4_ROOT/lerobot/.venv`: LeRobot pinned at `146d847` with the training/flux3/peft extras (training
  includes the dataset extra), the matching NATTEN wheel, and `pytest` and `anthropic` for this runbook.
- Prefetched `flux-3-action-so101` and `flux-3-action-base` (VAE and Qwen text encoder) into the HF cache.
- Checked that `$HOME/liminal/ws4` exists.
- Written GPU checks to `$WS4_ROOT/bootstrap.log`.

Check that log first:
- The NATTEN 3D op must report ok on the GPU.
- `sm_120 in torch arch list` must be True.

If `ws4/` is missing on the box, run this from the laptop:

```bash
bash ws4/cloud/push_ws4.sh <ssh-host>
```

It rsyncs the laptop's `ws4/` to `~/liminal/ws4`; the imagination code needs nothing else from the repo.
Re-run it after any change to `ws4/imagine`.

In every new shell on the box:

```bash
export WS4_ROOT=${WS4_ROOT:-$HOME/ws4} LR_PY=$WS4_ROOT/lerobot/.venv/bin/python LIMINAL=$HOME/liminal
```

Run every command below from the checkout, so `python -m ws4.imagine...` resolves:

```bash
cd $LIMINAL
```

Keep the token bootstrap used, because LeRobot checks the gated repos' revisions even when the weights are
cached:

```bash
export HF_TOKEN=hf_...
```

```bash
export ANTHROPIC_API_KEY=...    # only for the Claude judge (team's USD 200 budget); never commit it
```

MP4 writing uses imageio-ffmpeg if present, otherwise OpenCV, which LeRobot already brings.

## 1. Plumbing check against the box's LeRobot code (2 min, CPU is enough)

```bash
$LR_PY -m pytest ws4/imagine/tests -q
```

Expect every test to pass and **none skipped**. `test_imagine_lerobot_real.py` builds the real `Flux3Policy`
of the installed LeRobot with a tiny random DiT. It checks that imagination leaves actions bit-identical,
returns 11 latent frames and 41 decoded frames, and that best-of-N executes exactly the chosen seed.

Bootstrap pins LeRobot to `146d847`, the commit these tests and the `_sample` hook were verified against. If
someone overrides `LEROBOT_REF`, this step catches the drift. Confirm the installed commit:

```bash
git -C $WS4_ROOT/lerobot log -1 --format='%h %cd'
```

## 2. A dataset to replay

The weights are already cached; the first policy load in a process takes a minute to read about 24 GiB
(SO-101 DiT 12.9 GiB, Qwen3-VL text encoder 8.3 GiB, video VAE 2.3 GiB). Replay data, best first:

1. The team's own SO-101 recordings (WS1). Use their camera keys below.
2. A training-distribution episode from `lerobot/community_dataset_v3`. In the current revision its scene
   camera is `top` and its wrist camera is `wrist`. Camera names in that collection are unreliable, so look
   at the GIF.

```bash
$WS4_ROOT/lerobot/.venv/bin/hf download lerobot/community_dataset_v3 --repo-type dataset --include "00ri/so100_battery/*" --local-dir data/cd3
```

The dataset root is then `data/cd3/00ri/so100_battery`.

## 3. Does the decoded timing match the code? (5 min)

`predict_with_video.VideoSpec.frame_offsets` derives, from the VAE's merge/expand code, that the SO-101 future
clip (42 frames) decodes to 41 frames showing control steps +2 .. +42. Check that on the real VAE:

```bash
$LR_PY -m ws4.imagine.run_imagine vaecheck --dataset data/cd3/00ri/so100_battery --episode 0 --start 100 --scene-key observation.images.top --wrist-key observation.images.wrist --device cuda:1 --out outputs/imagine/vaecheck
```

Read two values in `report.json`:
- `agreement`: the fraction of decoded frames whose nearest recorded frame is the predicted one. Expect close
  to 1.0 on a moving scene, lower on a static one.
- `psnr_db`: reconstruction quality.

Also note the log line `natten backend (3D tokens): ...`. On sm_120, expect `cutlass-fna`, because the
`blackwell-fna` kernels are SM100 only.

## 4. What does imagination cost? (10 min)

```bash
$LR_PY -m ws4.imagine.run_imagine bench --device cuda:0 --decode-device cuda:1 --out outputs/imagine/bench
```

Read in `report.json`:

| key | meaning | what to hope for |
|---|---|---|
| `stock_predict.median_s` | one chunk, stock policy | the control pause per chunk; compare with 1.07 s of motion |
| `imagine_no_decode` | chunk + latent capture | equal to stock (capture is a copy of 540 KB) |
| `imagine_sync_same_gpu` | chunk + decode on cuda:0 | stock + decode time |
| `decode_only_cuda:1` | decode 41 frames on cuda:1 | well under 1 s, so async decode fits the execution window |
| `imagine_async_cuda:1` | chunk, decode in background | equal to stock |
| `best_of_2/4_with_decode` | N seeds + N decodes | about N x stock |
| `peak_memory_gib` | per device | decode peak about 5 GB, est. from the VAE's own calibration constants |

If decode is slow, compare `F3_NATTEN_BACKEND=flex-fna` with `cutlass-fna` and keep the faster one. If memory
is tight (24 to 32 GB cards), add `--text-encoder-on-cpu` to `bench` / `replay`. It moves Qwen3-VL to host RAM
after loading and frees about 8.3 GiB; in your own code, call `policy.frozen.text_encoder.cpu()` instead.

## 5. Imagined vs recorded future (10 min, the first demo-video material)

```bash
$LR_PY -m ws4.imagine.run_imagine replay --dataset data/cd3/00ri/so100_battery --episode 0 --start 40 --chunks 6 --scene-key observation.images.top --wrist-key observation.images.wrist --decode-device cuda:1 --out outputs/imagine/replay_ep0
```

Outputs:
- `side_by_side.mp4` and `.gif`: ACTUAL | IMAGINED with chunk boundaries and a surprise trace.
- `chunk_XX_tYYY.png` contact sheets: the imagined row over the recorded row at +0.13 .. +1.4 s.
- `chunk_*.npz`: frames and latents.
- `report.json`, with `surprise_by_offset`.

This is open loop: the recorded future follows the demonstrator, so divergence late in the chunk is expected.
Surprise that rises with offset is the healthy pattern. A flat, high value means the frames are wrong; check
the camera keys and their order first.

## 6. Live on the robot

Wherever the team's loop creates the policy (ws4-flux-dual runner, LeRobot record/eval loop or WS5 UI), add:

```python
from ws4.imagine.compose import ActualFrame, Renderer, align, make_canvas, render_run
from ws4.imagine.predict_with_video import Imaginer

imaginer = Imaginer(policy, decode="async", on_frames=ui_push).install()  # same GPU is fine: it is idle while the arm moves
actual_log = []
# control loop, unchanged:
#   batch = preprocessor(obs); action = policy.select_action(batch); robot.send(postprocessor(action))
# then one line per tick:
#   actual_log.append(ActualFrame(make_canvas(scene_rgb, wrist_rgb, policy.config.canvas_hw), step=imaginer.last_step))
```

`ui_push(record)` receives each `Imagination` once its frames are decoded (`record.frames`,
`record.frame_offsets`, `record.step`). For the panel, `Renderer().render(row)` returns a ready image, and
`align(imaginer.history, actual_log)` pairs frames by control step. After the run:

```python
render_run([r.wait() for r in imaginer.history], actual_log, "outputs/demo/imagined_vs_actual.mp4", task=task)
```

If the runner calls `predict_action_chunk` itself (one model per arm, with its own queues), `install()` still
works. The records then carry no step, and `align` falls back to wall-clock time from `t_return`. At an
episode boundary, call `imaginer.new_episode()` together with the processors' reset. To force a replan,
call `imaginer.replan()`.

## 7. Decision points: imagine alternatives, let Claude pick (optional, after 6 works)

```python
from ws4.imagine.predict_with_video import Imaginer, alternatives_batch
from ws4.imagine.score import ClaudeJudge

judge = ClaudeJudge()  # claude-opus-5-5, effort low, server-side refusal fallback on
options = ["push the green box to the left", "lift the green box out of the way"]
recs = Imaginer(policy).imagine(alternatives_batch(preprocessed_batch, options))
result = judge.rank([r.frames for r in recs], current=recs[0].current_frame(), offsets=recs[0].frame_offsets,
                    task="clear the path to the red cube")
next_instruction = options[result.best]
```

The robot pauses while this runs (N chunks plus one API call, a few seconds), so show the candidate grid on
screen while it decides. For seed-level best-of-N inside `select_action`, use
`Imaginer(policy, n_candidates=3, scorer=judge)`.

## Troubleshooting

- `RuntimeError: solver output did not contain x_video`: the installed LeRobot changed `_sample`. Step 1 fails
  first. Re-run bootstrap with the default `LEROBOT_REF=146d847`, or tell ws4-imagine.
- `ImportError ... NATTEN`: the wheel does not match torch/CUDA. Re-run `bootstrap.sh` (it reinstalls NATTEN
  after every sync) and read its GPU-check section.
- Frames look like the wrong camera: check `--scene-key` and `--wrist-key`. The canvas is scene left, wrist
  right.
- `decode_skipped` in `record.timings`: the async queue (4 chunks) was full, meaning decoding is slower than
  control.
- The judge returns `[1, 0, 0]` and `judge.last_error` is set: an API error or a refusal; the policy's own
  seed runs.
