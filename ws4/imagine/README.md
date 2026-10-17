# ws4/imagine: FLUX 3 Action's imagination on the real-arm path

FLUX 3 Action denoises the next video together with every action chunk, but LeRobot and flux-action throw the
video away. This package keeps it, decodes it with the model's own frozen VAE and puts it next to what the
cameras really saw: **ACTUAL | IMAGINED**. Research, evidence and design: `ws4/research/imagination.md`.

Status (2026-10-17, 22:15 CEST):
- **Measured on real weights:** held-out Haller pen takes 101, 102, 103 (layout D, never trained on), served by
  `so101_haller_pen_B84` at guidance 1.0 on `real-serve-6000` (RTX PRO 6000 Blackwell):
  - **Bit-identical commands:** every one of the 23 chunks matched a plain request exactly (max abs diff 0.0;
    `compile_model` off).
  - **Compile on as well** (`/workspace/imagine/compile_check/`): lora_500, guidance 1.0, `compile_model` on,
    text encoder on the CPU, 5 synthetic requests, one fresh process per code copy. Plain requests are identical
    across the current code, the serve copy and this code (max 0.0). Within a process, plain, lazy imagination,
    plain-after-imagination and a forced recompile are all identical. The only difference anywhere is compile on
    vs off (max 0.70), which no imagination setting touches.
  - **No added round trip in `lazy` mode:** median 0.39 s with imagination vs 0.39 to 0.43 s plain, same pod.
  - **Decode cost:** 0.11 to 0.12 s for 41 frames at 256x512, run after the take while nothing is waiting.
- **Verified on CPU, against real code:** a test through the real `LeRobotFlux3Backend` (real LeRobot processors)
  behind the real `flux.remote` server; every mode returns bit-identical commands and the decoded frames.
- **End to end:** the patched runner on a fake robot, against a live HTTP server, writes the attempt folder below
  and the video.

## First clips (real data, held-out takes)

On this laptop, gitignored (`ws4/imagine/outputs/`). On the pod: `/workspace/imagine/replay_ep10{1,2,3}`.

| File | What it shows |
|---|---|
| `outputs/replay_ep101/imagination/imagined_vs_actual.mp4` | take 101, 10 s: ACTUAL \| IMAGINED, chunk boundaries, surprise trace |
| `outputs/replay_ep101/imagination/imagined_vs_actual_first3.gif` | the first 3 chunks as a GIF (14 MB) |
| `outputs/replay_ep101/contact_chunk1.png`, `contact_chunk4.png` | slide material: imagined row over actual row at +0.13 to +1.4 s |
| `outputs/replay_ep10{2,3}/imagination/imagined_vs_actual.mp4` | takes 102 and 103 |
| `outputs/lora500_ep10{1,2,3}/` | the same takes through the 500-step test LoRA |
| `outputs/offset_errors_*.json` | the accuracy numbers below |

How good is the imagination? Error is `score.frame_distance` (mean absolute RGB difference at 64x64, 0 =
identical), averaged over the 3 takes. The baseline is the VAE's own reconstruction of the current frame, held
still ("nothing moves", same blur as the imagination).

| Control offset | Scene camera, imagined vs held | Wrist camera, imagined vs held |
|---|---|---|
| +2 .. +8 (0.07 to 0.27 s) | **0.019** vs 0.021 | **0.064** vs 0.072 |
| +9 .. +20 (0.3 to 0.67 s) | **0.025** vs 0.028 | **0.139** vs 0.147 |
| +21 .. +32 (0.7 to 1.07 s) | **0.032** vs 0.035 | **0.207** vs 0.211 |
| +33 .. +42 (beyond the replan) | **0.036** vs 0.039 | **0.249** vs 0.256 |

- On the scene camera, the imagination beats "nothing moves" on 80 % of (chunk, offset) pairs: it predicts where
  the arm goes.
- On the fast-moving wrist view it wins only 43 % of pairs: the frames blur within about 0.5 s.
- Short horizons (under about 0.4 s) look near-identical to reality.
- **Fine-tuning sharpens the imagination.** The overnight pen LoRA at 3000 steps lowers the wrist-camera error
  by 11 % at +2..+8 (0.064 to 0.057), and by 9, 6 and 5 % in the later bands. The scene camera improves by about
  2 %. The 500-step test LoRA (28 takes) is indistinguishable from the base. The LoRA trains the video stream on
  our own scene too. Numbers: `outputs/offset_errors_b84_lora500_lora3000.json`.
- With the 3000-step LoRA, commands are again identical to plain requests on all 23 chunks, with the same round
  trip (0.38 to 0.39 s) and decode time (0.12 s); clips in `outputs/lora3000_ep10{1,2,3}/` (H.264).

## What runs where

| Piece | File | Who calls it |
|---|---|---|
| Capture + decode (policy side) | `predict_with_video.py` (`Imaginer`, `LatentDecoder`) | the server, offline tools |
| Server service | `serving.py` (`ImagineService`) | `flux.backends.LeRobotFlux3Backend` when a request asks for it |
| Transport | `ws4/flux/remote.py`: request option `imagine`, `GET /imagination/<package>/<id>` | `RemoteBackend(..., imagine=...)` |
| Robot-side recorder | `attempt.py` (`AttemptImagination`) | `run_single_arm.py --imagine` |
| Side-by-side video | `compose.py`, `python -m imagine.attempt <attempt_dir>` | after an attempt |
| Replay without the robot | `replay_server.py` | on the pod, tonight |

## Modes (per request, chosen by the client)

| `imagine` | What the server does | Added time on the control path |
|---|---|---|
| `""` (default) | nothing new; request and response bytes unchanged, `imagine/` not even imported | 0 |
| `lazy` | keeps the 11 latent frames (about 0.5 MB); decodes when the client fetches them | about 0 (one 540 KB device-to-host copy) |
| `async` | answers at once, decodes on a server thread while the model is idle | about 0, but a decode can delay the *next* request by up to one decode time |
| `sync` | decodes before answering; the frames travel in the response | one decode plus about 1.2 MB of JPEG per chunk |

The solver hook that captures the video stream is installed only while a capture is in progress and restored
right after, so the sampling code is untouched whenever no imagination is in flight.

**Use `lazy` on the robot.** The runner fetches every chunk's frames after the attempt, while the arm is
parked, so control is never slowed. Decoding always takes the model's lock and never overlaps a prediction.

## On the GPU box: nothing to configure

Any server built on `flux.remote.make_server` with `LeRobotFlux3Backend` serves imagination as soon as the
updated `ws4/` is on the box. That covers `python -m flux.remote ...`, `ws4/cloud/runpod_serve.sh` and
`ws4/single_arm/serve_lowmem.py`. Sync the code and restart the server:

```bash
bash ws4/cloud/push_ws4.sh <ssh-host>
```

VRAM: decoding needs about 5 GB more at peak (estimate from the VAE's own constants, `research/imagination.md`
1.4). If free memory is short, the VAE falls back to chunked decoding by itself. On a 24 GB card, use
`serve_lowmem.py` as before (Qwen on the CPU). `ws4/cloud/lora_adapter.py serve` (LoRA folded in memory)
also goes through `LeRobotFlux3Backend` and `make_server`, so it serves imagination too, but only once the
copy of `ws4/` it runs from includes the updated `flux/` and `imagine/`.

## On the robot computer

`run_single_arm.py` has the opt-in hook (approved by so101-zeroshot, 2026-10-17 22:30 CEST). Add the flag to the
usual remote command:

```bash
python ws4/single_arm/run_single_arm.py --backend remote ... --imagine lazy
```

Flags:
- `--imagine {lazy,async,sync}`.
- `--imagine-timeout` (default 30 s): the hard cap on fetching after an attempt.
- `--imagine-render`: write the video right away.

Without `--imagine` nothing changes: no import of ws4/imagine, and `RemoteBackend` is called exactly as before.
The graded B84 and LoRA eval rows run without it, and `eval_pen.sh` never passes it.

Guarantees:
- **In the 30 Hz loop:** `irec.tick` only queues the frames. The queue is bounded (3 s of frames) and drops
  frames, counted, when the encoder falls behind. A failure disables recording for the attempt with one printed
  line; it never raises.
- **After `meta.json` is written, before the scoring prompt:** `finish_with_timeout` fetches the imagined chunks
  on a thread. It returns within `--imagine-timeout` whatever the server does. It stops after the first server
  error, and chunks it did not get are listed as pending in `imagination.json`. The trace, video, meta, scoring
  prompt and log line always happen.
- **Fetching later:** the server keeps its last 256 chunks (about 8 attempts). Pending chunks can be fetched
  afterwards:

```bash
cd ws4 && python -m imagine.attempt fetch single_arm/runs/<session>/attempt01 --server http://127.0.0.1:8765 --package so101
```

How long the arm holds after an attempt (estimate):
- Per chunk: 0.85 MB of JPEG at about 5 MB/s over the laptop-to-pod SSH link (measured: 0.17 s), plus the server
  decode (measured: 0.11 s), plus request overhead (about 0.2 s). That is about 0.4 to 0.6 s per chunk.
- A 40 s attempt yields about 20 chunks, so about 8 to 12 s.
- Not yet measured end to end through the tunnel.

Each attempt folder then gets `imagination/`:
- `actual.npz`: every tick's scene | wrist canvas as the model sees it (JPEG), with tick indices and times.
- `chunk_XXX.npz`: per chunk, 41 imagined frames at control offsets +2 .. +42 and the tick where the chunk
  started.
- `imagination.json`: round trips, server predict and decode times, dropped frames, pending chunks.
- `imagined_vs_actual.mp4` with `--imagine-render`. Rendering takes about 20 to 40 s per minute of attempt on
  the laptop CPU. Skip the flag and render later instead:

```bash
cd ws4 && python -m imagine.attempt single_arm/runs/<session>/attempt01 --gif
```

Alignment: the runner requests a chunk at tick k, holds the arm while waiting, and starts executing at the
`replanned` tick r. Imagined offset o is drawn next to actual tick r + o. Ticks spent waiting for the next chunk
still show the previous imagination (offsets 33 to 42), labelled with their offset.

**Server requirement:** the policy server must run the updated `ws4/flux` (remote.py, backends.py) and
`ws4/imagine`. The arm's pen server on 8790 (`cloud/lora_adapter.py serve`) currently runs the training copy
of `ws4/`, so it needs a re-sync and restart before `--imagine` works against it.

## Replay test (no robot)

On the pod, with a server running on PORT (any of the servers above) and a per-arm pen dataset (tonight:
`/workspace/real_lora/pen_in_right_final`; held-out takes 101 to 103, then 20 to 22, 36 to 38 and 86 to 88).
`/workspace/imagine/run_replays.sh` loops over episodes, reading `PORT`, `PREFIX` and `EXTRA` from the
environment:

```bash
cd $LIMINAL/ws4 && python -m imagine.replay_server --url http://127.0.0.1:PORT --package so101 --token-file /workspace/flux_token --root <per-arm dataset root> --repo-id <name> --episode 0 --task "put the red pen in its slot in the cardboard box" --chunks 8 --gif --out /workspace/imagine/replay_ep0
```

It sends every chunk twice, plain and with `imagine=lazy`. It reports both round trips, whether the commands
are identical, and the decode times. Then it writes the same folder layout as a robot attempt, plus
`report.json`. Copy the clip back:

```bash
rsync -az -e "ssh -p <port> -i ~/.ssh/id_ed25519_osrdvs" root@<pod-ip>:/workspace/imagine/replay_ep0/ outputs/imagine/replay_ep0/
```

## Tests

```bash
python -m pytest ws4/imagine/tests -q
```

The real-LeRobot file needs LeRobot `146d847` (the cloud env, or a CPU venv). Elsewhere it skips.
