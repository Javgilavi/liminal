# FLUX 3 Action on Runpod, MuJoCo on this laptop

Real FLUX inference verified, 2026-10-17, on existing Pod `2uypll8r50gqc9`,
NVIDIA A100-SXM4-80GB. No new Pod or volume was created.

Latest state: at the user's request, the local simulator/UI, SSH tunnel and cloud
FLUX server were stopped. The Runpod Pod remains running. Restart manually using
the commands below; allow model loading to finish before launching the UI.
`runpod.sh start` explicitly selects CUDA on the remote Pod. `runpod.sh ui`
requires the managed SSH tunnel and verifies a real FLUX/CUDA health response;
it never loads a local VLA or falls back to a demo/CPU backend.

The browser's task prompt goes directly to FLUX for one selected arm. A separate
frontier planner is optional for this test; Claude Haiku can later break a longer
goal into these same bounded instructions through the Liminal harness.

## Files and environment

- Local application: `dual_so101_sim/flux/app.py` using
  `dual_so101_sim/.venv/bin/python` and `MUJOCO_GL=egl`.
- Remote source/logs/model cache: `/workspace/flux3-action/`.
- Remote environment: `/opt/flux3-action/.venv` with Python 3.12, torch 2.11 CUDA
  12.8 and matching NATTEN. The template's system torch 2.8 is preserved.
- Fast inference cache: `/opt/flux3-action/hf-cache/`; `stage_models.py` copies
  the mounted download cache here. `runpod.sh start` uses it after the
  `.complete` marker exists. Both copies take approximately 24 GB each.
- LeRobot source snapshot: `146d8474932c8b566717eb2738ba459337ab71c4`.
- SO-101 policy: `black-forest-labs/flux-3-action-so101` revision
  `c9e13b2aca6a0a472b3ea03fd90cff32d3e85849`.
- Frozen encoders: `black-forest-labs/flux-3-action-base` revision
  `62878e2925e59b7a89ec14463ce89932624c490d`.

The mounted global volume rejects ownership/mode/timestamp restoration. Source
was transferred with tar's metadata restoration disabled; executable environments
and package caches use the container disk. Reading large safetensors through this
mount also delayed the first startup, so inference uses a staged local copy while
retaining the mounted download cache. If `/opt` is lost when the Pod is recreated,
rerun dependency setup and staging. `python3 runpod_bootstrap.py` starts setup on
this already-configured Pod; follow `/workspace/flux3-action/setup.log`.

`runpod_setup.sh`, `download_models.py`, `stage_models.py` and `server.py` have
already been copied to the Pod. Setup records the installed versions in
`/workspace/flux3-action/runtime-packages.txt`.

The existing local Hugging Face credential was checked for both gated model
repositories. Setup passes it through encrypted SSH to an ephemeral process
environment; it is not printed, written into these scripts, or persisted with
`hf auth login` on the Pod. Cached inference runs offline without the token.

## Repeat the test

From a normal local terminal:

```bash
cd ~/erh_hack/dual_so101_sim/flux
bash runpod.sh start       # skip when the model server is already running
bash runpod.sh logs        # loading progress
bash runpod.sh status      # must report ok=true and fake=false
bash runpod.sh tunnel
bash runpod.sh ui
```

If your terminal environment reaps background SSH processes, use
`bash runpod.sh tunnel-foreground` in a separate terminal and leave it open.

Open **http://127.0.0.1:8000**. Select **right** and initially use **front** as the
scene camera so the cubes are visible. Start with **1–2 simulated seconds** and a bounded
instruction such as “Pick up the green cube and place it on the blue square.”
Only the selected arm receives policy targets. Switching arms starts a new
episode and resets policy/processor history; this server is a single session,
not concurrent learned bimanual control.

This is the same local launch as before, changing only the policy URL:

```bash
MUJOCO_GL=egl ../.venv/bin/python app.py --policy http://127.0.0.1:18100
```

Port `18100` is the local SSH tunnel to the Pod's loopback port `8100`. Local
port `8100` was already occupied and was preserved. The inference service is
bound to loopback; no unauthenticated public inference port is opened. Direct
SSH uses the user's existing key and a Pod host key obtained over the already
trusted Runpod proxy connection. Local connection metadata lives in `.runpod/`.

To close this connection/server without affecting other services:

```bash
bash runpod.sh stop-tunnel
bash runpod.sh stop-server
```

Stopping the model process does not stop Runpod Pod billing. This Pod existed
before setup and is left running for the user's tests.

## Verification and limits

With the UI idle, reproduce the integration smoke test:

```bash
MUJOCO_GL=egl ~/erh_hack/dual_so101_sim/.venv/bin/python smoke_remote.py \
  --arm right --scene-cam front --ticks 64 --output .runpod/real-smoke
```

This saves before/after images, every predicted/applied action and `result.json`.
It refuses a fake server by default. Sixty-four ticks exercise two 32-action
chunks, including history across the chunk boundary. Do not run the smoke client
and a UI episode concurrently: they share one stateful policy session.

The first real request found that the saved camera resize step applies only to
grid layouts, whereas this checkpoint uses `side_by_side`. The server now resizes
each camera to its checkpoint feature shape (`256×256`) before the saved
processor accumulates history. MuJoCo's scene (`640×480`) and wrist (`640×360`)
images otherwise fail the policy's equal-shape check.

- **Real GPU model loaded:** CUDA, `fake=false`, approximately 24.9 GiB GPU
  memory. Startup from the staged cache took about 122 seconds.
- **64-tick real MuJoCo smoke passed:** 64 finite six-dimensional control targets
  sent to the right arm, zero targets sent to the left arm. Physics can still
  settle the uncommanded arm; this does not mean both arms received VLA actions.
- **Chunk boundary exercised:** inference at ticks 0 and 32 took 1,888 ms and
  1,077 ms. Saved history/processor state continued through the second chunk.
- **Full wall time:** 112.983 seconds for 2.13 simulated seconds. Chunk inference
  timing excludes most per-tick HTTP, image processing and rendering overhead;
  this path is currently much slower than real time.
- **Artifacts:** `.runpod/real-smoke/{result.json,actions.jsonl,before.jpg,after.jpg}`.
  The earlier `.runpod/protocol-smoke/` used a fake server and is labeled as such.
- **Browser Run path passed:** `/run` completed 30 real-policy ticks (one
  simulated second) with `state=done`, `demo=false`, `error=null`; wall time
  64.085 seconds and chunk inference 1,082 ms. Evidence is in
  `.runpod/ui-smoke/{result.json,before.jpg,after.jpg}`. The scene was reset
  afterward. These services were initially left running, then stopped at the
  user's subsequent request (see the current state at the top).
- **Local checks:** EGL rendering, Python syntax, shell syntax and Ruff passed.

A successful inference/action loop does not by itself establish pick-and-place
success. The frames show arm motion; the short test did not demonstrate a
completed placement. The checkpoint was trained on real-camera data; rendered
scenes, camera pose and robot calibration differ. Task verification, calibration
and domain adaptation remain separate work. No real robot or Claude API was used.

Official integration guide: https://docs.bfl.ai/flux_3/flux3_action_so101
