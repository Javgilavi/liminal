# FLUX 3 Action in the Haller sim

Type an instruction in a browser and watch one simulated SO-101 try it with
[FLUX 3 Action SO-101](https://huggingface.co/black-forest-labs/flux-3-action-so101), zero-shot.

For the existing A100 Pod and this laptop, see [RUNPOD_FLUX.md](RUNPOD_FLUX.md)
for the configured SSH tunnel, launch helpers and integration smoke test.

```
laptop                                            GPU box (>= 40 GB, e.g. L40S / A100 / H100)
  app.py  (MuJoCo, cameras, browser UI)  --HTTP-->  server.py  (LeRobot Flux3Policy, BF16 ~32 GB)
  every 30 Hz tick: state + D455 + wrist JPEG          returns one absolute command
```

## 1. GPU box

Once, in the browser, accept the FLUX Kommunity License on both
[flux-3-action-so101](https://huggingface.co/black-forest-labs/flux-3-action-so101) and
[flux-3-action-base](https://huggingface.co/black-forest-labs/flux-3-action-base). Then, on the box (Linux, Python 3.12,
CUDA 12.8 driver), follow BFL's SO-101 guide:

```bash
git clone https://github.com/huggingface/lerobot.git && cd lerobot
uv venv --python 3.12 && source .venv/bin/activate
uv pip install 'torch==2.11.0' 'torchvision==0.26.0' --index-url https://download.pytorch.org/whl/cu128
uv pip install -e '.[flux3]'
uv pip install 'natten==0.21.6+torch2110cu128' --find-links https://whl.natten.org/
uv pip install fastapi uvicorn pillow
hf auth login

# copy server.py from this folder, then:
python server.py --port 8100     # first start downloads ~25 GB (policy 13.9 GB + text encoder + video VAE)
```

The log prints the checkpoint's cameras, fps, chunk size and history once it has loaded.

## 1b. Or on this laptop (8 GB RTX 5070), slowly

`server.py --offload` keeps the 7B transformer in pinned RAM and copies each of its blocks to the GPU only while
it runs. The text encoder streams the same way, once per new prompt, and the video VAE stays on the GPU. That's
~1.2 s of weight copying per transformer pass at the measured 11.6 GB/s, so expect several seconds per 32-action
chunk. It needs ~23 GB of RAM (transformer 13.9 + text encoder 8.9), so close big apps first. Swap covers the text
encoder.

```bash
cd ~/erh_hack/dual_so101_sim/flux
lerobot/.venv/bin/python server.py --offload --port 8100      # env from BFL's guide + the natten wheel
# second terminal:
MUJOCO_GL=egl ~/erh_hack/dual_so101_sim/.venv/bin/python app.py --policy http://localhost:8100
```

The video VAE needs NATTEN: `uv pip install --python lerobot/.venv/bin/python 'natten==0.21.6+torch2110cu128'
--find-links https://whl.natten.org/`.

## 2. Laptop

Tunnel the port instead of exposing it, then start the UI:

```bash
ssh -N -L 8100:localhost:8100 user@GPU_BOX          # keep this running
cd ~/erh_hack/dual_so101_sim/flux
MUJOCO_GL=egl ~/erh_hack/dual_so101_sim/.venv/bin/python app.py --policy http://localhost:8100
```

Open http://localhost:8000. Pick the arm, the scene camera and the episode length, type an instruction (or click one of
the examples), then press **Run**. **Shuffle cubes** re-deals them with ±5 cm jitter.

Testing without a GPU:
- `app.py --policy demo` runs a scripted wave, which checks the sim, cameras and UI.
- `server.py --fake` on the laptop, with `app.py --policy http://localhost:8100`, checks the exact client/server
  protocol. It echoes the state back, so the arm holds still.

## What the app does each tick

1. Renders the scene camera at 640×480 and the arm's wrist camera at 640×360. The server resizes both to the
   checkpoint's input shapes (256×256) before the saved processor accumulates history. FLUX builds a 512×256
   canvas (scene left, wrist right).
2. Reads the arm state in LeRobot units: degrees for shoulder_pan … wrist_roll (zero = mid-range, same as the MJCF)
   and 0–100 for the gripper (0 = closed).
3. Sends both to the server, which runs the saved `preprocessor → select_action → postprocessor`. The processors own
   the 8-step history and integrate the checkpoint's command deltas, so every tick must go through, and `/reset`
   runs at the start of each episode.
4. Clamps each arm joint to ±6° per tick (180°/s). The model bounds nothing itself (model card). Then it applies the
   command and steps 1/30 s of physics.

Sim time is the clock, so physics pauses while the model thinks. Expect roughly 10–15 ticks per wall second over a
tunnel, plus ~1 inference per 32 ticks.

## Set your expectations

- **Zero-shot from rendered images.** The checkpoint was trained on real SO-101 videos
  (`lerobot/community_dataset_v3`). Flat-shaded MuJoCo frames are out of distribution, so random-looking motion is a
  likely result and not a bug in the harness. The fix is a LoRA on sim demos (`lora.json` in the model repo, 10k
  steps, batch 2 × accumulation 4).
- **Scene camera.** `top` is Haller's D455 at its *estimated* pose, where the folded arms hide the side cubes.
  `front` sees the whole bench and is closer to the front/top views common in community datasets.
- **Joint signs (VERIFY).** The degrees assume the TheRobotStudio "new calib" MJCF uses the same sign per joint as
  LeRobot's calibration. If an arm moves the wrong way on a joint, flip that joint in `Sim.get_arm` / `Sim.set_arm`.
- **Instructions.** The cubes are red, green and yellow, and the place zone is a blue square.
