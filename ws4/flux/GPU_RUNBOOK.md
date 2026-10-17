# GPU runbook: FLUX 3 Action for the XLeRobot on 3x RTX PRO 6000 Blackwell

Companion to `ws4/research/flux_dual_arm.md` (the decision memo). Every step says what it proves. Anything
marked UNVERIFIED has not run on this hardware yet; record the numbers you measure in the memo.

**Start after `ws4/cloud/bootstrap.sh` has finished** (orchestrator-owned; do not repeat its steps). It builds
`$WS4_ROOT/lerobot` (LeRobot main + `training,flux3,peft,diffusion`, NATTEN matched to its torch) and
`$WS4_ROOT/flux-action` (+ `encoders,data,serve`, NATTEN), prefetches `flux-3-action-so101` and
`flux-3-action-base` into the HF cache with `HF_TOKEN`, and runs GPU and NATTEN checks. Weights are used by Hub
id from that cache below; no `--local-dir` copies.

The `ws4/flux` code was tested against LeRobot `146d8474932c8b566717eb2738ba459337ab71c4` (Oct 16 2026) and
flux-action `830f1ce178d7aea52f18fde31d8e8e36ce6bae21`; bootstrap pulls `main`, so note the commits it logs.

## 0. Session setup on the box (every shell)

```bash
export WS4_ROOT=${WS4_ROOT:-$HOME/ws4}
source $WS4_ROOT/lerobot/.venv/bin/activate          # LeRobot env: everything in ws4/flux runs here
export PYTHONPATH=<path-to-liminal-checkout>/ws4    # ws4/flux imports as `flux` (copy ws4/ without ws4/.cache)
export HF_TOKEN=hf_...                                # same account bootstrap used
grep -E "lerobot @|flux-action @|torch tag|natten|sm_|FAILED|WARNING" $WS4_ROOT/bootstrap.log
python -m pytest $PYTHONPATH/flux/tests -q           # 24 CPU-level tests, incl. LeRobot flux3 integration
```

Check in the log: both envs report `sm_120`, a NATTEN version, `natten 3D op ok`, and `lerobot Flux3Policy
import ok` for the LeRobot env (that import fails in the flux-action env by design; ignore it there). The
bootstrap check lets NATTEN choose its backend, while the VAE picks one explicitly (Blackwell SM100, Hopper,
generic CUTLASS, then flex; `lerobot f3/video_vae.py:47-76`, same in flux-action `models/video_vae.py:47-71`),
so the real answer is the VAE's own log line in section 2.

## 1. What the gated packages contain (2 min, no weight download)

```bash
python -m flux.inspect_checkpoint --repo black-forest-labs/flux-3-action-so101 --quantiles > ~/so101_quantiles.json
python -m flux.inspect_checkpoint --repo black-forest-labs/flux-3-action-base \
  --file flux-3-action-base.safetensors --keys --grep 'emb_in|final_layer' > ~/base_heads.txt
```

`so101_quantiles.json` (state and command-delta q01/q99, `config.json`, `dataset_statistics.json`) answers "do
the package's quantiles cover our arms' degrees?" (section 4). `base_heads.txt` lists the EE50 / gaming head
names and widths (option B, research only). Paste both into the memo.

## 2. Smoke test, latency and VRAM (GPU 0, 15 min)

```bash
export CUDA_VISIBLE_DEVICES=0
S=black-forest-labs/flux-3-action-so101
python -m flux.measure_latency --checkpoint $S --runs 10 | tee ~/lat_default.json
python -m flux.measure_latency --checkpoint $S --runs 10 \
  --override guidance_scale=1.0 --override guidance_scale_action=1.0 | tee ~/lat_cfg1.json    # one pass per step
python -m flux.measure_latency --checkpoint $S --runs 10 --override compile_model=true | tee ~/lat_compiled.json
python -m flux.measure_latency --checkpoint $S --runs 10 --two-instances | tee ~/lat_two.json
python -m flux.measure_latency --checkpoint $S --runs 10 --text-encoder-on-cpu | tee ~/lat_qwen_cpu.json
```

Expectations (estimates, memo section A.2): 0.6 to 1.0 s per chunk with the package defaults (Euler 4 steps,
guidance 3 on video and actions = 8 DiT passes over about 2,000 tokens); roughly half with guidance 1.0; about
30 GB per instance, 8.3 GiB less with Qwen on CPU. BFL publishes 772.6 ms on "RTX 6000 Pro" for the DROID BF16
4-step guided checkpoint (https://bfl.ai/models/flux-3-action). `measure_latency` logs at INFO, so the VAE's
line `natten backend (...): <name>` appears on the first prediction; on sm_120 expect `cutlass-fna` (the fused
Blackwell kernel is SM100). If NATTEN fails at runtime: `export F3_NATTEN_BACKEND=flex-fna` (pure PyTorch, slower).

Decision rule: if default latency is above about 1.2 s, serve with guidance 1.0 only after checking on the
robot that motion quality holds (same prompt and scene both ways).

**Cross-check against BFL's own implementation** (same package, independent code path, real weights):

```bash
python -m flux.crosscheck_so101 write --out ~/obs.npz                       # or --scene/--wrist PNGs from our cameras
(cd $WS4_ROOT/flux-action && uv run --extra encoders python examples/so101/inference.py \
   --weights $S --observation ~/obs.npz --prompt "pick up the red box" --output ~/bfl.npy)
python -m flux.crosscheck_so101 predict --observation ~/obs.npz --prompt "pick up the red box" --out ~/ours.npy
python -m flux.crosscheck_so101 compare ~/bfl.npy ~/ours.npy
```

Equal to rounding if both draw the same noise; otherwise "two samples of one policy": same direction and
scale of `net_motion`. A large systematic difference means a units, sign or camera-order bug in our path.

## 3. Serve to the robot (GPU 0, all night and during the demo)

```bash
# GPU box: bind to 127.0.0.1, the SSH tunnel is the only way in
export CUDA_VISIBLE_DEVICES=0 TOKEN=$(openssl rand -hex 16)
python -m flux.remote --port 8765 --token $TOKEN \
  --package so101=black-forest-labs/flux-3-action-so101 \
  --package left=$HOME/models/so101_left_renorm --package right=$HOME/models/so101_right_renorm   # after section 4
# Robot PC (LeRobot main, numpy, Pillow)
ssh -N -L 8765:127.0.0.1:8765 user@gpu-box &
PYTHONPATH=ws4 python -m flux.run_xlerobot --left-port /dev/ttyACM0 --right-port /dev/ttyACM1 \
  --head-cam <realsense-serial> --left-wrist-cam /dev/video2 --right-wrist-cam /dev/video4 \
  --token $TOKEN --left-package left --right-package right \
  --left-task "push the red box to the left" --seconds 20 --calibration-limits
```

Packages that point at the same weights share one loaded model (one lock, sequential calls). For two arms
replanning at the same time without queueing, start a second server on GPU 1 and point one arm at it. The
runner clamps every command (per-tick step and joint limits) before sending; keep the e-stop in hand.

## 4. Per-arm normalization check and re-normalized packages (option A1, 20 min)

Record 5 to 10 short teleop episodes per arm in LeRobot degrees mode (section 5), then compare our ranges
with `~/so101_quantiles.json`:

```bash
python - <<'EOF'
import json
from pathlib import Path
from flux.package_tools import compute_quantiles, dataset_arrays, describe_stats, renormalize_package
models = Path.home() / "models"
for arm, repo in (("left", "ORG/xlerobot_left_calib"), ("right", "ORG/xlerobot_right_calib")):
    a = dataset_arrays(repo)
    stats = compute_quantiles(a["action"], a["observation.state"], a["episode_index"], absolute_dims=(-1,))
    print(arm); print(describe_stats(stats, ["pan", "lift", "elbow", "wflex", "roll", "grip"]))
    renormalize_package("black-forest-labs/flux-3-action-so101", models / f"so101_{arm}_renorm", stats)
EOF
```

`renormalize_package` resolves the Hub id through the HF cache and symlinks the 14 GB weights. Serve `so101`
(package quantiles, A0) and `left` / `right` (our quantiles, A1) side by side and compare on the same scene.
Delta quantiles from 5 to 10 episodes are noisy; once task demos exist, recompute from them.

## 5. Recording XLeRobot demos (robot PC, LeRobot main)

Per-arm datasets for option A (one leader arm is enough; cameras already named for the SO-101 package):

```bash
lerobot-record --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=xlerobot_left \
  --robot.cameras='{scene: {type: intelrealsense, serial_number_or_name: "<serial>", width: 640, height: 480, fps: 30},
                    wrist: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30}}' \
  --teleop.type=so101_leader --teleop.port=/dev/ttyACM2 --teleop.id=haller_leader \
  --dataset.repo_id=ORG/xlerobot_left_push --dataset.single_task="push the red box to the left" \
  --dataset.num_episodes=50 --dataset.fps=30 --dataset.episode_time_s=30 --dataset.reset_time_s=15 \
  --dataset.push_to_hub=true
```

Bimanual 12-D datasets for option C need two leaders (`--teleop.type=bi_so_leader`) and
`--robot.type=bi_so_follower` with `--robot.cameras='{head: {...realsense...}}'` and per-arm
`--robot.left_arm_config.cameras='{wrist: {...}}'` (keys become `head`, `left_wrist`, `right_wrist`).
Calibrate each follower once in degrees mode; both arms need the same calibration procedure because degrees
are measured from the middle of each joint's calibrated range.

## 6. Training tonight

Measure first: start each run, read the step time after about 50 microsteps in the log, then set `--steps`
so the run ends by 06:00. Checkpoints land every `save_freq` microsteps; evaluate raw and EMA.

**GPU 1: per-arm task LoRA (option A).** Pool left and right demos into one 6-D dataset or train one adapter
per arm. Base = the package (A0) or a re-normalized package (A1):

```bash
cd $WS4_ROOT/lerobot && export CUDA_VISIBLE_DEVICES=1
python -m lerobot.scripts.lerobot_train --config_path=examples/flux3/lora.json \
  --policy.path=$HOME/models/so101_left_renorm --policy.device=cuda \
  --dataset.repo_id=ORG/xlerobot_left_push --steps=6000 --save_freq=500 \
  --output_dir=outputs/xlerobot_left_lora
```

(`--policy.path=black-forest-labs/flux-3-action-so101` for A0.) The package config resolves the VAE and text
encoder from its pinned base revision in the HF cache. Serve the adapter with
`--package left=$WS4_ROOT/lerobot/outputs/xlerobot_left_lora/checkpoints/last/pretrained_model` (or
`pretrained_model_ema`); LeRobot loads the referenced base and its processors.

**GPU 2: bimanual 12-D base + LoRA (option C, stretch, only with >= 50 bimanual demos).**

```bash
export CUDA_VISIBLE_DEVICES=2
python -m flux.make_bimanual_base --dataset ORG/xlerobot_bimanual_task --output $HOME/models/xlerobot_bimanual_base \
  --source-kind so101 --device cpu
cd $WS4_ROOT/lerobot && python -m lerobot.scripts.lerobot_train \
  --config_path=$HOME/models/xlerobot_bimanual_base/lora_train.json --steps=8000 \
  --output_dir=outputs/xlerobot_bimanual_lora
```

`lora_train.json` keeps the LeRobot preset and adds a head-first schedule (trunk frozen 800 microsteps, warmup
2,400, heads warmup 800) because the 12-D heads start at zero output. Serve the result as `--package
both=...` and drive it with `bimanual_target("both")` in the runner.

**Multi-GPU LoRA (if a single run is too slow):** `accelerate launch --num_processes=2 --multi_gpu -m
lerobot.scripts.lerobot_train ...` (LeRobot `docs/source/flux3.mdx:98-108`; BFL has not validated four GPUs).

**Not tonight:** flux-action full fine-tuning (option C from the base, or EE50). It needs FSDP over PCIe with
fp32 masters, AdamW and two fp32 power-EMA copies: about 56 to 67 GB per GPU before activations with
`shard_size=3`, and on the order of 10 h per 1,000 updates at batch 64 (memo, section C.3). The flux-action env
from bootstrap is ready for a validation run after the hackathon:

```bash
cd $WS4_ROOT/flux-action
uv run hf download black-forest-labs/flux-3-action-base --include 'flux-3-action-base.safetensors' \
  --include 'video_vae.safetensors' --include 'text_encoder/*' --local-dir outputs/weights   # train.json paths
uv run flux-action index-lerobot --source-root outputs/xlerobot/source --output-dir outputs/xlerobot/index \
  --camera head=observation.images.head --camera left_wrist=observation.images.left_wrist \
  --camera right_wrist=observation.images.right_wrist --absolute-action-dims 5,11 --chunk-size 42 --val-episodes 5
# configs/xlerobot/train.json: copy configs/so101/train.json; policy: action_dim 12, camera_layout "grid",
# camera_keys [images.head, images.left_wrist, images.right_wrist], canvas_hw [384, 512],
# absolute_action_dims [5, 11], action_channel_weights [1,1,1,1,1,2,1,1,1,1,1,2]; top level: frame_hw [192, 256],
# shard_size 3, windows_per_rank 2, frozen_steps 200, trunk_warmup_steps 600, heads_warmup_steps 200,
# ema_sigma_rels [0.1]
uv run torchrun --nproc_per_node 3 -m flux_action.cli train --config configs/xlerobot/train.json \
  --override steps=4    # memory and step-time validation first
```

## 7. CPU checks (this laptop)

```bash
cd ws4/.cache/flux && uv venv --python 3.12 venv-lerobot-cpu
uv pip install --python venv-lerobot-cpu/bin/python --extra-index-url https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match "torch==2.10.0+cpu" "torchvision==0.25.0+cpu" -e "./lerobot[flux3,peft]" pytest
cd ~/liminal && ws4/.cache/flux/venv-lerobot-cpu/bin/python -m pytest ws4/flux/tests -q   # 23+ tests, about 20 s
python3 -m pytest ws4/flux/tests -q                                                       # numpy-only subset
```
