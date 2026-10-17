# ws4/ttt: RoboTTT-style test-time-training memory for FLUX 3 Action

CPU-testable PyTorch prototype that goes with `ws4/research/robottt_flux.md`. Standalone: no imports from
`src/liminal`. Needs `torch` (and `einops` for the flux-action hook tests).

| File | What |
|---|---|
| `ttt_layer.py` | `TTTLayer`: multi-head fast weights (linear or 2-layer GELU MLP), KVB inner loss, mini-batch inner updates, learnable `W0` and per-token inner learning rate, hand-written inner gradients. `TanhGate`, `GatedTTT` (RoboTTT Eq. 3), `detach_state` |
| `temporal.py` | `TemporalTTTBlock`: per-timestep transformer block + gated TTT across timesteps (one timestep per inner mini-batch). Helpers: truncated BPTT slices, sequence action forcing noise levels, masked sequence loss |
| `flux_hook.py` | `FluxTTTAdapter`: hooks gated TTT layers after chosen joint blocks of flux-action's `JointSingleSeq` on the action-stream tokens, with commit semantics for denoising and TBPTT. `FluxFeatureTap`: per-re-plan FLUX features for a downstream memory. Pinned to flux-action@830f1ce |
| `toy_recall.py` | Toy long-horizon task: pretrained memory-less policy, wrapped with a gated TTT layer, fine-tuned with truncated BPTT, evaluated on 64x longer gaps |
| `progress_memory.py` | Pipeline P2: `EpisodeProgressMemory`, an episode memory (TTT, GRU or none) over one FLUX feature per re-plan that predicts the current subtask and a done flag for the orchestrator; `fit`, streaming `step`, and a synthetic aliased-stages check (GRU matched or beat TTT there) |
| `labels.py` | Shared stage-label format (`episode.json`, schema `ws4.stage_labels/1`): plan, segments, re-plan ticks; per-re-plan targets; orchestrator start/end event log -> segments |
| `sim_episodes.py` | Labelled multi-stage sim episodes of the demo skills (orchestrator pack-box scene and scripted stand-ins, read-only), 30 Hz state/commands, FLUX snapshot frames per re-plan |
| `features.py` | CPU `FrameFeaturizer` (colour grids + head motion + state), FLUX-feature loading, per-re-plan targets, plan posterior |
| `train_progress.py` | Train/evaluate the P2 memory (none, last-k, GRU, TTT) on held-out episodes; saves `ProgressMonitor` checkpoints |
| `progress_api.py` | `ProgressMonitor.load(...)`, `.reset(plan)`, `.update(frames=..., state=...)` -> `StageEstimate` (stage, done, progress, confidence): the orchestrator interface |
| `flux_sim_closed_loop.py` | Closed loop in sim: FLUX (remote server), the scripted stand-in or a fake policy drives the arm through `flux.dual_arm_runner`; ground-truth checks per re-plan, stand-in repair, live progress monitor, episodes in the `sim_episodes.py` layout plus `overlay.mp4` (report Sec. 11.10) |
| `sim_block_cup.py` | Sim track: scripted "put the block in the cup" demos on the station model as a LeRobot v3.0 dataset (30 fps, left arm 6-D degrees, head 480x640 as `scene`, left wrist as `wrist`), per-frame phases in `stages_left.jsonl`, held-out solvable layouts (`--eval-seeds`) |
| `eval_block_cup.py` | Closed loop of one served FLUX package on the cup task over the held-out layouts (`flux.dual_arm_runner`, default guard); success at the first passed check and after the policy keeps going; rollouts in the `sim_episodes.py` layout. Runs on the pod, so the token stays there |
| `cup_stages.py` | Three-stage tracking (reach, carry, in cup) for the cup task: trains none/GRU/TTT on the demos, scores FLUX rollouts with stages from privileged block and cup poses |
| `pod_sim_lora.sh` | Pod: A0r-sim cup package (demo quantiles), LoRA probe and run (LeRobot `lora.json`), adapter merge for `flux.remote` |
| `lora_train_te_cpu.py` | `lerobot_train` with FLUX's frozen text encoder kept on the CPU, which lets the LoRA preset fit the 32 GB card |
| `pod_sim_eval.sh`, `pod_job_watchdog.sh` | Pod: serve one package, run `eval_block_cup.py`, stop serving; idle stop that also counts running ws4/ttt jobs as busy |
| `extract_flux_features.py` | GPU: FLUX 3 Action tap features at every re-plan, with the runner's exact request building |
| `pod_job.py` | One model load for a time-boxed GPU slot: the scale check, then the feature extraction (what ran on the RunPod pod) |
| `flux_scale_check.py` | GPU (optional): gated TTT layers in the loaded 7B model, gate-0 identity, added latency and VRAM |
| `claude_stage_labels.py` | Stage boundaries from one Claude vision call per episode (about 20 frames, structured output), saved as `episode_claude.json`; `--eval` against sim ground truth |
| `models/` | Trained progress checkpoints (sim frame features, GRU is the default; `*_subset21_*`: FLUX-feature models from the pod run) |
| `results/` | Metrics JSON, the pod's scale check and extraction log, `episode_pack_lidded-100004.mp4` / `_sheet.jpg` (an annotated training episode), and the FLUX-in-sim closed loop: `flux_sim_closed_loop.json` (A0 vs A0r-sim), `flux_sim_so101sim_A0r_ep0_overlay.mp4` / `_sheet.jpg`, `stats_sim_left.json` (the A0r-sim quantiles) |
| `tests/` | 48 tests (2 of them run in the LeRobot CPU venv): inner gradients vs autograd, exact recall, streaming = one pass, causality, gate identity, TBPTT gradient cut, and the adapter on a tiny instance of flux-action's real DiT (packed training = streaming inference) |

```bash
python -m pytest ws4/ttt/tests -q          # about 10 s on a laptop CPU
python ws4/ttt/toy_recall.py --quick       # smoke run
python ws4/ttt/toy_recall.py               # full toy run, writes ws4/ttt/results/toy_recall.json
python ws4/ttt/progress_memory.py          # synthetic check of the P2 episode memory
MUJOCO_GL=egl python ws4/ttt/sim_episodes.py --mix --episodes 200 --workers 4 --seed 1   # labelled sim data
python ws4/ttt/train_progress.py --save-dir ws4/ttt/models --out ws4/ttt/results/progress_sim_frame.json
ws4/.cache/flux/venv-lerobot-cpu/bin/python -m pytest ws4/ttt/tests/test_lerobot_integration.py -q
```

Every CLI here runs at `nice 19` with at most 4 threads (`throttle.py`): the laptop also runs the 30 Hz robot
loops. `WS4_TTT_THREADS` changes the cap; `WS4_TTT_NICE=0` only on a machine with nothing else running.

The flux-action tests import `ws4/.cache/flux-action/src` (gitignored clone of
`black-forest-labs/flux-action@830f1ce`) and are skipped when it is absent. To get it:

```bash
git clone https://github.com/black-forest-labs/flux-action ws4/.cache/flux-action
git -C ws4/.cache/flux-action checkout 830f1ce178d7aea52f18fde31d8e8e36ce6bae21
```

Not done here: anything with real FLUX weights or a GPU (runbook: `ws4/research/robottt_flux.md` Sec. 11.7).
