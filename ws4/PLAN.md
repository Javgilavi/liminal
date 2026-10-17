# WS4 plan (orchestrator synthesis)

Sat 17 Oct 12:45 CEST. Built from all six agent reports; every suite was re-run by the orchestrator.
Submission: **Sun 18 Oct 14:00 CEST**. A snapshot of `ws4/` is on branch `ws4-handoff` of Javgilavi/liminal.

## Decisions

| Track | Verdict | Why (details in the linked report) |
|---|---|---|
| **A: FLUX SO-101 checkpoint on each XLeRobot arm + per-arm task LoRA** | **Sunday demo path** | Works with released weights and LeRobot's documented LoRA. One loaded model serves both arms with separate histories; parity with LeRobot's own loop verified on CPU. [flux_dual_arm.md](research/flux_dual_arm.md) |
| **D: behaviour tree / Claude on top of A** | **Yes, with WS3** | Coordination (who holds, who moves, retries) has to be scripted with per-arm policies. Claude picks from a fixed list of trained prompts and never reaches the motors. |
| **Imagination: FLUX's predicted video** | **Yes, the showcase** | Both LeRobot and flux-action compute the future video and throw it away. Our wrapper returns 41 decoded frames per chunk with bit-identical actions (CPU-verified on real LeRobot code). Nobody has shown real FLUX imagination yet. This is the strongest "Imagine, Then Act" story. [imagination.md](research/imagination.md) |
| **C1: 12-D bimanual heads via LeRobot PEFT** | Stretch, GPU 2, only with two leader arms and 50+ bimanual demos | Base builder works on CPU. BFL says PEFT on a new embodiment is not validated, and the heads start from zero: BFL removed the 14-D joint heads from the released base (safetensors metadata, `flux/checkpoint_facts.md`). |
| **B: EE50 (Simone's plan)** | **Not this weekend.** Kinematic gate PASSES; the model side is blocked. Post-hackathon track | IK: real robot motion round-trips exactly (25,168 frames, 100% success, p99 error 1e-7 mm), human handheld chunks are 98% position-feasible, median unexecutable rotation 0.5 deg, 1 to 3 ms per arm. Model side: BFL released no EE50 code, statistics or frame conventions; flux-action normalizes q01/q99, not per-offset z-scores; the pretrained EE50 heads do ship in the base (`action_prediction_ee50`, 50-D in and out, conditioned on state only, no past commands; `flux/checkpoint_facts.md`), but nothing released feeds them; flux-action has no LoRA. BFL's DROID result: 37.9% EE50 vs 40.1% joint heads. [ee50_ik.md](research/ee50_ik.md) |
| **C2: flux-action full fine-tune** | No | About 10 h per 1k updates on 3 PCIe GPUs (estimate). |
| **RoboTTT in the FLUX trunk** | No-go before Sunday | Gains come from a long-context pretraining run (16 GB200, 30K steps) plus hours of task data; no official code or weights. The hook fits FLUX cleanly (verified at tiny scale on flux-action's own DiT). [robottt_flux.md](research/robottt_flux.md) |
| **P2: TTT progress memory downstream of frozen FLUX** | Optional, decide at 18:00 | Reads one FLUX feature per replan, outputs task progress / current subtask to the orchestrator, no motor authority. Gates: FLUX running on the GPU, 30+ multi-stage episodes with subtask labels, WS3 wants the signal. Otherwise a research slide. |
| **Sim** | Built, use it | XLeRobot 0.4.0 CAD, D455 on a fixed head (confirmed by Oscar), your real soft fins (`SO101_soft_fin.stl`), env in LeRobot units with a per-arm FLUX view, props, domain randomisation, scripted picks. Camera poses, joint signs and gripper range still need one check on the robot. [sim/README.md](sim/README.md) |

## Data

- **Own demos are required.** They cover our task, the black fin-ray fingers, the RealSense mast pose and our
  calibration. Target 40 to 60 demos per skill for 3 to 4 skills, plus 10 held-out each (about 45 to 75 min
  of teleop, 2 to 3 h wall-clock for two people). See [xlerobot_recording_protocol.md](research/xlerobot_recording_protocol.md)
  and `flux/GPU_RUNBOOK.md` section 5.
  - One leader arm: record per arm (`so_follower` + `so_leader`, keys `scene` / `wrist`).
  - Two leader arms: record bimanual raw (`bi_so_follower`) and split with `data/convert_bimanual.py`. This
    keeps C1 possible.
- **Public data is cheap and fills a real gap.** FLUX SO-101 never saw a bimanual robot or an XLeRobot (BFL
  corpus file). About 900 public XLeRobot-style episodes from four labs (head camera behind the arms, two wrist
  cams, degrees, Apache-2.0). [datasets.md](research/datasets.md)
  - LeRobot main trains on one dataset only (`MultiLeRobotDataset` is deactivated, `datasets/factory.py:201`),
    so public and own data must be merged first. Merging needs identical features: same video codec and frame
    size (only encoder tuning keys are ignored, `configs/video.py:51-56`, `datasets/feature_utils.py:143-171`).
  - **Mergeable core set: `zonglin11/xlerobot_clean_table` + `cortx-labs/bimanual-fruit-sorting`, 9.5 GB, about
    6.5 arm-hours.** `rk000000` (h264) and `bigse0u1` (240x320 wrists) stay separate and optional.
  - **Our own recordings must be AV1 at 640x480** to merge with it (LeRobot's default `libsvtav1`, cameras at
    640x480 as in the recording protocol).
  - Tools: `data/fetch_public.sh` (download, convert per arm and 12-D, merge) and `data/merge_datasets.py`. The
    aggregate step itself is untested until real videos are on the training box.
- **Time-split evidence for the judges:** LoRA A (own only) vs LoRA B (own + at most 30 to 50% public) with
  identical steps. Report the held-out action error and hardware success. Keep public data only if B wins.

## Evaluation (graded: zero-shot and post-training both count)

The official BFL track brief grades **both** the released checkpoint zero-shot on our task **and** the
fine-tuned model on the same task under comparable conditions, with successful attempts, total attempts and
completed task steps recorded, plus where it failed and whether a human had to intervene.

- Zero-shot = `flux-3-action-so101` as released (revision `0713a7a`, weights identical to BFL's pinned `c9e13b2`),
  per arm, on the chosen task. Run it this afternoon even if it fails; failures are graded evidence. Keep
  testing it while LoRAs train.
  - Its normalization comes from one community dataset (`VoicAndrei/so100_kitchen`) in the pre-2025 SO-100
    degree convention (shoulder lift 85 to 184, elbow 61 to 179), not today's LeRobot degrees with zero at
    mid-range. So run two zero-shot variants: **A0** (package quantiles) and **A0r** (same weights, quantiles
    from 5 to 10 of our teleop episodes). Both leave the weights untouched; log both and say which is which.
- Post-training = LoRA A / B (and P): same task, same props and start layouts, same number of trials.
- Log every attempt with `eval/trial_log.py` (one line per attempt) and paste `trial_log.py summary --markdown`
  into the slides. Fix 3 start layouts and at least 5 attempts per condition before the first trial.

## Compute (decided by Oscar)

- **Training: the teammate's 3x RTX PRO 6000 box.** Only the teammate can reach it. Checklist:
  [HANDOFF_GPU.md](HANDOFF_GPU.md); code from branch `ws4-handoff` (snapshot at about 12:45, so pull the final
  reports when they are pushed).
- **Serving to the robot: RunPod pod `ws4-flux-inference`, live since about 13:30** (deployed by ws4-datasets
  with Oscar's approval): 1x RTX PRO 4500 Blackwell 32 GB in EU-RO-1, $0.72/h, on the 100 GB network volume
  `ws4-inference` at `/workspace`. No A40 was available in EU datacenters with network volumes; EU-RO-1 also has
  RTX PRO 6000 and A100 stock for a demo-hour upgrade on the same volume. Bootstrap runs there with `PREFETCH=1`;
  then `measure_latency --text-encoder-on-cpu` and `cloud/runpod_serve.sh` with the 20-minute idle self-stop.
  Details, SSH and gotchas (`source /etc/rp_environment`, env changes restart the container and change the SSH
  port): [cloud/RUNPOD.md](cloud/RUNPOD.md). LoRA adapters come from the training box via a private HF repo.
- **AWS: EUR 133 from the event, reimbursed against an invoice** (the team sets up the account and pays first;
  owner still being decided, RunPod stays the serving path meanwhile). Notes for whoever takes it:
  - New AWS accounts often start with a GPU vCPU quota of 0 ("Running On-Demand G and VT instances"); request
    the increase first, it can take hours.
  - G7e (RTX PRO 6000 Blackwell) is in London, Frankfurt and Stockholm. Third-party listing: g7e.2xlarge about
    $5.85/h in London vs $3.36/h in us-east-1 (UNVERIFIED, check the AWS calculator). EUR 133 is roughly 25 to
    45 GPU-hours, so the best use is the Sunday demo-window server or one extra training run.
  - Set an AWS Budgets alert at EUR 133 and keep the invoice for reimbursement.
- **Risks, orchestrator note.**
  - **Measured on the pod (RTX PRO 4500, Qwen on CPU, prompts pre-encoded, 10 chunks each):**

    | Setting | Median per chunk | p90 | Peak VRAM | Arm moving |
    |---|---|---|---|---|
    | default guidance 3 | 1.46 s | 1.47 s | 23.6 GiB | 42% |
    | guidance 3 + `compile_model` | 1.22 s | 1.23 s | 23.6 GiB | 47% |
    | **guidance 1.0 + compile** | **0.63 s** | 0.64 s | 23.6 GiB | **63%** |

    Model load is 6.4 min (plus about 2 min with compile) per server start. Without the prompt warm-up, the first
    use of a new prompt took 17 s.
  - **Open question: does guidance 1.0 hurt action quality?** The package default is guidance 3 on both streams.
    Run the zero-shot attempts at both settings and log them as separate variants before standardising on 1.0.
  - An RTX PRO 6000 pod for the Sunday demo window would roughly halve these times again (estimate) and fits two
    model instances for truly parallel arms.
  - Qwen must stay on the CPU here (23.6 + 8.3 GiB does not fit). About 8 GiB is left for imagination decoding.

### Training box GPUs (3x RTX PRO 6000, 96 GB)

| GPU | Afternoon (no own demos yet) | Night | Sunday |
|---|---|---|---|
| 0 | Smoke test, latency/VRAM, imagination `vaecheck` / `bench` / `replay` on a public XLeRobot episode (first imagined-vs-actual clip) | Offline evaluation of each checkpoint on held-out episodes | Eval / imagination clips for the video |
| 1 | **LoRA P: public XLeRobot core set only** (orchestrator suggestion: an early adapter that may fix the two-arms-in-view viewpoint, and a third point for the time-split table) | LoRA B (own + public) | Eval / spare |
| 2 | Public dataset download + per-arm conversion + merge | LoRA A (own only), or C1 if bimanual demos exist | Eval / spare |

Latency to expect (estimate): 0.6 to 1.0 s per SO-101 chunk against 1.07 s of executed motion, so motion will
pause between chunks. Levers: action and video guidance 1.0 (about 2x), `compile_model=true`, the other arm on
another GPU.

## Timeline (CEST)

| When | Who | What |
|---|---|---|
| Now to 14:00 | Oscar | HF gates: done (so101, base, droid all accessible with the token). Teammate starts [HANDOFF_GPU.md](HANDOFF_GPU.md). RunPod pod for serving. Calibrate both arms (LeRobot degrees, gripper range with the fin-ray fingers). Fix the demo task with the team. |
| 14:00 to 16:00 | training box | `flux/GPU_RUNBOOK.md` sections 1 and 2, `imagine/GPU_RUNBOOK.md` sections 1 to 5, public data fetch + merge, start LoRA P. RunPod pod: latency measurement first. |
| 14:00 to 19:00 | Oscar + 1 | Record own demos, skill by skill, with held-out episodes. |
| 16:00 to 18:00 | Oscar | Zero-shot A0 / A1 on the robot through the RunPod policy server, one arm at a time (runbook sections 3 and 4). Test LoRA P when it lands. |
| ~19:00 | training box | Start LoRA A and LoRA B overnight (identical steps). C1 on GPU 2 only if bimanual demos exist. |
| Sun 07:00 to 10:00 | team | Offline A vs B vs P, hardware trials, pick a checkpoint. Wire into the WS3 tree + imagination panel in the UI. |
| 10:00 to 12:30 | team | At least 5 trials per demo, demo video (imagined vs actual clips), slides. |
| 12:30 to 14:00 | team | Freeze, README, submit. |

## Open questions for Oscar

Answered: D455 on a fixed head (sim agent), HF gates accepted on account Oskrt, training on the teammate's box,
serving on RunPod.

1. How many leader arms can drive the XLeRobot? (One means per-arm only; two keeps C1 alive.)
2. Which demo task? Proposal: clear, open, pack, close ([recording protocol](research/xlerobot_recording_protocol.md)).
3. Wrist camera model, table height, and the on-robot checks the sim agent listed (one frame per camera at
   LeRobot zero, jog each joint once).

## Agent outputs (all CPU-tested, nothing run on a GPU yet, nothing committed)

| Path | Owner | Tests (re-run by the orchestrator) |
|---|---|---|
| `flux/` + [flux_dual_arm.md](research/flux_dual_arm.md) | ws4-flux-dual | 24 passed in the LeRobot-main env (`~/ws4/lerobot/.venv`) |
| `imagine/` + [imagination.md](research/imagination.md) | ws4-imagine | 157 passed, plus 5 of 5 against real LeRobot `flux3` code in the LeRobot-main env |
| `data/` + [datasets.md](research/datasets.md) | ws4-datasets | 34 passed (agent's count) |
| `kinematics/` + [ee50_ik.md](research/ee50_ik.md) | ws4-ik | 31 passed |
| `ttt/` + [robottt_flux.md](research/robottt_flux.md) | ws4-ttt | 30 passed |
| `sim/` + [sim/README.md](sim/README.md) | ws4-sim | 26 passed (headless EGL) |
| `eval/trial_log.py` (one JSONL line per attempt, `summary --markdown` for the slides) | orchestrator | 5 passed, ruff clean |
| `cloud/bootstrap.sh`, `cloud/push_ws4.sh` (orchestrator); `cloud/runpod_serve.sh`, `cloud/idle_stop.sh`, `cloud/RUNPOD.md`, `HANDOFF_GPU.md` (ws4-datasets) | orchestrator / ws4-datasets | bootstrap ran end to end on the laptop in env-only mode (both envs list sm_120; NATTEN 0.21.6 wheels match torch 2.10 and 2.11); not yet run on a GPU box |
