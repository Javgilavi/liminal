# WS4 plan (orchestrator synthesis)

Interim version, Sat 17 Oct 12:20 CEST. Built from the finished agent reports. Still pending: the IK verdict
(ws4-ik), RoboTTT (ws4-ttt) and the sim (ws4-sim). Their sections get filled in when they report.
Submission: **Sun 18 Oct 14:00 CEST** (about 26 h from now).

## Decisions

| Track | Verdict | Why (details in the linked report) |
|---|---|---|
| **A: FLUX SO-101 checkpoint on each XLeRobot arm + per-arm task LoRA** | **Sunday demo path** | Works with released weights and LeRobot's documented LoRA. One loaded model serves both arms with separate histories; parity with LeRobot's own loop verified on CPU. [flux_dual_arm.md](research/flux_dual_arm.md) |
| **D: behaviour tree / Claude on top of A** | **Yes, with WS3** | Coordination (who holds, who moves, retries) has to be scripted with per-arm policies. Claude picks from a fixed list of trained prompts and never reaches the motors. |
| **Imagination: FLUX's predicted video** | **Yes, the showcase** | Both LeRobot and flux-action compute the future video and throw it away. Our wrapper returns 41 decoded frames per chunk with bit-identical actions (CPU-verified on real LeRobot code). Nobody has shown real FLUX imagination yet. This is the strongest "Imagine, Then Act" story. [imagination.md](research/imagination.md) |
| **C1: 12-D bimanual heads via LeRobot PEFT** | Stretch, GPU 2, only with two leader arms and 50+ bimanual demos | Base builder works on CPU. BFL says PEFT on a new embodiment is not validated, and the heads start from zero. |
| **B: EE50 (Simone's plan)** | **Not this weekend**, post-hackathon research track | No EE50 code, statistics or frame conventions are released. flux-action has no LoRA. LeRobot cannot load the EE50 heads without a new processor written blind. BFL's own DROID result: 37.9% EE50 vs 40.1% joint heads. The data side (FK, UMI converters) is built and kept. IK numbers are pending from ws4-ik. |
| **C2: flux-action full fine-tune** | No | About 10 h per 1k updates on 3 PCIe GPUs (estimate). |
| **RoboTTT** | Pending (ws4-ttt) | |

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
  cams, degrees, Apache-2.0). Core set 16.3 GB: `zonglin11/xlerobot_clean_table`,
  `rk000000/xlerobot-handkerchief-quarters-mix-20260918-480`, `bigse0u1/xlerobot_scrub_7tool` (flip the gripper),
  `cortx-labs/bimanual-fruit-sorting`. Download on the VM, convert per arm (metadata only, minutes).
  [datasets.md](research/datasets.md)
- **Time-split evidence for the judges:** LoRA A (own only) vs LoRA B (own + at most 30 to 50% public) with
  identical steps. Report the held-out action error and hardware success. Keep public data only if B wins.

## GPUs (3x RTX PRO 6000, 96 GB)

| GPU | Afternoon (no own demos yet) | Night | Sunday |
|---|---|---|---|
| 0 | Smoke test, latency/VRAM, imagination `vaecheck` / `bench` / `replay` on a public XLeRobot episode (first imagined-vs-actual clip) | Policy server for robot tests (`flux/remote.py` over an SSH tunnel) | Demo server + async imagination decode |
| 1 | **LoRA P: public XLeRobot core set only** (orchestrator suggestion: an early adapter that may fix the two-arms-in-view viewpoint, and a third point for the time-split table) | LoRA B (own + public) | Eval / spare |
| 2 | Public dataset download + per-arm conversion | LoRA A (own only), or C1 if bimanual demos exist | Eval / spare |

Latency to expect (estimate): 0.6 to 1.0 s per SO-101 chunk against 1.07 s of executed motion, so motion will
pause between chunks. Levers: action and video guidance 1.0 (about 2x), `compile_model=true`, the other arm on
another GPU.

## Timeline (CEST)

| When | Who | What |
|---|---|---|
| Now to 14:00 | Oscar | Accept the FLUX licence on HF + token. `push_ws4.sh` + `cloud/bootstrap.sh` on the VM. Calibrate both arms (LeRobot degrees, gripper range with the fin-ray fingers). Answer the sim agent. Fix the demo task with the team. |
| 14:00 to 16:00 | VM | `flux/GPU_RUNBOOK.md` sections 1 and 2, `imagine/GPU_RUNBOOK.md` sections 1 to 5, dataset download + conversion, start LoRA P. |
| 14:00 to 19:00 | Oscar + 1 | Record own demos, skill by skill, with held-out episodes. |
| 16:00 to 18:00 | Oscar | Zero-shot A0 / A1 on the robot through the policy server, one arm at a time (runbook sections 3 and 4). Test LoRA P when it lands. |
| ~19:00 | VM | Start LoRA A and LoRA B overnight (identical steps). C1 on GPU 2 only if bimanual demos exist. |
| Sun 07:00 to 10:00 | team | Offline A vs B vs P, hardware trials, pick a checkpoint. Wire into the WS3 tree + imagination panel in the UI. |
| 10:00 to 12:30 | team | At least 5 trials per demo, demo video (imagined vs actual clips), slides. |
| 12:30 to 14:00 | team | Freeze, README, submit. |

## Open questions for Oscar

1. RealSense model, and does the head have pan/tilt servos? Finger STL or dimensions, wrist camera model, table
   height (the sim agent asks in its session).
2. How many leader arms can drive the XLeRobot? (One means per-arm only; two keeps C1 alive.)
3. Which HF account accepts the FLUX gates, and how does the robot PC reach the VM (SSH tunnel)?
4. Which demo task? Proposal: clear, open, pack, close ([recording protocol](research/xlerobot_recording_protocol.md)).
5. OK to download the 16.3 GB public core set on the VM?

## Agent outputs (all CPU-tested, nothing run on a GPU yet, nothing committed)

| Path | Owner | Tests (re-run by the orchestrator) |
|---|---|---|
| `flux/` + [flux_dual_arm.md](research/flux_dual_arm.md) | ws4-flux-dual | 24 passed in the LeRobot-main env (`~/ws4/lerobot/.venv`) |
| `imagine/` + [imagination.md](research/imagination.md) | ws4-imagine | 157 passed, plus 5 of 5 against real LeRobot `flux3` code in the LeRobot-main env |
| `data/` + [datasets.md](research/datasets.md) | ws4-datasets | 33 passed |
| `kinematics/` + `research/ee50_ik.md` | ws4-ik | in progress (26 passed, 1 failing mid-work) |
| `ttt/` + `research/robottt_flux.md` | ws4-ttt | in progress (30 passed so far) |
| `sim/` | ws4-sim | in progress |
| `cloud/bootstrap.sh`, `cloud/push_ws4.sh` | orchestrator | bootstrap ran end to end on the laptop in env-only mode (both envs list sm_120; NATTEN 0.21.6 wheels match torch 2.10 and 2.11); not yet run on a GPU box |
