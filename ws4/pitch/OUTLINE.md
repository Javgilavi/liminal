# Pitch outline: 3-minute track pitch and 5-minute final

Draft, Sat 17 Oct 2026 about 22:00 CEST. Numbers come from [NUMBERS.md](NUMBERS.md) only; every `[[NUM:...]]`,
`[[TEAM:...]]` and `[[CLIP:...]]` is a placeholder for the morning (`grep -rn "\[\[" ws4/pitch`). Scope (Oscar,
17 Oct): one team pitch, the Haller pen task is the live demo, the other workstreams get their own slots.

## The one-sentence story

A released world action model failed 16 of 16 times on our robot; we traced it to a silent calibration contract,
fixed it without training, taught the task by VR in one evening, and now FLUX does it `[[NUM:LORA_PEN_SUCCESS]]`
under a safety stack, chained by Claude, while showing us what it imagines before it acts.

## What the judges weigh, and where each beat scores

| Criterion (weight) | What earns it | Beats |
|---|---|---|
| Demo works on real hardware (25 %) | The Haller does the pen task live, Claude chains it, the safety stack visibly holds | Live demo, R1, R3 |
| Technical difficulty (20 %) | The contract diagnosis (A to E table), decoding FLUX's imagined video, TTT at 7B scale, an FK keep-out with the demo envelope | D1, D2, I1 |
| Use of FLUX (20 %) | FLUX in the loop for every motion, zero-shot vs post-training on the same task, using its video stream and its features, not just its actions | R1, I1, I2 |
| Idea and originality (15 %) | "Imagine, then act" made literal: imagined vs actual side by side; nobody we found shows decoded FLUX imagination | I1 |
| Real-world use (10 %) | A bring-up kit for any new arm: diagnose offline before touching hardware, teach by VR, bound a learned policy | W1 |
| Demo video (10 %) | See [STORYBOARD.md](STORYBOARD.md) | Final only |

The official brief also grades zero-shot vs post-training with successes, attempts, completed steps, failure points
and interventions. R1 shows exactly those five columns.

## 3-minute track pitch (180 s, followed by the live demo and Q&A)

Run order: start the live chain in the HMI during H1 if the robot is warm and the server is READY; talk over it. If
the server is not READY by 14:25, present first and run the demo after the pitch.

| t (s) | Beat | Slide | Say (draft, trim to taste) | Show |
|---|---|---|---|---|
| 0-15 | **H1 Hook** | Cover | "This is Haller, our XLeRobot. One arm, soft fin-ray fingers, a head camera. FLUX 3 Action had never seen a robot like it. Last night it learned to put this pen in its slot." | Robot on the table; HMI on the screen; chain starts |
| 15-35 | **P1 Problem** | "0 of 16" | "Zero-shot, the released checkpoint failed every one of our 16 graded attempts. Not slowly: the shoulder ran straight to its limit, whatever the prompt said." | Big 0/16, one frame of the runaway |
| 35-70 | **D1 Diagnosis** | A to E table | "The model was fine. The contract was not. The released package normalizes joints with one community dataset in the old SO-100 degree convention: lift 85 to 184 degrees. Today's LeRobot calibration puts our folded arm at minus 100. FLUX saw an impossible robot. We tested five normalizations offline on our own demos, before touching the arm: with our own quantiles, the lift error drops from 19.9 to 3.2 degrees and FLUX predicts the demonstrated direction 80 % of the time. Zero-shot, no training. The adapter proposed in the public flux-action issue does not transfer." | Table with B and D highlighted |
| 70-100 | **A1 Approach** | Architecture | "Then we taught the task. VR teleop with a Quest: 96 demos in one evening, recorded in FLUX's exact contract. A LoRA from the re-normalized base, overnight on one RTX PRO 6000. FLUX runs on a GPU server; our runner wraps every command in a safety stack; Claude Opus plans the chain, Claude Haiku checks every step from the camera; nothing an LLM says ever reaches a motor." | Architecture diagram, left to right |
| 100-130 | **R1 Results** | Zero-shot vs post-training | "Same task, same layouts, same time budget. Zero-shot, with the statistics fixed two ways and no training: our own quantiles `[[NUM:B84_PEN_SUCCESS]]`, community median statistics `[[NUM:MEDIAN_PEN_SUCCESS]]`. After the LoRA: `[[NUM:LORA_PEN_SUCCESS]]`, `[[NUM:LORA_PEN_STEPS]]` steps on average, `[[NUM:LORA_PEN_HELP]]` interventions. It still fails when `[[NUM:LORA_PEN_FAILURE]]`." Before the arm, on 12 held-out takes it never trained on: "zero-shot was about as good as holding still; after 2,500 of 20,000 steps the LoRA already has 4.3 degrees of arm error against 5.5, and 74 % direction against 66.5 %" (update with `[[NUM:LORA_CURVE]]`). (Use the offline line alone if the real-arm attempts are not in yet.) | The five-column table, plus the held-out curve |
| 130-160 | **I1 Imagine, then act** | Imagined vs actual | "FLUX predicts a video of the future with every action chunk, and every stack throws it away. We decode it: 41 frames, 1.4 seconds ahead, and its actions stay bit-identical. Left is what happened, right is what FLUX imagined, on a take it never trained on. When they diverge, we know before the step fails." | Replay clip, held-out take 101 (`ws4/imagine/outputs/replay_ep101/imagination/imagined_vs_actual.mp4`); a live Sunday clip if one exists (`[[CLIP:IMAGINED_VS_ACTUAL]]`) |
| 160-180 | **W1 Why it matters** | Close | "Anyone putting a released VLA on their own arm hits this contract silently. We built the kit: diagnose offline, teach by VR in an evening, and bound the learned policy so a wrong guess costs a frozen arm, not a broken servo. `[[TEAM:NAME]]`, thank you." | Repo link, team names |

Cut list if over time: drop the issue-2 sentence in D1 (5 s), drop the Haiku latency detail in A1, shorten W1 to the
last sentence.

## 5-minute final (300 s, with the demo video)

Same spine, plus the safety story, the team, the research results and what is next. Play the video at V1 (or open
with it if the organisers prefer: then cut H1 to 5 s).

| t (s) | Beat | Slide | Content |
|---|---|---|---|
| 0-15 | H1 Hook | Cover | As in the 3-minute version |
| 15-75 | V1 Video | Video | The 60 s cut of the demo video ([STORYBOARD.md](STORYBOARD.md), "60 s cut") |
| 75-95 | P1 Problem | "0 of 16" | As above, add: "the brief grades zero-shot and post-training, so we logged every attempt" |
| 95-135 | D1 Diagnosis | A to E table | As above |
| 135-160 | **S1 Safety** | Safety stack | "On Saturday afternoon, under zero-shot FLUX, our first arm's elbow pushed against an obstruction at 92 % load and its servo broke. So every command now goes through six layers: soft limits, 1 degree per tick, a forward-kinematics keep-out fitted to the demonstrations, the parked arm as collision capsules, a stall and load freeze, a 60 % torque cap. Replayed, the two damaging attempts are refused within 1.3 seconds; all 24 good demos pass with zero refusals; on the Haller the load never passed 29 %." |
| 160-185 | A1 Approach | Architecture | As above |
| 185-215 | R1 Results | Results table | As above, add the sim row: "in simulation the same contract: released 0 of 20, LoRA `[[NUM:SIM_LORA]]`" |
| 215-240 | I1 Imagination | Imagined vs actual | As above |
| 240-260 | **I2 Memory** | Memory | "FLUX re-plans every second with 0.27 s of memory. We fitted test-time-training layers inside the 7B model: switched off they are bit-identical, they cost 4 %. And a memory over FLUX's own features tracks the task stage at 0.90, against 0.54 without memory. A GRU gets 0.87, so the memory matters, not TTT itself. Simulation only." |
| 260-280 | **T1 Team** | Team | `[[TEAM:WORKSTREAMS]]`: one line per workstream (WS1 single SO-101 + LoRA, WS2 bimanual obstacle, WS3 long-horizon orchestration, WS4 Haller + FLUX embodiment, WS5 hardware + UI). Do not claim another workstream's result without its owner's number |
| 280-300 | **N1 Next + close** | What's next | Both arms at once (two model instances on one 96 GB card), the pretrained EE50 bimanual heads (our IK gate passes: 25,168 frames round-trip at 1e-7 mm), imagination-driven retries. Close as W1 |

## Demo moments (live, in order of value)

1. **Claude plans, the operator approves.** Type "put the red pen in its slot, then take it out and put it back" in
   the HMI; the plan appears as lanes; approve. (About 6 to 12 s for the plan: say something while it thinks.)
2. **FLUX grasps and places the pen.** The imagined-vs-actual panel runs next to the live camera.
3. **Haiku verifies the step** from the head camera and the chain moves on (median 1.2 to 1.3 s per check; on 282
   recorded Haller frames 97.1 % for "put" and 98.2 % for "take", no false pass).
4. **E-stop on the spacebar**, then Release and Resume: the arm holds at its measured pose. Only if time allows.
5. **Failure handled honestly.** If a grasp misses, the check fails, the orchestrator retries or asks; say "this is
   the failure mode in our table" and move on.

Live demo checklist (Sunday 14:00 to 14:30): server READY about 9 min after start (`ws4/PLAN.md` "Risks"); the
pod idle-stops after 20 to 30 min without traffic, so warm it at 14:15; one process holds `/dev/haller_arm_*`
(`fuser /dev/ttyACM*` empty); the parked robot-right arm readable (else the e-stop latches by design); pen and box
at layout `[[TEAM:DEMO_LAYOUT]]`; fallback video loaded on the laptop.

## Q&A prep (honest answers)

| Likely question | Answer |
|---|---|
| Is the gain from the LoRA or from the normalization? | Both are measured separately: zero-shot B84 (the LoRA's exact base) on the pen is `[[NUM:B84_PEN_SUCCESS]]`, the LoRA on that base `[[NUM:LORA_PEN_SUCCESS]]`. Offline, normalization alone fixes direction (chance to 80 %) but not magnitude (46 to 102 % displacement error) |
| Why not just use the issue-2 adapter? | Tested as variant C: lift error 16.8 deg, direction at chance. Its offsets belong to that reporter's calibration |
| Did zero-shot ever succeed? | No: 0/16 on hardware, 0/20 in sim. With our quantiles the shoulder runaway stops, but wrist flex still pins at its soft limit (cause UNVERIFIED) |
| How much data, and why your own? | 96 VR demos for one task, `[[NUM:RECORD_HOURS]]` of recording. The released corpus has zero XLeRobot data and 11 of 1,021 head or overhead cameras; our fin-ray fingers and head camera are new to it |
| Does Claude control the robot? | No. Claude picks skills from a fixed list and checks outcomes. Plans are validated (schema, vocabulary, dependencies) and every motion is FLUX or a scripted primitive through the guards |
| Is the imagined video real or rendered? | Decoded from the latents FLUX denoises with every chunk (LeRobot `modeling_flux3.py:578` drops them). Decoding leaves the actions unchanged: bit-identical with imagination on or off, with `compile_model` on and off (measured on the RTX PRO 6000 pod). If asked about small differences between servers: separately compiled server processes can differ slightly; imagination itself changes nothing. On the scene camera the imagined frame beats "nothing moves" on 80 % of offsets (`ws4/imagine/README.md`) |
| Why guidance 1.0? | Halves the server time (1.46 to 0.76 s on the 4500). Its effect on quality vs the default 3.0 is UNVERIFIED |
| Two arms at once? | One shared model makes the arms take turns (30 % motion each, measured). Simultaneous motion needs two instances: next step |
| What broke? | The red SO-101's elbow servo, Saturday 17:30, under zero-shot A0 before the keep-out existed. That is why the safety stack exists |
| How does it compare to a non-FLUX policy? | ws4-act trained an ACT baseline on the same demos: at 10k steps arm MAE mean 4.3 deg and direction 81 to 90 % on the held-out takes, ahead of the FLUX LoRA at 2,500 steps on direction (74 %) at the same arm error (cited in `ws4/research/pen_lora.md`; ACT's own section is not in yet). Say it plainly: FLUX vs ACT is a real comparison, final numbers come from the 20k checkpoint and the arm (`[[NUM:LORA_CURVE]]`, `[[NUM:LORA_PEN_SUCCESS]]`). FLUX also predicts the video it expects to see, which ACT does not |
| What is the "median" condition? | Stock weights with the median statistics of current-calibration SO-101 community datasets, taken from multimodalart's public Space (`multimodalart/flux-3-action-so101-sim@09edebb`). Offline on our 12 held-out takes it has the lowest arm error (4.8 deg mean vs 5.5 for B84 and 5.6 for holding still) and the most balanced motion (every joint at 0.57 to 0.89 of the demo path; B84 over-swings pan 1.49x and barely lifts, 0.32x). But overall direction is no better (69 vs 67 %), wrist roll is below chance (42 %), and its gripper range is 0..40 vs our 0..100. So the LoRA stays on B84 (`ws4/research/pen_lora.md`) |
| Why TTT, if a GRU does almost as well? | TTT's case is a constant-size state over long contexts; our episodes are too short to show it, so on this data a GRU (0.87) is nearly as good as TTT (0.90). The finding is that FLUX's own features plus any memory track the stage (0.54 without memory). Caveat: on FLUX's failed sim rollouts, monitors trained only on demos invented progress (GRU 43 %, TTT 37 % of re-plans said "in cup"); adding 12 failed rollouts to training fixed it (`ws4/research/robottt_flux.md` 11.12). The real chain does not use this monitor: it uses Claude Haiku checks (97.1 % and 98.2 % on recorded Haller frames) |
| How reliable are Claude's checks? | Offline on 282 recorded Haller head-camera frames: "put" 97.1 %, "take" 98.2 %, no false pass on 278 negatives (52 with the pen still in the gripper); odd episodes held out from prompt tuning score 97.8 % and 99.1 %. Not yet measured live (`ws4/research/orchestrator.md`, "Success checks on real frames") |

## Open team facts (ask the team, do not guess)

- `[[TEAM:NAME]]` team name; `[[TEAM:MEMBERS]]` names and roles; who presents and who drives the robot.
- `[[TEAM:WORKSTREAMS]]` what WS1, WS2, WS3 and WS5 each finished, with their own numbers and clips.
- `[[TEAM:DEMO_LAYOUT]]` which pen and box layout for the live demo.
- Whether WS3's tree or the ws4 orchestrator drives the live chain (both exist; say one).
