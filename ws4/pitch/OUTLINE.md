# Pitch outline: 3-minute track pitch and 5-minute final

Re-centred on the 4-skill chain, Sun 18 Oct about 07:45 CEST. Numbers come from [NUMBERS.md](NUMBERS.md) only
(section 0 is the final demo); every `[[NUM:...]]`, `[[TEAM:...]]` and `[[CLIP:...]]` is a placeholder with an owner
and a time in NUMBERS.md section 14 (`grep -rn "\[\[" ws4/pitch`). Scope (Oscar, 17 Oct): one team pitch, the
Haller chain is the live demo, the other workstreams get their own slot.

## The one-sentence story

A released world action model failed every zero-shot attempt on our robot; we traced it to a calibration contract,
taught four skills by VR overnight, and now one FLUX base with a small LoRA per skill puts away three objects with
one arm while the other holds a box open (a scripted replay, because the FLUX box skill never got going), all planned
and checked by Claude under one safety layer, while FLUX shows what it imagines before it acts.

## What the judges weigh, and where each beat scores

| Criterion (weight) | What earns it | Beats |
|---|---|---|
| Demo works on real hardware (25 %) | The chain live on the Haller (box scripted, three FLUX skills); real FLUX inserts for the red pen (3/5), the green pen and the cutter; the safety stack visibly holding | Live demo, R1 |
| Technical difficulty (20 %) | The contract diagnosis, the wrist-view knife edge found by a field-by-field request diff, two-arm safety (start gate, slow zone, lid held), decoding FLUX's imagined video | D1, K1, I1 |
| Use of FLUX (20 %) | FLUX in the loop for every learned step (green, cutter, red pen), one base and a LoRA per skill (all four public), zero-shot vs post-training on the same task, its video stream used (and improved by the fine-tune) | A1, R1, I1 |
| Idea and originality (15 %) | "Imagine, then act" made literal: imagined vs actual side by side; Claude plans and checks, never moves a motor | I1, A1 |
| Real-world use (10 %) | A bring-up kit: diagnose offline, teach by VR in a night, bound a learned policy; public data, adapter and a how-to | W1 |
| Demo video (10 %) | See [STORYBOARD.md](STORYBOARD.md) | Final |

The brief grades zero-shot vs post-training with successes, attempts, completed steps, failure points and
interventions: R1 shows all five.

## 3-minute track pitch (180 s, then the live demo and Q&A)

Run order: if the server is READY and the chain rehearsed, start it in the HMI during H1 and talk over it (the plan
takes about 6 s). Otherwise present first, then demo. Hard fallback at 12:30 (runbook): skills one by one.

| t (s) | Beat | Slide | Say (draft) | Show |
|---|---|---|---|---|
| 0-15 | **H1 Hook** | Cover | "This is Haller. One arm opens a box and holds the lid; the other puts away a green pen, a yellow cutter and a red pen with FLUX 3 Action. Claude plans the chain and checks every step." | Robot, HMI, chain starts |
| 15-45 | **D1 Problem and diagnosis** | 0/16, then the contract | "Zero-shot, the released checkpoint failed all 16 of our graded attempts: the shoulder ran to its limit whatever the prompt. The model was fine, the contract was not: its normalization assumes an old degree convention, so our arm looked impossible. Re-normalized offline, with no training, the lift error fell from 20 to 3 degrees." | 0/16, A to E table |
| 45-75 | **A1 Approach** | Architecture, data | "So we taught it. 337 VR takes over four skills, most of them recorded overnight in FLUX's exact contract, all public on the Hub. One FLUX base, one 0.45 GB LoRA per skill, about two hours each on one GPU. Claude Opus plans the chain, Claude Haiku checks each step from the camera, and nothing an LLM writes reaches a motor. The box step is a recorded demo replayed under the same safety layer, and the screen says so: our FLUX box skill would not start from rest." | Architecture, the chain table |
| 75-110 | **R1 Results** | Held-out, then the arm | "On takes it never trained on, the red-pen LoRA cut arm error from 5.5 to 3.2 degrees and predicts the right direction 88 % of the time, ahead of an ACT policy on the same data. On the arm: the first full red-pen inserts, 3 of 5, where ACT grasped but never inserted; this morning FLUX also inserted the green pen and the cutter, and Claude confirmed a red-pen insert in 33 seconds. And the full chain: two of three runs finished all four steps with no human touch." Say the caveats: small n, the chain ran in an unguarded demo mode, and in both completed runs the green-pen policy also slotted the red pen. | heldout, results |
| 110-140 | **K1 The knife edge** | Knife edge | "The hardest bug: after the robot moved, FLUX just held still. A field-by-field diff of live versus recorded requests found one difference, the wrist camera's view of the room. One pixel of shift flipped hold to reach. So we built a station-alignment tool and train with image augmentation." | Knife-edge slide |
| 140-165 | **I1 Imagine, then act** | Imagination | "FLUX also predicts a video of the next 1.4 seconds, and every stack throws it away. We decode it, with its actions bit-identical. Left is what happened, right what FLUX imagined. And fine-tuning made it act better and imagine better." | Clip, comparison |
| 165-180 | **W1 Close** | Close | "Anyone putting a released VLA on their own arm hits this. Our kit: diagnose offline, teach by VR in a night, bound the policy, and it is all public. `[[TEAM:NAME]]`, thank you." | Repo, Hub links, team |

Cut list if over time: drop the issue-2 and ACT clauses, shorten K1 to its first two sentences.

## 5-minute final (300 s, with the demo video)

| t (s) | Beat | Slide | Content |
|---|---|---|---|
| 0-10 | H1 Hook | Cover | As above, short |
| 10-70 | V1 Video | Video | The 60 s cut ([STORYBOARD.md](STORYBOARD.md)) |
| 70-100 | D1 | 0/16, contract, diagnosis | As above |
| 100-125 | A1 | Architecture, teach | As above, plus "each skill's workspace is built from its own demos" |
| 125-160 | R1 | heldout, results | As above, plus green 3.95 deg / 89 %, box 5.11 deg (the hardest skill, for ACT too), RAW beat EMA every time, guidance 1/1 as good as 3/3 at half the compute |
| 160-185 | K1 | Knife edge | As above |
| 185-210 | **S1 Safety** | Safety | "One honest note first: the final chain ran in an unguarded demo mode. The demonstrated-envelope zones caused 17 of 20 keep-out stops this morning, so for the demo the policies ran without keep-out and stall stops; joint limits, one degree per tick, the torque cap and the e-stop stayed on. In the single-skill runs: two arms share one box. The cutter has a slow zone next to the lid-holding arm, every pen step passes a start gate with the box arm's live pose, and every command goes through soft limits, one degree per tick, a keep-out built from the demos, a stall and load freeze and a torque cap. This morning the keep-out stepped in on 15 of 33 attempts and stopped 7; nothing broke. It exists because on Saturday, under zero-shot FLUX, our first arm's elbow pushed until its servo broke." |
| 210-235 | I1 | Imagination, imagine better | As above |
| 235-255 | T1 Team | Team | `[[TEAM:WORKSTREAMS]]`, one line each, no number without its owner. The teammates' behaviour-tree harness drove the real Haller through this stack (branch `claude/harness-bridge-11f8b6`) and placed the green pen in its best run |
| 255-285 | N1 Next | What's next | A FLUX box skill that starts from rest (every box LoRA failed the from-rest gate: 4.9 to 5.5 deg of first-chunk motion against the 20 required), LoRA hot-swap on one loaded base, robustness to the room, both arms at once; and the honest sim result: 0/20 |
| 285-300 | W1 Close | Close | As above |

## Demo moments (live, in order of value)

1. **Claude plans the chain.** The goal box is pre-filled; Plan shows two lanes (right arm: open the box; left arm:
   green, cutter, red pen) in about 6 s. Approve and run.
2. **The box opens and stays held.** The step card says "scripted replay of recorded demo 72, not FLUX" (take 72 in the final runs): say it out
   loud. Then the pen arm passes the start gate (shown in the events) and the FLUX steps run.
3. **Each Haiku check turns green** (held-out accuracy on the final scene: lid 63/63, green 114/119, cutter 91/91,
   red pen 65/66).
4. **The imagination panel** next to the live camera (`--imagine lazy`, no added round trip).
5. **E-stop on the spacebar**: both arms freeze, the lid stays held; Release, Resume.
6. **A failure, handled honestly**: a check fails, the executor retries or stops; say "that is in our table".

Live demo checklist (from notes `final_demo_runbook.md`): server real-serve-6000b READY and warmed (it stops after
45 min idle); station aligned with `ALIGN=1` (wrist offset < 15 px, scale 0.95 to 1.05, brightness 0.85 to 1.2);
one process per arm (`fuser /dev/haller_arm_right /dev/haller_arm_left` empty); hand near the power cable and on the
lid at every handover; box at P3, objects per the overview image; fallback video on the laptop.

## Q&A prep (honest answers)

| Likely question | Answer |
|---|---|
| Did the whole chain work? | Two of three runs finished 4/4 with no human touch (209 s and 302 s; one Claude replan in the second). The box was a scripted replay, the green pen and the cutter were FLUX, and the red pen was slotted by the green-pen policy on its way home, which the red step's check then confirmed. Unguarded demo mode: no keep-out or stall stops; joint limits, 1 deg per tick, torque cap and e-stop on |
| Why run unguarded? | 17 of 20 keep-out stops in the morning came from the demonstrated-envelope zones, not collision zones: they stopped good motion. Oscar approved the mode for the demo; the single-skill results are with the guards |
| Is the box opened by FLUX? | No, and the screen says so. The FLUX box LoRA went 0/3 in the runner and 0/1 in the HMI: from rest it barely moves (its first chunk moves 4.9 to 5.5 deg on held-out takes, demos move 39 to 150). The demo replays a recorded take (72 in the final runs). The point: the planner mixes learned skills and scripted primitives under one safety layer |
| How many real inserts, per skill? | Red pen: 3/5 in the benchmark block (9/23 over all RAW 5k attempts), 2/3 this morning, plus one in the HMI in 33 s confirmed by Haiku. Green pen: 1/2. Cutter: 1 insert (the trial log shows 1/6; it reached in all and grasped in most). Small n: say so |
| Can others use it? | Every adapter is public (`Oskrt/flux-3-action-so101-haller-{pen,open-box,green-pen,yellow-cutter}-lora`, all checkpoints inside), the five datasets too, plus `ws4/LORA_HOWTO.md` |
| Did zero-shot ever work? | No: 0/16 on the arm on Saturday, 0/20 in sim. Re-normalized offline it predicts the right direction (67 %) but no better than holding still in magnitude (5.5 vs 5.6 deg) |
| Is the gain from the LoRA or the normalization? | Both measured offline on the same 12 held-out takes: zero-shot B84 5.5 deg, median statistics 4.8, LoRA RAW 5k 3.24 (direction 66.5 to 87.5 %) |
| How sure are the 3/5 inserts? | Small n. Horizon 16 came with a 25 deg lift-stall threshold, two changes at once. Five of eight counting the three attempts logged right after. Same model at horizon 32: 0 inserts (stall freeze) |
| Compared with a non-FLUX policy? | ACT on the same red-pen data: offline 4.3 deg and 85 % (LoRA 3.24 and 87.5 %); on the arm it grasped and carried in 4 of 5 but never inserted (8/15 steps), FLUX RAW 5k at horizon 16 inserted 3 of 5. On the box ACT has better direction (80 vs 73 %), FLUX lower error (5.11 vs 5.6) |
| Isn't the knife edge a sign it is fragile? | Yes, and we measured it: one pixel of wrist-view shift flips hold vs reach on live frames. Recorded frames are robust to the same shifts, so it is the room view, not the pipeline. Fixes: align the station to the recording view (and its brightness) before running, and train with crop and colour augmentation |
| Why one LoRA per skill? | Each skill trains in about two hours on one GPU and can be checked on its own; the adapters are 0.45 GB on one shared base. Hot-swapping them on one loaded base is the next step |
| Why RAW and not EMA? | Measured: RAW beat EMA every time (pen 5k 3.24 vs 3.62 deg); at 1,250 updates a 0.999 EMA still carries much of the initial adapter |
| Why guidance 1.0? | Measured: 3/3 and 4/1 gave no better actions (3.83 and 3.67 vs 3.24 deg) at twice the compute per chunk |
| Does Claude control the robot? | No. It picks skills from a fixed list and checks outcomes; plans are validated; every motion is FLUX or a scripted primitive through the guards |
| How reliable are the checks? | Held-out frames of the final scene: lid 63/63, green 114/119 (2 false passes), cutter 91/91, red pen 65/66 (1 false pass); about 1.2 to 1.3 s and USD 0.0002 per check |
| Is the imagined video real? | Decoded from the latents FLUX denoises with every chunk. Bit-identical actions with imagination on or off, compile on and off, also on the live arm server. Separately compiled server processes can differ slightly; imagination itself changes nothing |
| What broke? | The red SO-101's elbow servo on Saturday, under zero-shot FLUX, before the keep-out existed. Nothing since; this morning the keep-out stopped 7 of 33 attempts |
| What is the "median" condition? | Median statistics of current-calibration SO-101 community datasets (multimodalart's Space, commit `09edebb`): lowest zero-shot arm error (4.8 deg), most balanced motion, but wrist roll below chance; the LoRAs train from our own quantiles |
| TTT / memory? | Backup only: a memory over FLUX features tracks the stage at 0.90 vs 0.54 in sim (GRU 0.87, so the memory matters, not TTT itself); the chain uses Claude checks |

## Open team facts (ask, do not guess)

- `[[TEAM:NAME]]`, `[[TEAM:MEMBERS]]`, who presents, who drives the robot.
- `[[TEAM:WORKSTREAMS]]`: what WS1, WS2, WS3 and WS5 each finished, with their own numbers.
- Live demo: the full chain (2 of 3 runs 4/4 this morning), box at P3 with the objects per the overview image.
