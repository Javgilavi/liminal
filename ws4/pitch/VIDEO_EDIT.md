# Submission video: edit list

Status: **v4 = final, chosen for submission by Oscar (`liminal_2min_v4.mp4`), Sun 18 Oct 12:50 CEST**: chain timelapse cold open, full-chain run on the console,
voice-over, original music, animated cards. v3 (12:32) is the same without the timelapse cold open.
Numbers in captions and voice come only from [NUMBERS.md](NUMBERS.md) (chain: section 0.1b; demo mode: 0.5).

| Cut | Drive | Local (laptop, not in the repo) |
|---|---|---|
| Main, 2:00 | `gdrive:liminal/05_video/liminal_2min_v4.mp4` | `~/liminal_video/out/liminal_2min_v4.mp4` |
| 60 s | `gdrive:liminal/05_video/liminal_60s_v4.mp4` | `~/liminal_video/out/liminal_60s_v4.mp4` |
| Older | `_v1` (silent), `_v2` (audio, placeholder chain), `_v3` (chain, red-pen cold open) | |

Format: 1920x1080, 30 fps, H.264 yuv420p, AAC 192 kb/s stereo 48 kHz, about -14 LUFS integrated, peaks under -1.5 dBFS.

## The chain section (run 12:05, `gdrive:liminal/06_full_chain/chain_20261018_120554_4of4/hmi.mp4`)

Cold open: `cam_head.mp4` of the same run, recording 80 s to the end at 36x (2:00) or 60x (60 s), last frame held
about a second; four step chips appear when each step starts (box at 80 s, green 99 s, cutter 157 s, check 329 s),
with the demo-mode line at the bottom. The head camera shares the console recording's timeline within about 1 s.

Chain shots: operator-console recording, top 1600x900 crop. Run time r maps to recording time t = r + about 28 s.

| Step | Recording in to out | What is on screen | Label |
|---|---|---|---|
| 1 box | 80 to 98 s | the replanned third try of the box replay opens the lid | SCRIPTED REPLAY OF A RECORDED DEMO, NOT FLUX |
| 2 green | 99 to 157 s | green-pen LoRA slots the green pen, then the red pen on its way home | FLUX 3 ACTION + GREEN-PEN LoRA |
| 3 cutter | 283 to 329 s | cutter LoRA, second try, inserts | FLUX 3 ACTION + CUTTER LoRA |
| 4 red pen | 329 to 336 s | Haiku confirms the red pen already in place; run succeeded | CLAUDE HAIKU CHECK |

- The box replays recorded take 72 in this run (the console says so); the card and voice say "a recorded demo".
- The red pen was placed by the green-pen LoRA, not by the red-pen LoRA (NUMBERS 0.1b); caption and voice say so.
- Demo mode: the final chain runs were unguarded (keep-out zones, stall and load stops and the start gate off;
  joint limits, 1 deg per tick, the 60 % torque cap and the e-stop on). The step-4 caption says "demo mode, guards
  off", and the safety card and voice say which guards were on in the chain and which only in the benchmark runs.
- The cutter step's first try timed out (120 s) and is skipped in the cut; the caption says "in on the second try".

## Audio

- Voice: Kokoro-82M (Apache-2.0, ONNX export `onnx-community/Kokoro-82M-v1.0-ONNX`), voice `af_heart`, speed 1.1,
  synthesized locally on the laptop CPU. Script per shot in the tables below (`~/liminal_video/vo_{2min,60s}.json`).
  To replace it with a human read, record each line and drop the WAVs into `~/liminal_video/vo/<cut>/` with the same
  names, then re-run the build.
- Music: original, synthesized for this video by `~/liminal_video/music.py` (numpy only: pads, bass, arpeggio,
  light drums, A minor, 96 BPM); no samples or third-party audio. The arrangement follows the sections (quiet under
  the problem, groove under results and the chain, drums drop for the imagination shot).
- Mix: voice normalized to -16 LUFS; music at -22 LUFS in the gaps and side-chain ducked about 8 to 9 dB under speech,
  so it sits more than 10 LU below the voice (AES guidance for commentary over music); limiter at -1 dBFS.

## Edit choices (from demo-video guidance)

- Cold open: the first 6 s show a real red-pen insert at real speed with the hook line, before the title.
- The narration says where to look ("on the right, its wrist camera"; "left, what happened; right, what FLUX
  imagined"), each point is made once, and numbers appear on screen as the voice says them.
- Cards build up in sync with the voice (staggered fade and slide); a "0 / 16" stamp on the zero-shot shot; chapter
  labels top-right; slow zoom on the stills; the chain is shown on the operator console so plan lanes, Claude's
  verdicts and the live imagined-vs-actual panel are all visible.

## How it is built

`~/liminal_video/cuts.py` holds both shot lists (source, in point, duration, speed, labels, caption);
`build.py <2min|60s> --version vN` (shot lists in `cuts2.py` since v2) renders each shot with ffmpeg (nice 19, 2 threads, one encode at a time) and joins
them; `gfx.py` draws the title, cards and caption overlays; `edl.py` writes the tables below. Phone clips are
downscaled first to `~/liminal_video/proxy/` (IMG_6396 proxy starts at 25.0 s of the original; the source times
below are original times). A full re-render of both cuts with audio takes about 5 minutes. To refine in an editor, cut the same
in and out points from the Drive originals and copy the caption text.

## Honesty rules applied

- Speed: every robot shot carries REAL SPEED or Nx SPEED in its top-left label.
- FLUX label only on FLUX runs: red pen (LoRA RAW 5k on B84), green pen (LoRA RAW 5k), cutter (augmented LoRA RAW 5k).
  Box footage appears only in chain step 1, labelled SCRIPTED REPLAY OF A RECORDED DEMO, NOT FLUX.
- Imagination clips are labelled as a held-out take replayed offline, not a live run.
- The safety card sits over IMG_6397, which is a failed attempt (05:36, layout A, 2 steps, load freeze); the card
  makes no success claim about it.
- The box step is labelled SCRIPTED REPLAY, NOT FLUX on the chain card and on the chain shot itself.

## Phone clip to run mapping (start times from the phone metadata, matched to `ws4/eval/trials.jsonl`)

| Phone clip (Drive root) | Starts | Run | Result in the trial log | Used |
|---|---|---|---|---|
| IMG_6396.MOV | 05:25:44 | `20261018_052519/attempt02`, layout B | success, 3 steps | hero, red pen (runner wrist view = phone + 2 s) |
| IMG_6397.MOV | 05:36:13 | `20261018_053419/attempt05`, layout A | fail, 2 steps (load freeze) | background of the safety card only |
| 20261018_074537.mp4 | 07:45:37 | `20261018_074025/attempt03`, layout C | success, 3 steps | 00_cold (cold open) |
| 20261018_092520.mp4 | 09:25:20 | `20261018_092508/attempt01`, cutter | 0 steps logged (grasp and carry visible) | 07b, caption and voice claim grasps only |
| 20261018_091619.mp4 | 09:16:19 | `20261018_091611`, green | 0 steps | title background frame (2.5 s) |
| 20261018_091845.mp4 | 09:18:45 | `20261018_091827`, green | 0 steps logged (grasp and carry visible) | not used: reconcile the grade first |
| 20261018_093134.mp4 | 09:31:34 | `20261018_093117` "pick up the yellow pen", stock checkpoint | not in the trial log | not used: ungraded |
| IMG_6401.MOV | 07:43:05 | `20261018_074025/attempt02` | fail, 0 steps | not used |
| IMG_6395.MOV | 04:53:55 | `20261018_044837` block, horizon 32 | 0/5 block | not used |
| IMG_6374 / 6387 / 6388 / 6392.MOV | Sat 19:10, 00:34, 00:44, 01:59 | VR teleop recording | n/a | 04a, 04b (6387, 6374) |
| IMG_6386 / 6378 / 6390.MOV | Sat 23:47, 20:11, 00:49 | unidentified run / VR recorder screen / another team | n/a | not used |

People are visible in 04a, 04b (VR operators), the hero background (05a, 05b) and the knife-edge wrist frames (09):
check with the team before publishing.

## Before submitting

- Done in v3: chain footage in, chain result from NUMBERS 0.1b. Remaining human checks: listen to the voice and
  music, and OK the people visible in the VR clips, the hero background, the knife-edge frames and the console's
  wrist cameras (spectators).

## Main cut (2:00)

| # | Out | Dur | Shot | Source | Source in to out | Speed | On-screen labels | Caption | Voice-over |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0:00.0 | 8.2 s | 00_cold | `gdrive:liminal/06_full_chain/chain_20261018_120554_4of4/cam_head.mp4` | 80.0 to 375.2 s | 36x | FULL CHAIN, RUN 12:05, REAL ARM; 36x SPEED | - | +0.2 s: This is FLUX 3 Action and Claude, clearing a table on our robot, with no human touch. On Saturday, the released model failed every attempt. |
| 2 | 0:08.2 | 3.4 s | 01_title | `title card over gdrive:liminal/20261018_091619.mp4 frame at 2.5 s` | still | 1x | - | Card: Liminal / Imagine, Then Act / Black Forest Labs FLUX 3 Action track / ERH Physical AI Hackathon, 17 to 18 Oct 2026 / A released world action model, our two-arm robot, and a weekend of VR teaching. | - |
| 3 | 0:11.6 | 5.3 s | 02_zeroshot | `01_real_robot_runs/20261017_171141/attempt01/video.mp4` | 1.0 to 6.3 s | 1x | ZERO-SHOT, RELEASED CHECKPOINT; REAL SPEED | Released FLUX 3 Action, zero-shot on our arms: 0 of 16 graded attempts succeeded | +0.3 s: Out of the box: zero of sixteen. The shoulder ran to its limit, whatever we asked. |
| 4 | 0:16.9 | 11.0 s | 03_contract | `card (NUMBERS.md section 3)` | still | 1x | - | Card: The model was fine. The contract was not. Released statistics from one community dataset in an older SO-100 convention; same weights with our statistics: shoulder-lift error 19.9 deg to 3.2 deg (offline, 200 frames of our demos). | +0.2 s: The model was fine. The contract was not. It expected an older joint convention. With statistics from our own demos, the lift error fell from twenty degrees to three. No training. |
| 5 | 0:27.9 | 3.5 s | 04a_vr_box | `gdrive:liminal/IMG_6387.MOV` | 11.0 to 14.5 s | 1x | VR TELEOP, META QUEST PRO; REAL SPEED | Then we taught it by VR: 337 teleop takes across four skills, all public on the Hub | +0.3 s: Then we taught it in VR: three hundred and thirty-seven tele-op takes, across four skills, all public. |
| 6 | 0:31.4 | 3.5 s | 04b_vr_pen | `gdrive:liminal/IMG_6374.MOV` | 9.5 to 13.0 s | 1x | VR TELEOP, META QUEST PRO; REAL SPEED | Then we taught it by VR: 337 teleop takes across four skills, all public on the Hub | - |
| 7 | 0:34.9 | 6.4 s | 05a_redpen_4x | `gdrive:liminal/IMG_6396.MOV + 01_real_robot_runs/20261018_052519/attempt02` | 35.0 to 60.6 s | 4x | FLUX 3 ACTION + LoRA, REAL ARM; 4x SPEED | First full FLUX inserts on the real arm: red pen, 3 of 5 in the benchmark block | +0.4 s: One FLUX base, one small LoRA per skill. Here's the red pen LoRA on the real arm, sped up, then at real speed. On the right, its wrist camera. Three of five inserts in our benchmark. |
| 8 | 0:41.3 | 9.0 s | 05b_redpen_1x | `gdrive:liminal/IMG_6396.MOV + 01_real_robot_runs/20261018_052519/attempt02` | 63.0 to 72.0 s | 1x | FLUX 3 ACTION + LoRA, REAL ARM; REAL SPEED | First full FLUX inserts on the real arm: red pen, 3 of 5 in the benchmark block | - |
| 9 | 0:50.3 | 3.0 s | 06a_green | `laptop ws4/single_arm/runs/20261018_083134/attempt01/video.mp4` | 20.0 to 32.0 s | 4x | FLUX 3 ACTION + LoRA, REAL ARM; ROBOT CAMERAS, 4x SPEED | One LoRA per skill, one shared base: green pen 1 of 2, cutter 1 of 6 inserts this morning | +0.3 s: The green pen and the cutter: same base, their own adapters. Each one inserted this morning. |
| 10 | 0:53.3 | 3.0 s | 06b_cutter | `laptop ws4/single_arm/runs/20261018_084210/attempt01/video.mp4` | 38.0 to 50.0 s | 4x | FLUX 3 ACTION + LoRA, REAL ARM; ROBOT CAMERAS, 4x SPEED | One LoRA per skill, one shared base: green pen 1 of 2, cutter 1 of 6 inserts this morning | - |
| 11 | 0:56.3 | 5.0 s | 07a_chain_card | `card (NUMBERS.md 0.1, 0.1b)` | still | 1x | - | Card: Claude plans. FLUX acts. Claude checks. Steps: 1 box arm opens and holds the lid (SCRIPTED REPLAY, NOT FLUX), 2 green pen, 3 yellow cutter, 4 red pen (FLUX 3 Action + LoRA). Footer: step 1 replays a recorded VR demo; our FLUX box skill would not start from rest. | +0.3 s: For the demo, Claude Opus plans the chain, and Claude Haiku checks every step. Step one is a scripted replay of a recorded demo, not FLUX. Then the green pen LoRA slots the green pen, and the red pen on its way home. The cutter goes in on the second try. Two of three chain runs finished with no human touch. |
| 12 | 1:01.3 | 3.0 s | 07b_chain_step1 | `gdrive:liminal/06_full_chain/chain_20261018_120554_4of4/hmi.mp4` | 80.0 to 98.0 s | 6x | STEP 1: SCRIPTED REPLAY OF A RECORDED DEMO, NOT FLUX; 6x SPEED | Box arm: scripted replay of a recorded demo, not FLUX (third try, after Claude replanned) | - |
| 13 | 1:04.3 | 7.2 s | 07c_chain_step2 | `gdrive:liminal/06_full_chain/chain_20261018_120554_4of4/hmi.mp4` | 99.0 to 157.0 s | 8x | STEP 2: FLUX 3 ACTION + GREEN-PEN LoRA; 8x SPEED | Green-pen LoRA: the green pen goes in, then the red pen on its way home | - |
| 14 | 1:11.5 | 5.8 s | 07d_chain_step3 | `gdrive:liminal/06_full_chain/chain_20261018_120554_4of4/hmi.mp4` | 283.0 to 329.0 s | 8x | STEP 3: FLUX 3 ACTION + CUTTER LoRA; 8x SPEED | Cutter LoRA: in on the second try | - |
| 15 | 1:17.3 | 3.6 s | 07e_chain_step4 | `gdrive:liminal/06_full_chain/chain_20261018_120554_4of4/hmi.mp4` | 329.0 to 336.2 s | 2x | STEP 4: CLAUDE HAIKU CHECK; 2x SPEED | 2 of 3 chain runs: 4 of 4 steps, no human touch | - |
| 16 | 1:20.9 | 10.0 s | 08a_imagine | `gdrive:liminal/00_highlights/imagination_lora5k_imagined_vs_actual_ep101.mp4` | 0.0 to 10.0 s | 1x | FLUX IMAGINATION, HELD-OUT TAKE REPLAYED | FLUX imagines the next 1.4 s. We decode it; its actions stay bit-identical. | +0.3 s: FLUX also predicts the next one point four seconds of video. We decode it, and the actions stay bit-identical. Left, what happened. Right, what FLUX imagined. Fine-tuning made it imagine better, too. |
| 17 | 1:30.9 | 4.0 s | 08b_imagine_cmp | `gdrive:liminal/00_highlights/imagination_b84_vs_lora5k_ep101.png` | still | 1x | FLUX IMAGINATION, OFFLINE | Fine-tuning made it imagine better: zero-shot smears the pen by 1 s ahead, the LoRA keeps it whole | - |
| 18 | 1:34.9 | 11.0 s | 09_knife | `gdrive:liminal/00_highlights/knife_edge_wrist_view_training_vs_now.jpg` | still | 1x | DIAGNOSIS, OFFLINE | After the robot moved, FLUX only held still. One pixel of wrist-camera shift flipped hold and reach, so we align the station and augment the images. | +0.2 s: Our hardest bug: after we moved the robot, FLUX just held still. One pixel of wrist-camera shift flipped hold and reach. So now we align the station, and augment the images. |
| 19 | 1:45.9 | 8.8 s | 10_safety | `card over gdrive:liminal/IMG_6397.MOV (05:36 attempt, not a success)` | 20.0 to 28.8 s | 1x | - | Card: always on: joint limits, 1 deg per tick, 60 % torque cap, e-stop. Benchmark runs add keep-out zones, stall and load freeze, start gate. The filmed chain ran in demo mode without those guards. Keep-out refused ticks in 15 of 33 benchmark attempts, ended 7, no collision reported. | +0.3 s: Every command is rate-limited and torque-capped. Our benchmark runs add keep-out zones and a stall freeze. The filmed chain ran without them. |
| 20 | 1:54.7 | 5.3 s | 11_end | `card` | still | 1x | - | Card: github.com/Javgilavi/liminal, huggingface.co/Oskrt adapters and datasets. | +0.3 s: The code, the adapters and the data are all public. Liminal. Imagine, then act. |

Total: 1:60.0

## 60 s cut

| # | Out | Dur | Shot | Source | Source in to out | Speed | On-screen labels | Caption | Voice-over |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0:00.0 | 5.0 s | 00_cold | `gdrive:liminal/06_full_chain/chain_20261018_120554_4of4/cam_head.mp4` | 80.0 to 380.0 s | 60x | FULL CHAIN, RUN 12:05, REAL ARM; 60x SPEED | - | +0.2 s: FLUX 3 Action and Claude, clearing a table with no human touch. On Saturday, the released model failed every attempt. |
| 2 | 0:05.0 | 2.6 s | 01_title | `title card over gdrive:liminal/20261018_091619.mp4 frame at 2.5 s` | still | 1x | - | Card: Liminal / Imagine, Then Act / Black Forest Labs FLUX 3 Action track / ERH Physical AI Hackathon, 17 to 18 Oct 2026 / A released world action model, our two-arm robot, and a weekend of VR teaching. | - |
| 3 | 0:07.6 | 4.0 s | 02_zeroshot | `01_real_robot_runs/20261017_171141/attempt01/video.mp4` | 2.0 to 6.0 s | 1x | ZERO-SHOT, RELEASED CHECKPOINT; REAL SPEED | Released FLUX 3 Action, zero-shot on our arms: 0 of 16 graded attempts succeeded | +0.9 s: Zero of sixteen, out of the box. |
| 4 | 0:11.6 | 4.0 s | 04_vr_pen | `gdrive:liminal/IMG_6374.MOV` | 9.0 to 13.0 s | 1x | VR TELEOP, META QUEST PRO; REAL SPEED | Then we taught it by VR: 337 teleop takes across four skills, all public on the Hub | +0.3 s: So we fixed its joint contract, and taught it in VR. |
| 5 | 0:15.6 | 5.0 s | 05a_redpen_4x | `gdrive:liminal/IMG_6396.MOV + 01_real_robot_runs/20261018_052519/attempt02` | 40.0 to 60.0 s | 4x | FLUX 3 ACTION + LoRA, REAL ARM; 4x SPEED | First full FLUX inserts on the real arm: red pen, 3 of 5 in the benchmark block | +0.3 s: One FLUX base, one small LoRA per skill. The red pen: three of five inserts. On the right, its wrist camera. |
| 6 | 0:20.6 | 7.0 s | 05b_redpen_1x | `gdrive:liminal/IMG_6396.MOV + 01_real_robot_runs/20261018_052519/attempt02` | 64.5 to 71.5 s | 1x | FLUX 3 ACTION + LoRA, REAL ARM; REAL SPEED | First full FLUX inserts on the real arm: red pen, 3 of 5 in the benchmark block | - |
| 7 | 0:27.6 | 2.0 s | 06a_green | `laptop ws4/single_arm/runs/20261018_083134/attempt01/video.mp4` | 23.0 to 31.0 s | 4x | FLUX 3 ACTION + LoRA, REAL ARM; ROBOT CAMERAS, 4x SPEED | One LoRA per skill, one shared base: green pen 1 of 2, cutter 1 of 6 inserts this morning | +0.3 s: Green pen and cutter: same base, their own adapters. |
| 8 | 0:29.6 | 2.0 s | 06b_cutter | `laptop ws4/single_arm/runs/20261018_084210/attempt01/video.mp4` | 42.0 to 50.0 s | 4x | FLUX 3 ACTION + LoRA, REAL ARM; ROBOT CAMERAS, 4x SPEED | One LoRA per skill, one shared base: green pen 1 of 2, cutter 1 of 6 inserts this morning | - |
| 9 | 0:31.6 | 3.5 s | 07a_chain_card | `card (NUMBERS.md 0.1, 0.1b)` | still | 1x | - | Card: Claude plans. FLUX acts. Claude checks. Steps: 1 box arm opens and holds the lid (SCRIPTED REPLAY, NOT FLUX), 2 green pen, 3 yellow cutter, 4 red pen (FLUX 3 Action + LoRA). Footer: step 1 replays a recorded VR demo; our FLUX box skill would not start from rest. | +0.2 s: Claude plans and checks each step. The box is a scripted replay, not FLUX. FLUX puts the pens and the cutter away. Two of three chain runs: no human touch. |
| 10 | 0:35.1 | 2.0 s | 07b_chain_step1 | `gdrive:liminal/06_full_chain/chain_20261018_120554_4of4/hmi.mp4` | 80.0 to 98.0 s | 9x | STEP 1: SCRIPTED REPLAY OF A RECORDED DEMO, NOT FLUX; 9x SPEED | Box arm: scripted replay of a recorded demo, not FLUX (third try, after Claude replanned) | - |
| 11 | 0:37.1 | 4.0 s | 07c_chain_step2 | `gdrive:liminal/06_full_chain/chain_20261018_120554_4of4/hmi.mp4` | 99.0 to 157.0 s | 14.5x | STEP 2: FLUX 3 ACTION + GREEN-PEN LoRA; 14.5x SPEED | Green-pen LoRA: the green pen goes in, then the red pen on its way home | - |
| 12 | 0:41.1 | 2.5 s | 07d_chain_step3 | `gdrive:liminal/06_full_chain/chain_20261018_120554_4of4/hmi.mp4` | 283.0 to 328.0 s | 18x | STEP 3: FLUX 3 ACTION + CUTTER LoRA; 18x SPEED | Cutter LoRA: in on the second try | - |
| 13 | 0:43.6 | 2.0 s | 07e_chain_step4 | `gdrive:liminal/06_full_chain/chain_20261018_120554_4of4/hmi.mp4` | 329.0 to 333.0 s | 2x | STEP 4: CLAUDE HAIKU CHECK; 2x SPEED | 2 of 3 chain runs: 4 of 4 steps, no human touch | - |
| 14 | 0:45.6 | 7.4 s | 08a_imagine | `gdrive:liminal/00_highlights/imagination_lora5k_imagined_vs_actual_ep101.mp4` | 1.0 to 8.4 s | 1x | FLUX IMAGINATION, HELD-OUT TAKE REPLAYED | FLUX imagines the next 1.4 s. We decode it; its actions stay bit-identical. | +0.3 s: FLUX also imagines the next one point four seconds. We decode it, and the actions stay bit-identical. |
| 15 | 0:53.0 | 7.0 s | 11_end | `card` | still | 1x | - | Card: github.com/Javgilavi/liminal, huggingface.co/Oskrt adapters and datasets. | +0.3 s: Everything is public. Liminal. Imagine, then act. |

Total: 1:00.0
