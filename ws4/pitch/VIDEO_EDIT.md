# Submission video: edit list

Status: **v2, Sun 18 Oct 11:00 CEST**: voice-over, original music, animated cards. The chain shots (07a, 07b) are
placeholders until the full-chain footage lands. Numbers in captions and voice come only from
[NUMBERS.md](NUMBERS.md).

| Cut | Drive | Local (laptop, not in the repo) |
|---|---|---|
| Main, 2:00 | `gdrive:liminal/05_video/liminal_2min_v2.mp4` | `~/liminal_video/out/liminal_2min_v2.mp4` |
| 60 s | `gdrive:liminal/05_video/liminal_60s_v2.mp4` | `~/liminal_video/out/liminal_60s_v2.mp4` |
| v1 (silent, kept) | `gdrive:liminal/05_video/liminal_{2min,60s}_v1.mp4` | |

Format: 1920x1080, 30 fps, H.264 yuv420p, AAC 192 kb/s stereo 48 kHz, about -14 LUFS integrated, peaks under -1.5 dBFS.

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
  labels top-right; slow zoom on the stills.

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
  The box step appears only as "SCRIPTED REPLAY, NOT FLUX" on the chain card; no box footage is in v1.
- Imagination clips are labelled as a held-out take replayed offline, not a live run.
- The safety card sits over IMG_6397, which is a failed attempt (05:36, layout A, 2 steps, load freeze); the card
  makes no success claim about it.

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

## To do for the final (by 13:00)

- Replace 07a and 07b with the full-chain run (HMI plan and lanes, the box step labelled SCRIPTED REPLAY, NOT FLUX,
  then green, cutter, red pen, each Haiku check) and fill the chain result from NUMBERS.md (`[[NUM:CHAIN_SUCCESS]]`).
- If the cutter step in the chain runs as the take-67 replay, label it SCRIPTED REPLAY, NOT FLUX too.

## Main cut (2:00)

| # | Out | Dur | Shot | Source | Source in to out | Speed | On-screen labels | Caption | Voice-over |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0:00.0 | 6.1 s | 00_cold | `gdrive:liminal/20261018_074537.mp4` | 0.0 to 6.1 s | 1x | FLUX 3 ACTION + LoRA, REAL ARM; REAL SPEED | Red pen this morning, real speed: 2 of 3 graded attempts inserted | +0.2 s: This is FLUX 3 Action, putting a pen away on our robot. On Saturday, the released model failed every attempt. |
| 2 | 0:06.1 | 4.0 s | 01_title | `title card over gdrive:liminal/20261018_091619.mp4 frame at 2.5 s` | still | 1x | - | Card: Liminal / Imagine, Then Act / Black Forest Labs FLUX 3 Action track / ERH Physical AI Hackathon, 17 to 18 Oct 2026 / A released world action model, our two-arm robot, and a weekend of VR teaching. | - |
| 3 | 0:10.1 | 6.0 s | 02_zeroshot | `01_real_robot_runs/20261017_171141/attempt01/video.mp4` | 1.0 to 7.0 s | 1x | ZERO-SHOT, RELEASED CHECKPOINT; REAL SPEED | Released FLUX 3 Action, zero-shot on our arms: 0 of 16 graded attempts succeeded | +0.3 s: Out of the box: zero of sixteen. The shoulder ran to its limit, whatever we asked. |
| 4 | 0:16.1 | 11.2 s | 03_contract | `card (NUMBERS.md section 3)` | still | 1x | - | Card: The model was fine. The contract was not. Released statistics from one community dataset in an older SO-100 convention; same weights with our statistics: shoulder-lift error 19.9 deg to 3.2 deg (offline, 200 frames of our demos). | +0.2 s: The model was fine. The contract was not. It expected an older joint convention. With statistics from our own demos, the lift error fell from twenty degrees to three. No training. |
| 5 | 0:27.3 | 3.5 s | 04a_vr_box | `gdrive:liminal/IMG_6387.MOV` | 11.0 to 14.5 s | 1x | VR TELEOP, META QUEST PRO; REAL SPEED | Then we taught it by VR: 337 teleop takes across four skills, all public on the Hub | +0.3 s: Then we taught it in VR: three hundred and thirty-seven tele-op takes, across four skills, all public. |
| 6 | 0:30.8 | 3.5 s | 04b_vr_pen | `gdrive:liminal/IMG_6374.MOV` | 9.5 to 13.0 s | 1x | VR TELEOP, META QUEST PRO; REAL SPEED | Then we taught it by VR: 337 teleop takes across four skills, all public on the Hub | - |
| 7 | 0:34.3 | 7.0 s | 05a_redpen_4x | `gdrive:liminal/IMG_6396.MOV + 01_real_robot_runs/20261018_052519/attempt02` | 35.0 to 63.0 s | 4x | FLUX 3 ACTION + LoRA, REAL ARM; 4x SPEED | First full FLUX inserts on the real arm: red pen, 3 of 5 in the benchmark block | +0.4 s: One FLUX base, one small LoRA per skill. Here's the red pen LoRA on the real arm, sped up, then at real speed. On the right, its wrist camera. Three of five inserts in our benchmark. |
| 8 | 0:41.3 | 9.0 s | 05b_redpen_1x | `gdrive:liminal/IMG_6396.MOV + 01_real_robot_runs/20261018_052519/attempt02` | 63.0 to 72.0 s | 1x | FLUX 3 ACTION + LoRA, REAL ARM; REAL SPEED | First full FLUX inserts on the real arm: red pen, 3 of 5 in the benchmark block | - |
| 9 | 0:50.3 | 3.0 s | 06a_green | `laptop ws4/single_arm/runs/20261018_083134/attempt01/video.mp4` | 20.0 to 32.0 s | 4x | FLUX 3 ACTION + LoRA, REAL ARM; ROBOT CAMERAS, 4x SPEED | One LoRA per skill, one shared base: green pen 1 of 2, cutter 1 of 6 inserts this morning | +0.3 s: The green pen and the cutter: same base, their own adapters. Each one inserted this morning. |
| 10 | 0:53.3 | 3.0 s | 06b_cutter | `laptop ws4/single_arm/runs/20261018_084210/attempt01/video.mp4` | 38.0 to 50.0 s | 4x | FLUX 3 ACTION + LoRA, REAL ARM; ROBOT CAMERAS, 4x SPEED | One LoRA per skill, one shared base: green pen 1 of 2, cutter 1 of 6 inserts this morning | - |
| 11 | 0:56.3 | 13.6 s | 07a_chain_card | `card (NUMBERS.md 0.1)` | still | 1x | - | Card: Claude plans. FLUX acts. Claude checks. Steps: 1 box arm opens and holds the lid (SCRIPTED REPLAY, NOT FLUX), 2 green pen, 3 yellow cutter, 4 red pen (FLUX 3 Action + LoRA). Footer: step 1 replays a recorded VR demo under the same safety layer. | +0.3 s: For the demo, Claude Opus plans the chain, and Claude Haiku checks every step from the head camera. No LLM output reaches a motor. Step one, the box, is a scripted replay of a recorded demo. Not FLUX. |
| 12 | 1:09.9 | 11.0 s | 07b_cutter_live | `gdrive:liminal/20261018_092520.mp4` | 10.0 to 32.0 s | 2x | FLUX 3 ACTION + LoRA, REAL ARM; 2x SPEED | Live this morning: the cutter LoRA grasps almost every time; inserts are rarer | +0.4 s: Then FLUX puts the objects away. The cutter grasps almost every time. Full inserts are rarer. |
| 13 | 1:20.9 | 10.0 s | 08a_imagine | `gdrive:liminal/00_highlights/imagination_lora5k_imagined_vs_actual_ep101.mp4` | 0.0 to 10.0 s | 1x | FLUX IMAGINATION, HELD-OUT TAKE REPLAYED | FLUX imagines the next 1.4 s. We decode it; its actions stay bit-identical. | +0.3 s: FLUX also predicts the next one point four seconds of video. We decode it, and the actions stay bit-identical. Left, what happened. Right, what FLUX imagined. Fine-tuning made it imagine better, too. |
| 14 | 1:30.9 | 4.0 s | 08b_imagine_cmp | `gdrive:liminal/00_highlights/imagination_b84_vs_lora5k_ep101.png` | still | 1x | FLUX IMAGINATION, OFFLINE | Fine-tuning made it imagine better: zero-shot smears the pen by 1 s ahead, the LoRA keeps it whole | - |
| 15 | 1:34.9 | 11.0 s | 09_knife | `gdrive:liminal/00_highlights/knife_edge_wrist_view_training_vs_now.jpg` | still | 1x | DIAGNOSIS, OFFLINE | After the robot moved, FLUX only held still. One pixel of wrist-camera shift flipped hold and reach, so we align the station and augment the images. | +0.2 s: Our hardest bug: after we moved the robot, FLUX just held still. One pixel of wrist-camera shift flipped hold and reach. So now we align the station, and augment the images. |
| 16 | 1:45.9 | 8.0 s | 10_safety | `card over gdrive:liminal/IMG_6397.MOV (05:36 attempt, not a success)` | 20.0 to 28.0 s | 1x | - | Card: 1 deg per tick, soft limits, 60 % torque cap; keep-out and per-skill workspace; stall and load freeze; start gate; e-stop. Keep-out refused ticks in 15 of 33 benchmark attempts, ended 7, no collision reported. | +0.3 s: Learned or scripted, every command passes one safety layer: one degree per tick, keep-out zones, stall freeze, and a torque cap. |
| 17 | 1:53.9 | 6.1 s | 11_end | `card` | still | 1x | - | Card: github.com/Javgilavi/liminal, huggingface.co/Oskrt adapters and datasets. | +0.5 s: The code, the adapters and the data are all public. Liminal. Imagine, then act. |

Total: 1:60.0

## 60 s cut

| # | Out | Dur | Shot | Source | Source in to out | Speed | On-screen labels | Caption | Voice-over |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0:00.0 | 4.0 s | 00_cold | `gdrive:liminal/20261018_074537.mp4` | 2.1 to 6.1 s | 1x | FLUX 3 ACTION + LoRA, REAL ARM; REAL SPEED | Red pen this morning, real speed: 2 of 3 graded attempts inserted | +0.2 s: This is FLUX 3 Action on our robot. On Saturday, the released model failed every attempt. |
| 2 | 0:04.0 | 3.0 s | 01_title | `title card over gdrive:liminal/20261018_091619.mp4 frame at 2.5 s` | still | 1x | - | Card: Liminal / Imagine, Then Act / Black Forest Labs FLUX 3 Action track / ERH Physical AI Hackathon, 17 to 18 Oct 2026 / A released world action model, our two-arm robot, and a weekend of VR teaching. | - |
| 3 | 0:07.0 | 4.0 s | 02_zeroshot | `01_real_robot_runs/20261017_171141/attempt01/video.mp4` | 2.0 to 6.0 s | 1x | ZERO-SHOT, RELEASED CHECKPOINT; REAL SPEED | Released FLUX 3 Action, zero-shot on our arms: 0 of 16 graded attempts succeeded | +0.6 s: Zero of sixteen, out of the box. |
| 4 | 0:11.0 | 4.0 s | 04_vr_pen | `gdrive:liminal/IMG_6374.MOV` | 9.0 to 13.0 s | 1x | VR TELEOP, META QUEST PRO; REAL SPEED | Then we taught it by VR: 337 teleop takes across four skills, all public on the Hub | +0.3 s: So we fixed its joint contract, and taught it in VR. |
| 5 | 0:15.0 | 5.0 s | 05a_redpen_4x | `gdrive:liminal/IMG_6396.MOV + 01_real_robot_runs/20261018_052519/attempt02` | 40.0 to 60.0 s | 4x | FLUX 3 ACTION + LoRA, REAL ARM; 4x SPEED | First full FLUX inserts on the real arm: red pen, 3 of 5 in the benchmark block | +0.3 s: One FLUX base, one small LoRA per skill. The red pen: three of five inserts. On the right, its wrist camera. |
| 6 | 0:20.0 | 7.0 s | 05b_redpen_1x | `gdrive:liminal/IMG_6396.MOV + 01_real_robot_runs/20261018_052519/attempt02` | 64.5 to 71.5 s | 1x | FLUX 3 ACTION + LoRA, REAL ARM; REAL SPEED | First full FLUX inserts on the real arm: red pen, 3 of 5 in the benchmark block | - |
| 7 | 0:27.0 | 2.0 s | 06a_green | `laptop ws4/single_arm/runs/20261018_083134/attempt01/video.mp4` | 23.0 to 31.0 s | 4x | FLUX 3 ACTION + LoRA, REAL ARM; ROBOT CAMERAS, 4x SPEED | One LoRA per skill, one shared base: green pen 1 of 2, cutter 1 of 6 inserts this morning | +0.3 s: Green pen and cutter: same base, their own adapters. |
| 8 | 0:29.0 | 2.0 s | 06b_cutter | `laptop ws4/single_arm/runs/20261018_084210/attempt01/video.mp4` | 42.0 to 50.0 s | 4x | FLUX 3 ACTION + LoRA, REAL ARM; ROBOT CAMERAS, 4x SPEED | One LoRA per skill, one shared base: green pen 1 of 2, cutter 1 of 6 inserts this morning | - |
| 9 | 0:31.0 | 6.0 s | 07a_chain_card | `card (NUMBERS.md 0.1)` | still | 1x | - | Card: Claude plans. FLUX acts. Claude checks. Steps: 1 box arm opens and holds the lid (SCRIPTED REPLAY, NOT FLUX), 2 green pen, 3 yellow cutter, 4 red pen (FLUX 3 Action + LoRA). Footer: step 1 replays a recorded VR demo under the same safety layer. | +0.1 s: Claude Opus plans, Claude Haiku checks each step. The box step is a scripted replay, not FLUX. |
| 10 | 0:37.0 | 8.0 s | 07b_cutter_live | `gdrive:liminal/20261018_092520.mp4` | 10.0 to 26.0 s | 2x | FLUX 3 ACTION + LoRA, REAL ARM; 2x SPEED | Live this morning: the cutter LoRA grasps almost every time; inserts are rarer | +0.4 s: The cutter grasps almost every time. Inserts are rarer. |
| 11 | 0:45.0 | 8.0 s | 08a_imagine | `gdrive:liminal/00_highlights/imagination_lora5k_imagined_vs_actual_ep101.mp4` | 1.0 to 9.0 s | 1x | FLUX IMAGINATION, HELD-OUT TAKE REPLAYED | FLUX imagines the next 1.4 s. We decode it; its actions stay bit-identical. | +0.3 s: FLUX also imagines the next one point four seconds. We decode it, and the actions stay bit-identical. |
| 12 | 0:53.0 | 7.0 s | 11_end | `card` | still | 1x | - | Card: github.com/Javgilavi/liminal, huggingface.co/Oskrt adapters and datasets. | +0.5 s: Everything is public. Liminal. Imagine, then act. |

Total: 1:00.0
