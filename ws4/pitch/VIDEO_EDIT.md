# Submission video: edit list

Status: **v1, Sun 18 Oct 10:15 CEST**. The full-chain shots (07a to 07c) are placeholders until the chain footage
lands; everything else is final-candidate. Numbers in captions come only from [NUMBERS.md](NUMBERS.md).

| Cut | Drive | Local (laptop, not in the repo) |
|---|---|---|
| Main, 2:00 | `gdrive:liminal/05_video/liminal_2min_v1.mp4` | `~/liminal_video/out/liminal_2min_v1.mp4` |
| 60 s | `gdrive:liminal/05_video/liminal_60s_v1.mp4` | `~/liminal_video/out/liminal_60s_v1.mp4` |

Format: 1920x1080, 30 fps, H.264 yuv420p, silent AAC track (no music), burned-in captions and labels.

## How it is built

`~/liminal_video/cuts.py` holds both shot lists (source, in point, duration, speed, labels, caption);
`build.py <2min|60s> --version vN` renders each shot with ffmpeg (nice 19, 2 threads, one encode at a time) and joins
them; `gfx.py` draws the title, cards and caption overlays; `edl.py` writes the tables below. Phone clips are
downscaled first to `~/liminal_video/proxy/` (IMG_6396 proxy starts at 25.0 s of the original; the source times
below are original times). A full re-render of both cuts takes about 2 minutes. To refine in an editor, cut the same
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
| 20261018_074537.mp4 | 07:45:37 | `20261018_074025/attempt03`, layout C | success, 3 steps | 07c |
| 20261018_092520.mp4 | 09:25:20 | `20261018_092508/attempt01`, cutter | 0 steps logged (grasp and carry visible) | 07b, caption claims grasps only |
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

- Replace 07a to 07c with the full-chain run (HMI plan and lanes, the box step labelled SCRIPTED REPLAY, NOT FLUX,
  then green, cutter, red pen, each Haiku check) and fill the chain result from NUMBERS.md (`[[NUM:CHAIN_SUCCESS]]`).
- If the cutter step in the chain runs as the take-67 replay, label it SCRIPTED REPLAY, NOT FLUX too.

## Main cut (2:00)

| # | Out | Dur | Shot | Source | Source in to out | Speed | On-screen labels | Caption |
|---|---|---|---|---|---|---|---|---|
| 1 | 0:00.0 | 5.0 s | 01_title | `title card over gdrive:liminal/20261018_091619.mp4 frame at 2.5 s` | still | 1x | - | Card: Liminal / Imagine, Then Act / Black Forest Labs FLUX 3 Action track / ERH Physical AI Hackathon, 17 to 18 Oct 2026 / A released world action model, our two-arm robot, and a weekend of VR teaching. |
| 2 | 0:05.0 | 6.0 s | 02_zeroshot | `01_real_robot_runs/20261017_171141/attempt01/video.mp4` | 1.0 to 7.0 s | 1x | ZERO-SHOT, RELEASED CHECKPOINT; REAL SPEED | Released FLUX 3 Action, zero-shot on our arms: 0 of 16 graded attempts succeeded |
| 3 | 0:11.0 | 8.0 s | 03_contract | `card (NUMBERS.md section 3)` | still | 1x | - | Card: The model was fine. The contract was not. Released statistics from one community dataset in an older SO-100 convention; same weights with our statistics: shoulder-lift error 19.9 deg to 3.2 deg (offline, 200 frames of our demos). |
| 4 | 0:19.0 | 4.0 s | 04a_vr_box | `gdrive:liminal/IMG_6387.MOV` | 10.0 to 14.0 s | 1x | VR TELEOP, META QUEST PRO; REAL SPEED | Then we taught it by VR: 337 teleop takes across four skills, all public on the Hub |
| 5 | 0:23.0 | 4.0 s | 04b_vr_pen | `gdrive:liminal/IMG_6374.MOV` | 9.0 to 13.0 s | 1x | VR TELEOP, META QUEST PRO; REAL SPEED | Then we taught it by VR: 337 teleop takes across four skills, all public on the Hub |
| 6 | 0:27.0 | 7.0 s | 05a_redpen_4x | `gdrive:liminal/IMG_6396.MOV + 01_real_robot_runs/20261018_052519/attempt02` | 35.0 to 63.0 s | 4x | FLUX 3 ACTION + LoRA, REAL ARM; 4x SPEED | First full FLUX inserts on the real arm: red pen, 3 of 5 in the benchmark block |
| 7 | 0:34.0 | 9.0 s | 05b_redpen_1x | `gdrive:liminal/IMG_6396.MOV + 01_real_robot_runs/20261018_052519/attempt02` | 63.0 to 72.0 s | 1x | FLUX 3 ACTION + LoRA, REAL ARM; REAL SPEED | First full FLUX inserts on the real arm: red pen, 3 of 5 in the benchmark block |
| 8 | 0:43.0 | 3.0 s | 06a_green | `laptop ws4/single_arm/runs/20261018_083134/attempt01/video.mp4` | 20.0 to 32.0 s | 4x | FLUX 3 ACTION + LoRA, REAL ARM; ROBOT CAMERAS, 4x SPEED | One LoRA per skill, one shared base: green pen 1 of 2, cutter 1 of 6 inserts this morning |
| 9 | 0:46.0 | 3.0 s | 06b_cutter | `laptop ws4/single_arm/runs/20261018_084210/attempt01/video.mp4` | 38.0 to 50.0 s | 4x | FLUX 3 ACTION + LoRA, REAL ARM; ROBOT CAMERAS, 4x SPEED | One LoRA per skill, one shared base: green pen 1 of 2, cutter 1 of 6 inserts this morning |
| 10 | 0:49.0 | 10.0 s | 07a_chain_card | `card (NUMBERS.md 0.1)` | still | 1x | - | Card: Claude plans. FLUX acts. Claude checks. Steps: 1 box arm opens and holds the lid (SCRIPTED REPLAY, NOT FLUX), 2 green pen, 3 yellow cutter, 4 red pen (FLUX 3 Action + LoRA). Footer: step 1 replays a recorded VR demo under the same safety layer. |
| 11 | 0:59.0 | 14.0 s | 07b_cutter_live | `gdrive:liminal/20261018_092520.mp4` | 10.0 to 38.0 s | 2x | FLUX 3 ACTION + LoRA, REAL ARM; 2x SPEED | Live this morning: the cutter LoRA grasps almost every time; inserts are rarer |
| 12 | 1:13.0 | 6.0 s | 07c_redpen_am | `gdrive:liminal/20261018_074537.mp4` | 0.0 to 6.0 s | 1x | FLUX 3 ACTION + LoRA, REAL ARM; REAL SPEED | Red pen in this morning's rehearsal: 2 of 3 graded attempts inserted |
| 13 | 1:19.0 | 9.8 s | 08a_imagine | `gdrive:liminal/00_highlights/imagination_lora5k_imagined_vs_actual_ep101.mp4` | 0.0 to 9.8 s | 1x | FLUX IMAGINATION, HELD-OUT TAKE REPLAYED | FLUX imagines the next 1.4 s. We decode it; its actions stay bit-identical. |
| 14 | 1:28.8 | 4.2 s | 08b_imagine_cmp | `gdrive:liminal/00_highlights/imagination_b84_vs_lora5k_ep101.png` | still | 1x | FLUX IMAGINATION, OFFLINE | Fine-tuning made it imagine better: zero-shot smears the pen by 1 s ahead, the LoRA keeps it whole |
| 15 | 1:33.0 | 10.0 s | 09_knife | `gdrive:liminal/00_highlights/knife_edge_wrist_view_training_vs_now.jpg` | still | 1x | DIAGNOSIS, OFFLINE | After the robot moved, FLUX only held still. One pixel of wrist-camera shift flipped hold and reach, so we align the station and augment the images. |
| 16 | 1:43.0 | 8.0 s | 10_safety | `card over gdrive:liminal/IMG_6397.MOV (05:36 attempt, not a success)` | 20.0 to 28.0 s | 1x | - | Card: 1 deg per tick, soft limits, 60 % torque cap; keep-out and per-skill workspace; stall and load freeze; start gate; e-stop. Keep-out refused ticks in 15 of 33 benchmark attempts, ended 7, no collision reported. |
| 17 | 1:51.0 | 9.0 s | 11_end | `card` | still | 1x | - | Card: github.com/Javgilavi/liminal, huggingface.co/Oskrt adapters and datasets. |

Total: 2:00.0

## 60 s cut

| # | Out | Dur | Shot | Source | Source in to out | Speed | On-screen labels | Caption |
|---|---|---|---|---|---|---|---|---|
| 1 | 0:00.0 | 3.0 s | 01_title | `title card over gdrive:liminal/20261018_091619.mp4 frame at 2.5 s` | still | 1x | - | Card: Liminal / Imagine, Then Act / Black Forest Labs FLUX 3 Action track / ERH Physical AI Hackathon, 17 to 18 Oct 2026 / A released world action model, our two-arm robot, and a weekend of VR teaching. |
| 2 | 0:03.0 | 4.0 s | 02_zeroshot | `01_real_robot_runs/20261017_171141/attempt01/video.mp4` | 2.0 to 6.0 s | 1x | ZERO-SHOT, RELEASED CHECKPOINT; REAL SPEED | Released FLUX 3 Action, zero-shot on our arms: 0 of 16 graded attempts succeeded |
| 3 | 0:07.0 | 4.0 s | 04_vr_pen | `gdrive:liminal/IMG_6374.MOV` | 9.0 to 13.0 s | 1x | VR TELEOP, META QUEST PRO; REAL SPEED | Then we taught it by VR: 337 teleop takes across four skills, all public on the Hub |
| 4 | 0:11.0 | 5.0 s | 05a_redpen_4x | `gdrive:liminal/IMG_6396.MOV + 01_real_robot_runs/20261018_052519/attempt02` | 40.0 to 60.0 s | 4x | FLUX 3 ACTION + LoRA, REAL ARM; 4x SPEED | First full FLUX inserts on the real arm: red pen, 3 of 5 in the benchmark block |
| 5 | 0:16.0 | 7.0 s | 05b_redpen_1x | `gdrive:liminal/IMG_6396.MOV + 01_real_robot_runs/20261018_052519/attempt02` | 64.5 to 71.5 s | 1x | FLUX 3 ACTION + LoRA, REAL ARM; REAL SPEED | First full FLUX inserts on the real arm: red pen, 3 of 5 in the benchmark block |
| 6 | 0:23.0 | 2.0 s | 06a_green | `laptop ws4/single_arm/runs/20261018_083134/attempt01/video.mp4` | 23.0 to 31.0 s | 4x | FLUX 3 ACTION + LoRA, REAL ARM; ROBOT CAMERAS, 4x SPEED | One LoRA per skill, one shared base: green pen 1 of 2, cutter 1 of 6 inserts this morning |
| 7 | 0:25.0 | 2.0 s | 06b_cutter | `laptop ws4/single_arm/runs/20261018_084210/attempt01/video.mp4` | 42.0 to 50.0 s | 4x | FLUX 3 ACTION + LoRA, REAL ARM; ROBOT CAMERAS, 4x SPEED | One LoRA per skill, one shared base: green pen 1 of 2, cutter 1 of 6 inserts this morning |
| 8 | 0:27.0 | 6.0 s | 07a_chain_card | `card (NUMBERS.md 0.1)` | still | 1x | - | Card: Claude plans. FLUX acts. Claude checks. Steps: 1 box arm opens and holds the lid (SCRIPTED REPLAY, NOT FLUX), 2 green pen, 3 yellow cutter, 4 red pen (FLUX 3 Action + LoRA). Footer: step 1 replays a recorded VR demo under the same safety layer. |
| 9 | 0:33.0 | 8.0 s | 07b_cutter_live | `gdrive:liminal/20261018_092520.mp4` | 10.0 to 26.0 s | 2x | FLUX 3 ACTION + LoRA, REAL ARM; 2x SPEED | Live this morning: the cutter LoRA grasps almost every time; inserts are rarer |
| 10 | 0:41.0 | 6.0 s | 07c_redpen_am | `gdrive:liminal/20261018_074537.mp4` | 0.0 to 6.0 s | 1x | FLUX 3 ACTION + LoRA, REAL ARM; REAL SPEED | Red pen in this morning's rehearsal: 2 of 3 graded attempts inserted |
| 11 | 0:47.0 | 8.0 s | 08a_imagine | `gdrive:liminal/00_highlights/imagination_lora5k_imagined_vs_actual_ep101.mp4` | 1.0 to 9.0 s | 1x | FLUX IMAGINATION, HELD-OUT TAKE REPLAYED | FLUX imagines the next 1.4 s. We decode it; its actions stay bit-identical. |
| 12 | 0:55.0 | 5.0 s | 11_end | `card` | still | 1x | - | Card: github.com/Javgilavi/liminal, huggingface.co/Oskrt adapters and datasets. |

Total: 1:00.0
