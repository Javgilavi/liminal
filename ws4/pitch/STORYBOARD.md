# Demo video storyboard and this morning's shot list

Re-centred on the 4-skill chain, Sun 18 Oct about 08:15 CEST. Two cuts from one shoot: the **main cut (2:00)** for
the submission and a **60 s cut** (`60s`) for the 5-minute final. Captions use [NUMBERS.md](NUMBERS.md) values only.
Status: **HAVE** (file exists, path given), **MAKE** (build on the laptop from existing material), **RECORD** (film
this morning, see the shot list).

## Shot list for THIS MORNING (priority order)

Filming rules (all shots): phone on a tripod, landscape, 1080p 30 fps, arm + objects + box in frame, started before
Enter / Approve; HMI screen recording at 1080p. Name each clip with the runner's run-folder id. Failed attempts are
fine: they are labelled as failed. People in frame: ask before publishing. Spots (notes `final_demo_runbook.md`): the
red pen runs at the **pen84 spot**; box, green and cutter at **last night's spot** with the box at P3; align with
`ALIGN=1` before each (wrist offset < 15 px, brightness 0.85 to 1.2).

| P | Shot | When (demo-rehearsal) | Spot | Notes |
|---|---|---|---|---|
| 1 | **Red-pen insert, RAW 5k at horizon 16**, one full insert from reach to release, phone wide + runner video | about 07:00 to 07:45, first block (3 graded attempts, phone on each) | pen84 | Runner records scene and wrist automatically; the phone adds the wide shot. If no new success: use this morning's runner videos (HAVE, below) |
| 2 | **Each skill once**: open the box and hold the lid (scripted replay of demo 59, film it as "not FLUX"); green pen; yellow cutter; red pen with the lid held | box (P3) and cutter about 08:15 to 09:00 after the augmented swap; green about 09:00 to 09:30; red pen lid-held: one check attempt with RAW 5k (expect a hold, labelled failed) unless an augmented red pen lands | last night's, P3 | Through the HMI (one run per goal) so the Haiku check shows; 10 to 20 s of each is enough |
| 3 | **Full chain with the HMI on screen**: Plan, lanes, Approve, the four steps, each Haiku check turning green | about 09:30 to 11:30, 2 to 3 runs, best one kept | last night's, P3 | Screen-record the HMI and run the phone at the same time; this is `[[CLIP:HMI_CHAIN]]` |
| 4 | **Imagination panel live** during the chain (`--imagine lazy`) | with P3; a first take in the pen84 HMI red-pen run at about 07:45 | | Screen recording of the panel; optional, replay clips exist (HAVE) |
| 5 | **E-stop**: spacebar during a step, both arms freeze, the lid stays held, Release, Resume | about 07:45, during the pen84 HMI red-pen run (Release + Resume after) | any | 5 s; only if it fits |
| 6 | **VR teleop b-roll**: Oscar in the Quest Pro, the arm following | sim mode in any 5-minute gap (for example a model swap); real arm only after 12:30 if time allows | any | 10 s; sim mode is fine, caption "sim" |

Slots from demo-rehearsal (07:30 reply); all depend on the Haller being plugged in (not yet at 06:45). Run-folder ids
arrive from demo-rehearsal as clips land. Hard stop for filming: the 12:30 decision; if the chain is unreliable, P3
becomes one HMI clip per skill. After that, cut the video from what exists.

## Main cut (2:00)

| # | Time | Dur | Shot | Caption / voice-over | Source | Status |
|---|---|---|---|---|---|---|
| 1 | 0:00 | 5 s | Haller station photo, title "liminal: imagine, then act" | none, music in | `ws4/assets/photos/xlerobot_station.jpg` | HAVE |
| 2 | 0:05 | 6 s | Zero-shot runaway (red SO-101): the arm never reaches the cups (`60s`, 4 s) | "Released FLUX 3 Action, zero-shot: 0 of 16" | `ws4/single_arm/runs/20261017_171141/attempt01/video.mp4` | HAVE |
| 3 | 0:11 | 8 s | The contract: A to E table builds; lift error 19.9 to 3.2 deg | "A calibration contract, fixed offline, no training" | deck slides "contract", "diagnosis" (Present mode capture) | MAKE |
| 4 | 0:19 | 8 s | VR teleop b-roll + the four datasets as captions (`60s`, 4 s) | "337 VR takes, four skills, one night" | shot P6 | RECORD |
| 5 | 0:27 | 22 s | **Red-pen insert**: reach, grasp, insert, at 1x then 2x; then 3 s each of the green-pen and cutter inserts (`60s`, 12 s) | "First full FLUX inserts on the real arm: red pen 3 of 5, then green pen and cutter" | Drive `liminal/00_highlights/redpen_insert_layoutC_48s_flux_raw5k_h16.mp4` (or `redpen_insert_morning_C_36s.mp4`); green and cutter run folders from demo-rehearsal | HAVE (red pen) / RECORD (green, cutter clips) |
| 6 | 0:49 | 30 s | **Full chain**: HMI plan and lanes, then the box opens and stays held (caption on screen: "scripted replay of a recorded demo, not FLUX"), green, cutter, red pen (FLUX), 2x to 4x with step labels; each Haiku check green (`60s`, 20 s) | "Claude plans and checks; FLUX does the learned skills; one safety layer for both" + chain result `[[NUM:CHAIN_SUCCESS]]` | shots P3 (+ P2 if the chain is not filmed) | RECORD |
| 7 | 1:19 | 14 s | **Imagined vs actual**, then 3 s on the B84 vs 5k comparison sheet (`60s`, 10 s) | "FLUX imagines the next 1.4 s; its actions stay bit-identical. Fine-tuning made it imagine better." | `ws4/imagine/outputs/lora5000_ep101/imagination/imagined_vs_actual.mp4`, `ws4/imagine/outputs/b84_vs_lora5000_ep101_chunk1.png`; live shot P4 if filmed | HAVE |
| 8 | 1:33 | 10 s | **The knife edge**: recording vs now vs blend of the wrist view | "One pixel of wrist view flipped hold and reach. We align the station and augment." | `ws4/single_arm/runs/station_check/station_vs_pen84.jpg` (people in frame: check) | HAVE |
| 9 | 1:43 | 8 s | **Safety**: keep-out freeze on a run, then the e-stop (shot P5) | "Per-skill workspaces, a start gate, a slow zone: nothing collided" | `ws4/single_arm/runs/20261017_210715/attempt01/video.mp4` (keep-out holds a wrong-sign run) + P5 | HAVE + RECORD |
| 10 | 1:51 | 9 s | End card: team, repo, Hub links (`60s`, 3 s) | `[[TEAM:NAME]]`, github.com/Javgilavi/liminal, huggingface.co/Oskrt | deck slide "close" | MAKE |

60 s cut = shots 2 (4 s), 4 (4 s), 5 (12 s), 6 (20 s), 7 (10 s), 10 (3 s) plus 7 s of transitions.

## What already exists (HAVE)

| Material | Path | Notes |
|---|---|---|
| Edited highlight clips (red-pen inserts A 94 s, B 91 s, C 48 s, morning A 69 s and C 36 s; imagination clips and comparison; knife-edge figure; zero-shot strip; sim overlay) | Google Drive `liminal/00_highlights` | ready to cut; file names in NUMBERS.md 0.6b |
| This morning's five red-pen inserts (runner video, scene and wrist, 640x240) | `ws4/single_arm/runs/20261018_051926/attempt01` (A, 94 s), `20261018_052519/attempt02` (B, 91 s), `20261018_052519/attempt03` (C, 48 s, the best for 2x), `20261018_053419/attempt04` (D), `20261018_053815/attempt04` (D) | gitignored, laptop only, backed up in `~/haller_rescue/single_arm_2026-10-18/runs/` |
| Imagined vs actual, held-out takes, B84 and 5k LoRA | `ws4/imagine/outputs/replay_ep10{1,2,3}/`, `ws4/imagine/outputs/lora5000_ep10{1,2,3}/imagination/imagined_vs_actual.mp4` | H.264 1048x340, about 10 s; captions say "held-out take, replayed" |
| B84 vs 5k imagination comparison | `ws4/imagine/outputs/b84_vs_lora5000_ep101_chunk1.png` | slide "imaginebetter" |
| Wrist view: recording vs now | `ws4/single_arm/runs/station_check/station_vs_pen84.jpg`, `ws4/single_arm/runs/wrist_view_training_vs_now.jpg` | people visible |
| Zero-shot failures (Saturday) | `ws4/single_arm/runs/20261017_*` | A0 runaway, A0r-B wrist pinned, A0r-D keep-out hold |
| Sim | `ws4/ttt/results/*.gif`, `ws4/site/img/compare_photo_vs_sim.jpg` | backup only |
| VR demos (head camera) | `~/.cache/huggingface/lerobot/ws4/haller_*/videos/` and on the Hub | people visible in some takes: crop to the table |

## Edit notes

- 1920x1080, 30 fps. Runner videos are 640x240 (two 320x240 halves): show them as picture-in-picture at native size
  or 3x nearest-neighbour; do not stretch to 16:9.
- Every robot clip is a real attempt at real speed unless the caption says 2x or 4x. Failed attempts are labelled.
- Captions: one line, bottom, high contrast, numbers exactly as in NUMBERS.md section 0.
- Export H.264, under 100 MB, upload where the submission form asks (`[[TEAM:VIDEO_URL]]`).
