# Demo video storyboard

Draft, Sat 17 Oct 2026 about 22:15 CEST. Two cuts from one shoot: the **main cut (2:00)** for the submission and a
**60 s cut** for the 5-minute final (marked `60s` in the table). Numbers on captions come from
[NUMBERS.md](NUMBERS.md). Status: **HAVE** = the file exists now (path given), **MAKE** = can be built on the laptop
from existing data, **RECORD** = must be filmed Sunday morning.

Where the files are:
- Runner attempt videos (`ws4/single_arm/runs/<session>/attempt<NN>/video.mp4`, 640x240, scene | wrist side by side,
  with a task and time caption burned in) are **gitignored**: they are on the laptop only. Keep them out of git; cut
  from them.
- VR demo footage: the LeRobot datasets in `~/.cache/huggingface/lerobot/ws4/haller_put_the_red_pen_in_its_slot_in_the_cardboard_box/`
  (104 takes) and `.../haller_take_the_red_pen_out_of_its_slot/` (11 takes), `videos/observation.images.head/` and
  `.../right_wrist/`, 640x480 at 30 fps. **People are in frame** in some head-camera takes (Oscar and passers-by):
  pick takes without faces, or crop to the table (crop `x 160..640, y 110..480` removes the operator's face in
  file-026) or blur, and ask anyone visible before publishing.
- Sim clips: `ws4/ttt/results/` (GIFs and JPGs are in git; the MP4s are local only).

## Main cut (2:00)

| # | Time | Dur | Shot | Caption / voice-over | Source | Status |
|---|---|---|---|---|---|---|
| 1 | 0:00 | 5 s | Haller station photo, slow push-in; title "liminal: imagine, then act" | none, music in | `ws4/assets/photos/xlerobot_station.jpg` | HAVE |
| 2 | 0:05 | 8 s | Zero-shot runaway, red SO-101: shoulder drives up, wrist folds, operator stops it (`60s`) | "FLUX 3 Action, released checkpoint, zero-shot: 0 of 16 graded attempts" | `ws4/single_arm/runs/20261017_171141/attempt01/video.mp4` (12.5 s, variant A0) | HAVE |
| 3 | 0:13 | 6 s | Zero-shot on the Haller with our quantiles: reaches toward the cup, wrist flex pins, gripper curls back (4x speed) | "Even re-normalized: the wrist runs to its limit" | `ws4/single_arm/runs/20261017_204828/attempt01/video.mp4` (60 s, A0r-B), use 10 to 34 s at 4x | HAVE |
| 4 | 0:19 | 12 s | The A to E table builds row by row; B and D light up (`60s`, 6 s) | "Root cause: a calibration contract. Our own quantiles: lift error 19.9 to 3.2 deg, direction 80 %. No training." | Slide "The contract" of the deck, screen-captured in Present mode | MAKE (from the deck) |
| 5 | 0:31 | 6 s | Optional plot: commanded shoulder lift over one A0 attempt vs the package's q01..q99 band (85 to 184 deg) | "FLUX saw an impossible robot" | `trace.npz` of `20261017_171141/attempt01` + `ws4/flux/checkpoint_facts.md:38`; small matplotlib script | MAKE (15 min, skip if short on time) |
| 6 | 0:37 | 10 s | Oscar teleoperating in the Quest Pro, robot copying the hand (third-person, phone) (`60s`, 5 s) | "Teaching by VR: 96 demos in one evening" | none yet | RECORD (2 takes, landscape, 1080p) |
| 7 | 0:47 | 6 s | Three VR demos from the head camera in a 3-up grid, 2x speed | "Recorded in FLUX's exact contract: 30 fps, head + wrist, LeRobot degrees" | dataset head videos, pick 3 takes with no faces from different layouts (A to D) | HAVE (pick and crop) |
| 8 | 0:53 | 5 s | LoRA eval-loss curve, one line | "One LoRA overnight on one RTX PRO 6000" | trainer log on `real-train-6000` | MAKE after the run (`[[NUM:LORA_EVAL_LOSS]]`) |
| 9 | 0:58 | 20 s | **Post-training success**: the Haller grasps the pen and puts it in its slot, real time. Phone wide shot, with the runner's scene and wrist video picture-in-picture (`60s`, 15 s) | "After post-training: `[[NUM:LORA_PEN_SUCCESS]]` on the same task" | phone + `runs/<sunday>/attemptNN/video.mp4` | RECORD |
| 10 | 1:18 | 15 s | **Imagined vs actual**: actual left, FLUX's decoded imagination right, chunk boundaries and surprise trace (`60s`, 10 s) | "FLUX imagines the next 1.4 s with every chunk. We decode it, and its actions stay bit-identical." | `ws4/imagine/outputs/replay_ep101/imagination/imagined_vs_actual.mp4` (held-out take 101 replayed, H.264 1048x340, 9.9 s; also ep102, ep103); optional live attempt via `compose.render_run` | HAVE (replay; caption "held-out take, replayed"), RECORD live optional |
| 11 | 1:33 | 14 s | **Claude chains it**: HMI screen capture, goal typed, plan lanes appear, approve; robot does pen out then pen in; Haiku check turns green (`60s`, 12 s) | "Claude plans and checks. It never touches a motor." | OBS / screen recording of `python -m orchestrator hmi` + phone | RECORD (`[[CLIP:HMI_CHAIN]]`) |
| 12 | 1:47 | 7 s | **Safety holds**: the keep-out freezes a wrong-sign zero-shot run at the mast and base (2x speed), then an e-stop on the spacebar | "Every command: soft limits, 1 deg per tick, FK keep-out, stall and load freeze, 60 % torque cap" | `ws4/single_arm/runs/20261017_210715/attempt01/video.mp4` (A0r-D, keep-out refused 198 mast and 96 base ticks, `meta.json`) + e-stop RECORD | HAVE + RECORD |
| 13 | 1:54 | 6 s | End card: team names, repo URL, "FLUX 3 Action by Black Forest Labs" (`60s`, 3 s) | `[[TEAM:NAME]]`, `[[TEAM:MEMBERS]]`, github.com/Javgilavi/liminal | deck end slide | MAKE |

60 s cut = shots 2, 4 (6 s), 6 (5 s), 9 (15 s), 10 (10 s), 11 (12 s), 13 (3 s) plus 4 s of transitions.

### Optional B-roll (5-minute final, Q&A backup)

| Shot | Source | Status |
|---|---|---|
| Sim: scripted demo, "put the block in the cup" | `ws4/ttt/results/block_cup_demo_ep0.gif` (11.3 s) | HAVE |
| Sim: released checkpoint knocks the cup over | `ws4/ttt/results/block_cup_a0_seed9000000_3x.gif` (8.1 s) | HAVE |
| Sim: TTT / GRU stage tracker overlaid on a FLUX rollout | `ws4/ttt/results/flux_sim_so101sim_A0r_ep0_overlay.mp4` (1280x590, 117.6 s; use 20 s) | HAVE (local) |
| Sim: progress monitor on stand-in lid cycles | `ws4/ttt/results/inference_lid_cycles_4-100001.mp4` (24 s) | HAVE (local) |
| Sim vs photo of the station | `ws4/site/img/compare_photo_vs_sim.jpg` | HAVE |
| Sim LoRA success, same 20 layouts | `[[NUM:SIM_LORA]]` rollout video from `ws4-sim-4500b` | PENDING |
| Teammates' workstreams | `[[TEAM:WORKSTREAMS]]` clips from WS1, WS2, WS3, WS5 | ASK the team |

## Shoot list for Sunday morning (in this order)

1. **Post-training attempts** (shot 9): the graded LoRA attempts are filmed anyway by the runner. Add a phone on a
   tripod, landscape, 1080p 30 fps, wide enough for arm, pen and box, fixed for all attempts. Start the phone before
   pressing Enter, so every graded attempt has a matching wide shot. Name files with the runner's session id.
2. **Imagined vs actual** (shot 10): run at least one successful attempt with imagination on and render it with
   `compose.render_run`; also keep one failed attempt with a visible divergence (it makes the point better).
3. **HMI chain** (shot 11): screen recording at 1080p plus the phone. Two full chains, keep the cleaner one.
4. **VR teleop** (shot 6): 2 short takes of Oscar in the headset with the robot following. Sim mode is fine if the
   real arm is busy, but say "sim" on the caption if so.
5. **E-stop** (shot 12): spacebar during a ramp, the arm holds, Release and Resume.

Do not stage anything: every robot clip in the video is a real, unedited-speed attempt unless the caption says "2x"
or "4x". Failed attempts are labelled as failed.

## Edit notes

- 1920x1080, 30 fps. Runner videos are 640x240 (two 320x240 halves): scale 3x with nearest-neighbour or show them
  as a picture-in-picture at native size; do not stretch to 16:9.
- Captions: one line, bottom, high contrast, numbers exactly as in NUMBERS.md.
- Music low; no voice-over required, but a 2-minute voice-over follows the 3-minute pitch script in
  [OUTLINE.md](OUTLINE.md) beats P1, D1, A1, R1, I1.
- Export H.264, under 100 MB, and upload where the submission form asks (`[[TEAM:VIDEO_URL]]`).
