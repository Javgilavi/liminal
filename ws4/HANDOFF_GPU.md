# GPU handoff: what to run on the 3x RTX PRO 6000 box (Sat 17 to Sun 18 Oct)

You are the only person on the team who can reach the 3x RTX PRO 6000 Blackwell (96 GB) machine. This page
is everything WS4 needs from it, in order. Oscar runs the robot. You run the GPUs and send results back.

**Deadline: Sunday 14:00 CEST.** Track judging is 14:30 to 15:30, with a live demo that needs the policy
server running.

## What we are doing, in five lines

- Our robot is an XLeRobot station: two SO-101 arms on a box, a RealSense head camera, a wrist camera on each
  arm, and soft fin-ray fingers.
- The plan runs BFL's **FLUX 3 Action SO-101 checkpoint once per arm**, then adds a **task LoRA** trained on
  our own demos.
- FLUX's **predicted video** ("imagination") is the showcase.
- Your box does three jobs:
  1. serve the policy to the robot over the network;
  2. train the LoRAs overnight;
  3. run the imagination checks.
- Decisions and reasons are in `ws4/PLAN.md`. You do not need to read the research reports to do the steps
  below.

Each step lists its rough time and what to **send back** (paste into the shared chat, or put the files where
Oscar can fetch them).

---

## 0. Before you start (Oscar and you, about 15 min)

- [ ] **Get the code onto the box.** `ws4/` is on the `ws4-handoff` branch of the team repo (you need read
  access to `Javgilavi/liminal`):
  `git clone -b ws4-handoff git@github.com:Javgilavi/liminal.git ~/liminal`

  Run `git -C ~/liminal pull` whenever Oscar says the code changed.
- [ ] **Hugging Face access.** With the HF account you will use on the box, open each repo and click
  "Agree and access" (each one is gated separately):
  - <https://huggingface.co/black-forest-labs/flux-3-action-so101>
  - <https://huggingface.co/black-forest-labs/flux-3-action-base>
  - <https://huggingface.co/black-forest-labs/flux-3-action-droid>

  Then create a read token and run `export HF_TOKEN=hf_...` in every shell. The public datasets need no
  gate, but the token also avoids anonymous rate limits.
- [ ] **A way for the robot to reach the policy server.** The server listens only on the box's
  `127.0.0.1:8765`. Pick one:
  - (a) Add Oscar's SSH public key to `~/.ssh/authorized_keys` on the box. Oscar then runs the tunnel from
    the robot laptop (`flux/GPU_RUNBOOK.md` §3). This is the simplest.
  - (b) Install Tailscale on the box and share that machine with Oscar's tailnet (`oscardvs@github`).
  - (c) If neither is allowed, open a reverse tunnel from the box to a machine Oscar can reach. Agree on it
    with Oscar.

  Tell Oscar which one, and the host and user to use.
- [ ] Check the box has `ffmpeg` with AV1 decode (`ffmpeg -decoders | grep -i av1`) and about **250 GB free**
  (200 GB is workable with the cleanups in the disk budget below).

## 1. Bootstrap (about 45 to 60 min, mostly unattended)

```bash
export HF_TOKEN=hf_...            # from step 0
bash ~/liminal/ws4/cloud/bootstrap.sh
```

It checks your HF access before downloading anything, starts the ~40 GB weight download in the background,
builds two Python environments (`~/ws4/lerobot`, `~/ws4/flux-action`) and runs GPU checks. It is safe to
re-run. Log: `~/ws4/bootstrap.log`.

**Check** in the log:
- `sm_120 in torch arch list: True`
- `natten 3D op ok`
- `lerobot Flux3Policy import ok` (LeRobot env)
- `prefetch ok`

**Send back:** the output of
`grep -E "lerobot @|flux-action @|torch tag|natten|sm_|FAILED|WARNING|prefetch" ~/ws4/bootstrap.log`.

In every new shell after this:

```bash
export WS4_ROOT=$HOME/ws4 PYTHONPATH=$HOME/liminal/ws4 HF_TOKEN=hf_... DATA_ROOT=$HOME/data
source $WS4_ROOT/lerobot/.venv/bin/activate
```

## 2. First checks on GPU 0 (about 30 min)

Follow **`ws4/flux/GPU_RUNBOOK.md` §0 to §2**: tests, package contents, latency/VRAM, and a cross-check
against BFL's own code.

**Send back:**
- `~/so101_quantiles.json`
- `~/base_heads.txt`. This also settles whether the EE50 heads exist; that track is parked for this weekend,
  but the answer is useful.
- the five `~/lat_*.json`
- the cross-check verdict

Then follow **`ws4/imagine/GPU_RUNBOOK.md` §1 to §5**. For §2 ("a dataset to replay"), once step 3 has
finished, use a converted public XLeRobot episode: `$DATA_ROOT/per_arm/xlerobot_clean_table_left`, keys `scene` and
`wrist`. It is the same camera layout as our robot.

**Send back:** every `report.json` it writes, plus the imagined-vs-recorded clip from §5. That clip is
demo-video material.

## 3. Public data, in parallel with step 2 (about 20 min, CPU and network only)

```bash
bash ~/liminal/ws4/data/fetch_public.sh            # 9.5 GB: zonglin11 + cortx fruit sorting
```

What it does:
- downloads two public XLeRobot-style datasets at pinned revisions;
- converts each to per-arm format (6 joints in degrees, cameras `scene` and `wrist`, task text "... with the
  left arm") and to 12-D format;
- merges the per-arm sets into `$DATA_ROOT/public_per_arm`, about 200 arm-episodes and 6.5 h.

Conversion only rewrites metadata and hard-links the videos, so it takes minutes. `EXTRA=1` adds two more
datasets, which are kept separate because their videos cannot be merged.

**Check:** the last lines show `vs flux3_so101: convertible` with no issues listed.

**Send back:** the script's final output, about 15 lines.

If the merge step fails (LeRobot's aggregate has not been run on this data yet), send the error, and use a
single per-arm set for LoRA P instead, e.g. `$DATA_ROOT/per_arm/bimanual-fruit-sorting_left`.

## 4. LoRA P: public data only, GPU 1 (start as soon as step 3 is done)

This is an early adapter that may teach FLUX the "two arms in view, head camera on a mast" viewpoint, which
it has never seen. It also gives a third data point for the judges' "justify your time split" question.

```bash
cd $WS4_ROOT/lerobot && export CUDA_VISIBLE_DEVICES=1
python -m lerobot.scripts.lerobot_train --config_path=examples/flux3/lora.json \
  --policy.path=black-forest-labs/flux-3-action-so101 --policy.device=cuda \
  --dataset.repo_id=local/public_per_arm --dataset.root=$DATA_ROOT/public_per_arm \
  --steps=6000 --save_freq=500 --output_dir=outputs/lora_P
```

After about 50 microsteps, read the step time in the log. Lower `--steps` if the run would not finish by
about 19:00, because GPU 1 is needed for LoRA B tonight.

**Send back:**
- the step time and peak memory;
- when the run finishes, the checkpoint path, so the server can load it as
  `--package P=$WS4_ROOT/lerobot/outputs/lora_P/checkpoints/last/pretrained_model`.

## 5. Policy server for the robot, GPU 0 (from mid-afternoon until the demo)

Follow **`ws4/flux/GPU_RUNBOOK.md` §3**, using the connection you set up in step 0.

```bash
export CUDA_VISIBLE_DEVICES=0 TOKEN=$(openssl rand -hex 16)
python -m flux.remote --port 8765 --token $TOKEN --package so101=black-forest-labs/flux-3-action-so101
```

Add packages as they appear: Oscar's re-normalized `left`/`right` packages from runbook §4, and `P`, `A`, `B`.
Run it inside `tmux` so it survives disconnects.

**Send Oscar:** the `$TOKEN` (privately), plus the host and port for the tunnel.

Keep this server up through Sunday's demo. If it dies, restarting it is the top priority.

## 6. Overnight: LoRA A vs LoRA B (about 19:00, GPUs 1 and 2)

Oscar records our own demos in the afternoon and pushes them to the Hub (readable by your HF account). Oscar
will tell you two things:
- the repo id (`ORG/xlerobot_<task>`);
- which base to use: the package (`black-forest-labs/flux-3-action-so101`) or a re-normalized one from
  runbook §4.

Use the **same base and the same `--steps`** for A and B.

**6a. Get our data.**

```bash
OWN=ORG/xlerobot_task                          # from Oscar
BASE=black-forest-labs/flux-3-action-so101     # or the re-normalized package path, from Oscar
hf download $OWN --repo-type dataset --local-dir $DATA_ROOT/own
python -I ~/liminal/ws4/data/inspect_dataset.py $DATA_ROOT/own | head -5    # note robot_type and frames
```

If the demos were recorded bimanually (12 joints, cameras `head`, `left_wrist`, `right_wrist`), download into
`$DATA_ROOT/own_raw` instead, then split per arm and merge the two halves into `$DATA_ROOT/own`:

```bash
for s in left right; do python -I ~/liminal/ws4/data/convert_bimanual.py $DATA_ROOT/own_raw \
  $DATA_ROOT/own_$s --to per_arm --side $s; done
python -I ~/liminal/ws4/data/merge_datasets.py $DATA_ROOT/own $DATA_ROOT/own_left $DATA_ROOT/own_right
```

**6b. LoRA A: our data only, GPU 2.**

```bash
cd $WS4_ROOT/lerobot && CUDA_VISIBLE_DEVICES=2 python -m lerobot.scripts.lerobot_train \
  --config_path=examples/flux3/lora.json --policy.path=$BASE --policy.device=cuda \
  --dataset.repo_id=local/own --dataset.root=$DATA_ROOT/own \
  --steps=8000 --save_freq=500 --output_dir=outputs/lora_A
```

**6c. LoRA B: our data plus about 30% public, GPU 1.** Do not merge in all of the public data: the public
set is about 6x bigger than ours.

1. Keep a fraction `f` of the public episodes, where `f = 0.43 × own_frames / public_frames`. The public
   frames are printed at the end of step 3. Example: 1 h of own data gives `f` of about 0.07.

   ```bash
   lerobot-edit-dataset --repo_id local/public_per_arm --root $DATA_ROOT/public_per_arm \
     --new_root $DATA_ROOT/public_split --operation.type split --operation.splits '{"mix": 0.07, "rest": 0.93}'
   ```

2. Check, then merge, our data with that slice:

   ```bash
   python -I ~/liminal/ws4/data/merge_datasets.py $DATA_ROOT/own_plus_public \
     $DATA_ROOT/own $DATA_ROOT/public_split/mix --check-only
   python -I ~/liminal/ws4/data/merge_datasets.py $DATA_ROOT/own_plus_public \
     $DATA_ROOT/own $DATA_ROOT/public_split/mix --repo-id local/own_plus_public
   ```

   If the check says `robot_type` differs, re-run step 3 with `ROBOT_TYPE_ARM=<own value>` into a fresh
   `DATA_ROOT`. If it says the codec or frame size differs, tell Oscar: our recording settings need to match
   the public data (AV1, 640x480).

3. Train B:

   ```bash
   cd $WS4_ROOT/lerobot && CUDA_VISIBLE_DEVICES=1 python -m lerobot.scripts.lerobot_train \
     --config_path=examples/flux3/lora.json --policy.path=$BASE --policy.device=cuda \
     --dataset.repo_id=local/own_plus_public --dataset.root=$DATA_ROOT/own_plus_public \
     --steps=8000 --save_freq=500 --output_dir=outputs/lora_B
   ```

Set `--steps` so both runs end by about 06:00, using the step time from LoRA P.

The 12-D bimanual LoRA (runbook §6, GPU 2) replaces A **only if** Oscar says there are 50 or more bimanual
demos. Ask Oscar.

**Send back by 07:00:**
- the final loss for each run;
- the checkpoint paths (`last` and `pretrained_model_ema`);
- any crash logs.

Then add `--package A=... --package B=...` to the server in step 5, so Oscar can compare A, B and P on the
robot.

## 7. Sunday

- Keep GPU 0 serving. On GPUs 1 and 2, run whatever evaluation Oscar asks for. That is usually offline action
  error on our held-out episodes, plus imagination clips (`imagine/GPU_RUNBOOK.md` §6 and §7) for the video.
- From 12:30, change nothing on the box except restarting the server.

## GPU plan at a glance

| GPU | Sat afternoon | Sat night | Sun |
|---|---|---|---|
| 0 | checks (step 2), then policy server (step 5) | policy server | policy server for tests and the 14:30 demo |
| 1 | LoRA P (step 4) | LoRA B (step 6) | evaluation |
| 2 | data (step 3, CPU), imagination checks | LoRA A, or the 12-D LoRA (step 6) | evaluation |

## Disk budget

| What | GB | How we know |
|---|---|---|
| FLUX weights: SO-101 package 13.9 + base 25.4 (DiT, video VAE, Qwen text encoder) | 39 | exact Hub file sizes |
| Two Python envs (torch 2.10 cu128 each) + uv cache | about 20 | estimate |
| Public data: raw 9.5, merged per-arm copy about 13, LoRA B split about 13 | about 35 | Hub sizes + estimate |
| Our demos plus per-arm and merged copies | about 15 | estimate (public AV1 data is about 1.1 GB per camera-hour) |
| LoRA checkpoints for P, A, B | 20 to 70 | UNVERIFIED: LeRobot keeps every checkpoint, 16 per run at `--save_freq=500` |
| HF download cache, logs, imagination clips | about 15 | estimate |
| **Total** | **about 145 to 195** | |

To stay under 200 GB: use `--save_freq=1000`, delete `$DATA_ROOT/public_split/rest` after step 6c, and delete
old `checkpoints/0*` folders once a later one is evaluated. `EXTRA=1` in step 3 adds about 10 GB.

## If something breaks

| Symptom | Fix |
|---|---|
| NATTEN errors at runtime | `export F3_NATTEN_BACKEND=flex-fna` (slower, pure PyTorch). See `flux/GPU_RUNBOOK.md` §2. |
| `NO ACCESS` or 401 on `black-forest-labs/...` | The token's account has not clicked "Agree and access" on that repo (step 0). |
| A Python step in `fetch_public.sh` hangs after printing | It is killed after 30 min. Outputs are written before the hang. Re-run the script; it skips finished work. |
| Out of memory in a LoRA run | Keep batch 2; move the text encoder to the CPU (LeRobot doc, about 8.3 GiB saved). Do not run two trainings on one GPU. |
| Latency above about 1.2 s per chunk | Try the guidance-1.0 override from `flux/GPU_RUNBOOK.md` §2, but only after Oscar checks motion quality on the robot. |

Owners, if you need to ask about one area (through Oscar):

| Area | Owner and where |
|---|---|
| Data steps | ws4-datasets: `ws4/research/datasets.md` |
| Serving and LoRA | ws4-flux-dual: `ws4/research/flux_dual_arm.md` |
| Imagination | ws4-imagine: `ws4/research/imagination.md` |
| Bootstrap | the WS4 orchestrator: `ws4/cloud/` |
