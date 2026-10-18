# Submission checklist, Sun 18 Oct 2026 (deadline 14:00 CEST, track judging 14:30)

The brief asks for three things: the **GitHub repository**, the **slides**, the **demo video**. Everything below is
in service of those three, in time order. Tick each line; the owner is who does it, not who is told.

## Timeline

| By (CEST) | What | Owner |
|---|---|---|
| 08:00 to 08:30 | Augmented LoRA gates (box, cutter about 08:00; green about 08:25) and cutter offline numbers into NUMBERS.md section 0 | ws4-datasets, then pitch kit |
| 07:00 to 11:30 | Filming per [STORYBOARD.md](STORYBOARD.md): red-pen insert 07:00 to 07:45, e-stop and first imagination take about 07:45, box and cutter 08:15 to 09:00, green 09:00 to 09:30, full chain 09:30 to 11:30 | demo-rehearsal + Oscar |
| **12:30** | **Demo decision** (runbook): full chain, or skills one by one. Freeze the results | Oscar + orchestrator |
| 12:45 | `python ws4/eval/trial_log.py summary --markdown` pasted into README_submission.md "Results"; fill every `[[NUM:...]]` (table below) | pitch kit / orchestrator |
| 13:00 | Video cut from STORYBOARD.md (2:00 main, 60 s for the final), exported H.264 under 100 MB | Oscar or a teammate |
| 13:15 | README on main (section A) committed and pushed | orchestrator batches, Oscar pushes |
| 13:30 | Video uploaded, link into README and the deck's close slide | Oscar |
| 13:40 | Slides shared and exported (section B) | Oscar |
| 13:50 | Final checks (section D) | orchestrator |
| **13:55** | **Submit the form**: repo URL, slides link or PDF, video link | Oscar |
| 14:15 | Warm the FLUX server and align the station for the live demo (section E) | Oscar |

## A. Repository README on main

- [ ] **ws4/ is on main.** At the last fetch, `origin/main` had no `ws4/` directory (it has the team's `src/`,
      `configs/`, `sim/`, `experiments/`, `integrations/`). Merge `ws4-handoff` into main, or every README link
      breaks.
- [ ] Copy `ws4/pitch/README_submission.md` to the repo root as `README.md` (it is written for the root), merging in
      the team's existing README sections for WS1, WS2, WS3 and WS5 (`[[TEAM:WORKSTREAMS]]`). Delete the HTML comment
      at its top.
- [ ] Every link opens on GitHub (ws4 paths, Hub datasets, the LoRA, `ws4/LORA_HOWTO.md`).
- [ ] The results table is the fresh `trial_log.py summary --markdown`.
- [ ] Team name and members filled.

## B. Slides

- [ ] Hosted deck: https://claude.ai/artifact/FsptG2MfRYqkmaAqL4EDQm (19 slides). It is **private until shared**: share
      it from the page's Share menu, or export PDF / PPTX from the page and attach that.
- [ ] Every `[[...]]` on the slides replaced or the element deleted (results, orchestrator chain pill, team, cover and
      close pills).
- [ ] The box step is called a scripted replay everywhere (slides, README, video captions), never FLUX.
- [ ] Cutter count reconciled (trial log 1/6 vs the orchestrator's 0/5 inserts at 09:00) before it is quoted.
- [ ] Chain stated with its caveats: unguarded demo mode, red pen placed by the green-pen LoRA.
- [ ] Video v3 from video-cut (Drive `liminal/05_video`, about 12:50) checked against NUMBERS.md 0.1b captions.
- [ ] 3-minute order: cover, zeroshot, contract (optional), diagnosis, architecture, orchestrator, teach (optional),
      heldout, results, knifeedge, imagination, imaginebetter (optional), realworld, close. Skip safety, memory, sim,
      team and next in the 3-minute pitch (their notes say FINAL ONLY).
- [ ] People visible in two images (knife-edge wrist view, the imagination clip): fine with them, or swap the image.

## C. Demo video

- [ ] Main cut 2:00 per STORYBOARD.md; 60 s cut for the final.
- [ ] Captions match NUMBERS.md section 0; replays captioned "replayed", speed-ups captioned 2x / 4x, failures labelled.
- [ ] Uploaded (unlisted is fine), link tested in a private window: `[[TEAM:VIDEO_URL]]`.

## D. Final checks before submitting

- [ ] `grep -rn "\[\[" ws4/pitch` and the repo README: nothing left that a judge will see.
- [ ] No em dashes in the README and slides (`grep -rnP "\x{2014}" README.md ws4/pitch`).
- [ ] No secrets in git: `git grep -nE "hf_[A-Za-z0-9]{20,}|sk-ant-|flux_token" -- . ':!*.md'` returns nothing that is
      a real token.
- [ ] No videos, checkpoints or datasets committed (they are gitignored; the Hub holds the data and the adapter).
- [ ] No AI attribution in commit messages or the README.
- [ ] The five datasets and the LoRA open without login on the Hub (checked public 18 Oct 07:20).

## E. Live demo at 14:30

- [ ] FLUX server real-serve-6000b warm at 14:15 (it stops after 45 min idle), tunnel open.
- [ ] Station aligned (`ALIGN=1`: wrist offset < 15 px, scale 0.95 to 1.05, brightness 0.85 to 1.2), box at P3,
      objects per the overview image, both arms folded, one process per arm.
- [ ] Fallback video on the laptop, ready to play if the arm misbehaves.
- [ ] Presenter has OUTLINE.md (3-minute script and Q&A) open.

## Placeholders still open (owner and expected time)

Mirror of NUMBERS.md section 14; grep for the exact tokens.

| Placeholder | What | Owner | Expected |
|---|---|---|---|
| `[[CLIP:HMI_CHAIN]]`, `[[CLIP:IMAGINED_VS_ACTUAL]]` | chain screen recording; optional live imagination clip | demo-rehearsal | with the chain |
| `[[NUM:GPU_SPEND]]`, `[[NUM:CLAUDE_SPEND]]`, `[[NUM:RECORD_HOURS]]`, `[[NUM:TRAIN_HOURS]]` | cost and time split (optional) | Oscar | 13:00 |
| `[[TEAM:NAME]]`, `[[TEAM:MEMBERS]]`, `[[TEAM:WORKSTREAMS]]`, `[[TEAM:WS1]]` to `[[TEAM:WS5]]`, `[[TEAM:WS1_WHO]]` to `[[TEAM:WS5_WHO]]`, `[[TEAM:CREDITS]]` | team facts | Oscar / team | as early as possible |
| `[[TEAM:VIDEO_URL]]`, `[[TEAM:SLIDES_URL]]`, `[[TEAM:DEMO_LAYOUT]]` | links, live layout | Oscar | 13:30 |
