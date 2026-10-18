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

- [ ] **Everything on main**: PR #3 and #4 (https://github.com/Javgilavi/liminal/pull/3, /pull/4); otherwise README
      links break.
- [ ] Copy `ws4/pitch/README_submission.md` to the repo root as `README.md` (it is written for the root), merging in
      the team's existing README sections (harness, twin lab, project page). Delete the HTML comment
      at its top.
- [ ] Every link opens on GitHub (ws4 paths, Hub datasets, the LoRA, `ws4/LORA_HOWTO.md`).
- [ ] The results table is the fresh `trial_log.py summary --markdown`.
- [x] Team name and members filled: Liminal, Oscar Devos, Javier Gil, Tomas Leggaet, Simone Utili, Stanislaw Sufin.

## B. Slides

- [ ] Hosted deck: https://claude.ai/artifact/FsptG2MfRYqkmaAqL4EDQm (21 slides, 13 shown). It is **private until shared**: share
      it from the page's Share menu. PDF: `~/liminal_video/out/liminal_slides.pdf`, also in `gdrive:liminal/05_video/`
      (rendered locally from the deck source; PPTX from the page's Share > Export).
- [ ] Every `[[...]]` on the slides replaced or the element deleted (results, orchestrator chain pill, team, cover and
      close pills).
- [ ] The box step is called a scripted replay everywhere (slides, README, video captions), never FLUX.
- [ ] Cutter count reconciled (trial log 1/6 vs the orchestrator's 0/5 inserts at 09:00) before it is quoted.
- [ ] Chain stated with its caveats: unguarded demo mode, red pen placed by the green-pen LoRA.
- [ ] Video v3 from video-cut (Drive `liminal/05_video`, about 12:50) checked against NUMBERS.md 0.1b captions.
- [ ] 5-minute pitch: 13 main slides (cover, zeroshot, diagnosis, the chain, data, held-out, results, knife edge,
      imagination, harness, twin, team, close); 8 backup slides are hidden for Q&A.
- [ ] People visible in two images (knife-edge wrist view, the imagination clip): fine with them, or swap the image.

## C. Demo video

- [ ] Main cut 2:00 per STORYBOARD.md; 60 s cut for the final.
- [ ] Captions match NUMBERS.md section 0; replays captioned "replayed", speed-ups captioned 2x / 4x, failures labelled.
- [ ] Submission video `gdrive:liminal/05_video/liminal_2min_v4.mp4`; shareable link (https://drive.google.com/file/d/1et5ffQpXLLh4a1qOvociNB1weNlaCGPU/view, anyone with the link can view) tested in a private window.

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
- [ ] Presenter has OUTLINE.md (5-minute script and Q&A) open.

## Placeholders still open (owner and expected time)

Mirror of NUMBERS.md section 14; grep for the exact tokens.

| Placeholder | What | Owner | Expected |
|---|---|---|---|
