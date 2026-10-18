# ws4/pitch: submission pitch kit

Drafted Sat 17 Oct 2026 evening so Sunday morning only drops in numbers and clips. Owner: the pitch-kit session.
Nothing here is committed by this session.

| File | What |
|---|---|
| [NUMBERS.md](NUMBERS.md) | Every number with its source and status (MEASURED / UNVERIFIED / PENDING), plus the placeholder index |
| [OUTLINE.md](OUTLINE.md) | 3-minute track pitch and 5-minute final, beat by beat with timings, judging-weight map, demo moments, Q&A prep |
| [STORYBOARD.md](STORYBOARD.md) | Demo video shot list (2:00 main cut, 60 s cut), what exists, what to film Sunday |
| [README_submission.md](README_submission.md) | Submission README for the repo root on main |
| `deck/project/` | Source of the hosted slide deck (18 slides, one HTML file per slide, speaker notes in each `<aside>`) |
| `deck/img/` | Images derived for the deck (zero-shot frame strip, demo frame cropped to the table, photo vs sim) |

**Slide deck (hosted, private until shared):** https://claude.ai/artifact/FsptG2MfRYqkmaAqL4EDQm. It presents from the
browser and exports to PPTX or PDF. Share it from the page's Share menu before sending the link to anyone.
Slides in the 3-minute pitch: cover, zeroshot, contract (optional), diagnosis, architecture, heldout (optional),
results, imagination, imaginebetter (optional), realworld, close. The rest are for the 5-minute final or Q&A (their notes say "FINAL ONLY").

## Sunday morning, in order

1. Graded attempts: 5 each of zero-shot A0r-B84, zero-shot A0r-M (median) and the LoRA (NUMBERS.md section 1),
   phone filming per STORYBOARD.md "Shoot list".
2. `python ws4/eval/trial_log.py summary --markdown`, then fill NUMBERS.md sections 2, 4, 10 and 14.
3. `grep -rn "\[\[" ws4/pitch` and replace every placeholder in the markdown files.
4. Deck: edit the numbers in the slide editor (click the text), or edit `deck/project/slides/*.html` and ask Claude to
   republish them to the deck URL. Replace the dashed box on "imagination" with the uploaded clip.
5. Copy `README_submission.md` to the repo root as `README.md` on main (links are written for the root).
6. Cut the video from STORYBOARD.md; captions use NUMBERS.md values only.
