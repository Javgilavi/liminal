# Replace the faulty pen — easy mode

A minimal orchestrator for **two SO-101 arms, five fixed steps and five checkpoint photos**.
The sequence is written in Python. Claude only checks photos; it never plans or changes robot commands.

[Five-step diagram (PNG)](docs/maintenance.png).

| Step | Left arm | Right arm | Camera must confirm |
|---|---|---|---|
| 1 | `Open the box lid` | Stay still | Lid fully open and resting |
| 2 | `Move the cable aside and hold it`, then scripted position freeze | Stay still | Cable held clear; all pens visible and reachable |
| 3 | Scripted hold, no policy | `Pick up the red pen and put it in the faulty cup` | Cable clear; red pen in faulty cup; original slot empty |
| 4 | Scripted hold, no policy | `Pick up a pen from the replacement cup and place it in the empty slot` | Cable clear; replacement in that slot; right arm retracted |
| 5 | `Release the cable`, then `Close the box lid` | Stay still | Lid closed |

Choose another faulty color with `--color blue`. There are **six policy prompts** in total.
“Stay still” is a local position hold, not an inference call. The left hold continues through steps 3–4
and through pauses; no FLUX calls are sent to the left arm during the swap.

## Run the page without connecting motors

The existing `.env` and its Anthropic API key are preserved and remain gitignored.

```bash
cd /home/jgilaviles/erh_hack/liminal
uv --cache-dir /tmp/liminal-uv-cache pip install --python .venv/bin/python -e '.[maintenance]'
.venv/bin/python -m liminal --color red
```

Open **http://127.0.0.1:8780**. This starts **photo review only**: see the five steps and upload an actual
checkpoint photo for the highlighted step. Claude returns the individual checks and a short reason.
An unclear or failed check pauses at that step. There is no scripted success or simulated robot.
Photo uploads are disabled in robot mode so old photographs cannot advance a connected robot.

## Run the actual arms

Prepare the scene as described: lid closed, pens flat in individual slots, cable over the slots,
replacement and faulty cups outside the box, both arms in home poses. Homing is not automatic.

1. Edit [configs/maintenance_robot.yaml](configs/maintenance_robot.yaml): serial ports, calibration ids,
   three camera indices and measured safe joint limits.
2. Use a running FLUX SO-101 HTTP server with `GET /health`, `POST /reset`, and `POST /act`.
   It must identify real FLUX (`fake=false`), returning five joint positions in **degrees** and gripper 0–100.
   Only this controller should use that single-session policy server during the job.
3. Install LeRobot's Feetech hardware support in the same environment and launch:

```bash
.venv/bin/python -m liminal --color red \
  --robot configs/maintenance_robot.yaml --flux http://127.0.0.1:18100 --seconds 8
```

Click **Run demo**. Each prompt runs for the bounded duration, then the active arm freezes.
Step 5 issues the release prompt, then the close prompt, each with the same duration limit. The scene camera takes one checkpoint;
Claude checks it and advances only if **all** required conditions are visibly true.
Compare checkpoint 2→3 to identify the emptied slot and 3→4 to verify the same slot was refilled.
Verified checkpoints advance automatically through the five steps. A failed or unclear check pauses
the job. After correcting the scene, click **Retry current step** to retry the same fixed prompt.
There is no automatic retry loop. The other arm always stays in its scripted hold.

**Stop** latches both arms into position hold and rejects late policy/model responses. A stopped job
cannot resume automatically. Shutting down disconnects the motors; unload the held cable safely first.

The six manipulation skills must already work in the selected FLUX policy (normally after collecting
and fine-tuning on this scene). This orchestrator does not train the policy or guarantee the motions.
Hardware execution has not been verified on this machine: no arm serial ports or cameras are attached.

## Only the small active path

- [maintenance.py](src/liminal/maintenance.py): fixed steps, checkpoint verifier, sequential state.
- [hardware/maintenance.py](src/liminal/hardware/maintenance.py): FLUX requests and idle/cable holds.
- [ui/maintenance.py](src/liminal/ui/maintenance.py): page and four operator actions.
- [maintenance.html](src/liminal/ui/web/maintenance.html): the operator page.

The driver reuses the low-level SO-101 adapter and safety supervisor. It does not import the old planner,
behavior trees, memory retrieval, replay scenes, evaluations or video-to-plan pipeline. `liminal`,
`pen-demo`, and `python -m liminal` all open this minimal controller. Older code remains for reference;
it is not part of this execution path. The previous branch and its stashed changes are preserved.

## Saved evidence

Each job writes `runs/pen-demo/<timestamp>/state.json`, a readable `log.txt`, and `checkpoint-1.png`
through `checkpoint-5.png`. Failed/retried photos are also kept as `step-N-attempt-M.png`.
No database, long-history memory, generated task tree or model-authored prompts.

For data collection: vary lid grip/angle and cable position. Move the faulty color between slots,
and collect extra removal demonstrations. Vary the replacement cup position and empty slot.
Record release and close as separate episodes. Do not record the scripted holding interval as a policy skill.
