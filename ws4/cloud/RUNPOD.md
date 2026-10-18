# RunPod: inference pod for the robot (training on the teammate's box)

Runs in parallel with the teammate's 3x RTX PRO 6000 box (`ws4/HANDOFF_GPU.md`). Same code, same steps; only
the machines differ. Prices are RunPod on-demand, read from the console on 2026-10-17. This account shows no
spot option.

| | Training pod | Inference pod |
|---|---|---|
| GPU | 1x A100 SXM 80 GB, $1.79/h | 1x A40 48 GB, $0.59/h (for the Sunday demo hours, optionally an L40S at $1.09/h or an RTX PRO 6000 at $2.49/h) |
| Why | LoRA training is estimated at 40 to 50 GB (unmeasured; the frozen bf16 model alone is about 25 GB), so 80 GB is the safe size | one shared FLUX instance is about 32 GB in BF16 (BFL docs), or about 24 GB with the text encoder on the CPU |
| Runs | LoRA P in the afternoon, then LoRA A and B one after the other overnight, about 16 h | only while the robot is tested, about 15 h over both days |
| Cost | about $29 | about $9 |

Add about $5 for bootstrap time and storage. **Total about $40 to $45** if nothing ever idles. The next
section gets this lower.

## Live deployment (2026-10-17, 13:08 CEST)

| What | Value |
|---|---|
| Network volume | `ws4-inference`, id `6j5vbwvfu8`, 100 GB STANDARD, EU-RO-1 (about $0.23 per day until deleted) |
| Pod | `ws4-flux-inference`, id `4inbmv6d4xb876`; 1x RTX PRO 4500 Blackwell 32 GB, 62 GB RAM, 32 vCPU, host CUDA 13.0, $0.72/h |
| Image | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, container disk 30 GB, network volume at `/workspace` |
| SSH | `ssh root@213.173.107.24 -p 13684 -i ~/.ssh/id_ed25519_osrdvs` (since 13:18). The port changes on every container restart, including env updates: read it from the MCP `get-pod` or the console. |
| Proxy SSH | `ssh 4inbmv6d4xb876-64411f24@ssh.runpod.io`: interactive only, no rsync |

Measured on the pod:
- `nvidia-smi` reports 32,623 MiB.
- `runpodctl` 1.14.15 and a pod-scoped `RUNPOD_API_KEY` are present. This version uses `runpodctl stop pod <id>`;
  the docs' `pod stop` belongs to a newer CLI, and `idle_stop.sh` now detects which one is installed.
  - With the pod's key, `get pod` and the REST API return Unauthorized/403.
  - `stop pod <non-existent id>` passes authorization ("does not exist"), so self-stop is expected to work.
  - The first version of the watchdog used the wrong syntax and failed at 13:16 UTC. It was re-armed fixed at
    13:24 UTC.
- The server token is read from `/workspace/flux_token` (`--token-file`), so it never appears in `ps`.
- Pod environment variables reach non-interactive SSH only after `source /etc/rp_environment`.
- Updating a pod's env through the API restarts its container: `/root` is wiped and the SSH port changes;
  `/workspace` survives.
  A restart in the middle of `bootstrap.sh` can leave a half-installed package in a venv on the volume.
  `uv sync` then reports "Audited" (the dist-info exists) while `import torch` fails. Fix it with
  `rm -rf $WS4_ROOT/<env>/.venv` and re-run the bootstrap; the uv cache on the volume makes the rebuild quick.
- Older copies of `bootstrap.sh` skipped the weight download on cards under 40 GB (we ran it with `PREFETCH=1`).
  The current one prefetches on any GPU with 16 GB or more.
- The 32 GB card needs `EXTRA_ARGS=--text-encoder-on-cpu` for `runpod_serve.sh` (unmeasured until the first
  load).

### Measured on the pod (SO-101 package, BF16, RTX PRO 4500 32 GB, text encoder on the CPU)

10 runs each, 3 warm-up passes so every prompt is already encoded. Each model load took about 6.1 to 6.5 min
from the network volume.

| Run | Median per chunk | p90 | Peak VRAM | Arm moving (sync loop, 1.07 s chunks) |
|---|---|---|---|---|
| default (guidance 3) | 1.461 s | 1.466 s | 23.63 GiB | 42% |
| guidance 1.0 (video and action) | 0.758 s | 0.762 s | 23.63 GiB | 59% |
| `compile_model=true` | 1.222 s | 1.225 s | 23.63 GiB | 47% |
| guidance 1.0 + `compile_model=true` | 0.629 s | 0.638 s | 23.63 GiB | 63% |

- The same default run with only 2 warm-up passes had a 17.3 s p90. That was one first-time CPU encode of a
  prompt the warm-ups had not covered, which is why the server now pre-encodes its prompts.
- `compile_model` adds roughly 2 min of compilation to startup.
- **VRAM:** the 23.63 GiB peaks above include the load-time peak (Qwen briefly on the GPU before moving to the
  CPU). After the move the policy holds 15.3 GiB, and peaks at 15.8 GiB per chunk.
- **Imagination** (`imagine.run_imagine bench`, default guidance, 5 reps, decoding on the same GPU, Qwen on the
  CPU):

  | Mode | Per chunk |
  |---|---|
  | stock predict | 1.45 s |
  | imagination without decode | 1.46 s |
  | synchronous decode | 1.69 s |
  | decode only (41 frames) | 0.23 s |
  | background decode on the same GPU | 1.67 s median (1.46 s best) |
  | best of 2 with decode | 3.27 s |

  Peak 17.6 GiB, so the 32 GB card has about 14 GiB to spare for imagination.
- **End to end from the laptop** (Oscar's home connection, SSH tunnel to EU-RO-1, 2026-10-17 12:58 UTC):
  - round trip 1.78 s median (max 1.93 s) = server 1.47 s + network and JPEG encoding 0.31 s;
  - request 312 KB per chunk;
  - response 42 x 6 commands.

  Upload speed at the venue decides the 0.31 s; `RemoteBackend(quality=...)` lowers it.
- **Server start to READY:** about 9 min (6.4 min load, then 6 prompts at about 15 s each, the first 34 s).
- Guidance 1.0 changes the sampling. Check motion quality on the robot against guidance 3 (same prompt and
  scene) before serving with it (`flux/GPU_RUNBOOK.md` §2).

- **Pause math:** LeRobot's synchronous loop, and our per-arm runner, start the next inference only once the
  32-action queue is empty. So the arm stands still for the whole inference after each 1.07 s chunk, and at
  1.46 s it moves about 42% of the time.
- **Prompt warm-up:** `runpod_serve.sh` now encodes every prompt in `/workspace/prompts.txt` right after the
  server starts (`warm_prompts.py`), so no prompt is encoded live. The defaults are the recording protocol's
  S1 to S6. Edit the file so it matches the runner's strings exactly.
- **Memory:** Qwen stays on the CPU (added automatically on GPUs under 40 GB): 23.6 + 8.3 GiB would not fit
  in 31.9 GiB.

## Live deployment 2: single-arm SO-101 pod (2026-10-17, 16:18 CEST)

Dedicated to single-arm SO-101 work (the red arm, zero-shot "stack the cups"). It is a separate pod with its
own volume, so the 4500 pod above stays free for the XLeRobot work.

| What | Value |
|---|---|
| Network volume | `so101-inference`, id `u1ffu7k5f9`, 100 GB, EU-CZ-1 (about $0.23 per day until deleted) |
| Pod | `so101-flux-serve`, id `k4p4f8hgs0kroz`; 1x RTX PRO 6000 96 GB, EU-CZ-1, $2.49/h |
| SSH | `ssh root@213.192.6.98 -p 40082 -i ~/.ssh/id_ed25519_osrdvs` (since the 14:38 UTC start; it was 40093 before). The port changes on every start: read it from the MCP `get-pod` |
| Tunnel for the robot | `ssh -N -L 8765:127.0.0.1:8765 -L 8766:127.0.0.1:8766 root@213.192.6.98 -p 40082`, then `RemoteBackend("http://127.0.0.1:8765", "so101", token=...)` or `RemoteBackend("http://127.0.0.1:8766", "so101r", token=...)` |
| Token | `/workspace/flux_token` on the pod (root only), shared by both servers; read it over SSH into an env var, never into argv |
| Server 1, port 8765 | package `so101` = `black-forest-labs/flux-3-action-so101`, `guidance_scale=1.0`, `guidance_scale_action=1.0`, `compile_model=true`, text encoder on the GPU (started by `runpod_serve.sh`) |
| Server 2, port 8766 | package `so101r` = `/workspace/packages/so101r_red_follower` (from 15:27 UTC), same overrides. Released SO-101 weights at revision `0713a7af6768562e618e81d534372c6795d41b2e` with history quantiles from the red_follower hand-guided demos (`ws4/single_arm/handguided/stats_red_follower.json`, state and action keys) |
| Prompts | so101: 9 pre-encoded from `/workspace/prompts.txt`: "stack the cups", "put the left cup into the right cup", "stack the white paper cups", then S1 to S6. so101r: the first two, from `/workspace/prompts_so101r.txt` |
| Watchdog | `IDLE_MIN=45` for the robot attempt sessions (20 min was too short for the robot-side setup) |
| Restart | MCP `pod-action start`, then, with the new port: `ssh -n root@213.192.6.98 -p <port> 'IDLE_MIN=45 EXTRA_ARGS="--override guidance_scale=1.0 --override guidance_scale_action=1.0 --override compile_model=true" nohup bash /workspace/liminal/ws4/cloud/runpod_serve.sh > /workspace/logs/serve.out 2>&1 < /dev/null &'`. Then start server 2 by hand (below) |

Server 2 by hand, on the pod: `source /etc/rp_environment; source $WS4_ROOT/lerobot/.venv/bin/activate; cd $LIMINAL/ws4; export PYTHONPATH=$PWD`, then

```bash
nohup python -m flux.remote --host 127.0.0.1 --port 8766 --token-file /workspace/flux_token --package so101r=/workspace/packages/so101r_red_follower --override guidance_scale=1.0 --override guidance_scale_action=1.0 --override compile_model=true >> /workspace/logs/flux_remote_so101r.log 2>&1 < /dev/null &
```
```bash
python cloud/warm_prompts.py --url http://127.0.0.1:8766 --prompts-file /workspace/prompts_so101r.txt --token-file /workspace/flux_token
```

- so101r runs as a second server because `flux.remote` loads its packages only at start. Restarting server 1
  with both packages would have cut so101 off for 3 to 4 min during the A0 attempts. The cost is a second Qwen
  copy: GPU memory went from 27 GB to 54 GB of 96 GB.
- The package was built offline on the pod (`HF_HUB_OFFLINE=1`) with
  `flux.package_tools.renormalize_package("black-forest-labs/flux-3-action-so101", "/workspace/packages/so101r_red_follower", stats)`.
  The weights are a symlink into the HF cache, and `normalization_source.json` records the snapshot path.
  Read back from the three saved history-normalizer files, the quantiles match the input exactly.
- so101r from the laptop ("stack the cups", 5 timed runs): 0.503 s median round trip (max 0.580 s), server
  0.340 s. Its server was serving by 15:27:34, at most about 3 min after the 15:24 build; the first prompt took
  12.6 s, including compile.

Measured on the pod (UTC):
- Bootstrap on the fresh volume, weights included, finished at 14:15.
- Server start to READY: 2.5 min (14:15:50 to 14:18:19). The model loaded in 1.9 min, then the prompt
  warm-up took 37 s: 31.4 s for the first prompt, which includes `compile_model`, and 0.36 s for each of the
  other 8.
- GPU memory with everything resident (Qwen on the GPU): 27,175 MiB of 96 GB.
- **End to end from the laptop** (Oscar's home connection, SSH tunnel to EU-CZ-1, "stack the cups",
  7 timed runs after 1 warm-up):
  - round trip 0.479 s median (max 0.510 s) = server 0.333 s + network and JPEG encoding 0.145 s;
  - request 312 KB per chunk; response 42 x 6 commands.
- **Arm moving** (sync loop, 1.07 s chunks): 69% of the time at the 0.48 s round trip, up from 42% on the
  4500 pod at default guidance.
- Guidance 1.0 still needs the motion-quality check on the robot against guidance 3 (`flux/GPU_RUNBOOK.md`
  §2). To serve guidance 3 instead, restart the server without the two guidance overrides.
- A prompt outside `/workspace/prompts.txt` costs little here: with Qwen on the GPU, a first-time prompt took
  0.36 to 0.48 s for the whole call, compared with about 0.33 s for a cached one.

Run log (UTC; the pod costs $2.49/h while running, and the volume bills about $0.23/day all the time):

| Run | Started | Server READY | Stopped | GPU time | Cost |
|---|---|---|---|---|---|
| 1 (create, bootstrap, serve, latency test) | 14:07:28 | 14:18:19 | 14:35:51 by `idle_stop.sh` (20 min idle) | 28.4 min | about $1.18 |
| 2 (restart for so101-zeroshot, Oscar approved in chat) | 14:38:27 | 14:44:18 | running, `IDLE_MIN=45` | | |

- The 20 min watchdog stopped the pod as designed: `pod "k4p4f8hgs0kroz" stopped` (with a harmless
  "config file not found" warning from runpodctl).
- Run 2 timing: the server started at 14:41:10 and was READY 3.1 min later; the first prompt took 34.1 s,
  including compile.
- Run 2 end to end from the laptop ("stack the cups", 5 timed runs): 0.555 s median round trip (max 0.595 s),
  server 0.340 s.
- Between a start and `runpod_serve.sh` the pod has no watchdog, so run the server right after the start.
  For run 2 the gap was 2.7 min (about $0.11) while waiting for Oscar's OK.

## Costs and receipts (event budget about USD 100, any provider; keep every invoice)

| Resource | Id | Where | Rate | Created (UTC) | Purpose |
|---|---|---|---|---|---|
| Pod `ws4-flux-inference` | `4inbmv6d4xb876` | EU-RO-1, RTX PRO 4500 32 GB | $0.72/h | 2026-10-17 11:07 | XLeRobot: TTT, bimanual and robot tests |
| Volume `ws4-inference` | `6j5vbwvfu8` | EU-RO-1, 100 GB | $0.07/GB/month (about $0.23/day) | 2026-10-17 11:07 | weights + envs for the 4500 pod |
| Pod `so101-flux-serve` (created as `ws4-flux-serve-6000`) | `k4p4f8hgs0kroz` | EU-CZ-1, RTX PRO 6000 96 GB | $2.49/h | 2026-10-17 14:07 | single-arm SO-101 serving (zero-shot "stack the cups", teammate single-arm runs) |
| Volume `so101-inference` (created as `ws4-serve-cz`) | `u1ffu7k5f9` | EU-CZ-1, 100 GB | about $0.23/day | 2026-10-17 14:07 | weights + envs for the single-arm pod |

- Spend reported by RunPod's billing API at 14:10 UTC (it lags about 1 h): $1.47. At 14:47 UTC: $1.89 for
  2026-10-17 (pods $1.86, volumes $0.03), still without most of the single-arm pod's first run.
- Both pods run `idle_stop.sh`, which stops the pod after a stretch of idle GPU: 20 min by default, 45 min on
  the single-arm pod for robot sessions. A stopped pod bills only its volume. Per-run costs for the single-arm
  pod are in its run log above.
- **Receipts:** RunPod bills prepaid credits, so the invoices are the credit purchases. Download them from the
  console, Billing → Recent transactions → Receipt / Invoice (Oscar's account; the agent does not download or
  pay anything). Usage per day per resource is in Billing → Billing explorer, or the MCP `list-billing`.
- After the hackathon: terminate both pods and delete both volumes (volumes bill until deleted).

## Paying nothing while idle (from docs.runpod.io, read 2026-10-17)

- **Pods.** "Pods are billed by the second for compute and storage, with no fees for data ingress or egress."
  Stopping a Pod "releases the GPU": a stopped Pod costs only storage. Container disk is free and wiped;
  volume disk costs $0.20/GB/month; a network volume costs $0.07/GB/month either way
  (`/pods/pricing`, `/pods/manage-pods`).
  - Deploying needs at least one hour's worth of credits.
  - A restarted Pod can come back with zero GPUs. With a network volume, deploy a new Pod on the same volume
    instead.
- **Serverless flex workers** "scale to zero when there are no requests". Billing covers cold start (mostly
  loading the model into GPU memory), execution, and the idle timeout (default 5 s). A scaled-down worker is not
  billed. Per-second rates (`/serverless/pricing`, `/serverless/endpoints/endpoint-configurations`):

  | GPU class | Rate | About |
  |---|---|---|
  | A40 / A6000, 48 GB | $0.00034/s | $1.22/h |
  | L40S, 48 GB | $0.00053/s | $1.91/h |
  | A100, 80 GB | $0.00076/s | $2.74/h |

  That is about twice the Pod rate, but it is paid only while a robot is actually calling the policy.
- **Load-balancing Serverless endpoints** send HTTP requests straight to a worker
  (`https://ENDPOINT_ID.api.runpod.ai/<path>`): no queue, a 30 MB payload limit and a 5.5 min processing
  timeout (`/serverless/load-balancing/overview`).
  - Our `flux/remote.py` already fits this model: stateless HTTP, one POST per chunk, with the history kept on
    the robot side.
  - What is missing: bind `0.0.0.0` on `$PORT`, a health route (`HEALTH_CHECK_PATH`), and the client sending
    `Authorization: Bearer <RunPod API key>` with retries during cold starts.

**Cheapest plan:**

| Job | Where | Rule | Cost |
|---|---|---|---|
| Training | Teammate's Blackwell box (free) | The teammate pushes each LoRA checkpoint to a private HF repo, so any inference machine can pull it. | $0 |
| Training fallback | A100 Pod | Ends with `...lerobot_train ... ; runpodctl pod stop $RUNPOD_POD_ID`, so it stops itself when the run finishes. Whether `runpodctl` is authenticated inside a Pod is UNVERIFIED; check `runpodctl pod list` there first. | $1.79/h of actual training |
| Bootstrap | A cheap Pod on the same network volume, e.g. RTX A5000 24 GB at $0.27/h | Building the envs and downloading weights needs no big GPU. | small |
| Inference | A40 Pod that stops itself after 20 idle minutes, or a Serverless load-balancing endpoint (about 1 to 2 h of work to adapt `flux/remote.py`) | Serverless starts on the first request and stops by itself. | about $6 to $9 for about 7 h of actual robot use |
| Storage | 300 GB network volume | about $0.70 per day | about $1.50 |

If the teammate's box does all the training, **the RunPod bill is about $10 for the weekend**, close to the
current $8.46 balance. If RunPod also trains (A, B and P, about 14 h on an A100), it is **about $35**.

**Driving RunPod from the agent:** Runpod's hosted MCP server (`https://mcp.getrunpod.io/`,
`/get-started/mcp-servers`) can create, stop and delete Pods, endpoints and volumes. Oscar adds it in Claude
Desktop under Settings → Connectors → Add custom connector, and approves the "Sign in with Runpod" OAuth
prompt; no API key is stored on disk. The skills plugin (`npx skills add runpod/runpod-plugins-official`,
`/get-started/agent-skills`) needs an API key in the shell, so the MCP connector is the lighter option.

## Chosen plan (Oscar, 2026-10-17): inference only on RunPod

The teammate's box trains. RunPod runs one self-stopping A40 inference pod. The training-pod material further
down stays as a fallback.

### One-time setup (Oscar)

1. **RunPod MCP connector:** Claude Desktop → Settings → Connectors → Add custom connector →
   `https://mcp.getrunpod.io/`, then approve "Sign in with Runpod". A new Claude session may be needed before
   its tools appear.
2. **Hugging Face:**
   - With the HF account whose token the pod will use, click "Agree and access" on
     `black-forest-labs/flux-3-action-so101`, `-base` and `-droid`.
   - Create a read token and store it in RunPod **Settings → Secrets** as `HF_TOKEN`. The pod references it as
     `{{ RUNPOD_SECRET_HF_TOKEN }}`; the token never appears in a form, a log or a chat.
   - The same account needs read access to the private HF repo where the teammate pushes LoRA adapters.
3. **SSH key:** RunPod **Settings → SSH public keys**, add `~/.ssh/id_ed25519_osrdvs.pub`.
4. **Credits:** $8.46 covers about 10 h of A40 plus storage. Add $5 to $10 for headroom; deploying needs at
   least one hour's worth of credits.

### Deploy (agent, through the MCP, after the steps above)

| Setting | Value |
|---|---|
| Network volume | 100 GB in a datacenter with A40 stock, about $0.23 per day. Weights 39 GB + two envs + caches. |
| Pod | 1x A40 48 GB, on-demand $0.59/h, template RunPod PyTorch `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` |
| Disks | container disk 30 GB, network volume at `/workspace` |
| Ports | expose TCP 22 (SSH); the policy server binds 127.0.0.1 and is reached only through an SSH tunnel |

Environment variables:

```
HF_TOKEN={{ RUNPOD_SECRET_HF_TOKEN }}
WS4_ROOT=/workspace/ws4
LIMINAL=/workspace/liminal
HF_HOME=/workspace/hf
UV_CACHE_DIR=/workspace/uv-cache
UV_PYTHON_INSTALL_DIR=/workspace/uv-python
```

A pod's home directory is on the container disk and is **wiped on every stop**. Everything that must survive
lives under `/workspace` through these variables.

### First start (once, about 45 to 60 min, about $0.60)

From the laptop, copy the current `ws4/` (latest code from every agent, no GitHub access needed on the pod):

```bash
rsync -az -e "ssh -p <port> -i ~/.ssh/id_ed25519_osrdvs" --exclude .cache --exclude __pycache__ --exclude outputs \
  ~/liminal/ws4 root@<pod-ip>:/workspace/liminal/
```

On the pod:

```bash
ssh root@<pod-ip> -p <port> -i ~/.ssh/id_ed25519_osrdvs
nohup bash $LIMINAL/ws4/cloud/bootstrap.sh > /workspace/bootstrap.out 2>&1 &   # weights go to $HF_HOME
runpodctl pod list    # must work inside the pod, or idle_stop.sh cannot stop the pod (then stop it via the MCP)
```

Re-run the same `rsync` whenever the code changes. The `ws4-handoff` branch (`git clone -b ws4-handoff ...`
with `ssh -A`) is the alternative.

Then measure latency once:

```bash
python -m flux.measure_latency --checkpoint black-forest-labs/flux-3-action-so101 --runs 10
```

Run it from `$LIMINAL/ws4` in the LeRobot env, as in `flux/GPU_RUNBOOK.md` §2.

### Every robot session (pod start to robot connected, about 3 min)

1. Start the pod: through the MCP, or the console's play button.
2. Start the server and open the tunnel from the laptop:

   ```bash
   ssh root@<pod-ip> -p <port> 'nohup bash /workspace/liminal/ws4/cloud/runpod_serve.sh >/workspace/logs/serve.out 2>&1 &'
   ssh root@<pod-ip> -p <port> cat /workspace/flux_token                  # --token for flux.run_xlerobot
   ssh -N -L 8765:127.0.0.1:8765 root@<pod-ip> -p <port> &              # robot uses http://127.0.0.1:8765
   ```

   - `runpod_serve.sh` loads the model (about 1 to 2 min from the volume) and serves `PACKAGES`
     (default `so101=black-forest-labs/flux-3-action-so101`; add the teammate's LoRAs as `NAME=ORG/repo`).
   - It also starts `idle_stop.sh`, which **stops the pod after 20 minutes with an idle GPU**
     (`IDLE_MIN` changes that, 0 disables it).
   - A stopped pod costs only the volume.
3. Run the robot with `ws4/flux/GPU_RUNBOOK.md` §3 (`flux.run_xlerobot`).

If a restarted pod comes back with zero GPUs, deploy a new A40 pod on the same volume with the same settings.
Nothing is re-downloaded.

### Weekend cost

| Item | Cost |
|---|---|
| About 10 to 15 h of robot sessions, A40 at $0.59/h | about $6 to $9 |
| Volume | about $0.50 |
| First bootstrap | about $0.60 |
| **Total** | **about $8 to $10** |

For the live demo hours only, an L40S ($1.09/h) or RTX PRO 6000 ($2.49/h) pod on the same volume gives
faster chunks, if the A40 latency turns out to be poor.

## Fallback: training on RunPod too

Use this only if the teammate's box fails. One A100 80 GB pod ($1.79/h) on a 300 GB network volume, same
environment variables plus `DATA_ROOT=/workspace/data`.

```bash
ssh -A root@<pod-ip> -p <port> -i ~/.ssh/id_ed25519_osrdvs
apt-get update && apt-get install -y ffmpeg tmux
git clone -b ws4-handoff git@github.com:Javgilavi/liminal.git $LIMINAL
tmux new -s boot 'bash $LIMINAL/ws4/cloud/bootstrap.sh'
```

Then follow `ws4/HANDOFF_GPU.md` steps 3, 4 and 6 with `CUDA_VISIBLE_DEVICES=0`.
- Run the LoRAs one after the other.
- End each training command with `; runpodctl pod stop $RUNPOD_POD_ID` so the pod stops itself when the run
  finishes.
- About $35 in total with the inference pod.

## Unknowns to measure first, and when to switch

| Unknown | Measure | If bad |
|---|---|---|
| LoRA memory on the A100 | `nvidia-smi` after 50 microsteps | over 80 GB: move to an RTX PRO 6000 at $2.49/h |
| LoRA step time | trainer log after 50 microsteps | if 10,000 steps need more than about 10 h, cut the steps, or move to an H100 at $3.99/h (likely 2 to 3x faster) |
| A40 chunk latency | `python -m flux.measure_latency --checkpoint black-forest-labs/flux-3-action-so101 --runs 10` | above about 1.5 s: use an L40S, or an RTX PRO 6000 for the demo hours |

## Sim-track pod ws4-sim-4500 (orchestrator note, 17 Oct evening)

- Pod `ws4-sim-4500` (fnxekskwlu0095), 1x RTX PRO 4500 32 GB, EU-RO-1, $0.72/h, volume `ws4-inference` at
  /workspace. Created 16:07 UTC after both earlier 4500 pods failed: 4inbmv6d4xb876 could not restart (host GPUs
  taken) and voqbbqhku7bz8v never booted (runtime null, zero container and system logs, twice). Both are terminated.
- Closed-loop sim test (closed-loop-test-flux-dr): `so101` READY 7 min after runpod_serve.sh (first prompt 121 s
  including compile, others 13 to 15 s). `so101sim` (released weights, sim left-arm quantiles) built offline at
  /workspace/packages/so101sim_left, READY in 9 min. Server 0.65 s, round trip 0.84 s per chunk (guidance 1.0 +
  compile, Qwen on CPU). Tunnel on local port 18765 to avoid the single-arm 8765 tunnel.
- Server stopped 17:14:40 UTC; idle watchdogs (IDLE_MIN 30) armed; pod handed to ws4-ttt's sim LoRA job.

## Capacity notes and pods (orchestrator, 17 Oct ~19:30 CEST)

- so101-flux-serve (k4p4f8hgs0kroz, RTX PRO 6000, EU-CZ-1) idle-stopped at about 17:55 CEST and then could not
  restart: "not enough free GPUs on the host machine". A stopped pod only restarts on its own host.
- `so101-diag-4090` (bd63s5j8xtpves): 1x RTX 4090 24 GB, EU-CZ-1, $0.89/h, volume so101-inference, created 17:31 UTC
  for the data-format diagnostic (A/B/C/D frame variants, flux-action issue #2). Temporary; terminate after.
- Stock at 17:31 UTC: EU-CZ-1 only 24 GB cards; EU-RO-1 has an A100 80GB PCIe ($1.79/h, LOW), RTX PRO 4500/4000,
  L4, 4090, B200; RTX PRO 6000 HIGH in EUR-IS-1/2 and the US. Plan: overnight real LoRA on an A100 80 GB in EU-RO-1
  on the ws4-inference volume (weights + envs already there).

## Pods created 18:21 to 18:25 UTC (orchestrator)

- `so101-serve-4090` (wjaszpojorczgm): RTX 4090 24 GB, EU-CZ-1, $0.89/h, volume so101-inference. Serves the stock
  `so101` package for the real-arm variant E test (lift/elbow mapping, client-side), so so101-diag-4090 keeps
  running diagnostics. An A100 80 GB in EU-RO-1 was gone at create time.
- `real-train-6000` (bqmpv0ulsmb14a): RTX PRO 6000 96 GB, EUR-IS-1, $2.49/h, NEW volume `real-train-is`
  (hnh342zg4i, 150 GB, EUR-IS-1). Tonight's real-track LoRA (approved by Oscar). Bootstrap + training by ws4-datasets.

## Real-track LoRA pod real-train-6000 (ws4-datasets, 17 Oct evening)

| What | Value |
|---|---|
| Pod | `real-train-6000`, id `bqmpv0ulsmb14a`, 1x RTX PRO 6000 96 GB, 32 vCPU, EUR-IS-1, $2.49/h, container disk 40 GB; started 18:24:44 UTC |
| Volume | `real-train-is`, id `hnh342zg4i`, 150 GB, EUR-IS-1, at `/workspace` |
| SSH | `ssh root@157.157.221.177 -p 16301 -i ~/.ssh/id_ed25519_osrdvs` (changes on every start) |
| Watchdog | `ws4/ttt/pod_job_watchdog.sh 30`, armed 18:27:35 UTC: stops the pod after 30 min with GPU use at or under 5% and no `lerobot_train`, `bootstrap.sh` or other listed job running (`bootstrap.sh` added to its list for this fresh volume) |
| Run script | `ws4/cloud/real_lora.sh package STATS.json \| probe \| train \| merge` |
| Data | `/workspace/real_lora/pen_in_right_all`: snapshot of `haller_put_the_red_pen_in_its_slot_in_the_cardboard_box` at 18:30 UTC (36 episodes, 22,946 frames, 0.21 h), `convert_bimanual --to per_arm --side right --src-units degrees --task-suffix ""` (robot-left task arm; scene = head, wrist = right_wrist, both 480x640 h264) |

Timeline (UTC):
- 18:27 ws4/ synced (70 MB), watchdog armed, `bootstrap.sh` started. The weight prefetch (so101, base, droid
  metadata) had finished by 18:33.
- 18:31 dataset snapshot uploaded (242 MB).
- 18:38:34 bootstrap done, rc 0, 11 min: torch 2.10.0+cu128 with sm_120, NATTEN 0.21.6 3D op OK in both envs,
  `Flux3Policy` imports in the LeRobot env. A fresh EUR-IS-1 volume bootstraps 4x faster than the 45 to 60 min
  planned.
- 18:38 provisional base package `/workspace/packages/so101_haller_pen_B_probe` (variant B quantiles from
  so101-zeroshot's `a0r_B_haller_pen.json`, 20 episodes), only for the step-time probe.
- 18:38 to 18:52 step-time probes (below).
- 18:53 provisional run `/workspace/real_lora/lora_pen_prov`: 2,000 microsteps on the provisional split (28 train
  episodes, eval on held-out 20, 21, 22). Base `/workspace/packages/so101_haller_pen_B_prov` uses variant B
  quantiles of exactly those 28 training episodes (`a0r_stats.py`, 19,256 frames). It tests eval, checkpoints and
  merge before the overnight run, and keeps the pod busy until the final split (about 19:30 UTC). A watchdog stop
  in that gap could end in "not enough free GPUs on the host" at restart.
- Pipeline test on the provisional run:
  - 19:06 checkpoint 500 (eval loss 0.285 on the held-out takes), raw and EMA adapters 0.47 GB each.
  - `real_lora.sh merge` of the EMA adapter: 2.7 min on the CPU, giving a 13.9 GB package
    (`/workspace/packages/so101_haller_pen_B_prov_lora500`) with the B normalizers.
  - Serving check (`/workspace/real_lora/serve_test.sh`, port 8770, guidance 1.0, no compile): `flux.remote`
    loads it and returns finite 42 x 6 chunks for "put the red pen in its slot in the cardboard box", 0.80 s per
    chunk while sharing the GPU with training. The test server was stopped afterwards.
- Oscar, 17 Oct evening: the overnight run waits for the final split (enough data first), then 20,000
  microsteps.
- Test run losses (28 takes): eval 0.285, 0.271, 0.257, 0.254 at microsteps 500 to 2,000; train loss per 500
  microsteps 0.222, 0.178, 0.172, 0.167 (action MSE 0.208 to 0.140), max 0.48, gradient norm at most 1.9.
  Flattening on 28 takes, so with about three times the data the 20,000 microsteps stand.
- The test run's pruner did not prune: under `set -e` its first `find`, before `checkpoints/` existed, ended the
  loop. It now runs under `set +e`.
- 19:40 UTC: final dataset snapshot, identical to the live recording apart from the split files. 104 episodes,
  46,372 frames, layouts A 24, B 15, C 50, D 15. Converted per arm (all episodes; the run selects them) and
  uploaded (475 MB) to `/workspace/real_lora/pen_in_right_final`. That folder is also the held-out per-arm copy
  for the morning offline eval.
- Final split (`ws4_split.json` "final" at 19:40:18 UTC, copies on the pod and in `ws4/.cache/real_lora/`):
  - 82 train;
  - 14 held-out: A 20 to 22, B 36 to 38, C 85 to 88, D 99 and 101 to 103;
  - 8 dropped: 15, 16, 23, 28, 34, 43, 61, 100, which equal review.json's rejects.
- Base `/workspace/packages/so101_haller_pen_B`: variant B quantiles of the 82 train takes (38,030 frames).
- 19:43:00 UTC: first launch `/workspace/real_lora/lora_pen` (82 train, 14 eval). Stopped at 19:50 after
  7 min: at 19:43:05 ws4-vr2 froze the split at 12 held-out (the last 3 good takes per layout), which moved 85
  and 99 into train. The 82-take split files are kept as `ws4/.cache/real_lora/*_82.*`.
- **19:50:20 UTC: overnight run `/workspace/real_lora/lora_pen84` started on the frozen split.**
  - 84 train, 12 held-out (A 20 to 22, B 36 to 38, C 86 to 88, D 101 to 103), 8 dropped.
  - Base `/workspace/packages/so101_haller_pen_B84`: B quantiles of the 84 train takes (38,686 frames).
  - 20,000 microsteps, batch 2 x accum 4, checkpoints (raw + EMA) and eval loss on the 12 held-out takes
    (256 samples) every 500. LeRobot reports "84 train, 12 eval".
  - ETA about 02:35 UTC (04:35 CEST). Log `/workspace/logs/train_final84.out`.
  - Eval loss on the 12 held-out takes: 0.261, 0.249, 0.231, 0.231, 0.233 at 500 to 2,500 microsteps.
  - Held-out action-error eval of the 2.5k EMA adapter: done at about 20:53 UTC (`ws4/research/pen_lora.md`).
  - Pruner checked at 20:21: checkpoint 500 kept only its raw and EMA adapters, 1,000 also its training state;
    3.1 GB for both, so about 37 GB for the run.
- Learning curve (`ws4/research/pen_lora.md`), on the training pod:
  - `/workspace/real_lora/curve_watch.sh` evaluates the EMA adapters at 2.5k, 5k, 10k, 15k and 20k as they
    land, using `ws4/cloud/lora_adapter.py eval` (offline_diag.py on the 12 held-out takes, adapter folded in
    memory) and `/workspace/real_lora/eval_ckpt.sh`.
  - Each eval takes about 5 min next to training (17 GB, text encoder on the CPU).
  - Zero-shot B baseline done at 19:57.
- Robot-facing server (orchestrator, 20:00 UTC): on real-serve-6000, not the training GPU.
  - `bash /workspace/serve/pen_serve.sh 005000` runs one `lora_adapter.py serve` with two packages, as
    so101-zeroshot's `eval_pen.sh` wants: `base` (B84) and `lora_5000` (B84 + the 5k EMA adapter folded in
    memory). Port 8790, guidance 1.0 + compile, pen prompt pre-encoded, logs in `/workspace/serve/`; about 50 GB.
  - It starts when the 5k checkpoint lands. 10k, 15k and 20k replace it on the same port; each swap has about
    5 min of downtime and is announced to so101-zeroshot first. `pen_serve.sh stop` ends it when the arm
    session ends.
  - Tested at 20:05 to 20:08 UTC with `pen_serve.sh 000500`: READY in about 3 min.
    - Warm-up: `base` 50 s (first call, compile), `lora_500` 10 s.
    - Serve-pod GPU 81 of 96 GB with ws4-imagine's 8780 server.
    - From the laptop through an SSH tunnel: both packages 6-D, finite 42 x 6 chunks; round trip about 0.9 s,
      server 0.5 to 0.6 s (EUR-IS-1 is further from the laptop than EU-CZ-1, where the round trip was 0.48 s).
    - `lora_500` differs from `base` on identical requests (mean 0.83, max 3.0 in command units), so the adapter is
      applied.
    - Stopped at 20:10 UTC (`pen_serve.sh stop`), since no arm session can happen yet (the Haller is lent out).
      A running pen server counts as busy for the serve pod's watchdog.
    - Oscar (via the orchestrator, 20:20 UTC): keep real-serve-6000 and a pen server on 8790 up all night, so the
      arm can use the 5k checkpoint whenever it comes back; later checkpoints when the held-out curve says they
      are better. The serve pod's watchdog is now 8 h idle and counts `flux.remote` too.
    - Swaps keep the gap short: `pen_serve.sh <step>` starts `lora_adapter.py serve --replace-pid <old>`, which
      loads and folds the new models on the CPU while the old server keeps serving, then stops it and moves to
      the GPU. `/workspace/serve/swap_5k.sh` runs the 5k swap as soon as that adapter lands.
    - **5k swap, 21:37:53 to 21:43 UTC** (`swap_5k.sh`):
      - memory OK (97.2 GB available, 64.2 GB needed for 3 packages);
      - code `/workspace/serve/ws4`;
      - new models loaded and folded on the CPU, then the old server (pid 431318) replaced;
      - serving `base`, `lora_5000`, `median` since 21:42:05, warm-up done 21:43 (`base` 51 s with compile);
      - 50.2 GB on the GPU.
      - Laptop check: all three 6-D, chunk 42, finite; round trip 0.62 to 0.65 s, server 0.36 s.
      - so101-zeroshot's fake-robot rehearsal on it (about 21:50 UTC, nothing logged, no arm): base, median and
        lora_5000 ran through the runner in Oscar's order; round trip 0.66 to 0.69 s, server 0.36 to 0.37 s.
    - **4-package swap, 22:09 to 22:15 UTC:** `lora_5000_raw` added through `pen_extra.txt` (the orchestrator chose RAW 5k for
      the graded LoRA row). Memory OK: 97.2 GB available, 82.2 GB needed. Serving base, median, lora_5000,
      lora_5000_raw since 22:14:05; 66 GB. Laptop check: all 6-D, server 0.35 s. so101-zeroshot's fake-robot check
      passed on all four (graded order base, median, lora_5000_raw; warm-up round trip 0.73 to 0.80 s, server
      0.36 to 0.37 s).
    - **10k swap, 23:44 to 23:51 UTC:** RAW 10k beat RAW 5k on the held-out eval (arm MAE 3.15 vs 3.24 deg, direction
      88.6 vs 83.3 %, gripper direction 92 vs 62 %).
      - Announced to so101-zeroshot; no hold.
      - Serving base, median, lora_10000 (EMA) and lora_10000_raw since 23:50:01; 66 GB.
      - Laptop check: all 6-D, server 0.36 to 0.37 s.
      - RAW 10k is staged for the Hub, not uploaded.
    - Median normalization check (orchestrator, 20:30 UTC), on real-serve-6000 at 20:36 to 20:43 UTC:
      - `/workspace/packages/so101_haller_pen_M` (stats `/workspace/real_lora/a0r_M_median.json`): held-out eval and
        speed check, results in `ws4/research/pen_lora.md`.
      - NO-GO for a median-base LoRA: overall direction 69 % against the 80 % bar.
      - `median` joins 8790 at the 5k swap via `/workspace/serve/pen_extra.txt`; 3 packages need 64 GB available.
      - Staged, not launched: `/workspace/packages/so101_haller_pen_M84g` and `/workspace/real_lora/launch_penM.sh`.
    - Code check for imagination (orchestrator's request: serve from ws4-imagine's updated copy if default requests
      are unchanged), 20:35 to 20:45 UTC, `lora_500` on three servers, same command, 5 identical default requests:
      - current code (`/workspace/liminal/ws4`): bit-identical between 2-model 8790 and 1-model 8792, and on repeats;
      - `/workspace/serve/ws4` (ws4-imagine's copy plus `lora_adapter.py`): max 0.255, mean 0.019 off. `flux/` is
        identical; the new `imagine/` hooks the samplers (likely interplay with `compile_model`, unverified).
      - Lazy imagination itself works there: same commands as its default request, 0.68 s round trip, frames
        fetched in 0.78 s.
      - So the arm server stays on the current code (the orchestrator's fallback). `/workspace/serve/ws4` is kept
        for ws4-imagine.
      - Settled at 20:58 UTC, after ws4-imagine could not reproduce the drift. A fresh one-model server of the
        *current* code also differs from the live 8790, by exactly the same max 0.255 / mean 0.019. So the cause is
        compile-time variance: with `compile_model` on, each fresh process lands in one of two numeric states,
        whatever the code copy.
      - ws4-imagine's code does not change default commands; `imagine/` is not imported unless a request asks for
        imagination. `pen_serve.sh` now runs from the refreshed `/workspace/serve/ws4` (lazy imagination
        available), with `/workspace/liminal/ws4` as the fallback.
      - Every server restart may shift commands by up to about 0.25 (deg or gripper %), which is small next to the
        4 to 6 deg action errors.
    - so101-zeroshot's `eval_pen.sh` rehearsal (fake robot, nothing logged) passed against `base` and `lora_500`:
      tunnel, token, 6-D contract and warm-up OK; chunk round trip 0.71 to 1.04 s median, server 0.36 to 0.71 s.
    - Restarted 20:19 UTC for so101-zeroshot's fake-robot rehearsal, now with the text encoder on the CPU
      (`TE_CPU=1`, the new default): 34.6 GB for both packages instead of about 52. The serve-pod GPU was at
      93.4 of 97.9 GB with ws4-imagine's two servers (8780 stock B84, 8781 the test merge
      `so101_haller_pen_B_prov_lora500`, which ws4-imagine now uses, so keep it).
  - No merged 14 GB packages are written: the volume could hold only about one more.
- Watchdog: re-armed at 60 min idle from 19:40 while waiting for the split, back to 30 min at 19:47:38. The run
  keeps the pod busy, so it stops about 30 min after the run ends (about 03:00 UTC).
- Pod cost if it stops then: 18:24:44 to about 03:00 UTC, about 8.6 h, about $21 at $2.49/h. The 150 GB volume
  adds about $0.35 per day until deleted.
- Not done, at the safety check's request: removing the probe outputs (`/workspace/real_lora/lora_pen_probe_b2`,
  `_b8`), the test merge (`/workspace/packages/so101_haller_pen_B_prov_lora500`, 13.9 GB) and the probe base
  package. They are not needed for the run (91 of 150 GB used before it); delete them by hand if space runs short.
- Base variant B (quantiles from our demos, no transforms; gripper absolute as recorded), confirmed by
  so101-zeroshot: B and D tie over seeds 0 and 1 on the offline diagnostic, and B needs no flip in the data or
  at deployment.

LoRA step time (LeRobot `lerobot_train`, `examples/flux3/lora.json`, text encoder on the GPU, 480x640 h264 data):

| Micro-batch x accumulation, workers | Startup to first step | Steady state | Samples/s | Peak GPU memory |
|---|---|---|---|---|
| 2 x 4, 4 spawn workers | model load 1.2 min, first batch 1.8 min | 1.02 s per microstep (data wait 0.004 s) | 1.96 | 34.4 GiB |
| 8 x 1, 8 spawn workers | 1.0 min + 3.3 min for 8 workers | 4.11 s per microstep | 1.95 | 47.8 GiB |

- A bigger micro-batch gives no speed-up (compute-bound), so the run uses 2 x 4 with 4 workers.
- Eval pass (`max_eval_samples` 256): 60 to 95 s. A checkpoint is 0.45 GB raw + 0.45 GB EMA + 1.4 GB training
  state; `real_lora.sh train` keeps the training state only on the newest checkpoint.
- With eval and a checkpoint every 500 microsteps: about 3,000 microsteps (750 updates) per hour, so 20,000
  microsteps (5,000 updates, about 1.3 epochs of a 55-episode set) take about 6.7 h, about $17 at $2.49/h.

## Serving pod real-serve-6000 (orchestrator, 17 Oct 19:50 UTC)

| What | Value |
|---|---|
| Pod | `real-serve-6000`, id `6jfpc4cliq27jk`, 1x RTX PRO 6000 96 GB, EUR-IS-1, $2.49/h, container disk 40 GB |
| Volume | `real-train-is` (hnh342zg4i) at `/workspace`, shared with real-train-6000 (RunPod allowed the second attach) |
| SSH | `ssh root@157.157.221.177 -p 15377 -i ~/.ssh/id_ed25519_osrdvs` (changes on every start) |
| Watchdog | `pod_job_watchdog.sh 480` with `WATCH_JOBS=lora_adapter.py\|flux.remote\|imagine.replay_server\|warm_prompts\|offline_diag\|real_lora.sh\|ckpt_eval\|bootstrap.sh` (20:16 UTC), log `/workspace/logs/serve_watchdog.log`. Oscar, 17 Oct 22:16 CEST: keep the pod up all night so the arm can use the 5k checkpoint whenever it comes back (about $2.49/h overnight, approved) |

- Purpose: serving only, so the overnight LoRA keeps its GPU. Covers ws4-imagine's replay test (base
  `so101_haller_pen_B84`, held-out takes from `pen_in_right_final`), the real-arm checkpoint serving and the
  offline checkpoint evals.
- The volume's envs work on it as they are (torch 2.11 cu128, sm_120, NATTEN 0.21.6).
- Ports: 8780 and 8781 ws4-imagine (B84 replay tests), 8790 the real-arm checkpoint server (ws4-datasets: `base` + `lora_500` from 20:13 UTC, `lora_5000` from about 21:38 UTC).
- Also on this volume since 20:08 UTC: `act-train-4500` (id zhkilyhem41c2f, RTX PRO 4500, $0.72/h), the ACT baseline session's training pod (third attach of the volume).
- Spend on this RunPod account on 17 Oct as of 20:15 UTC: $13.38 billed; four pods running at $6.42/h.
- Writes go under `/workspace/serve/` and `/workspace/imagine/` only. `/workspace/liminal/ws4` is the training
  run's code, so other ws4 copies go to their own folders (for example `/workspace/imagine/ws4`).

## Per-skill LoRAs for the 4-skill demo chain (ws4-datasets, 18 Oct night)

Plan (orchestrator, 17 Oct about 22:30 UTC):
- Three new VR datasets, about 60 takes each: open the box (robot RIGHT arm, operator `left_*`), green pen and yellow
  cutter (robot LEFT arm, `right_*`). Each gets the pen recipe on its own RTX PRO 6000 pod in EUR-IS-1 on
  real-train-is.
- Pods: one per skill, $2.49/h each, 30 min job watchdog. Oscar approved the agent directly (18 Oct about 00:35
  CEST) to create each one when its dataset is ready, so the orchestrator does not create them.
  - Spec: copy of real-train-6000. SECURE cloud, `dataCenterIds` ["EUR-IS-1"], GPU "NVIDIA RTX PRO 6000 Blackwell
    Server Edition" x1, image `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, disk 40 GB, ports ["22/tcp"],
    network mount hnh342zg4i at /workspace.
  - Env: HF_HOME, HF_TOKEN (RunPod secret reference), LIMINAL, UV dirs, WS4_ROOT, and the same PUBLIC_KEY.
  - Created right after the skill's `prep` upload, then `skill_lora.sh train` arms the watchdog and starts the run.
- Tool `ws4/cloud/skill_lora.sh`:
  - `prep SKILL SRC SIDE`: laptop, low CPU, because the 30 Hz VR recording runs there. Checks the final split, reads
    the task string verbatim, and rsyncs the closed raw dataset to `/workspace/skills/SKILL/raw` under
    `nice -n 19`, `ionice -c3` and 2 threads;
  - `train SKILL HOST PORT`: on the skill's pod; per-arm conversion and B stats on the train takes, base
    `/workspace/packages/so101_haller_SKILL_B`, watchdog, `real_lora.sh train`, 5,000 microsteps, about 1 h 40 min;
  - `eval`: zero-shot B, RAW and EMA at 5k on the held-out takes, same protocol as the pen curve;
  - `report`: the table and the pick.
- Checked end to end on the closed pen dataset (pen_dryrun):
  - `prep` through the serve pod took 30 s wall and 3 s of laptop CPU for 475 MB, all 104 episodes arrived;
  - `train` with NO_TRAIN=1 on the serve pod converted, computed stats and built the package in 1.8 min, with
    quantiles identical to B84;
  - `eval` with DRY=1 generates a correct script.
- Split: ws4-vr2's `ws4_split.json` per dataset (last 3 good takes per box position held out).
- Serving:
  - 8790 keeps base, median, lora_5000 and lora_5000_raw until Oscar's graded pen session is done.
  - The four skill packages (about 66 GB, one 7B copy is about 16.5 GB with the text encoder on the CPU) go on a
    second serve pod that the orchestrator creates: the final-demo server.
  - No shared-trunk work tonight.
- Tests of `skill_lora.sh` run on the laptop CPU or the serve pod, never on a training GPU.
- Storage (orchestrator, 18 Oct about 01:10 CEST): the orchestrator grew real-train-is to 250 GB (about $0.23/day
  more) after it reached 109 of 150 GB.
  - Skill runs save every 1,000 microsteps, about 6 GB each.
  - `skill_lora.sh train` checks `du` first and keeps at least 40 GB free; 120 GB were used at 01:15 CEST.
  - Deleting the pen probes, `lora_pen_prov` and the test merge needs a person; the safety check blocks the agent.
  - The skills' `raw/` copies stay: the ACT session reuses them.
- Datasets (ws4-vr2): box `~/.cache/huggingface/lerobot/ws4/haller_open_the_box`, task "open the box by pulling its
  handle". Every new dataset records head, left_wrist and right_wrist (both wrists rotated 180). The pen arm's
  calibration was restored to the pen dataset's at 00:20 CEST, so skills 2 to 4 share it.
- Timing (UTC): box data about 23:30 to 00:00, LoRA done about 01:45; green about 02:45; yellow about 03:45.
- Hub uploads for these need Oscar's OK, as for the pen.
- 18 Oct 01:15 CEST (orchestrator): volume `real-train-is` grown from 150 to 250 GB. It held 109 GB with the pen run at 9k,
  and the rest of the pen run plus three 5k skill LoRAs would have passed 150 GB mid-checkpoint (the sim volume hit
  its quota that way on 17 Oct). About $0.23 per day more until the volume is deleted.

### open_box (robot RIGHT arm, `left_*`), 18 Oct
- Dataset `haller_open_the_box`: 75 episodes, 31,678 frames, task "open the box by pulling its handle".
- Split final at 01:25:06 CEST: positions P1 0 to 2, P2 3 to 38, P3 39 to 74. 59 train (P1 folded in), 6 held-out
  (P2 36 to 38, P3 72 to 74), 10 dropped.
- 23:25 UTC prep: 39 s wall, 3.6 s laptop CPU, 524 MB raw to `/workspace/skills/open_box/raw` through the serve
  pod.
- 23:26:41 UTC pod `skill-open-box-6000`, id `g4n67t55ac58jk`: 1x RTX PRO 6000 (18 vCPU), EUR-IS-1, $2.49/h,
  volume real-train-is.
- SSH `root@157.157.221.29 -p 58324`. Watchdog `pod_job_watchdog.sh 30` armed 23:27:42 UTC.
- 23:29:46 UTC LoRA `/workspace/skills/open_box/lora` started.
  - Conversion on the pod (--side left, scene head, wrist left_wrist), B stats on 59 train takes (24,609
    frames), base `so101_haller_open_box_B`, all within 2 min.
  - LeRobot reports 59 train / 6 eval. 1.25 s per microstep (pen pod 1.0 s); the first model load took 4 min
    on the cold host. ETA about 01:35 UTC.
- `/workspace/skills/eval_after_train.sh open_box 5000` waits on the pod for training to exit, then runs
  zero-shot B, RAW 5k and EMA 5k on the 6 held-out takes. The pod idle-stops 30 min after that.

### green_pen (robot LEFT arm, `right_*`), 18 Oct
- Dataset `haller_put_the_green_pen_in_the_green_slot`: 66 episodes, 28,832 frames, task "put the green pen in the
  green slot". One box position. The box arm (`left_*`) holds the lid at a constant pose.
- Split final: 58 train, 3 held-out (63 to 65), 5 dropped (14, 18, 25, 44, 50).
- 23:52 UTC prep: 26 s wall, 3.5 s laptop CPU, 420 MB raw.
- 23:52:56 UTC pod `skill-green-pen-6000`, id `uru7zrg7iun19g`: 1x RTX PRO 6000 (32 vCPU), EUR-IS-1, $2.49/h.
- SSH `root@157.157.221.30 -p 56923`. Watchdog armed 23:55:16 UTC.
- 23:56:47 UTC LoRA `/workspace/skills/green_pen/lora` started: LeRobot reports 58 train / 3 eval; stats on
  25,435 frames. Base `so101_haller_green_pen_B`.
- `eval_after_train.sh` (it now waits for the checkpoint first, then for training to exit) runs the held-out
  evals.

### One serve pod, used in sequence (orchestrator, 18 Oct about 02:05 CEST)
- RunPod billed $34.5 for 17 Oct. A second serve pod would push the hackathon past Oscar's $100, so real-serve-6000
  stays the only server.
- First Oscar's graded pen session (base, median, the best pen LoRA). Then one swap to the final-demo set:
  skill_open_box, skill_green_pen, skill_yellow_cutter, skill_red_pen, about 4 x 16.5 GB.
- `/workspace/serve/demo_serve.sh` does that swap.
  - Packages from `/workspace/serve/demo_packages.txt` (NAME=BASE+ADAPTER, filled in as each winner lands),
    prompts from `/workspace/serve/demo_prompts.txt`.
  - Same memory check and short-gap replacement as `pen_serve.sh`; DRY=1 checks only (passed at 00:05 UTC).
  - Not run until the orchestrator or Oscar says the graded session is done; announced to so101-zeroshot and
    ws4-orchestrator first.
- Skill pods idle-stop 30 min after their evals. real-train-6000's watchdog stops it after the pen run's 20k eval;
  terminating it is left to the orchestrator or Oscar.

### HOLD on new pods (Oscar via the orchestrator, 18 Oct about 02:05 CEST)
- Oscar: "I would hold on deploying another pod. My friend has credits on his account we should use instead of
  reloading mine".
- No new pods on this RunPod account, the yellow cutter's included, until the orchestrator says otherwise. The cutter
  gets `prep` (raw upload to the volume) as usual and stops before pod creation.
- Running pods keep running. GPU work left at 00:04 UTC:
  - pen: about 160 min, to about 02:45 UTC;
  - open_box: about 96 min, to about 01:41;
  - green_pen: about 110 min, to about 01:55.
  - Each pod idle-stops about 30 min after its last job.

### Budget stop (orchestrator, 18 Oct about 02:30 CEST): about $7 left on Oscar's account
- Stopped by the orchestrator: real-serve-6000 and real-train-6000. The pen run ends at 12,145 microsteps (last
  checkpoint 012000); RAW 10k stays the red-pen model, so there are no 15k/20k evals. Neither pod is restarted.
- Finishing: skill-open-box-6000 and skill-green-pen-6000 (5k LoRAs and held-out evals of RAW and EMA 5k). The
  zero-shot B evals are skipped to save budget (orchestrator, option (a)); the pen has the zero-shot
  comparisons.
- `/workspace/skills/stop_after_rescue.sh SKILL` on each pod stops it as soon as its evals have ended and the laptop
  copy is done (marker `/workspace/skills/SKILL/rescued`, at most 30 min wait), instead of after 30 idle minutes.
- Rescue to the laptop, `~/haller_rescue/`, by a nice/ionice rsync job:
  - pen: RAW and EMA 10k adapters, B84 and median normalizers, stats, split, evals, training log;
  - each skill: RAW and EMA 5k adapters, base normalizers, stats, split, evals, training log;
  - no 14 GB weights (they are the stock `black-forest-labs/flux-3-action-so101`).
- Serving and the cutter LoRA move to a friend's RunPod account later.

### yellow_cutter (robot LEFT arm, `right_*`), 18 Oct: prep only (hold on new pods)
- Dataset `haller_put_the_yellow_cutter_in_the_yellow_slot`: 70 episodes, 30,514 frames, task "put the yellow cutter in
  the yellow slot". One box position (P3).
- Split final: 59 train, 3 held-out (67 to 69), 8 dropped.
- 00:38 UTC prep: raw (440 MB) to `/workspace/skills/yellow_cutter/raw` through the open_box pod, 45 s, 3.8 s laptop
  CPU. No pod created.
- The friend's RunPod account needs its own network volume (volumes belong to one account and one data center):
  bootstrap (about 11 min with the weight prefetch), ws4/ sync, then `skill_lora.sh prep` with UPLOAD_HOST/PORT on
  that pod and `train`.
