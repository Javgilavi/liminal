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
