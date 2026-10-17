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
- `runpodctl` and a pod-scoped `RUNPOD_API_KEY` are present, so `idle_stop.sh` can stop the pod.
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

| Run | Median per chunk | p90 | Peak VRAM | Notes |
|---|---|---|---|---|
| default guidance, 2 warm-up passes | 1.46 s | 17.3 s | 23.6 GiB | load 6.4 min from the network volume; the p90 is one first-time CPU encode of a third prompt that the 2 warm-ups did not cover |

- **Pause math:** LeRobot's synchronous loop, and our per-arm runner, start the next inference only once the
  32-action queue is empty. So the arm stands still for the whole inference after each 1.07 s chunk, and at
  1.46 s it moves about 42% of the time.
- **Prompt warm-up:** `runpod_serve.sh` now encodes every prompt in `/workspace/prompts.txt` right after the
  server starts (`warm_prompts.py`), so no prompt is encoded live. The defaults are the recording protocol's
  S1 to S6. Edit the file so it matches the runner's strings exactly.
- **Memory:** Qwen stays on the CPU (added automatically on GPUs under 40 GB): 23.6 + 8.3 GiB would not fit
  in 31.9 GiB.

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
