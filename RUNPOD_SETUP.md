# Runpod agent setup

Source guide: https://docs.runpod.io/agent-setup.md

## Current connection status

**Connected and authenticated in Codex.** A later session now loads the Runpod
plugin and MCP tools. `codex mcp list` reports the hosted server enabled with
OAuth; a successful `list-pods` call independently verified account access and
returned one running Pod (`2uypll8r50gqc9`, NVIDIA A100-SXM4-80GB). No resources
were created, changed or stopped by this verification. CLI/Flash credentials are
not implied by MCP OAuth authentication.

The unqualified `npx @runpod/mcp-server@latest add` command still exits in this
non-interactive shell; its installer does not list Codex among its supported
clients. No additional installer run is needed for the now-working Codex MCP
connection. The sections below preserve the original local setup and optional
normal-terminal installation instructions.

This setup installs agent skills and configures Runpod's MCP connection. It does
not create a Pod, rent a GPU, deploy an endpoint, or install runpodctl/Flash.

## Initial setup verification

- Node.js `v22.23.3` meets the guide's `22.20+` prerequisite.
- Codex CLI `0.163.0` is installed and exposes plugin and MCP commands.
- All **15** official skills are installed at `.agents/skills/` in this repository:
  `runpod`, `runpod-mcp`, `runpodctl`, `flash`, `runpod-usage`,
  `runpod-templates`, `companion-clis`, `runpod-migrate`, `discovery`,
  `lifecycle-crud`, `serverless-deploy`, `pod-deploy`, `pod-doctor`,
  `endpoint-ops`, and `cost-audit`.
- Source revision: `ef53bfd056ba183d46dc31dfa54bf784cad5bc3d` in
  https://github.com/runpod/runpod-plugins-official (plugin manifest version 1.7.4).
- Project configuration `.codex/config.toml` declares the hosted MCP URL
  `https://mcp.getrunpod.io/` without credentials. Codex only loads project
  configuration in trusted projects.
- During initial setup, `codex mcp get runpod --json` reported no loaded server.
  Global registration/restart and sign-in were pending then; the current
  connection check above supersedes that earlier state.

The normal global destinations `~/.codex` and `~/.agents` are not writable from
this managed session. Therefore the guide's **global plugin install and global
server registration were not performed here**. The local skill copies are the
same official skill directories; they were installed with Codex's skill-installer
helper using an explicit project destination. They become available when Codex
loads skills for this repository in a new turn/session; this turn has not gained
new Runpod MCP tools.

## Complete setup from your terminal

Run this from the repository in a normal terminal, outside this agent's restricted
shell:

```bash
cd /home/jgilaviles/erh_hack/liminal
bash scripts/setup_runpod_agent.sh
```

The script runs the guide's Codex skills/marketplace/plugin commands and adds the
hosted server directly if no named Runpod entry exists. It never launches an
interactive sign-in process. It does not change existing MCP servers.

Then **fully quit and reopen Codex**. Sign in to the Runpod MCP connection in the
client, or start the browser sign-in from a normal terminal:

```bash
codex mcp login runpod
```

Do not paste a Runpod key into chat or this document. No `RUNPOD_API_KEY` was
present in the process environment; the interactive guide uses OAuth instead.
On a new installation, authentication remains pending until the browser flow
completes. Afterward ask the agent to list Pods. An empty successful response also
verifies connectivity. This installation has now passed that check.

## Local files and verification

`.agents/` and `.codex/` are personal local configuration and are gitignored.
`scripts/setup_runpod_agent.sh` and this document can be reviewed/committed with
the project if desired. The installer script is syntax-checked locally; its
global install commands must run in your writable terminal.

No inference or hardware result follows from the onboarding checks alone. The
subsequent inference work below uses the user's existing GPU Pod. The existing
harness continuation document is `WS3_HANDOFF.md`; newer work from the other
active development session was preserved during Runpod setup.

## Existing Pod: real FLUX with standalone MuJoCo

On 2026-10-17, real `black-forest-labs/flux-3-action-so101` inference was set up on
the existing A100 Pod `2uypll8r50gqc9`. The standalone browser simulator in
`../dual_so101_sim/flux/` connects through a local SSH tunnel on port **18100** to
the Pod's loopback inference port **8100**. Browser UI: **http://127.0.0.1:8000**.

See [the simulator's Runpod guide](../dual_so101_sim/flux/RUNPOD_FLUX.md) for exact
model/source revisions, repeatable launch helpers, setup and stop commands, and
the saved real-inference smoke results. Start it from that simulator directory:

```bash
bash runpod.sh start
bash runpod.sh tunnel
bash runpod.sh ui
```

A 64-tick test applied real FLUX targets to the right simulated arm and exercised
two action chunks (approximately 1.9 and 1.1 seconds of chunk inference). Full wall
time was 113 seconds; start with 1–2 simulated seconds in the UI. The server now
resizes each camera to the checkpoint's 256×256 feature shape before history.

This verifies the standalone prompt-to-FLUX simulation path. It does not establish
pick-and-place success, Liminal harness integration, simultaneous bimanual policy
execution or hardware operation. No Claude key is needed for this direct prompt
test. The existing Pod is left running; stopping the model process does not stop
Pod billing.

The source snapshot is also versioned under
[`examples/dual_so101_sim/`](examples/dual_so101_sim/README.md), including the
FLUX server, browser app, Runpod helpers and licensed robot meshes. Environments,
weights, private research notes and `.runpod/` connection metadata are excluded.
The local UI, tunnel and cloud FLUX process were subsequently stopped at the
user's request; restart manually using the commands above. The Runpod Pod stays
running. The launcher checks the managed SSH tunnel and real CUDA-backed FLUX
health before starting the simulator.
