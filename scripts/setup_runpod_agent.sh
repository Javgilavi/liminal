#!/usr/bin/env bash
# Finish Runpod's official Codex onboarding from a normal, writable terminal.
# Reference: https://docs.runpod.io/agent-setup.md
# Installs skills and MCP registration only; never provisions a GPU or logs secrets.
set -euo pipefail

for program in node npx codex; do
  if ! command -v "$program" >/dev/null 2>&1; then
    printf 'Required program is missing: %s\n' "$program" >&2
    exit 1
  fi
done

node -e '
  const version = process.versions.node.split(".").map(Number);
  if (version[0] < 22 || (version[0] === 22 && version[1] < 20)) {
    process.stderr.write("Runpod setup requires Node.js 22.20 or newer.\n");
    process.exit(1);
  }
'

npx -y skills add runpod/runpod-plugins-official --skill '*' --agent codex --yes --global
npx -y skills list --global
codex plugin marketplace add https://github.com/runpod/runpod-plugins-official.git
codex plugin add runpod@runpod

# The official guide permits direct registration if bundled tools do not appear.
# Avoid dumping an existing server configuration because it might contain headers.
if ! codex mcp get runpod >/dev/null 2>&1; then
  codex mcp add runpod --url https://mcp.getrunpod.io/
fi

printf '\nRunpod files and MCP registration are installed.\n'
printf 'Fully quit and reopen Codex, then sign in to Runpod through its MCP connection.\n'
printf 'Alternatively run: codex mcp login runpod\n'
printf 'Authentication is not verified until listing your Pods succeeds.\n'
