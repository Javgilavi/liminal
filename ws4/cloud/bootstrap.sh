#!/usr/bin/env bash
# WS4 GPU bootstrap for FLUX 3 Action on RTX PRO 6000 Blackwell (sm_120).
#
# Run on the cloud GPU box, as early as possible. The weight prefetch (~40 GB) starts first and runs in the
# background while the two Python environments build.
#
#   export HF_TOKEN=hf_...   # a real read token of the HF account that clicked "Agree and access" on
#                            # flux-3-action-base, -so101 and -droid (each repo is gated separately)
#   bash bootstrap.sh        # idempotent, safe to re-run
#
# On a machine whose largest GPU has under 16 GB (e.g. the 4 GB laptop) it only builds the environments for
# CPU tests and skips the weights. Serving cards such as a 32 GB RTX PRO 4500 do prefetch (FLUX SO-101 fits there
# with the Qwen text encoder on CPU). Force either way with PREFETCH=1 or PREFETCH=0.
#
# Produces:
#   $WS4_ROOT/lerobot        LeRobot main + flux3/peft extras (LoRA task adaptation, robot I/O)
#   $WS4_ROOT/flux-action    BFL's standalone repo (new embodiments, full fine-tune, predicted video)
#   HF cache                 flux-3-action-so101 + flux-3-action-base (VAE + Qwen text encoder)
#
# Versions follow flux-action docs/setup.md (H200 reference: CUDA 12.8, Python 3.12, torch 2.10.0+cu128,
# NATTEN 0.21.6). Blackwell workstation cards are sm_120: torch cu128 wheels support it, NATTEN kernel
# coverage for sm_120 is UNVERIFIED, so the GPU check runs a real NATTEN op to find out.
#
# Everything lives inside main() so bash parses the whole file before running it: editing this file while
# it runs cannot corrupt the running copy.

set -euo pipefail

WS4_ROOT="${WS4_ROOT:-$HOME/ws4}"
PY_VER="${PY_VER:-3.12}"
# Pinned to the commits the ws4/flux and ws4/imagine code was read and tested against. Override to move.
LEROBOT_REF="${LEROBOT_REF:-146d847}"
FLUX_ACTION_REF="${FLUX_ACTION_REF:-830f1ce}"
# Where the liminal checkout (with ws4/) lives on this box; push it from the laptop with ws4/cloud/push_ws4.sh.
LIMINAL="${LIMINAL:-$HOME/liminal}"
PREFETCH="${PREFETCH:-auto}"
TORCH_INDEX="https://download.pytorch.org/whl/cu128"
NATTEN_VER="0.21.6"
GATED_REPOS=(black-forest-labs/flux-3-action-so101 black-forest-labs/flux-3-action-base black-forest-labs/flux-3-action-droid)

step() { printf '\n=== %s (%s)\n' "$*" "$(date +%H:%M:%S)"; }
warn() { printf 'WARNING: %s\n' "$*"; }
hf() { uvx --quiet --from "huggingface_hub[hf_xet]" hf "$@"; }

natten_wheel() {  # prints the NATTEN wheel tag matching the torch inside the given venv python
  "$1" - <<'EOF'
import torch
v = torch.__version__.split("+")[0].split(".")
cu = (torch.version.cuda or "").replace(".", "")
print(f"torch{v[0]}{v[1]}{v[2] if len(v) > 2 else 0}cu{cu}")  # 2.10.0 + 12.8 -> torch2100cu128
EOF
}

checkout_ref() {  # $1 = dir, $2 = ref; fetches only when the ref is missing locally
  git -C "$1" cat-file -e "$2^{commit}" 2>/dev/null || git -C "$1" fetch --quiet origin
  git -C "$1" checkout --quiet "$2"
  git -C "$1" log -1 --format="$(basename "$1") @ %h %cd"
}

check_hf_access() {  # fail fast, before 40 GB of downloads, with an actionable message
  case "${HF_TOKEN:-}" in
    "" | hf_xxx | hf_... )
      echo "HF_TOKEN is not set to a real token (got '${HF_TOKEN:-}')."
      echo "1. With your HF account, open each repo and click 'Agree and access':"
      printf '     https://huggingface.co/%s\n' "${GATED_REPOS[@]}"
      echo "2. Create a read token at https://huggingface.co/settings/tokens"
      echo "3. export HF_TOKEN=hf_<your token> && bash $0"
      return 1 ;;
  esac
  hf auth whoami >/dev/null 2>&1 || { echo "HF_TOKEN is rejected by the Hub (hf auth whoami failed)."; return 1; }
  local probe repo denied=0
  probe=$(mktemp -d)
  for repo in "${GATED_REPOS[@]}"; do
    if hf download "$repo" README.md --local-dir "$probe" >/dev/null 2>&1; then
      echo "access ok: $repo"
    else
      echo "NO ACCESS: $repo  (click 'Agree and access' on https://huggingface.co/$repo with this token's account)"
      denied=1
    fi
  done
  rm -rf "$probe"
  return "$denied"
}

main() {
  mkdir -p "$WS4_ROOT"
  exec > >(tee -a "$WS4_ROOT/bootstrap.log") 2>&1

  step "Preflight"
  local max_mib=0
  if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,driver_version,compute_cap --format=csv
    max_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | tail -1)
  else
    warn "no nvidia-smi: CPU-only machine"
  fi
  if [ "$PREFETCH" = auto ]; then
    if [ "$max_mib" -ge 16000 ]; then PREFETCH=1; else PREFETCH=0; fi
  fi
  [ "$PREFETCH" = 1 ] || echo "largest GPU ${max_mib} MiB: building environments only, no weight download (PREFETCH=1 forces it)"
  local free_gb
  free_gb=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
  [ "$PREFETCH" = 0 ] || [ "$free_gb" -ge 150 ] || warn "only ${free_gb} GB free in \$HOME; weights + 2 envs + datasets want ~150 GB"
  if ! command -v ffmpeg >/dev/null; then
    warn "ffmpeg missing (apt-get install -y ffmpeg); LeRobot's torchcodec video backend uses the system FFmpeg"
  elif ! ffmpeg -hide_banner -decoders 2>/dev/null | grep -qiE 'libdav1d|libaom-av1|\bav1\b'; then
    warn "system ffmpeg has no AV1 decoder; LeRobot v3 videos are AV1, so torchcodec decoding will fail"
  fi
  if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi

  local prefetch_pid=""
  if [ "$PREFETCH" = 1 ]; then
    step "Hugging Face access"
    check_hf_access || exit 1
    step "Weight prefetch in the background (log: $WS4_ROOT/prefetch.log)"
    # Into the shared HF cache, not --local-dir: LeRobot resolves the frozen encoders from a pinned revision
    # of the base repo, and the cache reuses identical blobs across revisions instead of re-downloading.
    if [ ! -f "$WS4_ROOT/.prefetch_done" ]; then
      (
        set -e
        hf download black-forest-labs/flux-3-action-so101
        hf download black-forest-labs/flux-3-action-base
        # DROID: configs only (13.9 GB weights not needed unless we test the DROID policy).
        hf download black-forest-labs/flux-3-action-droid --include "*.json" --include "*.md"
        touch "$WS4_ROOT/.prefetch_done"
      ) > "$WS4_ROOT/prefetch.log" 2>&1 &
      prefetch_pid=$!
    else
      echo "already done"
    fi
  fi

  step "flux-action env"
  cd "$WS4_ROOT"
  [ -d flux-action ] || git clone https://github.com/black-forest-labs/flux-action
  checkout_ref "$WS4_ROOT/flux-action" "$FLUX_ACTION_REF"
  cd flux-action
  uv sync --extra encoders --extra data --extra serve
  local fa_py="$WS4_ROOT/flux-action/.venv/bin/python" fa_tag
  # NATTEN is not in uv.lock: reinstall after every `uv sync`.
  fa_tag=$(natten_wheel "$fa_py")
  echo "flux-action torch tag: $fa_tag"
  uv pip install --python "$fa_py" "natten==${NATTEN_VER}+${fa_tag}" --find-links https://whl.natten.org \
    || warn "no NATTEN wheel for $fa_tag; check https://whl.natten.org or build from source"

  step "LeRobot env (main @ $LEROBOT_REF, flux3 + peft extras, ws4 test deps)"
  cd "$WS4_ROOT"
  [ -d lerobot ] || git clone https://github.com/huggingface/lerobot
  checkout_ref "$WS4_ROOT/lerobot" "$LEROBOT_REF"
  cd lerobot
  [ -d .venv ] || uv venv --python "$PY_VER" .venv
  local lr_py="$WS4_ROOT/lerobot/.venv/bin/python" lr_tag
  uv pip install --python "$lr_py" -e ".[training,flux3,peft,diffusion]" \
    --extra-index-url "$TORCH_INDEX" --index-strategy unsafe-best-match
  lr_tag=$(natten_wheel "$lr_py")
  echo "lerobot torch tag: $lr_tag"
  uv pip install --python "$lr_py" "natten==${NATTEN_VER}+${lr_tag}" --find-links https://whl.natten.org \
    || warn "no NATTEN wheel for $lr_tag; check https://whl.natten.org or build from source"
  # ws4 code and tests run in this env: pytest for the suites, anthropic for the Claude judge.
  uv pip install --python "$lr_py" pytest anthropic matplotlib pyyaml

  step "GPU checks in both envs"
  local py
  for py in "$fa_py" "$lr_py"; do
    echo "--- $py"
    "$py" - <<'EOF' || echo "GPU check FAILED for this env"
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    archs = torch.cuda.get_arch_list()
    print("  sm_120 in torch arch list:", "sm_120" in archs, archs)
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  gpu{i}: {p.name}, sm_{p.major}{p.minor}, {p.total_memory / 2**30:.0f} GiB")
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    print("  bf16 matmul ok:", float((x @ x).float().abs().mean()) > 0)
    try:
        import natten
        import natten.functional as F
        print("  natten", natten.__version__)
        # Function names differ across NATTEN releases (UNVERIFIED for 0.21.6): try the known ones.
        fn = getattr(F, "na3d", None) or getattr(F, "neighborhood_attention_generic", None)
        q = torch.randn(1, 8, 16, 16, 4, 64, device="cuda", dtype=torch.bfloat16)  # [B, T, H, W, heads, dim]
        out = fn(q, q, q, kernel_size=(3, 7, 7))
        print("  natten 3D op ok on this GPU:", tuple(out.shape))
    except Exception as e:  # noqa: BLE001
        print("  natten GPU op check FAILED:", type(e).__name__, str(e)[:300])
        print("  (the VAE needs NATTEN; if this fails on sm_120, check whl.natten.org or build from source)")
    # This probes NATTEN's default backend only. The FLUX VAE picks its own backend (blackwell, hopper,
    # cutlass, then flex); on sm_120 expect cutlass-fna. ws4/imagine GPU_RUNBOOK `vaecheck` logs the real one.
try:
    from lerobot.policies.flux3 import Flux3Policy  # noqa: F401
    print("  lerobot Flux3Policy import ok")
except Exception as e:  # noqa: BLE001
    print("  lerobot Flux3Policy import:", type(e).__name__, str(e)[:200])
EOF
  done

  if [ -n "$prefetch_pid" ]; then
    step "Waiting for weight prefetch"
    if wait "$prefetch_pid"; then echo "prefetch ok"; else warn "prefetch failed, see $WS4_ROOT/prefetch.log"; exit 1; fi
  fi
  du -sh "${HF_HOME:-$HOME/.cache/huggingface}/hub" 2>/dev/null || true

  step "liminal checkout"
  if [ -d "$LIMINAL/ws4" ]; then
    echo "found $LIMINAL/ws4"
  else
    warn "no $LIMINAL/ws4 on this box. From the laptop run: bash ws4/cloud/push_ws4.sh <ssh-host>"
  fi

  step "Done"
  cat <<EOF
Next:
  - SO-101 smoke test with BFL's own example (read it first):
      cd $WS4_ROOT/flux-action && uv run --extra encoders python examples/so101/inference.py --help
  - Then follow ws4/flux/GPU_RUNBOOK.md (latency/VRAM, LoRA) and ws4/imagine/GPU_RUNBOOK.md (predicted video).
  - Log: $WS4_ROOT/bootstrap.log
EOF
}

main "$@"
