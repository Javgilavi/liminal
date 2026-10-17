# ws4/ttt: RoboTTT-style test-time-training memory for FLUX 3 Action

CPU-testable PyTorch prototype that goes with `ws4/research/robottt_flux.md`. Standalone: no imports from
`src/liminal`. Needs `torch` (and `einops` for the flux-action hook tests).

| File | What |
|---|---|
| `ttt_layer.py` | `TTTLayer`: multi-head fast weights (linear or 2-layer GELU MLP), KVB inner loss, mini-batch inner updates, learnable `W0` and per-token inner learning rate, hand-written inner gradients. `TanhGate`, `GatedTTT` (RoboTTT Eq. 3), `detach_state` |
| `temporal.py` | `TemporalTTTBlock`: per-timestep transformer block + gated TTT across timesteps (one timestep per inner mini-batch). Helpers: truncated BPTT slices, sequence action forcing noise levels, masked sequence loss |
| `flux_hook.py` | `FluxTTTAdapter`: hooks gated TTT layers after chosen joint blocks of flux-action's `JointSingleSeq` on the action-stream tokens, with commit semantics for denoising and TBPTT. `FluxFeatureTap`: per-re-plan FLUX features for a downstream memory. Pinned to flux-action@830f1ce |
| `toy_recall.py` | Toy long-horizon task: pretrained memory-less policy, wrapped with a gated TTT layer, fine-tuned with truncated BPTT, evaluated on 64x longer gaps |
| `progress_memory.py` | Pipeline P2: `EpisodeProgressMemory`, a TTT memory over one FLUX feature per re-plan that predicts the current subtask and a done flag for the orchestrator; `fit`, streaming `step`, and a synthetic aliased-stages check |
| `tests/` | 30 tests: inner gradients vs autograd, exact recall, streaming = one pass, causality, gate identity, TBPTT gradient cut, and the adapter on a tiny instance of flux-action's real DiT (packed training = streaming inference) |

```bash
python -m pytest ws4/ttt/tests -q          # about 10 s on a laptop CPU
python ws4/ttt/toy_recall.py --quick       # smoke run
python ws4/ttt/toy_recall.py               # full toy run, writes ws4/ttt/results/toy_recall.json
python ws4/ttt/progress_memory.py          # synthetic check of the P2 episode memory
```

The flux-action tests import `ws4/.cache/flux-action/src` (gitignored clone of
`black-forest-labs/flux-action@830f1ce`) and are skipped when it is absent. To get it:

```bash
git clone https://github.com/black-forest-labs/flux-action ws4/.cache/flux-action
git -C ws4/.cache/flux-action checkout 830f1ce178d7aea52f18fde31d8e8e36ce6bae21
```

Not done here: anything with real FLUX weights or a GPU (see the run checklist in the report).
