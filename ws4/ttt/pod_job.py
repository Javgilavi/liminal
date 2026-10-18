"""One model load, two GPU jobs: the 7B scale check, then FLUX feature extraction (for a time-boxed pod slot).

    PYTHONPATH=. python -m ttt.pod_job --data /workspace/ttt/sim --text-encoder-on-cpu \\
        --override guidance_scale=1.0 --override guidance_scale_action=1.0 --scale-out /workspace/ttt/scale_check.json

Same code as ``flux_scale_check.py`` and ``extract_flux_features.py``; loading the package once saves its
roughly 6 min load. The scale check runs with the same overrides as the extraction (hooks need the eager DiT).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--checkpoint", default="black-forest-labs/flux-3-action-so101")
    ap.add_argument("--override", action="append", default=[])
    ap.add_argument("--text-encoder-on-cpu", action="store_true")
    ap.add_argument("--adapter-layers", type=int, nargs="+", default=[24, 25, 26, 27])
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--layers", type=int, nargs="+", default=[16, 27])
    ap.add_argument("--scale-out", type=Path, default=Path("/workspace/ttt/scale_check.json"))
    ap.add_argument("--skip-scale-check", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)

    from flux.backends import LeRobotFlux3Backend

    from ttt.extract_flux_features import GROUPS, episode_requests, extract_episodes, streams_for
    from ttt.flux_hook import FluxFeatureTap
    from ttt.flux_scale_check import run_scale_check

    overrides = {"compile_model": False}
    for kv in args.override:
        k, v = kv.split("=", 1)
        overrides[k] = json.loads(v) if v[:1] in "0123456789-[{tfn" else v
    t0 = time.time()
    backend = LeRobotFlux3Backend.from_pretrained(
        args.checkpoint, device=args.device, overrides=overrides, text_encoder_on_cpu=args.text_encoder_on_cpu
    )
    load_s = time.time() - t0
    eps = sorted(p.parent for p in args.data.glob("*/episode.json") if not (p.parent / "features_flux.npz").exists())
    print(f"loaded in {load_s:.0f}s; {len(eps)} episodes to extract; overrides {overrides}", flush=True)

    if not args.skip_scale_check and eps:
        reqs = list(episode_requests(eps[0], backend.contract))
        report = run_scale_check(backend, reqs[len(reqs) // 2][1], args.adapter_layers, 512, args.runs)
        report.update(
            checkpoint=args.checkpoint,
            overrides=overrides,
            load_s=load_s,
            observation=f"{eps[0].name} re-plan {len(reqs) // 2}",
        )
        args.scale_out.parent.mkdir(parents=True, exist_ok=True)
        args.scale_out.write_text(json.dumps(report, indent=1))
        print(
            "SCALE_CHECK " + json.dumps({k: v for k, v in report.items() if k not in ("plain", "with_adapter")}),
            flush=True,
        )
        print(
            f"plain median {report['plain']['median_s']:.3f}s, with adapter {report['with_adapter']['median_s']:.3f}s",
            flush=True,
        )

    dit, modality = backend.policy.dit, backend.policy.modality
    taps = [
        (layer, g, FluxFeatureTap(layer, streams_for(g, modality)).attach(dit)) for layer in args.layers for g in GROUPS
    ]
    log = extract_episodes(backend, taps, eps, args.checkpoint, args.layers, GROUPS, overrides)
    log["load_s"] = load_s
    log["total_s"] = round(time.time() - t0, 1)
    (args.data / f"extract_flux_log_{int(t0)}.json").write_text(json.dumps(log, indent=1))
    print(f"EXTRACT_DONE in {log['total_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
