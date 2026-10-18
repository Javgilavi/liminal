"""Record size on disk, licence, revision and data-file list of Hub datasets (one API call each, nothing downloaded).

Usage:
    python -I ws4/data/hub_sizes.py MrC4t/bi_so_bag zonglin11/xlerobot_clean_table --out ws4/.cache/hub_sizes.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def dataset_sizes(repo_id: str) -> dict:
    from huggingface_hub import HfApi

    for _ in range(4):
        try:
            info = HfApi().dataset_info(repo_id, files_metadata=True)
            break
        except Exception as e:  # rate limits
            if "429" not in str(e):
                return {"error": str(e)[:200]}
            time.sleep(60)
    else:
        return {"error": "rate limited"}
    tot = {"data": 0, "videos": 0, "meta": 0, "other": 0}
    files = {}
    for s in info.siblings:
        top = s.rfilename.split("/")[0]
        tot[top if top in tot else "other"] += s.size or 0
        files[s.rfilename] = s.size
    card = info.card_data.to_dict() if info.card_data else {}
    return {
        "bytes": tot,
        "license": card.get("license"),
        "gated": info.gated,
        "sha": info.sha,
        "last_modified": str(info.last_modified)[:10],
        "downloads": info.downloads,
        "data_files": {k: v for k, v in files.items() if k.startswith("data/")},
        "n_video_files": sum(1 for k in files if k.startswith("videos/")),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--out", type=Path, default=Path("ws4/.cache/hub_sizes.json"))
    args = ap.parse_args(argv)
    out = json.loads(args.out.read_text()) if args.out.exists() else {}
    for rid in args.ids:
        out[rid] = r = dataset_sizes(rid)
        if "bytes" in r:
            b = r["bytes"]
            print(f"{rid:<70} {sum(b.values()) / 1e9:7.2f} GB  data {b['data'] / 1e6:8.1f} MB  lic={r['license']}")
        else:
            print(rid, r)
    args.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
