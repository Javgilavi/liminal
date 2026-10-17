"""Survey LeRobot datasets on the Hugging Face Hub by fetching only their small metadata files.

Usage:
    python -I ws4/data/survey_hub.py --search xlerobot bi_so101 --out ws4/.cache/hub_meta
    python -I ws4/data/survey_hub.py --ids lerobot/svla_so101_pickplace --out ws4/.cache/hub_meta

For each dataset it downloads `meta/info.json` (LeRobot v2.x/v3.0) into `<out>/<owner>__<name>/info.json`
and writes one summary row per dataset to `<out>/summary.jsonl`. Nothing but JSON is downloaded and the
JSON is only parsed, never executed. Network access is required (this is a CLI, not imported by tests).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def summarize_info(repo_id: str, info: dict) -> dict:
    """Reduce a LeRobot info.json to the fields we rank on."""
    feats = info.get("features", {}) or {}
    cams = {}
    vectors = {}
    for key, f in feats.items():
        if not isinstance(f, dict):
            continue
        if f.get("dtype") in ("video", "image"):
            shape = f.get("shape") or []
            cams[key] = "x".join(str(s) for s in shape[:2])
        elif key.startswith(("observation.", "action")) and f.get("dtype", "").startswith("float"):
            names = f.get("names")
            if isinstance(names, dict):  # v2.0 style {"motors": [...]}
                names = next(iter(names.values()), None)
            vectors[key] = {"shape": f.get("shape"), "names": names}
    return {
        "id": repo_id,
        "codebase_version": info.get("codebase_version"),
        "robot_type": info.get("robot_type"),
        "fps": info.get("fps"),
        "episodes": info.get("total_episodes"),
        "frames": info.get("total_frames"),
        "tasks": info.get("total_tasks"),
        "hours": round(info["total_frames"] / info["fps"] / 3600, 3)
        if info.get("total_frames") and info.get("fps")
        else None,
        "cameras": cams,
        "vectors": vectors,
    }


def fetch_info(repo_id: str, out: Path, retries: int = 4) -> dict:
    """`repo_id` is `owner/name`, or `owner/name::sub/dir` for a dataset nested inside a collection repo."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, GatedRepoError, RepositoryNotFoundError

    repo, _, sub = repo_id.partition("::")
    dest = out / repo_id.replace("/", "__").replace("::", "___")
    cached = dest / "info.json"
    if cached.exists():
        return summarize_info(repo_id, json.loads(cached.read_text()))
    for attempt in range(retries):
        try:
            fname = f"{sub}/meta/info.json" if sub else "meta/info.json"
            p = hf_hub_download(repo, fname, repo_type="dataset", local_dir=dest / "_dl")
            info = json.loads(Path(p).read_text())
            dest.mkdir(parents=True, exist_ok=True)
            cached.write_text(json.dumps(info, indent=1))
            return summarize_info(repo_id, info)
        except (EntryNotFoundError, RepositoryNotFoundError):
            return {"id": repo_id, "error": "no meta/info.json (not LeRobot v2+/v3)"}
        except GatedRepoError:
            return {"id": repo_id, "error": "gated"}
        except Exception as e:  # rate limits and transient network errors
            msg = str(e)
            if "429" in msg or "Too Many" in msg:
                time.sleep(30 * (attempt + 1))
                continue
            if attempt == retries - 1:
                return {"id": repo_id, "error": msg[:200]}
            time.sleep(2)
    return {"id": repo_id, "error": "rate limited"}


def search_ids(queries: list[str], limit: int) -> list[str]:
    from huggingface_hub import HfApi

    api = HfApi()
    seen: dict[str, None] = {}
    for q in queries:
        for d in api.list_datasets(search=q, limit=limit, sort="downloads"):
            seen.setdefault(d.id, None)
    return list(seen)


def collection_ids(repo: str) -> list[str]:
    """Sub-datasets of a collection repo such as lerobot/community_dataset_v3 (one meta/info.json each)."""
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(repo, repo_type="dataset")
    return [f"{repo}::{f[: -len('/meta/info.json')]}" for f in files if f.endswith("/meta/info.json")]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--search", nargs="*", default=[], help="Hub search terms")
    ap.add_argument("--ids", nargs="*", default=[], help="explicit dataset repo ids")
    ap.add_argument("--collection", default=None, help="repo holding many LeRobot datasets in subfolders")
    ap.add_argument("--limit", type=int, default=400, help="max results per search term")
    ap.add_argument("--out", type=Path, default=Path("ws4/.cache/hub_meta"))
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)

    ids = args.ids + search_ids(args.search, args.limit)
    if args.collection:
        ids += collection_ids(args.collection)
    ids = list(dict.fromkeys(ids))
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"{len(ids)} datasets", file=sys.stderr)
    with ThreadPoolExecutor(args.workers) as ex:
        rows = list(ex.map(lambda r: fetch_info(r, args.out), ids))
    with (args.out / "summary.jsonl").open("a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    ok = sum("error" not in r for r in rows)
    print(f"{ok}/{len(rows)} with info.json -> {args.out / 'summary.jsonl'}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
