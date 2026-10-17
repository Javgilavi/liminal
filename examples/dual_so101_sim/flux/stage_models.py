"""Stage the mounted HF cache on container disk for fast safetensors loading."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path


def main() -> None:
    source_root = Path('/workspace/flux3-action/hf-cache/hub')
    destination_root = Path('/opt/flux3-action/hf-cache/hub')
    if not source_root.is_dir():
        raise SystemExit('Download the models before staging their cache.')
    destination_root.mkdir(parents=True, exist_ok=True)
    marker = destination_root.parent / '.complete'
    marker.unlink(missing_ok=True)
    started = time.monotonic()
    for source in source_root.rglob('*'):
        relative = source.relative_to(source_root)
        if '.locks' in relative.parts:
            continue
        target = destination_root / relative
        if source.is_symlink():
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.is_symlink():
                target.symlink_to(source.readlink())
            elif target.readlink() != source.readlink():
                raise RuntimeError(f'Unexpected cache link: {target}')
        elif source.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif source.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            size = source.stat().st_size
            if target.is_file() and target.stat().st_size == size:
                continue
            if shutil.disk_usage(destination_root).free < size + 2 * 2**30:
                raise SystemExit(f'Insufficient container disk space to stage {relative}')
            print(f'Staging {relative}: {size / 2**30:.2f} GiB', flush=True)
            temporary = target.with_name(target.name + '.staging')
            shutil.copyfile(source, temporary)
            temporary.replace(target)
    seconds = round(time.monotonic() - started, 1)
    marker.write_text(json.dumps({'source': str(source_root), 'seconds': seconds}))
    print(f'LOCAL_MODEL_CACHE_COMPLETE in {seconds}s', flush=True)


if __name__ == '__main__':
    main()
