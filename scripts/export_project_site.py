"""Stage only the reviewed static project; never copy source profiles or raw runs."""

import argparse
import shutil
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

PAGES = {
    ".nojekyll",
    "README.md",
    "index.html",
    "live.html",
    "style.css",
    "project.css",
    "project.js",
    "live.css",
    "live.js",
    "viewer.js",
    "scene-preview.js",
    "evidence/results.json",
    "evidence/notes.html",
}
ASSETS = {
    "liminal-overview.mp4",
    "preview-liminal-overview.jpg",
    "real-cutter-trial.mp4",
    "preview-real-cutter-trial.jpg",
    "paper-texture.svg",
    "preview-zero-shot-red.jpg",
    "preview-lora-red.jpg",
    "preview-real-prediction.jpg",
    "preview-flux-session.jpg",
    "preview-perception-session.jpg",
    "preview-predicted-future.jpg",
    "preview-real-insert-A.jpg",
    "preview-real-insert-B.jpg",
    "preview-real-insert-C.jpg",
    "preview-real-morning-A.jpg",
    "preview-real-morning-C.jpg",
    "real-insert-A.mp4",
    "real-insert-B.mp4",
    "real-insert-C.mp4",
    "real-morning-A.mp4",
    "real-morning-C.mp4",
    "real-prediction.mp4",
    "real-insert-poster.jpg",
    "station-offset.jpg",
    "MODEL-LICENSE.txt",
    "architecture.png",
    "demo-poster.jpg",
    "demonstration.mp4",
    "favicon.svg",
    "flux-session.mp4",
    "gripper-comparison.png",
    "kinematics.png",
    "lora-red.mp4",
    "loss.png",
    "perception-session.mp4",
    "predicted-future.mp4",
    "real-head.jpg",
    "real-scene.jpg",
    "real-vs-sim.jpg",
    "real-wrist.png",
    "replay-left-wrist.mp4",
    "replay-right-wrist.mp4",
    "replay.json",
    "scene.json.gz",
    "setup.png",
    "real-setup-twin.jpg",
    "twin-scene.png",
    "zero-shot-red.mp4",
}
VENDOR = {
    "OrbitControls.js",
    "THREE-LICENSE.txt",
    "dm-sans-normal.woff2",
    "dmsans-OFL.txt",
    "instrument-serif-italic.woff2",
    "instrument-serif-normal.woff2",
    "instrumentserif-OFL.txt",
    "three.core.js",
    "three.module.js",
}


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name in ("href", "src") and value:
                self.links.append(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new staging directory; must not exist")
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1] / "site"
    files = PAGES | {"assets/" + a for a in ASSETS} | {"assets/vendor/" + a for a in VENDOR}
    for name in sorted(files):
        path = source / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Missing or linked public asset: {name}")
        if path.stat().st_size > 25 * 1024 * 1024:
            raise ValueError(f"Public asset exceeds 25 MiB: {name}")
        if path.suffix in (".html", ".js", ".css", ".json", ".md"):
            text = path.read_text()
            if any(marker in text for marker in ("/home/jgilaviles/", "ssh.runpod.io", "ghp_", "PRIVATE KEY-----")):
                raise ValueError(f"Private machine or credential marker in {name}")
        if path.suffix == ".html":
            links = Links()
            links.feed(path.read_text())
            for value in links.links:
                url = urlsplit(value)
                if url.scheme or url.netloc or not url.path:
                    continue
                target = (path.parent / unquote(url.path)).resolve()
                if target.is_dir():
                    target /= "index.html"
                relative = str(target.relative_to(source))
                if relative not in files:
                    raise ValueError(f"Unexported local link in {name}: {value}")
    args.output.mkdir(parents=True, exist_ok=False)
    for name in sorted(files):
        destination = args.output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, destination)
    print(f"Staged {len(files)} reviewed public files in {args.output}")


if __name__ == "__main__":
    main()
