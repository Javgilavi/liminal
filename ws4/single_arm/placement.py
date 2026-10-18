"""Live placement page for the start layouts: the held-out take's start frame next to the live scene camera, with
the reference's edges drawn over the live view, so the box and pen can be pushed onto their recorded positions.

OpenCV on the laptop is built without a GUI, so the view is a local web page (http://127.0.0.1:8899) that reloads
the composite a few times a second. The runner feeds it frames while it waits for Enter; nothing here touches the
arm. ``ref_shift`` reports how far the live background sits from the reference (camera moved?).
"""

from __future__ import annotations

import http.server
import threading
from pathlib import Path

import numpy as np

PAGE = b"""<!doctype html><html><head><meta charset="utf-8"><title>placement</title>
<style>body{margin:0;background:#111;color:#ddd;font:14px sans-serif}img{width:100%;display:block}</style></head>
<body><img id="v" src="frame.jpg"><script>
const v=document.getElementById('v');setInterval(()=>{v.src='frame.jpg?t='+Date.now()},300);
</script></body></html>"""


def load_refs(ref_dir: Path) -> dict[str, list[Path]]:
    """``<layout>_ep<NN>_head.jpg`` -> {layout: [paths in take order]}."""
    refs: dict[str, list[Path]] = {}
    for path in sorted(Path(ref_dir).glob("*_ep*_head.jpg")):
        refs.setdefault(path.name.split("_")[0], []).append(path)
    return refs


def read_rgb(path: Path) -> np.ndarray:
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def composite(live: np.ndarray, ref: np.ndarray, label: str) -> np.ndarray:
    """[reference | live with the reference's edges in green] over [50/50 blend], RGB."""
    import cv2

    h, w = ref.shape[:2]
    if live.shape[:2] != (h, w):
        live = cv2.resize(live, (w, h))
    edges = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY), (5, 5), 0), 40, 120) > 0
    marked = live.copy()
    marked[edges] = (0, 255, 0)
    blend = (0.5 * live.astype(np.float32) + 0.5 * ref.astype(np.float32)).astype(np.uint8)
    top = np.concatenate([ref, marked], axis=1)
    bottom = np.concatenate([blend, np.zeros_like(blend)], axis=1)
    out = np.ascontiguousarray(np.concatenate([top, bottom], axis=0))
    for i, text in enumerate((label, "left: reference start | right: live + its edges",
                              "bottom: 50/50 blend", "align box and pen, then Enter in the terminal")):  # fmt: skip
        cv2.putText(out, text, (w + 12, h + 30 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 1)
    return out


BASE_BAND = (0.78, 1.0)  # head camera rows showing the robot's own base plate: fixed to the mast-mounted camera


def ref_shift(live: np.ndarray, ref: np.ndarray, band: tuple[float, float] = BASE_BAND) -> tuple[float, float, float]:
    """Euclidean ECC alignment of live onto ref inside a band of rows, on a half-size grayscale: (dx px, dy px,
    rotation deg) at full size. On the head camera the default band is the robot's base plate, which is fixed to the
    mast camera whatever the arm, box or pen do, so a shift of more than a few px means the camera mount moved."""
    import cv2

    def gray(img: np.ndarray) -> np.ndarray:
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return cv2.resize(g, (g.shape[1] // 2, g.shape[0] // 2)).astype(np.float32) / 255

    if live.shape[:2] != ref.shape[:2]:
        live = cv2.resize(live, (ref.shape[1], ref.shape[0]))
    g_ref, g_live = gray(ref), gray(live)
    mask = np.zeros(g_live.shape, np.uint8)
    mask[int(band[0] * mask.shape[0]):int(band[1] * mask.shape[0])] = 1
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-5)
    _, warp = cv2.findTransformECC(g_ref, g_live, warp, cv2.MOTION_EUCLIDEAN, criteria, mask, 5)
    return 2 * float(warp[0, 2]), 2 * float(warp[1, 2]), float(np.degrees(np.arctan2(warp[1, 0], warp[0, 0])))


class PlacementView:
    """Serves the latest composite on 127.0.0.1 in a daemon thread; ``update`` is cheap enough to call at 5 Hz."""

    def __init__(self, port: int = 8899):
        self._jpeg = b""
        self._lock = threading.Lock()
        view = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 (http.server API)
                body, kind = (PAGE, "text/html") if self.path in ("/", "/index.html") else (view.jpeg(), "image/jpeg")
                self.send_response(200)
                self.send_header("Content-Type", kind)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args: object) -> None:
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.url = f"http://127.0.0.1:{port}/"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def jpeg(self) -> bytes:
        with self._lock:
            return self._jpeg

    def update(self, live: np.ndarray, ref: np.ndarray, label: str) -> np.ndarray:
        return self.show(composite(live, ref, label))

    def show(self, img: np.ndarray) -> np.ndarray:
        """Serve an already composed RGB image (align_station.py builds its own)."""
        import cv2

        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self._lock:
                self._jpeg = buf.tobytes()
        return img

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
