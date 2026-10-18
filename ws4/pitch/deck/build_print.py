"""Build a print HTML of the deck source (one 1920x1080 page per slide) for headless Chrome's print-to-pdf.

Usage: python3 build_print.py ROOT, where ROOT holds deck/project (this folder's project/) and deck_img/ (the images
uploaded to the hosted deck). Then: google-chrome --headless=new --no-pdf-header-footer --allow-file-access-from-files
--virtual-time-budget=15000 --print-to-pdf=liminal_slides.pdf file://ROOT/pdf/print.html
"""
import json, re, sys
from pathlib import Path

SCR = Path(sys.argv[1])
deck = SCR / "deck" / "project"
img = SCR / "deck_img"
blobs = {
    "c5728b82844d582438fe23991149cbad": "haller_station.jpg",
    "7e2470a771aff813257d1218899b7d65": "zeroshot_red_strip.png",
    "0ce85c8f7e76ec1956d7dbdc179a9b65": "demo_pen_grasp.jpg",
    "6332d21c0ed3e2c926f45aac17b0d217": "photo_vs_sim.jpg",
    "f304bf99eed97d4fef7685ae2bb9ef97": "imagine_still.jpg",
    "4b97d00729199c4d1ad84ca8865a3f4a": "imagine_contact_ep101.png",
    "4348af4beaf83288b7da00352b00ea00": "imagine_lora5k_still.jpg",
    "b2977fb60b6c1471d479bba37a8a3038": "b84_vs_lora5k_ep101.png",
    "f25d638b8d67b23a39d12d7cd90de910": "wrist_view_recording_vs_now.jpg",
    "d309d6edc931a2e2483b0e5733225a67": "cover_haller_venue.jpg",
    "b0e129329e2b5f9a0b0e78c6d1be9b6b": "sim_venue_vs_sim.jpg",
}
order = json.loads((deck / "deck.json").read_text())["order"]


def connector(m):
    a = dict(re.findall(r'(\w[\w-]*)="([^"]*)"', m.group(1)))
    x1, y1, x2, y2 = (float(a[k]) for k in ("x1", "y1", "x2", "y2"))
    color = re.search(r"color:\s*(#[0-9a-fA-F]{3,6})", a.get("style", ""))
    color = color.group(1) if color else "#1c1830"
    head = a.get("head", "end") != "none"
    left, top = min(x1, x2) - 12, min(y1, y2) - 12
    w, h = abs(x2 - x1) + 24, abs(y2 - y1) + 24
    marker = f'<defs><marker id="m{int(x1)}{int(y1)}{int(x2)}{int(y2)}" markerWidth="4" markerHeight="4" refX="2" refY="2" orient="auto" markerUnits="strokeWidth"><path d="M0 0 L4 2 L0 4 Z" fill="{color}"/></marker></defs>' if head else ""
    me = f' marker-end="url(#m{int(x1)}{int(y1)}{int(x2)}{int(y2)})"' if head else ""
    return (f'<svg style="position:absolute;left:{left}px;top:{top}px;overflow:visible" width="{w}" height="{h}">{marker}'
            f'<line x1="{x1-left}" y1="{y1-top}" x2="{x2-left}" y2="{y2-top}" stroke="{color}" stroke-width="3" stroke-linecap="round"{me}/></svg>')


pages = []
for sid in order:
    html = (deck / "slides" / f"{sid}.html").read_text()
    html = re.sub(r"<aside>.*?</aside>", "", html, flags=re.S)
    html = re.sub(r"<!--.*?-->", "", html, flags=re.S)
    for bid, name in blobs.items():
        html = html.replace(f"/_blob/{bid}", (img / name).as_uri())
    html = re.sub(r'\sdata-video="[^"]*"', "", html)
    html = re.sub(r"<x-connector([^>]*)></x-connector>", connector, html)
    html = re.sub(r'<x-icon name="[^"]*" style="color:([^;]*);width:(\d+)px;height:(\d+)px"></x-icon>',
                  r'<div style="width:\2px;height:\3px;border-radius:50%;border:5px solid \1"></div>', html)
    pages.append(html)

css = """
@page { size: 1920px 1080px; margin: 0 }
* { margin: 0; box-sizing: border-box }
html, body { background: #fbf8f4 }
section { position: relative; width: 1920px; height: 1080px; overflow: hidden; break-after: page }
h1 { font-size: 96px; font-weight: 600; line-height: 1.1 } h2 { font-size: 64px; font-weight: 600; line-height: 1.15 }
h3 { font-size: 44px; font-weight: 600; line-height: 1.2 } p { font-size: 32px; line-height: 1.4 }
img { display: block }
table { border-collapse: collapse; width: 100% }
th, td { padding: 0.35em 0.6em; border-bottom: 1px solid #ddd7ea; text-align: left; vertical-align: top }
th { font-weight: 700; border-bottom: 2px solid #1c1830 }
"""
fonts = ('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800'
         '&family=JetBrains+Mono:wght@400;500;700&display=swap">')
out = SCR / "pdf" / "print.html"
out.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>liminal slides</title>{fonts}<style>{css}</style></head><body>"
               + "\n".join(pages) + "</body></html>")
print(out, len(pages))
