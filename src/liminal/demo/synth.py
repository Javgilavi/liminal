"""Deterministic synthetic demo frames (stdlib PNG writer; no image dependency).

``python -m liminal.demo.synth [configs/demos]`` (re)writes ``frames/*.png`` for the shipped demos. The
pictures are schematic top-down scenes (table, containers, coloured blocks, a hand); they exist so the
frame pipeline, ids and provenance can be exercised offline. They are not realistic video.
"""

from __future__ import annotations

import struct
import sys
import zlib
from pathlib import Path

W, H = 160, 96
TABLE = (176, 154, 120)
WHITE = (236, 236, 236)
INNER = (206, 206, 206)
BIN = (96, 116, 160)
BIN_INNER = (72, 90, 128)
SKIN = (232, 184, 142)
BLOCK = {"red": (206, 40, 40), "blue": (40, 72, 206), "green": (44, 168, 72)}
BLOCK_SIZE = 14


class Canvas:
    def __init__(self, bg: tuple[int, int, int] = TABLE):
        self.px = bytearray(bytes(bg) * (W * H))

    def rect(self, x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        for y in range(max(0, y0), min(H, y1)):
            for x in range(max(0, x0), min(W, x1)):
                self.px[(y * W + x) * 3:(y * W + x) * 3 + 3] = bytes(color)

    def container(self, x0: int, y0: int, x1: int, y1: int, outer: tuple[int, int, int],
                  inner: tuple[int, int, int]) -> None:
        self.rect(x0, y0, x1, y1, outer)
        self.rect(x0 + 4, y0 + 4, x1 - 4, y1 - 4, inner)

    def block(self, color: str, x: int, y: int) -> None:
        self.rect(x, y, x + BLOCK_SIZE, y + BLOCK_SIZE, BLOCK[color])

    def hand(self, x: int, y: int) -> None:
        self.rect(x, y + 3, W, y + 9, SKIN)  # forearm from the right edge
        self.rect(x - 2, y, x + 12, y + 12, SKIN)

    def png(self) -> bytes:
        raw = b"".join(b"\x00" + bytes(self.px[y * W * 3:(y + 1) * W * 3]) for y in range(H))

        def chunk(tag: bytes, data: bytes) -> bytes:
            return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

        return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


# -- two_blocks_demo: a human puts the red block, then the blue block, in the box -------------

_RED_T, _BLUE_T = (18, 62), (48, 62)
_RED_IN, _BLUE_IN = (108, 30), (128, 30)


def _two_blocks(red: tuple[int, int], blue: tuple[int, int], hand: tuple[int, int] | None) -> Canvas:
    c = Canvas()
    c.container(100, 20, 152, 76, WHITE, INNER)
    c.block("red", *red)
    c.block("blue", *blue)
    if hand:
        c.hand(*hand)
    return c


def two_blocks_frames() -> list[Canvas]:
    return [
        _two_blocks(_RED_T, _BLUE_T, None),
        _two_blocks(_RED_T, _BLUE_T, (_RED_T[0] + 2, _RED_T[1] - 2)),
        _two_blocks(_RED_IN, _BLUE_T, None),
        _two_blocks(_RED_IN, _BLUE_T, (_BLUE_T[0] + 2, _BLUE_T[1] - 2)),
        _two_blocks(_RED_IN, _BLUE_IN, (_BLUE_IN[0] + 14, 8)),
        _two_blocks(_RED_IN, _BLUE_IN, None),
    ]


# -- stack_and_hidden_demo: green -> left bin, an unsupported stack, an occluded move -----------

_RED_P, _BLUE_P, _GREEN_P = (66, 28), (84, 28), (75, 46)


def _bench(blocks: dict[str, tuple[int, int]], hand: tuple[int, int] | None = None, occluder: bool = False,
           ) -> Canvas:
    c = Canvas()
    c.container(4, 14, 48, 62, BIN, BIN_INNER)
    c.container(112, 14, 156, 62, BIN, BIN_INNER)
    c.container(58, 66, 102, 94, WHITE, INNER)
    for color, pos in blocks.items():
        c.block(color, *pos)
    if occluder:
        c.rect(108, 10, 160, 66, SKIN)
    if hand:
        c.hand(*hand)
    return c


def stack_and_hidden_frames() -> list[Canvas]:
    start = {"red": _RED_P, "blue": _BLUE_P, "green": _GREEN_P}
    green_in = {**start, "green": (18, 30)}
    stacked = {**green_in, "blue": (_RED_P[0], _RED_P[1] - 10)}
    done = {**stacked, "blue": (72, 70)}
    return [
        _bench(start),
        _bench(start, (_GREEN_P[0] + 2, _GREEN_P[1] - 2)),
        _bench(green_in),
        _bench(green_in, (_BLUE_P[0] + 2, _BLUE_P[1] - 2)),
        _bench(stacked),
        _bench(stacked, occluder=True),
        _bench(stacked, (_RED_P[0] + 2, _RED_P[1] - 12)),
        _bench(done, (80, 74)),
    ]


DEMOS = {"two_blocks_demo": two_blocks_frames, "stack_and_hidden_demo": stack_and_hidden_frames}


def write_all(root: Path) -> list[Path]:
    written = []
    for name, build in DEMOS.items():
        folder = root / name / "frames"
        folder.mkdir(parents=True, exist_ok=True)
        for i, canvas in enumerate(build()):
            path = folder / f"f{i:03d}.png"
            path.write_bytes(canvas.png())
            written.append(path)
    return written


if __name__ == "__main__":
    for p in write_all(Path(sys.argv[1] if len(sys.argv) > 1 else "configs/demos")):
        print(p)
