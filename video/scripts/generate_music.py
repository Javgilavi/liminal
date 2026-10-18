"""Synthesize the original 120 BPM score, so the music is ours and carries no licence.

Arrangement follows config.json: impacts land on the cuts, a breakdown under the imagination shot,
and the drums stop for the end card. Run from video/: python3 scripts/generate_music.py
Writes public/music.wav (48 kHz stereo). Replace it with any 120 BPM track to change the music.
"""

import json
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
CFG = json.loads((HERE / "config.json").read_text())
SR = 48_000
BPM = CFG["bpm"]
BEAT = 60 / BPM
DUR = CFG["durationSec"]
SHOTS = CFG["shots"]
RNG = np.random.default_rng(7)
N = int(DUR * SR) + SR * 3

mix = {k: np.zeros((N, 2)) for k in ("drums", "bass", "pad", "arp", "fx")}


def midi(n: float) -> float:
    return 440.0 * 2 ** ((n - 69) / 12)


def lowpass(x: np.ndarray, cutoff: float, res_db: float = 0.0) -> np.ndarray:
    """Zero-phase FFT low-pass with a soft knee and an optional resonant bump at the cutoff."""
    n = len(x)
    f = np.fft.rfftfreq(n, 1 / SR)
    gain = 1 / np.sqrt(1 + (f / cutoff) ** 8)
    if res_db:
        gain *= 1 + (10 ** (res_db / 20) - 1) * np.exp(-((f - cutoff) / (cutoff * 0.15)) ** 2)
    return np.fft.irfft(np.fft.rfft(x, axis=0) * (gain[:, None] if x.ndim == 2 else gain), n, axis=0)


def highpass(x: np.ndarray, cutoff: float) -> np.ndarray:
    n = len(x)
    f = np.fft.rfftfreq(n, 1 / SR)
    gain = 1 / np.sqrt(1 + (cutoff / np.maximum(f, 1)) ** 8)
    return np.fft.irfft(np.fft.rfft(x, axis=0) * (gain[:, None] if x.ndim == 2 else gain), n, axis=0)


def saw(freq: float, n: int, phase: float = 0.0) -> np.ndarray:
    t = np.arange(n) / SR
    p = (freq * t + phase) % 1.0
    return 2 * p - 1


def place(bus: str, t: float, sig: np.ndarray, pan: float = 0.0) -> None:
    i = int(t * SR)
    if sig.ndim == 1:
        sig = np.stack([sig * np.sqrt(0.5 * (1 - pan)), sig * np.sqrt(0.5 * (1 + pan))], 1)
    j = min(N, i + len(sig))
    mix[bus][i:j] += sig[: j - i]


def env(n: int, attack: float, decay: float) -> np.ndarray:
    t = np.arange(n) / SR
    a = np.clip(t / max(attack, 1e-4), 0, 1)
    return a * np.exp(-t / decay)


# --- instruments -------------------------------------------------------------------------------

def kick(gain: float = 1.0) -> np.ndarray:
    n = int(0.5 * SR)
    t = np.arange(n) / SR
    f = 46 + 110 * np.exp(-t / 0.035)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.28)
    click = highpass(RNG.standard_normal(n) * np.exp(-t / 0.004), 2000) * 0.25
    return np.tanh((body + click) * 1.6) * 0.9 * gain


def hat(open_: bool = False, gain: float = 1.0) -> np.ndarray:
    n = int((0.25 if open_ else 0.06) * SR)
    noise = highpass(RNG.standard_normal(n), 7000)
    return noise * env(n, 0.001, 0.07 if open_ else 0.018) * 0.22 * gain


def clap() -> np.ndarray:
    n = int(0.35 * SR)
    t = np.arange(n) / SR
    noise = lowpass(highpass(RNG.standard_normal(n), 900), 5000)
    bursts = sum(np.exp(-np.clip(t - d, 0, None) / 0.012) * (t >= d) for d in (0, 0.011, 0.022))
    tail = np.exp(-t / 0.12)
    return noise * (0.6 * bursts + 0.5 * tail) * 0.35


def bass_note(note: int, length: float) -> np.ndarray:
    n = int(length * SR)
    f = midi(note)
    s = 0.7 * saw(f, n) + 0.5 * np.sin(2 * np.pi * f * np.arange(n) / SR)
    s = lowpass(s, 420, 4)
    return s * env(n, 0.004, length * 0.7) * 0.42


def pad_chord(notes: list[int], length: float) -> np.ndarray:
    n = int(length * SR)
    out = np.zeros((n, 2))
    for note in notes:
        for d, side in ((-0.09, 0), (0.09, 1)):
            out[:, side] += saw(midi(note + d), n, phase=RNG.random())
            out[:, side] += 0.5 * saw(midi(note + 12 - d), n, phase=RNG.random())
    out = lowpass(out, 1400)
    t = np.arange(n) / SR
    shape = np.clip(t / 0.35, 0, 1) * np.clip((length - t) / 0.4, 0, 1)
    return out * shape[:, None] * 0.045


def pluck(note: int, cutoff: float) -> np.ndarray:
    n = int(0.28 * SR)
    f = midi(note)
    s = saw(f, n) + 0.6 * saw(f * 1.005, n)
    s = lowpass(s, cutoff, 3)
    return s * env(n, 0.002, 0.08) * 0.16


def riser(length: float) -> np.ndarray:
    n = int(length * SR)
    t = np.arange(n) / SR
    noise = RNG.standard_normal(n)
    out = np.zeros(n)
    steps = 12
    for k in range(steps):
        a, b = k * n // steps, (k + 1) * n // steps
        out[a:b] = highpass(lowpass(noise[a:b], 600 + 9000 * (k / steps) ** 2), 300)[: b - a]
    return out * (t / length) ** 2.2 * 0.22


def impact() -> np.ndarray:
    n = int(2.4 * SR)
    t = np.arange(n) / SR
    boom = np.sin(2 * np.pi * np.cumsum(38 + 70 * np.exp(-t / 0.08)) / SR) * np.exp(-t / 0.7)
    air = lowpass(RNG.standard_normal(n), 3000) * np.exp(-t / 0.5) * 0.25
    return np.tanh((boom + air) * 1.4) * 0.55


# --- arrangement -------------------------------------------------------------------------------

beats = np.arange(0, DUR, BEAT)
chords = [[57, 60, 64, 67], [53, 57, 60, 64], [48, 52, 55, 59], [55, 59, 62, 66]]  # Am7 Fmaj7 Cmaj7 G(add#11)
roots = [45, 41, 48, 43]
book, reveal, sim, hero, imag, diagram, end = (
    SHOTS[k] for k in ("book", "reveal", "sim", "hero", "imagination", "diagram", "end"))
BOOK = CFG["book"]
flips = [BOOK["coverOpenAt"] + i * BOOK["flipEvery"] for i in range(len(BOOK["pages"]))]


def section(t: float) -> str:
    if t < reveal["start"]:
        return "book"
    if t < sim["start"]:
        return "reveal"
    if imag["start"] <= t < imag["end"]:
        return "break"
    if t >= end["start"]:
        return "outro"
    return "groove"


def music_box(note: int) -> np.ndarray:
    n = int(0.9 * SR)
    t = np.arange(n) / SR
    f = midi(note)
    s = np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * 2 * f * t) + 0.12 * np.sin(2 * np.pi * 4.01 * f * t)
    return s * env(n, 0.002, 0.35) * 0.09


def page_flip() -> np.ndarray:
    n = int(0.45 * SR)
    t = np.arange(n) / SR
    noise = lowpass(highpass(RNG.standard_normal(n), 1200), 7000)
    shape = np.sin(np.pi * np.clip(t / 0.4, 0, 1)) ** 2 * (1 + 0.6 * np.sin(2 * np.pi * 9 * t))
    return noise * shape * 0.16


for t in beats:
    sec = section(t)
    if sec == "groove":
        place("drums", t, kick())
        place("drums", t + BEAT / 2, hat(open_=(int(t / BEAT) % 4 == 3)), pan=0.25)
        place("drums", t + BEAT / 4, hat(gain=0.45), pan=-0.3)
        place("drums", t + 3 * BEAT / 4, hat(gain=0.45), pan=-0.3)
        if int(round(t / BEAT)) % 2 == 1:
            place("drums", t, clap(), pan=0.05)
    elif sec == "book" and t >= flips[4]:
        # The pulse arrives as the history reaches learned, two-handed arms.
        place("drums", t, kick(0.55))
        place("drums", t + BEAT / 2, hat(gain=0.6), pan=0.2)

# Bass: off-beat eighths on the chord root in the grooves, a held sub under the reveal.
for t in beats:
    root = roots[int(t // (4 * BEAT)) % 4]
    if section(t) == "groove":
        place("bass", t + BEAT / 2, bass_note(root, BEAT / 2 * 0.95))
        if int(t / BEAT) % 4 == 3:
            place("bass", t + 3 * BEAT / 4, bass_note(root + 12, BEAT / 4 * 0.9))
place("bass", reveal["start"], bass_note(33, reveal["end"] - reveal["start"]) * 0.8)

# Pad: one chord per bar for the whole piece, held into the end card.
bar_len = 4 * BEAT
for b in range(int(np.ceil(DUR / bar_len))):
    t = b * bar_len
    length = bar_len + 0.3 if t + bar_len < end["start"] else DUR - t + 2.5
    place("pad", t, pad_chord(chords[b % 4], length) * (0.7 if t < reveal["start"] else 1.0))
    if t + bar_len >= end["start"]:
        break

# Music box under the book: one note per beat, walking the chord tones.
for k, t in enumerate(beats):
    if section(t) != "book" or t < 0.5:
        continue
    chord = chords[int(t // bar_len) % 4]
    place("arp", t, music_box(chord[[0, 2, 1, 3][k % 4]] + 24), pan=0.3 * np.sin(k))
for t in flips:
    place("fx", t, page_flip(), pan=0.15)

# Arp: sixteenths through the chord from the twin to the end card, the filter opening as it goes.
step = BEAT / 4
pattern = [0, 2, 1, 3, 2, 1, 3, 2]
t = sim["start"]
k = 0
while t < end["start"]:
    sec = section(t)
    chord = chords[int(t // bar_len) % 4]
    note = chord[pattern[k % len(pattern)]] + 12
    progress = (t - sim["start"]) / (end["start"] - sim["start"])
    cutoff = 900 + 3200 * progress if sec != "break" else 2600
    place("arp", t, pluck(note, cutoff) * (1.0 if k % 4 == 0 else 0.7), pan=0.35 * np.sin(k * 0.7))
    t += step
    k += 1

# FX: risers into the big cuts, impacts on them.
for cut in (reveal["start"], sim["start"], hero["start"], diagram["start"], end["start"]):
    place("fx", cut, impact())
for cut, length in ((reveal["start"], 1.5), (sim["start"], 1.0), (hero["start"], 1.5), (end["start"], 2.0)):
    place("fx", cut - length, riser(length), pan=0.0)

# --- mix ----------------------------------------------------------------------------------------

t_axis = np.arange(N) / SR
duck = np.ones(N)
for t in beats:
    if section(t) == "groove":
        i = int(t * SR)
        d = 1 - 0.55 * np.exp(-np.arange(int(0.3 * SR)) / SR / 0.09)
        j = min(N, i + len(d))
        duck[i:j] = np.minimum(duck[i:j], d[: j - i])


def reverb(x: np.ndarray, seconds: float, wet: float) -> np.ndarray:
    n = int(seconds * SR)
    t = np.arange(n) / SR
    ir = RNG.standard_normal((n, 2)) * np.exp(-t / (seconds / 5))[:, None]
    ir = lowpass(ir, 6000)
    ir /= np.sqrt((ir ** 2).sum(0))
    size = 1 << int(np.ceil(np.log2(len(x) + n)))
    y = np.fft.irfft(np.fft.rfft(x, size, axis=0) * np.fft.rfft(ir, size, axis=0), size, axis=0)[: len(x)]
    return x + wet * y


pad = reverb(mix["pad"], 3.0, 0.9) * duck[:, None]
arp = reverb(mix["arp"], 2.0, 0.55) * (0.5 + 0.5 * duck[:, None])
bass = mix["bass"] * duck[:, None]
fx = reverb(mix["fx"], 2.5, 0.6)
levels = {"drums": 0.6, "bass": 5.0, "pad": 0.8, "arp": 2.4, "fx": 0.75}
buses = {"drums": mix["drums"], "bass": bass, "pad": pad, "arp": arp, "fx": fx}
master = sum(buses[k] * g for k, g in levels.items())
for name, g in levels.items():
    print(f"{name:6s} rms {20 * np.log10(np.sqrt(((buses[name] * g) ** 2).mean()) + 1e-9):6.1f} dB")

# Fade with the end card and trim to the video length.
fade = np.clip((DUR - t_axis) / 1.2, 0, 1)
master = master * fade[:, None]
master = master[: int(DUR * SR)]
master = np.tanh(master * 1.3) / np.tanh(1.3)
# Peak to -1 dBFS, then down 4 dB so the mix sits near -14 LUFS, the usual streaming target.
master *= 10 ** (-5 / 20) / np.abs(master).max()

out = HERE / "public/music.wav"
out.parent.mkdir(exist_ok=True)
with wave.open(str(out), "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes((master * 32767).astype("<i2").tobytes())
print("wrote", out)
