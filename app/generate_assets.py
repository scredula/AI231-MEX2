#!/usr/bin/env python3
"""
Generate pre-rendered demo assets (NOT at runtime):

  assets/jpgs/<INTENT>.jpg   for every supported intent (text on colored card)
  assets/audio/weather.mp3   "The weather today is sunny." (via espeak + ffmpeg)

Usage:
    python demo/generate_assets.py
    python demo/generate_assets.py --no-audio   # jpgs only
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent

# The 31 MEX2 classes (13 fixed intents + 18 slot variants)
INTENTS = [
    "PLAY_MUSIC", "VOLUME_UP", "VOLUME_DOWN", "NEXT", "PAUSE", "STOP",
    "LIGHT_ON", "LIGHT_OFF", "WEATHER", "TIME", "CALL", "MESSAGE", "LIST_REMINDERS",
    "BRIGHTNESS_20", "BRIGHTNESS_60", "BRIGHTNESS_100",
    "COLOR_RED", "COLOR_GREEN", "COLOR_BLUE",
    "TEMPERATURE_18", "TEMPERATURE_22", "TEMPERATURE_26",
    "TIMER_10s", "TIMER_30s", "TIMER_1m",
    "ALARM_6_00AM", "ALARM_8_00AM", "ALARM_9_00PM",
    "CREATE_REMINDER_DRINK_WATER", "CREATE_REMINDER_EXERCISE", "CREATE_REMINDER_STUDY",
]

CATEGORY_COLORS = {
    "PLAY_MUSIC": (30, 90, 200), "VOLUME_UP": (30, 90, 200), "VOLUME_DOWN": (30, 90, 200),
    "NEXT": (30, 90, 200), "PAUSE": (30, 90, 200), "STOP": (30, 90, 200),
    "LIGHT_ON": (230, 170, 30), "LIGHT_OFF": (120, 120, 130),
    "BRIGHTNESS_20": (230, 170, 30), "BRIGHTNESS_60": (230, 170, 30), "BRIGHTNESS_100": (230, 170, 30),
    "COLOR_RED": (200, 60, 60), "COLOR_GREEN": (60, 160, 80),
    "COLOR_BLUE": (60, 90, 200),
    "TEMPERATURE_18": (60, 150, 200), "TEMPERATURE_22": (60, 150, 200), "TEMPERATURE_26": (60, 150, 200),
    "TIMER_10s": (140, 80, 180), "TIMER_30s": (140, 80, 180), "TIMER_1m": (140, 80, 180),
    "ALARM_6_00AM": (140, 80, 180), "ALARM_8_00AM": (140, 80, 180), "ALARM_9_00PM": (140, 80, 180),
    "CREATE_REMINDER_DRINK_WATER": (80, 120, 90),
    "CREATE_REMINDER_EXERCISE": (80, 120, 90),
    "CREATE_REMINDER_STUDY": (80, 120, 90),
    "WEATHER": (40, 140, 150), "TIME": (90, 90, 110),
    "CALL": (150, 70, 120), "MESSAGE": (150, 70, 120),
    "LIST_REMINDERS": (80, 120, 90),
}
DEFAULT_COLOR = (90, 90, 110)


def wrap(text, width=14):
    words = text.replace("_", " ").split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= width:
            cur = (cur + " " + w).strip()
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return "\n".join(lines)


def generate_jpgs(out_dir: Path, size=(640, 480)):
    from PIL import Image, ImageDraw, ImageFont
    out_dir.mkdir(parents=True, exist_ok=True)

    def load_font(sz):
        for p in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                  "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
            if Path(p).exists():
                return ImageFont.truetype(p, sz)
        return ImageFont.load_default()

    title_font = load_font(46)
    made = 0
    for intent in INTENTS:
        color = CATEGORY_COLORS.get(intent, DEFAULT_COLOR)
        img = Image.new("RGB", size, color)
        d = ImageDraw.Draw(img)
        d.rectangle([10, 10, size[0] - 10, size[1] - 10], outline=(255, 255, 255), width=3)
        text = wrap(intent)
        bbox = d.multiline_textbbox((0, 0), text, font=title_font, align="center")
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        d.multiline_text(((size[0] - w) / 2, (size[1] - h) / 2), text,
                         font=title_font, fill=(255, 255, 255), align="center")
        img.save(out_dir / f"{intent}.jpg", quality=88)
        made += 1
    print(f"Generated {made} JPGs in {out_dir}")


def generate_beep_audio(out_dir: Path, sample_rate: int = 22050):
    """Short two-tone A5 -> E6 'listening' cue (assets/audio/beep.wav)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import soundfile as sf

    parts = []
    for f in (880.0, 1318.51):
        n = max(1, int(0.09 * sample_rate))
        t = np.arange(n) / sample_rate
        parts.append(np.sin(2 * np.pi * f * t))
    x = np.concatenate(parts)
    a = max(1, int(0.008 * sample_rate))
    r = max(1, int(0.05 * sample_rate))
    env = np.ones(x.shape[0])
    env[:a] = np.linspace(0, 1, a)
    env[-r:] = np.linspace(1, 0, r)
    y = np.clip(x * env * 0.5, -1, 1).astype("float32")
    path = out_dir / "beep.wav"
    sf.write(str(path), y, sample_rate)
    print(f"Generated {path}")
    return path


def generate_weather_audio(out_dir: Path, phrase="The weather today is sunny."):
    out_dir.mkdir(parents=True, exist_ok=True)
    mp3 = out_dir / "weather.mp3"
    wav = out_dir / "weather.wav"
    espeak = shutil.which("espeak") or shutil.which("espeak-ng")
    if not espeak:
        print("espeak not found; skipping weather.mp3 (install espeak-ng)")
        return False
    subprocess.run([espeak, "-s", "145", "-w", str(wav), phrase], check=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        subprocess.run([ffmpeg, "-loglevel", "error", "-y", "-i", str(wav),
                        "-codec:a", "libmp3lame", "-qscale:a", "5", str(mp3)], check=True)
        wav.unlink(missing_ok=True)
        print(f"Generated {mp3}")
    else:
        print(f"ffmpeg not found; kept {wav} instead of mp3")
    return True


# ---------------------------------------------------------------------------
# Demo music (short, original synthesized clips so music commands work
# out of the box; replace with your own songs any time)
# ---------------------------------------------------------------------------
_NOTE_BASE = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5,
              "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}


def _midi(name):
    if isinstance(name, int):
        return name
    return 12 * (int(name[-1]) + 1) + _NOTE_BASE[name[:-1]]


def _freq(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)


def _tone(name, dur_s, sr, wave="sine", gain=0.5):
    import numpy as np
    n = max(1, int(dur_s * sr))
    t = np.arange(n) / sr
    f = _freq(_midi(name))
    if wave == "sine":
        x = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(4 * np.pi * f * t)
    elif wave == "tri":
        x = (2 / np.pi) * np.arcsin(np.sin(2 * np.pi * f * t))
    else:  # square-ish
        x = np.sign(np.sin(2 * np.pi * f * t)) * 0.7 + 0.3 * np.sin(2 * np.pi * f * t)
    # ADSR-ish envelope (attack + release)
    a = min(0.02, 0.1 * dur_s)
    r = min(0.25, 0.5 * dur_s)
    env = np.ones(n)
    na = max(1, int(a * sr)); nr = max(1, int(r * sr))
    env[:na] = np.linspace(0, 1, na)
    env[-nr:] = np.linspace(1, 0, nr)
    return (x * env * gain).astype("float32")


def _render(melody, bass_seq, bpm, sr, wave, out_path):
    import numpy as np
    import soundfile as sf
    beat = 60.0 / bpm
    parts = [_tone(n, b * beat, sr, wave, 0.5) for n, b in melody]
    track = np.concatenate(parts) if parts else np.zeros(1, "float32")
    # bass line (soft sine, an octave or two down), mixed under the melody
    if bass_seq:
        bparts = [_tone(n, b * beat, sr, "sine", 0.28) for n, b in bass_seq]
        bass = np.concatenate(bparts)
        L = max(len(track), len(bass))
        track = np.pad(track, (0, L - len(track)))
        bass = np.pad(bass, (0, L - len(bass)))
        track = track + bass
    # gentle headroom + fade out
    peak = float(np.max(np.abs(track))) or 1.0
    track = (track / peak * 0.85).astype("float32")
    nf = min(len(track), int(0.5 * sr))
    track[-nf:] *= np.linspace(1, 0, nf)
    sf.write(str(out_path), track, sr)
    return out_path


def generate_music(music_dir: Path, overwrite=False):
    music_dir.mkdir(parents=True, exist_ok=True)
    existing = [p for p in music_dir.glob("*") if p.suffix.lower() in (".mp3", ".wav", ".ogg", ".flac")]
    if existing and not overwrite:
        print(f"Music already present ({len(existing)} files); skipping (use --overwrite to replace).")
        return
    sr = 22050
    tracks = {
        # happy C-major
        "demo_sunny.wav": (
            [("C5", 1), ("E5", 1), ("G5", 1), ("E5", 1),
             ("F5", 1), ("A5", 1), ("G5", 2),
             ("D5", 1), ("F5", 1), ("A5", 1), ("F5", 1),
             ("E5", 1), ("G5", 1), ("C5", 2)],
            [("C3", 4), ("F3", 4), ("D3", 4), ("C3", 4)], 120, "sine"),
        # mellow A-minor
        "demo_chill.wav": (
            [("A4", 1), ("C5", 1), ("E5", 2),
             ("G4", 1), ("B4", 1), ("D5", 2),
             ("F4", 1), ("A4", 1), ("C5", 2),
             ("E4", 1), ("G4", 1), ("B4", 2)],
            [("A2", 4), ("G2", 4), ("F2", 4), ("E2", 4)], 92, "tri"),
        # arpeggio synth-pop
        "demo_synthpop.wav": (
            [("C5", 0.5), ("E5", 0.5), ("G5", 0.5), ("C6", 0.5),
             ("G5", 0.5), ("E5", 0.5), ("C5", 0.5), ("G4", 0.5)] * 2
            + [("F5", 0.5), ("A5", 0.5), ("C6", 0.5), ("F5", 0.5),
               ("E5", 0.5), ("G5", 0.5), ("C5", 1.0)],
            [("C3", 4), ("F3", 4), ("C3", 4)], 128, "square"),
    }
    made = []
    for name, (mel, bass, bpm, wave) in tracks.items():
        p = _render(mel, bass, bpm, sr, wave, music_dir / name)
        made.append(p)
        print(f"Generated {p}")
    return made


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(DEMO_DIR / "assets"))
    ap.add_argument("--music_dir", default=str(DEMO_DIR / "music"))
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--no-music", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out = Path(args.out_dir)
    generate_jpgs(out / "jpgs")
    if not args.no_audio:
        generate_weather_audio(out / "audio")
        generate_beep_audio(out / "audio")
    if not args.no_music:
        generate_music(Path(args.music_dir), overwrite=args.overwrite)
    return 0


if __name__ == "__main__":
    sys.exit(main())
