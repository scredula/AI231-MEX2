#!/usr/bin/env python3
"""
Download & prepare negative samples for wake-word training.

Sources:
  speech  : LibriSpeech dev-clean (OpenSLR-12)       -> data/negative/speech
  noise   : MUSAN noise+music subsets (OpenSLR-17)   -> data/negative/noise
  esc50   : ESC-50 environmental sounds              -> data/negative/noise
  silence : synthetically generated room tone        -> data/negative/silence
  mex2    : MEX2 command dataset (local path)        -> data/negative/mex2

All clips are resampled to 16 kHz mono and chunked into fixed-length WAV files.
A metadata.csv is written with columns: path, source, duration, split.

Requires: ffmpeg (or sox) on PATH. No Python audio deps needed.

Usage:
    python scripts/download_negatives.py --all
    python scripts/download_negatives.py --sources speech,silence
    python scripts/download_negatives.py --sources mex2 --mex2_path /path/to/MEX2/Data
    python scripts/download_negatives.py --all --max_clips_speech 2000 --max_clips_noise 1000
"""
import argparse
import csv
import os
import random
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
import wave
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LIBRISPEECH_URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
MUSAN_URL = "https://www.openslr.org/resources/17/musan.tar.gz"
ESC50_URL = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"

SAMPLE_RATE = 16000
CLIP_DURATION = 1.0
FFMPEG = shutil.which("ffmpeg")
SOX = shutil.which("sox")


def log(msg):
    print(f"[negatives] {msg}", flush=True)


def ensure_tool():
    if FFMPEG is None and SOX is None:
        raise RuntimeError("Neither ffmpeg nor sox found on PATH. Please install one.")


def download_file(url: str, dest: Path, chunk_size: int = 1 << 20):
    """Download a URL to dest with a simple progress log."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        log(f"Already downloaded: {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest

    log(f"Downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    start = time.time()
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    dt = time.time() - start
    log(f"Saved {dest.name} ({dest.stat().st_size / 1e6:.1f} MB in {dt:.1f}s)")
    return dest


def to_wav_clip(src: Path, dst: Path, start: float, duration: float,
                sample_rate: int = SAMPLE_RATE) -> bool:
    """Extract a mono 16kHz WAV clip using ffmpeg (fallback to sox)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return True
    if FFMPEG:
        cmd = [
            FFMPEG, "-loglevel", "error", "-y",
            "-ss", f"{start:.4f}", "-t", f"{duration:.4f}",
            "-i", str(src), "-ac", "1", "-ar", str(sample_rate),
            "-sample_fmt", "s16", str(dst),
        ]
    else:
        cmd = [
            SOX, str(src), "-c", "1", "-r", str(sample_rate),
            str(dst), "trim", f"{start:.4f}", f"{duration:.4f}",
        ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return dst.exists() and dst.stat().st_size > 0
    except subprocess.CalledProcessError:
        return False


def audio_duration(src: Path) -> float:
    """Return duration in seconds using ffprobe, or 0 if unknown."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 0.0
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(src)],
            capture_output=True, text=True, check=True,
        )
        return float(out.stdout.strip())
    except Exception:
        return 0.0


def find_audio_files(root: Path, exts=(".wav", ".flac", ".mp3", ".ogg")):
    files = []
    for ext in exts:
        files.extend(root.rglob(f"*{ext}"))
    return sorted(files)


def chunk_file(src: Path, out_dir: Path, prefix: str, max_clips: int,
               clip_duration: float, rng: random.Random) -> list:
    """Chunk one audio file into fixed-length clips. Returns list of paths."""
    dur = audio_duration(src)
    produced = []
    if dur <= 0:
        # fallback: single clip of full file
        dst = out_dir / f"{prefix}_{len(produced):05d}.wav"
        if to_wav_clip(src, dst, 0.0, clip_duration):
            produced.append(dst)
        return produced

    n = min(max_clips, max(1, int(dur // clip_duration)))
    starts = [i * clip_duration for i in range(n)]
    rng.shuffle(starts)
    for i, st in enumerate(starts):
        dst = out_dir / f"{prefix}_{i:05d}.wav"
        if to_wav_clip(src, dst, st, clip_duration):
            produced.append(dst)
    return produced


# ---------------------------------------------------------------------------
# Source: LibriSpeech (speech)
# ---------------------------------------------------------------------------
def prepare_speech(out_root: Path, tmp_dir: Path, max_clips: int,
                   clip_duration: float, rng: random.Random):
    out_dir = out_root / "speech"
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(out_dir.glob("*.wav"))
    if len(existing) >= max_clips:
        log(f"speech: {len(existing)} clips already present, skipping")
        return existing[:max_clips]

    archive = download_file(LIBRISPEECH_URL, tmp_dir / "dev-clean.tar.gz")
    extract_dir = tmp_dir / "librispeech"
    if not extract_dir.exists():
        log("Extracting LibriSpeech dev-clean...")
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(tmp_dir)
        extracted = tmp_dir / "LibriSpeech" / "dev-clean"
        if extracted.exists():
            extracted.rename(extract_dir)

    files = find_audio_files(extract_dir, exts=(".flac", ".wav"))
    rng.shuffle(files)
    log(f"speech: found {len(files)} source files, target {max_clips} clips")

    produced = list(existing)
    idx = len(produced)
    for src in files:
        if idx >= max_clips:
            break
        remaining = max_clips - idx
        clips = chunk_file(src, out_dir, f"speech_{idx:05d}", remaining,
                           clip_duration, rng)
        produced.extend(clips)
        idx += len(clips)
    log(f"speech: produced {len(produced)} clips")
    return produced[:max_clips]


# ---------------------------------------------------------------------------
# Source: MUSAN (noise + music)
# ---------------------------------------------------------------------------
def prepare_noise(out_root: Path, tmp_dir: Path, max_clips: int,
                  clip_duration: float, rng: random.Random):
    out_dir = out_root / "noise"
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(out_dir.glob("*.wav"))
    if len(existing) >= max_clips:
        log(f"noise: {len(existing)} clips already present, skipping")
        return existing[:max_clips]

    extract_dir = tmp_dir / "musan"
    if not extract_dir.exists():
        archive = download_file(MUSAN_URL, tmp_dir / "musan.tar.gz")
        log("Stream-extracting MUSAN noise/ and music/ subsets (this can take a while)...")
        extract_dir.mkdir(parents=True, exist_ok=True)
        wanted = ("musan/noise/", "musan/music/")
        with tarfile.open(archive, "r:gz") as tar:
            members = [m for m in tar.getmembers()
                       if m.name.startswith(wanted) and m.isfile()]
            for m in members:
                tar.extract(m, extract_dir, filter="data")

    files = find_audio_files(extract_dir, exts=(".wav", ".flac"))
    rng.shuffle(files)
    log(f"noise: found {len(files)} source files, target {max_clips} clips")

    produced = list(existing)
    idx = len(produced)
    for src in files:
        if idx >= max_clips:
            break
        remaining = max_clips - idx
        clips = chunk_file(src, out_dir, f"noise_{idx:05d}", remaining,
                           clip_duration, rng)
        produced.extend(clips)
        idx += len(clips)
    log(f"noise: produced {len(produced)} clips")
    return produced[:max_clips]


# ---------------------------------------------------------------------------
# Source: ESC-50 (environmental noise; smaller alternative/addition to MUSAN)
# ---------------------------------------------------------------------------
def prepare_esc50(out_root: Path, tmp_dir: Path, max_clips: int,
                  clip_duration: float, rng: random.Random):
    out_dir = out_root / "noise"
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = download_file(ESC50_URL, tmp_dir / "esc50.zip")
    extract_dir = tmp_dir / "esc50"
    if not extract_dir.exists():
        log("Extracting ESC-50...")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(extract_dir)

    files = find_audio_files(extract_dir, exts=(".wav",))
    rng.shuffle(files)
    log(f"esc50: found {len(files)} source files, target {max_clips} clips")

    produced = []
    for i, src in enumerate(files):
        if len(produced) >= max_clips:
            break
        dst = out_dir / f"esc50_{i:05d}.wav"
        if to_wav_clip(src, dst, 0.0, clip_duration):
            produced.append(dst)
    log(f"esc50: produced {len(produced)} clips")
    return produced


# ---------------------------------------------------------------------------
# Source: synthetic silence / room tone
# ---------------------------------------------------------------------------
def prepare_silence(out_root: Path, tmp_dir: Path, max_clips: int,
                    clip_duration: float, rng: random.Random,
                    sample_rate: int = SAMPLE_RATE):
    import math
    import struct

    out_dir = out_root / "silence"
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(out_dir.glob("*.wav"))
    if len(existing) >= max_clips:
        log(f"silence: {len(existing)} clips already present, skipping")
        return existing[:max_clips]

    n_samples = int(clip_duration * sample_rate)
    produced = []
    for i in range(max_clips):
        dst = out_dir / f"silence_{i:05d}.wav"
        if dst.exists():
            produced.append(dst)
            continue
        # Low-level pink-ish room tone + faint mains hum
        amp = rng.uniform(0.0008, 0.004)
        hum_freq = rng.choice([50.0, 60.0])
        hum_amp = amp * rng.uniform(0.0, 0.5)
        prev = 0.0
        frames = bytearray()
        for t in range(n_samples):
            white = rng.uniform(-1.0, 1.0)
            pink = 0.98 * prev + 0.02 * white
            prev = pink
            hum = math.sin(2 * math.pi * hum_freq * t / sample_rate)
            sample = max(-1.0, min(1.0, amp * pink + hum_amp * hum))
            frames += struct.pack("<h", int(sample * 32767))
        with wave.open(str(dst), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(bytes(frames))
        produced.append(dst)
    log(f"silence: produced {len(produced)} clips")
    return produced


# ---------------------------------------------------------------------------
# Source: MEX2 commands (local dataset)
# ---------------------------------------------------------------------------
def prepare_mex2(out_root: Path, tmp_dir: Path, max_clips: int,
                 clip_duration: float, rng: random.Random, mex2_path: Path):
    out_dir = out_root / "mex2"
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = list(out_dir.glob("*.wav"))
    if len(existing) >= max_clips:
        log(f"mex2: {len(existing)} clips already present, skipping")
        return existing[:max_clips]

    if mex2_path is None or not Path(mex2_path).exists():
        log(f"mex2: source path not provided or missing ({mex2_path}); skipping")
        return existing

    files = find_audio_files(Path(mex2_path), exts=(".wav",))
    rng.shuffle(files)
    log(f"mex2: found {len(files)} source files, target {max_clips} clips")

    produced = list(existing)
    idx = len(produced)
    for src in files:
        if idx >= max_clips:
            break
        remaining = max_clips - idx
        clips = chunk_file(src, out_dir, f"mex2_{idx:05d}", remaining,
                           clip_duration, rng)
        produced.extend(clips)
        idx += len(clips)
    log(f"mex2: produced {len(produced)} clips")
    return produced[:max_clips]


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
def write_metadata(out_root: Path, clips_by_source: dict, rng: random.Random):
    """Write metadata.csv by scanning the output directories on disk.

    This makes re-runs idempotent and merges sources across invocations.
    Columns: path, source, duration, split.
    """
    # Discover all clips present on disk (source == subdirectory name)
    rows = []
    for source_dir in sorted(p for p in out_root.iterdir() if p.is_dir()):
        if source_dir.name.startswith("_"):
            continue
        source = source_dir.name
        paths = sorted(source_dir.glob("*.wav"))
        if not paths:
            continue
        n = len(paths)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        for i, p in enumerate(paths):
            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "val"
            else:
                split = "test"
            rel = p.relative_to(out_root)
            rows.append({
                "path": str(rel),
                "source": source,
                "duration": CLIP_DURATION,
                "split": split,
            })

    meta_path = out_root / "metadata.csv"
    with open(meta_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "source", "duration", "split"])
        writer.writeheader()
        writer.writerows(rows)
    log(f"Wrote {len(rows)} rows to {meta_path}")

    from collections import Counter
    by_source = Counter(r["source"] for r in rows)
    by_split = Counter(r["split"] for r in rows)
    log(f"By source: {dict(by_source)}")
    log(f"By split:  {dict(by_split)}")


def main():
    parser = argparse.ArgumentParser(description="Download negative samples for wake-word training")
    parser.add_argument("--out_dir", type=str, default="data/negative")
    parser.add_argument("--tmp_dir", type=str, default="data/negative/_downloads")
    parser.add_argument("--sources", type=str, default=None,
                        help="Comma list of: speech,noise,esc50,silence,mex2")
    parser.add_argument("--all", action="store_true", help="All sources except esc50")
    parser.add_argument("--mex2_path", type=str, default=None,
                        help="Path to local MEX2 dataset (when released)")
    parser.add_argument("--max_clips_speech", type=int, default=2000)
    parser.add_argument("--max_clips_noise", type=int, default=1000)
    parser.add_argument("--max_clips_esc50", type=int, default=1000)
    parser.add_argument("--max_clips_silence", type=int, default=500)
    parser.add_argument("--max_clips_mex2", type=int, default=2000)
    parser.add_argument("--clip_duration", type=float, default=CLIP_DURATION)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    rng = random.Random(args.seed)
    ensure_tool()

    out_root = Path(args.out_dir)
    tmp_dir = Path(args.tmp_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.all:
        sources = ["speech", "noise", "silence", "mex2"]
    elif args.sources:
        sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    else:
        parser.error("Specify --all or --sources")

    clips_by_source = {}

    if "speech" in sources:
        clips_by_source["speech"] = prepare_speech(
            out_root, tmp_dir, args.max_clips_speech, args.clip_duration, rng)
    if "noise" in sources:
        clips_by_source["noise"] = prepare_noise(
            out_root, tmp_dir, args.max_clips_noise, args.clip_duration, rng)
    if "esc50" in sources:
        clips_by_source.setdefault("noise", []).extend(prepare_esc50(
            out_root, tmp_dir, args.max_clips_esc50, args.clip_duration, rng))
    if "silence" in sources:
        clips_by_source["silence"] = prepare_silence(
            out_root, tmp_dir, args.max_clips_silence, args.clip_duration, rng)
    if "mex2" in sources:
        clips_by_source["mex2"] = prepare_mex2(
            out_root, tmp_dir, args.max_clips_mex2, args.clip_duration, rng,
            Path(args.mex2_path) if args.mex2_path else None)

    # Drop empty sources
    clips_by_source = {k: v for k, v in clips_by_source.items() if v}
    if not clips_by_source:
        log("No clips produced. Nothing to write.")
        return 1

    write_metadata(out_root, clips_by_source, rng)
    log("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())