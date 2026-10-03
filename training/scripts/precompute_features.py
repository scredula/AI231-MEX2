#!/usr/bin/env python3
"""
Precompute log-mel features for the wake-word dataset.

Converts every WAV used by training/validation into an un-normalized log-mel
spectrogram saved as float16 `.npy` in a flat features directory (keyed by
basename). This removes audio decoding + mel computation from the training loop,
which is essential on slow/network filesystems.

Usage:
    python scripts/precompute_features.py --config configs/wake_word_train.yaml
    python scripts/precompute_features.py --config configs/wake_word_train.yaml --workers 16

The dataset automatically uses these features when `data.features_dir` is set
in the config (see WakeWordDataset.use_feature_cache).
"""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_config
from src.data.wake_word_dataset import (
    LogMelSpectrogram, load_audio_file,
)


def build_transform(cfg):
    a = cfg.audio
    return LogMelSpectrogram(
        sample_rate=a.sample_rate, n_fft=a.n_fft, hop_length=a.hop_length,
        win_length=a.win_length, n_mels=a.n_mels, f_min=a.f_min, f_max=a.f_max,
        power=a.power, normalized=a.normalized,
    )


def fix_length(wav, target_samples, train=False):
    cur = wav.shape[-1]
    if cur > target_samples:
        start = (cur - target_samples) // 2
        wav = wav[:, start:start + target_samples]
    elif cur < target_samples:
        pad = target_samples - cur
        import torch
        wav = torch.nn.functional.pad(wav, (0, pad))
    return wav


def process_one(wav_path: Path, feat_dir: Path, transform, target_samples):
    out = feat_dir / (wav_path.stem + ".npy")
    if out.exists():
        return "skip"
    try:
        wav = load_audio_file(wav_path, transform.sample_rate)
        wav = fix_length(wav, target_samples)
        spec = transform(wav)  # (n_mels, T), un-normalized dB
        np.save(out, spec.numpy().astype(np.float16))
        return "ok"
    except Exception as e:
        print(f"  ! failed {wav_path.name}: {e}")
        return "fail"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wake_word_train.yaml")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    cfg = load_config(args.config)
    feat_dir = Path(cfg.data.get("features_dir", "data/features"))
    feat_dir.mkdir(parents=True, exist_ok=True)

    transform = build_transform(cfg)
    target_samples = int(cfg.audio.sample_rate * cfg.audio.clip_duration)

    # Collect audio files: positives (all splits) + all negatives in used sources
    paths = []
    for split in ["train", "val", "test"]:
        pdir = Path(cfg.data.get(f"positive_{split}"))
        if pdir.exists():
            paths += sorted(pdir.glob("*.wav"))
    neg_root = Path(cfg.data.negative_root)
    for source in cfg.data.negative_ratios.keys():
        sdir = neg_root / source
        if sdir.exists():
            paths += sorted(sdir.glob("*.wav"))

    # Deduplicate (val/test symlinks may overlap; basename key)
    uniq = {}
    for p in paths:
        uniq.setdefault(p.stem, p)
    paths = list(uniq.values())

    print(f"Precomputing {len(paths)} features -> {feat_dir} (workers={args.workers})")

    ok = skip = fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process_one, p, feat_dir, transform, target_samples)
                   for p in paths]
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            ok += r == "ok"
            skip += r == "skip"
            fail += r == "fail"
            if i % 200 == 0 or i == len(paths):
                print(f"  [{i}/{len(paths)}] ok={ok} skip={skip} fail={fail}")

    print(f"Done: ok={ok} skip={skip} fail={fail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
