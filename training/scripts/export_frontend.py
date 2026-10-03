#!/usr/bin/env python3
"""
Export the exact log-mel frontend parameters (mel filterbank + Hann window)
used during training, so the Raspberry Pi 5 demo reproduces preprocessing
bit-for-bit WITHOUT needing torch/torchaudio.

Saves to <out_dir>:
    mel_fbank.npy        (n_mels, 1 + n_fft//2)
    hann_window.npy      (win_length,)
    frontend.json        (n_fft, hop, win, n_mels, f_min, f_max, sample_rate,
                          power, top_db, amp_eps, mean, std)

Usage:
    python scripts/export_frontend.py --config configs/wake_word_train.yaml \
        --out_dir models/wake_word
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wake_word_train.yaml")
    ap.add_argument("--out_dir", default="models/wake_word")
    ap.add_argument("--norm_stats", default=None,
                    help="Path to norm_stats.json (defaults to <out_dir>/norm_stats.json)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    a = cfg.audio
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    n_freqs = 1 + a.n_fft // 2
    fbank = torchaudio.functional.melscale_fbanks(
        n_freqs=n_freqs,
        f_min=a.f_min,
        f_max=a.f_max if a.f_max else a.sample_rate / 2.0,
        n_mels=a.n_mels,
        sample_rate=a.sample_rate,
        norm=None,          # matches torchaudio MelSpectrogram(normalized=False)
        mel_scale="htk",    # torchaudio default
    )
    window = torch.hann_window(a.win_length, periodic=True)

    np.save(out / "mel_fbank.npy", fbank.numpy().astype(np.float32))
    np.save(out / "hann_window.npy", window.numpy().astype(np.float32))

    # norm stats
    norm_path = Path(args.norm_stats) if args.norm_stats else (out / "norm_stats.json")
    mean = std = None
    if norm_path.exists():
        with open(norm_path) as f:
            ns = json.load(f)
            mean, std = ns.get("mean"), ns.get("std")

    params = {
        "sample_rate": a.sample_rate,
        "n_fft": a.n_fft,
        "hop_length": a.hop_length,
        "win_length": a.win_length,
        "n_mels": a.n_mels,
        "f_min": a.f_min,
        "f_max": a.f_max if a.f_max else a.sample_rate / 2.0,
        "power": a.power,
        "top_db": 80.0,       # AmplitudeToDB(top_db=80)
        "amp_eps": 1e-10,     # torchaudio AmplitudeToDB default eps
        "clip_duration": a.clip_duration,
        "normalize": bool(a.get("normalize", True)),
        "mean": mean,
        "std": std,
    }
    with open(out / "frontend.json", "w") as f:
        json.dump(params, f, indent=2)

    print(f"Saved frontend to {out}")
    print(f"  mel_fbank: {fbank.shape}, hann: {window.shape}")
    print(f"  params: {json.dumps(params)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
