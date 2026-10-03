"""
Log-mel frontend for the wake-word / command demo.

Replicates torchaudio.transforms.MelSpectrogram + AmplitudeToDB EXACTLY,
using NumPy only (no torch/torchaudio needed on the Raspberry Pi).

The mel filterbank and Hann window are loaded from files exported by
scripts/export_frontend.py, guaranteeing identical preprocessing to training.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np


class LogMelFrontend:
    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir)
        self.fbank = np.load(model_dir / "mel_fbank.npy").astype(np.float32)      # (n_mels, n_freqs)
        self.window = np.load(model_dir / "hann_window.npy").astype(np.float32)   # (win_length,)
        with open(model_dir / "frontend.json") as f:
            p = json.load(f)
        self.p = p
        self.n_fft = int(p["n_fft"])
        self.hop = int(p["hop_length"])
        self.win_length = int(p["win_length"])
        self.n_mels = int(p["n_mels"])
        self.sample_rate = int(p["sample_rate"])
        self.power = float(p["power"])
        self.top_db = float(p["top_db"])
        self.eps = float(p["amp_eps"])
        self.mean = p.get("mean")
        self.std = p.get("std")
        self.normalize = bool(p.get("normalize", True))
        self.clip_duration = float(p.get("clip_duration", 1.0))
        self.n_samples = int(self.sample_rate * self.clip_duration)

    # ---- core DSP ----
    def _stft_power(self, wav: np.ndarray) -> np.ndarray:
        """Center-padded, reflect STFT power spectrum. Returns (n_freqs, T)."""
        n_fft = self.n_fft
        pad = n_fft // 2
        wav = np.pad(wav, (pad, pad), mode="reflect")
        # frames: (T, n_fft)
        frames = np.lib.stride_tricks.sliding_window_view(wav, n_fft)[:: self.hop]
        frames = frames * self.window[: n_fft]
        spec = np.fft.rfft(frames, n=n_fft, axis=1)          # (T, n_freqs)
        power = (np.abs(spec) ** self.power).T               # (n_freqs, T)
        return power.astype(np.float32)

    def __call__(self, wav: np.ndarray) -> np.ndarray:
        """wav: float32 mono at self.sample_rate, length n_samples. Returns (n_mels, T)."""
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        if wav.shape[0] < self.n_samples:
            wav = np.pad(wav, (0, self.n_samples - wav.shape[0]))
        elif wav.shape[0] > self.n_samples:
            wav = wav[: self.n_samples]

        power = self._stft_power(wav)                        # (n_freqs, T)
        # torchaudio stores the filterbank as (n_freqs, n_mels); transpose as needed
        if self.fbank.shape[0] == self.n_mels and self.fbank.shape[1] == power.shape[0]:
            mel = self.fbank @ power                         # (n_mels, T)
        else:
            mel = self.fbank.T @ power                       # (n_mels, T)

        # AmplitudeToDB(power, top_db)
        db = 10.0 * np.log10(np.maximum(mel, self.eps))
        db = np.maximum(db, db.max() - self.top_db)

        if self.normalize and self.mean is not None and self.std is not None:
            db = (db - self.mean) / (self.std + 1e-8)

        return db.astype(np.float32)


class MetadataFrontend:
    """Log-mel frontend for the command classifier.

    Unlike :class:`LogMelFrontend` (the torchaudio-compatible wake-word
    frontend), this reads *everything* from a single metadata JSON that ships
    the Hanning window and mel filterbank inline, so it stays bit-compatible
    with training for the command model:

        16 kHz mono -> hann window -> rfft per frame
        -> power = (re^2 + im^2) / n_fft
        -> mel = filterbank @ power.T
        -> log(mel + log_eps) -> pad/truncate to max_frames  (n_mels, max_frames)
    """

    def __init__(self, meta_path: str | Path):
        with open(meta_path) as f:
            meta = json.load(f)
        self.sr = int(meta["sr"])
        self.n_fft = int(meta["n_fft"])
        self.hop = int(meta["hop"])
        self.n_mels = int(meta["n_mels"])
        self.max_frames = int(meta["max_frames"])
        self.log_eps = float(meta.get("log_eps", 1e-4))
        self.window = np.asarray(meta["hanning_window"], dtype=np.float32)
        self.fb = np.asarray(meta["mel_filterbank"], dtype=np.float32)

    @property
    def window_samples(self) -> int:
        """Number of input samples that yield exactly ``max_frames`` frames."""
        return self.hop * (self.max_frames - 1) + self.n_fft

    def __call__(self, wav: np.ndarray) -> np.ndarray:
        """wav: 1-D float32 mono at ``sr``. Returns (1, 1, n_mels, max_frames)."""
        y = np.asarray(wav, dtype=np.float32).reshape(-1)
        if y.shape[0] < self.window_samples:
            y = np.pad(y, (0, self.window_samples - y.shape[0]))
        else:
            y = y[-self.window_samples:]
        idx = np.arange(self.n_fft)[None, :] + self.hop * np.arange(self.max_frames)[:, None]
        spec = np.fft.rfft(y[idx] * self.window, axis=1)
        power = (spec.real ** 2 + spec.imag ** 2) / self.n_fft
        logmel = np.log(self.fb @ power.T + self.log_eps).astype(np.float32)
        return logmel[None, None]


def resample_to_16k(x: np.ndarray, orig_sr: int, target_sr: int = 16000) -> np.ndarray:
    """Resample mono audio using polyphase filtering (scipy)."""
    if orig_sr == target_sr:
        return np.asarray(x, dtype=np.float32)
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(int(orig_sr), int(target_sr))
    up = target_sr // g
    down = orig_sr // g
    return resample_poly(np.asarray(x, dtype=np.float32), up, down).astype(np.float32)


def read_wav_mono_16k(path: str | Path, target_sr: int = 16000) -> np.ndarray:
    """Read a wav file as mono float32 at target_sr (soundfile + scipy)."""
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    return resample_to_16k(mono, sr, target_sr)
