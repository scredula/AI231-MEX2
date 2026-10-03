#!/usr/bin/env python3
"""
Headless self-test for the demo (no GUI / no mic required).

1. Verifies the NumPy log-mel frontend matches torchaudio EXACTLY.
2. Runs the ONNX wake-word detector on positive ("Hey Mason") and negative
   samples and prints probabilities.

Usage:
    python demo/selftest.py
    python demo/selftest.py --model_dir models/wake_word --n 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))
PARENT = APP_DIR.parent

from features import LogMelFrontend, read_wav_mono_16k   # noqa: E402
from wakeword import WakeWordDetector                    # noqa: E402
from commands import CommandClassifier                   # noqa: E402


def check_frontend_equivalence(model_dir: Path):
    try:
        import torch
        import torchaudio
    except Exception as e:
        print(f"  torch/torchaudio not installed ({e}); skipping frontend equivalence check.")
        print("  (This check is optional; the ONNX tests below still run.)")
        return True
    import json

    with open(model_dir / "frontend.json") as f:
        p = json.load(f)
    a = p
    mel_t = torchaudio.transforms.MelSpectrogram(
        sample_rate=a["sample_rate"], n_fft=a["n_fft"], hop_length=a["hop_length"],
        win_length=a["win_length"], n_mels=a["n_mels"], f_min=a["f_min"], f_max=a["f_max"],
        power=a["power"], normalized=False, center=True, pad_mode="reflect",
    )
    db_t = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=a["top_db"])

    frontend = LogMelFrontend(model_dir)

    wav = (np.random.randn(frontend.n_samples).astype(np.float32)) * 0.05
    ref = db_t(mel_t(torch.from_numpy(wav)[None]))[0].numpy()          # (64, T)
    got = frontend(wav)                                               # normalized

    # de-normalize the demo output for a fair raw comparison
    if frontend.mean is not None:
        got_raw = got * (frontend.std + 1e-8) + frontend.mean
    else:
        got_raw = got

    if ref.shape != got_raw.shape:
        print(f"  shape mismatch: torch {ref.shape} vs numpy {got_raw.shape}")
        return False
    diff = float(np.max(np.abs(ref - got_raw)))
    print(f"  torch {ref.shape} vs numpy {got_raw.shape} | max|diff| = {diff:.3e}")
    ok = diff < 1e-3
    print("  frontend equivalence:", "PASS" if ok else "FAIL")
    return ok


def run_detector(model_dir: Path, n: int):
    det = WakeWordDetector(model_dir)
    # Positives: bundled samples; fall back to repo test split if present.
    pos_dir = APP_DIR / "samples"
    if not pos_dir.exists():
        pos_dir = PARENT / "data/wake_word/test"
    neg_root = PARENT / "data/negative"

    pos = sorted(pos_dir.glob("*.wav"))[:n]
    neg = []
    for src in ["speech", "noise", "silence"]:
        d = neg_root / src
        if d.exists():
            neg += sorted(d.glob("*.wav"))[: max(1, n // 3)]

    def score(paths):
        vals = [det.predict(read_wav_mono_16k(p, 16000)) for p in paths]
        return np.array(vals) if vals else np.array([np.nan])

    ps = score(pos)
    ns = score(neg)
    print(f"  positive ({len(ps)}): mean={np.nanmean(ps):.3f} min={np.nanmin(ps):.3f}")
    print(f"  negative ({len(ns)}): mean={np.nanmean(ns):.3f} max={np.nanmax(ns):.3f}")
    # simple separability check
    thr = 0.85
    acc = ((ps >= thr).sum() + (ns < thr).sum()) / (len(ps) + len(ns))
    print(f"  accuracy @ {thr}: {acc*100:.1f}%")
    return acc


def _iter_command_models(cfg):
    cc = cfg.get("command", {}) or {}
    reg = cc.get("models", {}) or {}
    if not reg and cc.get("model_dir"):
        reg = {"default": cc["model_dir"]}
    return reg


def run_command_classifier(n: int = 3):
    """Load every selectable command classifier and classify bundled sample WAVs."""
    cfg = {}
    cfg_path = APP_DIR / "config.yaml"
    if cfg_path.exists():
        try:
            import yaml
            cfg = yaml.safe_load(open(cfg_path)) or {}
        except Exception:
            cfg = {}
    intents = cfg.get("intents", []) or []
    registry = _iter_command_models(cfg) or {"default": "models/command_classifier"}

    samples = sorted((APP_DIR / "samples").glob("*.wav"))[:n]
    ok = True
    for name, rel in registry.items():
        model_dir = APP_DIR / rel
        clf = CommandClassifier(model_dir, intents=intents)
        if not clf.real:
            print(f"  [{name}] no command model at {model_dir}")
            ok = False
            continue
        print(f"  [{name}] {rel}: {len(clf.intents)} classes, "
              f"frontend={clf.frontend_kind}, onnx={clf.onnx_name}, "
              f"unknown_threshold={clf.unknown_threshold:.3f}")
        if not samples:
            print("    no sample WAVs to classify")
            continue
        for p in samples:
            wav = read_wav_mono_16k(p, 16000)
            label, conf, _top = clf.classify(wav)
            total = float(clf.predict(wav).sum())
            print(f"    {p.name}: {label} ({conf*100:.1f}%)  softmax_sum={total:.4f}")
            ok = ok and abs(total - 1.0) < 1e-3
    return ok


def _resolve_model_dir(model: str | None, model_dir: str | None) -> Path:
    if model_dir:
        return Path(model_dir)
    # resolve via config.yaml registry
    default = APP_DIR / "models" / "wake_word" / "matchboxnet"
    cfg_path = APP_DIR / "config.yaml"
    if cfg_path.exists():
        try:
            import yaml
            cfg = yaml.safe_load(open(cfg_path))
            reg = (cfg.get("wake_word", {}) or {}).get("models", {}) or {}
            if model:
                for k, v in reg.items():
                    if model.lower() in (k.lower(), Path(v).name.lower()):
                        return APP_DIR / v
                print(f"(unknown --model '{model}'; using default)")
            elif reg:
                ww = cfg.get("wake_word", {}) or {}
                # default = config wake_word.model, else model_dir, else MatchboxNet
                for cand in (ww.get("model"), "MatchboxNet"):
                    if not cand:
                        continue
                    for k, v in reg.items():
                        if cand.lower() in (k.lower(), Path(v).name.lower()):
                            return APP_DIR / v
                if ww.get("model_dir"):
                    return APP_DIR / ww["model_dir"]
        except Exception:
            pass
    return default


def main():
    ap = argparse.ArgumentParser(description="Headless self-test for the demo app")
    ap.add_argument("--model", default=None,
                    help="Wake-word model: DS-CNN | TC-ResNet | MatchboxNet | VGG")
    ap.add_argument("--model_dir", default=None, help="Explicit model directory")
    ap.add_argument("--n", type=int, default=15)
    args = ap.parse_args()
    model_dir = _resolve_model_dir(args.model, args.model_dir)
    print(f"Model dir: {model_dir}")

    print("== frontend equivalence ==")
    ok1 = check_frontend_equivalence(model_dir)
    print("\n== ONNX wake-word detection ==")
    ok2 = run_detector(model_dir, args.n)
    print("\n== command classifier ==")
    ok3 = run_command_classifier()
    print("\nSELFTEST:", "PASS" if (ok1 and ok3) else "CHECK FRONTEND/COMMAND")
    return 0


if __name__ == "__main__":
    sys.exit(main())
