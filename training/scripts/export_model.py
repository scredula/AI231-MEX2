#!/usr/bin/env python3
"""
Export a trained checkpoint (.pt) to ONNX and benchmark it.

Usage:
    python scripts/export_model.py \
        --config configs/wake_word_train.yaml \
        --model runs/wake_word/wake_word_hey_mason/<ts>/best_model.pt \
        --out_dir models/wake_word
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_config, save_config
from src.models.kws import create_model
from src.export.export_onnx import export_to_onnx, benchmark_onnx, get_onnx_model_info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True, help="Path to best_model.pt")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--arch", type=str, default=None,
                    help="Override model.arch: dscnn|tcresnet|matchboxnet2d|vgg_small")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.arch:
        cfg.model.arch = args.arch
    device = torch.device("cpu")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    ckpt = torch.load(args.model, map_location=device)
    model = create_model(cfg).to(device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    norm = ckpt.get("norm_stats", {}) if isinstance(ckpt, dict) else {}
    if norm:
        with open(out_dir / "norm_stats.json", "w") as f:
            json.dump(norm, f, indent=2)

    time_frames = int(cfg.audio.clip_duration * cfg.audio.sample_rate / cfg.audio.hop_length) + 1
    input_shape = (1, 1, cfg.model.n_mels, time_frames)
    onnx_path = out_dir / "best_model.onnx"

    export_to_onnx(model, str(onnx_path), input_shape=input_shape,
                   opset_version=args.opset, simplify=True, verify=True)

    info = get_onnx_model_info(str(onnx_path))
    with open(out_dir / "onnx_info.json", "w") as f:
        json.dump(info, f, indent=2)
    print("ONNX info:", json.dumps(info, indent=2)[:600])

    bench = benchmark_onnx(str(onnx_path), input_shape=input_shape, num_runs=200)
    with open(out_dir / "onnx_benchmark.json", "w") as f:
        json.dump(bench, f, indent=2)
    print("CPU latency (ms):", {k: round(v, 3) for k, v in bench.items()
                                if isinstance(v, (int, float))})

    # Also copy the .pt for reference
    shutil.copy(args.model, out_dir / "best_model.pt")
    print(f"Artifacts written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
