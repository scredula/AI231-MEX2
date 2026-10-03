#!/usr/bin/env python
"""
Single-command validation entry point.

Usage:
    python validate.py --config configs/wake_word_val.yaml --model runs/wake_word/<ts>/best_model.onnx
    python validate.py --config configs/wake_word_val.yaml --model runs/wake_word/<ts>/best_model.pt

Loads the model (ONNX or PyTorch), evaluates on the test/holdout split, and prints:
  - Accuracy, Precision, Recall, F1
  - EER, FAR, FRR at multiple thresholds
  - ROC-AUC, PR-AUC
  - Confusion matrix
and saves plots + metrics.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.config import load_config
from src.utils.seeding import set_seed
from src.utils.visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_det_curve,
    plot_threshold_metrics,
)
from src.data.wake_word_dataset import LogMelSpectrogram, create_dataloaders, create_single_loader
from src.models.kws import create_model
from src.training.train_loop import validate as validate_loader
from src.training.metrics import compute_all_metrics
from src.export.export_onnx import benchmark_onnx, get_onnx_model_info


def parse_args():
    parser = argparse.ArgumentParser(description="Validate wake-word / command model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--model", type=str, required=True, help="Path to .onnx or .pt model")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--benchmark", action="store_true", help="Run latency benchmark")
    parser.add_argument("--arch", type=str, default=None,
                        help="Override model.arch: dscnn|tcresnet|matchboxnet2d|vgg_small")
    return parser.parse_args()


def load_pytorch_model(model_path, config, device):
    """Load a PyTorch checkpoint and return the model + norm stats."""
    ckpt = torch.load(model_path, map_location=device)
    model = create_model(config).to(device)

    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    norm = ckpt.get("norm_stats", {}) if isinstance(ckpt, dict) else {}
    return model, norm


class ONNXWrapper(nn.Module):
    """Wrap an ONNX model so it behaves like an nn.Module for validate()."""

    def __init__(self, onnx_path, num_classes):
        super().__init__()
        import onnxruntime as ort
        self.sess = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.sess.get_inputs()[0].name
        self.num_classes = num_classes

    def forward(self, x):
        out = self.sess.run(None, {self.input_name: x.cpu().numpy()})[0]
        return torch.from_numpy(out)


def _evaluate_onnx(model, loader, num_classes, thresholds):
    """Evaluate a wrapped ONNX model (no loss available)."""
    all_probs = []
    all_labels = []

    for inputs, targets in loader:
        logits = model(inputs)
        if num_classes == 1:
            probs = torch.sigmoid(logits).squeeze(-1)
            labels = targets.squeeze(-1)
        else:
            probs = torch.softmax(logits, dim=-1)
            labels = targets.long()
        all_probs.append(probs.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    task = "binary" if num_classes == 1 else "multiclass"
    details = compute_all_metrics(all_labels, all_probs, task=task,
                                  thresholds=thresholds, num_classes=num_classes)
    details["y_true"] = all_labels
    details["y_prob"] = all_probs

    metrics = {"accuracy": details["summary"].get("accuracy", 0.0)}
    metrics.update(details["summary"])
    return metrics, details


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.arch:
        config.model.arch = args.arch
    set_seed(config.experiment.get("seed", 42))

    device = torch.device("cpu")
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        return 1

    num_classes = config.model.num_classes
    is_onnx = model_path.suffix == ".onnx"

    norm = {}
    if is_onnx:
        print(f"Loading ONNX model: {model_path}")
        model = ONNXWrapper(str(model_path), num_classes)
        norm_file = model_path.parent / "norm_stats.json"
        if norm_file.exists():
            with open(norm_file) as f:
                norm = json.load(f)
    else:
        print(f"Loading PyTorch model: {model_path}")
        model, norm = load_pytorch_model(model_path, config, device)

    audio_cfg = config.audio
    transform = LogMelSpectrogram(
        sample_rate=audio_cfg.sample_rate,
        n_fft=audio_cfg.n_fft,
        hop_length=audio_cfg.hop_length,
        win_length=audio_cfg.win_length,
        n_mels=audio_cfg.n_mels,
        f_min=audio_cfg.f_min,
        f_max=audio_cfg.f_max,
        power=audio_cfg.power,
        normalized=audio_cfg.normalized,
        mean=norm.get("mean"),
        std=norm.get("std"),
    )

    loader = create_single_loader(config, transform, args.split)
    thresholds = [float(t) for t in config.validation.get("thresholds", [0.1, 0.5, 0.9])]

    out_dir = Path(args.output_dir) if args.output_dir else model_path.parent / f"validation_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating on '{args.split}' split ({len(loader.dataset)} samples)...")
    start = time.time()

    if is_onnx:
        metrics, details = _evaluate_onnx(model, loader, num_classes, thresholds)
    else:
        criterion = (nn.BCEWithLogitsLoss() if num_classes == 1 else nn.CrossEntropyLoss())
        metrics, details = validate_loader(model, loader, criterion, device, thresholds)

    elapsed = time.time() - start
    _print_report(details, elapsed, args.split)
    report = {
        "model_path": str(model_path),
        "split": args.split,
        "summary": details["summary"],
        "per_threshold": details.get("per_threshold", {}),
        "per_class": details.get("per_class", {}),
        "eval_time_sec": elapsed,
    }
    _maybe_benchmark(args, config, model_path, is_onnx, report, audio_cfg)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nMetrics saved to {out_dir / 'metrics.json'}")
    _save_plots(details, out_dir, num_classes, args.split)
    print(f"Plots saved to {out_dir}")
    return 0


def _print_report(details, elapsed, split):
    print("\n" + "=" * 70)
    print(f"VALIDATION REPORT ({split.upper()} SPLIT)")
    print("=" * 70)
    for key in sorted(details["summary"].keys()):
        val = details["summary"][key]
        print(f"  {key:<25}: {val:.4f}" if isinstance(val, float) else f"  {key:<25}: {val}")
    print("-" * 70)
    print(f"  {'evaluation_time_s':<25}: {elapsed:.2f}")
    print("=" * 70)
    if details.get("per_threshold"):
        print("\nPer-threshold metrics:")
        print(f"  {'Thresh':<8}{'Acc':<10}{'Prec':<10}{'Rec':<10}{'F1':<10}{'FAR':<10}{'FRR':<10}")
        for t, m in sorted(details["per_threshold"].items()):
            print(f"  {t:<8}{m['accuracy']:<10.4f}{m['precision']:<10.4f}"
                  f"{m['recall']:<10.4f}{m['f1']:<10.4f}{m['far']:<10.4f}{m['frr']:<10.4f}")


def _maybe_benchmark(args, config, model_path, is_onnx, report, audio_cfg):
    enabled = args.benchmark or (
        is_onnx and config.validation.get("benchmark", {}).get("enabled", False)
    )
    if not enabled:
        return
    print("\nBenchmarking ONNX latency (CPU)...")
    time_frames = int(audio_cfg.clip_duration * audio_cfg.sample_rate / audio_cfg.hop_length) + 1
    bench_cfg = config.validation.get("benchmark", {})
    bench = benchmark_onnx(
        str(model_path), input_shape=(1, 1, config.model.n_mels, time_frames),
        num_runs=bench_cfg.get("num_runs", 500),
        warmup_runs=bench_cfg.get("warmup_runs", 100),
    )
    report["onnx_benchmark"] = bench
    print(f"  mean={bench['mean_ms']:.2f}ms p50={bench['p50_ms']:.2f}ms "
          f"p95={bench['p95_ms']:.2f}ms p99={bench['p99_ms']:.2f}ms")
    report["onnx_info"] = get_onnx_model_info(str(model_path))


def _save_plots(details, out_dir, num_classes, split):
    cm = details["confusion_matrix"]
    if num_classes == 1:
        plot_confusion_matrix(cm, out_dir, class_names=["negative", "positive"],
                              model_name=f"wake_word_{split}")
        plot_confusion_matrix(cm, out_dir, class_names=["negative", "positive"],
                              model_name=f"wake_word_{split}", normalize=True)
        if "y_true" in details:
            plot_roc_curve(details["y_true"], details["y_prob"], out_dir,
                           model_name=f"wake_word_{split}")
            plot_det_curve(details["y_true"], details["y_prob"], out_dir,
                           model_name=f"wake_word_{split}")
        if details.get("per_threshold"):
            plot_threshold_metrics(details["per_threshold"], out_dir,
                                   model_name=f"wake_word_{split}")
    else:
        plot_confusion_matrix(cm, out_dir, model_name=f"command_{split}")


if __name__ == "__main__":
    sys.exit(main())