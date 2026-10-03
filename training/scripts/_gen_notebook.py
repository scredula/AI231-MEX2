#!/usr/bin/env python3
"""Generate notebooks/02_wake_word_training.ipynb from embedded cell sources."""
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parent.parent / "notebooks" / "02_wake_word_training.ipynb"

CELLS = []


def md(src):
    CELLS.append(("markdown", src))


def code(src):
    CELLS.append(("code", src))


md(r"""# Wake-Word Training — "Hey Mason"

Full training pipeline for the custom wake-word detector, run on a **single A100 GPU**.

This notebook:
1. Builds the log-mel frontend + negative samples
2. Trains a lightweight **MatchboxNet** (CNN) with SpecAugment + audio augmentations
3. Reports **per-epoch time** and metrics (train/val)
4. Plots **loss** and **accuracy (0-100%)** curves
5. Evaluates on the **holdout/test** set: confusion matrix, EER, FAR/FRR at thresholds, ROC, DET
6. Exports to **ONNX** and benchmarks CPU latency (proxy for Raspberry Pi 5)

> The exact same pipeline is available as scripts: `python train.py --config configs/wake_word_train.yaml`.
""")

code(r"""import os, sys, json, time, random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib

# Repo root = one level above notebooks/
REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(REPO_ROOT))

from src.utils.config import load_config
from src.utils.seeding import set_seed
from src.data.wake_word_dataset import (
    LogMelSpectrogram, SpecAugment, AudioAugmentation, create_dataloaders,
)
from src.models.kws import create_model
from src.training.train_loop import train_one_epoch, validate, make_grad_scaler
from src.training.metrics import compute_all_metrics
from src.export.export_onnx import export_to_onnx, benchmark_onnx, get_onnx_model_info

print("REPO_ROOT:", REPO_ROOT)
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
""")

md(r"""## 1. Configuration & reproducibility

We load `configs/wake_word_train.yaml` and fix all seeds. Only **one GPU** is used
(`torch.cuda.set_device(0)`), even if more are visible.""")

code(r"""config = load_config(REPO_ROOT / "configs" / "wake_word_train.yaml")

SEED = config.experiment.get("seed", 42)
set_seed(SEED, deterministic=True)

# Force single GPU
if torch.cuda.is_available():
    torch.cuda.set_device(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Uncomment to run a fast sanity pass with fewer epochs:
# config.training.epochs = 3
print("Epochs:", config.training.epochs)
""")

md(r"""## 2. Audio frontend & data exploration

Each clip is clipped/padded to a fixed duration, converted to a **log-mel spectrogram**
(64 mel bins), and normalized. Let's inspect clip counts and a few spectrograms.""")

code(r"""audio_cfg = config.audio
transform = LogMelSpectrogram(
    sample_rate=audio_cfg.sample_rate,
    n_fft=audio_cfg.n_fft, hop_length=audio_cfg.hop_length,
    win_length=audio_cfg.win_length, n_mels=audio_cfg.n_mels,
    f_min=audio_cfg.f_min, f_max=audio_cfg.f_max,
    power=audio_cfg.power, normalized=audio_cfg.normalized,
)

spec_augment = SpecAugment(
    freq_mask_param=config.augmentation.spec_augment.freq_mask_param,
    time_mask_param=config.augmentation.spec_augment.time_mask_param,
    num_freq_masks=config.augmentation.spec_augment.num_freq_masks,
    num_time_masks=config.augmentation.spec_augment.num_time_masks,
) if config.augmentation.get("enabled", False) else None

audio_augment = AudioAugmentation(sample_rate=audio_cfg.sample_rate) \
    if config.augmentation.get("enabled", False) else None

train_loader, val_loader, test_loader = create_dataloaders(
    config, transform, spec_augment, audio_augment
)
print("Train:", len(train_loader.dataset))
print("Val:  ", len(val_loader.dataset))
print("Test: ", len(test_loader.dataset))
""")

code(r"""# Compute normalization stats on the validation set (un-augmented)
if audio_cfg.get("normalize", False):
    mean, std = transform.compute_stats(val_loader.dataset, num_samples=500)
    transform.mean, transform.std = float(mean), float(std)
    print(f"mean={mean:.4f}, std={std:.4f}")

# Visualize a few positive vs negative spectrograms
fig, axes = plt.subplots(2, 3, figsize=(15, 6))
picks = {"positive": [], "negative": []}
for i in range(len(train_loader.dataset)):
    x, y = train_loader.dataset[i]
    key = "positive" if y.item() == 1.0 else "negative"
    if len(picks[key]) < 3:
        picks[key].append(x.squeeze(0).numpy())
    if all(len(v) >= 3 for v in picks.values()):
        break

for r, key in enumerate(["positive", "negative"]):
    for c in range(3):
        ax = axes[r, c]
        ax.imshow(picks[key][c], aspect="auto", origin="lower", cmap="viridis")
        ax.set_title(f"{key} #{c}")
        ax.set_xlabel("frames"); ax.set_ylabel("mel")
plt.suptitle("Log-mel spectrograms (train)")
plt.tight_layout(); plt.show()
""")

md(r"""## 3. Model — MatchboxNet (lightweight CNN for RPi5)

Depthwise-separable residual CNN. Small enough for real-time inference on a
Raspberry Pi 5 via ONNX Runtime, but expressive enough for reliable KWS.""")

code(r"""model = create_model(config).to(device)
n_params = model.count_parameters()
print(f"Model: {config.model.get('arch', '?')} | trainable params: {n_params:,}")

with torch.no_grad():
    xb, yb = next(iter(train_loader))
    out = model(xb.to(device))
    print("input batch:", tuple(xb.shape), "-> logits:", tuple(out.shape))
""")

md(r"""## 4. Training

Loss, optimizer, scheduler and AMP scaler. We record per-epoch wall-clock time,
train/val loss and train/val accuracy, plus val F1 and EER.""")

code(r"""loss_cfg = config.training.loss
pos_weight = None
if config.model.num_classes == 1:
    w = train_loader.dataset.get_class_weights()
    pos_weight = (w[1] / w[0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
else:
    criterion = nn.CrossEntropyLoss(
        label_smoothing=loss_cfg.get("label_smoothing", 0.0))
print("Loss:", criterion)

optimizer = torch.optim.AdamW(
    model.parameters(), lr=config.training.optimizer.lr,
    weight_decay=config.training.optimizer.get("weight_decay", 1e-4))

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=config.training.scheduler.get("T_0", 10),
    T_mult=config.training.scheduler.get("T_mult", 1),
    eta_min=config.training.scheduler.get("eta_min", 1e-6))

use_amp = config.training.get("amp", True) and device.type == "cuda"
scaler = make_grad_scaler(use_amp)
print("AMP:", use_amp)
""")

code(r"""epochs = config.training.epochs
grad_clip = config.training.get("grad_clip_norm", None)
val_thresholds = list(config.validation.get("thresholds", [0.5]))

history = {k: [] for k in
           ["train_loss", "val_loss", "train_acc", "val_acc",
            "train_f1", "val_f1", "val_eer", "lr", "epoch_time"]}

best_score = float("inf"); best_epoch = -1; best_details = None
total_start = time.time()

for epoch in range(1, epochs + 1):
    t0 = time.time()
    tr = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch,
                         scaler=scaler, grad_clip_norm=grad_clip,
                         log_interval=10**9, amp=use_amp)
    va, va_details = validate(model, val_loader, criterion, device,
                              thresholds=val_thresholds)
    scheduler.step()
    dt = time.time() - t0

    history["train_loss"].append(tr["loss"]); history["val_loss"].append(va["loss"])
    history["train_acc"].append(tr["accuracy"]); history["val_acc"].append(va["accuracy"])
    history["train_f1"].append(tr.get("f1", 0.0)); history["val_f1"].append(va.get("f1", 0.0))
    history["val_eer"].append(va.get("eer", float("nan")))
    history["lr"].append(optimizer.param_groups[0]["lr"])
    history["epoch_time"].append(dt)

    if va["eer"] < best_score:
        best_score = va["eer"]; best_epoch = epoch; best_details = va_details
        torch.save({"model_state_dict": model.state_dict(),
                    "norm_stats": {"mean": transform.mean, "std": transform.std}},
                   REPO_ROOT / "models" / "wake_word" / "best_model.pt")

    print(f"Epoch {epoch:>3}/{epochs} | "
          f"train_loss={tr['loss']:.4f} val_loss={va['loss']:.4f} | "
          f"train_acc={tr['accuracy']*100:5.1f}% val_acc={va['accuracy']*100:5.1f}% | "
          f"val_eer={va.get('eer', float('nan')):.4f} | "
          f"lr={optimizer.param_groups[0]['lr']:.2e} | {dt:.1f}s")

total_train_time = time.time() - total_start
print(f"\nBest epoch: {best_epoch} (val EER={best_score:.4f})")
print(f"Total training time: {total_train_time/60:.2f} min")
""")

md(r"""## 5. Training curves

**Loss** (train vs val) and **accuracy** (train vs val, y-axis forced to 0-100%).""")

code(r"""ep = range(1, len(history["train_loss"]) + 1)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(ep, history["train_loss"], "b-o", ms=4, label="Train Loss")
ax.plot(ep, history["val_loss"], "r-o", ms=4, label="Val Loss")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend(); ax.grid(alpha=.3)
ax.set_title("Wake Word — Training & Validation Loss")
plt.tight_layout(); plt.show()
""")

code(r"""fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(ep, np.array(history["train_acc"]) * 100, "b-o", ms=4, label="Train Accuracy")
ax.plot(ep, np.array(history["val_acc"]) * 100, "r-o", ms=4, label="Val Accuracy")
ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 100)
ax.legend(); ax.grid(alpha=.3)
ax.set_title("Wake Word — Training & Validation Accuracy")
plt.tight_layout(); plt.show()
""")

code(r"""pd.DataFrame({
    "epoch": list(ep),
    "train_loss": history["train_loss"], "val_loss": history["val_loss"],
    "train_acc(%)": np.array(history["train_acc"]) * 100,
    "val_acc(%)": np.array(history["val_acc"]) * 100,
    "val_f1": history["val_f1"], "val_eer": history["val_eer"],
    "lr": history["lr"], "epoch_time(s)": history["epoch_time"],
}).round(4)
""")

md(r"""## 6. Final holdout / validation-set evaluation

Load the best checkpoint and evaluate on the untouched test set. Print keyword
accuracy and a confusion matrix, plus EER and FAR/FRR at multiple thresholds.""")

code(r"""ckpt = torch.load(REPO_ROOT / "models" / "wake_word" / "best_model.pt", map_location=device)
model.load_state_dict(ckpt["model_state_dict"])

test_metrics, test_details = validate(model, test_loader, criterion, device,
                                      thresholds=val_thresholds)

print("Holdout metrics:")
for k in sorted(test_metrics):
    v = test_metrics[k]
    if isinstance(v, float):
        print(f"  {k:<20}: {v:.4f}")
""")

code(r"""from sklearn.metrics import confusion_matrix
cm = test_details["confusion_matrix"]
fig, ax = plt.subplots(figsize=(4.5, 4))
ax.imshow(cm, cmap="Blues")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > cm.max()/2 else "black")
ax.set_xticks([0, 1]); ax.set_xticklabels(["neg", "pos"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["neg", "pos"])
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("Confusion Matrix (holdout)")
plt.tight_layout(); plt.show()
""")

code(r"""rows = []
for t, m in sorted(test_details["per_threshold"].items()):
    rows.append({"threshold": float(t), **m})
pd.DataFrame(rows).round(4)
""")

code(r"""from sklearn.metrics import roc_curve, auc
yt, yp = test_details["y_true"], test_details["y_prob"]
fpr, tpr, _ = roc_curve(yt, yp); roc_auc = auc(fpr, tpr)
fnr = 1 - tpr

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(fpr, tpr, "b-", lw=2, label=f"AUC={roc_auc:.4f}")
axes[0].plot([0, 1], [0, 1], "k--", lw=1)
axes[0].set_xlabel("FAR"); axes[0].set_ylabel("TPR")
axes[0].set_title("ROC Curve"); axes[0].legend(); axes[0].grid(alpha=.3)

axes[1].plot(fpr*100, fnr*100, "b-", lw=2)
d = np.abs(fpr - fnr); i = int(np.argmin(d))
axes[1].plot(fpr[i]*100, fnr[i]*100, "ro", ms=8,
             label=f"EER={(fpr[i]+fnr[i])/2*100:.2f}%")
axes[1].set_xlabel("False Acceptance Rate (%)"); axes[1].set_ylabel("False Rejection Rate (%)")
axes[1].set_title("DET Curve"); axes[1].legend(); axes[1].grid(alpha=.3)
plt.tight_layout(); plt.show()
""")

md(r"""## 7. Training-time summary

Key numbers for the report: parameters, best epoch, and total training time on a single A100.""")

code(r"""print("=" * 55)
print(f"Model                  : {config.model.get('arch', '?')}")
print(f"Parameters             : {n_params:,}")
print(f"Best epoch             : {best_epoch}")
print(f"Best val EER           : {best_score:.4f}")
print(f"Holdout accuracy       : {test_metrics['accuracy']*100:.2f}%")
print(f"Holdout EER            : {test_metrics.get('eer', float('nan')):.4f}")
print(f"Holdout ROC-AUC        : {test_metrics.get('roc_auc', float('nan')):.4f}")
print(f"Total training time    : {total_train_time/60:.2f} min")
print(f"Mean epoch time        : {np.mean(history['epoch_time']):.2f} s")
print("=" * 55)
""")

md(r"""## 8. ONNX export & RPi5 latency benchmark

Export the trained model to ONNX (opset 17, dynamic batch/time) and measure
CPU latency as a proxy for Raspberry Pi 5.""")

code(r"""time_frames = int(audio_cfg.clip_duration * audio_cfg.sample_rate / audio_cfg.hop_length) + 1
input_shape = (1, 1, config.model.n_mels, time_frames)

onnx_dir = REPO_ROOT / "models" / "wake_word"
onnx_path = onnx_dir / "best_model.onnx"
export_to_onnx(model, str(onnx_path), input_shape=input_shape,
               opset_version=17, simplify=True, verify=True)

info = get_onnx_model_info(str(onnx_path))
print(json.dumps(info, indent=2)[:800])

bench = benchmark_onnx(str(onnx_path), input_shape=input_shape, num_runs=200)
print("CPU latency (ms):",
      {k: round(v, 3) for k, v in bench.items() if isinstance(v, (int, float))})
""")

md(r"""## Summary

- Trained a lightweight **MatchboxNet** for "Hey Mason" wake-word detection.
- Negative samples from LibriSpeech (speech), ESC-50 (noise) and synthetic room tone.
- Reported loss & accuracy curves, confusion matrix, EER/FAR/FRR at thresholds, ROC/DET.
- Exported to **ONNX** and benchmarked CPU latency (target < 500 ms on RPi5).
""")


def main():
    nb = {
        "cells": [
            {"cell_type": ctype, "metadata": {},
             "source": [l + "\n" for l in src.split("\n")[:-1]] + [src.split("\n")[-1]]}
            for ctype, src in CELLS
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    # code cells need "outputs" and "execution_count"
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(NB_PATH, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"Wrote {NB_PATH} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
