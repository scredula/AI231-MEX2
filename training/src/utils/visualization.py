"""
Plotting utilities for training curves and evaluation results.
"""
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc


def plot_training_curves(
    history: Dict[str, List[float]],
    output_dir: Path,
    model_name: str = "model",
) -> Dict[str, str]:
    """
    Plot training and validation loss and accuracy curves.
    
    history keys expected: train_loss, val_loss, train_acc, val_acc
    
    Returns dict of saved plot paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = {}
    
    epochs = range(1, len(history.get("train_loss", [])) + 1)
    
    # --- Loss curve ---
    fig, ax = plt.subplots(figsize=(10, 6))
    if "train_loss" in history:
        ax.plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=4)
    if "val_loss" in history:
        ax.plot(epochs, history["val_loss"], "r-o", label="Val Loss", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{model_name} - Training & Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = output_dir / f"{model_name}_loss_curve.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved["loss_curve"] = str(path)
    
    # --- Accuracy curve (0-100%) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    if "train_acc" in history:
        train_acc_pct = [a * 100.0 for a in history["train_acc"]]
        ax.plot(epochs, train_acc_pct, "b-o", label="Train Accuracy", markersize=4)
    if "val_acc" in history:
        val_acc_pct = [a * 100.0 for a in history["val_acc"]]
        ax.plot(epochs, val_acc_pct, "r-o", label="Val Accuracy", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{model_name} - Training & Validation Accuracy")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)
    path = output_dir / f"{model_name}_accuracy_curve.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    saved["accuracy_curve"] = str(path)
    
    return saved


def plot_confusion_matrix(
    cm: np.ndarray,
    output_dir: Path,
    class_names: Optional[List[str]] = None,
    model_name: str = "model",
    normalize: bool = False,
) -> str:
    """Plot confusion matrix heatmap."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if normalize:
        cm_plot = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        fmt = ".2f"
    else:
        cm_plot = cm
        fmt = "d"
    
    fig, ax = plt.subplots(figsize=(max(8, len(cm_plot) * 0.6), max(6, len(cm_plot) * 0.5)))
    
    if class_names is None:
        class_names = [str(i) for i in range(len(cm_plot))]
    
    sns.heatmap(
        cm_plot, annot=len(cm_plot) <= 12, fmt=fmt, cmap="Blues",
        xticklabels=class_names, yticklabels=class_names, ax=ax,
        cbar_kws={"label": "Proportion" if normalize else "Count"},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{model_name} - Confusion Matrix{' (Normalized)' if normalize else ''}")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    
    suffix = "_normalized" if normalize else ""
    path = output_dir / f"{model_name}_confusion_matrix{suffix}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    
    return str(path)


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_dir: Path,
    model_name: str = "model",
) -> str:
    """Plot ROC curve for binary classification."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, "b-", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    ax.set_xlabel("False Positive Rate (FAR)")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{model_name} - ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    path = output_dir / f"{model_name}_roc_curve.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    
    return str(path)


def plot_det_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_dir: Path,
    model_name: str = "model",
) -> str:
    """Plot DET curve (FAR vs FRR)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fnr = 1 - tpr
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr * 100, fnr * 100, "b-", lw=2)
    ax.set_xlabel("False Acceptance Rate (%)")
    ax.set_ylabel("False Rejection Rate (%)")
    ax.set_title(f"{model_name} - DET Curve")
    ax.grid(True, alpha=0.3)
    
    # Mark EER
    diff = np.abs(fpr - fnr)
    idx = np.argmin(diff)
    eer = (fpr[idx] + fnr[idx]) / 2 * 100
    ax.plot(fpr[idx] * 100, fnr[idx] * 100, "ro", markersize=8,
            label=f"EER = {eer:.2f}%")
    ax.legend()
    
    path = output_dir / f"{model_name}_det_curve.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    
    return str(path)


def plot_threshold_metrics(
    per_threshold: Dict[str, Dict[str, float]],
    output_dir: Path,
    model_name: str = "model",
) -> str:
    """Plot FAR/FRR/accuracy/F1 vs threshold."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    thresholds = sorted([float(k) for k in per_threshold.keys()])
    accs = [per_threshold[f"{t:.1f}"]["accuracy"] * 100 for t in thresholds]
    fars = [per_threshold[f"{t:.1f}"]["far"] * 100 for t in thresholds]
    frrs = [per_threshold[f"{t:.1f}"]["frr"] * 100 for t in thresholds]
    f1s = [per_threshold[f"{t:.1f}"]["f1"] * 100 for t in thresholds]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, accs, "b-o", label="Accuracy")
    ax.plot(thresholds, f1s, "g-s", label="F1")
    ax.plot(thresholds, fars, "r-^", label="FAR")
    ax.plot(thresholds, frrs, "m-v", label="FRR")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric (%)")
    ax.set_title(f"{model_name} - Metrics vs Threshold")
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    path = output_dir / f"{model_name}_threshold_metrics.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    
    return str(path)


def plot_mel_spectrogram(
    spec: np.ndarray,
    output_dir: Path,
    title: str = "Log-Mel Spectrogram",
    filename: str = "spectrogram.png",
) -> str:
    """Plot a single log-mel spectrogram."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if spec.ndim == 3:
        spec = spec.squeeze(0)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    img = ax.imshow(spec, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xlabel("Time frames")
    ax.set_ylabel("Mel bins")
    ax.set_title(title)
    fig.colorbar(img, ax=ax, label="dB")
    
    path = output_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    
    return str(path)