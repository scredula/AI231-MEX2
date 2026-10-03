"""
Evaluation metrics for wake-word (binary) and command (multi-class) models.

Includes:
- Accuracy, Precision, Recall, F1
- EER (Equal Error Rate), FAR, FRR at multiple thresholds
- ROC-AUC, PR-AUC
- Top-K accuracy (for multi-class)
- Confusion matrix
"""
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    average_precision_score,
    confusion_matrix,
    top_k_accuracy_score,
)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[List[float]] = None,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    Compute binary classification metrics (wake-word detection).
    
    Args:
        y_true: (N,) binary labels {0, 1}
        y_prob: (N,) predicted probabilities for class 1
        thresholds: List of thresholds for per-threshold metrics
    
    Returns:
        (summary_metrics, per_threshold_metrics)
    """
    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob).ravel()
    
    summary = {}
    
    # Default threshold 0.5
    y_pred = (y_prob >= 0.5).astype(int)
    
    summary["accuracy"] = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    summary["precision"] = float(precision)
    summary["recall"] = float(recall)
    summary["f1"] = float(f1)
    
    # ROC-AUC and PR-AUC
    try:
        summary["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        summary["roc_auc"] = float("nan")
    try:
        summary["pr_auc"] = float(average_precision_score(y_true, y_prob))
    except ValueError:
        summary["pr_auc"] = float("nan")
    
    # EER computation
    eer, eer_threshold = compute_eer(y_true, y_prob)
    summary["eer"] = float(eer)
    summary["eer_threshold"] = float(eer_threshold)
    
    # Confusion matrix at 0.5
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        summary["tn"] = int(tn)
        summary["fp"] = int(fp)
        summary["fn"] = int(fn)
        summary["tp"] = int(tp)
        summary["far"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        summary["frr"] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    
    # Per-threshold metrics
    per_threshold = {}
    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        cm_t = confusion_matrix(y_true, y_pred_t, labels=[0, 1])
        if cm_t.shape == (2, 2):
            tn, fp, fn, tp = cm_t.ravel()
        else:
            tn = fp = fn = tp = 0
        
        acc_t = accuracy_score(y_true, y_pred_t)
        prec_t, rec_t, f1_t, _ = precision_recall_fscore_support(
            y_true, y_pred_t, average="binary", zero_division=0
        )
        far_t = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        frr_t = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        per_threshold[f"{t:.1f}"] = {
            "accuracy": float(acc_t),
            "precision": float(prec_t),
            "recall": float(rec_t),
            "f1": float(f1_t),
            "far": float(far_t),
            "frr": float(frr_t),
        }
    
    return summary, per_threshold


def compute_eer(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER) and the threshold at which FAR == FRR.
    
    Returns:
        (eer, threshold)
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob).ravel()
    
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    fnr = 1.0 - tpr
    
    # Find intersection where FPR == FNR
    diff = fpr - fnr
    idx = np.argmin(np.abs(diff))
    
    eer = float((fpr[idx] + fnr[idx]) / 2.0)
    threshold = float(thresholds[idx]) if idx < len(thresholds) else float("nan")
    
    return eer, threshold


def compute_multiclass_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_classes: Optional[int] = None,
    class_names: Optional[List[str]] = None,
) -> Tuple[Dict[str, float], np.ndarray, Dict[str, List[float]]]:
    """
    Compute multi-class classification metrics (command classifier).
    
    Args:
        y_true: (N,) integer labels
        y_prob: (N, C) predicted probabilities
        num_classes: Number of classes
        class_names: Optional class names
    
    Returns:
        (summary_metrics, confusion_matrix, per_class_metrics)
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob)
    
    if num_classes is None:
        num_classes = y_prob.shape[1]
    
    y_pred = y_prob.argmax(axis=1)
    
    summary = {}
    summary["accuracy"] = float(accuracy_score(y_true, y_pred))
    
    # Macro / weighted F1
    _, _, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    summary["f1_macro"] = float(f1_macro)
    summary["f1_weighted"] = float(f1_weighted)
    
    # Top-K accuracy
    for k in [1, 3, 5]:
        if k <= num_classes:
            try:
                summary[f"top{k}_accuracy"] = float(
                    top_k_accuracy_score(y_true, y_prob, k=k, labels=list(range(num_classes)))
                )
            except Exception:
                summary[f"top{k}_accuracy"] = float("nan")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    
    # Per-class precision/recall/f1
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0
    )
    per_class = {
        "precision": precisions.tolist(),
        "recall": recalls.tolist(),
        "f1": f1s.tolist(),
        "support": supports.tolist(),
    }
    if class_names is not None:
        per_class["class_names"] = list(class_names)
    
    return summary, cm, per_class


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    task: str = "binary",
    thresholds: Optional[List[float]] = None,
    num_classes: Optional[int] = None,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Unified metric computation entry point.
    
    Returns a dict with:
        - summary: scalar metrics
        - per_threshold: per-threshold metrics (binary only)
        - confusion_matrix: ndarray
        - per_class: per-class metrics (multiclass only)
    """
    if task == "binary":
        summary, per_threshold = compute_binary_metrics(y_true, y_prob, thresholds)
        cm = confusion_matrix(
            np.asarray(y_true).astype(int).ravel(),
            (np.asarray(y_prob).ravel() >= 0.5).astype(int),
            labels=[0, 1],
        )
        return {
            "summary": summary,
            "per_threshold": per_threshold,
            "confusion_matrix": cm,
        }
    else:
        summary, cm, per_class = compute_multiclass_metrics(
            y_true, y_prob, num_classes, class_names
        )
        return {
            "summary": summary,
            "confusion_matrix": cm,
            "per_class": per_class,
        }