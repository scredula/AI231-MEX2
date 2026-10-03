"""
Training and validation loops for wake-word / command models.
"""
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import compute_all_metrics


class AverageMeter:
    """Tracks running average of a metric."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)


def _compute_loss(criterion, outputs, targets, num_classes):
    """Compute loss depending on task type."""
    if num_classes == 1:
        return criterion(outputs.squeeze(-1), targets.squeeze(-1))
    return criterion(outputs, targets.long())


def _autocast_ctx():
    """Return an autocast context manager compatible with old/new PyTorch."""
    try:
        return torch.amp.autocast(device_type="cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast()


def make_grad_scaler(enabled: bool):
    """Create a GradScaler compatible with old/new PyTorch."""
    if not enabled:
        return None
    try:
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    grad_clip_norm: Optional[float] = None,
    log_interval: int = 50,
    logger=None,
    amp: bool = True,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    loss_meter = AverageMeter()
    correct = 0
    total = 0
    
    all_probs = []
    all_labels = []
    
    start_time = time.time()
    
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        if amp and scaler is not None:
            with _autocast_ctx():
                outputs = model(inputs)
                loss = _compute_loss(criterion, outputs, targets, model.num_classes)
            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = _compute_loss(criterion, outputs, targets, model.num_classes)
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
        
        batch_size = inputs.size(0)
        loss_meter.update(loss.item(), batch_size)
        
        with torch.no_grad():
            if model.num_classes == 1:
                probs = torch.sigmoid(outputs).squeeze(-1)
                preds = (probs >= 0.5).float()
                labels = targets.squeeze(-1)
            else:
                probs = torch.softmax(outputs, dim=-1)
                preds = probs.argmax(dim=-1)
                labels = targets.long()
            
            correct += (preds == labels).sum().item()
            total += batch_size
            
            all_probs.append(probs.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
        
        if logger is not None and (batch_idx + 1) % log_interval == 0:
            logger.log_metrics(
                {"loss": loss_meter.avg, "acc": correct / max(total, 1)},
                step=epoch * len(loader) + batch_idx,
                prefix="train_batch",
                print_console=False,
            )
    
    epoch_time = time.time() - start_time
    
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    metrics = {
        "loss": loss_meter.avg,
        "accuracy": correct / max(total, 1),
        "epoch_time": epoch_time,
    }
    
    if model.num_classes == 1:
        from .metrics import compute_binary_metrics
        summary, _ = compute_binary_metrics(all_labels, all_probs, thresholds=[])
        metrics["f1"] = summary.get("f1", 0.0)
        metrics["eer"] = summary.get("eer", float("nan"))
    else:
        from sklearn.metrics import f1_score
        metrics["f1"] = float(
            f1_score(all_labels, all_probs.argmax(-1), average="macro", zero_division=0)
        )
    
    return metrics


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    thresholds: Optional[list] = None,
) -> Tuple[Dict[str, float], Dict]:
    """
    Validate the model.
    
    Returns:
        (metrics_dict, detailed_results)
    """
    model.eval()
    
    loss_meter = AverageMeter()
    correct = 0
    total = 0
    
    all_probs = []
    all_labels = []
    
    start_time = time.time()
    
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        outputs = model(inputs)
        loss = _compute_loss(criterion, outputs, targets, model.num_classes)
        
        batch_size = inputs.size(0)
        loss_meter.update(loss.item(), batch_size)
        
        if model.num_classes == 1:
            probs = torch.sigmoid(outputs).squeeze(-1)
            preds = (probs >= 0.5).float()
            labels = targets.squeeze(-1)
        else:
            probs = torch.softmax(outputs, dim=-1)
            preds = probs.argmax(dim=-1)
            labels = targets.long()
        
        correct += (preds == labels).sum().item()
        total += batch_size
        
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    
    epoch_time = time.time() - start_time
    
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    task = "binary" if model.num_classes == 1 else "multiclass"
    detailed = compute_all_metrics(
        all_labels, all_probs, task=task, thresholds=thresholds,
        num_classes=model.num_classes if task == "multiclass" else None,
    )
    
    metrics = {
        "loss": loss_meter.avg,
        "accuracy": correct / max(total, 1),
        "epoch_time": epoch_time,
    }
    metrics.update(detailed["summary"])
    
    # Stash raw predictions for plotting
    detailed["y_true"] = all_labels
    detailed["y_prob"] = all_probs
    
    return metrics, detailed