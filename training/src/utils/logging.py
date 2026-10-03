"""
Logging utilities: TensorBoard, CSV, console.
"""
import os
import csv
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from contextlib import contextmanager

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = None


class CSVLogger:
    """Simple CSV logger for metrics."""
    
    def __init__(self, log_dir: Path, filename: str = "metrics.csv"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.filepath = self.log_dir / filename
        self._file = None
        self._writer = None
        self._header_written = False
        self._fieldnames = None
    
    def log(self, metrics: Dict[str, Any], step: int) -> None:
        if self._file is None:
            self._file = open(self.filepath, "w", newline="")
            self._fieldnames = ["step"] + sorted([k for k in metrics.keys() if k != "step"])
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
            self._writer.writeheader()
            self._header_written = True
        
        row = {"step": step}
        row.update({k: v for k, v in metrics.items() if k != "step"})
        self._writer.writerow(row)
        self._file.flush()
    
    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


class TensorBoardLogger:
    """TensorBoard logger wrapper."""
    
    def __init__(self, log_dir: Path, enabled: bool = True):
        self.enabled = enabled and TENSORBOARD_AVAILABLE
        self.writer = None
        if self.enabled:
            self.writer = SummaryWriter(log_dir=str(log_dir))
    
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        if self.enabled and self.writer:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, tag: str, values: Dict[str, float], step: int) -> None:
        if self.enabled and self.writer:
            self.writer.add_scalars(tag, values, step)
    
    def log_histogram(self, tag: str, values, step: int) -> None:
        if self.enabled and self.writer:
            self.writer.add_histogram(tag, values, step)
    
    def log_figure(self, tag: str, figure, step: int) -> None:
        if self.enabled and self.writer:
            self.writer.add_figure(tag, figure, step)
    
    def log_model_graph(self, model, input_sample) -> None:
        if self.enabled and self.writer:
            try:
                self.writer.add_graph(model, input_sample)
            except Exception:
                pass  # Graph logging can fail for complex models
    
    def close(self) -> None:
        if self.enabled and self.writer:
            self.writer.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


class MetricsLogger:
    """Combined logger for console, CSV, and TensorBoard."""
    
    def __init__(
        self,
        log_dir: Path,
        experiment_name: str,
        use_tensorboard: bool = True,
        use_csv: bool = True,
        log_interval: int = 50,
    ):
        self.log_dir = Path(log_dir) / experiment_name / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_interval = log_interval
        self.step = 0
        self.epoch = 0
        
        # Separate CSV files: epoch-level metrics vs per-batch metrics
        # (they have different schemas, so they must not share a CSV).
        self.csv_logger = CSVLogger(self.log_dir, "metrics.csv") if use_csv else None
        self.batch_csv_logger = CSVLogger(self.log_dir, "train_batches.csv") if use_csv else None
        self.tb_logger = TensorBoardLogger(self.log_dir, enabled=use_tensorboard)
        
        # Console logging
        self._console = sys.stdout
        
        # Track best metrics
        self.best_metrics = {}
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        prefix: str = "",
        print_console: bool = True,
    ) -> None:
        if step is not None:
            self.step = step
        
        # Console
        if print_console:
            self._log_console(metrics, prefix)
        
        # CSV (route batch vs epoch to separate files)
        if self.csv_logger:
            target = self.batch_csv_logger if prefix.startswith("train_batch") else self.csv_logger
            if target is not None:
                target.log(metrics, self.step)
        
        # TensorBoard
        if self.tb_logger.enabled:
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    tag = f"{prefix}/{key}" if prefix else key
                    self.tb_logger.log_scalar(tag, value, self.step)
    
    def log_epoch_metrics(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        epoch: int,
        lr: float,
        epoch_time: float,
    ) -> None:
        self.epoch = epoch
        
        # Combine metrics
        all_metrics = {}
        all_metrics.update({f"train/{k}": v for k, v in train_metrics.items()})
        all_metrics.update({f"val/{k}": v for k, v in val_metrics.items()})
        all_metrics["lr"] = lr
        all_metrics["epoch_time"] = epoch_time
        
        self.log_metrics(all_metrics, step=epoch, prefix="epoch", print_console=True)
        
        # Update best metrics
        for key, value in val_metrics.items():
            if key not in self.best_metrics:
                self.best_metrics[key] = value
            else:
                # Assume lower is better for loss/eer, higher for accuracy/f1
                if "loss" in key or "eer" in key or "error" in key:
                    if value < self.best_metrics[key]:
                        self.best_metrics[key] = value
                else:
                    if value > self.best_metrics[key]:
                        self.best_metrics[key] = value
    
    def _log_console(self, metrics: Dict[str, float], prefix: str) -> None:
        prefix_str = f"[{prefix}] " if prefix else ""
        parts = [f"{prefix_str}Step {self.step}"]
        for key, value in sorted(metrics.items()):
            if isinstance(value, float):
                parts.append(f"{key}={value:.4f}")
            else:
                parts.append(f"{key}={value}")
        print(" | ".join(parts), flush=True)
    
    def log_figure(self, tag: str, figure, step: Optional[int] = None) -> None:
        if step is None:
            step = self.step
        if self.tb_logger.enabled:
            self.tb_logger.log_figure(tag, figure, step)
    
    def log_model_graph(self, model, input_sample) -> None:
        if self.tb_logger.enabled:
            self.tb_logger.log_model_graph(model, input_sample)
    
    def get_log_dir(self) -> Path:
        return self.log_dir
    
    def close(self) -> None:
        if self.csv_logger:
            self.csv_logger.close()
        if getattr(self, "batch_csv_logger", None):
            self.batch_csv_logger.close()
        self.tb_logger.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


def setup_logging(log_dir: Path, level: str = "INFO") -> None:
    """Setup root logger."""
    import logging
    
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "training.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )