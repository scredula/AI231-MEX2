#!/usr/bin/env python
"""
Single-command training entry point.

Usage:
    python train.py --config configs/wake_word_train.yaml
    python train.py --config configs/wake_word_train.yaml --output_dir runs/my_exp

This script:
  1. Loads config
  2. Builds datasets (log-mel frontend, negatives, augmentations)
  3. Builds the MatchboxNet model
  4. Trains with early stopping
  5. Saves best checkpoint (.pt), ONNX model, metrics.json, and plots
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

# Add repo root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.config import load_config, save_config, Config
from src.utils.seeding import set_seed, worker_init_fn
from src.utils.logging import MetricsLogger
from src.utils.visualization import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_det_curve,
    plot_threshold_metrics,
)
from src.data.wake_word_dataset import (
    LogMelSpectrogram,
    SpecAugment,
    AudioAugmentation,
    create_dataloaders,
)
from src.data.command_dataset import create_command_dataloaders
from src.models.kws import create_model
from src.training.train_loop import train_one_epoch, validate, make_grad_scaler
from src.export.export_onnx import export_to_onnx, benchmark_onnx, get_onnx_model_info


def parse_args():
    parser = argparse.ArgumentParser(description="Train wake-word / command model")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output dir")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--arch", type=str, default=None,
                        help="Override model.arch: dscnn|tcresnet|matchboxnet2d|vgg_small")
    return parser.parse_args()


def build_criterion(config, train_dataset, device):
    """Build loss function from config."""
    loss_cfg = config.training.loss
    loss_type = loss_cfg.get("type", "bce_with_logits")
    
    if loss_type == "bce_with_logits":
        pos_weight = loss_cfg.get("pos_weight", None)
        if pos_weight is None:
            # Auto-compute from class balance
            weights = train_dataset.get_class_weights()
            pos_weight = weights[1] / weights[0]  # ratio
            print(f"Auto pos_weight (neg/pos ratio) = {pos_weight:.3f}")
        pos_weight = torch.tensor([float(pos_weight)], device=device)
        label_smoothing = float(loss_cfg.get("label_smoothing", 0.0))
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight), pos_weight
    elif loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(label_smoothing=float(loss_cfg.get("label_smoothing", 0.0))), None
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def build_optimizer(config, model):
    opt_cfg = config.training.optimizer
    opt_type = opt_cfg.get("type", "adamw").lower()
    lr = float(opt_cfg.lr)
    wd = float(opt_cfg.get("weight_decay", 1e-4))

    if opt_type == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=wd,
            betas=tuple(float(b) for b in opt_cfg.get("betas", [0.9, 0.999])),
            eps=float(opt_cfg.get("eps", 1e-8)),
        )
    elif opt_type == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt_type == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr,
                               momentum=float(opt_cfg.get("momentum", 0.9)),
                               weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")


def build_scheduler(config, optimizer):
    sch_cfg = config.training.get("scheduler", None)
    if sch_cfg is None:
        return None
    sch_type = sch_cfg.get("type", "cosine_annealing_warm_restarts").lower()
    
    if sch_type == "cosine_annealing_warm_restarts":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=int(sch_cfg.get("T_0", 10)),
            T_mult=int(sch_cfg.get("T_mult", 1)),
            eta_min=float(sch_cfg.get("eta_min", 1e-6)),
        )
    elif sch_type == "cosine_annealing":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training.epochs,
            eta_min=float(sch_cfg.get("eta_min", 1e-6)),
        )
    elif sch_type == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(sch_cfg.get("step_size", 10)),
            gamma=float(sch_cfg.get("gamma", 0.1)),
        )
    elif sch_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=int(sch_cfg.get("patience", 5)),
        )
    else:
        raise ValueError(f"Unknown scheduler: {sch_type}")

def run_training(
    config, args, model, train_loader, val_loader, test_loader,
    criterion, optimizer, scheduler, scaler, use_amp, transform,
    logger, run_dir, device, num_params,
):
    """Execute the training loop, evaluation, plotting and export."""
    epochs = int(config.training.epochs)
    grad_clip = config.training.get("grad_clip_norm", None)
    if grad_clip is not None:
        grad_clip = float(grad_clip)
    val_thresholds = [float(t) for t in config.validation.get("thresholds", [0.5])]

    early_cfg = config.training.get("early_stopping", None)
    es_enabled = early_cfg is not None and early_cfg.get("enabled", False)
    es_patience = int(early_cfg.get("patience", 10)) if early_cfg else 10
    es_min_delta = float(early_cfg.get("min_delta", 0.0)) if early_cfg else 0.0

    ckpt_cfg = config.training.get("checkpoint", None)
    ckpt_metric = ckpt_cfg.get("metric", "val_loss") if ckpt_cfg else "val_loss"
    ckpt_mode = ckpt_cfg.get("mode", "min") if ckpt_cfg else "min"

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [],
               "train_f1": [], "val_f1": [], "val_eer": [], "epoch_time": []}

    best_score = float("inf") if ckpt_mode == "min" else float("-inf")
    best_epoch = -1
    es_counter = 0
    best_details = None

    total_start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch,
            scaler=scaler, grad_clip_norm=grad_clip,
            log_interval=config.logging.get("log_interval", 50),
            logger=logger, amp=use_amp,
        )

        val_metrics, val_details = validate(
            model, val_loader, criterion, device, thresholds=val_thresholds,
        )

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics["loss"])
            else:
                scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.time() - epoch_start

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["train_f1"].append(train_metrics.get("f1", 0.0))
        history["val_f1"].append(val_metrics.get("f1", 0.0))
        history["val_eer"].append(val_metrics.get("eer", float("nan")))
        history["epoch_time"].append(epoch_time)

        logger.log_epoch_metrics(train_metrics, val_metrics, epoch, lr, epoch_time)

        monitor_value = val_metrics.get(ckpt_metric.replace("val_", ""),
                                        val_metrics.get("loss"))
        improved = (monitor_value < best_score - es_min_delta) if ckpt_mode == "min" \
            else (monitor_value > best_score + es_min_delta)

        if improved:
            best_score = monitor_value
            best_epoch = epoch
            best_details = val_details
            es_counter = 0

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_score": best_score,
                "config": config.to_dict(),
                "norm_stats": {"mean": transform.mean, "std": transform.std},
                "val_metrics": val_metrics,
            }, run_dir / "best_model.pt")
            print(f"  --> New best model saved (epoch {epoch}, {ckpt_metric}={best_score:.4f})")
        else:
            es_counter += 1
            if es_enabled and es_counter >= es_patience:
                print(f"Early stopping triggered at epoch {epoch} (patience={es_patience})")
                break

    total_time = time.time() - total_start
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = total_time % 60
    print(f"\nTotal training time: {hours}h {minutes}m {seconds:.1f}s")

    checkpoint = torch.load(run_dir / "best_model.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("\nEvaluating on test/holdout set...")
    test_metrics, test_details = validate(
        model, test_loader, criterion, device, thresholds=val_thresholds,
    )

    results = {
        "experiment": config.experiment.name,
        "config_path": str(args.config),
        "best_epoch": best_epoch,
        "num_params": num_params,
        "total_train_time_sec": total_time,
        "history": history,
        "test_metrics": {k: v for k, v in test_metrics.items()},
        "test_per_threshold": test_details.get("per_threshold", {}),
        "norm_stats": {"mean": transform.mean, "std": transform.std},
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Metrics saved to {run_dir / 'metrics.json'}")

    # ---- Plots ----
    plot_training_curves(history, run_dir, model_name=config.experiment.name)

    if model.num_classes == 1:
        plot_confusion_matrix(test_details["confusion_matrix"], run_dir,
                              class_names=["negative", "positive"],
                              model_name=config.experiment.name)
        plot_roc_curve(test_details["y_true"], test_details["y_prob"], run_dir,
                       model_name=config.experiment.name)
        plot_det_curve(test_details["y_true"], test_details["y_prob"], run_dir,
                       model_name=config.experiment.name)
        if test_details.get("per_threshold"):
            plot_threshold_metrics(test_details["per_threshold"], run_dir,
                                   model_name=config.experiment.name)
    else:
        plot_confusion_matrix(test_details["confusion_matrix"], run_dir,
                              model_name=config.experiment.name)

    return _finish(config, model, transform, run_dir, results, test_metrics, test_details)



def _finish(config, model, transform, run_dir, results, test_metrics, test_details):
    """Export ONNX + print final summary."""
    print("\nExporting to ONNX...")
    tmean = transform.mean if transform.mean is not None else 0.0
    tstd = transform.std if transform.std is not None else 1.0

    model.eval()
    time_frames = int(config.audio.clip_duration * config.audio.sample_rate
                      / config.audio.hop_length) + 1
    input_shape = (1, 1, config.model.n_mels, time_frames)

    onnx_path = run_dir / "best_model.onnx"
    try:
        export_to_onnx(model, str(onnx_path), input_shape=input_shape,
                       opset_version=17, simplify=True, verify=True)
        info = get_onnx_model_info(str(onnx_path))
        with open(run_dir / "onnx_info.json", "w") as f:
            json.dump(info, f, indent=2)

        print("Benchmarking ONNX inference (CPU)...")
        bench = benchmark_onnx(str(onnx_path), input_shape=input_shape, num_runs=200)
        with open(run_dir / "onnx_benchmark.json", "w") as f:
            json.dump(bench, f, indent=2)
        print(f"  Latency: mean={bench['mean_ms']:.2f}ms, p95={bench['p95_ms']:.2f}ms, "
              f"p99={bench['p99_ms']:.2f}ms")
        results["onnx_benchmark"] = bench
        with open(run_dir / "metrics.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
    except Exception as e:
        print(f"ONNX export failed: {e}")

    print("\n" + "=" * 70)
    print("FINAL TEST/HOLDOUT METRICS")
    print("=" * 70)
    for key in sorted(test_metrics.keys()):
        val = test_metrics[key]
        if isinstance(val, float):
            print(f"  {key:<25}: {val:.4f}")
    print("=" * 70)
    print(f"Artifacts saved in: {run_dir}")
    return 0




def main():
    args = parse_args()
    config = load_config(args.config)

    if args.arch:
        config.model.arch = args.arch
        print(f"Using architecture: {args.arch}")

    # Avoid "storage that is not resizable" issues with multi-worker loaders
    try:
        torch.multiprocessing.set_sharing_strategy("file_system")
    except Exception:
        pass

    if args.output_dir:
        config.experiment.output_dir = args.output_dir

    seed = config.experiment.get("seed", 42)
    set_seed(seed, deterministic=True)
    print(f"Random seed set to {seed}")

    device_str = config.experiment.get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device_str = "cpu"
    device = torch.device(device_str)
    print(f"Using device: {device}")

    logger = MetricsLogger(
        log_dir=config.experiment.output_dir,
        experiment_name=config.experiment.name,
        use_tensorboard=config.logging.get("tensorboard", True),
        use_csv=config.logging.get("csv_log", True),
        log_interval=config.logging.get("log_interval", 50),
    )
    run_dir = logger.get_log_dir()
    print(f"Run directory: {run_dir}")
    save_config(config, run_dir / "config.yaml")

    # ---- Log-mel transform ----
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
    )

    spec_augment = None
    audio_augment = None
    aug_cfg = config.augmentation
    if aug_cfg.get("enabled", False):
        sa = aug_cfg.spec_augment
        spec_augment = SpecAugment(
            freq_mask_param=sa.freq_mask_param,
            time_mask_param=sa.time_mask_param,
            num_freq_masks=sa.num_freq_masks,
            num_time_masks=sa.num_time_masks,
        )
        ts = aug_cfg.time_stretch
        ps = aug_cfg.pitch_shift
        ni = aug_cfg.noise_injection
        gn = aug_cfg.gain
        audio_augment = AudioAugmentation(
            sample_rate=audio_cfg.sample_rate,
            time_stretch_range=tuple(ts.rate_range) if ts.enabled else (1.0, 1.0),
            pitch_shift_range=tuple(ps.n_steps_range) if ps.enabled else (0, 0),
            gain_db_range=(gn.min_gain_db, gn.max_gain_db) if gn.enabled else (0, 0),
            noise_snr_range=tuple(ni.snr_range) if ni.enabled else (0, 0),
        )

    # ---- Dataloaders ----
    # A command config declares `data.manifest`; a wake-word config declares
    # `data.positive_train` etc. Pick the matching builder.
    if config.data.get("manifest"):
        print(f"Command dataset: {config.data.manifest}")
        train_loader, val_loader, test_loader = create_command_dataloaders(
            config, transform, spec_augment, audio_augment
        )
    else:
        train_loader, val_loader, test_loader = create_dataloaders(
            config, transform, spec_augment, audio_augment
        )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # ---- Normalization stats ----
    if audio_cfg.get("normalize", False):
        if audio_cfg.get("mean") is not None and audio_cfg.get("std") is not None:
            transform.mean = audio_cfg.mean
            transform.std = audio_cfg.std
        else:
            print("Computing normalization stats from validation set (un-augmented)...")
            mean, std = transform.compute_stats(val_loader.dataset, num_samples=256)
            transform.mean = float(mean)
            transform.std = float(std)
            print(f"Computed mean={mean:.4f}, std={std:.4f}")
        with open(run_dir / "norm_stats.json", "w") as f:
            json.dump({"mean": float(transform.mean), "std": float(transform.std)}, f, indent=2)

    # ---- Model ----
    model = create_model(config).to(device)
    num_params = model.count_parameters()
    print(f"Model: {config.model.get('arch', '?')} | Params: {num_params:,}")

    criterion, pos_weight = build_criterion(config, train_loader.dataset, device)
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer)

    use_amp = config.training.get("amp", True) and device.type == "cuda"
    scaler = make_grad_scaler(use_amp)

    rc = run_training(
        config, args, model, train_loader, val_loader, test_loader,
        criterion, optimizer, scheduler, scaler, use_amp, transform,
        logger, run_dir, device, num_params,
    )
    logger.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())