#!/usr/bin/env python3
"""
Build ALL wake-word architectures: train each, export ONNX + frontend into
app/models/wake_word/<key>/, and print a comparison table.

Usage:
    python scripts/build_wake_models.py
    python scripts/build_wake_models.py --arches dscnn,tcresnet
    python scripts/build_wake_models.py --epochs 40

Each architecture is trained with the SAME data/config (only model.arch differs),
then exported so the demo can pick it with  python app.py --model <NAME>.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
APP_MODELS = REPO / "app" / "models" / "wake_word"

# arch (model.arch) -> demo directory / display name
ARCHS = [
    ("dscnn", "dscnn", "DS-CNN"),
    ("tcresnet", "tcresnet", "TC-ResNet"),
    ("matchboxnet2d", "matchboxnet", "MatchboxNet"),
    ("vgg_small", "vgg", "VGG"),
]


def run(cmd):
    print("\n$", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.call(cmd)


def latest_run(runs_subdir: Path) -> Path:
    cands = sorted(runs_subdir.glob("*/*/"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(f"No runs found under {runs_subdir}")
    return cands[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wake_word_train.yaml")
    ap.add_argument("--arches", default=",".join(a[0] for a in ARCHS))
    ap.add_argument("--epochs", type=int, default=None, help="override training.epochs")
    args = ap.parse_args()

    wanted = set(a.strip() for a in args.arches.split(",") if a.strip())
    cfg_path = args.config

    # optional epoch override -> temp config
    if args.epochs:
        import yaml
        with open(REPO / cfg_path) as f:
            cfg = yaml.safe_load(f)
        cfg["training"]["epochs"] = args.epochs
        tmp = REPO / "configs" / "_build_tmp.yaml"
        with open(tmp, "w") as f:
            yaml.safe_dump(cfg, f)
        cfg_path = "configs/_build_tmp.yaml"

    py = sys.executable
    results = []
    for arch, key, name in ARCHS:
        if arch not in wanted:
            continue
        out_runs = f"runs/wake_{key}"
        print(f"\n{'='*70}\n=== {name} ({arch}) ===\n{'='*70}", flush=True)

        rc = run([py, "train.py", "--config", cfg_path, "--arch", arch,
                  "--output_dir", out_runs])
        if rc != 0:
            print(f"!! training failed for {arch} (rc={rc})")
            continue

        run_dir = latest_run(REPO / out_runs)
        ckpt = run_dir / "best_model.pt"
        out_dir = APP_MODELS / key
        out_dir.mkdir(parents=True, exist_ok=True)

        run([py, "scripts/export_model.py", "--config", cfg_path, "--arch", arch,
             "--model", str(ckpt), "--out_dir", str(out_dir)])
        run([py, "scripts/export_frontend.py", "--config", cfg_path, "--out_dir", str(out_dir)])

        mfile = run_dir / "metrics.json"
        if mfile.exists():
            with open(mfile) as f:
                d = json.load(f)
            results.append((name, arch, key, d))

    # comparison table
    print("\n" + "=" * 90)
    print(f"{'model':<12}{'params':>10}{'acc':>8}{'eer':>8}{'roc_auc':>9}{'far@0.5':>9}{'epoch_time':>11}")
    print("-" * 90)
    for name, arch, key, d in results:
        tm = d.get("test_metrics", {})
        print(f"{name:<12}{d.get('num_params', 0):>10,}"
              f"{tm.get('accuracy', float('nan')):>8.4f}"
              f"{tm.get('eer', float('nan')):>8.4f}"
              f"{tm.get('roc_auc', float('nan')):>9.4f}"
              f"{tm.get('far', float('nan')):>9.4f}"
              f"{tm.get('epoch_time', float('nan')):>11.2f}")
    print("=" * 90)

    # write a summary for the app/README
    summary = [{"name": n, "arch": a, "dir": k,
                "num_params": d.get("num_params"),
                "test_metrics": d.get("test_metrics")} for n, a, k, d in results]
    with open(APP_MODELS / "models_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {APP_MODELS / 'models_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
