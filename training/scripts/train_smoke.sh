#!/usr/bin/env bash
# Fast end-to-end smoke test: 2 epochs on the current data.
# Verifies data loading, model, training loop, metrics, plots and ONNX export.
#
# Usage:  bash scripts/train_smoke.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# shellcheck disable=SC1091
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi

python - <<'PY'
import yaml, copy
from pathlib import Path
from src.utils.config import load_config, save_config

cfg = load_config("configs/wake_word_train.yaml")
cfg.training.epochs = 2
cfg.experiment.name = "wake_word_smoke"
cfg.experiment.device = "cuda" 
cfg.dataloader.batch_size = 64
cfg.dataloader.num_workers = 4
save_config(cfg, "configs/_smoke.yaml")
print("Wrote configs/_smoke.yaml")
PY

python train.py --config configs/_smoke.yaml --output_dir runs/_smoke
echo "Smoke test complete."
