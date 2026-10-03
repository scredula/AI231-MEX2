#!/usr/bin/env bash
# Create a Python environment and install project dependencies.
#
# Usage:
#   bash scripts/setup_env.sh                 # CPU torch (safe default)
#   bash scripts/setup_env.sh --cuda cu121    # CUDA 12.1 torch (for A100 training)
#
# Creates a venv at .venv (repo root) and installs requirements.txt.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CUDA_CHANNEL=""
if [ "${1:-}" = "--cuda" ] && [ -n "${2:-}" ]; then
  CUDA_CHANNEL="$2"
fi

echo "==> Creating virtual environment at .venv"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Upgrading pip"
pip install --upgrade pip wheel setuptools

echo "==> Installing PyTorch + torchaudio"
# Default PyPI wheels bundle CUDA and support recent Python versions.
# If a specific CUDA channel was requested, try it first and fall back to PyPI.
if [ -n "${CUDA_CHANNEL}" ]; then
  echo "    Trying CUDA channel: ${CUDA_CHANNEL}"
  if ! pip install torch torchaudio --index-url "https://download.pytorch.org/whl/${CUDA_CHANNEL}"; then
    echo "    CUDA channel install failed; falling back to default PyPI wheels."
    pip install torch torchaudio
  fi
else
  pip install torch torchaudio
fi

echo "==> Installing requirements.txt"
pip install -r requirements.txt

echo "==> Registering Jupyter kernel"
python -m ipykernel install --user --name mex2-wakeword --display-name "MEX2 WakeWord"

echo
echo "Done. Activate with:  source .venv/bin/activate"
echo "Run training with:    python train.py --config configs/wake_word_train.yaml"
