#!/usr/bin/env bash
# Thin wrapper around download_negatives.py
#
# Examples:
#   bash scripts/download_negatives.sh                      # all sources
#   bash scripts/download_negatives.sh --sources silence    # no download (fast smoke test)
#   bash scripts/download_negatives.sh --sources speech,noise
#   bash scripts/download_negatives.sh --sources mex2 --mex2_path /path/to/MEX2/Data
#
# Requires: python3 + ffmpeg (or sox) on PATH.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

if [ "$#" -eq 0 ]; then
  python3 scripts/download_negatives.py --all
else
  python3 scripts/download_negatives.py "$@"
fi
