#!/usr/bin/env bash
# Launch the Hey Mason demo on Linux / macOS / Raspberry Pi.
# First run:  python3 setup.py   (creates ./venv and installs requirements)
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -x "$DIR/venv/bin/python" ]; then
  exec "$DIR/venv/bin/python" "$DIR/app.py" "$@"
else
  echo "[ERROR] Virtual environment not found. Run: python3 setup.py"
  exit 1
fi
