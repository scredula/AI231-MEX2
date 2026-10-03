#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-shot validation runner for the Raspberry Pi.
#   - installs the notebook/runtime deps into the app's ./venv
#   - executes a validation notebook top-to-bottom (writes outputs + JSON)
#   - prints where the report landed
#
# Usage (from anywhere):
#   ./rpi_validation/run_on_pi.sh                       # keyword intent notebook
#   ./rpi_validation/run_on_pi.sh command_classifier_validation.ipynb
#   ./rpi_validation/run_on_pi.sh keyword_intent_validation.ipynb
# ---------------------------------------------------------------------------
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../rpi_validation
APP="$(cd "$DIR/.." && pwd)"                          # app folder (has setup.py/app.py)
PY="$APP/venv/bin/python"

if [ ! -x "$PY" ]; then
  echo "[!] venv not found at $PY"
  echo "    Run the initial setup first:"
  echo "      cd \"$APP\" && python3 setup.py --recreate"
  exit 1
fi

NB="${1:-keyword_intent_validation.ipynb}"
NBPATH="$DIR/$NB"
if [ ! -f "$NBPATH" ]; then
  echo "[!] notebook not found: $NBPATH"
  echo "    Available:"; ls -1 "$DIR"/*.ipynb
  exit 1
fi

echo "[*] Python : $PY"
echo "[*] Install notebook deps (jupyter / nbconvert / ipykernel) ..."
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet jupyter nbconvert ipykernel
# optional extras (plots / CSVs / RAM) -- ignore failures
"$PY" -m pip install --quiet matplotlib pandas psutil || true

echo "[*] Executing $NB (this runs the whole notebook headless) ..."
cd "$DIR"
"$PY" -m jupyter nbconvert --to notebook --execute --inplace "$NB"

echo
echo "[*] Done."
echo "    Notebook (with outputs): $NBPATH"
if [ -d "$DIR/results" ]; then
  echo "    Reports in            : $DIR/results"
  ls -1 "$DIR/results" || true
fi
echo
echo "Tip: add labeled clips to get accuracy/FAR:"
echo "    $DIR/test_data/<CLASS>/*.wav   and   $DIR/test_data/_negative/*.wav"
