# Command Classifier — Raspberry Pi Validation

Two ways to validate the 31-class command ("keyword intent") classifier on the
target device, both using the **same production code path** as the app
(`commands.CommandClassifier` + `features.MetadataFrontend`):

| File | Outputs |
|---|---|
| `keyword_intent_validation.ipynb` | **keyword intent accuracy**, **false accept rate (FAR/FRR)**, **latency**, **runtime/thermal** |
| `command_classifier_validation.ipynb` | accuracy + plots + latency + runtime (dataset explorer) |
| `bench_latency.py` | headless **latency/runtime** only (no Jupyter needed) |
| `validate_classifier.py` | headless **accuracy (top-1/3/5, per-class P/R/F1, FAR) + latency** for **any** classifier |

## Run

**One command on the Pi (installs deps + runs a notebook headless):**

```bash
./rpi_validation/run_on_pi.sh                        # keyword intent notebook
./rpi_validation/run_on_pi.sh command_classifier_validation.ipynb
```

**Quick latency/runtime metrics (no Jupyter needed):**

```bash
cd ..                                   # app folder
./venv/bin/python rpi_validation/bench_latency.py
./venv/bin/python rpi_validation/bench_latency.py --runs 200 --seconds 60
./venv/bin/python rpi_validation/bench_latency.py --data my_wavs/ --json out.json
```

Prints frontend / ONNX / end-to-end latency (mean, median, p95, p99), real-time
factor, throughput, a thread sweep, and (with `--seconds`) a sustained
thermal-drift run. Report saved to `rpi_validation/results/latency.json`.

**Validate any command classifier (accuracy + latency, no Jupyter):**

```bash
cd ..                                   # app folder
./venv/bin/python rpi_validation/validate_classifier.py                  # config default
./venv/bin/python rpi_validation/validate_classifier.py --classifier MEX2-trained
./venv/bin/python rpi_validation/validate_classifier.py --all
./venv/bin/python rpi_validation/validate_classifier.py --data my_wavs --manifest my.csv
```

Uses the app's production path (`commands.CommandClassifier`), so it works with
**both** classifier formats (the bundled `command_metadata.json` model and the
pipeline-trained `frontend.json` model). It prints top-1/3/5, macro-F1,
per-class precision/recall/F1, the false-accept rate on negatives, coverage, and
end-to-end latency (mean/p50/p95/p99) + CPU temperature, and saves JSON to
`rpi_validation/results/validate_<model>.json`.

**Full metrics notebook (adds accuracy, plots):**

```bash
pip install -r ../requirements.txt      # numpy, onnxruntime, soundfile, scipy, pyyaml
pip install matplotlib pandas psutil    # optional: plots / CSVs / RAM
pip install jupyter nbconvert ipykernel # to run the notebook

jupyter notebook command_classifier_validation.ipynb
# or, headless:
jupyter nbconvert --execute --to notebook --inplace command_classifier_validation.ipynb
```

Launch Jupyter **from this folder** so relative paths resolve. The notebook
auto-locates the app directory by walking up the tree looking for `commands.py`
plus `models/command_classifier/`.

## What it reports

* **Environment** — OS, CPU, RAM, Raspberry Pi model, CPU temp, ONNX Runtime + providers.
* **Model** — classes, UNKNOWN threshold, frontend params, shapes, load time.
* **Latency** — frontend / ONNX / end-to-end ms (mean, median, p95, p99),
  real-time factor, throughput, plus a 1/2/4-thread sweep.
* **Accuracy** (needs labeled data) — top-1/3/5, coverage, per-class
  precision/recall/F1, confusion matrix, UNKNOWN-threshold sweep, macro F1.
* **Runtime** — sustained run: latency drift and CPU temperature (thermal throttling).
* **Report** — `results/command_classifier_validation.json` (+ optional CSVs / PNGs).

## Labeled test set (for the accuracy section)

Add clips in any of these layouts (auto-detected; edit `TEST_DATA_DIR` /
`MANIFEST_CSV` in the config cell to override):

```
test_data/<CLASS>/<any>.wav        # folder per class  (recommended)
test_data/manifest.csv             # columns: path,label
```

`<CLASS>` must match the model's `class_to_idx` (run the notebook to print the
31 class names). Without a dataset, the accuracy cells are skipped and all
latency/runtime/model metrics still run.
