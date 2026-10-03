# MEX2 — Wake-Word + Voice-Command Assistant (FINAL)

A self-contained, ready-to-run release of the MEX2 (AI231) on-device voice
assistant for the **Raspberry Pi 5**, plus the full training/validation pipeline
used to produce the models.

- **Wake word:** `"Hey Mason"` — 4 selectable CNN architectures.
- **Command classifier:** 2 selectable 31-class intent models.
- **Everything ships pre-trained** — no training needed to run the app.

```
FINAL/
├── training/   # training + validation pipeline (train.py, validate.py, src/, configs/, scripts/)
├── app/        # Raspberry Pi 5 demo app (selectable models) + validation notebooks
├── runs/       # training logs (TensorBoard + CSV + plots) and final checkpoints
└── data/       # dataset placeholders (upload your own data; paths preserved)
```

---

## 1. Run the app out of the box (Raspberry Pi 5 / Linux / macOS)

```bash
cd app
python3 setup.py            # creates ./venv and installs dependencies (once)
./run.sh                    # launches the demo GUI
```

Pick a specific model (wake-word + classifier are independent):

```bash
./run.sh --list-models                         # show every available model
./run.sh --model MatchboxNet                   # DS-CNN | TC-ResNet | MatchboxNet | VGG
./run.sh --classifier MEX2-31class             # MEX2-31class | MEX2-trained
./run.sh --model VGG --classifier MEX2-trained
```

No microphone? Use the GUI's **Simulate Wake** / **Simulate Command** buttons.

### Bundled, pre-trained models

| Kind | Name | Path | Notes |
|---|---|---|---|
| Wake word | DS-CNN | `app/models/wake_word/dscnn` | |
| Wake word | TC-ResNet | `app/models/wake_word/tcresnet` | |
| Wake word | **MatchboxNet** (default) | `app/models/wake_word/matchboxnet` | best wake-word model |
| Wake word | VGG | `app/models/wake_word/vgg` | |
| Classifier | **MEX2-31class** (default) | `app/models/command_classifier` | validated on the Pi 5 |
| Classifier | MEX2-trained | `app/models/command_classifier/trained` | produced by `training/` |

Headless self-test (no GUI/mic needed):

```bash
cd app && ./venv/bin/python selftest.py
```

---

## 2. Validate on the Raspberry Pi 5

```bash
cd app
./rpi_validation/run_on_pi.sh                              # keyword-intent accuracy / FAR / latency / thermal
./rpi_validation/run_on_pi.sh command_classifier_validation.ipynb
```

These run the **same production code path as the app** and report keyword
accuracy, false-accept rate, per-class precision/recall/F1, latency
(mean/p50/p95/p99), real-time factor, throughput, and CPU temperature / thermal
drift. Results land in `app/rpi_validation/results/`.
The bundled `app/rpi_validation/test_data/holdout/` set makes the accuracy
section run immediately; add your own clips under
`test_data/<CLASS>/*.wav` (+ `test_data/_negative/*.wav`) to extend it.

Quick latency-only run (no Jupyter):

```bash
cd app && ./venv/bin/python rpi_validation/bench_latency.py --runs 200
```

---

## 3. One-line training & validation

From `training/` (after uploading the datasets — see `data/README.md`):

```bash
cd training
bash scripts/setup_env.sh                          # create .venv + install requirements
source .venv/bin/activate

# ---- training (one line) ----
python train.py --config configs/wake_word_train.yaml      # wake word ("Hey Mason")
python train.py --config configs/command_train.yaml        # command classifier

# ---- validation (one line) ----
python validate.py --config configs/wake_word_val.yaml \
    --model runs/wake_word/wake_word_hey_mason/<ts>/best_model.onnx --split test --benchmark
python validate.py --config configs/command_val.yaml \
    --model runs/command/command_mex2/<ts>/best_model.onnx --split test
```

Training writes plots, `metrics.json`, `best_model.pt`, `best_model.onnx`,
`norm_stats.json` and TensorBoard/CSV logs to `training/runs/<experiment>/<name>/<timestamp>/`.
The **final** such publications are mirrored under `runs/` in this folder.

---

## 4. Data

No datasets are committed (they are large). Keep the directory layout and drop
your files in — the configs already point at these paths. See
**`data/README.md`** for exactly what goes where and the download/prepare
commands.

## 5. License / provenance

Built for the AI231 MEX2 project. See `training/README.md` and
`app/README.md` for component-level details.
