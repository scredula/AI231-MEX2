# MEX2 — Wake-Word + Voice Command (AI231)

Custom on-device voice assistant pipeline for a **Raspberry Pi 5**:

1. **Always-on wake-word detector** for `"Hey Mason"` (binary CNN).
2. **Command classifier** over the fixed MEX2 intent + slot set (multi-class CNN).

Both models use the same **log-mel → MatchboxNet (CNN)** front-end and are exported
to **ONNX** for fast CPU inference via ONNX Runtime on the Pi 5.

---

## Repository layout

```
.
├── train.py                     # single-command training entry point
├── validate.py                  # single-command validation entry point
├── configs/
│   ├── wake_word_train.yaml
│   ├── wake_word_val.yaml
│   ├── command_train.yaml       # (used once the MEX2 dataset is released)
│   └── command_val.yaml
├── src/
│   ├── data/                    # datasets, log-mel frontend, augmentations
│   ├── models/                  # MatchboxNet definition
│   ├── training/                # train/val loops + metrics (EER, FAR/FRR, CM)
│   ├── export/                  # ONNX export + latency benchmark
│   └── utils/                   # config, seeding, logging, plotting
├── notebooks/
│   └── 02_wake_word_training.ipynb   # full A100 training run with plots/metrics
├── scripts/
│   ├── download_negatives.py    # fetch LibriSpeech / MUSAN / ESC-50 / silence / MEX2
│   ├── download_negatives.sh    # wrapper
│   └── setup_env.sh             # create venv + install requirements
├── models/                      # trained artifacts (best_model.pt / .onnx)
└── requirements.txt
```

---

## Quick start

### 1. Environment

```bash
bash scripts/setup_env.sh --cuda cu121     # A100 (CUDA 12.1)
# or:  bash scripts/setup_env.sh           # CPU-only
source .venv/bin/activate
```

### 2. Data

Wake-word **positives** live in `data/wake_word/{train,val,test}`.
Wake-word **negatives** are downloaded with:

```bash
# Speech (LibriSpeech) + noise (ESC-50) + synthetic room tone
bash scripts/download_negatives.sh --sources speech,esc50,silence

# Full set including MUSAN (large) and MEX2 commands (when available)
bash scripts/download_negatives.sh --all --mex2_path /path/to/MEX2/Data
```

This writes `data/negative/<source>/*.wav` + `data/negative/metadata.csv`.

### 3. Train

```bash
python train.py --config configs/wake_word_train.yaml
```

Artifacts (plots, `metrics.json`, `best_model.pt`, `best_model.onnx`) are written to
`runs/wake_word/<experiment>/<timestamp>/`.

### 4. Validate (professor's test)

```bash
python validate.py --config configs/wake_word_val.yaml \
                   --model runs/wake_word/<exp>/<ts>/best_model.onnx \
                   --split test --benchmark
```

Prints accuracy, precision/recall/F1, EER, FAR/FRR at each threshold, ROC-AUC,
confusion matrix, and (optionally) ONNX latency, and saves plots + `metrics.json`.

---

## Model — MatchboxNet

Lightweight depthwise-separable residual CNN (~75K params for wake-word):

```
input (1, 64, T)  →  stem Conv1d → [Matchbox blocks] → head → logit(s)
```

- `n_mels = 64`, `hop = 160` (10 ms), 25 ms window, 16 kHz, 1 s clips.
- Train: AdamW + CosineAnnealingWarmRestarts, AMP, grad-clip, SpecAugment + audio aug.
- Loss: BCEWithLogits (wake-word, class-balanced) / CrossEntropy (commands).

## Metrics reported

| Kind | Metrics |
|------|---------|
| Per-epoch | train/val loss, train/val accuracy (0–100 %), F1, val EER, LR, epoch time |
| Holdout | accuracy, precision, recall, F1, EER, FAR, FRR, ROC-AUC, PR-AUC |
| Thresholds | accuracy/precision/recall/F1/FAR/FRR at 0.1 … 0.9 |
| Plots | loss curve, accuracy curve, confusion matrix, ROC, DET, metrics-vs-threshold |
| Deployment | ONNX size/ops, CPU latency (mean/p50/p95/p99) |

---

## Demo app (`app/`)

A **self-contained** Tkinter app runs the wake-word model with ONNX Runtime,
shows a live probability bar, and on trigger displays the top-10 command
predictions and performs the action (music / weather / JPG popup). It bundles the
trained model, the exact NumPy log-mel frontend, assets, and sample audio, so the
folder can be downloaded and run on its own.

```bash
cd app
python setup.py        # creates ./venv + installs requirements.txt
./venv/bin/python app.py --list-models
./venv/bin/python app.py --model MatchboxNet   # or DS-CNN | TC-ResNet | VGG
```

Four wake-word architectures are trained and shipped as ONNX
(`DS-CNN`, `TC-ResNet`, `MatchboxNet` (default, best), `VGG`); see `app/README.md`
for the comparison table. Rebuild them with `python scripts/build_wake_models.py`.

**Test without a mic:** **Simulate Wake** (feeds `app/samples/*.wav`) and
**Simulate Command** (type e.g. `play music`, `weather`, `color red`).
See `app/README.md`. `app/selftest.py` verifies the NumPy frontend matches
torchaudio exactly and checks detection accuracy headlessly.

> The command classifier is a **placeholder** until the MEX2 dataset/model is
> trained; drop `models/command/best_model.onnx` + `labels.json` into
> `app/models/command/` and it switches to real inference automatically.

## Deployment on Raspberry Pi 5

- Convert/train on the A100, ship `best_model.onnx` (+ `norm_stats.json`).
- On the Pi: ONNX Runtime, sliding-window wake-word detection every ~100–200 ms;
  on trigger, run the command classifier and dispatch the action.
- The wake-word model is intentionally tiny to avoid thermal throttling during
  continuous inference.

> **Demo UI** (threshold bar → top-10 commands → music/weather/popup actions) is
> planned as a separate optional component, built after both models are verified.
