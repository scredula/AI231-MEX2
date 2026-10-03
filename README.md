# MEX2 — Wake-Word + Voice-Command Assistant (FINAL)

A self-contained, ready-to-run release of the MEX2 (AI231) on-device voice
assistant for the **Raspberry Pi 5**, plus the full training/validation pipeline
used to produce the models.

- **Wake word:** `"Hey Mason"` — 4 selectable CNN architectures.
- **Command classifier:** 2 selectable 31-class intent models.
- **Everything ships pre-trained** — no training needed to run the app.

---

## Architecture & Data Flow Overview

```mermaid
flowchart TD
    %% ============================================================
    %% INPUT / FRONTEND
    %% ============================================================
    subgraph Input["🎤 Voice Input & Frontend"]
        A["Microphone\n16 kHz mono"] --> B["VAD / Ring Buffer\n(voice activity detection)"]
        B --> C["Log-Mel Spectrogram\n64 mel bins, 400/160 hop/win"]
        C --> D["Normalize (mean/std)\nor Clamp (top_db=80)"]
    end

    %% ============================================================
    %% WAKE WORD STAGE
    %% ============================================================
    subgraph Wake["🔊 Wake Word Detection\n\"Hey Mason\" — Binary Classification"]
        D --> E["Selectable CNN Backbone\n(shared log-mel input)"]
        
        subgraph WW_Models["Wake Word Architectures (4)"]
            E1["DS-CNN\nDepthwise-Separable\n~64k params"]
            E2["MatchboxNet (default)\nResidual Depthwise-Separable\n~180k params"]
            E3["TC-ResNet\nTemporal Conv (k×1) spanning mels\n~210k params"]
            E4["VGG-Small\nVGG-style 3×3 conv blocks\n~1.2M params"]
        end
        
        E --> E1 & E2 & E3 & E4
        E1 & E2 & E3 & E4 --> F["Sigmoid → P(wake)\nThreshold ≈ 0.5"]
        F -->|"P(wake) > thr"| G["✅ WAKE DETECTED\n→ Activate Classifier"]
        F -.->|"P(wake) ≤ thr"| H["⏳ Keep Listening\n(return to VAD)"]
    end

    %% ============================================================
    %% CLASSIFIER STAGE
    %% ============================================================
    subgraph Classifier["🧠 Command Classifier\n31 Intents + UNKNOWN"]
        G --> I["Log-Mel Frontend\n(two variants)"]
        
        subgraph FE_Variants["Frontend Variants"]
            I1["MetadataFrontend\n(bundled MEX2-31class)\nn_fft=512, 150 frames\nlog(fb·power/n_fft)\n→ (1,1,64,150)"]
            I2["LogMelFrontend\n(trained MEX2-trained)\nn_fft=400, 201 frames\n10·log10 + top_db + mean/std\n→ (1,1,64,201)"]
        end
        
        I --> I1 & I2
        I1 & I2 --> J["ONNX Runtime Inference\n(intra_op_threads=2)"]
        
        subgraph CLF_Models["Classifier Architectures (2)"]
            J1["MEX2-31class (bundled)\nMatchboxNet2D, 32 classes\n(incl. OUT_OF_SCOPE)\n27 MB ONNX"]
            J2["MEX2-trained (pipeline)\nMatchboxNet2D, 31 classes\n(OUT_OF_SCOPE dropped)\n2.3 MB ONNX"]
        end
        
        J --> J1 & J2
        J1 & J2 --> K["Softmax + Temperature\n→ probs[31]"]
        K --> L{"max(probs) >\nunknown_thr?"}
        L -->|Yes| M["✅ Intent = argmax\n(confidence = max)"]
        L -->|No| N["❓ UNKNOWN\n(below threshold)"]
    end

    %% ============================================================
    %% COMMAND EXECUTION
    %% ============================================================
    subgraph Action["⚡ Command Execution\n(app/app.py)"]
        M --> O["Dispatch → Handler"]
        O --> P1["PLAY_MUSIC\n🎵 Show album art / play"]
        O --> P2["SET_ALARM\n⏰ Set alarm UI"]
        O --> P3["GET_WEATHER\n☁️ Fetch & display"]
        O --> P4["...28 more intents"]
        O --> P5["PLACEHOLDER\n🖼️ Pop-up image\n🔊 Play TTS\n💡 Trigger GPIO"]
    end

    %% ============================================================
    %% TRAINING PIPELINE (separate)
    %% ============================================================
    subgraph Training["🏋️ Training Pipeline (training/)"]
        T1["Raw Audio\n(wav, 16 kHz)"] --> T2["LogMelSpectrogram\n(n_mels=64, n_fft=400)"]
        T2 --> T3["Augmentations\n(SpecAug, noise, shift)"]
        T3 --> T4["MatchboxNet2D\n(num_classes=1 or 32)"]
        T4 --> T5["Loss: BCEWithLogitsLoss\n(wake) / CrossEntropy (cmd)"]
        T5 --> T6["Train / Val / Test\n(speaker-disjoint splits)"]
        T6 --> T7["Export ONNX\n+ norm_stats.json"]
        T7 --> T8["runs/<exp>/<ts>/\nbest_model.onnx/.pt\nmetrics.json, plots"]
    end

    %% Styling
    classDef input fill:#e3f2fd,stroke:#1976d2,stroke-width:2px;
    classDef wake fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    classDef clf fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px;
    classDef action fill:#e8f5e9,stroke:#388e3c,stroke-width:2px;
    classDef train fill:#fce4ec,stroke:#c2185b,stroke-width:2px;
    classDef model fill:#fafafa,stroke:#616161,stroke-width:1px,stroke-dasharray: 5 5;
    
    class A,B,C,D input;
    class E,F,G,H wake;
    class I,J,K,L,M,N clf;
    class O,P1,P2,P3,P4,P5 action;
    class T1,T2,T3,T4,T5,T6,T7,T8 train;
    class E1,E2,E3,E4,I1,I2,J1,J2 model;
```

### Quick Data-Flow Summary (Voice → Intent)

| Stage | What Happens | Key Artifacts |
|-------|--------------|---------------|
| **1. Capture** | Mic → 16 kHz mono → VAD ring buffer | `audio.wav` chunks (1 s) |
| **2. Frontend** | STFT → mel filterbank (64) → log-power → normalize/clamp | `LogMelSpectrogram` / `MetadataFrontend` |
| **3. Wake Word** | 2-D CNN (MatchboxNet default) → sigmoid → `P(wake)` | `models/wake_word/matchboxnet/best_model.onnx` |
| **4. Gate** | If `P(wake) > 0.5` → **activate classifier**; else loop | — |
| **5. Classifier** | Same mel → different frontend → ONNX → softmax(31) | `models/command_classifier/…/best_model.onnx` |
| **6. Threshold** | `max(prob) > unknown_thr` → **Intent**; else **UNKNOWN** | `unknown_threshold` in metadata/frontend.json |
| **7. Action** | `app.py` dispatches to handler → UI / TTS / GPIO / API | `commands.py` handlers (placeholder images, etc.) |

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

Validate a **specific** command classifier (accuracy + latency + FAR, no Jupyter):

```bash
cd app
./venv/bin/python rpi_validation/validate_classifier.py --classifier MEX2-trained
./venv/bin/python rpi_validation/validate_classifier.py --all
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
