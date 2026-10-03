# Hey Mason — Demo App (self-contained)

Wake-word ("Hey Mason") + command demo you can download and run as a folder.
Wake-word detection runs on **ONNX Runtime**; the log-mel frontend is pure NumPy
(no PyTorch needed).

```
┌───────────────────────────────┬───────────────────────────────┐
│  Wake Word: "Hey Mason"       │  Top-10 Commands              │
│  [===== prob bar =====|    ]  │  62.1%  PLAY_MUSIC            │
│  trigger threshold ──────     │  11.4%  NEXT                  │
│  Status: Listening…           │   …                           │
│  [Start Mic][Simulate Wake]   │  Simulate command: [____] Run │
│                               │  Action log                   │
└───────────────────────────────┴───────────────────────────────┘
```

## Quick start

```bash
python setup.py                      # creates ./venv and installs requirements.txt
./venv/bin/python app.py             # launch with the default model (Tkinter window)
./venv/bin/python app.py --ui web    # browser smart-home simulator (see below)
./venv/bin/python app.py --model DS-CNN      # pick a wake-word model
./venv/bin/python app.py --list-models       # show available models
```
On Windows use `py setup.py` then `venv\Scripts\python app.py --model DS-CNN`.

On Raspberry Pi 5 also install the system libs (Windows needs none):
```bash
sudo apt-get install -y libportaudio2 libsdl2-mixer-2.0-0 python3-tk
```

## Wake-word models
Four architectures are trained on the same data and shipped as ONNX. Choose at launch:

| `--model` | Architecture | Params | Notes |
|-----------|--------------|-------:|-------|
| `DS-CNN` | depthwise-separable CNN | ~20k | lightest, best for Pi thermals |
| `TC-ResNet` | temporal-conv ResNet | ~63k | fastest, (k,1) kernels span all mels |
| `MatchboxNet` | residual depthwise-separable | ~560k | strongest robustness (default) |
| `VGG` | small VGG-style CNN | ~1.2M | accuracy reference, heaviest |

Each lives in `models/wake_word/<dir>/` with `best_model.onnx` + the frontend files.
`models/wake_word/models_summary.json` holds the trained metrics for comparison.

### Measured holdout results (same data, single A100)
| model | params | accuracy | EER | ROC-AUC | FAR @0.5 |
|-------|-------:|---------:|----:|--------:|---------:|
| DS-CNN | 19,969 | 0.9200 | 0.0117 | 0.9992 | 0.0933 |
| TC-ResNet | 62,769 | 0.9514 | 0.0100 | 0.9996 | 0.0567 |
| **MatchboxNet** | 559,041 | **0.9914** | **0.0033** | 0.9996 | **0.0067** |
| VGG | 1,172,897 | 0.9586 | 0.0100 | 0.9988 | 0.0483 |

Rebuild all models with: `python scripts/build_wake_models.py` (from the repo root).

## Folder layout
```
app/
├── setup.py            # creates ./venv + installs requirements.txt (+ optional assets/selftest)
├── requirements.txt
├── run.sh              # optional launcher (passes args: ./run.sh --model DS-CNN)
├── config.yaml         # paths (relative to this folder), thresholds, 32 intents
├── app.py              # Tkinter GUI + orchestration (--ui tk) / switches to web UI
├── webui.py            # browser UI server: stdlib HTTP + Server-Sent Events (--ui web)
├── web/                # the dashboard: index.html + style.css + app.js (no build step)
├── device.py           # simulated smart-home state (bulb, thermostat, timer, alarms…)
├── weather.py          # live Cebu weather via Open-Meteo (key-less, cached, offline-safe)
├── features.py         # exact log-mel frontend (NumPy; matches torchaudio training)
├── wakeword.py         # ONNX wake-word detector + sliding-window debounce
├── commands.py         # command classifier (real ONNX OR placeholder) + action mapping
├── audio_io.py         # microphone capture + WAV feeder (test without a mic)
├── actions.py          # music (pygame), weather clip, JPG popups
├── generate_assets.py  # pre-renders intent JPGs + weather.mp3
├── selftest.py         # headless verification (frontend equivalence + detection)
├── models/wake_word/   # per-architecture: dscnn/ , tcresnet/ , matchboxnet/ , vgg/
│                       #   each = best_model.onnx + frontend.json + mel_fbank.npy
│                       #          + hann_window.npy + norm_stats.json
├── models/command_classifier/  # command_classifier.onnx + command_metadata.json
│                       #   (metadata carries frontend + class_to_idx + unknown_threshold)
├── assets/jpgs/        # 32 pre-rendered intent JPGs (popups)
├── assets/audio/       # weather.mp3 + beep.wav (wake-word "listening" cue)
├── samples/            # optional real "Hey Mason" WAVs (reference/test clips)
└── music/              # bundled synthesized demo tracks (demo_*.wav) + your own songs
```

> The three `music/demo_*.wav` files are short **original synthesized** clips so the
> music commands work immediately. Replace/add your own `.mp3`/`.wav` songs for the demo.

## Testing without a microphone
- **Simulate Wake** — injects a wake-word trigger (plays the cue) and then
  listens for a command, exactly like saying "Hey Mason". The microphone is
  auto-started if needed; say your command during the capture window.
- **Simulate Command** — type e.g. `play music`, `volume up`, `turn on the lights`,
  `weather`, `set brightness to 100`, `color red`, `timer 30 seconds` and press **Run**.

## Actions
| Intent group | Action |
|--------------|--------|
| `PLAY_MUSIC`, `NEXT`, `PAUSE`, `STOP`, `VOLUME_UP`, `VOLUME_DOWN` | pygame music player (`music/`) |
| `WEATHER` | plays `assets/audio/weather.mp3` |
| the other 24 intents | shows `assets/jpgs/<INTENT>.jpg` popup |

## Browser UI — smart-home simulator (`--ui web`)

A richer, animated dashboard that **simulates** each command instead of flashing a
JPG. It is served by `webui.py` (Python **stdlib** HTTP + Server-Sent Events) and
reuses the *same* `config.yaml` models, so a spoken command and a clicked/typed one
land in the same simulated device (`device.py`).

```bash
./venv/bin/python app.py --ui web                 # open http://<host>:8000/
./venv/bin/python app.py --ui web --classifier MEX2-trained
./venv/bin/python webui.py --port 8080 --open     # or run it directly
./venv/bin/python webui.py --no-mic               # simulation only
```

| Panel | Intent(s) | What you see |
|-------|-----------|--------------|
| 💡 Lights | `LIGHT_ON/OFF`, `BRIGHTNESS_*`, `COLOR_*` | the bulb lights up, dims to 20/60/100 %, recolours red/green/blue |
| ☁️ Weather | `WEATHER` | **live Cebu City conditions** (Open-Meteo, no key) |
| 🌡️ Thermostat | `TEMPERATURE_18/22/26` | dial sweeps to the new setpoint |
| ⏱️ Timer | `TIMER_10s/30s/1m` | countdown ring (server-side countdown) |
| ⏰ Alarms / reminders | `ALARM_*`, `CREATE_REMINDER_*`, `LIST_REMINDERS` | chips + list |
| 🎵 Music | `PLAY_MUSIC/NEXT/PAUSE/STOP/VOLUME_*` | now-playing card + volume meter |
| 🕐 Assistant | `TIME`, `CALL`, `MESSAGE` | clock highlight, call state, message bubble |
| Wake word | the detector | live probability meter with the trigger threshold marker |

* **Weather** is fetched **server-side** in `weather.py` from
  [Open-Meteo](https://open-meteo.com) — free, key-less, JSON — cached for
  `weather.ttl_s` and falling back to the last reading (marked `stale`) when
  offline. Change `weather.place` / `lat` / `lon` in `config.yaml` (default
  **Cebu City, Philippines**).
* **Typed text** uses the keyword fast-path (`commands.placeholder_from_text`);
  clicking a button or saying “Hey Mason” exercises the **real ONNX model**.
* Endpoints: `GET /` · `/events` (SSE) · `/state` · `/weather` · `/models`,
  `POST /command` · `/wake` · `/mic` · `/reset`. The page is plain HTML/CSS/JS with
  no build step and no CDN; on the Pi 5 point Chromium at the server for kiosk mode.

## Command classifier status
The real 31-class command model is bundled at `models/command_classifier/`:
`command_classifier.onnx` (input `logmel [1,1,64,150]`, output `[1,31]` logits)
plus `command_metadata.json`, which carries the log-mel frontend (Hanning window
+ mel filterbank inline), the ordered `class_to_idx` labels and the calibrated
`unknown_threshold` (0.163). `commands.py` loads it automatically and uses
`features.MetadataFrontend` for bit-compatible preprocessing.

- If the top-1 softmax is **below `unknown_threshold`**, the label is `UNKNOWN`
  and the app logs *"Sorry, I didn't catch that."* without running an action.
- Set `command.unknown_threshold` in `config.yaml` to override the metadata value.
- Without the ONNX file the app falls back to the keyword-simulator placeholder,
  so the GUI still runs.

## Verify (headless)
```bash
# Windows
venv\Scripts\python selftest.py
# Linux/macOS/RPi
./venv/bin/python selftest.py
```
Reports frontend-vs-torchaudio equivalence and positive/negative detection stats.

## Notes
- Wake-word threshold default **0.15** (overridable in `config.yaml` / the slider).
- Command `UNKNOWN` threshold default **0.163** (from `command_metadata.json`).
- When the wake word fires, a short two-tone **"listening" beep**
  (`assets/audio/beep.wav`) plays as the command capture starts. The capture is
  extended by the beep's length so the cue is not scored by the classifier.
  Toggle it with `assets.beep: false` and set level via `assets.beep_volume`.
- `num_threads` capped at 2 to avoid thermal throttling on the Pi 5.
