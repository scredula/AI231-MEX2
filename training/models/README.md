# Models

Training artifacts used by the demo live **inside the self-contained app**:

- Wake-word model + frontend: `app/models/wake_word/`
  (`best_model.onnx`, `frontend.json`, `mel_fbank.npy`, `hann_window.npy`, `norm_stats.json`)

This top-level folder is used by the **training pipeline** for the command model:

```
models/
└── command/        # populated once the MEX2 command dataset is released
    ├── best_model.onnx
    ├── best_model.pt
    ├── labels.json
    ├── frontend.json
    ├── mel_fbank.npy
    ├── hann_window.npy
    └── norm_stats.json
```

## Regenerating

Wake-word (writes `runs/wake_word/<exp>/<ts>/`, then export):
```bash
python scripts/precompute_features.py --config configs/wake_word_train.yaml
python train.py --config configs/wake_word_train.yaml
python scripts/export_model.py --config configs/wake_word_train.yaml \
    --model runs/wake_word/wake_word_hey_mason/<ts>/best_model.pt --out_dir app/models/wake_word
python scripts/export_frontend.py --config configs/wake_word_train.yaml --out_dir app/models/wake_word
```

Command (once MEX2 is released):
```bash
python scripts/prepare_mex2.py --src /path/to/MEX2/Data --out data/command
python train.py --config configs/command_train.yaml
python scripts/export_model.py --config configs/command_train.yaml \
    --model runs/command/<exp>/<ts>/best_model.pt --out_dir app/models/command
```

## Notes
- `.pt` / `.onnx` are git-ignored except `*/best_model.onnx` (see `.gitignore`).
- `norm_stats.json` + `frontend.json` + `mel_fbank.npy` + `hann_window.npy` are
  **required at inference** so the Pi reproduces the training preprocessing exactly.
