# Data — what to upload

No datasets are committed here (they are large). The directory layout is kept so
you can drop the files in and run training directly — the configs already point
at these paths (relative to `training/`).

```
data/
├── wake_word/{train,val,test}/   # "Hey Mason" positive clips  (*.wav, 16 kHz mono)
├── negative/                     # background negatives: speech / noise / silence / mex2
│   └── metadata.csv              #   (columns: path, source, split)
├── features/                     # OPTIONAL precomputed log-mel *.npy (speed-up)
└── command/                      # command-classifier manifest + labels
    ├── manifest.csv              #   path, class_name, class_id, split, speaker, condition
    └── labels.json               #   num_classes / class_to_id / id_to_class
```

## 1. Wake word — positives
Put the `"Hey Mason"` recordings here:

```
data/wake_word/train/*.wav
data/wake_word/val/*.wav
data/wake_word/test/*.wav
```
16 kHz mono WAV. (In the source workspace these live under `Wake_Word_Model/data/`.)

## 2. Wake word — negatives
```bash
cd training
python scripts/download_negatives.py --sources speech,esc50,silence   # small
python scripts/download_negatives.py --all                           # adds MUSAN (+ large)
```
This writes `data/negative/<source>/*.wav` and `data/negative/metadata.csv`.
(They are referenced by `configs/wake_word_train.yaml` → `data.negative_ratios`.)

## 3. Optional feature cache (faster epochs)
```bash
python scripts/precompute_features.py --config configs/wake_word_train.yaml
```
Writes `data/features/*.npy`.

## 4. Command-classifier dataset (MEX2 "gold")
Upload the released folder
`ai231-me2-gold-dataset/` (`train/ test/ holdout/ numerals/`, each with
`manifest.csv` + `audio/`) anywhere, then build the flat manifest:

```bash
cd training
python scripts/prepare_gold_mex2.py \
    --gold /path/to/ai231-me2-gold-dataset \
    --out  ../data/command
```
Produces `data/command/manifest.csv` + `data/command/labels.json` (32 classes:
31 intents + `OUT_OF_SCOPE`) and carves a speaker-disjoint `val` split out of
`train`. Then point `configs/command_train.yaml` → `data.audio_root` at that
gold folder and train:

```bash
python train.py --config configs/command_train.yaml
```

> The committed `data/command/{manifest.csv,labels.json}` are the ones used to
> produce the released `MEX2-trained` classifier (audio_root pointed at the gold
> dataset on the build machine); re-run `prepare_gold_mex2.py` if your upload
> path differs.

## 5. Validation test set (rpi_validation)
`app/rpi_validation/test_data/holdout/` is the small labeled holdout used by the
Pi validation notebook (kept in the repo so the notebook runs out of the box).
Add your own clips as `app/rpi_validation/test_data/<CLASS>/*.wav` and
`.../test_data/_negative/*.wav`.
