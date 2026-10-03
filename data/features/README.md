# Precomputed features (optional)

Optional cache of log-mel `.npy` features used to speed up training epochs.
Generate with:

```bash
cd ../../training
python scripts/precompute_features.py --config configs/wake_word_train.yaml
```

Not committed (large). See `../README.md`.
