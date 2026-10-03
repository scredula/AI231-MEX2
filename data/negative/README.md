# Wake-word negatives

`metadata.csv` (columns: `path,source,split`) indexes the negative clips used by
training; the audio itself is **not** committed. Regenerate it with:

```bash
cd ../../training
python scripts/download_negatives.py --sources speech,esc50,silence
python scripts/download_negatives.py --all          # adds MUSAN (large)
```

Writes `data/negative/<source>/*.wav` + `data/negative/metadata.csv`.
See `../README.md`.
