#!/usr/bin/env python3
"""
Prepare the released AI231 "gold" MEX2 command dataset for the command classifier.

The released dataset ships one manifest per split:

    <gold>/train/manifest.csv
    <gold>/test/manifest.csv
    <gold>/holdout/manifest.csv
    <gold>/numerals/manifest.csv      (number-only, not used for the 32-class task)

This script flattens them into what the training pipeline expects:

    <out>/manifest.csv   -> path, class_name, class_id, split, speaker, condition
    <out>/labels.json    -> num_classes / class_to_id / id_to_class

Class granularity is INTENT (plus the SLOT value for slotted commands) plus a
single OUT_OF_SCOPE class — i.e. the 32 classes present in the released
test/holdout splits. Train rows whose slot value falls outside the schema
(bucket "<COMMAND> (other slot value)") are mapped to OUT_OF_SCOPE.

The released dataset has no validation split, so a speaker-disjoint `val`
split is carved out of `train`.

Usage:
    python scripts/prepare_gold_mex2.py \
        --gold /path/to/ai231-me2-gold-dataset --out data/command
"""
import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

# Splits read from the released dataset (numerals are intentionally skipped).
SOURCE_SPLITS = ["train", "test", "holdout"]
OUT_SPLITS = ["train", "val", "test", "holdout"]


def class_name_for(row: dict) -> str:
    """Map a gold-dataset row to one of the 32 intent(+slot) classes."""
    cmd = (row.get("command") or "").strip()
    if not cmd or cmd == "OUT_OF_SCOPE":
        return "OUT_OF_SCOPE"
    bucket = row.get("bucket") or ""
    if "other slot value" in bucket:
        # A slotted command whose slot value is outside the schema.
        return "OUT_OF_SCOPE"
    slot = (row.get("slot_value") or "").strip()
    if slot:
        return f"{cmd}_{slot.replace(' ', '_')}"
    return cmd


def load_split(gold: Path, split: str):
    """Read one split's manifest and return normalised rows."""
    manifest = gold / split / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest}")
    rows = []
    with open(manifest, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rel = r.get("file")
            if not rel:
                continue
            rows.append({
                "path": f"{split}/{rel}",              # relative to --gold root
                "class_name": class_name_for(r),
                "speaker": (r.get("speaker_id") or "").strip(),
                "condition": "synthetic" if r.get("is_synthetic") == "1" else "real",
            })
    return rows


def carve_val(rows, val_fraction: float, seed: int):
    """Speaker-disjoint val split carved out of the train rows."""
    speakers = sorted({r["speaker"] for r in rows if r["speaker"]})
    rng = random.Random(seed)
    rng.shuffle(speakers)
    n_val = max(1, int(len(speakers) * val_fraction))
    val_speakers = set(speakers[:n_val])
    for r in rows:
        r["split"] = "val" if r["speaker"] in val_speakers else "train"
    return len(val_speakers)


def main():
    ap = argparse.ArgumentParser(description="Prepare the gold MEX2 command dataset")
    ap.add_argument("--gold", type=str, required=True,
                    help="Path to ai231-me2-gold-dataset (contains train/test/holdout)")
    ap.add_argument("--out", type=str, default="data/command")
    ap.add_argument("--val-fraction", type=float, default=0.10,
                    help="Fraction of train speakers held out for validation")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    gold = Path(args.gold).expanduser().resolve()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for split in SOURCE_SPLITS:
        split_rows = load_split(gold, split)
        if split == "train":
            n_val_spk = carve_val(split_rows, args.val_fraction, args.seed)
            print(f"  train: carved val from {n_val_spk} speaker(s)")
        else:
            for r in split_rows:
                r["split"] = split
        rows.extend(split_rows)
        print(f"  {split}: {len(split_rows)} rows")

    if not rows:
        print("No rows found. Check --gold.")
        return 1

    classes = sorted({r["class_name"] for r in rows})
    class_to_id = {c: i for i, c in enumerate(classes)}

    out_manifest = out / "manifest.csv"
    with open(out_manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["path", "class_name", "class_id", "split", "speaker", "condition"])
        w.writeheader()
        for r in rows:
            w.writerow({
                "path": r["path"],
                "class_name": r["class_name"],
                "class_id": class_to_id[r["class_name"]],
                "split": r["split"],
                "speaker": r["speaker"],
                "condition": r["condition"],
            })
    print(f"Wrote {len(rows)} rows to {out_manifest}")

    labels = {
        "num_classes": len(classes),
        "class_to_id": class_to_id,
        "id_to_class": {str(i): c for c, i in class_to_id.items()},
    }
    with open(out / "labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)
    print(f"Wrote {len(classes)} classes to {out / 'labels.json'}")

    print("By split:", dict(Counter(r["split"] for r in rows)))
    print("Classes:", classes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
