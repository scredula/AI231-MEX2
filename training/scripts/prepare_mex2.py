#!/usr/bin/env python3
"""
Prepare the MEX2 command dataset for the command classifier.

Reads the released MEX2 `manifest.csv` (columns include: path, intent, speaker,
split, phrase, condition, transcript, slot value, duration) and produces:

  data/command/manifest.csv   -> path, class_name, class_id, split, speaker, condition
  data/command/labels.json    -> class_to_id / id_to_class

Class granularity is INTENT + SLOT (e.g. BRIGHTNESS_100, COLOR_RED, TEMPERATURE_22,
ALARM_4_00AM ...), i.e. the folder names in MEX2/Data (32 classes).

Usage:
    python scripts/prepare_mex2.py --src /path/to/MEX2/Data --out data/command
    python scripts/prepare_mex2.py --src /path/to/MEX2/Data \
        --manifest /path/to/MEX2/Data/manifest.csv --out data/command
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def discover_from_dirs(src: Path):
    """Fallback: derive classes from directory layout + filename convention."""
    rows = []
    for intent_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        cls = intent_dir.name
        for wav in sorted(intent_dir.glob("*.wav")):
            parts = wav.stem.split("_")  # <FOLDER>_s<speaker>_v<variation>_<condition>
            speaker, condition = "", ""
            for tok in parts:
                if tok.startswith("s") and tok[1:].isdigit():
                    speaker = tok[1:]
                if tok in ("clean", "noisy"):
                    condition = tok
            rows.append({
                "path": str(wav.relative_to(src)),
                "class_name": cls,
                "speaker": speaker,
                "condition": condition,
            })
    return rows


def load_from_manifest(manifest_path: Path, src: Path):
    """Derive classes from a released manifest.csv."""
    rows = []
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []

        def pick(*cands):
            for c in cands:
                for fn in fieldnames:
                    if fn.lower().replace(" ", "_") == c:
                        return fn
            return None

        path_c = pick("audio_path", "path", "filepath", "file")
        intent_c = pick("intent", "folder", "label")
        slot_c = pick("slot_value", "slot", "value")
        split_c = pick("split")
        speaker_c = pick("speaker", "speaker_id")
        cond_c = pick("condition")

        for r in reader:
            if not path_c:
                continue
            rel = r[path_c]
            folder = Path(rel).parent.name
            if folder:
                cls = folder
            else:
                intent = r.get(intent_c, "") if intent_c else ""
                slot = r.get(slot_c, "") if slot_c else ""
                cls = f"{intent}_{slot}".strip("_") if slot else intent
            rows.append({
                "path": rel,
                "class_name": cls,
                "speaker": r.get(speaker_c, "") if speaker_c else "",
                "condition": r.get(cond_c, "") if cond_c else "",
                "split": r.get(split_c, "") if split_c else "",
            })
    return rows


def main():
    ap = argparse.ArgumentParser(description="Prepare MEX2 command dataset")
    ap.add_argument("--src", type=str, required=True, help="Path to MEX2/Data")
    ap.add_argument("--manifest", type=str, default=None,
                    help="Path to MEX2 manifest.csv (if released)")
    ap.add_argument("--out", type=str, default="data/command")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.manifest and Path(args.manifest).exists():
        print(f"Reading manifest: {args.manifest}")
        rows = load_from_manifest(Path(args.manifest), src)
    else:
        print(f"No manifest provided; scanning directories under {src}")
        rows = discover_from_dirs(src)

    if not rows:
        print("No rows found. Check --src / --manifest.")
        return 1

    classes = sorted({r["class_name"] for r in rows})
    class_to_id = {c: i for i, c in enumerate(classes)}

    # Speaker-disjoint split fallback (if manifest lacks split)
    have_split = any(r.get("split") for r in rows)
    if not have_split:
        print("No 'split' column; creating speaker-disjoint 80/10/10 split...")
        speakers = sorted({r["speaker"] for r in rows if r["speaker"]})
        n = len(speakers)
        n_tr, n_va = int(n * 0.8), int(n * 0.1)
        sp_split = {}
        for i, sp in enumerate(speakers):
            sp_split[sp] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
        for r in rows:
            r["split"] = sp_split.get(r.get("speaker", ""), "train")

    out_manifest = out / "manifest.csv"
    with open(out_manifest, "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["path", "class_name", "class_id", "split", "speaker", "condition"])
        w.writeheader()
        for r in rows:
            w.writerow({
                "path": r["path"], "class_name": r["class_name"],
                "class_id": class_to_id[r["class_name"]],
                "split": r.get("split", "train"),
                "speaker": r.get("speaker", ""), "condition": r.get("condition", ""),
            })
    print(f"Wrote {len(rows)} rows to {out_manifest}")

    labels = {
        "num_classes": len(classes),
        "class_to_id": class_to_id,
        "id_to_class": {str(i): c for c, i in class_to_id.items()},
    }
    with open(out / "labels.json", "w") as f:
        json.dump(labels, f, indent=2)
    print(f"Wrote {len(classes)} classes to {out / 'labels.json'}")

    print("By split:", dict(Counter(r.get("split", "train") for r in rows)))
    print("Top classes:", dict(Counter(r["class_name"] for r in rows).most_common(5)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
