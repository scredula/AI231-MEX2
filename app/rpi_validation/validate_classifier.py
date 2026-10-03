#!/usr/bin/env python3
"""
Headless accuracy + latency validation for a command classifier.

Works with BOTH on-disk formats in app/models/command_classifier (the bundled
`command_metadata.json` model and the pipeline-trained `frontend.json` model),
because it uses the app's production path: commands.CommandClassifier, which
auto-detects the frontend (MetadataFrontend or LogMelFrontend).

Reports (when labeled data is available):
  * top-1 / top-3 / top-5 accuracy
  * per-class precision / recall / F1  (+ confusion matrix)
  * false-accept rate (FAR) on negatives / out-of-scope clips and coverage
  * latency: frontend / onnx / end-to-end (mean, p50, p95, p99), real-time factor
  * CPU temperature (Raspberry Pi) if available

Usage (from the app folder):
    ./venv/bin/python rpi_validation/validate_classifier.py                     # config default
    ./venv/bin/python rpi_validation/validate_classifier.py --classifier MEX2-trained
    ./venv/bin/python rpi_validation/validate_classifier.py --all
    ./venv/bin/python rpi_validation/validate_classifier.py --data my_wavs --manifest my.csv
    ./venv/bin/python rpi_validation/validate_classifier.py --out results/val.json

Only numpy + onnxruntime (+ soundfile/scipy for WAVs, pyyaml for config) needed.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent          # .../app/rpi_validation
APP = HERE.parent                               # .../app


def find_app_dir(start: Path):
    for base in [start, *start.parents]:
        if (base / "commands.py").exists() and (base / "models" / "command_classifier").exists():
            return base
    return None


APP_DIR = find_app_dir(HERE) or APP
sys.path.insert(0, str(APP_DIR))

from commands import CommandClassifier            # noqa: E402
from features import read_wav_mono_16k           # noqa: E402


# ---- dataset label -> app class name (the 31 app classes) --------------------
NAME_MAP = {
    "TIMER_10_seconds": "TIMER_10s", "TIMER_30_seconds": "TIMER_30s",
    "TIMER_1_minute": "TIMER_1m",
    "ALARM_6:00_AM": "ALARM_6_00AM", "ALARM_8:00_AM": "ALARM_8_00AM",
    "ALARM_9:00_PM": "ALARM_9_00PM",
    "BRIGHTNESS_20_percent": "BRIGHTNESS_20", "BRIGHTNESS_60_percent": "BRIGHTNESS_60",
    "BRIGHTNESS_100_percent": "BRIGHTNESS_100",
    "COLOR_Red": "COLOR_RED", "COLOR_Green": "COLOR_GREEN", "COLOR_Blue": "COLOR_BLUE",
    "TEMPERATURE_18_degrees": "TEMPERATURE_18", "TEMPERATURE_22_degrees": "TEMPERATURE_22",
    "TEMPERATURE_26_degrees": "TEMPERATURE_26",
    "CREATE_REMINDER_Drink_water": "CREATE_REMINDER_DRINK_WATER",
    "CREATE_REMINDER_Exercise": "CREATE_REMINDER_EXERCISE",
    "CREATE_REMINDER_Study": "CREATE_REMINDER_STUDY",
}
NEG_NAMES = {"_negative", "negatives", "negative", "unknown", "background", "noise",
             "noncommand", "non_command"}


def map_label(command: str, slot: str):
    """(command, slot_value) -> app class name, or None if out-of-scope."""
    cmd = (command or "").strip()
    slot = (slot or "").strip()
    if not cmd or cmd == "OUT_OF_SCOPE":
        return None
    name = cmd if not slot else f"{cmd}_{slot.replace(' ', '_')}"
    return NAME_MAP.get(name, name)


def _read_manifest(path: Path, root: Path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return []
    fields = {f.lower() for f in rows[0].keys()}
    items = []
    if "command" in fields:                       # holdout/exported format
        for r in rows:
            f = r.get("file") or r.get("path")
            if not f:
                continue
            wav = root / f
            if r.get("out_of_scope") == "1":
                items.append((wav, None))
            else:
                lab = map_label(r.get("command", ""), r.get("slot_value", ""))
                if lab:
                    items.append((wav, lab))
    else:                                          # path,label format
        for r in rows:
            f = r.get("path") or r.get("file")
            lab = (r.get("label") or "").strip()
            if not f:
                continue
            items.append((root / f, None if lab.lower() in NEG_NAMES or not lab else lab))
    return items


def discover_data(data_root: Path, manifest: Path | None):
    """Return a list of (wav_path, label_or_None). label None = negative."""
    if manifest and Path(manifest).exists():
        mp = Path(manifest)
        return _read_manifest(mp, mp.parent)
    if data_root:
        data_root = Path(data_root)
        for cand in (data_root / "holdout" / "manifest.csv", data_root / "manifest.csv"):
            if cand.exists():
                return _read_manifest(cand, cand.parent)
        items = []                                 # folder-per-class
        for d in sorted(p for p in data_root.iterdir() if p.is_dir()):
            neg = d.name.lower() in NEG_NAMES
            for w in sorted(d.glob("*.wav")):
                items.append((w, None if neg else d.name))
        return items
    return []


def cpu_temp_c():
    try:
        v = Path("/sys/class/thermal/thermal_zone0/temp").read_text().strip()
        if v.isdigit():
            return int(v) / 1000.0
    except Exception:
        pass
    try:
        out = subprocess.check_output(["vcgencmd", "measure_temp"],
                                      stderr=subprocess.DEVNULL).decode()
        return float(out.strip().split("=")[1].split("'")[0])
    except Exception:
        return None


def _stats(a):
    a = np.asarray(a, dtype=float)
    return {"mean_ms": float(a.mean()), "p50_ms": float(np.percentile(a, 50)),
            "p95_ms": float(np.percentile(a, 95)), "p99_ms": float(np.percentile(a, 99))}


def run_items(clf, items):
    """Classify every item through the app wrapper; collect preds + timings."""
    rows = []
    for wav_path, label in items:
        if not wav_path.exists():
            continue
        wav = read_wav_mono_16k(wav_path, 16000)
        t = time.perf_counter()
        probs = clf.predict(wav)
        e2e = (time.perf_counter() - t) * 1e3
        fe = float("nan")
        if clf.frontend is not None:
            tf = time.perf_counter()
            clf.frontend(wav)
            fe = (time.perf_counter() - tf) * 1e3
        order = np.argsort(-probs)
        top = [clf.intents[i] for i in order[:5]]
        rows.append({"path": str(wav_path), "label": label, "pred": top[0],
                     "conf": float(probs[order[0]]), "top5": top,
                     "e2e_ms": e2e, "fe_ms": fe})
    return rows


def summarize(rows, intents, threshold):
    idx = {c: i for i, c in enumerate(intents)}
    scoped = [r for r in rows if r["label"] in idx]
    negs = [r for r in rows if r["label"] is None]
    n = len(scoped)

    def acc(pred_of):
        return float(sum(pred_of(r) for r in scoped) / n) if n else 0.0

    top1 = acc(lambda r: r["pred"] == r["label"])
    top3 = acc(lambda r: r["label"] in r["top5"][:3])
    top5 = acc(lambda r: r["label"] in r["top5"])

    cm = np.zeros((len(intents), len(intents)), dtype=int)
    for r in scoped:
        if r["label"] in idx and r["pred"] in idx:
            cm[idx[r["label"]], idx[r["pred"]]] += 1
    per_class = {}
    for c in intents:
        tp = cm[idx[c], idx[c]]
        fp = cm[:, idx[c]].sum() - tp
        fn = cm[idx[c], :].sum() - tp
        p = tp / (tp + fp) if (tp + fp) else 0.0
        rc = tp / (tp + fn) if (tp + fn) else 0.0
        per_class[c] = {"precision": float(p), "recall": float(rc),
                        "f1": float(2 * p * rc / (p + rc)) if (p + rc) else 0.0,
                        "support": int(cm[idx[c], :].sum())}

    accepted = lambda r: r["conf"] >= threshold
    far = (sum(accepted(r) for r in negs) / len(negs)) if negs else None
    coverage = (sum(accepted(r) for r in scoped) / n) if n else 0.0
    macro_f1 = float(np.mean([v["f1"] for v in per_class.values()])) if per_class else 0.0

    e2e = [r["e2e_ms"] for r in rows]
    fe = [r["fe_ms"] for r in rows if not np.isnan(r["fe_ms"])]
    return {
        "n_labeled": n, "n_neg": len(negs), "n_total": len(rows),
        "top1": top1, "top3": top3, "top5": top5, "macro_f1": macro_f1,
        "unknown_threshold": threshold,
        "far": far, "coverage": float(coverage),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "latency": {"e2e": _stats(e2e), "frontend": _stats(fe) if fe else None},
    }


def main():
    ap = argparse.ArgumentParser(description="Validate a command classifier (accuracy + latency)")
    ap.add_argument("--classifier", default=None,
                    help="MEX2-31class | MEX2-trained (default: config command.model)")
    ap.add_argument("--all", action="store_true", help="Validate every configured classifier")
    ap.add_argument("--data", default=None,
                    help="test-data dir (default: rpi_validation/test_data)")
    ap.add_argument("--manifest", default=None, help="explicit manifest CSV")
    ap.add_argument("--out", default=None, help="output JSON path")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(APP_DIR / "config.yaml"))
    intents = cfg.get("intents", []) or []
    cmd = cfg.get("command", {}) or {}
    reg = cmd.get("models", {}) or {}
    if not reg and cmd.get("model_dir"):
        reg = {"default": cmd["model_dir"]}

    if args.all:
        names = list(reg)
    elif args.classifier:
        names = [k for k in reg if args.classifier.lower() in (k.lower(), Path(reg[k]).name.lower())]
        if not names:
            print(f"Unknown --classifier '{args.classifier}'. Options: {', '.join(reg)}")
            return 2
    else:
        d = cmd.get("model")
        names = [k for k in reg if d and d.lower() in (k.lower(), Path(reg[k]).name.lower())]
        names = names or list(reg)[:1]

    data_root = Path(args.data) if args.data else (HERE / "test_data")
    items = discover_data(data_root, args.manifest)
    n_pos = sum(1 for _, l in items if l is not None)
    n_neg = sum(1 for _, l in items if l is None)
    print(f"data: {data_root}  ->  {n_pos} labeled + {n_neg} negatives")

    results = []
    for name in names:
        rel = reg[name]
        clf = CommandClassifier(APP_DIR / rel, intents=intents,
                                num_threads=int(cmd.get("num_threads", 2)))
        print(f"\n=== {name}  ({rel}) ===")
        if not clf.real or clf.frontend is None:
            print("  !! model or frontend not found")
            continue
        print(f"  onnx={clf.onnx_name}  frontend={clf.frontend_kind}  "
              f"classes={len(clf.intents)}  unknown_threshold={clf.unknown_threshold:.3f}")
        rows = run_items(clf, items)
        rep = summarize(rows, clf.intents, clf.unknown_threshold)
        print(f"  top1={rep['top1']:.4f}  top3={rep['top3']:.4f}  top5={rep['top5']:.4f}  "
              f"macro_f1={rep['macro_f1']:.4f}  (n={rep['n_labeled']})")
        if rep["far"] is not None:
            print(f"  FAR={rep['far']:.4f}  coverage={rep['coverage']:.4f}  (neg n={rep['n_neg']})")
        lat = rep["latency"]["e2e"]
        print(f"  latency e2e: mean={lat['mean_ms']:.2f}ms  p95={lat['p95_ms']:.2f}ms  "
              f"p99={lat['p99_ms']:.2f}ms")
        rep.update({"classifier": name, "path": rel, "classes": clf.intents,
                    "frontend_kind": clf.frontend_kind, "onnx": clf.onnx_name,
                    "cpu_temp_c": cpu_temp_c(), "target": platform.platform(),
                    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")})
        out = Path(args.out) if args.out else (HERE / "results" / f"validate_{Path(rel).name}.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump(rep, open(out, "w"), indent=2)
        print(f"  saved: {out}")
        results.append(rep)

    print("\nDONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())



