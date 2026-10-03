#!/usr/bin/env python3
"""
Headless latency / runtime benchmark for the command classifier (no Jupyter).

Measures, using the SAME production code path as the app
(commands.CommandClassifier + features.MetadataFrontend):

  * frontend   -- NumPy log-mel feature extraction
  * onnx       -- pure session.run on pre-computed features
  * end-to-end -- what the app does per command (predict); mean/median/p95/p99
  * real-time factor + throughput, and an intra-op thread sweep
  * optional sustained run with CPU temperature (thermal-throttling check)

Usage (from the app folder):
    ./venv/bin/python rpi_validation/bench_latency.py
    ./venv/bin/python rpi_validation/bench_latency.py --runs 200 --seconds 60
    ./venv/bin/python rpi_validation/bench_latency.py --data my_wavs/ --json out.json

Only numpy + onnxruntime (+ soundfile/scipy for WAVs) are required.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def find_app_dir(start: Path):
    """Locate the app folder (contains commands.py + the command model)."""
    marker = ("models", "command_classifier", "command_metadata.json")
    for base in [start, *start.parents]:
        if (base / "commands.py").exists() and (base / Path(*marker)).exists():
            return base
    for base in [start, *start.parents]:
        for p in base.glob("*/commands.py"):
            if (p.parent / Path(*marker)).exists():
                return p.parent
    return None


HERE = Path(__file__).resolve().parent
APP_DIR = find_app_dir(HERE)
if APP_DIR is None:
    sys.exit("Could not locate the app folder (commands.py + models/command_classifier).")
sys.path.insert(0, str(APP_DIR))

from commands import CommandClassifier                    # noqa: E402
from features import MetadataFrontend, read_wav_mono_16k  # noqa: E402
import yaml                                               # noqa: E402


def cpu_temp_c():
    v = None
    try:
        v = Path("/sys/class/thermal/thermal_zone0/temp").read_text().strip()
    except Exception:
        pass
    if v and v.isdigit():
        return int(v) / 1000.0
    try:
        out = subprocess.check_output(["vcgencmd", "measure_temp"],
                                      stderr=subprocess.DEVNULL).decode()
        return float(out.strip().split("=")[1].split("'")[0])
    except Exception:
        return None


def rpi_model():
    for p in ("/proc/device-tree/model", "/sys/firmware/devicetree/base/model"):
        try:
            return Path(p).read_bytes().decode(errors="ignore").strip("\x00").strip()
        except Exception:
            pass
    return None


def stats(times):
    t = np.asarray(times, dtype=float)
    return {"mean_ms": float(t.mean()), "median_ms": float(np.median(t)),
            "p95_ms": float(np.percentile(t, 95)), "p99_ms": float(np.percentile(t, 99)),
            "min_ms": float(t.min()), "max_ms": float(t.max()), "std_ms": float(t.std())}


def bench_frontend(fe, wavs, warm, runs):
    for i in range(warm):
        fe(wavs[i % len(wavs)])
    ts = []
    for i in range(runs):
        w = wavs[i % len(wavs)]
        t = time.perf_counter_ns(); fe(w); ts.append((time.perf_counter_ns() - t) / 1e6)
    return ts


def bench_onnx(sess, inp, mels, warm, runs):
    for i in range(warm):
        sess.run(None, {inp: mels[i % len(mels)]})
    ts = []
    for i in range(runs):
        m = mels[i % len(mels)]
        t = time.perf_counter_ns(); sess.run(None, {inp: m})
        ts.append((time.perf_counter_ns() - t) / 1e6)
    return ts


def bench_e2e(clf, wavs, warm, runs):
    for i in range(warm):
        clf.predict(wavs[i % len(wavs)])
    ts = []
    for i in range(runs):
        w = wavs[i % len(wavs)]
        t = time.perf_counter_ns(); clf.predict(w); ts.append((time.perf_counter_ns() - t) / 1e6)
    return ts


def load_wavs(data_dir, app_dir, window_samples):
    if data_dir:
        wavs = sorted(Path(data_dir).glob("**/*.wav"))[:20]
    else:
        wavs = sorted((app_dir / "samples").glob("*.wav"))[:5]
    out = [read_wav_mono_16k(p, 16000) for p in wavs] if wavs else []
    if not out:
        out = [np.zeros(window_samples, np.float32)]
        return out, "silence"
    return out, ("data" if data_dir else "bundled samples")


def sustained(clf, wavs, seconds, temp_every=1.0):
    t_end = time.time() + seconds
    lat, temps, stamps = [], [], []
    next_temp = time.time()
    t0 = cpu_temp_c()
    while time.time() < t_end:
        w = wavs[len(lat) % len(wavs)]
        t = time.perf_counter_ns(); clf.predict(w)
        lat.append((time.perf_counter_ns() - t) / 1e6)
        now = time.time()
        if now >= next_temp:
            tv = cpu_temp_c()
            if tv is not None:
                temps.append(tv); stamps.append(now - t_end + seconds)
            next_temp = now + temp_every
    return np.asarray(lat), temps, t0, cpu_temp_c()


def main():
    ap = argparse.ArgumentParser(description="Command-classifier latency benchmark")
    ap.add_argument("--runs", type=int, default=100, help="timed inferences per benchmark")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--threads", type=str, default="1,2,4",
                    help="comma-separated intra-op thread counts to sweep")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="sustained/thermal test duration (0 = skip)")
    ap.add_argument("--data", default=None, help="directory of WAVs to use (optional)")
    ap.add_argument("--json", default=None, help="write results JSON here")
    ap.add_argument("--model-dir", default=None, help="override model dir")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(APP_DIR / "config.yaml"))
    cmd = cfg["command"]
    model_dir = Path(args.model_dir) if args.model_dir else (APP_DIR / cmd["model_dir"])
    clf = CommandClassifier(model_dir, intents=cfg["intents"],
                            onnx_name=cmd.get("onnx_name", "command_classifier.onnx"),
                            meta_name=cmd.get("meta_name", "command_metadata.json"),
                            num_threads=int(cmd.get("num_threads", 2)))
    if not clf.real:
        sys.exit(f"No command model at {model_dir}")

    fe = clf.frontend
    win = fe.window_samples
    clip_s = win / fe.sr
    wavs, src = load_wavs(args.data, APP_DIR, win)
    mels = [fe(w) for w in wavs]

    print("=" * 66)
    print("Command classifier latency benchmark")
    print("=" * 66)
    print(f"target            : {rpi_model() or platform.platform()}")
    print(f"cpu_count         : {os.cpu_count()}")
    print(f"model             : {model_dir.name}  ({len(clf.intents)} classes, "
          f"thr={clf.unknown_threshold})")
    print(f"audio             : {src} ({len(wavs)} clips)")
    print(f"clip length       : {clip_s*1000:.1f} ms   warmup={args.warmup} runs={args.runs}")

    fe_t = bench_frontend(fe, wavs, args.warmup, args.runs)
    onnx_t = bench_onnx(clf.sess, clf.input_name, mels, args.warmup, args.runs)
    e2e_t = bench_e2e(clf, wavs, args.warmup, args.runs)
    F, O, E = stats(fe_t), stats(onnx_t), stats(e2e_t)
    rtf = E["mean_ms"] / (clip_s * 1000)

    print("\n-- latency (ms) --")
    print(f"{'stage':12s} {'mean':>8s} {'median':>8s} {'p95':>8s} {'p99':>8s} {'std':>8s}")
    for name, s in (("frontend", F), ("onnx", O), ("end-to-end", E)):
        print(f"{name:12s} {s['mean_ms']:8.2f} {s['median_ms']:8.2f} "
              f"{s['p95_ms']:8.2f} {s['p99_ms']:8.2f} {s['std_ms']:8.2f}")
    print(f"\nreal-time factor  : {rtf:.4f}  (x{1/rtf:.1f} faster than real time)")
    print(f"throughput        : {1000.0/E['mean_ms']:.1f} inferences/s")

    sweep = {}
    mx = os.cpu_count() or 1
    threads = sorted({int(x) for x in args.threads.split(",") if int(x) <= mx} | {min(2, mx)})
    print("\n-- thread sweep (end-to-end) --")
    for nt in threads:
        c = CommandClassifier(model_dir, intents=cfg["intents"],
                              onnx_name=cmd.get("onnx_name", "command_classifier.onnx"),
                              meta_name=cmd.get("meta_name", "command_metadata.json"),
                              num_threads=nt)
        s = stats(bench_e2e(c, wavs, max(3, args.warmup // 2), max(20, args.runs // 2)))
        s["throughput_per_s"] = 1000.0 / s["mean_ms"]
        sweep[nt] = s
        print(f"threads={nt}: mean={s['mean_ms']:6.2f}  p95={s['p95_ms']:6.2f}  "
              f"({s['throughput_per_s']:5.1f}/s)")
    runtime = None
    if args.seconds > 0:
        lat_s, temps, tc0, tc1 = sustained(clf, wavs, args.seconds)
        half = max(1, len(lat_s) // 2)
        first, second = float(lat_s[:half].mean()), float(lat_s[half:].mean())
        runtime = {"seconds": args.seconds, "n": int(len(lat_s)),
                   "mean_ms": float(lat_s.mean()), "p95_ms": float(np.percentile(lat_s, 95)),
                   "first_half_ms": first, "second_half_ms": second,
                   "drift_ms": second - first,
                   "drift_pct": (second / first - 1) * 100 if first else 0.0,
                   "temp_start_c": tc0, "temp_end_c": tc1,
                   "temp_max_c": (max(temps) if temps else None)}
        print(f"\n-- sustained {args.seconds:.0f}s --")
        print(f"inferences        : {runtime['n']}")
        print(f"latency mean/p95  : {runtime['mean_ms']:.2f} / {runtime['p95_ms']:.2f} ms")
        print(f"drift (1st->2nd)  : {runtime['drift_ms']:+.2f} ms ({runtime['drift_pct']:+.1f}%)")
        if tc0 is not None:
            print(f"cpu temp start/end: {tc0:.1f} / {tc1:.1f} C   max {runtime['temp_max_c']:.1f} C")
        else:
            print("cpu temp          : n/a")

    result = {"target": rpi_model() or platform.platform(), "cpu_count": os.cpu_count(),
              "model_dir": str(model_dir), "classes": len(clf.intents),
              "unknown_threshold": clf.unknown_threshold, "audio_source": src,
              "clip_ms": clip_s * 1000, "warmup": args.warmup, "runs": args.runs,
              "latency": {"frontend": F, "onnx": O, "end_to_end": E,
                          "real_time_factor": rtf, "throughput_per_s": 1000.0 / E["mean_ms"]},
              "thread_sweep": sweep, "runtime": runtime,
              "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    out = Path(args.json) if args.json else (HERE / "results" / "latency.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nsaved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

