#!/usr/bin/env python3
"""
Raspberry Pi 5 demo: always-on wake word ("Hey Mason") + command classifier UI.

Layout
------
Left  : wake-word probability bar (real time), threshold slider, mic controls.
Right : top-10 command predictions and an action log.

Actions
-------
- music intents (PLAY_MUSIC, VOLUME_UP, VOLUME_DOWN, NEXT, PAUSE, STOP) drive pygame
- WEATHER plays assets/audio/weather.mp3
- every other intent shows a pre-generated JPG popup

Run:
    python demo/app.py
Without a mic, use the "Simulate Wake" / "Simulate Command" controls to test.
"""
from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

from features import read_wav_mono_16k                          # noqa: E402
from wakeword import WakeWordDetector, SlidingWakeWord          # noqa: E402
from commands import CommandClassifier, action_for_intent       # noqa: E402
from audio_io import MicSource, list_input_devices                  # noqa: E402
from actions import MusicPlayer, ClipPlayer, PopupManager, BeepPlayer  # noqa: E402


def resolve(p: str | Path) -> Path:
    """Resolve a path relative to the app folder (so it is portable)."""
    p = Path(p)
    return p if p.is_absolute() else (APP_DIR / p)


def load_config(path: str | Path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


class DemoApp:
    def __init__(self, cfg):
        self.cfg = cfg

        # ---- models ----
        ww_cfg = cfg["wake_word"]
        self.detector = WakeWordDetector(
            resolve(ww_cfg["model_dir"]),
            onnx_name=ww_cfg.get("onnx_name", "best_model.onnx"),
            threshold=float(ww_cfg.get("threshold", 0.85)),
            num_threads=int(ww_cfg.get("num_threads", 2)),
        )
        self.slider = SlidingWakeWord(
            self.detector,
            hop_ms=int(ww_cfg.get("hop_ms", 100)),
            trigger_threshold=float(ww_cfg.get("threshold", 0.85)),
            consecutive_hits=int(ww_cfg.get("consecutive_hits", 2)),
            cooldown_s=float(ww_cfg.get("cooldown_s", 2.0)),
        )

        cmd_cfg = cfg["command"]
        self.classifier = CommandClassifier(
            resolve(cmd_cfg["model_dir"]),
            intents=cfg["intents"],
            onnx_name=cmd_cfg.get("onnx_name", "command_classifier.onnx"),
            meta_name=cmd_cfg.get("meta_name", "command_metadata.json"),
            num_threads=int(cmd_cfg.get("num_threads", 2)),
            unknown_threshold=cmd_cfg.get("unknown_threshold"),
        )
        self.capture_s = float(cmd_cfg.get("capture_s", 1.5))

        # ---- actions ----
        self.music = MusicPlayer(resolve(cfg["music"]["dir"]), volume=cfg["music"].get("volume", 0.7))
        self.clip_player = ClipPlayer()
        assets_cfg = cfg["assets"]
        self.weather_clip = resolve(assets_cfg["weather_clip"])
        beep_wav = assets_cfg.get("beep_wav", "assets/audio/beep.wav")
        self.beep = BeepPlayer(
            resolve(beep_wav) if beep_wav else None,
            volume=float(assets_cfg.get("beep_volume", 0.4)),
            enabled=bool(assets_cfg.get("beep", True)),
        )

        # ---- state ----
        self.audio_q: "queue.Queue[np.ndarray]" = queue.Queue()
        self.mic = None
        self.capturing = False
        self.capture_buf = []
        self.capture_until = 0.0
        self.prob = 0.0

        # ---- UI ----
        self._build_ui()

    # ---------------------------------------------------------------- UI
    def _build_ui(self):
        import tkinter as tk
        from tkinter import ttk

        self.root = tk.Tk()
        self.root.title("Hey Mason — Wake Word + Command Demo")
        self.root.geometry("980x620")

        self.popup = PopupManager(self.root,
                                  resolve(self.cfg["assets"]["jpgs_dir"]),
                                  int(self.cfg["assets"].get("popup_duration_ms", 2500)))

        left = ttk.Frame(self.root, padding=12)
        left.grid(row=0, column=0, sticky="nsew")
        right = ttk.Frame(self.root, padding=12)
        right.grid(row=0, column=1, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # --- wake-word pane ---
        ttk.Label(left, text='Wake Word: "Hey Mason"', font=("TkDefaultFont", 14, "bold")).pack(anchor="w")
        self.canvas = tk.Canvas(left, height=46, width=420, bg="#20232a", highlightthickness=0)
        self.canvas.pack(fill="x", pady=(8, 8))
        self._draw_bar(0.0)

        ttk.Label(left, text="Trigger threshold").pack(anchor="w")
        self.thresh_var = tk.DoubleVar(value=self.slider.trigger_threshold)
        ttk.Scale(left, from_=0.05, to=0.99, variable=self.thresh_var, orient="horizontal",
                  command=self._on_threshold).pack(fill="x")

        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(left, textvariable=self.status_var, font=("TkDefaultFont", 12)).pack(anchor="w", pady=(10, 4))

        row = ttk.Frame(left)
        row.pack(fill="x", pady=6)
        self.start_btn = ttk.Button(row, text="Start Mic", command=self.toggle_mic)
        self.start_btn.pack(side="left")
        ttk.Button(row, text="Simulate Wake", command=self.simulate_wake).pack(side="left", padx=6)

        self.mic_info = tk.StringVar(value="mic: (not started)")
        ttk.Label(left, textvariable=self.mic_info, font=("TkDefaultFont", 9)).pack(anchor="w", pady=(6, 0))

        # --- command pane ---
        ttk.Label(right, text="Top-10 Commands", font=("TkDefaultFont", 14, "bold")).pack(anchor="w")
        self.topk = tk.Listbox(right, height=11, font=("TkFixedFont", 10))
        self.topk.pack(fill="both", expand=True, pady=(8, 8))

        sim_row = ttk.Frame(right)
        sim_row.pack(fill="x")
        ttk.Label(sim_row, text="Simulate command:").pack(side="left")
        self.sim_entry = ttk.Entry(sim_row)
        self.sim_entry.pack(side="left", fill="x", expand=True, padx=6)
        self.sim_entry.bind("<Return>", lambda e: self.simulate_command())
        ttk.Button(sim_row, text="Run", command=self.simulate_command).pack(side="left")

        ttk.Label(right, text="Action log", font=("TkDefaultFont", 11, "bold")).pack(anchor="w", pady=(10, 2))
        self.logbox = tk.Text(right, height=8, font=("TkFixedFont", 9), state="disabled")
        self.logbox.pack(fill="both", expand=True)

        if not self.classifier.real:
            self._log("NOTE: command model is a PLACEHOLDER (MEX2 model not trained yet).")
        self._log("Ready.")
        self.root.after(50, self._tick)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _draw_bar(self, p):
        c = self.canvas
        c.delete("all")
        w = int(c["width"]); h = int(c["height"])
        fill_w = int(w * max(0.0, min(1.0, p)))
        color = "#3ec46d" if p < 0.5 else ("#e0a52b" if p < 0.85 else "#e0483f")
        c.create_rectangle(0, 0, fill_w, h, fill=color, width=0)
        tx = int(w * self.slider.trigger_threshold)
        c.create_line(tx, 0, tx, h, fill="white", width=2)
        c.create_text(8, h // 2, anchor="w", fill="white", text=f"prob = {p:0.3f}")

    # ------------------------------------------------------------- helpers
    def _log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self.logbox.configure(state="normal")
        self.logbox.insert("end", f"[{ts}] {msg}\n")
        self.logbox.see("end")
        self.logbox.configure(state="disabled")

    def _on_threshold(self, _=None):
        self.slider.trigger_threshold = float(self.thresh_var.get())
        self._draw_bar(self.prob)

    # -------------------------------------------------------------- audio
    def on_audio(self, chunk: np.ndarray):
        self.audio_q.put(chunk)

    def _ensure_mic(self) -> bool:
        """Start the microphone if it is not already running."""
        if self.mic is None:
            self.toggle_mic()
        return self.mic is not None

    def _begin_listening(self):
        """Open a command-capture window (play the cue, then listen)."""
        if self.beep.enabled:
            self._log(f"Wake cue: {self.beep.play()}")
        self.capturing = True
        self.capture_buf = []
        # The cue plays first; the extra lead keeps the beep out of the
        # classifier window (the frontend scores only the last window_samples).
        lead = self.beep.duration_s if self.beep.enabled else 0.0
        self.capture_until = time.time() + lead + self.capture_s
        self.status_var.set("Listening for a command…")

    def toggle_mic(self):
        if self.mic is None:
            try:
                self.mic = MicSource(self.on_audio, block_ms=int(self.cfg["audio"].get("block_ms", 32)),
                                     device=self.cfg["audio"].get("device"))
                sr = self.mic.start()
                self.start_btn.configure(text="Stop Mic")
                self.mic_info.set(f"mic: running @ {sr} Hz")
                self.status_var.set("Listening…")
                self._log(f"Microphone started ({sr} Hz).")
            except Exception as e:
                self.mic = None
                self._log(f"Mic error: {e}")
                self.mic_info.set("mic: unavailable — install sounddevice/libportaudio2")
        else:
            self.mic.stop(); self.mic = None
            self.start_btn.configure(text="Start Mic")
            self.mic_info.set("mic: stopped")
            self.status_var.set("Idle")
            self._log("Microphone stopped.")

    # ---------------------------------------------------------- main loop
    def _tick(self):
        try:
            while True:
                chunk = self.audio_q.get_nowait()
                self.slider.push(chunk)
                if self.capturing:
                    self.capture_buf.append(chunk)
        except queue.Empty:
            pass

        self.prob = self.slider.prob
        self._draw_bar(self.prob)

        if self.slider.poll_trigger():
            self._log(f"WAKE WORD DETECTED (prob={self.prob:.3f})")
            if not self.capturing:
                self._begin_listening()

        if self.capturing and time.time() >= self.capture_until:
            self.capturing = False
            self._run_command(extra_audio=self.capture_buf)
            self.status_var.set("Listening…" if self.mic else "Ready")

        self.root.after(50, self._tick)

    # ------------------------------------------------------------ commands
    def _run_command(self, extra_audio=None, text=None):
        if text is not None:
            # typed simulation: keyword matcher (no audio to run the model on)
            probs = self.classifier.placeholder_from_text(text)
            top = self.classifier.top_k(probs, 10)
            best, conf = top[0]
        elif self.classifier.real and extra_audio:
            wav = np.concatenate(extra_audio) if extra_audio else np.zeros(1, dtype=np.float32)
            best, conf, top = self.classifier.classify(wav)
        else:
            probs = self.classifier.placeholder_probs()
            top = self.classifier.top_k(probs, 10)
            best, conf = top[0]

        self.topk.delete(0, "end")
        for name, p in top:
            self.topk.insert("end", f"{p*100:5.1f}%  {name}")

        if best == "UNKNOWN":
            thr = self.classifier.unknown_threshold * 100
            self._log(f"Command → UNKNOWN (top-1 {conf*100:.1f}% < {thr:.1f}%)")
            self._log("Sorry, I didn't catch that.")
            return
        self._log(f"Command → {best} ({conf*100:.1f}%)")
        self.execute_action(best)

    def execute_action(self, intent: str):
        kind = action_for_intent(intent)
        if kind == "music":
            method = {
                "PLAY_MUSIC": self.music.play,
                "NEXT": self.music.next,
                "PAUSE": self.music.pause_toggle,
                "STOP": self.music.stop,
                "VOLUME_UP": self.music.volume_up,
                "VOLUME_DOWN": self.music.volume_down,
            }.get(intent, self.music.play)
            self._log("music: " + method())
        elif kind == "weather":
            self._log("weather: " + self.clip_player.play(self.weather_clip))
            self._log('  "The weather today is sunny."')
        else:
            self._log("popup: " + self.popup.show(intent))

    # ------------------------------------------------------------ controls
    def simulate_wake(self):
        """Inject a wake-word trigger, then listen for a spoken command.

        No wake-word WAV is replayed: the trigger is forced directly, the wake
        cue plays, and the microphone is (auto)started so you can say a command
        during the normal capture window.
        """
        if self.capturing:
            self._log("Already listening for a command…")
            return
        self._log("Simulated wake word trigger.")
        if not self._ensure_mic():
            self._log("No microphone — cannot hear a command (start the mic first).")
            return
        self._begin_listening()

    def simulate_command(self):
        text = self.sim_entry.get().strip()
        if not text:
            return
        self._log(f"Simulated command: '{text}'")
        self._run_command(text=text)
        self.sim_entry.delete(0, "end")

    def on_close(self):
        try:
            if self.mic is not None:
                self.mic.stop()
        finally:
            self.root.destroy()

    def run(self):
        self.root.mainloop()


def _model_registry(cfg):
    return (cfg.get("wake_word", {}) or {}).get("models", {}) or {}


def _classifier_registry(cfg):
    cmd = cfg.get("command", {}) or {}
    reg = cmd.get("models", {}) or {}
    if not reg and cmd.get("model_dir"):
        reg = {"default": cmd["model_dir"]}
    return reg


def _canon_model(registry, name):
    """Match a --model value by display name (case-insensitive) or dir basename."""
    if not name:
        return None
    for k, v in registry.items():
        if name.lower() == k.lower() or name.lower() == Path(v).name.lower():
            return k
    return None


def main():
    ap = argparse.ArgumentParser(description="Hey Mason wake-word + command demo")
    ap.add_argument("--config", default=str(APP_DIR / "config.yaml"))
    ap.add_argument("--model", default=None,
                    help="Wake-word model: DS-CNN | TC-ResNet | MatchboxNet | VGG")
    ap.add_argument("--classifier", default=None,
                    help="Command classifier: MEX2-31class | MEX2-trained")
    ap.add_argument("--list-models", action="store_true",
                    help="List available wake-word + classifier models and exit")
    ap.add_argument("--ui", choices=["tk", "web"], default="tk",
                    help="tk = desktop window (default); web = browser simulator")
    ap.add_argument("--host", default=None, help="[--ui web] bind address")
    ap.add_argument("--port", type=int, default=None, help="[--ui web] HTTP port")
    ap.add_argument("--no-mic", action="store_true",
                    help="[--ui web] run without the microphone")
    args = ap.parse_args()

    if args.ui == "web":
        # Hand over to the browser UI, forwarding the shared flags.
        import webui
        argv = [sys.argv[0], "--config", args.config]
        if args.model:
            argv += ["--model", args.model]
        if args.classifier:
            argv += ["--classifier", args.classifier]
        if args.host:
            argv += ["--host", args.host]
        if args.port:
            argv += ["--port", str(args.port)]
        if args.no_mic:
            argv += ["--no-mic"]
        old_argv, sys.argv = sys.argv, argv
        try:
            return webui.main()
        finally:
            sys.argv = old_argv

    cfg = load_config(args.config)
    registry = _model_registry(cfg)
    clf_registry = _classifier_registry(cfg)

    if args.list_models:
        print("Available wake-word models (use --model <NAME>):")
        for k, v in registry.items():
            present = (APP_DIR / v / "best_model.onnx").exists()
            print(f"  {k:<12} -> {v:<32} {'[present]' if present else '[missing]'}")
        print("\nAvailable command classifiers (use --classifier <NAME>):")
        for k, v in clf_registry.items():
            present = any((APP_DIR / v / f).exists()
                          for f in ("best_model.onnx", "command_classifier.onnx"))
            print(f"  {k:<14} -> {v:<32} {'[present]' if present else '[missing]'}")
        return 0

    # Wake word: --model overrides the config default (wake_word.model).
    ww_name = args.model or (cfg.get("wake_word", {}) or {}).get("model")
    if ww_name:
        key = _canon_model(registry, ww_name)
        if key is None:
            if args.model:
                opts = ", ".join(registry) or "(none configured)"
                print(f"Unknown --model '{args.model}'. Options: {opts}")
                return 2
        else:
            cfg.setdefault("wake_word", {})["model_dir"] = registry[key]
            print(f"Using wake-word model: {key}  ({registry[key]})")

    # Command classifier: --classifier overrides the config default.
    clf_name = args.classifier or (cfg.get("command", {}) or {}).get("model")
    if clf_name:
        key = _canon_model(clf_registry, clf_name)
        if key is None:
            if args.classifier:
                opts = ", ".join(clf_registry) or "(none configured)"
                print(f"Unknown --classifier '{args.classifier}'. Options: {opts}")
                return 2
        else:
            cfg.setdefault("command", {})["model_dir"] = clf_registry[key]
            print(f"Using command classifier: {key}  ({clf_registry[key]})")

    try:
        app = DemoApp(cfg)
    except Exception as e:
        print(f"Failed to start demo: {e}")
        print("Tip: run 'python generate_assets.py' first, and check that "
              "models/wake_word/<model>/best_model.onnx exists.")
        return 1
    app.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())

