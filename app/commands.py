"""
Command classifier wrapper for the demo.

Supports **two** on-disk command-model formats (auto-detected from the files in
``model_dir``), so different trained classifiers can be selected at runtime:

  1. Metadata format (bundled / validated on the Pi 5)::

        command_classifier.onnx     # input logmel [1,1,64,150]
        command_metadata.json       # inline Hann window + mel fbank + class_to_idx
                                    # + calibrated unknown_threshold

     Loaded with ``features.MetadataFrontend``.

  2. Training format (exported by the ``training/`` pipeline)::

        best_model.onnx             # input logmel [1,1,64,<T>]
        frontend.json               # torchaudio-equivalent log-mel params (+mean/std)
        mel_fbank.npy, hann_window.npy
        labels.json                 # class_to_id / id_to_class

     Loaded with ``features.LogMelFrontend`` (bit-compatible with training).

If neither model is present, a PLACEHOLDER keeps the GUI demo-able.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from features import LogMelFrontend, MetadataFrontend


MUSIC_INTENTS = {"PLAY_MUSIC", "VOLUME_UP", "VOLUME_DOWN", "NEXT", "PAUSE", "STOP"}
WEATHER_INTENTS = {"WEATHER"}


class CommandClassifier:
    def __init__(
        self,
        model_dir: str | Path,
        intents: List[str],
        onnx_name: Optional[str] = None,
        meta_name: Optional[str] = None,
        num_threads: int = 2,
        temperature: float = 1.0,
        unknown_threshold: Optional[float] = None,
    ):
        self.model_dir = Path(model_dir)
        self.intents = list(intents)
        self.temperature = float(temperature)
        # UNKNOWN is returned when the top-1 softmax is below this value.
        self.unknown_threshold = float(unknown_threshold) if unknown_threshold is not None else 0.5

        # Auto-detect the ONNX file (accept either naming convention).
        candidates = ([onnx_name] if onnx_name else []) + ["best_model.onnx", "command_classifier.onnx"]
        self.onnx_name = next((c for c in candidates if c and (self.model_dir / c).exists()),
                              candidates[-1])
        self.real = (self.model_dir / self.onnx_name).exists()
        self.sess = None
        self.frontend = None
        self.frontend_kind = None

        if self.real:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.intra_op_num_threads = num_threads
            so.inter_op_num_threads = 1
            self.sess = ort.InferenceSession(
                str(self.model_dir / self.onnx_name),
                sess_options=so,
                providers=["CPUExecutionProvider"],
            )
            self.input_name = self.sess.get_inputs()[0].name

            # Optional per-class names (training-format export).
            labels_path = self.model_dir / "labels.json"
            if labels_path.exists():
                with open(labels_path) as f:
                    classes = list(json.load(f).get("class_to_id", {}).keys())
                if classes:
                    self.intents = classes

            # Metadata format (bundled model): inline frontend + class_to_idx.
            meta_path = self.model_dir / (meta_name or "command_metadata.json")
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                if "class_to_idx" in meta:
                    classes = [None] * len(meta["class_to_idx"])
                    for name, i in meta["class_to_idx"].items():
                        classes[int(i)] = name
                    self.intents = classes
                elif "class_names" in meta:
                    self.intents = list(meta["class_names"])
                if unknown_threshold is None:
                    self.unknown_threshold = float(meta.get("unknown_threshold", 0.5))
                self.frontend = MetadataFrontend(meta_path)
                self.frontend_kind = "metadata"
            # Training format: torchaudio-equivalent log-mel from frontend.json.
            elif (self.model_dir / "frontend.json").exists():
                self.frontend = LogMelFrontend(self.model_dir)
                self.frontend_kind = "logmel"

    # ---- inference ----
    def predict(self, wav_16k: np.ndarray) -> np.ndarray:
        """Return probabilities over intents."""
        if self.real and self.frontend is not None:
            feat = self.frontend(wav_16k).astype(np.float32)
            if feat.ndim == 2:                        # LogMelFrontend -> (n_mels, T)
                feat = feat[None, None, :, :]
            logits = np.asarray(self.sess.run(None, {self.input_name: feat})[0]).reshape(-1)
            logits = logits / max(self.temperature, 1e-6)
            e = np.exp(logits - logits.max())
            return e / e.sum()
        return self.placeholder_probs()

    def classify(self, wav_16k: np.ndarray) -> Tuple[str, float, List[Tuple[str, float]]]:
        """Return (label, confidence, top10); label is "UNKNOWN" below threshold."""
        probs = self.predict(wav_16k)
        top = self.top_k(probs, 10)
        label, conf = top[0]
        if conf < self.unknown_threshold:
            label = "UNKNOWN"
        return label, conf, top

    def top_k(self, probs: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        idx = np.argsort(-probs)[:k]
        return [(self.intents[i], float(probs[i])) for i in idx]

    # ---- placeholder (until MEX2 command model is trained) ----
    def placeholder_probs(self) -> np.ndarray:
        p = np.ones(len(self.intents), dtype=np.float64)
        p += 0.15 * np.random.rand(len(self.intents))
        return p / p.sum()

    def placeholder_from_text(self, text: str) -> np.ndarray:
        """Fake a confident prediction by keyword-matching a typed phrase."""
        import re
        text = text.lower()
        nums = [int(n) for n in re.findall(r"\d+", text)]
        score = np.zeros(len(self.intents), dtype=np.float64)

        def hit(name, w=1.0):
            if name in self.intents:
                score[self.intents.index(name)] += w

        # simple keyword intents
        simple = {
            "PLAY_MUSIC": ["play music", "play a song", "play some music", "start music"],
            "VOLUME_UP": ["volume up", "louder", "increase the volume", "turn the volume up"],
            "VOLUME_DOWN": ["volume down", "quieter", "lower the volume", "turn the volume down"],
            "NEXT": ["next", "skip"],
            "PAUSE": ["pause", "pause for now"],
            "STOP": ["stop", "end playback", "stop playing"],
            "LIGHT_ON": ["lights on", "turn on the lights", "power on the lights", "light on"],
            "LIGHT_OFF": ["lights off", "turn off the lights", "lights out", "light off"],
            "WEATHER": ["weather"],
            "TIME": ["time"],
            "CALL": ["call", "phone"],
            "MESSAGE": ["message"],
            "LIST_REMINDERS": ["reminders", "list my reminders", "show my reminders"],
        }
        for name, kws in simple.items():
            for kw in kws:
                if re.search(rf"\b{re.escape(kw)}\b", text):
                    hit(name, 1.0 + 0.1 * len(kw))

        # slot intents keyed by value
        def by_number(prefix, table, default):
            val = None
            for n in nums:
                if n in table:
                    val = n
            val = val if val is not None else default
            hit(f"{prefix}_{table[val]}", 1.4)

        if "brightness" in text or "percent" in text:
            by_number("BRIGHTNESS", {20: "20", 60: "60", 100: "100"}, 100)
        if "temperature" in text or "degrees" in text or "degree" in text:
            by_number("TEMPERATURE", {18: "18", 22: "22", 26: "26"}, 22)
        if "timer" in text or "countdown" in text or "second" in text or "minute" in text:
            if any(w in text for w in ("1 minute", "one minute", "60 second")):
                hit("TIMER_1m", 1.4)
            elif "10" in text:
                hit("TIMER_10s", 1.4)
            elif "30" in text:
                hit("TIMER_30s", 1.4)
            else:
                hit("TIMER_30s", 1.0)
        if "alarm" in text or "wake me" in text:
            if "4" in text:
                hit("ALARM_4_00AM", 1.4)
            elif "8" in text:
                hit("ALARM_8_00AM", 1.4)
            elif "9" in text:
                hit("ALARM_9_00PM", 1.4)
            else:
                hit("ALARM_8_00AM", 1.0)
        if "reminder" in text or "remind" in text:
            if "water" in text:
                hit("CREATE_REMINDER_DRINK_WATER", 1.4)
            elif "exercise" in text or "work out" in text:
                hit("CREATE_REMINDER_EXERCISE", 1.4)
            elif "study" in text:
                hit("CREATE_REMINDER_STUDY", 1.4)
        if "color" in text:
            for name, kw in [("COLOR_RED", "red"), ("COLOR_GREEN", "green"),
                             ("COLOR_BLUE", "blue"), ("COLOR_YELLOW", "yellow")]:
                if kw in text:
                    hit(name, 1.4)

        if score.sum() == 0:
            return self.placeholder_probs()
        score += 0.05 * np.random.rand(len(self.intents))
        e = np.exp(score * 3.0)
        return e / e.sum()


def action_for_intent(intent: str) -> str:
    """Map an intent to a demo action type: music | weather | popup."""
    if intent in MUSIC_INTENTS:
        return "music"
    if intent in WEATHER_INTENTS:
        return "weather"
    return "popup"
