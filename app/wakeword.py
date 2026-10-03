"""
Wake-word detector using ONNX Runtime (runs on Raspberry Pi 5).

Sliding-window "Hey Mason" detection: a 1 s window is scored every hop_ms and
debounced so a single utterance fires once.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

from features import LogMelFrontend


class WakeWordDetector:
    def __init__(
        self,
        model_dir: str | Path,
        onnx_name: str = "best_model.onnx",
        threshold: float = 0.5,
        providers: Optional[list] = None,
        num_threads: int = 2,
    ):
        import onnxruntime as ort

        model_dir = Path(model_dir)
        self.frontend = LogMelFrontend(model_dir)
        self.threshold = float(threshold)

        so = ort.SessionOptions()
        so.intra_op_num_threads = num_threads
        so.inter_op_num_threads = 1
        # Cap CPU threads to reduce thermal load during always-on inference.
        self.sess = ort.InferenceSession(
            str(model_dir / onnx_name),
            sess_options=so,
            providers=providers or ["CPUExecutionProvider"],
        )
        self.input_name = self.sess.get_inputs()[0].name

    def predict(self, wav_16k: np.ndarray) -> float:
        """Probability that a 1 s window contains the wake word."""
        spec = self.frontend(wav_16k)            # (n_mels, T)
        x = spec[None, None, :, :].astype(np.float32)   # (1, 1, n_mels, T)
        out = self.sess.run(None, {self.input_name: x})[0]
        logit = float(np.asarray(out).reshape(-1)[0])
        return 1.0 / (1.0 + np.exp(-logit))      # sigmoid


class SlidingWakeWord:
    """
    Maintains a rolling buffer and scores a wake-word window every `hop_ms`.
    Returns a smoothed probability and a boolean "triggered" (debounced).
    """

    def __init__(
        self,
        detector: WakeWordDetector,
        hop_ms: int = 100,
        trigger_threshold: Optional[float] = None,
        release_threshold: float = 0.3,
        consecutive_hits: int = 2,
        cooldown_s: float = 2.0,
        smooth: float = 0.6,
    ):
        self.det = detector
        self.sample_rate = detector.frontend.sample_rate
        self.win = detector.frontend.n_samples
        self.hop = int(self.sample_rate * hop_ms / 1000)
        self.trigger_threshold = trigger_threshold if trigger_threshold is not None else detector.threshold
        self.release_threshold = release_threshold
        self.consecutive_hits = consecutive_hits
        self.cooldown_s = cooldown_s
        self.smooth = smooth

        self._buf = np.zeros(0, dtype=np.float32)
        self._since_last = 0
        self._hits = 0
        self._last_trigger = -1e9
        self.prob = 0.0
        self.triggered = False

    def push(self, chunk_16k: np.ndarray) -> float:
        """Add new 16 kHz samples; run inference at the configured hop. Returns prob."""
        self._buf = np.concatenate([self._buf, np.asarray(chunk_16k, dtype=np.float32)])
        # keep only the last window (plus one hop) of audio
        if self._buf.shape[0] > self.win + self.hop:
            self._buf = self._buf[-(self.win + self.hop):]

        self._since_last += chunk_16k.shape[0]
        if self._since_last >= self.hop and self._buf.shape[0] >= self.win:
            self._since_last = 0
            window = self._buf[-self.win:]
            raw = self.det.predict(window)
            self.prob = self.smooth * self.prob + (1 - self.smooth) * raw

            now = time.time()
            if self.prob >= self.trigger_threshold:
                self._hits += 1
            else:
                self._hits = 0

            if (self._hits >= self.consecutive_hits
                    and (now - self._last_trigger) > self.cooldown_s):
                self._last_trigger = now
                self._hits = 0
                self.triggered = True
        return self.prob

    def poll_trigger(self) -> bool:
        """Return True once for each debounced trigger and clear the flag."""
        if self.triggered:
            self.triggered = False
            return True
        return False
