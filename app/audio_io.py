"""
Audio input for the demo: live microphone (sounddevice) and a file feeder.

Live mic audio is resampled to 16 kHz mono and delivered to a callback as
float32 NumPy chunks. On the Raspberry Pi 5 use a USB mic.
"""
from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from features import resample_to_16k, read_wav_mono_16k

TARGET_SR = 16000


def list_input_devices() -> List[str]:
    try:
        import sounddevice as sd
        return [f"{i}: {d['name']}" for i, d in enumerate(sd.query_devices())
                if d.get("max_input_channels", 0) > 0]
    except Exception as e:
        return [f"(sounddevice unavailable: {e})"]


class MicSource:
    """Continuously captures microphone audio -> 16 kHz float32 chunks."""

    def __init__(
        self,
        on_audio: Callable[[np.ndarray], None],
        block_ms: int = 32,
        device: Optional[int] = None,
    ):
        self.on_audio = on_audio
        self.block_ms = block_ms
        self.device = device
        self._stream = None
        self._sr = TARGET_SR
        self._resample = False
        self._stop = threading.Event()

    def _callback(self, indata, frames, time_info, status):
        if self._stop.is_set():
            return
        mono = indata[:, 0].copy()
        if self._resample:
            mono = resample_to_16k(mono, self._sr, TARGET_SR)
        self.on_audio(mono)

    def start(self):
        import sounddevice as sd
        block = int(TARGET_SR * self.block_ms / 1000)
        # Prefer native 16 kHz; fall back to device default + software resample.
        try:
            self._sr = TARGET_SR
            self._resample = False
            self._stream = sd.InputStream(
                samplerate=TARGET_SR, channels=1, dtype="float32",
                blocksize=block, device=self.device, callback=self._callback,
            )
            self._stream.start()
        except Exception:
            info = sd.query_devices(self.device, "input")
            self._sr = int(info["default_samplerate"])
            self._resample = True
            blk = int(self._sr * self.block_ms / 1000)
            self._stream = sd.InputStream(
                samplerate=self._sr, channels=1, dtype="float32",
                blocksize=blk, device=self.device, callback=self._callback,
            )
            self._stream.start()
        return self._sr

    def stop(self):
        self._stop.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None


class FileFeeder:
    """Streams a wav file as 16 kHz chunks in a background thread (for tests)."""

    def __init__(self, path: str | Path, on_audio: Callable[[np.ndarray], None],
                 block_ms: int = 32, realtime: bool = True, loop: bool = False):
        self.path = Path(path)
        self.on_audio = on_audio
        self.block = int(TARGET_SR * block_ms / 1000)
        self.realtime = realtime
        self.loop = loop
        self._thread = None
        self._stop = threading.Event()

    def start(self):
        def run():
            wav = read_wav_mono_16k(self.path, TARGET_SR)
            while not self._stop.is_set():
                for i in range(0, len(wav), self.block):
                    if self._stop.is_set():
                        break
                    self.on_audio(wav[i:i + self.block])
                    if self.realtime:
                        time.sleep(self.block / TARGET_SR)
                if not self.loop:
                    break
        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
