"""
Demo actions: music playback (pygame), weather clip, and JPG popups.

All are defensively written so they can be imported/instantiated even without
an audio device or display (useful for testing).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional

AUDIO_EXTS = (".mp3", ".ogg", ".wav", ".flac", ".m4a")


class MusicPlayer:
    """Simple random-file music player with play/pause/next/stop/volume."""

    def __init__(self, music_dir: str | Path = "music", volume: float = 0.7):
        self.music_dir = Path(music_dir)
        self.volume = float(volume)
        self.files: List[Path] = sorted(
            p for p in self.music_dir.glob("*") if p.suffix.lower() in AUDIO_EXTS
        ) if self.music_dir.exists() else []
        self.current: Optional[Path] = None
        self.playing = False
        self.paused = False
        self._ok = False
        self._init_mixer()

    def _init_mixer(self):
        try:
            import pygame
            pygame.mixer.init()
            pygame.mixer.music.set_volume(self.volume)
            self._ok = True
        except Exception as e:
            print(f"[MusicPlayer] audio unavailable: {e}")
            self._ok = False

    def _pick(self) -> Optional[Path]:
        if not self.files:
            return None
        if self.current is not None and len(self.files) > 1:
            choices = [f for f in self.files if f != self.current]
            return random.choice(choices)
        return random.choice(self.files)

    def play(self) -> str:
        if not self._ok:
            return "no audio device"
        f = self._pick()
        if f is None:
            return "no music files in /music"
        import pygame
        pygame.mixer.music.load(str(f))
        pygame.mixer.music.play()
        self.current, self.playing, self.paused = f, True, False
        return f"playing {f.name}"

    def next(self) -> str:
        return self.play()

    def pause_toggle(self) -> str:
        if not self._ok or self.current is None:
            return "nothing playing"
        import pygame
        if self.paused:
            pygame.mixer.music.unpause()
            self.paused = False
            return "resumed"
        pygame.mixer.music.pause()
        self.paused = True
        return "paused"

    def stop(self) -> str:
        if not self._ok:
            return "no audio device"
        import pygame
        pygame.mixer.music.stop()
        self.playing, self.paused, self.current = False, False, None
        return "stopped"

    def _set_vol(self, v: float) -> str:
        self.volume = max(0.0, min(1.0, v))
        if self._ok:
            import pygame
            pygame.mixer.music.set_volume(self.volume)
        return f"volume {int(self.volume * 100)}%"

    def volume_up(self) -> str:
        return self._set_vol(self.volume + 0.1)

    def volume_down(self) -> str:
        return self._set_vol(self.volume - 0.1)


class ClipPlayer:
    """Plays a single audio clip (e.g. weather.mp3)."""

    def __init__(self):
        self._ok = False
        try:
            import pygame
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            self._ok = True
        except Exception:
            self._ok = False

    def play(self, path: str | Path) -> str:
        path = Path(path)
        if not self._ok:
            return f"(audio unavailable) would play {path.name}"
        if not path.exists():
            return f"missing {path.name}"
        import pygame
        pygame.mixer.music.load(str(path))
        pygame.mixer.music.play()
        return f"playing {path.name}"


class BeepPlayer:
    """Short non-blocking "listening" cue played when the wake word fires.

    Prefers a pre-generated WAV (e.g. assets/audio/beep.wav); if it is missing
    or fails to load, a short two-tone beep is synthesized in memory. It plays
    on a dedicated pygame ``Sound`` channel, so it never interrupts music
    playback (which runs on ``pygame.mixer.music``).
    """

    def __init__(self, beep_path: str | Path | None = None, volume: float = 0.4,
                 enabled: bool = True, sample_rate: int = 22050):
        self.enabled = bool(enabled)
        self.volume = float(volume)
        self.duration_s = 0.0
        self._ok = False
        self._sound = None
        if not self.enabled:
            return
        try:
            import pygame
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            self._ok = True
            snd = None
            if beep_path is not None and Path(beep_path).exists():
                try:
                    snd = pygame.mixer.Sound(str(beep_path))
                except Exception as e:
                    print(f"[BeepPlayer] could not load {beep_path}: {e}")
            if snd is None:
                snd = self._synth(sample_rate)
            snd.set_volume(self.volume)
            self._sound = snd
            self.duration_s = float(snd.get_length())
        except Exception as e:
            print(f"[BeepPlayer] audio unavailable: {e}")
            self._ok = False

    @staticmethod
    def _synth(sample_rate: int = 22050):
        """Two-tone A5 -> E6 'ding' matching the generated asset."""
        import numpy as np
        import pygame
        init = pygame.mixer.get_init() or (sample_rate, -16, 2)
        mixer_freq, channels = init[0], init[2]
        parts = []
        for f in (880.0, 1318.51):
            n = max(1, int(0.09 * mixer_freq))
            t = np.arange(n) / mixer_freq
            parts.append(np.sin(2 * np.pi * f * t))
        x = np.concatenate(parts)
        a = max(1, int(0.008 * mixer_freq))
        r = max(1, int(0.05 * mixer_freq))
        env = np.ones(x.shape[0])
        env[:a] = np.linspace(0, 1, a)
        env[-r:] = np.linspace(1, 0, r)
        pcm = (np.clip(x * env * 0.5, -1, 1) * 32767).astype(np.int16)
        if channels >= 2:
            pcm = np.repeat(pcm[:, None], channels, axis=1)
        return pygame.sndarray.make_sound(np.ascontiguousarray(pcm))

    def play(self) -> str:
        if not self.enabled:
            return "beep disabled"
        if not self._ok or self._sound is None:
            return "(audio unavailable) beep"
        try:
            self._sound.play()
            return "beep"
        except Exception as e:
            return f"beep failed: {e}"


class PopupManager:
    """Shows a pre-generated JPG in a Toplevel window that auto-closes."""

    def __init__(self, root, jpg_dir: str | Path = "assets/jpgs", duration_ms: int = 2500):
        self.root = root
        self.jpg_dir = Path(jpg_dir)
        self.duration_ms = duration_ms
        self._photo = None  # keep reference

    def path_for(self, intent: str) -> Path:
        return self.jpg_dir / f"{intent}.jpg"

    def show(self, intent: str) -> str:
        p = self.path_for(intent)
        if not p.exists():
            return f"no image for {intent}"
        try:
            import tkinter as tk
            from PIL import Image, ImageTk
        except Exception as e:
            return f"popup unavailable: {e}"

        img = Image.open(p)
        img.thumbnail((520, 380))
        top = tk.Toplevel(self.root)
        top.title(intent)
        top.attributes("-topmost", True)
        photo = ImageTk.PhotoImage(img)
        lbl = tk.Label(top, image=photo)
        lbl.image = photo
        lbl.pack()
        top.after(self.duration_ms, top.destroy)
        return f"popup: {intent}"
