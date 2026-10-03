"""
Simulated smart-home device for the MEX2 demo.

Turns the 31 command intents into observable **device state** so the web UI can
animate a light bulb, a thermostat, a timer, alarms, reminders, a music player
and a live weather card.

Pure Python (no third-party imports) so it can be unit-tested headlessly and
reused by both the Tkinter app and the browser UI.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, List, Optional

# ---------------------------------------------------------------- effect tables
LIGHT_ONOFF = {"LIGHT_ON": True, "LIGHT_OFF": False}
BRIGHTNESS = {"BRIGHTNESS_20": 20, "BRIGHTNESS_60": 60, "BRIGHTNESS_100": 100}
COLOR = {
    "COLOR_RED": ("#ff4136", "red"),
    "COLOR_GREEN": ("#2ecc40", "green"),
    "COLOR_BLUE": ("#0074d9", "blue"),
}
TEMPERATURE = {"TEMPERATURE_18": 18, "TEMPERATURE_22": 22, "TEMPERATURE_26": 26}
TIMERS = {"TIMER_10s": (10, "10 s"), "TIMER_30s": (30, "30 s"), "TIMER_1m": (60, "1 min")}
ALARMS = {
    "ALARM_6_00AM": ("06:00", "6:00 AM"),
    "ALARM_8_00AM": ("08:00", "8:00 AM"),
    "ALARM_9_00PM": ("21:00", "9:00 PM"),
}
REMINDERS = {
    "CREATE_REMINDER_DRINK_WATER": "drink water",
    "CREATE_REMINDER_EXERCISE": "exercise",
    "CREATE_REMINDER_STUDY": "study",
}
MUSIC = {"PLAY_MUSIC", "NEXT", "PAUSE", "STOP", "VOLUME_UP", "VOLUME_DOWN"}

WARM_WHITE = "#ffd479"
MAX_LOG = 60


def _now() -> str:
    return time.strftime("%H:%M:%S")


class DeviceState:
    """Thread-safe simulated device. ``apply(intent)`` mutates it and notifies.

    ``music_backend`` (optional) is any object exposing the ``MusicPlayer`` API
    (``play``/``next``/``pause_toggle``/``stop``/``volume_up``/``volume_down``);
    when supplied, music intents drive real audio as well as the on-screen state.
    """

    def __init__(self, music_backend=None, listener: Optional[Callable] = None):
        self._lock = threading.RLock()
        self._listeners: List[Callable[[Dict[str, Any]], None]] = []
        if listener is not None:
            self._listeners.append(listener)
        self.music_backend = music_backend
        self._stop = threading.Event()
        self._timer_thread = None
        self.reset(notify=False)
        self._start_ticker()

    # ------------------------------------------------------------- listeners
    def add_listener(self, cb: Callable[[Dict[str, Any]], None]):
        with self._lock:
            self._listeners.append(cb)

    def _notify(self):
        state = self.snapshot()
        for cb in list(self._listeners):
            try:
                cb(state)
            except Exception:            # a dead SSE client must not break us
                pass

    # ----------------------------------------------------------------- state
    def reset(self, notify: bool = True):
        with self._lock:
            self.s: Dict[str, Any] = {
                "lights": {"on": False, "brightness": 100,
                           "color": WARM_WHITE, "color_name": "warm white"},
                "thermostat": {"target_c": 22},
                "timer": {"duration_s": 0, "remaining_s": 0,
                          "running": False, "label": ""},
                "alarms": [],
                "reminders": [],
                "music": {"playing": False, "paused": False,
                          "track": None, "volume": 0.7},
                "weather": {"visible": False, "place": "Cebu City"},
                "clock": {"highlight": False},
                "call": {"active": False},
                "message": {"unread": 0, "last": None},
                "last": {"intent": None, "confidence": 0.0, "text": None,
                         "ts": None, "group": None},
                "log": [],
            }
            self._log("Device reset.")
        if notify:
            self._notify()

    def _log(self, msg: str):
        """Append to the on-screen action log (caller holds the lock)."""
        self.s["log"].append({"t": _now(), "msg": msg})
        del self.s["log"][:-MAX_LOG]

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "lights": dict(self.s["lights"]),
                "thermostat": dict(self.s["thermostat"]),
                "timer": dict(self.s["timer"]),
                "alarms": [dict(a) for a in self.s["alarms"]],
                "reminders": [dict(r) for r in self.s["reminders"]],
                "music": dict(self.s["music"]),
                "weather": dict(self.s["weather"]),
                "clock": {"now": time.strftime("%H:%M:%S"),
                          "highlight": self.s["clock"]["highlight"]},
                "call": dict(self.s["call"]),
                "message": dict(self.s["message"]),
                "last": dict(self.s["last"]),
                "log": self.s["log"][-MAX_LOG:],
            }

    # ---------------------------------------------------------------- ticker
    def _start_ticker(self):
        def run():
            while not self._stop.is_set():
                time.sleep(0.2)
                with self._lock:
                    t = self.s["timer"]
                    if not t["running"]:
                        continue
                    rem = t["remaining_s"] - 0.2
                    if rem <= 0:
                        t["remaining_s"] = 0
                        t["running"] = False
                        self._log(f"\u23f1 Timer finished ({t['label']}).")
                        finished = True
                    else:
                        t["remaining_s"] = round(rem, 1)
                        finished = False
                if finished:
                    self._notify()

    def stop(self):
        self._stop.set()

    # ----------------------------------------------------------------- apply
    def apply(self, intent: str, confidence: float = 0.0,
              text: Optional[str] = None) -> Dict[str, Any]:
        """Apply an intent, log a sentence, notify listeners, return the state."""
        with self._lock:
            group, detail = self._apply_locked(intent)
            self.s["last"] = {
                "intent": intent, "confidence": round(float(confidence), 4),
                "text": text, "ts": _now(), "group": group,
            }
            self._log(f"{intent} \u2192 {detail}")
            state = self.snapshot()
        for cb in list(self._listeners):
            try:
                cb(state)
            except Exception:
                pass
        return state

    def update(self, mutator: Callable[[Dict[str, Any]], None]) -> Dict[str, Any]:
        """Mutate the raw state under the lock, then notify listeners."""
        with self._lock:
            mutator(self.s)
            state = self.snapshot()
        for cb in list(self._listeners):
            try:
                cb(state)
            except Exception:
                pass
        return state

    def note(self, msg: str, intent: Optional[str] = None,
             confidence: float = 0.0, text: Optional[str] = None,
             group: Optional[str] = None) -> Dict[str, Any]:
        """Log a free-form sentence (optionally recording the last command)."""
        def mutate(s):
            if intent is not None:
                s["last"] = {"intent": intent,
                             "confidence": round(float(confidence), 4),
                             "text": text, "ts": _now(), "group": group}
            s["log"].append({"t": _now(), "msg": msg})
            del s["log"][:-MAX_LOG]
        return self.update(mutate)

    def _apply_locked(self, intent: str):
        lights = self.s["lights"]
        if intent in LIGHT_ONOFF:
            lights["on"] = LIGHT_ONOFF[intent]
            return "light", "lights on" if lights["on"] else "lights off"
        if intent in BRIGHTNESS:
            lights["on"] = True
            lights["brightness"] = BRIGHTNESS[intent]
            return "light", f"brightness {lights['brightness']}%"
        if intent in COLOR:
            lights["on"] = True
            lights["color"], lights["color_name"] = COLOR[intent]
            return "light", f"colour {lights['color_name']}"
        if intent in TEMPERATURE:
            self.s["thermostat"]["target_c"] = TEMPERATURE[intent]
            return "climate", f"thermostat {TEMPERATURE[intent]}\u00b0C"
        if intent in TIMERS:
            secs, label = TIMERS[intent]
            self.s["timer"] = {"duration_s": secs, "remaining_s": float(secs),
                               "running": True, "label": label}
            return "timer", f"timer {label}"
        if intent in ALARMS:
            hhmm, label = ALARMS[intent]
            if not any(a["time"] == hhmm for a in self.s["alarms"]):
                self.s["alarms"].append({"time": hhmm, "label": label,
                                         "enabled": True})
            return "alarm", f"alarm armed for {label}"
        if intent in REMINDERS:
            what = REMINDERS[intent]
            self.s["reminders"].append({"text": what, "done": False})
            return "reminder", f"reminder \u201c{what}\u201d added"
        if intent == "LIST_REMINDERS":
            return "reminder", f"{len(self.s['reminders'])} reminder(s)"
        if intent == "WEATHER":
            self.s["weather"]["visible"] = True
            return "weather", "showing Cebu weather"
        if intent == "TIME":
            self.s["clock"]["highlight"] = True
            return "time", f"the time is {time.strftime('%H:%M')}"
        if intent == "CALL":
            self.s["call"]["active"] = True
            return "call", "placing a call\u2026"
        if intent == "MESSAGE":
            m = self.s["message"]
            m["unread"] += 1
            m["last"] = "Sure, I'll be there in 10 minutes."
            return "message", f"{m['unread']} unread message(s)"
        if intent in MUSIC:
            return self._apply_music(intent)
        return "unknown", "no mapped action"

    def _apply_music(self, intent: str):
        m = self.s["music"]
        msg = None
        if self.music_backend is not None:
            fn = {"PLAY_MUSIC": "play", "NEXT": "next", "PAUSE": "pause_toggle",
                  "STOP": "stop", "VOLUME_UP": "volume_up",
                  "VOLUME_DOWN": "volume_down"}.get(intent)
            if fn:
                try:
                    msg = getattr(self.music_backend, fn)()
                except Exception as e:            # no audio device is fine
                    msg = f"audio unavailable ({e})"
        if intent in ("PLAY_MUSIC", "NEXT"):
            m["playing"], m["paused"] = True, False
            cur = getattr(self.music_backend, "current", None) if self.music_backend else None
            if cur:
                m["track"] = str(cur).replace("\\", "/").split("/")[-1]
            elif not m.get("track"):
                m["track"] = "demo track"
        elif intent == "PAUSE":
            m["paused"] = not m["paused"]
        elif intent == "STOP":
            m["playing"], m["paused"] = False, False
        elif intent == "VOLUME_UP":
            m["volume"] = round(min(1.0, m["volume"] + 0.1), 2)
        elif intent == "VOLUME_DOWN":
            m["volume"] = round(max(0.0, m["volume"] - 0.1), 2)
        return "music", (msg or f"volume {int(m['volume'] * 100)}%")

