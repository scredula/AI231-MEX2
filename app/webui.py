#!/usr/bin/env python3
"""
Browser UI ("smart-home simulator") for the Hey Mason wake-word + command demo.

The **same two ONNX models** drive this UI, so a real spoken command and a typed
or clicked command end up in the same place: the simulated device in
``device.py``. The page is plain HTML/CSS/JS (no build step, no CDN) and talks
to this Python process over Server-Sent Events (state pushes) plus small JSON
POSTs.

Run:
    ./venv/bin/python webui.py                    # http://<pi-ip>:8000
    ./venv/bin/python webui.py --port 8080 --no-mic
    ./venv/bin/python webui.py --classifier MEX2-trained --open

Endpoints
---------
GET  /                   dashboard (web/index.html)
GET  /events             SSE: state / wake-probability / prediction
GET  /state              JSON snapshot
GET  /weather[?force=1]  live Cebu weather (Open-Meteo)
GET  /models             loaded model info
POST /command            {"text": "..."} or {"intent": "COLOR_RED"}
POST /wake               simulate a wake-word trigger (opens a capture window)
POST /mic                {"on": true|false}
POST /reset              reset the simulated device
"""
from __future__ import annotations

import argparse
import json
import queue
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

from wakeword import WakeWordDetector, SlidingWakeWord            # noqa: E402
from commands import CommandClassifier                            # noqa: E402
from audio_io import MicSource                                    # noqa: E402
from actions import MusicPlayer, ClipPlayer, BeepPlayer           # noqa: E402
from device import DeviceState                                    # noqa: E402
from weather import WeatherService                                # noqa: E402
from app import resolve, load_config                              # noqa: E402
from app import _model_registry, _classifier_registry, _canon_model  # noqa: E402

WEB_DIR = APP_DIR / "web"


def speak(text: str) -> bool:
    """Best-effort TTS via espeak-ng/espeak (silent when unavailable)."""
    exe = shutil.which("espeak-ng") or shutil.which("espeak")
    if not exe:
        return False
    try:
        subprocess.Popen([exe, "-s", "165", text],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False


class WebDemo:
    """Owns the models, the simulated device, the mic loop and the SSE clients."""

    def __init__(self, cfg: Dict[str, Any], use_mic: bool = True):
        self.cfg = cfg
        self.lock = threading.RLock()
        self.clients: List[queue.Queue] = []
        self._stop = threading.Event()
        self.wake_name = "MatchboxNet"
        self.clf_name = "MEX2-31class"

        ww = cfg["wake_word"]
        cmd = cfg["command"]
        self.detector = WakeWordDetector(
            resolve(ww["model_dir"]),
            onnx_name=ww.get("onnx_name", "best_model.onnx"),
            threshold=float(ww.get("threshold", 0.15)),
            num_threads=int(ww.get("num_threads", 2)),
        )
        self.slider = SlidingWakeWord(
            self.detector, hop_ms=int(ww.get("hop_ms", 100)),
            trigger_threshold=float(ww.get("threshold", 0.15)),
            consecutive_hits=int(ww.get("consecutive_hits", 2)),
            cooldown_s=float(ww.get("cooldown_s", 2.0)),
        )
        self.classifier = CommandClassifier(
            resolve(cmd["model_dir"]), intents=cfg["intents"],
            onnx_name=cmd.get("onnx_name"), meta_name=cmd.get("meta_name"),
            num_threads=int(cmd.get("num_threads", 2)),
            unknown_threshold=cmd.get("unknown_threshold"),
        )
        self.capture_s = float(cmd.get("capture_s", 1.6))

        self.music = MusicPlayer(resolve(cfg["music"]["dir"]),
                                 volume=cfg["music"].get("volume", 0.7))
        self.clips = ClipPlayer()
        assets = cfg["assets"]
        self.weather_clip = resolve(assets["weather_clip"])
        self.beep = BeepPlayer(
            resolve(assets.get("beep_wav", "assets/audio/beep.wav")),
            volume=float(assets.get("beep_volume", 0.4)),
            enabled=bool(assets.get("beep", True)),
        )

        self.device = DeviceState(music_backend=self.music,
                                  listener=self.on_state)
        wcfg = cfg.get("weather") or {}
        self.weather = WeatherService(
            lat=float(wcfg.get("lat", 10.3157)), lon=float(wcfg.get("lon", 123.8854)),
            tz=str(wcfg.get("tz", "Asia/Manila")),
            place=str(wcfg.get("place", "Cebu City, PH")),
            ttl=float(wcfg.get("ttl_s", 600)),
        )

        self.mic = None
        self._audio_q: "queue.Queue[np.ndarray]" = queue.Queue()
        self._capturing = False
        self._capture_buf: List[np.ndarray] = []
        self._capture_until = 0.0
        self._last_prob = 0.0
        threading.Thread(target=self._audio_worker, daemon=True).start()
        if use_mic:
            self.start_mic()

    # -------------------------------------------------------------- broadcast
    def on_state(self, _state=None):
        """DeviceState listener: push a fresh snapshot to every browser."""
        self.broadcast({"type": "state", "state": self.snapshot()})

    def broadcast(self, msg: Dict[str, Any]):
        dead = []
        for q in list(self.clients):
            try:
                q.put_nowait(msg)
            except queue.Full:
                dead.append(q)
        for q in dead:
            self.sse_remove(q)

    def sse_add(self) -> "queue.Queue[Dict[str, Any]]":
        q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=128)
        with self.lock:
            self.clients.append(q)
        return q

    def sse_remove(self, q):
        with self.lock:
            if q in self.clients:
                self.clients.remove(q)

    def model_info(self) -> Dict[str, Any]:
        return {
            "wake": {"name": self.wake_name,
                     "dir": str(self.cfg["wake_word"]["model_dir"]),
                     "threshold": round(float(self.slider.trigger_threshold), 3)},
            "classifier": {"name": self.clf_name,
                           "dir": str(self.cfg["command"]["model_dir"]),
                           "real": bool(self.classifier.real),
                           "frontend": self.classifier.frontend_kind,
                           "unknown_threshold": round(float(self.classifier.unknown_threshold), 3)},
            "intents": list(self.classifier.intents),
            "weather": {"source": "Open-Meteo (keyless)", "place": self.weather.place},
        }

    def snapshot(self) -> Dict[str, Any]:
        s = self.device.snapshot()
        s["weather"]["data"] = self.weather.get()
        s["models"] = self.model_info()
        s["mic"] = self.mic is not None
        s["wake"] = {"prob": round(float(self.slider.prob), 4),
                     "threshold": round(float(self.slider.trigger_threshold), 3),
                     "listening": self._capturing}
        return s

    def prewarm(self):
        threading.Thread(target=lambda: self.weather.get(force=True),
                         daemon=True).start()

    # -------------------------------------------------------------------- mic
    def start_mic(self) -> bool:
        if self.mic is not None:
            return True
        try:
            self.mic = MicSource(self._on_audio,
                                 block_ms=int(self.cfg["audio"].get("block_ms", 32)),
                                 device=self.cfg["audio"].get("device"))
            sr = self.mic.start()
            self.broadcast({"type": "mic", "on": True, "sr": sr})
            self.device.note(f"Microphone started ({sr} Hz).")
            return True
        except Exception as exc:
            self.mic = None
            self.device.note(f"Microphone unavailable: {exc}")
            return False

    def stop_mic(self):
        if self.mic is not None:
            try:
                self.mic.stop()
            except Exception:
                pass
            self.mic = None
        self.broadcast({"type": "mic", "on": False})
        self.device.note("Microphone stopped.")

    def _on_audio(self, chunk: np.ndarray):
        self._audio_q.put(chunk)

    def _audio_worker(self):
        while not self._stop.is_set():
            try:
                chunk = self._audio_q.get(timeout=0.2)
            except queue.Empty:
                chunk = None
            if chunk is not None:
                self.slider.push(chunk)
                if self._capturing:
                    self._capture_buf.append(chunk)
                if self.slider.poll_trigger():
                    self.wake(auto=True)
                now = time.time()
                if now - self._last_prob > 0.1:
                    self._last_prob = now
                    self.broadcast({"type": "wake",
                                    "prob": round(float(self.slider.prob), 4),
                                    "threshold": round(float(self.slider.trigger_threshold), 3),
                                    "listening": self._capturing})
            if self._capturing and time.time() >= self._capture_until:
                self._capturing = False
                data, self._capture_buf = self._capture_buf, []
                wav = np.concatenate(data) if data else np.zeros(1, dtype=np.float32)
                self._recognize(wav)

    # ------------------------------------------------------------------- wake
    def wake(self, auto: bool = True):
        if self._capturing or (auto and self.mic is None):
            return
        self.broadcast({"type": "wake", "prob": round(float(self.slider.prob), 4),
                        "triggered": True, "listening": True})
        if self.beep.enabled:
            self.beep.play()
        self._capturing = True
        self._capture_buf = []
        lead = self.beep.duration_s if self.beep.enabled else 0.0
        self._capture_until = time.time() + lead + self.capture_s
        self.device.note("Wake word detected — listening for a command…")

    def simulate_wake(self) -> Dict[str, Any]:
        if self.mic is None and not self.start_mic():
            return {"ok": False, "reason": "no microphone — type a command instead"}
        self.wake(auto=False)
        return {"ok": True, "listening": True}

    # --------------------------------------------------------------- commands
    def _recognize(self, wav: np.ndarray):
        try:
            if self.classifier.real:
                label, conf, top = self.classifier.classify(wav)
            else:
                probs = self.classifier.placeholder_probs()
                top = self.classifier.top_k(probs, 10)
                label, conf = top[0]
        except Exception as exc:
            label, conf, top = "UNKNOWN", 0.0, [("UNKNOWN", 1.0)]
            self.device.note(f"Classifier error: {exc}")
        self.broadcast({"type": "prediction", "source": "mic",
                        "intent": label, "confidence": round(float(conf), 4),
                        "top": [{"intent": n, "p": round(float(p), 4)} for n, p in top]})
        self._dispatch(label, conf, source="mic")

    def command(self, text: str = None, intent: str = None) -> Dict[str, Any]:
        """Run a typed phrase (keyword matcher) or a clicked intent (exact)."""
        if intent:
            return self._dispatch(intent, 1.0, source="click")
        probs = self.classifier.placeholder_from_text(text or "")
        top = self.classifier.top_k(probs, 10)
        label, conf = top[0]
        self.broadcast({"type": "prediction", "source": "text",
                        "intent": label, "confidence": round(float(conf), 4),
                        "top": [{"intent": n, "p": round(float(p), 4)} for n, p in top]})
        return self._dispatch(label, conf, text=text, source="text")

    def _dispatch(self, label: str, conf: float, text: str = None,
                  source: str = "mic") -> Dict[str, Any]:
        if (source != "click" and label != "UNKNOWN"
                and conf < self.classifier.unknown_threshold):
            label = "UNKNOWN"
        if label == "UNKNOWN":
            self.device.note("Sorry, I didn't catch that.", intent="UNKNOWN",
                             confidence=conf, text=text, group="unknown")
            return {"ok": False, "intent": "UNKNOWN",
                    "confidence": round(float(conf), 4), "state": self.snapshot()}
        self.device.apply(label, conf, text)
        self._after_intent(label)
        return {"ok": True, "intent": label, "confidence": round(float(conf), 4),
                "state": self.snapshot()}

    def _after_intent(self, intent: str):
        if intent == "WEATHER":
            threading.Thread(target=self._refresh_weather, daemon=True).start()
        elif intent == "CALL":
            threading.Timer(8.0, lambda: self.device.update(
                lambda s: s["call"].update({"active": False}))).start()
        elif intent == "TIME":
            threading.Timer(5.0, lambda: self.device.update(
                lambda s: s["clock"].update({"highlight": False}))).start()

    def _refresh_weather(self):
        w = self.weather.get(force=True)
        self.device.update(
            lambda s: s["weather"].update({"data": w, "visible": True}))
        brief = self.weather.brief()
        self.device.note(brief)
        if w.get("ok"):
            speak(brief)
        else:
            self.clips.play(self.weather_clip)

    def reset(self) -> Dict[str, Any]:
        self.device.reset()
        return self.snapshot()

    def stop(self):
        self._stop.set()
        self.device.stop()
        self.stop_mic()

# --------------------------------------------------------------------------- #
# HTTP layer
# --------------------------------------------------------------------------- #
_MIME = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml", ".png": "image/png",
    ".ico": "image/x-icon", ".json": "application/json",
}


class Handler(BaseHTTPRequestHandler):
    """Small JSON + SSE server; the browser holds one long-lived /events stream."""

    server_version = "MEX2web/1.0"

    @property
    def demo(self) -> "WebDemo":
        return self.server.demo                      # type: ignore[attr-defined]

    def log_message(self, fmt, *args):               # keep the console quiet
        pass

    # ------------------------------------------------------------- responses
    def _json(self, obj, code: int = 200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _body(self) -> Dict[str, Any]:
        try:
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b"{}"
            return json.loads(raw.decode("utf-8") or "{}")
        except Exception:
            return {}

    def _file(self, path: Path):
        try:
            data = path.read_bytes()
        except OSError:
            self.send_error(404, "not found")
            return
        self.send_response(200)
        self.send_header("Content-Type",
                         _MIME.get(path.suffix.lower(), "application/octet-stream"))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _static(self, route: str):
        target = (WEB_DIR / route.lstrip("/")).resolve()
        root = WEB_DIR.resolve()
        if root not in target.parents and target != root:
            self.send_error(403, "forbidden")
        elif target.is_file():
            self._file(target)
        else:
            self.send_error(404, "not found")

    # -------------------------------------------------------------------- SSE
    def _sse(self):
        demo = self.demo
        q = demo.sse_add()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        self.close_connection = True

        def send(obj):
            payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.wfile.write(b"data: " + payload + b"\n\n")
            self.wfile.flush()

        try:
            send({"type": "hello", "state": demo.snapshot()})
            while True:
                try:
                    send(q.get(timeout=15))
                except queue.Empty:                  # keep proxies from closing
                    self.wfile.write(b": keep-alive\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            demo.sse_remove(q)

    # ----------------------------------------------------------------- routes
    def do_GET(self):
        try:
            self._get()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:                 # never drop the connection
            self._json({"ok": False, "error": str(exc)}, 500)

    def _get(self):
        parsed = urllib.parse.urlparse(self.path)
        route = parsed.path
        demo = self.demo
        if route == "/events":
            return self._sse()
        if route == "/state":
            return self._json(demo.snapshot())
        if route == "/models":
            return self._json(demo.model_info())
        if route == "/weather":
            qs = urllib.parse.parse_qs(parsed.query)
            force = qs.get("force", ["0"])[0].lower() in ("1", "true", "yes")
            return self._json(demo.weather.get(force=force))
        if route in ("/", "/index.html"):
            return self._file(WEB_DIR / "index.html")
        return self._static(route)

    def do_POST(self):
        try:
            self._post()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:
            self._json({"ok": False, "error": str(exc)}, 500)

    def _post(self):
        route = urllib.parse.urlparse(self.path).path
        body = self._body()
        demo = self.demo
        if route == "/command":
            text = (body.get("text") or "").strip()
            intent = body.get("intent")
            if not text and not intent:
                return self._json({"ok": False, "reason": "empty"}, 400)
            return self._json(demo.command(text=text or None, intent=intent or None))
        if route == "/wake":
            return self._json(demo.simulate_wake())
        if route == "/mic":
            on = bool(body.get("on"))
            if on:
                ok = demo.start_mic()
            else:
                demo.stop_mic()
                ok = True
            return self._json({"ok": bool(ok), "mic": demo.mic is not None})
        if route == "/reset":
            return self._json(demo.reset())
        return self._json({"ok": False, "reason": "unknown route"}, 404)

# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Hey Mason — browser smart-home simulator (same ONNX models)")
    ap.add_argument("--config", default=str(APP_DIR / "config.yaml"))
    ap.add_argument("--model", default=None,
                    help="Wake-word model: DS-CNN | TC-ResNet | MatchboxNet | VGG")
    ap.add_argument("--classifier", default=None,
                    help="Command classifier: MEX2-31class | MEX2-trained")
    ap.add_argument("--host", default="0.0.0.0", help="bind address")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-mic", action="store_true", help="run without the microphone")
    ap.add_argument("--open", action="store_true", help="open the dashboard")
    ap.add_argument("--list-models", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    registry = _model_registry(cfg)
    clf_registry = _classifier_registry(cfg)

    if args.list_models:
        print("Wake-word models (--model):")
        for k, v in registry.items():
            print(f"  {k:<12} -> {v}")
        print("Command classifiers (--classifier):")
        for k, v in clf_registry.items():
            print(f"  {k:<14} -> {v}")
        return 0

    wake_key = _canon_model(registry, args.model or
                            (cfg.get("wake_word") or {}).get("model"))
    if wake_key:
        cfg.setdefault("wake_word", {})["model_dir"] = registry[wake_key]
    elif args.model:
        print(f"Unknown --model '{args.model}'. Options: {', '.join(registry)}")
        return 2

    clf_key = _canon_model(clf_registry, args.classifier or
                           (cfg.get("command") or {}).get("model"))
    if clf_key:
        cfg.setdefault("command", {})["model_dir"] = clf_registry[clf_key]
    elif args.classifier:
        print(f"Unknown --classifier '{args.classifier}'. "
              f"Options: {', '.join(clf_registry)}")
        return 2

    try:
        demo = WebDemo(cfg, use_mic=not args.no_mic)
    except Exception as exc:
        print(f"Failed to start: {exc}")
        print("Tip: run 'python generate_assets.py' first, and check that "
              "models/wake_word/<model>/best_model.onnx exists.")
        return 1
    demo.wake_name = wake_key or (cfg.get("wake_word") or {}).get("model") or "?"
    demo.clf_name = clf_key or (cfg.get("command") or {}).get("model") or "?"
    demo.prewarm()

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True
    httpd.demo = demo                               # type: ignore[attr-defined]

    shown = "localhost" if args.host in ("0.0.0.0", "::") else args.host
    url = f"http://{shown}:{args.port}/"
    print(f"Hey Mason web UI  ->  {url}")
    print(f"  wake word  : {demo.wake_name}  ({cfg['wake_word']['model_dir']})")
    print(f"  classifier : {demo.clf_name}  ({cfg['command']['model_dir']}, "
          f"{'real ONNX' if demo.classifier.real else 'PLACEHOLDER'})")
    print(f"  microphone : {'on' if demo.mic else 'off'}")
    print("  Ctrl-C to stop.")
    if args.open:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        httpd.shutdown()
        demo.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())




