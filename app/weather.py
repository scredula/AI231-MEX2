"""
Live weather for the demo — **Open-Meteo** (https://open-meteo.com), free and
key-less (no signup, no API key, JSON).

The fetch runs **server-side** with the stdlib (``urllib``) so the browser never
needs a key and CORS is irrelevant. Default location: **Cebu City, Philippines**.

Results are cached for ``ttl`` seconds and degrade gracefully offline: the last
good reading is returned with ``source="stale"`` (or ``"unavailable"`` if there
has never been one), so the UI always has something to draw.

CLI check:  python weather.py
"""
from __future__ import annotations

import json
import threading
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional, Tuple

# Cebu City, Philippines
CEBU = {"lat": 10.3157, "lon": 123.8854, "tz": "Asia/Manila",
        "place": "Cebu City, PH"}

# WMO weather interpretation codes (as published by Open-Meteo)
WMO: Dict[int, Tuple[str, str]] = {
    0: ("Clear sky", "\u2600\ufe0f"),
    1: ("Mainly clear", "\U0001f324\ufe0f"),
    2: ("Partly cloudy", "\u26c5"),
    3: ("Overcast", "\u2601\ufe0f"),
    45: ("Fog", "\U0001f32b\ufe0f"),
    48: ("Depositing rime fog", "\U0001f32b\ufe0f"),
    51: ("Light drizzle", "\U0001f326\ufe0f"),
    53: ("Moderate drizzle", "\U0001f326\ufe0f"),
    55: ("Dense drizzle", "\U0001f327\ufe0f"),
    56: ("Light freezing drizzle", "\U0001f327\ufe0f"),
    57: ("Dense freezing drizzle", "\U0001f327\ufe0f"),
    61: ("Slight rain", "\U0001f327\ufe0f"),
    63: ("Moderate rain", "\U0001f327\ufe0f"),
    65: ("Heavy rain", "\U0001f327\ufe0f"),
    66: ("Light freezing rain", "\U0001f327\ufe0f"),
    67: ("Heavy freezing rain", "\U0001f327\ufe0f"),
    71: ("Slight snowfall", "\U0001f328\ufe0f"),
    73: ("Moderate snowfall", "\U0001f328\ufe0f"),
    75: ("Heavy snowfall", "\u2744\ufe0f"),
    77: ("Snow grains", "\U0001f328\ufe0f"),
    80: ("Slight rain showers", "\U0001f326\ufe0f"),
    81: ("Moderate rain showers", "\U0001f327\ufe0f"),
    82: ("Violent rain showers", "\u26c8\ufe0f"),
    85: ("Slight snow showers", "\U0001f328\ufe0f"),
    86: ("Heavy snow showers", "\u2744\ufe0f"),
    95: ("Thunderstorm", "\u26c8\ufe0f"),
    96: ("Thunderstorm with slight hail", "\u26c8\ufe0f"),
    97: ("Heavy thunderstorm", "\u26c8\ufe0f"),
    99: ("Thunderstorm with heavy hail", "\u26c8\ufe0f"),
}
NIGHT: Dict[int, Tuple[str, str]] = {
    0: ("Clear night", "\U0001f319"), 1: ("Mainly clear", "\U0001f319"),
    2: ("Partly cloudy", "\u2601\ufe0f"),
}

_API = ("https://api.open-meteo.com/v1/forecast"
        "?latitude={lat}&longitude={lon}"
        "&current=temperature_2m,relative_humidity_2m,apparent_temperature,"
        "weather_code,wind_speed_10m,is_day&timezone={tz}")


class WeatherService:
    def __init__(self, lat: float = CEBU["lat"], lon: float = CEBU["lon"],
                 tz: str = CEBU["tz"], place: str = CEBU["place"],
                 ttl: float = 600.0, timeout: float = 8.0):
        self.lat, self.lon, self.tz, self.place = lat, lon, tz, place
        self.ttl, self.timeout = float(ttl), float(timeout)
        self._lock = threading.Lock()
        self._cache: Optional[Dict[str, Any]] = None
        self._fetched_at = 0.0

    # ------------------------------------------------------------------ public
    def get(self, force: bool = False) -> Dict[str, Any]:
        with self._lock:
            now = time.time()
            if not force and self._cache and (now - self._fetched_at) < self.ttl:
                return dict(self._cache, source="cache")
            try:
                data = self._fetch()
                data["source"] = "live"
                data["fetched_at"] = time.strftime("%H:%M:%S")
                self._cache, self._fetched_at = data, now
                return dict(data)
            except Exception as exc:
                if self._cache is not None:
                    return dict(self._cache, source="stale", error=str(exc))
                return {"ok": False, "place": self.place,
                        "source": "unavailable", "error": str(exc),
                        "fetched_at": time.strftime("%H:%M:%S"),
                        "description": "weather unavailable", "icon": "\u26a0\ufe0f"}

    def brief(self) -> str:
        """One-line spoken-style summary (can replace the canned weather clip)."""
        w = self.get()
        if not w.get("ok"):
            return "Sorry, I couldn't reach the weather service."
        return (f"The weather in {w['place']} is {w['description'].lower()} "
                f"at {w['temp_c']}\u00b0C, {w['humidity']}% humidity, "
                f"wind {w['wind_kmh']} km/h.")

    # ----------------------------------------------------------------- private
    def _fetch(self) -> Dict[str, Any]:
        url = _API.format(lat=self.lat, lon=self.lon,
                          tz=urllib.parse.quote(self.tz))
        req = urllib.request.Request(url, headers={"User-Agent": "MEX2-demo/1.0"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        cur = payload["current"]
        units = payload.get("current_units", {})
        code = int(cur["weather_code"])
        is_day = int(cur.get("is_day", 1))
        desc, icon = self._describe(code, is_day)
        return {
            "ok": True, "place": self.place,
            "latitude": payload.get("latitude"),
            "longitude": payload.get("longitude"),
            "timezone": payload.get("timezone"), "time": cur.get("time"),
            "temp_c": cur.get("temperature_2m"),
            "feels_c": cur.get("apparent_temperature"),
            "humidity": cur.get("relative_humidity_2m"),
            "wind_kmh": cur.get("wind_speed_10m"),
            "code": code, "description": desc, "icon": icon,
            "is_day": bool(is_day),
            "units": {"temp": units.get("temperature_2m", "\u00b0C"),
                      "wind": units.get("wind_speed_10m", "km/h")},
            "attribution": "Weather data by Open-Meteo.com",
        }

    @staticmethod
    def _describe(code: int, is_day: int = 1) -> Tuple[str, str]:
        if is_day == 0 and code in NIGHT:
            return NIGHT[code]
        return WMO.get(code, ("Unknown", "\U0001f321\ufe0f"))


if __name__ == "__main__":
    print(json.dumps(WeatherService().get(force=True), indent=2, ensure_ascii=False))

