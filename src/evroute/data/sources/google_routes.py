# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Saadettin Yıldırım (Yıldsam Teknoloji)
"""
EN: Single-source Google routing for a route. Fetches the real A->B
    route (dense step polyline) AND, for every station C, the A->C->B
    waypoint route; extracts the extra cost (km/min) and the deviation
    geometry (only the part that leaves the A->B route near C). One
    consistent source: the main line and every detour come from Google,
    so they overlap perfectly except at the genuine station access.
TR: Bir rota için tek-kaynak Google rotalama. Gerçek A->B rotasını
    (yoğun step polyline) VE her istasyon C için A->C->B waypoint
    rotasını çeker; fazladan maliyeti (km/dk) ve sapma geometrisini
    (yalnız C civarında A->B'den ayrılan kısım) çıkarır. Tek tutarlı
    kaynak: ana hat ve tüm sapmalar Google'dan; C erişimi dışında
    birebir çakışırlar.

Output / Çıktı: data/processed/<route>_route_google.json
Run: ./venv/bin/python -m src.fetch_route_google istanbul_ankara
"""

import json
import math
import sys
import time
from pathlib import Path

import requests

from evroute.config import get_data_dir
from evroute.data.secrets import get_secret

RAW = get_data_dir() / "raw"
PROC = get_data_dir() / "processed"
DEV_THRESH_KM = 0.30          # A->B'den bu kadar uzaksa "sapma" sayılır
_BACKOFF = [4, 12, 30, 60]


def _decode(enc):
    pts, i, lat, lng = [], 0, 0, 0
    while i < len(enc):
        for k in range(2):
            shift = result = 0
            while True:
                b = ord(enc[i]) - 63
                i += 1
                result |= (b & 0x1f) << shift
                shift += 5
                if b < 0x20:
                    break
            d = ~(result >> 1) if (result & 1) else (result >> 1)
            if k == 0:
                lat += d
            else:
                lng += d
        pts.append([round(lat / 1e5, 6), round(lng / 1e5, 6)])
    return pts


def _hav(a, b):
    R = 6371.0
    p = math.radians
    x = (math.sin(p(b[0] - a[0]) / 2) ** 2 +
         math.cos(p(a[0])) * math.cos(p(b[0])) *
         math.sin(p(b[1] - a[1]) / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(x))


def _directions(params, key):
    for attempt in range(len(_BACKOFF) + 1):
        try:
            j = requests.get(
                "https://maps.googleapis.com/maps/api/directions/json",
                params={**params, "key": key}, timeout=30).json()
            st = j.get("status")
            if st == "OK":
                leg = j["routes"][0]["legs"]
                km = sum(x["distance"]["value"] for x in leg) / 1000.0
                mn = sum(x["duration"]["value"] for x in leg) / 60.0
                path = []
                for x in leg:
                    for s in x["steps"]:
                        seg = _decode(s["polyline"]["points"])
                        if path and seg and path[-1] == seg[0]:
                            seg = seg[1:]
                        path += seg
                return km, mn, path
            if attempt < len(_BACKOFF):
                time.sleep(_BACKOFF[attempt])
                continue
            raise RuntimeError(f"Directions {st} {j.get('error_message','')}")
        except RuntimeError:
            raise
        except Exception as e:
            if attempt < len(_BACKOFF):
                time.sleep(_BACKOFF[attempt])
                continue
            raise RuntimeError(f"Directions ağ hatası: {e}")


def _cum(path):
    c = [0.0]
    for i in range(1, len(path)):
        c.append(c[-1] + _hav(path[i - 1], path[i]))
    return c


def build(route_key: str):
    key = get_secret("GOOGLE_MAPS_API_KEY")
    geom = json.load(open(RAW / f"{route_key}_route_geometry.json",
                          encoding="utf-8"))["coordinates"]
    A, B = geom[0], geom[-1]
    stations = json.load(open(PROC / f"{route_key}_stations_ocm.json",
                              encoding="utf-8"))

    print(f"=== {route_key}: Google A->B ===")
    ab_km, ab_min, ab_path = _directions(
        {"origin": f"{A[0]},{A[1]}", "destination": f"{B[0]},{B[1]}"}, key)
    ab_cum = _cum(ab_path)
    print(f"  A->B: {ab_km:.1f} km / {ab_min:.0f} dk / {len(ab_path)} nokta")
    ab_s = ab_path[::6]  # hızlı uzaklık testi için seyrek kopya

    out_stations = []
    for n, c in enumerate(stations, 1):
        clat, clng = c["lat"], c["lng"]
        wkm, wmin, wpath = _directions(
            {"origin": f"{A[0]},{A[1]}", "destination": f"{B[0]},{B[1]}",
             "waypoints": f"{clat},{clng}"}, key)
        time.sleep(0.25)
        # Sapma geometrisi: A->C->B'nin A->B'den ayrılan kısmı (C civarı)
        dev = [p for p in wpath
               if min(_hav(p, g) for g in ab_s) > DEV_THRESH_KM]
        rec = {
            "name": c["name"], "road_km": c.get("road_km"),
            "power_kw": c.get("power_kw", 120), "lat": clat, "lng": clng,
            "extra_km": round(wkm - ab_km, 2),
            "extra_min": round(wmin - ab_min, 1),
            "dev": dev,                      # temiz gerçek sapma (görsel)
        }
        out_stations.append(rec)
        print(f"  [{n}/{len(stations)}] {c['name'][:34]:34s} "
              f"+{rec['extra_km']:5.1f}km/+{rec['extra_min']:4.0f}dk "
              f"sapma_nokta={len(dev)}")

    fp = PROC / f"{route_key}_route_google.json"
    fp.write_text(json.dumps({
        "route_key": route_key,
        "ab_km": round(ab_km, 2), "ab_min": round(ab_min, 1),
        "ab_path": ab_path, "ab_cum_km": [round(x, 4) for x in ab_cum],
        "stations": out_stations,
    }, ensure_ascii=False), encoding="utf-8")
    print(f"Kaydedildi: {fp}")


if __name__ == "__main__":
    rk = sys.argv[1] if len(sys.argv) > 1 else "istanbul_ankara"
    build(rk)
