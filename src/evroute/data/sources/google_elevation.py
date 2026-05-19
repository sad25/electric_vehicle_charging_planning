# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Saadettin Yıldırım (Yıldsam Teknoloji)
"""
EN: Adds real Google Elevation to each station's detour ('dev')
    polyline in <route>_route_google.json, turning [lat,lng] into
    [lat,lng,elev]. The environment then integrates detour energy over
    the real grade profile — exactly like the main route — instead of
    assuming a flat detour. Idempotent / resumable.
TR: Her istasyonun sapma ('dev') polyline'ına gerçek Google Elevation
    ekler ([lat,lng] -> [lat,lng,elev]). Ortam, sapma enerjisini ana
    rota gibi GERÇEK eğim profili boyunca integre eder (düz kabul yok).
    Tekrar çalıştırılabilir.

Run: ./venv/bin/python -m src.fetch_dev_elevation
"""

import json
import time
from pathlib import Path

import requests

from evroute.config import get_data_dir
from evroute.data.secrets import get_secret

PROC = get_data_dir() / "processed"
ROUTES = ["istanbul_ankara", "istanbul_izmir", "ankara_antalya"]
ELEV_MAX = 256
_BACKOFF = [4, 12, 30, 60]


def _elevations(pts, gkey):
    """[[lat,lng],...] -> [elev,...] (sıra korunur)."""
    out = []
    for i in range(0, len(pts), ELEV_MAX):
        chunk = pts[i:i + ELEV_MAX]
        loc = "|".join(f"{p[0]},{p[1]}" for p in chunk)
        for attempt in range(len(_BACKOFF) + 1):
            try:
                r = requests.get(
                    "https://maps.googleapis.com/maps/api/elevation/json",
                    params={"locations": loc, "key": gkey}, timeout=30)
                j = r.json()
                if j.get("status") == "OK":
                    out.extend(round(x["elevation"], 1) for x in j["results"])
                    break
                if attempt < len(_BACKOFF):
                    time.sleep(_BACKOFF[attempt])
                    continue
                out.extend(0.0 for _ in chunk)
            except Exception:
                if attempt < len(_BACKOFF):
                    time.sleep(_BACKOFF[attempt])
                    continue
                out.extend(0.0 for _ in chunk)
        time.sleep(0.15)
    return out


def main():
    gkey = get_secret("GOOGLE_MAPS_API_KEY")
    for rk in ROUTES:
        fp = PROC / f"{rk}_route_google.json"
        if not fp.exists():
            print(f"[atla] {fp.name} yok")
            continue
        g = json.loads(fp.read_text(encoding="utf-8"))
        sts = g.get("stations", [])
        print(f"\n=== {rk} ({len(sts)} istasyon) ===")
        changed = 0
        for s in sts:
            dev = s.get("dev") or []
            if not dev:
                continue
            if len(dev[0]) >= 3:          # zaten yükseklikli
                continue
            elev = _elevations(dev, gkey)
            s["dev"] = [[p[0], p[1], elev[i] if i < len(elev) else 0.0]
                        for i, p in enumerate(dev)]
            changed += 1
            print(f"  + {s['name'][:40]:40s} {len(dev)} nokta yükseklikli")
        if changed:
            bak = PROC / f"{rk}_route_google.backup_noelev.json"
            if not bak.exists():
                bak.write_text(fp.read_text(encoding="utf-8"),
                               encoding="utf-8")
            fp.write_text(json.dumps(g, ensure_ascii=False),
                          encoding="utf-8")
            print(f"  {changed} istasyon güncellendi -> {fp.name}")
        else:
            print("  (hepsi zaten yükseklikli)")


if __name__ == "__main__":
    main()
