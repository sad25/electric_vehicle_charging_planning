# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Saadettin Yıldırım (Yıldsam Teknoloji)
"""
EN: Visual-only detour geometry. For every station, queries Google
    Directions FROM the route exit point (the station's projection on
    the route) TO the station and back, decoding the dense per-STEP
    polylines (~50 m spacing) so the drawn detour follows the real
    access road smoothly. Stored as vis_out / vis_back on the station
    JSON. Physics fields (detour_km, detour_min, detour_out_path,
    detour_back_path) are NOT touched -> trained models stay valid.
TR: Yalnızca görsel sapma geometrisi. Her istasyon için Google
    Directions'ı rota çıkış noktasından (istasyonun rota üzerindeki
    izdüşümü) istasyona ve geri sorgular; yoğun ADIM (step) polyline'ını
    (~50 m) çözer ki çizilen sapma gerçek erişim yolunu pürüzsüz izlesin.
    İstasyon JSON'una vis_out / vis_back olarak yazılır. Fizik alanları
    (detour_km, detour_min, detour_out_path, detour_back_path) DEĞİŞMEZ
    -> eğitilmiş modeller geçerli kalır, yeniden eğitim gerekmez.

Run / Çalıştırma:
    ./venv/bin/python -m src.fetch_detour_vis
"""

import json
import time
from pathlib import Path

import requests

from evroute.config import get_data_dir
from evroute.data.secrets import get_secret

PROC = get_data_dir() / "processed"
ROUTES = ["istanbul_ankara", "istanbul_izmir", "ankara_antalya"]
_BACKOFF = [4, 12, 30, 60]


def _decode(enc: str):
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


def _steps_polyline(o_lat, o_lng, d_lat, d_lng, gkey):
    """Dense path = all step polylines concatenated. None on failure."""
    params = {"origin": f"{o_lat},{o_lng}",
              "destination": f"{d_lat},{d_lng}", "key": gkey}
    for attempt in range(len(_BACKOFF) + 1):
        try:
            r = requests.get(
                "https://maps.googleapis.com/maps/api/directions/json",
                params=params, timeout=25)
            j = r.json()
            st = j.get("status")
            if st == "OK":
                leg = j["routes"][0]["legs"][0]
                path = []
                for s in leg["steps"]:
                    seg = _decode(s["polyline"]["points"])
                    if path and seg and path[-1] == seg[0]:
                        seg = seg[1:]
                    path += seg
                return path
            if st == "ZERO_RESULTS":
                return None
            if attempt < len(_BACKOFF):
                time.sleep(_BACKOFF[attempt])
                continue
            print(f"    Directions {st} {j.get('error_message','')}")
            return None
        except Exception as e:
            if attempt < len(_BACKOFF):
                time.sleep(_BACKOFF[attempt])
                continue
            print(f"    Directions ağ hatası: {e}")
            return None
    return None


def main():
    gkey = get_secret("GOOGLE_MAPS_API_KEY")
    for route in ROUTES:
        fp = PROC / f"{route}_stations_ocm.json"
        stations = json.load(open(fp, encoding="utf-8"))
        print(f"\n=== {route} ({len(stations)} istasyon) ===")
        changed = 0
        for s in stations:
            if s.get("vis_out") and s.get("vis_back"):
                continue  # resumable: zaten var
            ex = (s.get("exit_lat"), s.get("exit_lng"))
            st = (s.get("lat"), s.get("lng"))
            if None in ex or None in st:
                continue
            out = _steps_polyline(ex[0], ex[1], st[0], st[1], gkey)
            time.sleep(0.2)
            back = _steps_polyline(st[0], st[1], ex[0], ex[1], gkey)
            time.sleep(0.2)
            # En az çıkış ve istasyon uçları olsun
            s["vis_out"] = out or [[ex[0], ex[1]], [st[0], st[1]]]
            s["vis_back"] = back or [[st[0], st[1]], [ex[0], ex[1]]]
            changed += 1
            print(f"  + {s['name'][:40]:40s} out={len(s['vis_out'])} "
                  f"back={len(s['vis_back'])} nokta")
        if changed:
            bak = PROC / f"{route}_stations_ocm.backup_vis.json"
            if not bak.exists():
                bak.write_text(fp.read_text(encoding="utf-8"),
                               encoding="utf-8")
            fp.write_text(json.dumps(stations, ensure_ascii=False, indent=2),
                          encoding="utf-8")
            print(f"  {changed} istasyon güncellendi -> {fp.name}")
        else:
            print("  (hepsi zaten vis içeriyor)")


if __name__ == "__main__":
    main()
