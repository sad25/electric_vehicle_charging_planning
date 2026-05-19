# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Saadettin Yıldırım (Yıldsam Teknoloji)
"""
EN: Builds the charging-station dataset for each route by combining Open
    Charge Map with Google Directions and Google Elevation.
TR: Her rota için şarj istasyonu veri setini, Open Charge Map'i Google
    Directions ve Google Elevation ile birleştirerek üretir.

Pipeline / Akış:
  1. EN: Real DC fast-charging stations from Open Charge Map, sampled
        densely along the route polyline.
     TR: Rota polyline'ı boyunca yoğun örneklenmiş, Open Charge Map'ten
        gerçek DC hızlı şarj istasyonları.
  2. EN: Filters — DC >= 50 kW, operational, at least one CCS connector,
        publicly accessible (private / "Notice Required" excluded).
     TR: Filtreler — DC >= 50 kW, operasyonel, en az bir CCS konnektör,
        herkese açık (özel / "Notice Required" hariç).
  3. EN: Each station is projected onto the route polyline to obtain the
        highway exit point (road_km) and straight-line offset.
     TR: Her istasyon rota polyline'ına projekte edilerek otoyol çıkış
        noktası (road_km) ve kuş uçuşu sapma elde edilir.
  4. EN: Google Directions, two asymmetric legs — exit->station (outbound)
        and station->exit (return) — with traffic-aware duration.
     TR: Google Directions, iki asimetrik bacak — çıkış->istasyon (gidiş)
        ve istasyon->çıkış (dönüş) — trafikli süreyle.
  5. EN: Google Elevation per polyline point so the environment can model
        grade-aware energy along the detour too.
     TR: Sapma boyunca da eğimli enerji modellenebilsin diye polyline'ın
        her noktası için Google Elevation.
  6. EN: Prior data is backed up before being overwritten.
     TR: Önceki veri, üzerine yazılmadan önce yedeklenir.

Per-station record / Her istasyon kaydı:
  name, road_km, power_kw, type, slots, lat, lng, operator,
  usage_type, last_verified, distance_from_route_km,
  exit_lat, exit_lng,
  detour_km, detour_min            (real round-trip / gerçek gidiş+dönüş)
  detour_out_path, detour_back_path ([[lat, lng, elev], ...])

Usage / Kullanım:
    ./venv/bin/python -m evroute.data.sources.openchargemap
    evroute fetch stations
"""

import json
import math
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import requests

from evroute.config import get_data_dir
from evroute.data.secrets import get_secret

RAW_DIR = get_data_dir() / "raw"
PROC_DIR = get_data_dir() / "processed"

ROUTES = ["istanbul_ankara", "istanbul_izmir", "ankara_antalya"]

MIN_POWER_KW = 50.0
MAX_STRAIGHT_DETOUR_KM = 10.0      # kuş uçuşu ön eleme
MAX_ROAD_DETOUR_KM = 40.0          # fazladan (extra) mesafe üst sınırı
ANCHOR_OFFSET_KM = 3.0             # istasyon projeksiyonundan önce/sonra mesafe
SAMPLE_EVERY_KM = 20.0
OCM_RADIUS_KM = 15.0
DEDUPE_KM = 3.0
ELEV_MAX_PER_REQ = 256             # Google Elevation locations limiti


# ---------- Geometri ----------

def _haversine_km(lat1, lng1, lat2, lng2) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _decode_polyline(enc: str) -> List[List[float]]:
    points, index, lat, lng = [], 0, 0, 0
    while index < len(enc):
        for k in range(2):
            shift, result = 0, 0
            while True:
                b = ord(enc[index]) - 63
                index += 1
                result |= (b & 0x1f) << shift
                shift += 5
                if b < 0x20:
                    break
            d = ~(result >> 1) if (result & 1) else (result >> 1)
            if k == 0:
                lat += d
            else:
                lng += d
        points.append([lat / 1e5, lng / 1e5])
    return points


def _load_polyline(route_key: str) -> List[List[float]]:
    geom = json.load(open(RAW_DIR / f"{route_key}_route_geometry.json",
                          encoding="utf-8"))
    return geom["coordinates"]


def _cumulative_km(coords) -> List[float]:
    cum = [0.0]
    for i in range(1, len(coords)):
        cum.append(cum[-1] + _haversine_km(coords[i-1][0], coords[i-1][1],
                                           coords[i][0], coords[i][1]))
    return cum


def _project(lat, lng, coords, cum):
    best_d, best_km, best_pt = float("inf"), 0.0, (coords[0][0], coords[0][1])
    for i in range(len(coords)):
        d = _haversine_km(lat, lng, coords[i][0], coords[i][1])
        if d < best_d:
            best_d, best_km = d, cum[i]
            best_pt = (coords[i][0], coords[i][1])
    return best_km, best_d, best_pt


def _point_at_km(target_km, coords, cum):
    """Rota üzerinde verilen road_km'ye en yakın noktayı döndürür (lat,lng)."""
    target_km = max(cum[0], min(cum[-1], target_km))
    lo, hi = 0, len(cum) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if cum[mid] < target_km:
            lo = mid + 1
        else:
            hi = mid
    return (coords[lo][0], coords[lo][1])


# ---------- OCM ----------

def _ocm_query(lat, lng) -> List[dict]:
    params = {"output": "json", "countrycode": "TR",
              "latitude": lat, "longitude": lng,
              "distance": OCM_RADIUS_KM, "distanceunit": "KM",
              "maxresults": 200, "levelid": 3,
              "key": get_secret("OCM_API_KEY")}
    for attempt in range(3):
        try:
            r = requests.get("https://api.openchargemap.io/v3/poi/",
                             params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
            print(f"    OCM HTTP {r.status_code}: {r.text[:100]}")
            return []
        except Exception as e:
            print(f"    OCM hata (deneme {attempt+1}): {e}")
            time.sleep(2)
    return []


def _max_power_kw(poi) -> float:
    conns = poi.get("Connections") or []
    powers = [c.get("PowerKW") for c in conns if c.get("PowerKW")]
    return float(max(powers)) if powers else 0.0


def _has_ccs(poi) -> bool:
    for c in poi.get("Connections") or []:
        ct = c.get("ConnectionType") or {}
        title = (ct.get("Title") or "")
        if "CCS" in title or "Combo" in title:
            return True
    return False


def _is_operational(poi) -> bool:
    st = poi.get("StatusType")
    return True if not st else bool(st.get("IsOperational", True))


def _is_public(poi) -> bool:
    """Özel mülk / izinli erişim dışlanır; 'Public*' kabul."""
    ut = poi.get("UsageType") or {}
    title = (ut.get("Title") or "").strip()
    if not title:
        return True  # bilinmiyorsa dahil et (OCM çoğu zaman böyle bırakır)
    low = title.lower()
    if low.startswith("private"):
        return False
    if "notice required" in low and "public" not in low:
        return False
    return True


# ---------- Google ----------

def _gkey() -> str:
    return get_secret("GOOGLE_MAPS_API_KEY")


# Geçici (retry edilebilir) Google API durumları
_RETRY_STATUS = {"OVER_QUERY_LIMIT", "REQUEST_DENIED", "UNKNOWN_ERROR",
                 "RESOURCE_EXHAUSTED", "INTERNAL_ERROR"}
_BACKOFF = [5, 15, 40, 90, 150]   # saniye


class _GoogleAbort(RuntimeError):
    """Kalıcı Google hatası -> rota iptal (yedek korunur)."""


def _directions(o_lat, o_lng, d_lat, d_lng, gkey):
    """
    Tek yön sürüş rotası -> (km, trafikli_dk, [[lat,lng],...]).
    ZERO_RESULTS -> None (yol yok, istasyon atlanır).
    Kalıcı hata -> _GoogleAbort (rota iptal, yarım veri yazılmaz).
    """
    params = {"origin": f"{o_lat},{o_lng}",
              "destination": f"{d_lat},{d_lng}",
              "departure_time": "now", "key": gkey}
    for attempt in range(len(_BACKOFF) + 1):
        try:
            r = requests.get("https://maps.googleapis.com/maps/api/directions/json",
                             params=params, timeout=25)
            j = r.json()
            status = j.get("status")
            if status == "OK":
                leg = j["routes"][0]["legs"][0]
                km = leg["distance"]["value"] / 1000.0
                sec = leg.get("duration_in_traffic", leg["duration"])["value"]
                path = _decode_polyline(j["routes"][0]["overview_polyline"]["points"])
                return km, sec / 60.0, path
            if status == "ZERO_RESULTS":
                return None
            if status in _RETRY_STATUS and attempt < len(_BACKOFF):
                wait = _BACKOFF[attempt]
                print(f"    Directions {status} -> {wait}s sonra tekrar "
                      f"({attempt+1}/{len(_BACKOFF)})")
                time.sleep(wait)
                continue
            raise _GoogleAbort(f"Directions kalıcı hata: {status} "
                               f"{j.get('error_message','')}")
        except _GoogleAbort:
            raise
        except Exception as e:
            if attempt < len(_BACKOFF):
                time.sleep(_BACKOFF[attempt])
                continue
            raise _GoogleAbort(f"Directions ağ hatası: {e}")
    raise _GoogleAbort("Directions: tüm denemeler tükendi")


def _elevations(path: List[List[float]], gkey: str) -> List[float]:
    """Polyline noktaları için yükseklik (m). Sıra korunur."""
    out: List[float] = []
    for i in range(0, len(path), ELEV_MAX_PER_REQ):
        chunk = path[i:i + ELEV_MAX_PER_REQ]
        loc = "|".join(f"{p[0]},{p[1]}" for p in chunk)
        for attempt in range(len(_BACKOFF) + 1):
            try:
                r = requests.get("https://maps.googleapis.com/maps/api/elevation/json",
                                 params={"locations": loc, "key": gkey}, timeout=25)
                j = r.json()
                status = j.get("status")
                if status == "OK":
                    out.extend(round(x["elevation"], 1) for x in j["results"])
                    break
                if status in _RETRY_STATUS and attempt < len(_BACKOFF):
                    wait = _BACKOFF[attempt]
                    print(f"    Elevation {status} -> {wait}s sonra tekrar")
                    time.sleep(wait)
                    continue
                raise _GoogleAbort(f"Elevation kalıcı hata: {status} "
                                   f"{j.get('error_message','')}")
            except _GoogleAbort:
                raise
            except Exception as e:
                if attempt < len(_BACKOFF):
                    time.sleep(_BACKOFF[attempt])
                    continue
                raise _GoogleAbort(f"Elevation ağ hatası: {e}")
        time.sleep(0.2)
    return out


def _attach_elev(path: List[List[float]], gkey: str) -> List[List[float]]:
    elevs = _elevations(path, gkey)
    return [[round(p[0], 6), round(p[1], 6), elevs[i] if i < len(elevs) else 0.0]
            for i, p in enumerate(path)]


# ---------- Ana akış ----------

def fetch_route(route_key: str, gkey: str) -> List[Dict]:
    coords = _load_polyline(route_key)
    cum = _cumulative_km(coords)
    total_km = cum[-1]

    sample_idx, nxt = [], 0.0
    for i in range(len(coords)):
        if cum[i] >= nxt:
            sample_idx.append(i)
            nxt += SAMPLE_EVERY_KM
    sample_idx.append(len(coords) - 1)
    print(f"  {len(sample_idx)} OCM sorgu noktası, polyline {total_km:.0f} km")

    seen = {}
    for i in sample_idx:
        for poi in _ocm_query(coords[i][0], coords[i][1]):
            seen[poi["ID"]] = poi
        time.sleep(0.3)
    print(f"  OCM ham: {len(seen)} POI")

    cand = []
    drop_ccs = drop_priv = 0
    for poi in seen.values():
        ai = poi.get("AddressInfo") or {}
        plat, plng = ai.get("Latitude"), ai.get("Longitude")
        if plat is None or plng is None:
            continue
        if _max_power_kw(poi) < MIN_POWER_KW or not _is_operational(poi):
            continue
        if not _has_ccs(poi):
            drop_ccs += 1
            continue
        if not _is_public(poi):
            drop_priv += 1
            continue
        road_km, straight, exit_pt = _project(plat, plng, coords, cum)
        if straight > MAX_STRAIGHT_DETOUR_KM:
            continue
        if road_km < 1.0 or road_km > total_km - 1.0:
            continue
        op = poi.get("OperatorInfo") or {}
        ut = poi.get("UsageType") or {}
        cand.append({
            "name": (ai.get("Title") or "Bilinmeyen").strip(),
            "road_km": round(road_km, 1),
            "power_kw": int(round(_max_power_kw(poi))),
            "type": "DC_Fast",
            "slots": max(1, len(poi.get("Connections") or [])),
            "lat": round(plat, 6), "lng": round(plng, 6),
            "operator": (op.get("Title") or "Bilinmiyor").strip(),
            "usage_type": (ut.get("Title") or "Bilinmiyor").strip(),
            "last_verified": poi.get("DateLastVerified") or poi.get("DateLastStatusUpdate"),
            "distance_from_route_km": round(straight, 2),
            "exit_lat": round(exit_pt[0], 6),
            "exit_lng": round(exit_pt[1], 6),
            "ocm_id": poi["ID"],
        })
    cand.sort(key=lambda s: s["road_km"])
    print(f"  Eleme: CCS yok={drop_ccs}, özel={drop_priv} -> aday {len(cand)}")

    stations = []
    for c in cand:
        rk = c["road_km"]
        # Gidiş yönünde önce/sonra çapa noktaları (doğru şerit)
        before_km = max(0.5, rk - ANCHOR_OFFSET_KM)
        after_km = min(total_km - 0.5, rk + ANCHOR_OFFSET_KM)
        before_pt = _point_at_km(before_km, coords, cum)
        after_pt = _point_at_km(after_km, coords, cum)
        # EN: Distance that would be driven by staying on the highway.
        # TR: Otoyolda kalsaydı kat edilecek mesafe.
        baseline_km = after_km - before_km

        # EN: before -> station -> after; direction-aware, since Google
        #     enforces one-way / divided-road constraints.
        # TR: önce -> istasyon -> sonra; yön-duyarlı, çünkü Google tek-yön
        #     ve bölünmüş yol kısıtlarını uygular.
        leg1 = _directions(before_pt[0], before_pt[1], c["lat"], c["lng"], gkey)
        time.sleep(0.2)
        leg2 = _directions(c["lat"], c["lng"], after_pt[0], after_pt[1], gkey)
        time.sleep(0.2)
        if leg1 is None or leg2 is None:
            continue

        extra_km = (leg1[0] + leg2[0]) - baseline_km
        # EN: Extra distance cannot be negative; floor as minimal approach.
        # TR: Fazladan mesafe negatif olamaz; minimum yaklaşım olarak taban.
        extra_km = max(0.2, extra_km)
        if extra_km > MAX_ROAD_DETOUR_KM:
            continue
        # EN: Extra time vs. covering the baseline on the highway at ~110 km/h.
        # TR: Fazladan süre: baseline'ı otoyolda ~110 km/h geçmeye kıyasla.
        baseline_min = baseline_km / 110.0 * 60.0
        extra_min = max(0.5, (leg1[1] + leg2[1]) - baseline_min)

        c["detour_km"] = round(extra_km, 2)
        c["detour_min"] = round(extra_min, 1)
        c["anchor_before_lat"] = round(before_pt[0], 6)
        c["anchor_before_lng"] = round(before_pt[1], 6)
        c["anchor_after_lat"] = round(after_pt[0], 6)
        c["anchor_after_lng"] = round(after_pt[1], 6)
        c["anchor_before_km"] = round(before_km, 1)
        c["anchor_after_km"] = round(after_km, 1)
        # EN: Elevation is costly, so it is fetched AFTER dedup, only for
        #     the surviving stations.
        # TR: Elevation maliyetli olduğundan dedup SONRASI, yalnızca kalan
        #     istasyonlar için çekilir.
        c["_raw_out"] = leg1[2]
        c["_raw_back"] = leg2[2]
        stations.append(c)
        print(f"    · {c['name'][:38]:38s} km={c['road_km']:5.0f} "
              f"fazladan={c['detour_km']:5.1f}km/{c['detour_min']:4.0f}dk "
              f"{c['power_kw']:3d}kW")

    # EN: Dedup — among nearby stations keep the correct-carriageway one
    #     (smallest detour), breaking ties by higher power. This preserves
    #     direction awareness.
    # TR: Dedup — yakın istasyonlardan doğru şerittekini (en az sapma) tut,
    #     eşitlikte yüksek güç. Yön-duyarlılığın korunması için gereklidir.
    stations.sort(key=lambda s: s["road_km"])
    deduped = []
    for s in stations:
        if deduped and abs(s["road_km"] - deduped[-1]["road_km"]) < DEDUPE_KM:
            prev = deduped[-1]
            better = (s["detour_km"], -s["power_kw"]) < (prev["detour_km"], -prev["power_kw"])
            if better:
                deduped[-1] = s
            continue
        deduped.append(s)

    # --- Elevation: yalnızca dedup sonrası kalan istasyonlar için ---
    print(f"  Dedup sonrası {len(deduped)} istasyon, yükseklik çekiliyor...")
    for s in deduped:
        s["detour_out_path"] = _attach_elev(s.pop("_raw_out"), gkey)
        s["detour_back_path"] = _attach_elev(s.pop("_raw_back"), gkey)
    return deduped


def main():
    gkey = _gkey()
    for route_key in ROUTES:
        print(f"\n=== {route_key} ===")
        try:
            stations = fetch_route(route_key, gkey)
        except Exception as e:
            print(f"  HATA: {e}; rota atlandı, mevcut veri korunuyor.")
            continue
        if not stations:
            print("  Hiç istasyon yok; mevcut veri korunuyor.")
            continue

        out = PROC_DIR / f"{route_key}_stations_ocm.json"
        if out.exists():
            (PROC_DIR / f"{route_key}_stations_ocm.backup.json").write_text(
                out.read_text(encoding="utf-8"), encoding="utf-8")
            try:
                old_n = len(json.load(open(out, encoding="utf-8")))
            except Exception:
                old_n = 0
        else:
            old_n = 0

        out.write_text(json.dumps(stations, ensure_ascii=False, indent=2),
                        encoding="utf-8")
        kms = [s["road_km"] for s in stations]
        gaps = [round(kms[i+1]-kms[i], 1) for i in range(len(kms)-1)]
        dts = [s["detour_km"] for s in stations]
        print(f"  Önceki {old_n} -> Yeni {len(stations)} istasyon")
        print(f"  road_km: {kms}")
        print(f"  En büyük boşluk: {max(gaps) if gaps else 0} km")
        print(f"  Sapma km ort={sum(dts)/len(dts):.1f} maks={max(dts):.1f}")
        print(f"  Kaydedildi: {out}")


if __name__ == "__main__":
    main()
