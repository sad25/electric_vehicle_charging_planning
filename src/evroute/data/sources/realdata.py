# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Saadettin Yıldırım (Yıldsam Teknoloji)
"""
Google Maps API'den Gercek Veri Cekme
======================================

1. Elevation API -> Rota boyunca gercek yukseklik profili
2. Directions API -> Saat bazli gercek trafik suresi

Veriler data/raw/ altina JSON olarak kaydedilir.
Bir kez cekilir, tekrar API cagirmaya gerek kalmaz.
"""

import os
import json
import time
import numpy as np
import requests
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime, timedelta

# API Key — .env / ortam değişkeninden (koda gömülmez); tembel okunur
# API key — read lazily from .env / env var (never hard-coded)
from evroute.config import get_data_dir
from evroute.data.secrets import get_secret

DATA_DIR = get_data_dir() / "raw"
PROCESSED_DIR = get_data_dir() / "processed"


# ---------- Rota Tanimlari ----------

ROUTES = {
    "istanbul_ankara": {
        "waypoints": [
            {"name": "Istanbul (Gebze)", "lat": 40.7995, "lng": 29.4310, "road_km": 0},
            {"name": "Sakarya",          "lat": 40.6940, "lng": 30.4030, "road_km": 120},
            {"name": "Duzce",            "lat": 40.8389, "lng": 31.1639, "road_km": 200},
            {"name": "Bolu",             "lat": 40.7260, "lng": 31.6090, "road_km": 270},
            {"name": "Ankara",           "lat": 39.9208, "lng": 32.8541, "road_km": 450},
        ],
        "total_km": 450,
    },
    "istanbul_izmir": {
        "waypoints": [
            {"name": "Istanbul (Gebze)", "lat": 40.7995, "lng": 29.4310, "road_km": 0},
            {"name": "Bursa",            "lat": 40.1826, "lng": 29.0665, "road_km": 155},
            {"name": "Balikesir",        "lat": 39.6484, "lng": 27.8826, "road_km": 280},
            {"name": "Manisa",           "lat": 38.6191, "lng": 27.4289, "road_km": 400},
            {"name": "Izmir",            "lat": 38.4237, "lng": 27.1428, "road_km": 480},
        ],
        "total_km": 480,
    },
    "ankara_antalya": {
        "waypoints": [
            {"name": "Ankara",   "lat": 39.9208, "lng": 32.8541, "road_km": 0},
            {"name": "Konya",    "lat": 37.8714, "lng": 32.4846, "road_km": 260},
            {"name": "Isparta",  "lat": 37.7648, "lng": 30.5566, "road_km": 370},
            {"name": "Burdur",   "lat": 37.7203, "lng": 30.2906, "road_km": 400},
            {"name": "Antalya",  "lat": 36.8969, "lng": 30.7133, "road_km": 480},
        ],
        "total_km": 480,
    },
}


# ============================================================
# 1. ELEVATION API - Gercek Yukseklik Profili
# ============================================================

def fetch_elevation_for_route(route_key: str, interval_km: float = 2.0) -> dict:
    """
    Google Elevation API'den rota boyunca yukseklik verisi ceker.
    Waypoint'ler arasinda duzgun aralikli noktalar olusturur.

    API: https://maps.googleapis.com/maps/api/elevation/json
    Limit: Her istekte max 512 nokta, path modu ile daha fazla alinabilir.
    """
    route = ROUTES[route_key]
    waypoints = route["waypoints"]

    # Waypoint'ler arasi interpolasyon
    all_points = []
    for i in range(len(waypoints) - 1):
        wp1, wp2 = waypoints[i], waypoints[i + 1]
        seg_km = wp2["road_km"] - wp1["road_km"]
        n_points = max(int(seg_km / interval_km), 2)

        for j in range(n_points):
            frac = j / n_points
            lat = wp1["lat"] + frac * (wp2["lat"] - wp1["lat"])
            lng = wp1["lng"] + frac * (wp2["lng"] - wp1["lng"])
            km = wp1["road_km"] + frac * seg_km
            all_points.append({"lat": lat, "lng": lng, "km": km})

    # Son nokta
    last = waypoints[-1]
    all_points.append({"lat": last["lat"], "lng": last["lng"], "km": last["road_km"]})

    print(f"  {route_key}: {len(all_points)} nokta icin yukseklik cekilecek")

    # API'ye batch halinde gonder (max 512 per request)
    batch_size = 300
    all_elevations = []

    for batch_start in range(0, len(all_points), batch_size):
        batch = all_points[batch_start:batch_start + batch_size]
        locations = "|".join(f"{p['lat']},{p['lng']}" for p in batch)

        url = "https://maps.googleapis.com/maps/api/elevation/json"
        params = {"locations": locations,
                  "key": get_secret("GOOGLE_MAPS_API_KEY", required=False)}

        response = requests.get(url, params=params)
        data = response.json()

        if data["status"] != "OK":
            print(f"    HATA: {data.get('error_message', data['status'])}")
            return None

        for result in data["results"]:
            all_elevations.append(result["elevation"])

        time.sleep(0.2)  # Rate limiting

    # Sonuclari birlestir
    result = {
        "route_key": route_key,
        "route_name": f"{waypoints[0]['name']}-{waypoints[-1]['name']}",
        "distances_km": [p["km"] for p in all_points],
        "latitudes": [p["lat"] for p in all_points],
        "longitudes": [p["lng"] for p in all_points],
        "elevations_m": all_elevations,
        "fetched_at": datetime.now().isoformat(),
        "source": "Google Elevation API",
        "num_points": len(all_points),
    }

    # Kaydet
    out_dir = DATA_DIR / "elevation"
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = out_dir / f"{route_key}_elevation_google.json"
    with open(filepath, "w") as f:
        json.dump(result, f, indent=2)
    print(f"    Kaydedildi: {filepath}")

    # Processed klasorune de kaydet (elevation_model.py uyumlu format)
    distances = np.array(result["distances_km"])
    altitudes = np.array(result["elevations_m"])

    # Smoothing
    if len(altitudes) > 5:
        kernel = np.ones(5) / 5
        altitudes_smooth = np.convolve(altitudes, kernel, mode='same')
        altitudes_smooth[:2] = altitudes[:2]
        altitudes_smooth[-2:] = altitudes[-2:]
        altitudes = altitudes_smooth.tolist()

    # Egim hesapla
    grades = []
    for i in range(len(distances) - 1):
        d_km = distances[i + 1] - distances[i]
        d_alt = altitudes[i + 1] - altitudes[i] if isinstance(altitudes, list) else float(altitudes[i + 1] - altitudes[i])
        if d_km > 0.001:
            grades.append((d_alt / (d_km * 1000)) * 100)
        else:
            grades.append(0.0)

    processed = {
        "route_name": result["route_name"],
        "distances_km": result["distances_km"],
        "altitudes_m": altitudes if isinstance(altitudes, list) else altitudes.tolist(),
        "grades_percent": grades,
    }

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    proc_path = PROCESSED_DIR / f"{result['route_name'].replace(' ', '_').replace('/', '-')}_elevation.json"
    with open(proc_path, "w") as f:
        json.dump(processed, f, indent=2)
    print(f"    Processed: {proc_path}")

    return result


# ============================================================
# 2. DIRECTIONS API - Gercek Trafik Verileri
# ============================================================

def fetch_traffic_for_route(route_key: str,
                            hours: List[int] = None) -> dict:
    """
    Google Directions API'den saat bazli trafik verisi ceker.
    Her waypoint cifti icin, farkli saatlerde surus suresi alir.

    departure_time ile gelecekteki bir gun icin trafik tahmini alir.
    """
    if hours is None:
        hours = [0, 3, 6, 8, 10, 12, 14, 16, 18, 20, 22]

    route = ROUTES[route_key]
    waypoints = route["waypoints"]

    # Gelecek Pazartesi gununu bul (hafta ici trafik icin)
    today = datetime.now()
    days_until_monday = (7 - today.weekday()) % 7
    if days_until_monday == 0:
        days_until_monday = 7
    next_monday = today + timedelta(days=days_until_monday)

    # Gelecek Cumartesi (hafta sonu trafik icin)
    days_until_saturday = (5 - today.weekday()) % 7
    if days_until_saturday == 0:
        days_until_saturday = 7
    next_saturday = today + timedelta(days=days_until_saturday)

    results = {
        "route_key": route_key,
        "route_name": f"{waypoints[0]['name']}-{waypoints[-1]['name']}",
        "segments": [],
        "fetched_at": datetime.now().isoformat(),
        "source": "Google Directions API",
    }

    for i in range(len(waypoints) - 1):
        wp1, wp2 = waypoints[i], waypoints[i + 1]
        segment_name = f"{wp1['name']} -> {wp2['name']}"
        print(f"  {segment_name}...")

        segment_data = {
            "name": segment_name,
            "from": wp1,
            "to": wp2,
            "weekday": {},
            "weekend": {},
        }

        for day_type, base_date in [("weekday", next_monday), ("weekend", next_saturday)]:
            for hour in hours:
                departure = base_date.replace(hour=hour, minute=0, second=0)
                departure_ts = int(departure.timestamp())

                url = "https://maps.googleapis.com/maps/api/directions/json"
                params = {
                    "origin": f"{wp1['lat']},{wp1['lng']}",
                    "destination": f"{wp2['lat']},{wp2['lng']}",
                    "departure_time": departure_ts,
                    "key": get_secret("GOOGLE_MAPS_API_KEY", required=False),
                }

                response = requests.get(url, params=params)
                data = response.json()

                if data["status"] == "OK":
                    leg = data["routes"][0]["legs"][0]
                    distance_m = leg["distance"]["value"]
                    duration_s = leg["duration"]["value"]
                    # Trafikli sure (varsa)
                    duration_traffic_s = leg.get("duration_in_traffic", {}).get("value", duration_s)

                    segment_data[day_type][str(hour)] = {
                        "distance_km": distance_m / 1000,
                        "duration_min": duration_s / 60,
                        "duration_traffic_min": duration_traffic_s / 60,
                        "avg_speed_kmh": (distance_m / 1000) / (duration_s / 3600) if duration_s > 0 else 0,
                        "avg_speed_traffic_kmh": (distance_m / 1000) / (duration_traffic_s / 3600) if duration_traffic_s > 0 else 0,
                    }
                else:
                    print(f"    HATA {hour}:00: {data.get('error_message', data['status'])}")
                    segment_data[day_type][str(hour)] = None

                time.sleep(0.15)  # Rate limiting

        results["segments"].append(segment_data)

    # Kaydet
    out_dir = DATA_DIR / "traffic"
    out_dir.mkdir(parents=True, exist_ok=True)
    filepath = out_dir / f"{route_key}_traffic_google.json"
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Kaydedildi: {filepath}")

    return results


def build_traffic_profile_from_google(route_key: str) -> dict:
    """
    Google verisinden 24 saatlik trafik profili olusturur.
    Eksik saatler interpolasyonla doldurulur.
    """
    filepath = DATA_DIR / "traffic" / f"{route_key}_traffic_google.json"
    if not filepath.exists():
        print(f"  Google trafik verisi bulunamadi: {filepath}")
        print(f"  Once fetch_traffic_for_route('{route_key}') calistirin.")
        return None

    with open(filepath) as f:
        data = json.load(f)

    profile = {"route_key": route_key, "segments": []}

    for seg in data["segments"]:
        seg_profile = {"name": seg["name"]}

        for day_type in ["weekday", "weekend"]:
            hours_data = seg[day_type]

            # Mevcut saatlerdeki hiz verisi
            known_hours = []
            known_speeds = []
            for h_str, info in hours_data.items():
                if info is not None:
                    known_hours.append(int(h_str))
                    known_speeds.append(info["avg_speed_traffic_kmh"])

            if not known_hours:
                seg_profile[day_type] = [80.0] * 24  # Fallback
                continue

            # Serbest akis hizi (gece en hizli)
            free_flow_speed = max(known_speeds)

            # 24 saate interpolasyon
            all_hours = list(range(24))
            all_speeds = np.interp(all_hours, known_hours, known_speeds)

            # Faktor: gercek_hiz / serbest_akis
            factors = (all_speeds / free_flow_speed).tolist()
            seg_profile[f"{day_type}_speeds"] = all_speeds.tolist()
            seg_profile[f"{day_type}_factors"] = factors
            seg_profile[f"{day_type}_free_flow"] = free_flow_speed

        profile["segments"].append(seg_profile)

    # Kaydet
    proc_path = PROCESSED_DIR / f"{route_key}_traffic_profile.json"
    with open(proc_path, "w") as f:
        json.dump(profile, f, indent=2)
    print(f"  Trafik profili olusturuldu: {proc_path}")

    return profile


# ============================================================
# ANA FONKSIYON
# ============================================================

def fetch_all_data():
    """Tum rotalar icin tum verileri ceker."""
    print("=" * 60)
    print("GERCEK VERI CEKME - Google Maps API")
    print("=" * 60)

    for route_key in ROUTES:
        print(f"\n{'='*40}")
        print(f"ROTA: {route_key}")
        print(f"{'='*40}")

        # 1. Yukseklik
        print(f"\n[1/2] Yukseklik verisi cekiliyor...")
        fetch_elevation_for_route(route_key, interval_km=2.0)

        # 2. Trafik
        print(f"\n[2/2] Trafik verisi cekiliyor...")
        fetch_traffic_for_route(route_key)

        # Profil olustur
        print(f"\nTrafik profili olusturuluyor...")
        build_traffic_profile_from_google(route_key)

    print("\n" + "=" * 60)
    print("TAMAMLANDI!")
    print("=" * 60)


if __name__ == "__main__":
    fetch_all_data()
