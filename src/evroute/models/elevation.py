# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Saadettin Yıldırım (Yıldsam Teknoloji)
"""
Yukseklik Profili ve Egim Modeli
=================================

Rota boyunca yukseklik verisi cekilir, islenir ve egim hesaplanir.
Open Elevation API veya onceden kaydedilmis veri kullanilir.

Referanslar:
- SRTM (Shuttle Radar Topography Mission) 30m verisi
- Open Elevation API (open-source, SRTM tabanli)
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


from evroute.config import get_data_dir


def _processed_dir() -> Path:
    """
    Processed-data directory, resolved lazily so it follows
    ``EVROUTE_DATA_DIR`` / cwd.

    İşlenmiş veri dizini; ``EVROUTE_DATA_DIR`` / cwd'yi izlemek için
    tembel çözülür.
    """
    return get_data_dir() / "processed"


@dataclass
class ElevationProfile:
    """Bir rota icin yukseklik profili."""
    route_name: str
    distances_km: np.ndarray    # Baslangictan mesafe (km)
    altitudes_m: np.ndarray     # Yukseklik (m)
    grades_percent: np.ndarray  # Egim (%, her segment icin)

    def grade_at_km(self, km: float) -> float:
        """Verilen km'deki egimi dondurur (%)."""
        if len(self.distances_km) < 2:
            return 0.0
        km_clamped = np.clip(km, self.distances_km[0], self.distances_km[-1])
        return float(np.interp(km_clamped, self.distances_km[:-1], self.grades_percent))

    def altitude_at_km(self, km: float) -> float:
        """Verilen km'deki yuksekligi dondurur (m)."""
        return float(np.interp(km, self.distances_km, self.altitudes_m))

    def segment_stats(self, km_start: float, km_end: float) -> dict:
        """Bir segment icin istatistikler."""
        mask = (self.distances_km >= km_start) & (self.distances_km <= km_end)
        if not mask.any():
            return {"avg_grade": 0.0, "max_grade": 0.0, "elevation_gain": 0.0, "elevation_loss": 0.0}

        alts = self.altitudes_m[mask]
        diffs = np.diff(alts)

        return {
            "avg_grade": float(np.mean(self.grades_percent[(self.distances_km[:-1] >= km_start) & (self.distances_km[:-1] <= km_end)])) if len(self.grades_percent) > 0 else 0.0,
            "max_grade": float(np.max(np.abs(self.grades_percent[(self.distances_km[:-1] >= km_start) & (self.distances_km[:-1] <= km_end)]))) if len(self.grades_percent) > 0 else 0.0,
            "elevation_gain": float(np.sum(diffs[diffs > 0])) if len(diffs) > 0 else 0.0,
            "elevation_loss": float(np.abs(np.sum(diffs[diffs < 0]))) if len(diffs) > 0 else 0.0,
        }


def _interpolate_waypoints(waypoints: List[dict], interval_km: float = 1.0) -> List[Tuple[float, float, float]]:
    """
    Waypoint'ler arasinda duzgun aralikli koordinat noktalarini interpolasyonla uretir.
    Returns: [(lat, lng, road_km), ...]
    """
    points = []
    for i in range(len(waypoints) - 1):
        wp1, wp2 = waypoints[i], waypoints[i + 1]
        d_start, d_end = wp1["road_km"], wp2["road_km"]
        n_points = max(int((d_end - d_start) / interval_km), 1)

        for j in range(n_points + (1 if i == len(waypoints) - 2 else 0)):
            frac = j / n_points
            lat = wp1["lat"] + frac * (wp2["lat"] - wp1["lat"])
            lng = wp1["lng"] + frac * (wp2["lng"] - wp1["lng"])
            km = d_start + frac * (d_end - d_start)
            points.append((lat, lng, km))

    return points


def fetch_elevation_from_api(waypoints: List[dict],
                             interval_km: float = 1.0) -> ElevationProfile:
    """
    Open Elevation API'den yukseklik verisi ceker.
    API: https://api.open-elevation.com/api/v1/lookup

    Args:
        waypoints: [{"name": str, "lat": float, "lng": float, "road_km": float}, ...]
        interval_km: Olcum noktasi araligi (km)
    """
    if not HAS_REQUESTS:
        raise ImportError("requests kutuphanesi gerekli: pip install requests")

    points = _interpolate_waypoints(waypoints, interval_km)

    # API'ye gonder (batch halinde, max 100 nokta per request)
    all_elevations = []
    batch_size = 100

    for batch_start in range(0, len(points), batch_size):
        batch = points[batch_start:batch_start + batch_size]
        locations = [{"latitude": p[0], "longitude": p[1]} for p in batch]

        response = requests.post(
            "https://api.open-elevation.com/api/v1/lookup",
            json={"locations": locations},
            timeout=30,
        )
        response.raise_for_status()
        results = response.json()["results"]
        all_elevations.extend([r["elevation"] for r in results])

    distances = np.array([p[2] for p in points])
    altitudes = np.array(all_elevations)

    # Savitzky-Golay benzeri smoothing (basit hareketli ortalama)
    if len(altitudes) > 5:
        kernel_size = 5
        kernel = np.ones(kernel_size) / kernel_size
        altitudes_smooth = np.convolve(altitudes, kernel, mode='same')
        # Uclar icin orijinal degerleri koru
        altitudes_smooth[:2] = altitudes[:2]
        altitudes_smooth[-2:] = altitudes[-2:]
        altitudes = altitudes_smooth

    # Egim hesapla
    grades = _calculate_grades(distances, altitudes)

    route_name = f"{waypoints[0]['name']}-{waypoints[-1]['name']}"
    profile = ElevationProfile(route_name, distances, altitudes, grades)

    # Kaydet
    _save_profile(profile)

    return profile


def _calculate_grades(distances_km: np.ndarray, altitudes_m: np.ndarray) -> np.ndarray:
    """Egim hesapla (%)."""
    grades = np.zeros(len(distances_km) - 1)
    for i in range(len(grades)):
        d_km = distances_km[i + 1] - distances_km[i]
        d_alt = altitudes_m[i + 1] - altitudes_m[i]
        if d_km > 0.001:
            grades[i] = (d_alt / (d_km * 1000)) * 100  # % cinsinden
        else:
            grades[i] = 0.0
    return grades


def _save_profile(profile: ElevationProfile):
    """Profili JSON olarak kaydet."""
    _processed_dir().mkdir(parents=True, exist_ok=True)
    filename = profile.route_name.replace(" ", "_").replace("/", "-") + "_elevation.json"
    filepath = _processed_dir() / filename

    data = {
        "route_name": profile.route_name,
        "distances_km": profile.distances_km.tolist(),
        "altitudes_m": profile.altitudes_m.tolist(),
        "grades_percent": profile.grades_percent.tolist(),
    }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_profile(filepath: str) -> ElevationProfile:
    """Kaydedilmis profili yukle."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return ElevationProfile(
        route_name=data["route_name"],
        distances_km=np.array(data["distances_km"]),
        altitudes_m=np.array(data["altitudes_m"]),
        grades_percent=np.array(data["grades_percent"]),
    )


# ---------- Onceden Tanimli Rotalar ----------

ISTANBUL_ANKARA_WAYPOINTS = [
    {"name": "Istanbul (Gebze)", "lat": 40.7995, "lng": 29.4310, "road_km": 0},
    {"name": "Sakarya", "lat": 40.6940, "lng": 30.4030, "road_km": 120},
    {"name": "Duzce", "lat": 40.8389, "lng": 31.1639, "road_km": 200},
    {"name": "Bolu", "lat": 40.7260, "lng": 31.6090, "road_km": 270},
    {"name": "Ankara", "lat": 39.9208, "lng": 32.8541, "road_km": 450},
]

ISTANBUL_IZMIR_WAYPOINTS = [
    {"name": "Istanbul (Gebze)", "lat": 40.7995, "lng": 29.4310, "road_km": 0},
    {"name": "Bursa", "lat": 40.1826, "lng": 29.0665, "road_km": 155},
    {"name": "Balikesir", "lat": 39.6484, "lng": 27.8826, "road_km": 280},
    {"name": "Manisa", "lat": 38.6191, "lng": 27.4289, "road_km": 400},
    {"name": "Izmir", "lat": 38.4237, "lng": 27.1428, "road_km": 480},
]

ANKARA_ANTALYA_WAYPOINTS = [
    {"name": "Ankara", "lat": 39.9208, "lng": 32.8541, "road_km": 0},
    {"name": "Konya", "lat": 37.8714, "lng": 32.4846, "road_km": 260},
    {"name": "Isparta", "lat": 37.7648, "lng": 30.5566, "road_km": 370},
    {"name": "Burdur", "lat": 37.7203, "lng": 30.2906, "road_km": 400},
    {"name": "Antalya", "lat": 36.8969, "lng": 30.7133, "road_km": 480},
]

ROUTES = {
    "istanbul_ankara": ISTANBUL_ANKARA_WAYPOINTS,
    "istanbul_izmir": ISTANBUL_IZMIR_WAYPOINTS,
    "ankara_antalya": ANKARA_ANTALYA_WAYPOINTS,
}


def create_synthetic_elevation(waypoints: List[dict],
                               route_key: str) -> ElevationProfile:
    """
    API erisimi yoksa sentetik yukseklik profili olusturur.
    Gercek topografya bilgisine dayali yaklasik profil.
    """
    total_km = waypoints[-1]["road_km"]
    distances = np.arange(0, total_km + 1, 1.0)

    # Rota bazli gercekci profiller
    if route_key == "istanbul_ankara":
        # Gebze (50m) -> Sakarya (30m) -> Duzce (200m) -> Bolu Dagi (1300m) -> Ankara (850m)
        key_points_km = [0, 50, 120, 170, 200, 230, 260, 280, 320, 380, 420, 450]
        key_altitudes = [50, 100, 30, 200, 200, 600, 1000, 1300, 900, 850, 870, 850]
    elif route_key == "istanbul_izmir":
        # Gebze (50m) -> Bursa (100m) -> Balikesir (150m) -> Manisa (70m) -> Izmir (10m)
        key_points_km = [0, 80, 155, 220, 280, 340, 400, 440, 480]
        key_altitudes = [50, 300, 100, 200, 150, 100, 70, 30, 10]
    elif route_key == "ankara_antalya":
        # Ankara (850m) -> Tuz Golu (900m) -> Konya (1000m) -> Toros Daglari -> Antalya (30m)
        key_points_km = [0, 80, 150, 260, 310, 370, 400, 430, 460, 480]
        key_altitudes = [850, 900, 950, 1000, 1100, 1200, 900, 500, 150, 30]
    else:
        # Duz profil
        key_points_km = [0, total_km]
        key_altitudes = [100, 100]

    # Interpolasyon
    altitudes = np.interp(distances, key_points_km, key_altitudes)

    # Hafif gurultu ekle (dogal topografya icin)
    np.random.seed(42)
    noise = np.cumsum(np.random.normal(0, 2, len(distances)))
    noise = noise - np.linspace(noise[0], noise[-1], len(noise))  # Baslangic/bitis sabit
    altitudes = altitudes + noise

    # Egim hesapla
    grades = _calculate_grades(distances, altitudes)

    route_name = f"{waypoints[0]['name']}-{waypoints[-1]['name']}"
    profile = ElevationProfile(route_name, distances, altitudes, grades)
    _save_profile(profile)

    return profile


def get_elevation_profile(route_key: str, use_api: bool = False) -> ElevationProfile:
    """
    Rota icin yukseklik profili dondurur.
    Oncelik sirasi:
    1. Google Elevation API verisi (data/processed/ altinda)
    2. Open Elevation API (use_api=True ise)
    3. Sentetik profil (fallback)
    """
    waypoints = ROUTES.get(route_key)
    if waypoints is None:
        raise ValueError(f"Bilinmeyen rota: {route_key}. Secenekler: {list(ROUTES.keys())}")

    route_name = f"{waypoints[0]['name']}-{waypoints[-1]['name']}"
    filename = route_name.replace(" ", "_").replace("/", "-") + "_elevation.json"
    filepath = _processed_dir() / filename

    if filepath.exists():
        return load_profile(str(filepath))

    if use_api and HAS_REQUESTS:
        return fetch_elevation_from_api(waypoints, interval_km=1.0)
    else:
        return create_synthetic_elevation(waypoints, route_key)


if __name__ == "__main__":
    for key in ROUTES:
        profile = get_elevation_profile(key, use_api=False)
        print(f"\n--- {profile.route_name} ---")
        print(f"  Mesafe: {profile.distances_km[-1]:.0f} km")
        print(f"  Min yukseklik: {profile.altitudes_m.min():.0f} m")
        print(f"  Max yukseklik: {profile.altitudes_m.max():.0f} m")
        print(f"  Max egim: %{np.max(np.abs(profile.grades_percent)):.1f}")

        stats = profile.segment_stats(0, profile.distances_km[-1])
        print(f"  Toplam tirmanis: {stats['elevation_gain']:.0f} m")
        print(f"  Toplam inis: {stats['elevation_loss']:.0f} m")
