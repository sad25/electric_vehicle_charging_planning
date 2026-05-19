# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Saadettin Yıldırım (Yıldsam Teknoloji)
"""
Trafik Profili Modeli
======================

Saat bazli trafik yogunluguna gore rota segmentlerinin ortalama hizini modeller.
Gercek API verisi (TomTom/Google) veya literatur tabanli sentetik profil destekler.

Referanslar:
- KGM Turkiye trafik yogunluk istatistikleri
- TomTom Traffic Index (Istanbul, Ankara sehir ici yogunluk verileri)
- Immers & Logghe (2002) "Traffic Flow Theory"
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class TrafficProfile:
    """Bir rota segmenti icin saat bazli trafik profili."""
    segment_name: str
    segment_start_km: float
    segment_end_km: float
    speed_limit_kmh: float               # Yasal hiz siniri
    hourly_speed_factor: np.ndarray      # 24 saat, 0-1 arasi (1 = serbest akis)


@dataclass
class RouteTrafficModel:
    """Bir rotanin tum segmentleri icin trafik modeli."""
    route_name: str
    segments: List[TrafficProfile]

    def speed_factor_at(self, km: float, hour: float) -> float:
        """
        Verilen km ve saatte hiz faktorunu dondurur (0-1).
        1.0 = serbest akis hizi, 0.3 = agir trafik.
        """
        hour_idx = int(hour) % 24
        hour_frac = hour - int(hour)

        for seg in self.segments:
            if seg.segment_start_km <= km < seg.segment_end_km:
                # Saat interpolasyonu
                f1 = seg.hourly_speed_factor[hour_idx]
                f2 = seg.hourly_speed_factor[(hour_idx + 1) % 24]
                return f1 + hour_frac * (f2 - f1)

        # Son segment
        if self.segments and km >= self.segments[-1].segment_end_km:
            seg = self.segments[-1]
            f1 = seg.hourly_speed_factor[hour_idx]
            f2 = seg.hourly_speed_factor[(hour_idx + 1) % 24]
            return f1 + hour_frac * (f2 - f1)

        return 1.0  # Default: serbest akis

    def get_speed_limit(self, km: float) -> float:
        """Verilen km'deki yasal hiz sinirini dondurur (km/h)."""
        for seg in self.segments:
            if seg.segment_start_km <= km < seg.segment_end_km:
                return seg.speed_limit_kmh
        if self.segments:
            return self.segments[-1].speed_limit_kmh
        return 120.0

    def effective_speed(self, km: float, hour: float,
                        desired_speed_kmh: float,
                        speed_tolerance: float = 0.0) -> float:
        """
        Trafigi ve hiz sinirini hesaba katarak efektif hizi dondurur.

        Args:
            km: Mevcut pozisyon
            hour: Gunun saati
            desired_speed_kmh: Surucunun istedigi hiz
            speed_tolerance: Hiz sinirini asma toleransi (0-1).
                0.0 = sinira uyar, 0.20 = %20 uzerinde surer (agresif)
        """
        factor = self.speed_factor_at(km, hour)
        speed_limit = self.get_speed_limit(km)

        # Toleransli hiz siniri (agresif surucu icin)
        # Ornek: limit=120, tolerance=0.25 -> max_allowed=150
        max_allowed = speed_limit * (1.0 + speed_tolerance)

        # Trafik siniri - toleransli limit uzerinden hesaplanir
        # Yogun trafikte herkes yavaslar, ama bos yolda toleransli surucu daha hizli gider
        max_traffic_speed = max_allowed * factor

        # Efektif hiz: istenen vs trafik
        return min(desired_speed_kmh, max_traffic_speed)

    def average_speed_for_segment(self, km_start: float, km_end: float,
                                  hour: float, desired_speed_kmh: float) -> float:
        """Bir segment boyunca ortalama efektif hizi hesaplar."""
        n_samples = max(int(km_end - km_start), 5)
        km_points = np.linspace(km_start, km_end, n_samples)
        speeds = [self.effective_speed(km, hour, desired_speed_kmh) for km in km_points]
        return float(np.mean(speeds))


# ---------- Sentetik Trafik Profilleri ----------

def _create_urban_exit_profile() -> np.ndarray:
    """
    Sehir cikisi trafik profili (yogun sabah/aksam).
    Turkiye sehir cikis yollarinda (D100, TEM baglanti):
    - En kotu durumda bile ~30-40 km/h (limit 82 -> factor 0.40)
    - Gece neredeyse serbest akis
    """
    factors = np.array([
        0.95,  # 00
        0.97,  # 01
        0.98,  # 02
        0.98,  # 03
        0.95,  # 04
        0.88,  # 05
        0.72,  # 06 - sabah yogunlugu baslangic
        0.58,  # 07 - yogun
        0.52,  # 08 - en yogun (~43 km/h @ limit 82)
        0.62,  # 09
        0.75,  # 10
        0.78,  # 11
        0.75,  # 12 - ogle
        0.78,  # 13
        0.75,  # 14
        0.68,  # 15
        0.58,  # 16 - aksam yogunlugu
        0.52,  # 17 - en yogun
        0.58,  # 18
        0.70,  # 19
        0.82,  # 20
        0.88,  # 21
        0.92,  # 22
        0.95,  # 23
    ])
    return factors


def _create_highway_profile() -> np.ndarray:
    """Sehirlerarasi otoban trafik profili (genel akici)."""
    factors = np.array([
        0.97, 0.98, 0.99, 0.99, 0.98, 0.95,  # 00-05
        0.90, 0.85, 0.82, 0.85, 0.88, 0.85,  # 06-11
        0.82, 0.85, 0.88, 0.85, 0.82, 0.80,  # 12-17
        0.85, 0.90, 0.93, 0.95, 0.97, 0.97,  # 18-23
    ])
    return factors


def _create_urban_entry_profile() -> np.ndarray:
    """Sehir girisi trafik profili (yogun aksam)."""
    factors = np.array([
        0.95, 0.97, 0.98, 0.98, 0.95, 0.88,  # 00-05
        0.78, 0.65, 0.60, 0.70, 0.78, 0.75,  # 06-11
        0.72, 0.75, 0.72, 0.65, 0.55, 0.50,  # 12-17 - aksam yogun (~41 km/h @ limit 82)
        0.58, 0.68, 0.78, 0.85, 0.92, 0.95,  # 18-23
    ])
    return factors


def _create_mountain_pass_profile() -> np.ndarray:
    """Dag gecidi profili (TIR'lar yuzunden yavas)."""
    factors = np.array([
        0.92, 0.95, 0.95, 0.95, 0.92, 0.85,  # 00-05
        0.78, 0.72, 0.70, 0.72, 0.75, 0.72,  # 06-11 - TIR yogunlugu
        0.70, 0.72, 0.75, 0.72, 0.70, 0.72,  # 12-17
        0.78, 0.85, 0.90, 0.92, 0.92, 0.92,  # 18-23
    ])
    return factors


def create_synthetic_traffic(route_key: str, is_weekend: bool = False) -> RouteTrafficModel:
    """
    Rota icin sentetik trafik modeli olusturur.
    Hafta sonu profilleri daha akici olur.
    """
    weekend_boost = 0.15 if is_weekend else 0.0

    def boost(factors):
        return np.clip(factors + weekend_boost, 0.3, 1.0)

    # Turkiye yasal hiz sinirlari (2918 sayili Karayollari Trafik Kanunu):
    # Otoban (bolunmus): 120 km/h (binek arac)
    # Devlet yolu (bolunmus): 110 km/h
    # Devlet yolu (tek yol): 90 km/h
    # Sehir ici: 50 km/h
    # Bolu Dagi gibi gecitlerde: 80-90 km/h (degisken tabela)

    if route_key == "istanbul_ankara":
        segments = [
            TrafficProfile("Istanbul Cikis (Gebze)", 0, 40, 82,     # Sehir cikis yolu, 82 km/h
                           boost(_create_urban_exit_profile())),
            TrafficProfile("O-4 Otobani (Gebze-Duzce)", 40, 200, 120,   # Otoban
                           boost(_create_highway_profile())),
            TrafficProfile("Bolu Dagi Gecidi", 200, 320, 80,             # Dag gecidi, sinirli hiz
                           boost(_create_mountain_pass_profile())),
            TrafficProfile("E-80 Otobani (Bolu-Ankara)", 320, 420, 120,  # Otoban
                           boost(_create_highway_profile())),
            TrafficProfile("Ankara Cevre Yolu", 420, 450, 82,            # Sehir girisi
                           boost(_create_urban_entry_profile())),
        ]
    elif route_key == "istanbul_izmir":
        segments = [
            TrafficProfile("Istanbul Cikis", 0, 40, 82,                  # Sehir cikis
                           boost(_create_urban_exit_profile())),
            TrafficProfile("Bursa Otobani", 40, 155, 120,                # Otoban
                           boost(_create_highway_profile())),
            TrafficProfile("Balikesir Devlet Yolu", 155, 350, 110,       # Bolunmus devlet yolu
                           boost(_create_highway_profile())),
            TrafficProfile("Izmir Otobani", 350, 440, 120,               # Otoban
                           boost(_create_highway_profile())),
            TrafficProfile("Izmir Girisi", 440, 480, 82,                 # Sehir girisi
                           boost(_create_urban_entry_profile())),
        ]
    elif route_key == "ankara_antalya":
        segments = [
            TrafficProfile("Ankara Cikis", 0, 40, 82,                    # Sehir cikis
                           boost(_create_urban_exit_profile())),
            TrafficProfile("Konya Otobani", 40, 260, 120,                # Otoban
                           boost(_create_highway_profile())),
            TrafficProfile("Isparta Devlet Yolu", 260, 400, 90,          # Tek seritli devlet yolu
                           boost(_create_highway_profile())),
            TrafficProfile("Toros Gecidi", 400, 450, 80,                  # Dag gecidi
                           boost(_create_mountain_pass_profile())),
            TrafficProfile("Antalya Girisi", 450, 480, 82,               # Sehir girisi
                           boost(_create_urban_entry_profile())),
        ]
    else:
        raise ValueError(f"Bilinmeyen rota: {route_key}")

    day_type = "HaftaSonu" if is_weekend else "HaftaIci"
    return RouteTrafficModel(f"{route_key}_{day_type}", segments)


def load_google_traffic(route_key: str, is_weekend: bool = False) -> Optional[RouteTrafficModel]:
    """
    Google Directions API'den cekilmis gercek trafik verisini yukler.
    data/processed/{route_key}_traffic_profile.json dosyasini okur.

    Gercek hiz verilerinden 24 saatlik faktor profili olusturur.
    """
    import json
    from evroute.config import get_data_dir

    # The data directory follows EVROUTE_DATA_DIR / cwd.
    # Veri dizini EVROUTE_DATA_DIR / cwd izler.
    proc_dir = get_data_dir() / "processed"
    filepath = proc_dir / f"{route_key}_traffic_profile.json"

    if not filepath.exists():
        return None

    with open(filepath) as f:
        profile = json.load(f)

    from evroute.models.elevation import ROUTES
    waypoints = ROUTES.get(route_key)
    if not waypoints:
        return None

    day_type = "weekend" if is_weekend else "weekday"
    segments = []

    for i, seg_data in enumerate(profile["segments"]):
        factors_key = f"{day_type}_factors"
        speeds_key = f"{day_type}_speeds"
        freeflow_key = f"{day_type}_free_flow"

        if factors_key not in seg_data:
            continue

        factors = np.array(seg_data[factors_key])
        free_flow = seg_data.get(freeflow_key, 90)

        # Segment sinirlarini waypoint'lerden al
        if i < len(waypoints) - 1:
            start_km = waypoints[i]["road_km"]
            end_km = waypoints[i + 1]["road_km"]
        else:
            continue

        # Hiz siniri: Turkiye yasal sinirlari (Google ortalama hizi yasal sinir degil)
        # Otoban: 120, devlet yolu: 90-110, dag gecidi: 80, sehir girisi: 82
        if free_flow > 85:
            speed_limit = 120  # Otoban
        elif free_flow > 70:
            speed_limit = 110  # Bolunmus devlet yolu
        else:
            speed_limit = 80   # Dag gecidi / sehir ici

        segments.append(TrafficProfile(
            segment_name=seg_data["name"],
            segment_start_km=start_km,
            segment_end_km=end_km,
            speed_limit_kmh=speed_limit,
            hourly_speed_factor=factors,
        ))

    if not segments:
        return None

    day_label = "HaftaSonu" if is_weekend else "HaftaIci"
    return RouteTrafficModel(f"{route_key}_{day_label}_google", segments)


def create_traffic_model(route_key: str, is_weekend: bool = False,
                         prefer_google: bool = True) -> RouteTrafficModel:
    """
    Trafik modeli olusturur. Google verisi varsa onu kullanir,
    yoksa sentetik profil olusturur.
    """
    if prefer_google:
        model = load_google_traffic(route_key, is_weekend)
        if model is not None:
            return model
    return create_synthetic_traffic(route_key, is_weekend)


def estimate_travel_time(traffic_model: RouteTrafficModel,
                         km_start: float, km_end: float,
                         departure_hour: float,
                         desired_speed_kmh: float) -> Tuple[float, float]:
    """
    Trafigi hesaba katarak surus suresini tahmin eder.

    Returns:
        (surus_suresi_saat, ortalama_hiz_kmh)
    """
    step_km = 5.0  # Her 5 km'de trafik guncelle
    current_km = km_start
    current_hour = departure_hour
    total_time_h = 0.0

    while current_km < km_end:
        next_km = min(current_km + step_km, km_end)
        segment_km = next_km - current_km

        eff_speed = traffic_model.effective_speed(
            current_km, current_hour, desired_speed_kmh
        )
        eff_speed = max(eff_speed, 20.0)  # Minimum 20 km/h

        segment_time_h = segment_km / eff_speed
        total_time_h += segment_time_h
        current_hour += segment_time_h
        current_km = next_km

    total_distance = km_end - km_start
    avg_speed = total_distance / total_time_h if total_time_h > 0 else desired_speed_kmh

    return total_time_h, avg_speed


if __name__ == "__main__":
    for route in ["istanbul_ankara", "istanbul_izmir", "ankara_antalya"]:
        for weekend in [False, True]:
            model = create_synthetic_traffic(route, weekend)
            day = "HaftaSonu" if weekend else "HaftaIci"
            print(f"\n--- {route} ({day}) ---")

            for hour in [7, 12, 17, 22]:
                travel_h, avg_v = estimate_travel_time(
                    model, 0, model.segments[-1].segment_end_km, hour, 120
                )
                print(f"  Kalkis {hour:02d}:00, 120 km/h istenen -> "
                      f"Ort. {avg_v:.0f} km/h, Sure: {travel_h:.1f} saat")
