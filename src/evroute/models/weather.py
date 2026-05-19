# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Saadettin Yıldırım (Yıldsam Teknoloji)
"""
Hava Durumu Etki Modeli
========================

Sicaklik, ruzgar ve yagmurun enerji tuketimine etkisini modeller.

Referanslar:
- Steinstraeter et al. (2021) "Effect of Low Temperature on EV Range"
- Iora & Tribioli (2019) "Effect of Ambient Temperature on EV Energy Consumption"
- AAA (2019) "AAA Electric Vehicle Range Testing"
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class WeatherCondition:
    """Anlık hava durumu kosullari."""
    temperature_c: float         # Hava sicakligi (°C)
    wind_speed_ms: float = 0.0   # Ruzgar hizi (m/s)
    wind_direction_deg: float = 0.0  # Ruzgar yonu (derece, 0=kuzey, saat yonu)
    is_raining: bool = False     # Yagmur var mi
    humidity_pct: float = 50.0   # Nem orani (%)


# ---------- Onceden Tanimli Hava Senaryolari ----------

WEATHER_SCENARIOS = {
    "yaz_gunesli": WeatherCondition(
        temperature_c=30.0, wind_speed_ms=2.0, wind_direction_deg=180,
        is_raining=False, humidity_pct=40,
    ),
    "yaz_sicak": WeatherCondition(
        temperature_c=38.0, wind_speed_ms=1.0, wind_direction_deg=180,
        is_raining=False, humidity_pct=30,
    ),
    "ilkbahar_normal": WeatherCondition(
        temperature_c=18.0, wind_speed_ms=3.0, wind_direction_deg=270,
        is_raining=False, humidity_pct=55,
    ),
    "kis_soguk": WeatherCondition(
        temperature_c=-5.0, wind_speed_ms=4.0, wind_direction_deg=0,
        is_raining=False, humidity_pct=70,
    ),
    "kis_cok_soguk": WeatherCondition(
        temperature_c=-15.0, wind_speed_ms=5.0, wind_direction_deg=45,
        is_raining=False, humidity_pct=80,
    ),
    "yagmurlu": WeatherCondition(
        temperature_c=12.0, wind_speed_ms=6.0, wind_direction_deg=225,
        is_raining=True, humidity_pct=90,
    ),
    "optimal": WeatherCondition(
        temperature_c=22.0, wind_speed_ms=0.0, wind_direction_deg=0,
        is_raining=False, humidity_pct=50,
    ),
}


def temperature_battery_factor(temperature_c: float) -> float:
    """
    Sicakligin batarya verimine etkisi (carpan).
    Sogukta lityum-iyon batarya ic direnci artar -> daha fazla enerji kaybi.
    Sicakta AC sistemi ek enerji harcar.

    Referans:
    - Steinstraeter et al. (2021): -10°C'de %35-40 menzil kaybi
    - AAA (2019): 20°F (-6.7°C) @ 95°F (35°C) testleri

    Returns:
        Carpan (1.0 = optimal 20-25°C arasi, >1.0 = fazla tuketim)
    """
    # Parca-bazli lineer model (literatur verilerine fit)
    if temperature_c <= -15:
        return 1.45
    elif temperature_c <= -10:
        factor = 1.45 + (temperature_c + 15) * (1.35 - 1.45) / 5
    elif temperature_c <= 0:
        factor = 1.35 + (temperature_c + 10) * (1.20 - 1.35) / 10
    elif temperature_c <= 10:
        factor = 1.20 + (temperature_c - 0) * (1.08 - 1.20) / 10
    elif temperature_c <= 20:
        factor = 1.08 + (temperature_c - 10) * (1.00 - 1.08) / 10
    elif temperature_c <= 30:
        factor = 1.00  # Optimal aralik
    elif temperature_c <= 35:
        factor = 1.00 + (temperature_c - 30) * 0.01  # AC etkisi baslar
    else:
        factor = 1.05 + (temperature_c - 35) * 0.015  # Agresif sogutma

    return max(factor, 1.0)


def wind_effect_on_consumption(v_vehicle_kmh: float,
                               wind_speed_ms: float,
                               wind_direction_deg: float,
                               road_bearing_deg: float = 90.0) -> float:
    """
    Ruzgarin enerji tuketimine etkisi (carpan).

    Karsidan ruzgar aerodinamik surtukunmeyi artirir:
    F_aero ~ (v_vehicle + v_headwind)^2

    Args:
        v_vehicle_kmh: Arac hizi (km/h)
        wind_speed_ms: Ruzgar hizi (m/s)
        wind_direction_deg: Ruzgar yonu (meteorolojik: ruzgarin geldigi yon)
        road_bearing_deg: Yol yonu (derece, 0=kuzey, 90=dogu)
    Returns:
        Aerodinamik tuketim carpani (>1 = karsidan ruzgar, <1 = arkadan)
    """
    v_vehicle_ms = v_vehicle_kmh / 3.6
    if v_vehicle_ms < 1.0:
        return 1.0

    # Ruzgarin yol eksenindeki bileseni
    # Meteorolojik yon: ruzgarin geldigi yon, 180 derece cevir
    wind_from_rad = np.radians(wind_direction_deg)
    road_rad = np.radians(road_bearing_deg)

    # Karsidan ruzgar bileseni (pozitif = karsidan)
    angle_diff = wind_from_rad - road_rad
    headwind_ms = wind_speed_ms * np.cos(angle_diff)

    # Aerodinamik kuvvet orani
    v_effective = v_vehicle_ms + headwind_ms
    aero_ratio = (v_effective / v_vehicle_ms) ** 2

    # Toplam tuketim icinde aerodinamigin payi ~%40-60 (otoban hizlarinda)
    aero_share = 0.50  # Otoban icin yaklasik
    consumption_factor = 1.0 + aero_share * (aero_ratio - 1.0)

    return max(consumption_factor, 0.5)  # Minimum %50 (cok guclu arka ruzgarda bile)


def rain_rolling_resistance_factor(is_raining: bool) -> float:
    """
    Yagmurun yuvarlanma direncine etkisi.
    Islak yolda lastik-yol arasindaki su filmi ek enerji harcar.

    Referans: Willis et al. (2004) ~ %10-20 artis

    Returns:
        C_rr carpani (1.0 = kuru, ~1.15 = islak)
    """
    return 1.15 if is_raining else 1.0


def combined_weather_factor(weather: WeatherCondition,
                            v_vehicle_kmh: float,
                            road_bearing_deg: float = 90.0) -> float:
    """
    Tum hava kosullarinin birlesik tuketim etkisi.

    Returns:
        Toplam carpan (1.0 = ideal kosullar)
    """
    temp_factor = temperature_battery_factor(weather.temperature_c)
    wind_factor = wind_effect_on_consumption(
        v_vehicle_kmh, weather.wind_speed_ms,
        weather.wind_direction_deg, road_bearing_deg
    )
    rain_factor = rain_rolling_resistance_factor(weather.is_raining)

    # Carpanlar bagimsiz etki eder
    # Ama toplam etki ustu uste binmez, hafif azaltma uygula
    combined = temp_factor * wind_factor * rain_factor

    # Makul sinirlar icinde tut
    return np.clip(combined, 0.7, 2.0)


def charging_speed_factor(temperature_c: float) -> float:
    """
    Sicakligin sarj hizina etkisi.
    Sogukta batarya on isitma yapilmazsa sarj gucu duser.

    Referans: Tomaszewska et al. (2019) "Lithium-ion battery fast charging"

    Returns:
        Sarj gucu carpani (1.0 = optimal, <1.0 = yavas sarj)
    """
    if temperature_c <= -10:
        return 0.50  # Cok soguk, sarj cok yavas
    elif temperature_c <= 0:
        return 0.50 + (temperature_c + 10) * 0.02  # 0.50 -> 0.70
    elif temperature_c <= 10:
        return 0.70 + (temperature_c - 0) * 0.02   # 0.70 -> 0.90
    elif temperature_c <= 15:
        return 0.90 + (temperature_c - 10) * 0.02  # 0.90 -> 1.00
    elif temperature_c <= 35:
        return 1.0  # Optimal aralik
    elif temperature_c <= 45:
        return 1.0 - (temperature_c - 35) * 0.01   # Cok sicakta da yavaslama
    else:
        return 0.90


if __name__ == "__main__":
    print("=== Sicaklik -> Batarya Verimi ===")
    for temp in [-15, -10, -5, 0, 5, 10, 15, 20, 25, 30, 35, 40]:
        f = temperature_battery_factor(temp)
        print(f"  {temp:+3d}°C -> x{f:.2f} tuketim ({(f-1)*100:+.0f}%)")

    print("\n=== Hava Senaryolari (120 km/h) ===")
    for name, weather in WEATHER_SCENARIOS.items():
        f = combined_weather_factor(weather, 120)
        cf = charging_speed_factor(weather.temperature_c)
        print(f"  {name:20s}: tuketim x{f:.2f} ({(f-1)*100:+.0f}%), "
              f"sarj hizi x{cf:.2f} ({(cf-1)*100:+.0f}%)")
