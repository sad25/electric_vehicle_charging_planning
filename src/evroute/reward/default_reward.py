# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Saadettin Yıldırım (Yıldsam Teknoloji)
"""
Moduler Reward (Odul) Fonksiyonu
==================================

Farkli bilesenlerden olusan, surucu profiline gore agirliklanan
reward fonksiyonu. Her bilesen bagimsiz olarak test edilebilir.

Bilesanler:
1. R_time     - Sure minimizasyonu (surus + sarj + bekleme)
2. R_cost     - Sarj maliyeti minimizasyonu
3. R_comfort  - Surucu konforu (mola-sarj senkronizasyonu, yorgunluk)
4. R_anxiety  - Range anxiety (menzil kaygisi)
5. R_degrad   - Batarya yipranma cezasi
6. R_death    - Batarya olum cezasi (terminal)
"""

import numpy as np
from typing import Optional
from evroute.models.driver import (
    DriverProfile, break_charge_overlap,
    fatigue_penalty, patience_penalty, range_anxiety_penalty
)
from evroute.models.charging import degradation_stress


# ---------- Sarj Istasyonu Fiyatlandirmasi ----------

STATION_PRICING = {
    "ZES_DC_50kW":   {"peak": 7.50, "offpeak": 5.50},
    "ZES_DC_120kW":  {"peak": 8.50, "offpeak": 6.50},
    "Esarj_DC":      {"peak": 9.00, "offpeak": 7.00},
    "Tesla_SC":      {"peak": 6.00, "offpeak": 4.50},
    "Generic_DC":    {"peak": 8.00, "offpeak": 6.00},
}


def get_electricity_price(station_type: str, hour: float) -> float:
    """
    Saat ve istasyon tipine gore elektrik fiyati (TL/kWh).
    Gece tarife: 22:00 - 06:00 arasi ucuz.
    """
    pricing = STATION_PRICING.get(station_type, STATION_PRICING["Generic_DC"])
    h = hour % 24
    if h >= 22 or h < 6:
        return pricing["offpeak"]
    return pricing["peak"]


def charge_cost_tl(energy_kwh: float, station_type: str, hour: float) -> float:
    """Sarj maliyeti (TL)."""
    price = get_electricity_price(station_type, hour)
    return energy_kwh * price


# ---------- Kuyruk Bekleme Suresi ----------

def queue_wait_time(hour: float, station_slots: int = 2,
                    popularity: float = 0.5) -> float:
    """
    Istasyondaki tahmini kuyruk bekleme suresi (dakika).
    Basitlestirilmis model: saat bazli yogunluk + rastgelelik.

    Args:
        hour: Gunun saati (0-24)
        station_slots: Istasyondaki sarj slotu sayisi
        popularity: Istasyon populerligi (0-1)
    Returns:
        Tahmini bekleme suresi (dakika)
    """
    h = hour % 24

    # Saat bazli yogunluk orani
    if 6 <= h < 9:
        occupancy = 0.3
    elif 9 <= h < 11:
        occupancy = 0.4
    elif 11 <= h < 14:
        occupancy = 0.6
    elif 14 <= h < 17:
        occupancy = 0.5
    elif 17 <= h < 20:
        occupancy = 0.7
    elif 20 <= h < 22:
        occupancy = 0.4
    else:
        occupancy = 0.1

    # Populerlik etkisi
    occupancy *= (0.5 + popularity)
    occupancy = min(occupancy, 0.95)

    # Bekleme suresi: slot sayisi arttikca azalir
    if occupancy < 0.5:
        wait = 0.0
    else:
        # Basit kuyruk modeli
        excess = (occupancy - 0.5) * 2  # 0-1 arasi
        wait = excess ** 2 * 30 / station_slots  # Max ~30/slot dk

    return max(wait, 0.0)


# ---------- Reward Bileşenleri ----------

def reward_time(drive_time_h: float, charge_time_min: float,
                wait_time_min: float) -> float:
    """
    Sure cezasi: toplam harcanan sure.
    Normalize edilmis (saat cinsinden).
    """
    total_hours = drive_time_h + (charge_time_min + wait_time_min) / 60.0
    return -total_hours


def reward_cost(energy_kwh: float, station_type: str, hour: float,
                max_cost: float = 500.0) -> float:
    """
    Maliyet cezasi: sarj maliyeti.
    Normalize edilmis (max_cost'a gore).
    """
    cost = charge_cost_tl(energy_kwh, station_type, hour)
    return -cost / max_cost


def reward_comfort(continuous_drive_min: float,
                   max_drive_min: float,
                   preferred_break_min: float,
                   charge_time_min: float) -> float:
    """
    Konfor odulu/cezasi: mola-sarj senkronizasyonu + yorgunluk.
    Sarj mola zamanina denk gelirse pozitif odul.
    """
    overlap_info = break_charge_overlap(
        continuous_drive_min, max_drive_min,
        preferred_break_min, charge_time_min
    )

    # Mola-sarj overlap bonusu (0 ile +1 arasi)
    overlap_bonus = overlap_info["saved_time"] / max(preferred_break_min, 1.0)

    # Yorgunluk cezasi
    fatigue = fatigue_penalty(continuous_drive_min, max_drive_min)

    return overlap_bonus - fatigue


def reward_anxiety(current_soc: float,
                   min_comfortable_soc: float,
                   anxiety_weight: float) -> float:
    """Range anxiety cezasi."""
    return -range_anxiety_penalty(current_soc, min_comfortable_soc, anxiety_weight)


def reward_degradation(target_soc_pct: float,
                       station_power_kw: float) -> float:
    """Batarya yipranma cezasi."""
    return -degradation_stress(target_soc_pct, station_power_kw)


def reward_death() -> float:
    """Batarya tamamen bosaldi - terminal ceza."""
    return -20.0


# ---------- Birlesik Reward ----------

def compute_reward(driver: DriverProfile,
                   drive_time_h: float,
                   charge_time_min: float,
                   wait_time_min: float,
                   energy_kwh: float,
                   station_type: str,
                   hour: float,
                   current_soc: float,
                   target_soc_pct: float,
                   station_power_kw: float,
                   continuous_drive_min: float,
                   is_dead: bool = False) -> dict:
    """
    Tum bilsenleri birlestirip agirlikli toplam reward hesaplar.

    Returns:
        dict: Her bilesen ve toplam reward
    """
    if is_dead:
        return {
            "r_time": 0.0, "r_cost": 0.0, "r_comfort": 0.0,
            "r_anxiety": 0.0, "r_degradation": 0.0,
            "r_death": reward_death(),
            "total": reward_death(),
        }

    r_time = reward_time(drive_time_h, charge_time_min, wait_time_min)
    r_cost = reward_cost(energy_kwh, station_type, hour)
    r_comfort = reward_comfort(
        continuous_drive_min, driver.max_continuous_drive_min,
        driver.preferred_break_min, charge_time_min
    )
    r_anxiety = reward_anxiety(current_soc, driver.min_comfortable_soc, driver.range_anxiety_weight)
    r_degrad = reward_degradation(target_soc_pct, station_power_kw)

    # Sabir cezasi (comfort icinde ama ayri hesaplanir)
    r_patience = -patience_penalty(
        charge_time_min, driver.patience_threshold_min, driver.patience_decay_rate
    )
    r_comfort += r_patience * 0.5

    total = (
        driver.w_time * r_time
        + driver.w_cost * r_cost
        + driver.w_comfort * r_comfort
        + driver.w_anxiety * r_anxiety
        + driver.w_degradation * r_degrad
    )

    return {
        "r_time": r_time,
        "r_cost": r_cost,
        "r_comfort": r_comfort,
        "r_anxiety": r_anxiety,
        "r_degradation": r_degrad,
        "r_death": 0.0,
        "total": total,
    }


if __name__ == "__main__":
    from evroute.models.driver import DRIVER_PROFILES

    print("=== Reward Ornekleri ===\n")

    # Senaryo: 120km surup, 20dk sarj, %40->%80, 120kW istasyon
    for name, driver in DRIVER_PROFILES.items():
        r = compute_reward(
            driver=driver,
            drive_time_h=1.0,
            charge_time_min=20,
            wait_time_min=5,
            energy_kwh=30,
            station_type="ZES_DC_120kW",
            hour=14.0,
            current_soc=0.40,
            target_soc_pct=80,
            station_power_kw=120,
            continuous_drive_min=60,
        )
        print(f"{driver.name}:")
        for k, v in r.items():
            print(f"  {k}: {v:.3f}")
        print()
