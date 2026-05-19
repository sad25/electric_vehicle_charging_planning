# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Saadettin Yıldırım (Yıldsam Teknoloji)
"""
Fizik Tabanli Elektrikli Arac Enerji Tuketim Modeli
====================================================

Longitudinal vehicle dynamics modelini kullanarak hiza, egime, hava kosullarina
ve arac yukune bagli enerji tuketimini hesaplar.

Referanslar:
- Fiori et al. (2016) "Power-based electric vehicle energy consumption model"
- De Cauwer et al. (2015) "Energy consumption prediction for EVs based on real-world data"
- Genikomsakis & Mitrentsis (2017) "A computationally efficient simulation model for EV energy consumption"

Fizik:
    F_total = F_rolling + F_aero + F_grade
    F_rolling = C_rr * m * g * cos(theta)
    F_aero    = 0.5 * rho * C_d * A_f * (v + v_wind)^2
    F_grade   = m * g * sin(theta)
    P_wheels  = F_total * v
    P_motor   = P_wheels / eta_drivetrain  (suruş)
    P_motor   = P_wheels * eta_regen       (rejeneratif frenleme, F_grade < 0)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ---------- Sabitler ----------
G = 9.81          # Yercekimi ivmesi (m/s^2)
RHO_STD = 1.225   # Standart hava yogunlugu (kg/m^3, 15°C, deniz seviyesi)
R_AIR = 287.05     # Kuru hava gas sabiti (J/(kg·K))
P_ATM = 101325     # Standart atmosfer basinci (Pa)


@dataclass
class VehicleParams:
    """Arac fiziksel parametreleri."""
    name: str
    mass_kg: float              # Bos arac kutlesi (kg)
    battery_total_kwh: float    # Toplam batarya kapasitesi (kWh)
    battery_usable_kwh: float   # Kullanilabilir kapasite (kWh)
    C_d: float                  # Aerodinamik surutunme katsayisi
    A_f: float                  # On kesit alani (m^2)
    C_rr: float                 # Yuvarlanma direnci katsayisi
    eta_drivetrain: float       # Aktarma organi verimi (0-1)
    eta_regen: float            # Rejeneratif frenleme verimi (0-1)
    aux_power_kw: float = 0.5   # Sabit aksesuar tuketimi (kW) - DC-DC, BMS, sogutucu pompa vb.
    hvac_power_kw: dict = field(default_factory=dict)  # HVAC guc tuketimleri


# ---------- Onceden Tanimli Araclar ----------

IONIQ5_84 = VehicleParams(
    name="Hyundai IONIQ 5 84 kWh (2024)",
    mass_kg=2100,
    battery_total_kwh=84.0,
    battery_usable_kwh=74.0,
    C_d=0.288,
    A_f=2.345,
    C_rr=0.0095,       # SUV EV lastik (Continental EcoContact)
    eta_drivetrain=0.88,
    eta_regen=0.60,
    aux_power_kw=0.6,   # BMS, sogutucu pompa, 12V sistem
    hvac_power_kw={
        "off": 0.0,
        "mild": 1.0,
        "cooling": 2.5,
        "heating": 4.0,
    },
)

TESLA_MODEL3_LR = VehicleParams(
    name="Tesla Model 3 Long Range (2023+)",
    mass_kg=1830,
    battery_total_kwh=78.1,
    battery_usable_kwh=72.0,
    C_d=0.23,
    A_f=2.22,
    C_rr=0.008,
    eta_drivetrain=0.90,
    eta_regen=0.65,
    aux_power_kw=0.4,   # Heat pump + verimli BMS
    hvac_power_kw={
        "off": 0.0,
        "mild": 0.8,
        "cooling": 2.0,
        "heating": 3.0,  # Heat pump daha verimli
    },
)

VEHICLES = {
    "ioniq5": IONIQ5_84,
    "tesla3": TESLA_MODEL3_LR,
}


def air_density(temperature_c: float, altitude_m: float = 0.0) -> float:
    """
    Hava yogunlugunu sicaklik ve yukseklige gore hesaplar.
    rho = P / (R * T)
    Basinc yukseklikle azalir (barometrik formul).

    Args:
        temperature_c: Hava sicakligi (°C)
        altitude_m: Yukseklik (m, deniz seviyesinden)
    Returns:
        Hava yogunlugu (kg/m^3)
    """
    T_kelvin = temperature_c + 273.15
    # Barometrik formul: P = P0 * exp(-g*M*h / (R*T))
    # Basitlestirilmis: her 100m'de ~%1.2 azalir
    P = P_ATM * np.exp(-G * altitude_m / (R_AIR * T_kelvin))
    return P / (R_AIR * T_kelvin)


def rolling_resistance(mass_kg: float, C_rr: float,
                       grade_rad: float = 0.0,
                       wet_road: bool = False) -> float:
    """
    Yuvarlanma direnci kuvveti (N).
    F_rr = C_rr * m * g * cos(theta)
    Islak yolda C_rr ~%15 artar.

    Referans: Gillespie (1992) "Fundamentals of Vehicle Dynamics"
    """
    c = C_rr * (1.15 if wet_road else 1.0)
    return c * mass_kg * G * np.cos(grade_rad)


def aerodynamic_drag(C_d: float, A_f: float, v_vehicle_ms: float,
                     v_headwind_ms: float = 0.0,
                     rho: float = RHO_STD) -> float:
    """
    Aerodinamik surutunme kuvveti (N).
    F_aero = 0.5 * rho * C_d * A_f * v_eff^2

    Args:
        v_vehicle_ms: Arac hizi (m/s)
        v_headwind_ms: Karsidan ruzgar hizi (m/s, pozitif = karsidan)
        rho: Hava yogunlugu (kg/m^3)
    """
    v_effective = v_vehicle_ms + v_headwind_ms
    return 0.5 * rho * C_d * A_f * v_effective ** 2


def grade_resistance(mass_kg: float, grade_rad: float) -> float:
    """
    Egim direnci kuvveti (N).
    F_grade = m * g * sin(theta)
    Pozitif = yokus yukari, Negatif = yokus asagi
    """
    return mass_kg * G * np.sin(grade_rad)


def total_traction_force(vehicle: VehicleParams,
                         v_kmh: float,
                         grade_percent: float = 0.0,
                         extra_mass_kg: float = 0.0,
                         headwind_ms: float = 0.0,
                         temperature_c: float = 20.0,
                         altitude_m: float = 0.0,
                         wet_road: bool = False) -> float:
    """
    Toplam cekis kuvveti (N).

    Args:
        vehicle: Arac parametreleri
        v_kmh: Arac hizi (km/h)
        grade_percent: Yol egimi (%, pozitif = yokus yukari)
        extra_mass_kg: Ek yuk (yolcu + bagaj, kg)
        headwind_ms: Karsidan ruzgar (m/s)
        temperature_c: Hava sicakligi (°C)
        altitude_m: Yukseklik (m)
        wet_road: Islak yol durumu
    Returns:
        Toplam kuvvet (N). Negatif olabilir (inis + dusuk hiz)
    """
    v_ms = v_kmh / 3.6
    m_total = vehicle.mass_kg + extra_mass_kg
    grade_rad = np.arctan(grade_percent / 100.0)
    rho = air_density(temperature_c, altitude_m)

    F_rr = rolling_resistance(m_total, vehicle.C_rr, grade_rad, wet_road)
    F_aero = aerodynamic_drag(vehicle.C_d, vehicle.A_f, v_ms, headwind_ms, rho)
    F_grade = grade_resistance(m_total, grade_rad)

    return F_rr + F_aero + F_grade


def power_at_wheels(vehicle: VehicleParams,
                    v_kmh: float,
                    **kwargs) -> float:
    """
    Tekerleklerdeki mekanik guc (W).
    P = F * v
    """
    v_ms = v_kmh / 3.6
    F = total_traction_force(vehicle, v_kmh, **kwargs)
    return F * v_ms


def energy_consumption_kwh_per_km(vehicle: VehicleParams,
                                  v_kmh: float,
                                  hvac_mode: str = "off",
                                  **kwargs) -> float:
    """
    Spesifik enerji tuketimi (kWh/km).
    Motor verimi ve HVAC dahil.

    Args:
        vehicle: Arac parametreleri
        v_kmh: Arac hizi (km/h)
        hvac_mode: "off", "mild", "cooling", "heating"
        **kwargs: total_traction_force'a gecilecek argumanlar
    Returns:
        Tuketim (kWh/km). Negatif olabilir (rejeneratif frenleme)
    """
    v_ms = v_kmh / 3.6
    if v_ms < 0.1:
        return 0.0

    P_wheels = power_at_wheels(vehicle, v_kmh, **kwargs)

    if P_wheels >= 0:
        # Suruş: motor gucu = tekerlek gucu / verim
        P_motor = P_wheels / vehicle.eta_drivetrain
    else:
        # Rejeneratif frenleme: geri kazanilan enerji
        P_motor = P_wheels * vehicle.eta_regen

    # HVAC + aksesuar ek tuketimi
    hvac_kw = vehicle.hvac_power_kw.get(hvac_mode, 0.0)
    P_total_kw = (P_motor / 1000.0) + hvac_kw + vehicle.aux_power_kw

    # kWh/km = kW / (km/h)
    return P_total_kw / v_kmh


def energy_for_segment(vehicle: VehicleParams,
                       distance_km: float,
                       v_kmh: float,
                       hvac_mode: str = "off",
                       **kwargs) -> float:
    """
    Bir rota segmenti icin toplam enerji tuketimi (kWh).

    Returns:
        Tuketim (kWh). Negatif = enerji geri kazanimi (uzun inis)
    """
    e_per_km = energy_consumption_kwh_per_km(vehicle, v_kmh, hvac_mode, **kwargs)
    return e_per_km * distance_km


def soc_change(vehicle: VehicleParams,
               distance_km: float,
               v_kmh: float,
               hvac_mode: str = "off",
               **kwargs) -> float:
    """
    Bir segment icin SoC degisimi (0-1 arasi birim).
    Negatif = tuketim, Pozitif = geri kazanim.
    """
    e_kwh = energy_for_segment(vehicle, distance_km, v_kmh, hvac_mode, **kwargs)
    return -e_kwh / vehicle.battery_usable_kwh


def temperature_battery_factor(temperature_c: float) -> float:
    """
    Sicakligin batarya verimine etkisi (carpan).
    Sogukta batarya ic direnci artar -> daha fazla enerji kaybi.

    Referans: Steinstraeter et al. (2021) "Effect of Low Temperature on EV Range"

    Returns:
        Carpan (1.0 = optimal, >1.0 = fazla tuketim)
    """
    if temperature_c < -10:
        return 1.40
    elif temperature_c < 0:
        return 1.25 + 0.015 * (-temperature_c)
    elif temperature_c < 10:
        return 1.10 + 0.015 * (10 - temperature_c)
    elif temperature_c < 25:
        return 1.0
    elif temperature_c < 35:
        return 1.0 + 0.005 * (temperature_c - 25)
    else:
        return 1.10 + 0.005 * (temperature_c - 35)


def validate_against_wltp(vehicle: VehicleParams):
    """
    Modeli WLTP / EPA referans degerlerine karsi dogrula.
    IONIQ 5: WLTP tuketim ~16.8 kWh/100km (karisik)
    Tesla Model 3 LR: WLTP tuketim ~14.4 kWh/100km (karisik)
    """
    # Karisik surus profili: sehir ici dusuk hiz + otoban
    speeds = [30, 50, 70, 90, 110, 130]
    weights = [0.10, 0.20, 0.25, 0.25, 0.15, 0.05]  # WLTP agirliklar (yaklasik)

    weighted_consumption = 0.0
    for v, w in zip(speeds, weights):
        e = energy_consumption_kwh_per_km(vehicle, v)
        weighted_consumption += e * w

    wltp_kwh_100km = weighted_consumption * 100

    # WLTP karisik referans (sehir ici ivmelenme/frenleme dahil - modelimiz sabit hiz varsayar)
    known_wltp = {
        "Hyundai IONIQ 5 84 kWh (2024)": 16.8,
        "Tesla Model 3 Long Range (2023+)": 14.4,
    }
    # Otoban referans (sabit hiz, modelimize daha uygun)
    known_highway = {
        "Hyundai IONIQ 5 84 kWh (2024)": {"90": 16.0, "120": 21.0, "130": 24.0},
        "Tesla Model 3 Long Range (2023+)": {"90": 12.5, "120": 16.5, "130": 19.0},
    }

    ref = known_wltp.get(vehicle.name, None)
    hw_ref = known_highway.get(vehicle.name, {})

    print(f"\n{'='*55}")
    print(f"Dogrulama: {vehicle.name}")
    print(f"  WLTP karisik tahmini:  {wltp_kwh_100km:.1f} kWh/100km", end="")
    if ref:
        error = abs(wltp_kwh_100km - ref) / ref * 100
        print(f"  (ref: {ref}, hata: %{error:.0f})")
        print("  NOT: WLTP şehir içi ivmelenme içerir; model sabit hız kullanır, fark beklenendir.")
    else:
        print()

    if hw_ref:
        print(f"  Otoban dogrulamasi (sabit hiz):")
        for spd, ref_val in hw_ref.items():
            model_val = energy_consumption_kwh_per_km(vehicle, float(spd)) * 100
            err = abs(model_val - ref_val) / ref_val * 100
            status = "OK" if err < 15 else "!"
            print(f"    {spd} km/h: model={model_val:.1f}, ref={ref_val}, hata=%{err:.0f} {status}")
    print(f"{'='*55}")

    return wltp_kwh_100km


if __name__ == "__main__":
    # Hizli test ve dogrulama
    for key, v in VEHICLES.items():
        print(f"\n--- {v.name} ---")
        for speed in [80, 100, 120, 140, 160]:
            e = energy_consumption_kwh_per_km(v, speed) * 100
            print(f"  {speed} km/h -> {e:.1f} kWh/100km")

        # Egim testi
        e_flat = energy_consumption_kwh_per_km(v, 120, grade_percent=0) * 100
        e_up = energy_consumption_kwh_per_km(v, 120, grade_percent=5) * 100
        e_down = energy_consumption_kwh_per_km(v, 120, grade_percent=-5) * 100
        print(f"  120 km/h duz: {e_flat:.1f}, yukari %5: {e_up:.1f}, asagi %5: {e_down:.1f}")

        # WLTP dogrulama
        validate_against_wltp(v)
