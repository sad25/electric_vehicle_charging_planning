# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Saadettin Yıldırım (Yıldsam Teknoloji)
"""
Non-Lineer Sarj Egrisi Modeli
===============================

Gercek olcum verilerinden (Fastned/Figshare) SoC-bagli sarj gucu egrisini
yukler ve numerik integrasyon ile sarj suresini hesaplar.

Referanslar:
- Fastned charging data (https://figshare.com)
- Pelletier et al. (2017) "Battery degradation and optimal charging for EVs"
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
from dataclasses import dataclass

from evroute.config import get_data_dir


def _curves_dir() -> Path:
    """
    Charging-curve CSV directory, resolved via the engine data dir.

    Şarj eğrisi CSV dizini; motor veri dizini üzerinden çözülür.

    Resolved lazily so it follows ``EVROUTE_DATA_DIR`` / cwd.

    ``EVROUTE_DATA_DIR`` / cwd'yi izlemek için tembel çözülür.
    """
    return get_data_dir() / "raw" / "charging_curves"


@dataclass
class ChargingCurve:
    """Bir aracin sarj egrisi verisi."""
    name: str
    soc_points: np.ndarray     # SoC degerleri (0-100)
    power_points: np.ndarray   # Sarj gucu (kW)

    def power_at_soc(self, soc_percent: float) -> float:
        """Verilen SoC'de sarj gucunu interpolasyon ile dondurur (kW)."""
        soc_clamped = np.clip(soc_percent, self.soc_points[0], self.soc_points[-1])
        return float(np.interp(soc_clamped, self.soc_points, self.power_points))


def load_charging_curve(csv_path: str, name: str = "") -> ChargingCurve:
    """
    CSV dosyasindan sarj egrisi yukler.
    CSV formati: SoC(%), Power(kW) - header'siz veya header'li.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Sarj egrisi dosyasi bulunamadi: {csv_path}")

    # Header olup olmadigini kontrol et
    with open(path, 'r') as f:
        first_line = f.readline().strip()

    try:
        float(first_line.split(',')[0].strip())
        df = pd.read_csv(path, header=None, names=['soc', 'power'])
    except ValueError:
        df = pd.read_csv(path)
        df.columns = ['soc', 'power']

    # Temizle ve sirala
    df = df.dropna()
    df = df.sort_values('soc').reset_index(drop=True)

    # Ayni SoC degerlerinde ortalama al
    df = df.groupby('soc').mean().reset_index()

    soc = df['soc'].values
    power = df['power'].values

    # 0-100 araligina ekstrapolasyon
    soc_full = np.linspace(0, 100, 1001)
    power_full = np.interp(soc_full, soc, power)

    # 0% ve 100% uclarinda guc dusurme (fiziksel gerceklik)
    # Dusuk SoC'de guc sinirli olabilir
    for i in range(len(soc_full)):
        if soc_full[i] < soc[0]:
            power_full[i] = power[0] * (soc_full[i] / max(soc[0], 1))
        elif soc_full[i] > soc[-1]:
            # Yuksek SoC'de guc hizla duser
            remaining = (100 - soc_full[i]) / max(100 - soc[-1], 1)
            power_full[i] = power[-1] * max(remaining, 0.05)

    return ChargingCurve(
        name=name or path.stem,
        soc_points=soc_full,
        power_points=power_full,
    )


# ---------- Onceden Tanimli Sarj Egrileri ----------

def load_ioniq5_curve() -> ChargingCurve:
    return load_charging_curve(
        _curves_dir() / "ioniq5_84kwh_300kw.csv",
        name="Hyundai IONIQ 5 84 kWh"
    )


def load_tesla3_curve() -> ChargingCurve:
    return load_charging_curve(
        _curves_dir() / "tesla_model3_lr_300kw.csv",
        name="Tesla Model 3 Long Range"
    )


def calculate_charge_time(curve: ChargingCurve,
                          soc_start_pct: float,
                          soc_end_pct: float,
                          station_power_kw: float,
                          battery_usable_kwh: float,
                          temperature_factor: float = 1.0) -> float:
    """
    Numerik integrasyon ile sarj suresini hesaplar (dakika).

    Sarj gucu = min(egri_gucu, istasyon_gucu) / sicaklik_faktoru

    Args:
        curve: Aracin sarj egrisi
        soc_start_pct: Baslangic SoC (%, 0-100)
        soc_end_pct: Hedef SoC (%, 0-100)
        station_power_kw: Istasyon maksimum gucu (kW)
        battery_usable_kwh: Kullanilabilir batarya kapasitesi (kWh)
        temperature_factor: Sicaklik etkisi carpani (>1 = soguk, yavas sarj)
    Returns:
        Sarj suresi (dakika)
    """
    if soc_end_pct <= soc_start_pct:
        return 0.0

    soc_start = np.clip(soc_start_pct, 0, 100)
    soc_end = np.clip(soc_end_pct, 0, 100)

    # Numerik integrasyon: kucuk SoC adimlariyla
    n_steps = int((soc_end - soc_start) * 10)  # Her %0.1 icin bir adim
    n_steps = max(n_steps, 10)
    soc_steps = np.linspace(soc_start, soc_end, n_steps + 1)

    total_time_hours = 0.0

    for i in range(n_steps):
        soc_mid = (soc_steps[i] + soc_steps[i + 1]) / 2.0
        d_soc = (soc_steps[i + 1] - soc_steps[i]) / 100.0  # 0-1 birim

        # Egri gucu ve istasyon gucu arasindan minimum
        curve_power = curve.power_at_soc(soc_mid)
        effective_power = min(curve_power, station_power_kw)

        # Sicaklik etkisi: sogukta sarj gucu duser
        effective_power = effective_power / temperature_factor

        # Enerji: d_E = d_SoC * kapasite
        d_energy_kwh = d_soc * battery_usable_kwh

        if effective_power > 0.1:
            d_time_hours = d_energy_kwh / effective_power
        else:
            d_time_hours = d_energy_kwh / 0.1  # Minimum guc

        total_time_hours += d_time_hours

    return total_time_hours * 60.0  # Dakikaya cevir


def energy_charged_kwh(soc_start_pct: float,
                       soc_end_pct: float,
                       battery_usable_kwh: float) -> float:
    """Sarj edilen enerji miktari (kWh)."""
    d_soc = (soc_end_pct - soc_start_pct) / 100.0
    return d_soc * battery_usable_kwh


def degradation_stress(target_soc_pct: float,
                       station_power_kw: float) -> float:
    """
    Batarya yipranma stres skoru (0-1).
    Yuksek SoC + yuksek guc = daha fazla stres.

    Referans: Pelletier et al. (2017)
    """
    soc_stress = max(0.0, (target_soc_pct / 100.0) - 0.80) * 5.0  # %80 ustu cezali
    power_stress = min(station_power_kw / 300.0, 1.0)
    return soc_stress * power_stress


if __name__ == "__main__":
    # Test
    print("Sarj egrileri yukleniyor...")

    for loader, bat_kwh in [(load_ioniq5_curve, 74.0), (load_tesla3_curve, 72.0)]:
        curve = loader()
        print(f"\n--- {curve.name} ---")
        print(f"  SoC aralik: {curve.soc_points[0]:.1f}% - {curve.soc_points[-1]:.1f}%")
        print(f"  Max guc: {curve.power_points.max():.1f} kW @ SoC={curve.soc_points[np.argmax(curve.power_points)]:.1f}%")

        # Sarj suresi ornekleri
        for start, end in [(10, 80), (20, 80), (20, 100), (50, 80)]:
            for power in [50, 120, 300]:
                t = calculate_charge_time(curve, start, end, power, bat_kwh)
                print(f"  %{start}->{end} @ {power}kW: {t:.1f} dk")
