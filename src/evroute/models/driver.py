# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Saadettin Yıldırım (Yıldsam Teknoloji)
"""
Surucu Profili (Persona) Modeli
================================

Literatur tabanli 3 surucu tipi: Eko, Normal, Agresif.
Her profil farkli hiz tercihi, sarj sabri, mola aliskanligi ve
konfor beklentisi tanimlar.

Referanslar:
- Franke & Krems (2013) "Interacting with limited mobility resources:
  Psychological range levels in electric vehicle use"
- Rauh et al. (2015) "Understanding the Impact of Electric Vehicle
  Driving Experience on Range Anxiety"
- Maister (2005) "The Psychology of Waiting Lines"
- Antonides et al. (2002) "Consumer Perception and Evaluation of Waiting Time"
- EU Regulation 561/2006 - Surucu dinlenme suresi kurallari
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DriverProfile:
    """Surucu persona parametreleri."""
    name: str
    description: str

    # --- Hiz tercihi ---
    preferred_speed_kmh: float      # Tercih ettigi otoban hizi
    speed_variance_kmh: float       # Hiz varyasyonu (+/-)
    speed_tolerance: float          # Hiz sinirini asma toleransi (0=uyar, 0.2=%20 asar)

    # --- Sarj sabri (bekleme toleransi) ---
    patience_threshold_min: float   # Bu sureye kadar rahat bekler
    patience_decay_rate: float      # Esik sonrasi ceza buyume hizi

    # --- Range anxiety ---
    min_comfortable_soc: float      # Bu SoC'nin altinda tedirgin olur (0-1)
    range_anxiety_weight: float     # Anxiety cezasi agirligi

    # --- Mola aliskanligi ---
    max_continuous_drive_min: float  # Mola vermeden max surus suresi (dk)
    preferred_break_min: float      # Tercih ettigi mola suresi (dk)

    # --- HVAC kullanimi ---
    hvac_mode: str                  # "off", "mild", "cooling", "heating"

    # --- Rejeneratif frenleme ---
    regen_efficiency_bonus: float   # 0.0 = standart, 0.1 = agresif regen (%10 bonus)

    # --- Reward agirliklari ---
    w_time: float       # Sure minimizasyonu agirligi
    w_cost: float       # Maliyet minimizasyonu agirligi
    w_comfort: float    # Konfor agirligi (mola, bekleme)
    w_anxiety: float    # Range anxiety cezasi agirligi
    w_degradation: float  # Batarya yipranma cezasi agirligi


# ---------- 3 Persona Tanimi ----------

ECO_DRIVER = DriverProfile(
    name="Eko Surucu",
    description="Yavas ama verimli surer, maliyet odakli, sabırli, batarya sagligina dikkat eder",
    preferred_speed_kmh=90,
    speed_variance_kmh=5,
    speed_tolerance=0.0,        # Hiz sinirına tamamen uyar
    patience_threshold_min=45,
    patience_decay_rate=0.5,
    min_comfortable_soc=0.30,
    range_anxiety_weight=0.8,
    max_continuous_drive_min=120,   # 2 saat
    preferred_break_min=25,
    hvac_mode="mild",
    regen_efficiency_bonus=0.05,   # Agresif regen kullanir
    w_time=0.3,
    w_cost=0.35,
    w_comfort=0.15,
    w_anxiety=0.10,
    w_degradation=0.10,
)

NORMAL_DRIVER = DriverProfile(
    name="Normal Surucu",
    description="Ortalama hiz, makul sabir, dengeli tercihler",
    preferred_speed_kmh=120,
    speed_variance_kmh=10,
    speed_tolerance=0.10,       # %10 uzerinde surer (120 limitte 132'ye cikar)
    patience_threshold_min=30,
    patience_decay_rate=1.0,
    min_comfortable_soc=0.20,
    range_anxiety_weight=0.5,
    max_continuous_drive_min=150,   # 2.5 saat
    preferred_break_min=20,
    hvac_mode="cooling",
    regen_efficiency_bonus=0.0,
    w_time=0.40,
    w_cost=0.25,
    w_comfort=0.15,
    w_anxiety=0.10,
    w_degradation=0.10,
)

AGGRESSIVE_DRIVER = DriverProfile(
    name="Agresif Surucu",
    description="Hizli surer, sabırsiz, sureyi minimize etmek ister",
    preferred_speed_kmh=150,
    speed_variance_kmh=15,
    speed_tolerance=0.25,       # %25 uzerinde surer (120 limitte 150'ye cikar)
    patience_threshold_min=15,
    patience_decay_rate=2.0,
    min_comfortable_soc=0.10,
    range_anxiety_weight=0.2,
    max_continuous_drive_min=180,   # 3 saat
    preferred_break_min=15,
    hvac_mode="cooling",
    regen_efficiency_bonus=-0.05,  # Daha az regen (gec frenleme)
    w_time=0.55,
    w_cost=0.10,
    w_comfort=0.15,
    w_anxiety=0.05,
    w_degradation=0.15,
)

DRIVER_PROFILES = {
    "eco": ECO_DRIVER,
    "normal": NORMAL_DRIVER,
    "aggressive": AGGRESSIVE_DRIVER,
}


# ---------- Yuk Senaryolari ----------

@dataclass
class LoadScenario:
    """Arac yuk senaryosu."""
    name: str
    extra_mass_kg: float    # Ek kutle (yolcu + bagaj)
    hvac_override: Optional[str]  # None = profil default, str = override
    description: str


LOAD_SCENARIOS = {
    "hafif": LoadScenario(
        name="Hafif (1 kisi)",
        extra_mass_kg=85,
        hvac_override=None,
        description="Tek surucu, 10 kg bagaj",
    ),
    "normal": LoadScenario(
        name="Normal (2 kisi)",
        extra_mass_kg=170,
        hvac_override=None,
        description="2 yolcu, 20 kg bagaj, klima acik",
    ),
    "agir": LoadScenario(
        name="Agir (4-5 kisi)",
        extra_mass_kg=400,
        hvac_override="cooling",
        description="4-5 yolcu, 40+ kg bagaj, klima acik",
    ),
}


# ---------- Mola-Sarj Senkronizasyonu ----------

def break_charge_overlap(continuous_drive_min: float,
                         max_continuous_drive_min: float,
                         preferred_break_min: float,
                         charge_time_min: float) -> dict:
    """
    Mola ve sarj suresinin cakismasini (overlap) hesaplar.

    Eger surucu mola zamani gelmisse VE sarj yapiyorsa,
    mola suresi sarj suresiyle overlap eder = bedava zaman!

    Args:
        continuous_drive_min: Mevcut kesintisiz surus suresi (dk)
        max_continuous_drive_min: Mola vermeden max surus suresi (dk)
        preferred_break_min: Tercih edilen mola suresi (dk)
        charge_time_min: Sarj suresi (dk)

    Returns:
        dict:
            needs_break: bool - Mola gerekli mi
            break_time: float - Mola suresi (dk)
            charge_time: float - Sarj suresi (dk)
            effective_time: float - Efektif toplam sure (overlap sonrasi)
            saved_time: float - Tasarruf edilen sure (dk)
            overlap_ratio: float - Overlap orani (0-1)
    """
    needs_break = continuous_drive_min >= max_continuous_drive_min
    break_time = preferred_break_min if needs_break else 0.0

    if needs_break and charge_time_min > 0:
        # Mola ve sarj cakisiyor!
        effective_time = max(charge_time_min, break_time)
        saved_time = break_time + charge_time_min - effective_time
        overlap = min(break_time, charge_time_min)
        overlap_ratio = overlap / max(break_time + charge_time_min, 0.01)
    else:
        effective_time = charge_time_min + break_time
        saved_time = 0.0
        overlap_ratio = 0.0

    return {
        "needs_break": needs_break,
        "break_time": break_time,
        "charge_time": charge_time_min,
        "effective_time": effective_time,
        "saved_time": saved_time,
        "overlap_ratio": overlap_ratio,
    }


def fatigue_penalty(continuous_drive_min: float,
                    max_continuous_drive_min: float) -> float:
    """
    Yorgunluk cezasi.
    Mola vermeden surmek giderek artan ceza uretir.

    Referans: EU Regulation 561/2006 - max 4.5 saat kesintisiz surus
              (Biz daha kisa sureler kullaniyoruz, persona bazli)
    """
    overtime = max(0, continuous_drive_min - max_continuous_drive_min)
    # Karesel artis: 10 dk gecikme kucuk, 60 dk gecikme buyuk ceza
    return (overtime / 60.0) ** 2


def patience_penalty(charge_time_min: float,
                     patience_threshold_min: float,
                     decay_rate: float = 1.0) -> float:
    """
    Sarj bekleme sabri cezasi.
    Esik suresinin ustundeki her dakika karesel ceza uretir.

    Referans: Maister (2005), Antonides et al. (2002)
    Bekleme psikolojisi: ilk dakikalar tolere edilir,
    sonra honutsuzluk hizla artar.
    """
    overtime = max(0, charge_time_min - patience_threshold_min)
    return decay_rate * (overtime / 30.0) ** 2


def range_anxiety_penalty(current_soc: float,
                          min_comfortable_soc: float,
                          weight: float = 1.0) -> float:
    """
    Range anxiety (menzil kaygisi) cezasi.
    SoC rahat seviyenin altina dustugunde artan ceza.

    Referans: Franke & Krems (2013) "range comfort zone"
    """
    deficit = max(0, min_comfortable_soc - current_soc)
    return weight * (deficit / 0.20) ** 2


if __name__ == "__main__":
    print("=== Surucu Profilleri ===")
    for key, p in DRIVER_PROFILES.items():
        print(f"\n{p.name} ({key}):")
        print(f"  Hiz: {p.preferred_speed_kmh} km/h (+/-{p.speed_variance_kmh})")
        print(f"  Sarj sabri: {p.patience_threshold_min} dk")
        print(f"  Min rahat SoC: %{p.min_comfortable_soc*100:.0f}")
        print(f"  Max kesintisiz surus: {p.max_continuous_drive_min} dk")
        print(f"  Mola suresi: {p.preferred_break_min} dk")
        print(f"  HVAC: {p.hvac_mode}")
        print(f"  Reward agirliklari: time={p.w_time}, cost={p.w_cost}, "
              f"comfort={p.w_comfort}, anxiety={p.w_anxiety}, degrad={p.w_degradation}")

    print("\n=== Mola-Sarj Overlap Ornekleri ===")
    # 2.5 saat surmis, mola gerekli, 25 dk sarj
    result = break_charge_overlap(150, 150, 20, 25)
    print(f"  150dk surus, 25dk sarj: efektif={result['effective_time']:.0f}dk, "
          f"tasarruf={result['saved_time']:.0f}dk")

    # 1 saat surmis, mola yok, 15 dk sarj
    result = break_charge_overlap(60, 150, 20, 15)
    print(f"  60dk surus, 15dk sarj: efektif={result['effective_time']:.0f}dk, "
          f"tasarruf={result['saved_time']:.0f}dk")

    print("\n=== Sabir Cezasi ===")
    for t in [10, 20, 30, 40, 50, 60]:
        eco_p = patience_penalty(t, ECO_DRIVER.patience_threshold_min, ECO_DRIVER.patience_decay_rate)
        agg_p = patience_penalty(t, AGGRESSIVE_DRIVER.patience_threshold_min, AGGRESSIVE_DRIVER.patience_decay_rate)
        print(f"  {t}dk sarj: Eko ceza={eco_p:.2f}, Agresif ceza={agg_p:.2f}")
