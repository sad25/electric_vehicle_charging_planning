# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Saadettin Yıldırım (Yıldsam Teknoloji)
"""
TAM DENEY MATRISI
==================
3 rota x 2 arac x 3 surucu x 4 hava x 2 trafik = 144 senaryo
Her senaryo icin: DQN, Double DQN, PPO + 4 baseline = 7 algoritma
Toplam: 144 x 7 = 1008 degerlendirme

Sonuclar:
  data/processed/results/full_results.csv          <- Ana tablo
  data/processed/results/training_all.json          <- Ogrenme egrileri
  data/processed/results/experiment_matrix.json     <- Deney matrisi detaylari
"""

import time, json, itertools
import numpy as np
import pandas as pd
from pathlib import Path

from evroute import make_env
from evroute.config import get_results_dir
from evroute.agents.dqn import DQNAgent, train_dqn, evaluate_agent
from evroute.agents.ppo import train_ppo, evaluate_ppo
from evroute.agents.baselines import run_all_baselines

# EN: Output location is configurable via EVROUTE_RESULTS_DIR; defaults to
#     <cwd>/data/processed/results so re-runs resume from the checkpoint.
# TR: Çıktı konumu EVROUTE_RESULTS_DIR ile yapılandırılır; varsayılan
#     <cwd>/data/processed/results — yeniden çalıştırma kontrol noktasından devam eder.
RESULTS_DIR = get_results_dir()

# EN: Every scenario's trained models are stored here (scenario-keyed),
#     enabling cross-condition validation from the visualization.
# TR: Her senaryonun eğitilmiş modelleri burada (senaryo-anahtarlı)
#     saklanır; görselleştirmeden çapraz-koşul doğrulaması yapılabilir.
MODELS_DIR = RESULTS_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# === DENEY MATRISI ===
VEHICLES = ["ioniq5", "tesla3"]
ROUTES = ["istanbul_ankara", "istanbul_izmir", "ankara_antalya"]
DRIVERS = ["eco", "normal", "aggressive"]
WEATHERS = ["optimal", "yaz_gunesli", "kis_soguk", "yagmurlu"]
TRAFFICS = [False, True]  # is_weekend: False=hafta ici, True=hafta sonu

NUM_TRAIN_EPISODES = 1000       # 2000'den dusuruldu - 800'de zaten yakiniyor
PPO_TIMESTEPS = 25000           # 50000'den dusuruldu
NUM_EVAL_EPISODES = 15          # 30'dan dusuruldu - istatistik icin yeterli

all_rows = []
all_curves = {}
scenario_count = 0
total_scenarios = len(VEHICLES) * len(ROUTES) * len(DRIVERS) * len(WEATHERS) * len(TRAFFICS)

# --- DİRENÇLİ KOŞU: istenildiği an durdurulabilir, kaldığı yerden devam ---
# EN: Resumable. Models are saved per scenario; metrics/curves are
#     checkpointed every scenario. On (re)start, prior checkpoint is
#     loaded and scenarios whose 3 models already exist are skipped, so
#     nothing done so far is ever lost.
# TR: Dirençli. Modeller her senaryoda kaydedilir; metrik/eğri her
#     senaryoda checkpoint'lenir. (Yeniden) başlarken önceki checkpoint
#     yüklenir ve 3 modeli zaten olan senaryolar atlanır — o ana kadarki
#     hiçbir şey kaybolmaz.
_ckpt = RESULTS_DIR / "full_results_checkpoint.csv"
if _ckpt.exists():
    try:
        all_rows = pd.read_csv(_ckpt).to_dict("records")
        print(f"[devam] önceki checkpoint yüklendi: {len(all_rows)} satır")
    except Exception:
        all_rows = []
_curves_ckpt = RESULTS_DIR / "training_curves_all.json"
if _curves_ckpt.exists():
    try:
        all_curves = json.load(open(_curves_ckpt))
        print(f"[devam] önceki öğrenme eğrileri yüklendi: {len(all_curves)}")
    except Exception:
        all_curves = {}


def _scenario_done(skey: str) -> bool:
    """3 modeli de diskte varsa bu senaryo tamamlanmış sayılır."""
    return ((MODELS_DIR / f"dqn__{skey}.pt").exists()
            and (MODELS_DIR / f"ddqn__{skey}.pt").exists()
            and (MODELS_DIR / f"ppo__{skey}.zip").exists())

print("=" * 70)
print(f"TAM DENEY MATRISI: {total_scenarios} senaryo")
print(f"Her senaryo: DQN + Double DQN + PPO + 4 baseline")
print(f"Toplam degerlendirme: {total_scenarios * 7}")
print("=" * 70)

t_start = time.time()

for vehicle in VEHICLES:
    for route in ROUTES:
        for driver in DRIVERS:
            for weather in WEATHERS:
                for is_weekend in TRAFFICS:
                    scenario_count += 1
                    traffic_label = "weekend" if is_weekend else "weekday"
                    scenario_key = f"{vehicle}_{route}_{driver}_{weather}_{traffic_label}"

                    # Zaten tamamlanmışsa atla (kaldığı yerden devam).
                    if _scenario_done(scenario_key):
                        print(f"[{scenario_count}/{total_scenarios}] "
                              f"{scenario_key}  -> ATLA (zaten var)")
                        continue

                    elapsed = time.time() - t_start
                    if scenario_count > 1:
                        eta = elapsed / (scenario_count - 1) * (total_scenarios - scenario_count + 1)
                        eta_str = f"ETA: {eta/60:.0f}dk"
                    else:
                        eta_str = ""

                    print(f"\n[{scenario_count}/{total_scenarios}] {scenario_key} {eta_str}")

                    try:
                        env = make_env(
                            vehicle=vehicle, route=route, driver=driver,
                            weather=weather, is_weekend=is_weekend,
                            seed=42
                        )
                    except Exception as e:
                        print(f"  HATA ortam olusturma: {e}")
                        continue

                    base_info = {
                        "vehicle": vehicle,
                        "route": route,
                        "driver": driver,
                        "weather": weather,
                        "traffic": traffic_label,
                        "distance_km": round(env.total_distance, 1),
                        "num_stations": len(env.stations),
                    }

                    # Bu kombinasyon "temsili" mi? Eger oyleyse egitilen modeli
                    # rotaya ozel adla kaydedeceğiz; gorsellestirme bu modelleri kullanir.
                    is_representative = (
                        vehicle == "ioniq5"
                        and driver == "normal"
                        and weather == "optimal"
                        and not is_weekend
                    )

                    # --- DQN ---
                    try:
                        agent_dqn = DQNAgent(state_dim=14, action_dim=25, double_dqn=False)
                        stats_dqn = train_dqn(env, agent_dqn, num_episodes=NUM_TRAIN_EPISODES,
                                              log_interval=99999, verbose=False)
                        eval_dqn = evaluate_agent(env, agent_dqn, NUM_EVAL_EPISODES)
                        all_rows.append({**base_info, "algorithm": "DQN", **{
                            k: round(v, 4) if isinstance(v, float) else v
                            for k, v in eval_dqn.items()
                        }})
                        all_curves[f"{scenario_key}_DQN"] = [float(x) for x in stats_dqn["rewards"]]
                        # EN: Save this scenario's model (scenario-keyed).
                        # TR: Bu senaryonun modelini kaydet (senaryo-anahtarlı).
                        agent_dqn.save(str(MODELS_DIR / f"dqn__{scenario_key}.pt"))
                        if is_representative:
                            save_path = RESULTS_DIR / f"dqn_{route}.pt"
                            agent_dqn.save(str(save_path))
                            print(f"  DQN:      r={eval_dqn['avg_total_reward']:.3f} t={eval_dqn['avg_total_time_h']:.1f}h [model->{save_path.name}]", end="")
                        else:
                            print(f"  DQN:      r={eval_dqn['avg_total_reward']:.3f} t={eval_dqn['avg_total_time_h']:.1f}h", end="")
                    except Exception as e:
                        print(f"  DQN HATA: {e}", end="")

                    # --- Double DQN ---
                    try:
                        agent_ddqn = DQNAgent(state_dim=14, action_dim=25, double_dqn=True)
                        stats_ddqn = train_dqn(env, agent_ddqn, num_episodes=NUM_TRAIN_EPISODES,
                                               log_interval=99999, verbose=False)
                        eval_ddqn = evaluate_agent(env, agent_ddqn, NUM_EVAL_EPISODES)
                        all_rows.append({**base_info, "algorithm": "Double DQN", **{
                            k: round(v, 4) if isinstance(v, float) else v
                            for k, v in eval_ddqn.items()
                        }})
                        all_curves[f"{scenario_key}_DDQN"] = [float(x) for x in stats_ddqn["rewards"]]
                        agent_ddqn.save(str(MODELS_DIR / f"ddqn__{scenario_key}.pt"))
                        if is_representative:
                            save_path = RESULTS_DIR / f"ddqn_{route}.pt"
                            agent_ddqn.save(str(save_path))
                            print(f" | DDQN: r={eval_ddqn['avg_total_reward']:.3f} [model->{save_path.name}]", end="")
                        else:
                            print(f" | DDQN: r={eval_ddqn['avg_total_reward']:.3f}", end="")
                    except Exception as e:
                        print(f" | DDQN HATA: {e}", end="")

                    # --- PPO ---
                    try:
                        ppo_model, ppo_log = train_ppo(env, total_timesteps=PPO_TIMESTEPS,
                                                       seed=42, verbose=False, device="cpu")
                        eval_ppo = evaluate_ppo(ppo_model, env, NUM_EVAL_EPISODES)
                        all_rows.append({**base_info, "algorithm": "PPO", **{
                            k: round(v, 4) if isinstance(v, float) else v
                            for k, v in eval_ppo.items()
                        }})
                        all_curves[f"{scenario_key}_PPO"] = [float(x) for x in ppo_log.episode_rewards]
                        ppo_model.save(str(MODELS_DIR / f"ppo__{scenario_key}.zip"))
                        if is_representative:
                            save_path = RESULTS_DIR / f"ppo_{route}.zip"
                            ppo_model.save(str(save_path))
                            print(f" | PPO:  r={eval_ppo['avg_total_reward']:.3f} [model->{save_path.name}]", end="")
                        else:
                            print(f" | PPO:  r={eval_ppo['avg_total_reward']:.3f}", end="")
                    except Exception as e:
                        print(f" | PPO HATA: {e}", end="")

                    # --- Baselines ---
                    try:
                        br = run_all_baselines(env, NUM_EVAL_EPISODES)
                        for bkey, bval in br.items():
                            all_rows.append({**base_info, "algorithm": bval["name"], **{
                                k: round(v, 4) if isinstance(v, float) else v
                                for k, v in bval.items() if k != "name"
                            }})
                        print(f" | BL: done", end="")
                    except Exception as e:
                        print(f" | BL HATA: {e}", end="")

                    print()  # newline

                    # HER senaryoda checkpoint: istenen an durdurulsa da
                    # o ana kadarki metrikler/eğriler kaybolmaz.
                    pd.DataFrame(all_rows).to_csv(
                        RESULTS_DIR / "full_results_checkpoint.csv", index=False)
                    if scenario_count % 5 == 0:
                        with open(RESULTS_DIR / "training_curves_all.json", "w") as _f:
                            json.dump(all_curves, _f)
                        print(f"  [Checkpoint: {len(all_rows)} satır, "
                              f"{len(all_curves)} eğri]")

# === SONUCLARI KAYDET ===
total_time = time.time() - t_start
print("\n" + "=" * 70)
print(f"TAMAMLANDI! {total_scenarios} senaryo, {len(all_rows)} sonuc satiri")
print(f"Toplam sure: {total_time/60:.1f} dakika")
print("=" * 70)

# Ana tablo
df = pd.DataFrame(all_rows)
df.to_csv(RESULTS_DIR / "full_results.csv", index=False)
print(f"\nAna tablo: {RESULTS_DIR / 'full_results.csv'} ({len(df)} satir)")

# Ozet tablo: algoritma bazinda ortalama
summary = df.groupby("algorithm").agg({
    "avg_total_time_h": "mean",
    "avg_charge_time_min": "mean",
    "avg_total_cost_tl": "mean",
    "avg_arrival_soc": "mean",
    "avg_total_reward": "mean",
}).round(3).sort_values("avg_total_reward", ascending=False)
summary.to_csv(RESULTS_DIR / "summary_by_algorithm.csv")
print(f"\nAlgoritma ozet: {RESULTS_DIR / 'summary_by_algorithm.csv'}")
print(summary.to_string())

# Arac bazinda ozet
summary_v = df.groupby(["vehicle", "algorithm"]).agg({
    "avg_total_reward": "mean",
    "avg_total_time_h": "mean",
}).round(3).sort_values("avg_total_reward", ascending=False)
summary_v.to_csv(RESULTS_DIR / "summary_by_vehicle.csv")
print(f"\nArac ozet: {RESULTS_DIR / 'summary_by_vehicle.csv'}")

# Rota bazinda ozet
summary_r = df.groupby(["route", "algorithm"]).agg({
    "avg_total_reward": "mean",
    "avg_total_time_h": "mean",
}).round(3).sort_values("avg_total_reward", ascending=False)
summary_r.to_csv(RESULTS_DIR / "summary_by_route.csv")
print(f"Rota ozet: {RESULTS_DIR / 'summary_by_route.csv'}")

# Hava bazinda ozet
summary_w = df.groupby(["weather", "algorithm"]).agg({
    "avg_total_reward": "mean",
}).round(3)
summary_w.to_csv(RESULTS_DIR / "summary_by_weather.csv")
print(f"Hava ozet: {RESULTS_DIR / 'summary_by_weather.csv'}")

# Surucu bazinda ozet
summary_d = df.groupby(["driver", "algorithm"]).agg({
    "avg_total_reward": "mean",
}).round(3)
summary_d.to_csv(RESULTS_DIR / "summary_by_driver.csv")
print(f"Surucu ozet: {RESULTS_DIR / 'summary_by_driver.csv'}")

# Ogrenme egrileri
with open(RESULTS_DIR / "training_curves_all.json", "w") as f:
    json.dump(all_curves, f)
print(f"Ogrenme egrileri: {RESULTS_DIR / 'training_curves_all.json'}")

# Deney bilgileri
exp_info = {
    "total_scenarios": total_scenarios,
    "total_results": len(all_rows),
    "total_time_minutes": round(total_time / 60, 1),
    "vehicles": VEHICLES,
    "routes": ROUTES,
    "drivers": DRIVERS,
    "weathers": WEATHERS,
    "traffics": ["weekday", "weekend"],
    "algorithms": ["DQN", "Double DQN", "PPO", "Her Istasyonda %70", "Minimum Sarj", "Hizli Surucu", "Eko Surucu"],
    "train_episodes_dqn": NUM_TRAIN_EPISODES,
    "ppo_timesteps": PPO_TIMESTEPS,
    "eval_episodes": NUM_EVAL_EPISODES,
    "seed": 42,
    "data_sources": {
        "trafik": "Google Directions API",
        "yukseklik": "Google Elevation API",
        "sarj_egrileri": "Fastned/Figshare gercek olcum",
        "istasyonlar": "Open Charge Map API",
        "yol_geometrisi": "Google Directions API polyline",
    },
}
with open(RESULTS_DIR / "experiment_matrix.json", "w") as f:
    json.dump(exp_info, f, indent=2, ensure_ascii=False)

print(f"\n{'='*70}")
print("TUM DOSYALAR:")
print(f"  full_results.csv            <- {len(df)} satirlik ana tablo")
print(f"  summary_by_algorithm.csv    <- Algoritma karsilastirmasi")
print(f"  summary_by_vehicle.csv      <- Arac karsilastirmasi")
print(f"  summary_by_route.csv        <- Rota karsilastirmasi")
print(f"  summary_by_weather.csv      <- Hava durumu etkisi")
print(f"  summary_by_driver.csv       <- Surucu profili etkisi")
print(f"  training_curves_all.json    <- Tum ogrenme egrileri")
print(f"  experiment_matrix.json      <- Deney ayarlari")
print(f"{'='*70}")
