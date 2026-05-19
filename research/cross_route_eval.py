# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Saadettin Yıldırım (Yıldsam Teknoloji)
"""
EN: Cross-route generalization evaluation. Each route-specific RL model is
    evaluated on every route, including routes it was NOT trained on, to
    measure how well a learned policy transfers across road networks.
TR: Çapraz-rota genelleme değerlendirmesi. Her rotaya özel RL modeli, ÜZERİNDE
    EĞİTİLMEDİĞİ rotalar dahil tüm rotalarda değerlendirilir; böylece
    öğrenilen politikanın rotalar arası transferi ölçülür.

Outputs / Çıktılar:
  data/processed/results/cross_route_matrix.csv
  data/processed/results/cross_route_matrix.json

EN: Matrix rows = (algorithm, train route), columns = test route; the
    diagonal is the model's own route (upper bound), off-diagonal cells
    quantify generalization.
TR: Matris satırları = (algoritma, eğitim rotası), sütunlar = test rotası;
    köşegen modelin kendi rotası (üst sınır), köşegen-dışı genellemeyi verir.

Usage (after models are trained) / Kullanım (modeller eğitildikten sonra):
    ./venv/bin/python -m src.cross_route_eval
"""

import json
from pathlib import Path

import numpy as np

from evroute import make_env
from evroute.config import get_results_dir
from evroute.agents.runner import run_simulation, _load_rl_agent
from evroute.agents.baselines import BASELINE_STRATEGIES

ROUTES = ["istanbul_ankara", "istanbul_izmir", "ankara_antalya"]
RL_ALGOS = ["dqn", "double_dqn", "ppo"]
RESULTS_DIR = get_results_dir()

# EN: Multiple departure hours are averaged for statistical robustness.
# TR: İstatistiksel kararlılık için birden çok kalkış saati ortalanır.
EVAL_HOURS = [6, 8, 14, 18]
DRIVER = "normal"
VEHICLE = "ioniq5"


def _eval_episode(env, agent=None, strategy_fn=None) -> dict:
    """
    EN: Runs one simulated trip and returns its summary metrics.
    TR: Tek bir simüle yolculuk koşar ve özet metriklerini döndürür.
    """
    run_simulation(env, agent=agent, strategy_fn=strategy_fn)
    s = env.get_trip_summary()
    return {
        "arrived": 1.0 if s["arrival_soc"] > 0.01 else 0.0,
        "time_h": s["total_time_h"],
        "charge_min": s["charge_time_min"],
        "cost_tl": s["total_cost_tl"],
        "arrival_soc": s["arrival_soc"],
        "detour_km": s.get("total_detour_km", 0.0),
    }


def _aggregate(rows):
    """
    EN: Returns the per-metric mean across episodes plus the sample count.
    TR: Bölümler arası metrik ortalamalarını ve örnek sayısını döndürür.
    """
    if not rows:
        return None
    arr = {k: np.mean([r[k] for r in rows]) for k in rows[0]}
    arr["n"] = len(rows)
    return arr


def main():
    matrix = []  # flat records / düz kayıtlar

    # EN: RL — evaluate each (algorithm, train route) model on every test route.
    # TR: RL — her (algoritma, eğitim rotası) modelini her test rotasında dene.
    for algo in RL_ALGOS:
        for train_route in ROUTES:
            agent = _load_rl_agent(algo, train_route)
            if agent is None:
                print(f"[skip] no model for {algo} @ {train_route}")
                continue
            for test_route in ROUTES:
                rows = []
                for hour in EVAL_HOURS:
                    env = make_env(vehicle=VEHICLE, route=test_route,
                                   driver=DRIVER, weather="optimal",
                                   departure_hour=hour, seed=42)
                    rows.append(_eval_episode(env, agent=agent))
                agg = _aggregate(rows)
                rec = {"algorithm": algo, "train_route": train_route,
                       "test_route": test_route,
                       "is_diagonal": train_route == test_route, **agg}
                matrix.append(rec)
                tag = "own-route " if train_route == test_route else "generalize"
                print(f"{algo:11s} train={train_route:16s} test={test_route:16s} "
                      f"{tag:10s} arrived={agg['arrived']:.0%} "
                      f"t={agg['time_h']:.1f}h SoC={agg['arrival_soc']:.0%}")

    # EN: Baseline reference (no training route).
    # TR: Baseline referansı (eğitim rotası yok).
    for sk, (_, fn) in BASELINE_STRATEGIES.items():
        for test_route in ROUTES:
            rows = []
            for hour in EVAL_HOURS:
                env = make_env(vehicle=VEHICLE, route=test_route,
                               driver=DRIVER, weather="optimal",
                               departure_hour=hour, seed=42)
                rows.append(_eval_episode(env, strategy_fn=fn))
            agg = _aggregate(rows)
            matrix.append({"algorithm": f"baseline_{sk}",
                           "train_route": "-", "test_route": test_route,
                           "is_diagonal": False, **agg})

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "cross_route_matrix.json", "w", encoding="utf-8") as f:
        json.dump(matrix, f, ensure_ascii=False, indent=2)

    import csv
    if matrix:
        with open(RESULTS_DIR / "cross_route_matrix.csv", "w", newline="",
                  encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(matrix[0].keys()))
            w.writeheader()
            w.writerows(matrix)

    print(f"\nSaved: {RESULTS_DIR/'cross_route_matrix.csv'}")
    print(f"Saved: {RESULTS_DIR/'cross_route_matrix.json'}")

    # EN: Summary — RL arrival rate on its own route vs. cross-route.
    # TR: Özet — RL'nin kendi rotası vs. çapraz-rota varış oranı.
    diag = [r for r in matrix if r.get("is_diagonal")]
    off = [r for r in matrix if r["algorithm"] in RL_ALGOS
           and not r.get("is_diagonal") and r["train_route"] != "-"]
    if diag and off:
        print(f"\nRL own-route mean arrival:   "
              f"{np.mean([r['arrived'] for r in diag]):.0%}")
        print(f"RL cross-route mean arrival: "
              f"{np.mean([r['arrived'] for r in off]):.0%}")


if __name__ == "__main__":
    main()
