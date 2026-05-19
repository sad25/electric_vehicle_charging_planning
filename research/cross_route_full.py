# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Saadettin Yıldırım (Yıldsam Teknoloji)
"""
EN: FULL cross-route generalization sweep over ALL 432 scenario-keyed
    models. Every trained scenario model (vehicle x route x driver x
    weather x weekday/weekend, for DQN/DoubleDQN/PPO) is evaluated on
    EVERY route. The scenario condition (vehicle, driver, weather,
    weekend) is held FIXED and only the route is swapped, so the metric
    isolates pure road-network transfer.

    The environment is fully deterministic (no per-episode randomness;
    traffic/deviation are deterministic functions of position+hour, and
    the seed has no effect). Evaluation therefore uses ONE deterministic
    run per (model, test route) at the SAME departure hour used in
    training (08:00). Statistics come from the genuine population of
    scenario conditions: each (algorithm, train_route -> test_route)
    cell aggregates the 2x3x4x2 = 48 scenario instances, reported as an
    arrival proportion with a Wilson 95% confidence interval plus
    mean +/- std for time / arrival-SoC / detour.

TR: TÜM 432 senaryo modeli üzerinde TAM çapraz-rota genelleme süpürmesi.
    Çevre tamamen deterministik (episode rastgeleliği yok; trafik/sapma
    konum+saatin deterministik fonksiyonu, seed etkisiz). Bu yüzden her
    (model, test rotası) için eğitimdeki kalkış saatiyle (08:00) TEK
    deterministik koşu yapılır. İstatistik, gerçek senaryo koşulu
    popülasyonundan gelir: her (algoritma, eğitim-rota -> test-rota)
    hücresi 2x3x4x2 = 48 senaryo örneğini toplar; varış oranı Wilson
    %95 güven aralığıyla, süre/varış-SoC/sapma ortalama ± std ile.

Outputs / Çıktılar:
  data/processed/results/cross_route_matrix_full.csv      (model bazında)
  data/processed/results/cross_route_matrix_full.json
  data/processed/results/cross_route_summary.csv          (hücre + GA)
"""

import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from evroute import make_env
from evroute.config import get_results_dir

# EN: Sibling module — works both as `python research/cross_route_full.py`
#     (research/ on sys.path) and `python -m research.cross_route_full`.
# TR: Kardeş modül — hem `python research/cross_route_full.py` hem de
#     `python -m research.cross_route_full` ile çalışır.
try:
    from cross_route_eval import _eval_episode
except ImportError:
    from research.cross_route_eval import _eval_episode

RESULTS_DIR = get_results_dir()
MODELS_DIR = RESULTS_DIR / "models"

ROUTES = ["istanbul_ankara", "istanbul_izmir", "ankara_antalya"]
DRIVERS = ["eco", "normal", "aggressive"]
VEHICLES = ["ioniq5", "tesla3"]

# EN: Training always departed at 08:00 (run_all_experiments default);
#     keep it fixed so we measure route transfer, not departure-time
#     generalization. The env is deterministic, so n=1 run is exact.
# TR: Eğitim hep 08:00'de kalkıyordu; sabit tutuyoruz ki saat
#     genellemesini değil rota transferini ölçelim. Çevre deterministik,
#     tek koşu kesin sonuç verir.
DEPARTURE_HOUR = 8.0

ALGO_PREFIX = {"dqn": "dqn", "double_dqn": "ddqn", "ppo": "ppo"}


def _wilson_ci(k: int, n: int, z: float = 1.96):
    """
    TR: Varış oranı için Wilson %95 güven aralığı (binom oran;
        n=48 gibi orta örneklemde normal yaklaşımdan daha doğru).
    """
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (p, max(0.0, center - half), min(1.0, center + half))


def _parse_scenario(skey: str):
    """'<arac>_<rota>_<surucu>_<hava>_<weekday|weekend>' ayrıştır."""
    traffic = skey.rsplit("_", 1)[-1]
    is_weekend = (traffic == "weekend")
    body = skey[: -(len(traffic) + 1)]

    vehicle = body.split("_", 1)[0]
    rest = body[len(vehicle) + 1:]

    route = next((r for r in ROUTES if rest.startswith(r + "_")), None)
    if route is None:
        return None
    rest2 = rest[len(route) + 1:]

    driver = next((d for d in DRIVERS if rest2.startswith(d + "_")), None)
    if driver is None:
        return None
    weather = rest2[len(driver) + 1:]

    return {"vehicle": vehicle, "route": route, "driver": driver,
            "weather": weather, "is_weekend": is_weekend}


def _load_agent(algo: str, model_path: Path):
    if algo in ("dqn", "double_dqn"):
        from evroute.agents.dqn import DQNAgent
        agent = DQNAgent(state_dim=14, action_dim=25,
                         double_dqn=(algo == "double_dqn"))
        agent.load(str(model_path))

        class _W:
            def __init__(self, a): self.a = a
            def select_action(self, obs, eval_mode=True):
                return self.a.select_action(obs, eval_mode=True)
        return _W(agent)

    if algo == "ppo":
        from stable_baselines3 import PPO
        model = PPO.load(str(model_path))

        class _P:
            def __init__(self, m): self.m = m
            def select_action(self, obs, eval_mode=True):
                action, _ = self.m.predict(obs, deterministic=True)
                return int(action)
        return _P(model)
    return None


def main():
    t0 = time.time()
    matrix = []

    jobs = []
    for algo, prefix in ALGO_PREFIX.items():
        ext = ".zip" if algo == "ppo" else ".pt"
        for mf in sorted(MODELS_DIR.glob(f"{prefix}__*{ext}")):
            skey = mf.name[len(prefix) + 2: -len(ext)]
            jobs.append((algo, skey, mf))

    total = len(jobs)
    print(f"TAM ÇAPRAZ-ROTA SÜPÜRMESİ: {total} model x {len(ROUTES)} rota "
          f"= {total * len(ROUTES)} deterministik koşu (kalkış 08:00)\n")

    for i, (algo, skey, mf) in enumerate(jobs, 1):
        sc = _parse_scenario(skey)
        if sc is None:
            print(f"[{i}/{total}] [skip] ayrıştırılamadı: {skey}")
            continue
        try:
            agent = _load_agent(algo, mf)
        except Exception as e:
            print(f"[{i}/{total}] [skip] yüklenemedi {skey}: {e}")
            continue

        for test_route in ROUTES:
            env = make_env(vehicle=sc["vehicle"], route=test_route,
                           driver=sc["driver"], weather=sc["weather"],
                           departure_hour=DEPARTURE_HOUR,
                           is_weekend=sc["is_weekend"], seed=42)
            r = _eval_episode(env, agent=agent)  # tek deterministik koşu
            matrix.append({
                "algorithm": algo, "vehicle": sc["vehicle"],
                "train_route": sc["route"], "test_route": test_route,
                "driver": sc["driver"], "weather": sc["weather"],
                "is_weekend": sc["is_weekend"],
                "is_diagonal": sc["route"] == test_route,
                "arrived": r["arrived"], "time_h": r["time_h"],
                "arrival_soc": r["arrival_soc"], "detour_km": r["detour_km"],
                "cost_tl": r["cost_tl"]})

        if i % 25 == 0 or i == total:
            el = time.time() - t0
            print(f"[{i}/{total}] {el/60:.1f}dk geçti, "
                  f"ETA ~{el/i*(total-i)/60:.0f}dk")

    # --- model bazında ham çıktı ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "cross_route_matrix_full.json", "w",
              encoding="utf-8") as f:
        json.dump(matrix, f, ensure_ascii=False, indent=2)
    with open(RESULTS_DIR / "cross_route_matrix_full.csv", "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(matrix[0].keys()))
        w.writeheader()
        w.writerows(matrix)

    # --- hücre (algoritma x eğitim-rota x test-rota) istatistikleri ---
    cells = defaultdict(list)
    for r in matrix:
        cells[(r["algorithm"], r["train_route"], r["test_route"])].append(r)

    summary = []
    for (algo, tr, te), rs in sorted(cells.items()):
        n = len(rs)
        k = int(sum(x["arrived"] for x in rs))
        p, lo, hi = _wilson_ci(k, n)
        th = np.array([x["time_h"] for x in rs])
        soc = np.array([x["arrival_soc"] for x in rs])
        dev = np.array([x["detour_km"] for x in rs])
        summary.append({
            "algorithm": algo, "train_route": tr, "test_route": te,
            "is_diagonal": tr == te, "n_scenarios": n,
            "arrived_k": k, "arrival_rate": round(p, 4),
            "ci95_low": round(lo, 4), "ci95_high": round(hi, 4),
            "time_h_mean": round(float(th.mean()), 3),
            "time_h_std": round(float(th.std(ddof=1)) if n > 1 else 0.0, 3),
            "arrival_soc_mean": round(float(soc.mean()), 4),
            "arrival_soc_std": round(float(soc.std(ddof=1)) if n > 1 else 0.0, 4),
            "detour_km_mean": round(float(dev.mean()), 2)})

    with open(RESULTS_DIR / "cross_route_summary.csv", "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    print(f"\nSaved: {RESULTS_DIR/'cross_route_matrix_full.csv'} "
          f"({len(matrix)} satır)")
    print(f"Saved: {RESULTS_DIR/'cross_route_summary.csv'} "
          f"({len(summary)} hücre)")

    # --- özet ---
    diag = [r for r in matrix if r["is_diagonal"]]
    off = [r for r in matrix if not r["is_diagonal"]]

    def _rate(rows):
        n = len(rows)
        k = int(sum(x["arrived"] for x in rows))
        p, lo, hi = _wilson_ci(k, n)
        return f"{p:.1%} [%95 GA {lo:.1%}–{hi:.1%}] (n={n})"

    print(f"\n=== GENEL ===")
    print(f"Kendi rotası (köşegen)   varış: {_rate(diag)}")
    print(f"Çapraz-rota (köşegen dışı) varış: {_rate(off)}")

    print(f"\n=== ALGORİTMA BAZINDA ===")
    for algo in ALGO_PREFIX:
        d = [r for r in diag if r["algorithm"] == algo]
        o = [r for r in off if r["algorithm"] == algo]
        print(f"  {algo:11s} kendi={_rate(d)}")
        print(f"  {algo:11s} çapraz={_rate(o)}")

    print(f"\n=== EĞİTİM ROTASI BAZINDA (çapraz) — 'orta model' hipotezi ===")
    for tr in ROUTES:
        o = [r for r in off if r["train_route"] == tr]
        print(f"  {tr:16s} -> başka rotalarda {_rate(o)}")

    by_model = defaultdict(list)
    for r in matrix:
        by_model[(r["algorithm"], r["vehicle"], r["train_route"],
                  r["driver"], r["weather"], r["is_weekend"])].append(
                      r["arrived"])
    universal = [k for k, v in by_model.items()
                 if len(v) == 3 and min(v) >= 0.999]
    print(f"\n=== EVRENSEL GENELLEYİCİLER (3/3 rotada varış) ===")
    print(f"  {len(universal)}/{len(by_model)} model TÜM rotalarda vardı")
    for algo in ALGO_PREFIX:
        c = sum(1 for k in universal if k[0] == algo)
        tot = sum(1 for k in by_model if k[0] == algo)
        print(f"  {algo:11s}: {c}/{tot}")


if __name__ == "__main__":
    main()
