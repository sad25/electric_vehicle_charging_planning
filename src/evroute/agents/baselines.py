# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Saadettin Yıldırım (Yıldsam Teknoloji)
"""
Kural Tabanli Baseline Stratejiler
====================================

DQN performansini kiyaslamak icin 4 farkli deterministik strateji.
"""

import numpy as np
from typing import Dict, List
from evroute.env.charging_env import EVChargingEnv
from evroute.env.spaces import SPEEDS, CHARGE_TARGETS


def _speed_to_idx(speed_kmh: int) -> int:
    """En yakin hiz indeksini dondurur."""
    diffs = [abs(s - speed_kmh) for s in SPEEDS]
    return int(np.argmin(diffs))


def _charge_to_idx(target: float) -> int:
    """En yakin sarj hedefi indeksini dondurur (None=0)."""
    if target is None:
        return 0
    diffs = [abs(t - target) if t is not None else 999 for t in CHARGE_TARGETS]
    return int(np.argmin(diffs))


def _make_action(speed_kmh: int, charge_target) -> int:
    """Hiz ve sarj hedefinden aksiyon uret."""
    return _speed_to_idx(speed_kmh) * len(CHARGE_TARGETS) + _charge_to_idx(charge_target)


# ---------- Strateji 1: Her Istasyonda %70 ----------

def always_70_strategy(obs: np.ndarray, info: dict) -> int:
    """Her istasyonda %90'a sarj et, 120 km/h sur. (Guvende kal)"""
    return _make_action(120, 0.90)


# ---------- Strateji 2: Minimum Sarj ----------

def minimum_charge_strategy(obs: np.ndarray, info: dict) -> int:
    """SoC %50 altindaysa %70'e sarj, 120 km/h."""
    soc = obs[0]
    if soc < 0.50:
        return _make_action(120, 0.70)
    return _make_action(120, None)


# ---------- Strateji 3: Hizli Surucu ----------

def fast_drive_strategy(obs: np.ndarray, info: dict) -> int:
    """Hizli sur, SoC %50 altinda %90'a sarj."""
    soc = obs[0]
    if soc < 0.50:
        return _make_action(140, 0.90)
    return _make_action(140, None)


# ---------- Strateji 4: Eko Surucu ----------

def eco_strategy(obs: np.ndarray, info: dict) -> int:
    """Yavas sur, her istasyonda %70'e sarj (guvenli, verimli)."""
    return _make_action(80, 0.70)


BASELINE_STRATEGIES = {
    "always_70": ("Her Istasyonda %70", always_70_strategy),
    "minimum_charge": ("Minimum Sarj", minimum_charge_strategy),
    "fast_drive": ("Hizli Surucu", fast_drive_strategy),
    "eco": ("Eko Surucu", eco_strategy),
}


def run_baseline(env: EVChargingEnv, strategy_fn, num_episodes: int = 50) -> Dict:
    """
    Bir baseline stratejisini degerlendirir.

    Returns:
        dict: Ortalama sonuclar
    """
    results = []

    for _ in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = strategy_fn(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        summary = env.get_trip_summary()
        summary["total_reward"] = total_reward
        results.append(summary)

    avg = {}
    for key in results[0]:
        vals = [r[key] for r in results]
        avg[f"avg_{key}"] = np.mean(vals)
        avg[f"std_{key}"] = np.std(vals)

    return avg


def run_all_baselines(env: EVChargingEnv, num_episodes: int = 50) -> Dict[str, Dict]:
    """Tum baseline stratejileri degerlendirir."""
    results = {}
    for key, (name, fn) in BASELINE_STRATEGIES.items():
        results[key] = run_baseline(env, fn, num_episodes)
        results[key]["name"] = name
    return results


if __name__ == "__main__":
    from evroute import make_env

    env = make_env(seed=42)
    print("Baseline stratejiler degerlendiriliyor...\n")

    all_results = run_all_baselines(env, num_episodes=10)

    print(f"{'Strateji':<25s} {'Sure(h)':>8s} {'Sarj(dk)':>9s} {'Maliyet(TL)':>12s} {'Varis SoC':>10s} {'Reward':>8s}")
    print("-" * 75)
    for key, r in all_results.items():
        print(f"{r['name']:<25s} "
              f"{r['avg_total_time_h']:>8.1f} "
              f"{r['avg_charge_time_min']:>9.0f} "
              f"{r['avg_total_cost_tl']:>12.0f} "
              f"{r['avg_arrival_soc']:>9.0%} "
              f"{r['avg_total_reward']:>8.3f}")
