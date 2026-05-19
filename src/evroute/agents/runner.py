# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Saadettin Yıldırım (Yıldsam Teknoloji)
"""
Simulation runner / Simülasyon koşturucu
========================================

Engine-side trip rollout and trained-agent loading. Pure simulation —
no drawing, no Folium, no HTML. The service/visualisation layer
(`evroute_serve.visualize`) consumes this; the core never imports it.

Motor tarafı yolculuk koşturma ve eğitilmiş ajan yükleme. Saf simülasyon —
çizim/Folium/HTML yok. Servis/görselleştirme katmanı
(`evroute_serve.visualize`) bunu tüketir; çekirdek onu import etmez.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from evroute.config import get_data_dir
from evroute.env.charging_env import EVChargingEnv
from evroute.env.spaces import SPEEDS


def run_simulation(env: EVChargingEnv, strategy_fn=None, agent=None) -> List[Dict]:
    """
    Ortamda bir yolculuk simule eder ve her km icin detayli veri toplar.

    Args:
        env: Gymnasium ortami
        strategy_fn: Baseline strateji fonksiyonu (agent yoksa)
        agent: DQN/PPO agent (varsa)

    Returns:
        Her adim icin detayli veri listesi
    """
    obs, info = env.reset()
    done = False
    trajectory = []

    # Baslangic noktasi
    trajectory.append({
        "km": 0,
        "lat": env.elevation.altitudes_m[0] if hasattr(env, '_route_coords') else 0,
        "soc": env.initial_soc,
        "speed": 0,
        "hour": env.departure_hour,
        "altitude": float(env.elevation.altitude_at_km(0)),
        "event": "start",
        "charge_time_min": 0,
        "cost_tl": 0,
        "total_time_min": 0,
    })

    total_time_min = 0

    while not done:
        if agent is not None:
            action = agent.select_action(obs, eval_mode=True)
        elif strategy_fn is not None:
            action = strategy_fn(obs, info)
        else:
            action = env.action_space.sample()

        soc_before = env.soc
        km_before = env.current_km

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Bu adimin detaylari
        step_data = env.history[-1] if env.history else {}
        speed = step_data.get("speed", SPEEDS[action // 5])
        charge_time = step_data.get("charge_time_min", 0)
        drive_time_h = step_data.get("drive_time_h", 0)

        total_time_min += drive_time_h * 60

        # Surus segmentini km bazinda parcala (animasyon icin)
        km_start = step_data.get("from_km", km_before)
        km_end = step_data.get("to_km", env.current_km)
        segment_km = km_end - km_start

        if segment_km > 0:
            n_points = max(int(segment_km / 2), 3)  # Her 2 km'de bir nokta
            soc_start = soc_before
            soc_end_drive = step_data.get("soc_after", env.soc) if charge_time == 0 else soc_before - (soc_before - step_data.get("soc_after", env.soc))

            # SoC'yi sarj oncesi hesapla
            if charge_time > 0:
                # Surus sonrasi SoC (sarj oncesi)
                energy_consumed = step_data.get("energy_charged", 0)
                soc_after_drive = soc_before - (segment_km * 0.20 / 74)  # Yaklasik
                soc_after_charge = env.soc
            else:
                soc_after_drive = env.soc
                soc_after_charge = env.soc

            for j in range(n_points):
                frac = j / n_points
                km = km_start + frac * segment_km
                soc = soc_before + frac * (soc_after_drive - soc_before)
                cur_hour = env.departure_hour + total_time_min / 60

                # Trafik ve tuketim bilgisi
                traffic_factor = env.traffic.speed_factor_at(km, cur_hour)
                grade = env.elevation.grade_at_km(km)

                # Anlik tuketim (kWh/100km)
                from evroute.models.vehicle import energy_consumption_kwh_per_km
                consumption = energy_consumption_kwh_per_km(
                    env.vehicle, speed, grade_percent=grade,
                    extra_mass_kg=env.load.extra_mass_kg
                ) * 100

                trajectory.append({
                    "km": float(km),
                    "soc": float(np.clip(soc, 0, 1)),
                    "speed": float(speed),
                    "hour": float(cur_hour),
                    "altitude": float(env.elevation.altitude_at_km(km)),
                    "grade": float(grade),
                    "traffic_factor": float(traffic_factor),
                    "consumption_kwh100": float(consumption),
                    "event": "driving",
                    "charge_time_min": 0,
                    "cost_tl": float(env.total_cost_tl),
                    "total_time_min": float(total_time_min * frac + (total_time_min - drive_time_h * 60) * (1 - frac)),
                })

        # Varis noktasi (sarj dahil)
        station_name = step_data.get("station", "")
        event = "charging" if charge_time > 0 else ("dead" if env.soc <= 0 else "driving")

        km_now = float(env.current_km)
        base_alt = float(env.elevation.altitude_at_km(km_now))

        # --- SAPMA animasyonu: yoldan çık -> istasyon -> yola dön ---
        out_path = step_data.get("detour_out_path") or []
        back_path = step_data.get("detour_back_path") or []
        detour_min = float(step_data.get("detour_min", 0) or 0)
        soc_before_charge = float(step_data.get("soc_before", env.soc))

        if charge_time > 0 and out_path:
            # Gidiş: yoldan istasyona (SoC hafifçe düşer)
            n_out = len(out_path)
            for j, p in enumerate(out_path):
                frac = (j + 1) / n_out
                trajectory.append({
                    "km": km_now,
                    "soc": float(soc_before_charge),
                    "speed": 45.0,
                    "hour": float(info["hour"]),
                    "altitude": float(p[2]) if len(p) > 2 else base_alt,
                    "event": "detour",
                    "station": station_name,
                    "charge_time_min": 0,
                    "cost_tl": float(env.total_cost_tl),
                    "total_time_min": float(total_time_min + (detour_min / 2) * frac),
                    "lat": float(p[0]), "lng": float(p[1]),
                })
            total_time_min += detour_min / 2

        if charge_time > 0:
            total_time_min += charge_time

        # İstasyon/şarj noktası (sapma varsa istasyon koordinatında)
        chg_pt = {
            "km": km_now,
            "soc": float(env.soc),
            "speed": 0 if charge_time > 0 else float(speed),
            "hour": float(info["hour"]),
            "altitude": (float(out_path[-1][2]) if (charge_time > 0 and out_path
                          and len(out_path[-1]) > 2) else base_alt),
            "event": event,
            "station": station_name,
            "charge_time_min": float(charge_time),
            "cost_tl": float(env.total_cost_tl),
            "total_time_min": float(total_time_min),
        }
        if charge_time > 0 and out_path:
            chg_pt["lat"] = float(out_path[-1][0])
            chg_pt["lng"] = float(out_path[-1][1])
        trajectory.append(chg_pt)

        # Dönüş: istasyondan yola
        if charge_time > 0 and back_path:
            n_back = len(back_path)
            for j, p in enumerate(back_path):
                frac = (j + 1) / n_back
                trajectory.append({
                    "km": km_now,
                    "soc": float(env.soc),
                    "speed": 45.0,
                    "hour": float(info["hour"]),
                    "altitude": float(p[2]) if len(p) > 2 else base_alt,
                    "event": "detour",
                    "station": station_name,
                    "charge_time_min": 0,
                    "cost_tl": float(env.total_cost_tl),
                    "total_time_min": float(total_time_min + (detour_min / 2) * frac),
                    "lat": float(p[0]), "lng": float(p[1]),
                })
            total_time_min += detour_min / 2

    return trajectory


class _PPOWrapper:
    def __init__(self, model):
        self.model = model

    def select_action(self, obs, eval_mode: bool = True) -> int:
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)


class _DQNWrapper:
    def __init__(self, agent):
        self.agent = agent

    def select_action(self, obs, eval_mode: bool = True) -> int:
        return self.agent.select_action(obs, eval_mode=True)


_RL_AGENT_CACHE: Dict[str, object] = {}


def _resolve_model_file(results_dir: Path, algo_key: str, route_key: str) -> Optional[Path]:
    """
    Sadece rotaya ozel olarak egitilmis modeli dondurur.
    Tek istisna: kok dosyalar (dqn_model.pt vb.) zaten istanbul_ankara'da
    egitildigi icin yalnizca o rota icin yedek olarak kabul edilir.
    Boylece gorsellestirmede sahte/yanlis veri olmaz.
    """
    if algo_key == "ppo":
        ext, stem = ".zip", "ppo"
    else:
        ext = ".pt"
        stem = "ddqn" if algo_key == "double_dqn" else "dqn"

    route_specific = results_dir / f"{stem}_{route_key}{ext}"
    if route_specific.exists():
        return route_specific

    # Fallback file name, valid only for istanbul_ankara.
    # Yedek dosya adı; yalnızca istanbul_ankara için geçerlidir.
    if route_key == "istanbul_ankara":
        fallback = results_dir / f"{stem}_model{ext}"
        if fallback.exists():
            return fallback

    return None


def _load_rl_agent(algo_key: str, route_key: str = "istanbul_ankara"):
    """DQN / Double DQN / PPO ajanını yükler (rotaya özel, cache'li)."""
    cache_key = f"{algo_key}::{route_key}"
    if cache_key in _RL_AGENT_CACHE:
        return _RL_AGENT_CACHE[cache_key]

    results_dir = get_data_dir() / "processed" / "results"
    model_file = _resolve_model_file(results_dir, algo_key, route_key)
    if model_file is None:
        return None

    if algo_key in ("dqn", "double_dqn"):
        from evroute.agents.dqn import DQNAgent
        agent = DQNAgent(state_dim=14, action_dim=25,
                         double_dqn=(algo_key == "double_dqn"))
        agent.load(str(model_file))
        wrapper = _DQNWrapper(agent)
    elif algo_key == "ppo":
        try:
            from stable_baselines3 import PPO
        except ImportError:
            return None
        model = PPO.load(str(model_file))
        wrapper = _PPOWrapper(model)
    else:
        return None

    _RL_AGENT_CACHE[cache_key] = wrapper
    return wrapper


# Public aliases / Genel takma adlar
load_rl_agent = _load_rl_agent

__all__ = ["run_simulation", "load_rl_agent", "_load_rl_agent",
           "_DQNWrapper", "_PPOWrapper", "_resolve_model_file"]
