# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Saadettin Yıldırım (Yıldsam Teknoloji)
"""
Environment factory.

Ortam fabrikası.

``make_env`` is the single entry point the rest of the project (research
scripts, server, agents) uses to build an :class:`EVChargingEnv`. A custom
reward function can be injected via ``reward_fn`` without touching the
environment internals.

``make_env`` projenin geri kalanının (deney betikleri, sunucu, ajanlar)
:class:`EVChargingEnv` kurmak için kullandığı tek giriş noktasıdır. Özel
bir ödül fonksiyonu ortam içine dokunmadan ``reward_fn`` ile enjekte
edilebilir.
"""
from __future__ import annotations

from typing import Optional

from evroute.env.charging_env import EVChargingEnv


def make_env(vehicle: str = "ioniq5",
             route: str = "istanbul_ankara",
             driver: str = "normal",
             load: str = "normal",
             weather: str = "optimal",
             departure_hour: float = 8.0,
             is_weekend: bool = False,
             initial_soc: float = 0.80,
             seed: int = 42,
             reward_fn=None) -> EVChargingEnv:
    """Ortam oluşturma yardımcı fonksiyonu. / Environment builder helper."""
    return EVChargingEnv(
        vehicle_key=vehicle,
        route_key=route,
        driver_key=driver,
        load_key=load,
        weather_key=weather,
        departure_hour=departure_hour,
        is_weekend=is_weekend,
        initial_soc=initial_soc,
        seed=seed,
        reward_fn=reward_fn,
    )


if __name__ == "__main__":
    # Hizli test: random agent / Quick smoke test: random agent.
    from evroute.env.spaces import SPEEDS, CHARGE_TARGETS

    env = make_env()
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Initial obs: {obs}")
    print(f"Initial info: {info}")

    total_reward = 0
    done = False
    step = 0

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        done = terminated or truncated
        step += 1

        speed_idx = action // 5
        charge_idx = action % 5
        print(f"  Step {step}: speed={SPEEDS[speed_idx]}, "
              f"charge={CHARGE_TARGETS[charge_idx]}, "
              f"soc={info['soc']:.2f}, km={info['km']:.0f}, "
              f"reward={reward:.3f}")

    print(f"\n{'='*50}")
    summary = env.get_trip_summary()
    print(f"YOLCULUK OZETI:")
    for k, v in summary.items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"  Total reward: {total_reward:.3f}")
