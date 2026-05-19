# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Saadettin Yıldırım (Yıldsam Teknoloji)
"""
PPO Agent (Stable-Baselines3 Wrapper)
=======================================

Stable-Baselines3 kutuphanesini kullanarak PPO egitimi.
Gymnasium uyumlu ortamimizla dogrudan calisir.

Referans: Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
"""

import numpy as np
from typing import Dict, Optional
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class TrainingLogger(BaseCallback):
    """Egitim sirasinda istatistik toplayan callback."""

    def __init__(self, log_interval: int = 100, verbose: int = 1):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.episode_rewards = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        # Episode bitti mi?
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_count += 1

                if self.verbose and self.episode_count % self.log_interval == 0:
                    avg_r = np.mean(self.episode_rewards[-self.log_interval:])
                    print(f"  PPO Episode {self.episode_count:4d} | "
                          f"Avg Reward: {avg_r:.3f}")
        return True


def train_ppo(env,
              total_timesteps: int = 50_000,
              lr: float = 3e-4,
              n_steps: int = 256,
              batch_size: int = 64,
              n_epochs: int = 10,
              gamma: float = 0.99,
              seed: int = 42,
              verbose: bool = True,
              save_path: Optional[str] = None,
              device: str = "auto") -> tuple:
    """
    PPO ajanini egitir.

    Args:
        env: Gymnasium ortami
        total_timesteps: Toplam egitim adimi
        save_path: Model kaydetme yolu (opsiyonel)

    Returns:
        (model, logger): Egitilmis model ve istatistikler
    """
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=lr,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        seed=seed,
        verbose=0,
        device=device,
        policy_kwargs={"net_arch": [128, 128]},
    )

    logger = TrainingLogger(log_interval=100, verbose=verbose)

    if verbose:
        print(f"PPO egitimi basliyor ({total_timesteps} timestep)...")

    model.learn(total_timesteps=total_timesteps, callback=logger)

    if save_path:
        model.save(save_path)
        if verbose:
            print(f"Model kaydedildi: {save_path}")

    return model, logger


def evaluate_ppo(model, env, num_episodes: int = 50) -> Dict:
    """PPO modelini degerlendirir."""
    results = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
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


if __name__ == "__main__":
    from evroute import make_env

    env = make_env(seed=42)
    model, logger = train_ppo(env, total_timesteps=5000, verbose=True)

    print("\nDegerlendirme...")
    results = evaluate_ppo(model, env, num_episodes=10)
    print(f"  Ort. sure: {results['avg_total_time_h']:.1f}h")
    print(f"  Ort. maliyet: {results['avg_total_cost_tl']:.0f} TL")
    print(f"  Ort. reward: {results['avg_total_reward']:.3f}")
