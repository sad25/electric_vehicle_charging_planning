# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Saadettin Yıldırım (Yıldsam Teknoloji)
"""
Generic, algorithm-agnostic training/evaluation loop.

Jenerik, algoritmadan bağımsız eğitim/değerlendirme döngüsü.

Works with any :class:`evroute.agents.base.AgentBase` (DQN today, a
custom algorithm tomorrow) and never imports a specific algorithm — that
keeps "algorithms" a swappable layer above the engine.

Herhangi bir :class:`evroute.agents.base.AgentBase` ile çalışır (bugün
DQN, yarın özel bir algoritma) ve belirli bir algoritmayı içe aktarmaz —
böylece "algoritmalar" motorun üstünde takılabilir bir katman kalır.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from evroute.agents.base import AgentBase


def train_agent(env, agent: AgentBase,
                num_episodes: int = 2000,
                target_update_freq: int = 10,
                log_interval: int = 100,
                verbose: bool = True) -> Dict:
    """
    Ajani egitir (algoritmadan bagimsiz). / Train any AgentBase.

    Returns:
        dict: Egitim istatistikleri
    """
    episode_rewards = []
    episode_times = []
    episode_arrivals = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store_transition(obs, action, reward, next_obs, float(done))
            agent.update()

            obs = next_obs
            total_reward += reward

        # Episode sonu
        agent.decay_epsilon()
        if episode % target_update_freq == 0:
            agent.soft_update_target()

        summary = env.get_trip_summary()
        episode_rewards.append(total_reward)
        episode_times.append(summary["total_time_h"])
        episode_arrivals.append(1.0 if summary["arrival_soc"] > 0 else 0.0)

        if verbose and (episode + 1) % log_interval == 0:
            avg_r = np.mean(episode_rewards[-log_interval:])
            avg_t = np.mean(episode_times[-log_interval:])
            avg_arr = np.mean(episode_arrivals[-log_interval:])
            print(f"  Episode {episode+1:4d} | "
                  f"Avg Reward: {avg_r:.3f} | "
                  f"Avg Time: {avg_t:.1f}h | "
                  f"Arrival Rate: {avg_arr:.0%} | "
                  f"Eps: {getattr(agent, 'eps', float('nan')):.3f}")

    agent.episode_rewards = episode_rewards

    return {
        "rewards": episode_rewards,
        "times": episode_times,
        "arrivals": episode_arrivals,
        "losses": agent.training_losses,
    }


def evaluate_agent(env, agent: AgentBase,
                   num_episodes: int = 50) -> Dict:
    """Egitilmis ajani degerlendir (greedy). / Evaluate any AgentBase."""
    results = []

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = agent.select_action(obs, eval_mode=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        summary = env.get_trip_summary()
        summary["total_reward"] = total_reward
        results.append(summary)

    # Ortalama sonuclar
    avg = {}
    for key in results[0]:
        vals = [r[key] for r in results]
        avg[f"avg_{key}"] = np.mean(vals)
        avg[f"std_{key}"] = np.std(vals)

    return avg
