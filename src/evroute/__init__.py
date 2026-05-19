# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Saadettin Yıldırım (Yıldsam Teknoloji)
"""
evroute — EV long-distance charging-planning RL engine (public package).

evroute — EV uzun yol şarj planlama RL motoru (genel paket).

Public API / Genel API::

    from evroute import make_env, train_agent, registry
    env = make_env(vehicle="ioniq5", route="istanbul_ankara", driver="normal")

Layering / Katmanlar: the core (env / models / reward) never imports an
agent; algorithms sit ON TOP via :class:`evroute.agents.base.AgentBase`
and the generic :func:`train_agent`, so a new algorithm is a drop-in.

Çekirdek (env / modeller / ödül) hiçbir ajanı içe aktarmaz; algoritmalar
:class:`evroute.agents.base.AgentBase` ve jenerik :func:`train_agent`
üzerinden ÜSTTE durur; yeni bir algoritma tak-çalıştırdır.
"""

from evroute.version import __version__

# Core (no agent import here -> engine stays agent-agnostic).
# Çekirdek (burada ajan importu yok -> motor ajandan bağımsız kalır).
from evroute import registry
from evroute.env import EVChargingEnv, make_env
from evroute.reward import RewardFunction, compute_reward

# Agent layer (sits on top of the core).
# Ajan katmanı (çekirdeğin üstünde durur).
from evroute.agents import AgentBase, train_agent, evaluate_agent

__all__ = [
    "__version__",
    "registry",
    "EVChargingEnv",
    "make_env",
    "RewardFunction",
    "compute_reward",
    "AgentBase",
    "train_agent",
    "evaluate_agent",
]
