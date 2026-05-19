# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Saadettin Yıldırım (Yıldsam Teknoloji)
"""
Agent contract.

Ajan sözleşmesi.

Algorithms are a layer ON TOP of the engine, not part of it: the core
(env / models / reward) never imports an agent, and an agent only ever
talks to the env through Gymnasium's ``reset``/``step``. To try a new
algorithm (something other than DQN/PPO) you subclass :class:`AgentBase`
and pass the instance to :func:`evroute.train_agent` — no core change.

Algoritmalar motorun ÜSTÜNDE bir katmandır, parçası değil: çekirdek
(env / modeller / ödül) hiçbir ajanı içe aktarmaz; ajan ortamla yalnız
Gymnasium ``reset``/``step`` üzerinden konuşur. Yeni bir algoritma
(DQN/PPO dışında) denemek için :class:`AgentBase`'ten türetip örneği
:func:`evroute.train_agent`'a verirsiniz — çekirdek değişmez.
"""
from __future__ import annotations

import abc
from typing import List, Optional

import numpy as np


class AgentBase(abc.ABC):
    """
    Minimal interface every trainable agent must satisfy.

    Eğitilebilir her ajanın sağlaması gereken asgari arayüz.

    ``decay_epsilon`` and ``soft_update_target`` default to no-ops so
    on/off-policy and exploration-free algorithms (e.g. PPO, custom ones)
    don't have to implement DQN-specific hooks; the generic training loop
    calls them unconditionally.

    ``decay_epsilon`` ve ``soft_update_target`` varsayılan olarak boş
    işlemdir; böylece on/off-policy ve keşifsiz algoritmaların (örn. PPO,
    özel olanlar) DQN'e özgü kancaları uygulaması gerekmez; jenerik
    eğitim döngüsü bunları koşulsuz çağırır.
    """

    #: Diagnostics filled by ``update``/the training loop.
    #: ``update``/eğitim döngüsünün doldurduğu teşhis verileri.
    training_losses: List[float]
    episode_rewards: List[float]

    @abc.abstractmethod
    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """Pick an action for ``state``. / ``state`` için aksiyon seç."""

    @abc.abstractmethod
    def store_transition(self, state, action, reward, next_state, done) -> None:
        """Record one transition. / Bir geçişi kaydet."""

    @abc.abstractmethod
    def update(self) -> Optional[float]:
        """One learning step; returns loss or None. / Bir öğrenme adımı."""

    @abc.abstractmethod
    def save(self, filepath: str) -> None:
        """Persist weights. / Ağırlıkları kaydet."""

    @abc.abstractmethod
    def load(self, filepath: str) -> None:
        """Restore weights. / Ağırlıkları geri yükle."""

    def decay_epsilon(self) -> None:
        """Exploration decay hook (no-op by default). / Keşif azaltma kancası."""

    def soft_update_target(self) -> None:
        """Target-network sync hook (no-op by default). / Hedef ağ senkron kancası."""
