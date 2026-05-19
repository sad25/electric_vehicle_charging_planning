# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Saadettin Yıldırım (Yıldsam Teknoloji)
"""
DQN ve Double DQN Agent
========================

PyTorch tabanli Deep Q-Network implementasyonu.
Double DQN opsiyonel olarak aktiflestirilebilir.

Referanslar:
- Mnih et al. (2015) "Human-level control through deep RL" (DQN)
- Van Hasselt et al. (2016) "Deep RL with Double Q-learning" (Double DQN)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple, deque
from typing import Optional, List, Dict
import random

from evroute.agents.base import AgentBase
# The generic, algorithm-agnostic training loop lives in
# :mod:`evroute.agents.loop` and is re-exported here.
# Jenerik, algoritmadan bağımsız eğitim döngüsü
# :mod:`evroute.agents.loop` içindedir ve burada yeniden dışa aktarılır.
from evroute.agents.loop import train_agent, evaluate_agent  # noqa: F401

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer:
    """FIFO Experience Replay Buffer."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """
    Q-deger agi.
    Input: state (14D)
    Output: Q-degerleri her aksiyon icin (25D)
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class DuelingQNetwork(nn.Module):
    """
    Dueling DQN agi: Q = V(s) + A(s,a) - mean(A)
    Referans: Wang et al. (2016) "Dueling Network Architectures"
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def forward(self, x):
        features = self.feature(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)


class DQNAgent(AgentBase):
    """
    DQN / Double DQN / Dueling DQN Agent.

    Args:
        state_dim: Gozlem boyutu (14)
        action_dim: Aksiyon sayisi (25)
        double_dqn: True ise Double DQN kullanir
        dueling: True ise Dueling Network kullanir
    """

    def __init__(self,
                 state_dim: int = 14,
                 action_dim: int = 25,
                 hidden_dim: int = 128,
                 lr: float = 5e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 batch_size: int = 64,
                 buffer_size: int = 50_000,
                 eps_start: float = 1.0,
                 eps_end: float = 0.01,
                 eps_decay: float = 0.995,
                 double_dqn: bool = False,
                 dueling: bool = False,
                 device: Optional[str] = None):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.double_dqn = double_dqn

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Networks
        NetworkClass = DuelingQNetwork if dueling else QNetwork
        self.q_net = NetworkClass(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = NetworkClass(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss
        self.buffer = ReplayBuffer(buffer_size)

        # Istatistikler
        self.training_losses = []
        self.episode_rewards = []

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """Epsilon-greedy aksiyon secimi."""
        if not eval_mode and random.random() < self.eps:
            return random.randint(0, self.action_dim - 1)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return int(q_values.argmax(dim=1).item())

    def store_transition(self, state, action, reward, next_state, done):
        """Deneyimi buffer'a kaydet."""
        self.buffer.push(state, action, reward, next_state, done)

    def update(self) -> Optional[float]:
        """Bir guncelleme adimi. Loss dondurur."""
        if len(self.buffer) < self.batch_size:
            return None

        transitions = self.buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        states = torch.FloatTensor(np.array(batch.state)).to(self.device)
        actions = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        dones = torch.FloatTensor(batch.done).unsqueeze(1).to(self.device)

        # Mevcut Q degerleri
        current_q = self.q_net(states).gather(1, actions)

        # Hedef Q degerleri
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: online ag en iyi aksiyonu secer, target ag degerlendirir
                next_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_states).gather(1, next_actions)
            else:
                # Standard DQN
                next_q = self.target_net(next_states).max(dim=1, keepdim=True)[0]

            target_q = rewards + self.gamma * next_q * (1 - dones)

        # Loss ve guncelleme
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        loss_val = loss.item()
        self.training_losses.append(loss_val)
        return loss_val

    def soft_update_target(self):
        """Target network'u soft update ile guncelle."""
        for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def decay_epsilon(self):
        """Epsilon'u azalt."""
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    def save(self, filepath: str):
        """Modeli kaydet."""
        torch.save({
            'q_net': self.q_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'eps': self.eps,
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'double_dqn': self.double_dqn,
            },
        }, filepath)

    def load(self, filepath: str):
        """Modeli yukle."""
        # weights_only=False: our own checkpoints embed a numpy-typed
        # config dict; trusted source. / kendi ürettiğimiz güvenilir dosya.
        checkpoint = torch.load(filepath, map_location=self.device,
                                weights_only=False)
        self.q_net.load_state_dict(checkpoint['q_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.eps = checkpoint['eps']


# ---------- Eğitim ----------
# Alias: the training loop is generic over any AgentBase.
# Takma ad: eğitim döngüsü herhangi bir AgentBase için jeneriktir.
train_dqn = train_agent


if __name__ == "__main__":
    from evroute import make_env

    env = make_env(seed=42)
    agent = DQNAgent(
        state_dim=14, action_dim=25,
        double_dqn=True,
        lr=5e-4, gamma=0.99,
    )

    print(f"Device: {agent.device}")
    print(f"Double DQN: {agent.double_dqn}")
    print(f"Network: {agent.q_net}")

    # Kisa test egitimi
    print("\nKisa test egitimi (50 episode)...")
    stats = train_dqn(env, agent, num_episodes=50, log_interval=25, verbose=True)
    print(f"Son 25 episode ort. reward: {np.mean(stats['rewards'][-25:]):.3f}")
