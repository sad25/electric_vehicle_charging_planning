# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Saadettin Yıldırım (Yıldsam Teknoloji)
"""
Action grid for the charging environment.

Şarj ortamı için aksiyon ızgarası.

The discrete action is ``speed_idx * len(CHARGE_TARGETS) + charge_idx``.
Kept in its own module so agents/baselines can reference the grid
without importing the environment (no layer leak).

Ayrık aksiyon ``speed_idx * len(CHARGE_TARGETS) + charge_idx`` şeklindedir.
Ajan/baseline'ların ortamı içe aktarmadan ızgaraya erişebilmesi için
ayrı modülde tutulur (katman sızıntısı yok).
"""
from __future__ import annotations

from typing import Optional, Tuple

# Speeds in km/h; charge targets as SoC fractions (None = skip charging).
# Hızlar km/sa; şarj hedefleri SoC oranı (None = şarjı atla).
SPEEDS = [80, 100, 120, 140, 160]
CHARGE_TARGETS = [None, 0.30, 0.50, 0.70, 0.90]

NUM_ACTIONS = len(SPEEDS) * len(CHARGE_TARGETS)  # 25


def decode_action(action: int) -> Tuple[float, Optional[float]]:
    """``action`` -> (desired_speed_kmh, charge_target). / Aksiyonu çöz."""
    speed_idx = action // len(CHARGE_TARGETS)
    charge_idx = action % len(CHARGE_TARGETS)
    return SPEEDS[speed_idx], CHARGE_TARGETS[charge_idx]


def encode_action(speed_idx: int, charge_idx: int) -> int:
    """(speed_idx, charge_idx) -> action. / Aksiyona kodla."""
    return speed_idx * len(CHARGE_TARGETS) + charge_idx
