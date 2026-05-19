# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Saadettin Yıldırım (Yıldsam Teknoloji)
"""
Reward function contract.

Ödül fonksiyonu sözleşmesi.

The environment scores every step through an injected callable that
satisfies :class:`RewardFunction`. The engine ships
:func:`evroute.reward.default_reward.compute_reward` as the default
implementation; a host application can pass any callable with the same
signature to ``make_env(..., reward_fn=...)`` without touching the
environment.

Ortam her adımı, :class:`RewardFunction`'ı sağlayan enjekte edilmiş bir
çağrılabilir üzerinden puanlar. Motor varsayılan olarak
:func:`evroute.reward.default_reward.compute_reward` ile gelir; bir
uygulama aynı imzaya sahip herhangi bir çağrılabiliri ortama dokunmadan
``make_env(..., reward_fn=...)`` ile geçebilir.
"""
from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable


@runtime_checkable
class RewardFunction(Protocol):
    """
    Callable scoring one environment transition.

    Bir ortam geçişini puanlayan çağrılabilir.

    Must return a ``dict`` whose ``"total"`` key is the scalar reward fed
    back to the agent; component keys (``r_time``, ``r_cost``, ...) are
    used for diagnostics/plots and may be present.

    ``"total"`` anahtarı ajana verilen skaler ödül olan bir ``dict``
    döndürmelidir; bileşen anahtarları (``r_time``, ``r_cost``, ...)
    teşhis/grafik içindir ve bulunabilir.
    """

    def __call__(
        self,
        driver: Any,
        drive_time_h: float,
        charge_time_min: float,
        wait_time_min: float,
        energy_kwh: float,
        station_type: str,
        hour: float,
        current_soc: float,
        target_soc_pct: float,
        station_power_kw: float,
        continuous_drive_min: float,
        is_dead: bool = False,
    ) -> Dict[str, float]:
        ...
