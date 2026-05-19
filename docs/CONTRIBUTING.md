# Katkı

Mimari için [`architecture.md`](architecture.md). Lisans
AGPL-3.0-or-later; ticari kullanım için
[`../COMMERCIAL-LICENSE.md`](../COMMERCIAL-LICENSE.md). Katkılar AGPL-3.0
koşullarıyla kabul edilir.

## Geliştirme kurulumu

```bash
python3 -m venv venv && source venv/bin/activate
pip install -e ".[all]"
pytest
```

## Kaynak ekleme

Yeni araç, rota, sürücü veya hava `registry.register_*` ile kaydedilir ve
`make_env` üzerinden anında kullanılabilir.

```python
import dataclasses
from evroute import make_env, registry

base = registry.get_vehicle("ioniq5")
registry.register_vehicle("my_ev", dataclasses.replace(
    base, name="My City EV 60 kWh", mass_kg=1650,
    battery_total_kwh=60.0, battery_usable_kwh=54.0, C_d=0.27))

env = make_env(vehicle="my_ev", route="istanbul_izmir", driver="eco")
env.reset(seed=42)
```

Yeni algoritma `evroute.agents.base.AgentBase` uygulanarak jenerik
`train_agent` ve `evaluate_agent` ile çalıştırılır. Çekirdek değişmez.
