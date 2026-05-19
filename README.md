# evroute: EV Uzun Yol Şarj Planlama RL Motoru

[![License: AGPL v3](https://img.shields.io/badge/license-AGPL--3.0--or--later-blue.svg)](LICENSE)
[![Commercial license](https://img.shields.io/badge/commercial-available-green.svg)](COMMERCIAL-LICENSE.md)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Version](https://img.shields.io/badge/version-0.1-orange.svg)](docs/CHANGELOG.md)
[![Code style: bilingual](https://img.shields.io/badge/docs-EN%20%2B%20TR-lightgrey.svg)](docs/architecture.md)

**Türkçe** · [English](#evroute-rl-engine-for-long-distance-ev-charging-planning)

Uzun mesafeli EV yolculuklarında **şarj durağı, şarj miktarı ve sürüş hızı**
kararlarını optimize eden, `pip` ile kurulabilen, eklenti ile genişletilebilir
bir pekiştirmeli öğrenme motoru. Üretici/belediye/araştırmacı; kendi
araç/rota/istasyonunu kaydedip motor kaynağına dokunmadan senaryo koşturur.

`AGPL-3.0-or-later` + ticari ikili lisans · paket `evroute` · v0.1 ·
TURKWAI'26 bildirisinin arkasındaki çalışma (atıf: `CITATION.cff`).

## Kurulum

```bash
python3 -m venv venv && source venv/bin/activate
pip install -e .            # çekirdek
pip install -e ".[all]"     # + veri/servis/grafik/test extra'ları
```

Çekirdek (`import evroute`) opsiyonel bağımlılıklar olmadan çalışır.
Extra'lar: `[data]` (fetch), `[serve]` (sunucu), `[viz]`, `[test]`.

## Hızlı başlangıç

```python
from evroute import make_env, train_agent, evaluate_agent
from evroute.agents import DQNAgent

env = make_env(vehicle="ioniq5", route="istanbul_ankara", driver="normal")
agent = DQNAgent(double_dqn=True)
train_agent(env, agent, num_episodes=2000)
print(evaluate_agent(env, agent, num_episodes=50))
```

```bash
evroute list routes
evroute train --agent dqn --route istanbul_ankara --episodes 2000 --out m.pt
evroute eval  --agent dqn --model m.pt --route istanbul_ankara
evroute serve --port 8000        # [serve]
```

Yerleşik: 3 rota, 2 araç, 3 sürücü, 7 hava, 2 trafik, 3 yük profili.
Gözlem 14B, eylem 25 (5 hız × 5 şarj hedefi); kararları RL ajanı seçer.

## Genişletme (çekirdeği değiştirmeden)

```python
import dataclasses
from evroute import make_env, registry

base = registry.get_vehicle("ioniq5")
registry.register_vehicle("my_ev", dataclasses.replace(
    base, name="My City EV 60 kWh", mass_kg=1650,
    battery_total_kwh=60.0, battery_usable_kwh=54.0, C_d=0.27))

make_env(vehicle="my_ev", route="istanbul_izmir", driver="eco").reset(seed=42)
```

Rota/hava/sürücü için `registry.register_*`; yeni algoritma için
`evroute.agents.base.AgentBase`. Ayrıntı: [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md).

## Veri

Depo veri/model içermez (Google Maps ToS); hepsi `.gitignore`'da ve
`evroute fetch realdata` ile kendi ücretsiz anahtarlarınızla yeniden
üretilir. Sabit `seed=42`. Dizinler `EVROUTE_DATA_DIR` /
`EVROUTE_RESULTS_DIR` ile ayarlanır.

## Belgeler

- [`docs/architecture.md`](docs/architecture.md): dört katmanlı tasarım
- [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md): geliştirme, test, eklenti
- [`docs/CHANGELOG.md`](docs/CHANGELOG.md): sürüm notları
- `research/`: deney ve rapor betikleri (motoru genel API ile tüketir)

## Lisans

İkili: açık kaynak [AGPL-3.0-or-later](LICENSE); AGPL istemeyen ticari
kullanım için [`COMMERCIAL-LICENSE.md`](COMMERCIAL-LICENSE.md)
(info@yildsamteknoloji.com). © 2026 Saadettin Yıldırım (Yıldsam Teknoloji).

---

# evroute: RL engine for long-distance EV charging planning

[Türkçe](#evroute-ev-uzun-yol-şarj-planlama-rl-motoru) · **English**

A pip-installable, extensible reinforcement-learning engine that optimises
**charging stop, charge amount, and driving speed** decisions for long-distance
electric-vehicle trips. Manufacturers, municipalities, and researchers can
register their own vehicles, routes, and stations and run scenarios without
touching the engine source.

`AGPL-3.0-or-later` + commercial dual licence · package `evroute` · v0.1 ·
the engine behind the TURKWAI'26 paper (cite via `CITATION.cff`).

## Installation

```bash
python3 -m venv venv && source venv/bin/activate
pip install -e .            # core
pip install -e ".[all]"     # + data / serve / viz / test extras
```

The core (`import evroute`) runs with no optional dependencies.
Extras: `[data]` (fetch), `[serve]` (server), `[viz]`, `[test]`.

## Quick start

```python
from evroute import make_env, train_agent, evaluate_agent
from evroute.agents import DQNAgent

env = make_env(vehicle="ioniq5", route="istanbul_ankara", driver="normal")
agent = DQNAgent(double_dqn=True)
train_agent(env, agent, num_episodes=2000)
print(evaluate_agent(env, agent, num_episodes=50))
```

```bash
evroute list routes
evroute train --agent dqn --route istanbul_ankara --episodes 2000 --out m.pt
evroute eval  --agent dqn --model m.pt --route istanbul_ankara
evroute serve --port 8000        # [serve]
```

Built in: 3 routes, 2 vehicles, 3 driver profiles, 7 weather scenarios, 2
traffic regimes, 3 load profiles. Observation 14D, action space 25 (5 speeds ×
5 charge targets); the RL agent picks the decisions.

## Extending (without touching the core)

```python
import dataclasses
from evroute import make_env, registry

base = registry.get_vehicle("ioniq5")
registry.register_vehicle("my_ev", dataclasses.replace(
    base, name="My City EV 60 kWh", mass_kg=1650,
    battery_total_kwh=60.0, battery_usable_kwh=54.0, C_d=0.27))

make_env(vehicle="my_ev", route="istanbul_izmir", driver="eco").reset(seed=42)
```

Use `registry.register_*` for routes, weather, and drivers; subclass
`evroute.agents.base.AgentBase` for new algorithms. Details:
[`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md).

## Data

The repository ships no data or models (Google Maps ToS); everything is
gitignored and reproducible via `evroute fetch realdata` with your own free
API keys. Fixed `seed=42`. Directories are configurable through
`EVROUTE_DATA_DIR` / `EVROUTE_RESULTS_DIR`.

## Documentation

- [`docs/architecture.md`](docs/architecture.md): four-layer design
- [`docs/CONTRIBUTING.md`](docs/CONTRIBUTING.md): development, testing, plugins
- [`docs/CHANGELOG.md`](docs/CHANGELOG.md): release notes
- `research/`: experiment and report scripts (consume the public API only)

## Licence

Dual: open source [AGPL-3.0-or-later](LICENSE); for commercial use that cannot
accept AGPL terms, see [`COMMERCIAL-LICENSE.md`](COMMERCIAL-LICENSE.md)
(info@yildsamteknoloji.com). © 2026 Saadettin Yıldırım (Yıldsam Teknoloji).
