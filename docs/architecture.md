# Mimari

`evroute`, bağımsız fizik ve davranış modellerinden kurulan bir Gymnasium
ortamıdır. Ajanlar ve kural tabanlı baseline'lar çekirdeğin üstüne takılır.
Çekirdek hiçbir ajanı veya opsiyonel bağımlılığı içe aktarmaz.

## Katmanlar

| Katman | Paket | Rol | Extra |
|---|---|---|---|
| L1 Çekirdek | `evroute/` (`env`, `models`, `reward`, `agents`, `registry`, `config`) | RL kütüphanesi | yok |
| L2 Veri | `evroute.data` (`loader`, `secrets`, `sources/`) | Google ve OCM çekiciler | `[data]` |
| L3 Araştırma | `research/` | deney, çapraz rota, rapor betikleri | `[viz]` |
| L4 Servis | `evroute_serve/` | FastAPI ve harita/animasyon | `[serve]` |

`import evroute` tek başına, L2 ile L4 kurulu olmadan çalışır.

## Çekirdek bileşenleri

- `env/charging_env.py` içinde `EVChargingEnv`: 14 boyutlu gözlem, 25 ayrık
  eylem (5 hız çarpı 5 şarj hedefi). `env/factory.py:make_env(...)` tek genel
  giriş noktasıdır ve veriyi `data/loader.py` ile `registry` üzerinden alır.
- `models/`: bağımsız modeller. `vehicle` (kuvvet denge tüketimi ve
  `VEHICLES`), `charging`, `elevation`, `traffic`, `weather`, `driver`.
- `reward/`: `base.RewardFunction` arayüzü ve `default_reward` (altı bileşen;
  ağırlıklar sürücü profiline göre değişir).
- `agents/`: `base.AgentBase` ile jenerik `train_agent` ve `evaluate_agent`;
  `dqn`, `ppo`, `baselines`, `runner`. Yeni algoritma `AgentBase` uygulanarak
  çekirdek değişmeden eklenir.
- `registry.py`: `register_vehicle/route/driver/weather/load`. Kayıtlar
  `make_env` üzerinden anında kullanılabilir. `config.py`: `get_data_dir()`
  ve `get_results_dir()` ortam değişkenleriyle ayarlanır.

Geçerli `make_env` anahtarları ve deney matrisi `configs/config.yaml`
dosyasındadır.

## Dizin ağacı

```
src/evroute/
├── __init__.py  version.py  config.py  registry.py  cli.py
├── env/      charging_env  factory  spaces
├── models/   vehicle charging elevation traffic weather driver
├── reward/   base  default_reward
├── agents/   base dqn ppo baselines runner loop
└── data/     loader  secrets  bundled/  sources/
evroute_serve/  server  visualize  templates/      # L4 [serve]
research/       run_all_experiments  cross_route_*  generate_report
tests/          test_smoke.py
```

## Tasarım kararı

Çekirdek girdi/çıktı yapmaz; veri enjekte edilir. Çekirdek ajan bilmez;
ajanlar üstte durur. Bu sayede yeni araç, rota veya algoritma kaynak
değiştirmeden eklenir ve motor FastAPI, requests veya matplotlib kurulu
olmadan kullanılabilir.
