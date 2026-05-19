# Değişiklik Günlüğü

Biçim [Keep a Changelog](https://keepachangelog.com), sürümleme
[SemVer](https://semver.org).

## [0.1.0] (2026-05-19)

İlk sürüm.

- `evroute` paketi ve genel API: `make_env`, `train_agent`,
  `evaluate_agent`, `registry`, `AgentBase`.
- Dört katmanlı mimari (çekirdek, `[data]`, `research/`, `[serve]`).
  Çekirdek opsiyonel bağımlılıklar olmadan çalışır.
- `evroute` CLI: `list`, `train`, `eval`, `fetch`, `serve`.
- `registry.register_*` ile çekirdeği değiştirmeden kaynak ekleme.
- `evroute.config`: `EVROUTE_DATA_DIR` ve `EVROUTE_RESULTS_DIR`.
- `tests/` çevrimdışı duman testleri.
- AGPL-3.0 ve ticari ikili lisans.
