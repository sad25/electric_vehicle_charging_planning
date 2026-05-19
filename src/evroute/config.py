# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Saadettin Yıldırım (Yıldsam Teknoloji)
"""
Data-directory resolution for the engine.

Motor için veri dizini çözümleme.

Resolution order / Çözüm sırası:
  1. EVROUTE_DATA_DIR environment variable, if set and existing.
     EVROUTE_DATA_DIR ortam değişkeni (tanımlı ve mevcutsa).
  2. A `data/` directory under the current working directory.
     Çalışılan dizindeki `data/` klasörü.
  3. The `bundled/` data shipped inside the installed package.
     Kurulu paketin içindeki `bundled/` verisi (son çare).

Used by :mod:`evroute.data.loader` to locate route data files.
:mod:`evroute.data.loader` rota veri dosyalarını bulmak için kullanır.
"""
from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "EVROUTE_DATA_DIR"
RESULTS_ENV_VAR = "EVROUTE_RESULTS_DIR"


def get_data_dir() -> Path:
    """Return the resolved data directory as a Path.

    Çözümlenmiş veri dizinini Path olarak döndürür.

    The bundled fallback is returned even if it does not exist yet, so
    callers always get a usable Path; existence is the caller's concern.
    Paketli son çare henüz yoksa bile döndürülür; varlık denetimi
    çağırana aittir.
    """
    # 1. Explicit override via environment variable.
    #    Ortam değişkeni ile açık geçersiz kılma.
    env_val = os.environ.get(ENV_VAR)
    if env_val:
        p = Path(env_val).expanduser()
        if p.is_dir():
            return p

    # 2. Project-local data/ next to the current working directory.
    #    Çalışılan dizinin yanındaki proje-yerel data/ klasörü.
    cwd_data = Path.cwd() / "data"
    if cwd_data.is_dir():
        return cwd_data

    # 3. Fallback: data bundled inside the installed package.
    #    Son çare: kurulu paketin içine gömülü veri.
    return Path(__file__).resolve().parent / "data" / "bundled"


def get_results_dir() -> Path:
    """Return the directory where research scripts write results/models.

    Araştırma betiklerinin sonuç/model yazdığı dizini döndürür.

    Resolution order / Çözüm sırası:
      1. EVROUTE_RESULTS_DIR environment variable, if set.
         EVROUTE_RESULTS_DIR ortam değişkeni (tanımlıysa).
      2. ``<cwd>/data/processed/results`` (project default).
         ``<cwd>/data/processed/results`` (proje varsayılanı).

    The directory is created if it does not exist, so callers can write
    immediately. Unlike :func:`get_data_dir`, results are an *output*
    location and are always rooted at the working directory, never inside
    the installed package.

    Dizin yoksa oluşturulur; çağıran hemen yazabilir. :func:`get_data_dir`
    aksine, sonuçlar bir *çıktı* konumudur ve daima çalışma dizinine
    köklenir, asla kurulu paketin içine değil.
    """
    env_val = os.environ.get(RESULTS_ENV_VAR)
    if env_val:
        p = Path(env_val).expanduser()
    else:
        p = Path.cwd() / "data" / "processed" / "results"
    p.mkdir(parents=True, exist_ok=True)
    return p
