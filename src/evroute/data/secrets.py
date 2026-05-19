# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Saadettin Yıldırım (Yıldsam Teknoloji)
"""
EN: Secret loader. Resolves API keys from environment variables first,
    then from the project-root .env file. API keys are never hard-coded
    in source; .env is excluded from version control.
TR: Gizli anahtar okuyucu. API anahtarlarını önce ortam değişkenlerinden,
    yoksa proje kökündeki .env dosyasından çözer. Anahtarlar koda gömülmez;
    .env sürüm kontrolüne dahil edilmez.
"""

import os
from pathlib import Path

from evroute.config import get_data_dir

_ENV_CACHE = None


def _env_path() -> Path:
    """
    EN: Locates the project-root .env. Tries the current working directory
        first, then the parent of the resolved data/ directory.
    TR: Proje kökündeki .env'i bulur. Önce çalışma dizini, sonra çözülen
        data/ dizininin üst klasörü denenir.
    """
    cwd_env = Path.cwd() / ".env"
    if cwd_env.exists():
        return cwd_env
    return get_data_dir().parent / ".env"


def _load_env_file() -> dict:
    """
    EN: Parses the project-root .env file into a dict (cached). Lines that
        are blank, commented, or lack '=' are skipped.
    TR: Proje kökündeki .env dosyasını sözlüğe ayrıştırır (önbellekli).
        Boş, yorum veya '=' içermeyen satırlar atlanır.
    """
    global _ENV_CACHE
    if _ENV_CACHE is not None:
        return _ENV_CACHE
    data = {}
    env_path = _env_path()
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            data[k.strip()] = v.strip().strip('"').strip("'")
    _ENV_CACHE = data
    return data


def get_secret(name: str, required: bool = True) -> str:
    """
    EN: Returns the requested secret. Raises a descriptive error when
        required and missing.
    TR: İstenen anahtarı döndürür. required=True iken bulunamazsa
        açıklayıcı bir hata yükseltir.
    """
    val = os.environ.get(name) or _load_env_file().get(name, "")
    if required and not val:
        raise RuntimeError(
            f"'{name}' not found. Create a .env file at the project root "
            f"(copy .env.example) and add the line {name}=... / "
            f"'{name}' bulunamadı. Proje kökünde .env oluşturun "
            f"(.env.example'ı kopyalayın) ve {name}=... satırını ekleyin."
        )
    return val
