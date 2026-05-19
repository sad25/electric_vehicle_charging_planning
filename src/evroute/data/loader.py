# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Saadettin Yıldırım (Yıldsam Teknoloji)
"""
Filesystem data loaders for routes.

Rotalar için dosya sistemi veri yükleyicileri.

These functions are the only place that reads route data files from disk.
The simulation environment receives the loaded data as plain Python
objects and never touches the filesystem itself, which keeps the engine
testable and lets a host application supply its own data source.

Bu fonksiyonlar rota veri dosyalarını diskten okuyan tek yerdir.
Simülasyon ortamı yüklenen veriyi düz Python nesnesi olarak alır ve
dosya sistemine hiç dokunmaz; bu sayede motor test edilebilir kalır ve
bir uygulama kendi veri kaynağını sağlayabilir.

Data directory resolution is delegated to :func:`evroute.config.get_data_dir`.

Veri dizini çözümlemesi :func:`evroute.config.get_data_dir`'e bırakılır.
"""
from __future__ import annotations

import json
from typing import List, Optional

import numpy as np

from evroute.config import get_data_dir
from evroute.models.elevation import ElevationProfile, get_elevation_profile


def load_google_route(route: str) -> Optional[dict]:
    """
    Return the single-source Google route document for ``route``, or None.

    ``route`` için tek-kaynak Google rota belgesini döndürür, yoksa None.

    The document carries A->C->B detour costs (``extra_km`` / ``extra_min``)
    and the real elevation-tagged detour path per station.

    Belge, istasyon başına A->C->B sapma maliyetlerini (``extra_km`` /
    ``extra_min``) ve gerçek yükseklikli sapma yolunu taşır.
    """
    fp = get_data_dir() / "processed" / f"{route}_route_google.json"
    if not fp.exists():
        return None
    with open(fp, encoding="utf-8") as f:
        return json.load(f)


def load_stations(route: str) -> Optional[List[dict]]:
    """
    Load OpenChargeMap stations for ``route``, or None if absent.

    ``route`` için OpenChargeMap istasyonlarını yükler; yoksa None.

    When the single-source Google route document exists, the REAL detour
    cost (``extra_km`` / ``extra_min`` from A->C->B routing) and the
    elevation-tagged detour profile are merged onto each station so the
    physics uses the true detour rather than an anchor-based estimate.

    Tek-kaynak Google rota belgesi varsa, GERÇEK sapma maliyeti
    (A->C->B'den ``extra_km`` / ``extra_min``) ve yükseklikli sapma
    profili her istasyona eklenir; böylece fizik, çapa tabanlı tahmin
    yerine gerçek sapmayı kullanır.
    """
    fp = get_data_dir() / "processed" / f"{route}_stations_ocm.json"
    if not fp.exists():
        return None
    with open(fp, encoding="utf-8") as f:
        stations = json.load(f)

    g = load_google_route(route)
    if g is not None:
        gx = {s["name"]: s for s in g.get("stations", [])}
        for st in stations:
            gs = gx.get(st.get("name"))
            if gs is not None:
                # A detour cannot shorten the trip; clamp at zero.
                # Sapma yolculuğu kısaltamaz; sıfırda kırpılır.
                st["extra_km"] = max(0.0, float(gs.get("extra_km", 0.0)))
                st["extra_min"] = max(0.0, float(gs.get("extra_min", 0.0)))
                # Real elevation-tagged detour profile; energy is
                # integrated over it like the main route.
                # Gerçek yükseklikli sapma profili; enerji bunun
                # üzerinden ana rota gibi integre edilir.
                dev = gs.get("dev") or []
                if dev and len(dev[0]) >= 3:
                    st["g_dev"] = dev
    return stations


def detect_route_distance(route: str) -> Optional[float]:
    """
    Trip length from the real route geometry, or None if unavailable.

    Gerçek rota geometrisinden yolculuk uzunluğu; yoksa None.

    Returning None lets the caller fall back to the registry's declared
    distance for the route.

    None döndürmek, çağıranın rotanın kayıt defterindeki bildirilmiş
    mesafesine geri dönmesini sağlar.
    """
    fp = get_data_dir() / "raw" / f"{route}_route_geometry.json"
    if not fp.exists():
        return None
    with open(fp) as f:
        coords = json.load(f)["coordinates"]
    total = 0.0
    for i in range(1, len(coords)):
        dlat = np.radians(coords[i][0] - coords[i - 1][0])
        dlng = np.radians(coords[i][1] - coords[i - 1][1])
        a = (np.sin(dlat / 2) ** 2
             + np.cos(np.radians(coords[i - 1][0]))
             * np.cos(np.radians(coords[i][0]))
             * np.sin(dlng / 2) ** 2)
        total += 6371 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return total


def load_elevation(route: str) -> ElevationProfile:
    """
    Elevation profile for ``route`` (fetched data if present, else synthetic).

    ``route`` için yükseklik profili (varsa çekilmiş veri, yoksa sentetik).
    """
    return get_elevation_profile(route, use_api=False)
