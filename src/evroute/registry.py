# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Saadettin Yıldırım (Yıldsam Teknoloji)
"""
Central registry for engine resources.

Motor kaynakları için merkezi kayıt defteri.

Vehicles, driver profiles, weather and load scenarios, and route
specifications are looked up here. Third parties extend the engine by
calling the ``register_*`` functions instead of editing source files, so
a freshly registered resource becomes usable through ``make_env``
immediately and without code changes.

Araçlar, sürücü profilleri, hava ve yük senaryoları ile rota tanımları
buradan sorgulanır. Üçüncü taraflar motoru, kaynak dosyaları düzenlemek
yerine ``register_*`` fonksiyonlarını çağırarak genişletir; yeni kaydedilen
bir kaynak kod değişikliği olmadan anında ``make_env`` üzerinden kullanılır.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

# The canonical resource tables live in the model modules. They are bound
# here (same object identity) so the registry stays a single source of
# truth: registering through this module mutates the very dicts the rest
# of the engine reads.
# Asıl kaynak tabloları model modüllerinde tutulur. Buraya (aynı nesne
# kimliğiyle) bağlanır; böylece kayıt defteri tek doğruluk kaynağı kalır:
# bu modül üzerinden kayıt, motorun okuduğu sözlüklerin ta kendisini
# değiştirir.
from evroute.models.vehicle import VEHICLES, VehicleParams
from evroute.models.driver import DRIVER_PROFILES, DriverProfile, LOAD_SCENARIOS, LoadScenario
from evroute.models.weather import WEATHER_SCENARIOS, WeatherCondition


# ---------- Route specification / Rota tanımı ----------

@dataclass
class RouteSpec:
    """
    Static description of a corridor.

    Bir koridorun statik tanımı.

    ``default_stations`` is used only when no fetched station file exists
    for the route; ``tesla_superchargers`` are appended on top of the
    station list for Tesla vehicles. ``distance_km`` is the fallback trip
    length used when no real route geometry is available.

    ``default_stations`` yalnızca rota için çekilmiş istasyon dosyası
    yoksa kullanılır; ``tesla_superchargers`` Tesla araçlarda istasyon
    listesine eklenir. ``distance_km`` gerçek rota geometrisi yoksa
    kullanılan yedek yolculuk uzunluğudur.
    """
    key: str
    distance_km: float
    default_stations: List[dict] = field(default_factory=list)
    tesla_superchargers: List[dict] = field(default_factory=list)


# Default (offline) station layouts per corridor. Used as a fallback when
# no fetched OpenChargeMap file is present for the route.
# Koridor başına varsayılan (çevrimdışı) istasyon dizilimleri. Rota için
# çekilmiş OpenChargeMap dosyası yoksa yedek olarak kullanılır.
_DEFAULT_STATIONS_IST_ANK = [
    {"name": "Gebze ZES",        "road_km": 15,  "power_kw": 120, "type": "ZES_DC_120kW", "slots": 2},
    {"name": "Izmit Esarj",      "road_km": 60,  "power_kw": 90,  "type": "Esarj_DC",     "slots": 2},
    {"name": "Sakarya ZES",      "road_km": 120, "power_kw": 180, "type": "ZES_DC_120kW", "slots": 3},
    {"name": "Duzce ZES",        "road_km": 200, "power_kw": 150, "type": "ZES_DC_120kW", "slots": 2},
    {"name": "Bolu ZES",         "road_km": 270, "power_kw": 120, "type": "ZES_DC_120kW", "slots": 2},
    {"name": "Gerede Esarj",     "road_km": 310, "power_kw": 90,  "type": "Esarj_DC",     "slots": 1},
    {"name": "Ankara Giris ZES", "road_km": 420, "power_kw": 180, "type": "ZES_DC_120kW", "slots": 4},
]

_DEFAULT_STATIONS_IST_IZM = [
    {"name": "Yalova ZES",       "road_km": 80,  "power_kw": 120, "type": "ZES_DC_120kW", "slots": 2},
    {"name": "Bursa ZES",        "road_km": 155, "power_kw": 180, "type": "ZES_DC_120kW", "slots": 3},
    {"name": "Balikesir Esarj",  "road_km": 280, "power_kw": 90,  "type": "Esarj_DC",     "slots": 2},
    {"name": "Akhisar ZES",      "road_km": 360, "power_kw": 120, "type": "ZES_DC_120kW", "slots": 2},
    {"name": "Manisa ZES",       "road_km": 400, "power_kw": 150, "type": "ZES_DC_120kW", "slots": 2},
    {"name": "Izmir Giris",      "road_km": 460, "power_kw": 180, "type": "ZES_DC_120kW", "slots": 3},
]

_DEFAULT_STATIONS_ANK_ANT = [
    {"name": "Polatli ZES",      "road_km": 80,  "power_kw": 120, "type": "ZES_DC_120kW", "slots": 2},
    {"name": "Aksaray Esarj",    "road_km": 180, "power_kw": 90,  "type": "Esarj_DC",     "slots": 2},
    {"name": "Konya ZES",        "road_km": 260, "power_kw": 180, "type": "ZES_DC_120kW", "slots": 3},
    {"name": "Isparta ZES",      "road_km": 370, "power_kw": 120, "type": "ZES_DC_120kW", "slots": 2},
    {"name": "Burdur Esarj",     "road_km": 400, "power_kw": 90,  "type": "Esarj_DC",     "slots": 1},
    {"name": "Antalya Giris",    "road_km": 460, "power_kw": 180, "type": "ZES_DC_120kW", "slots": 3},
]

# Tesla Superchargers, applied only to Tesla vehicles.
# Tesla Supercharger istasyonları, yalnızca Tesla araçlara uygulanır.
_TESLA_SUPERCHARGERS = {
    "istanbul_ankara": [
        {"name": "Bolu Tesla SC",  "road_km": 265, "power_kw": 250, "type": "Tesla_SC", "slots": 8},
    ],
    "istanbul_izmir": [
        {"name": "Bursa Tesla SC", "road_km": 150, "power_kw": 250, "type": "Tesla_SC", "slots": 6},
    ],
    "ankara_antalya": [],
}

ROUTES: Dict[str, RouteSpec] = {
    "istanbul_ankara": RouteSpec(
        "istanbul_ankara", 450, _DEFAULT_STATIONS_IST_ANK,
        _TESLA_SUPERCHARGERS["istanbul_ankara"]),
    "istanbul_izmir": RouteSpec(
        "istanbul_izmir", 480, _DEFAULT_STATIONS_IST_IZM,
        _TESLA_SUPERCHARGERS["istanbul_izmir"]),
    "ankara_antalya": RouteSpec(
        "ankara_antalya", 480, _DEFAULT_STATIONS_ANK_ANT,
        _TESLA_SUPERCHARGERS["ankara_antalya"]),
}

# Flat views of the route tables, keyed by route.
# Rota tablolarının rota anahtarına göre düz görünümleri.
ROUTE_STATIONS: Dict[str, List[dict]] = {k: v.default_stations for k, v in ROUTES.items()}
ROUTE_DISTANCES: Dict[str, float] = {k: v.distance_km for k, v in ROUTES.items()}
TESLA_SUPERCHARGERS: Dict[str, List[dict]] = {k: v.tesla_superchargers for k, v in ROUTES.items()}


# ---------- Lookups / Sorgular ----------

def _get(table: dict, key: str, kind: str):
    try:
        return table[key]
    except KeyError:
        raise KeyError(
            f"Unknown {kind} '{key}'. Known: {sorted(table)}"
        ) from None


def get_vehicle(key: str) -> VehicleParams:
    return _get(VEHICLES, key, "vehicle")


def get_driver(key: str) -> DriverProfile:
    return _get(DRIVER_PROFILES, key, "driver")


def get_load(key: str) -> LoadScenario:
    return _get(LOAD_SCENARIOS, key, "load")


def get_weather(key: str) -> WeatherCondition:
    return _get(WEATHER_SCENARIOS, key, "weather")


def get_route(key: str) -> RouteSpec:
    return _get(ROUTES, key, "route")


def list_vehicles() -> List[str]:
    return sorted(VEHICLES)


def list_drivers() -> List[str]:
    return sorted(DRIVER_PROFILES)


def list_loads() -> List[str]:
    return sorted(LOAD_SCENARIOS)


def list_weather() -> List[str]:
    return sorted(WEATHER_SCENARIOS)


def list_routes() -> List[str]:
    return sorted(ROUTES)


# ---------- Registration / Kayıt ----------

def register_vehicle(key: str, vehicle: VehicleParams, *, overwrite: bool = False) -> None:
    """Register a new vehicle. / Yeni bir araç kaydeder."""
    if key in VEHICLES and not overwrite:
        raise ValueError(f"Vehicle '{key}' already registered.")
    VEHICLES[key] = vehicle


def register_driver(key: str, driver: DriverProfile, *, overwrite: bool = False) -> None:
    """Register a new driver profile. / Yeni bir sürücü profili kaydeder."""
    if key in DRIVER_PROFILES and not overwrite:
        raise ValueError(f"Driver '{key}' already registered.")
    DRIVER_PROFILES[key] = driver


def register_load(key: str, load: LoadScenario, *, overwrite: bool = False) -> None:
    """Register a new load scenario. / Yeni bir yük senaryosu kaydeder."""
    if key in LOAD_SCENARIOS and not overwrite:
        raise ValueError(f"Load '{key}' already registered.")
    LOAD_SCENARIOS[key] = load


def register_weather(key: str, weather: WeatherCondition, *, overwrite: bool = False) -> None:
    """Register a new weather scenario. / Yeni bir hava senaryosu kaydeder."""
    if key in WEATHER_SCENARIOS and not overwrite:
        raise ValueError(f"Weather '{key}' already registered.")
    WEATHER_SCENARIOS[key] = weather


def register_route(spec: RouteSpec, *, overwrite: bool = False) -> None:
    """Register a new route specification. / Yeni bir rota tanımı kaydeder."""
    if spec.key in ROUTES and not overwrite:
        raise ValueError(f"Route '{spec.key}' already registered.")
    ROUTES[spec.key] = spec
    ROUTE_STATIONS[spec.key] = spec.default_stations
    ROUTE_DISTANCES[spec.key] = spec.distance_km
    TESLA_SUPERCHARGERS[spec.key] = spec.tesla_superchargers
