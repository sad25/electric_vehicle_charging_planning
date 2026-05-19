# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Saadettin Yıldırım (Yıldsam Teknoloji)
"""
Genisletilmis EV Sarj Optimizasyonu Gymnasium Ortami
======================================================

14 boyutlu observation space, 25 aksiyon (5 hiz x 5 sarj hedefi).
Fizik tabanli tuketim, yukseklik, trafik, hava durumu, surucu profili,
mola-sarj senkronizasyonu, istasyon kuyrugu ve maliyet entegrasyonu.

Observation Space (14D, Box [0,1]):
  0: current_soc               - Mevcut batarya durumu
  1: position_normalized        - Rota uzerindeki pozisyon
  2: dist_to_next_station       - Sonraki istasyona mesafe (norm.)
  3: next_station_power         - Sonraki istasyonun gucu (norm.)
  4: remaining_distance         - Kalan toplam mesafe (norm.)
  5: current_grade              - Mevcut egim (norm. [-1,1] -> [0,1])
  6: hour_sin                   - Gunun saati (sin kodlama)
  7: hour_cos                   - Gunun saati (cos kodlama)
  8: temperature_normalized     - Hava sicakligi (norm.)
  9: headwind_normalized        - Karsidan ruzgar (norm.)
  10: continuous_drive_norm     - Kesintisiz surus suresi (norm.)
  11: traffic_speed_ratio       - Trafik hiz faktoru
  12: station_occupancy         - Istasyon yogunlugu (0-1)
  13: charge_price_normalized   - Sarj fiyati (norm.)

Action Space (25 Discrete):
  action = speed_idx * 5 + charge_idx
  Hizlar:        [80, 100, 120, 140, 160] km/h
  Sarj hedefleri: [None, 30%, 50%, 70%, 90%]
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, List, Dict, Tuple, Any

from evroute.models.vehicle import (
    VehicleParams,
    energy_consumption_kwh_per_km, soc_change, temperature_battery_factor
)
from evroute.models.charging import (
    ChargingCurve, load_ioniq5_curve, load_tesla3_curve,
    calculate_charge_time, energy_charged_kwh
)
from evroute.models.elevation import ElevationProfile
from evroute.models.traffic import RouteTrafficModel, create_traffic_model
from evroute.models.weather import (
    WeatherCondition,
    combined_weather_factor, charging_speed_factor, rain_rolling_resistance_factor
)
from evroute.models.driver import (
    DriverProfile, LoadScenario,
    break_charge_overlap
)
from evroute.reward.default_reward import compute_reward, queue_wait_time, get_electricity_price
from evroute.env.spaces import SPEEDS, CHARGE_TARGETS

from evroute import registry
from evroute.data import loader as _default_loader

# Re-exports of the route tables, which are defined in
# :mod:`evroute.registry`.
# Rota tablolarının yeniden dışa aktarımı; tablolar
# :mod:`evroute.registry` içinde tanımlıdır.
from evroute.registry import (  # noqa: F401
    ROUTE_STATIONS, ROUTE_DISTANCES, TESLA_SUPERCHARGERS,
)


class EVChargingEnv(gym.Env):
    """
    Genisletilmis EV Sarj Optimizasyonu Ortami.

    Her adimda agent bir istasyona gider ve:
    1. Hiz secer (trafik ve fizikle sinirli)
    2. Sarj hedefi secer (veya atla)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self,
                 vehicle_key: str = "ioniq5",
                 route_key: str = "istanbul_ankara",
                 driver_key: str = "normal",
                 load_key: str = "normal",
                 weather_key: str = "optimal",
                 departure_hour: float = 8.0,
                 is_weekend: bool = False,
                 initial_soc: float = 0.80,
                 seed: Optional[int] = None,
                 data_loader=None,
                 route_registry=None,
                 reward_fn=None):
        super().__init__()

        # Injected reward function (RewardFunction protocol). Defaults to
        # the engine's 6-component weighted reward; a host application may
        # supply its own scoring without touching the environment.
        # Enjekte edilen ödül fonksiyonu (RewardFunction protokolü).
        # Varsayılan motorun 6 bileşenli ağırlıklı ödülüdür; bir uygulama
        # ortamı değiştirmeden kendi puanlamasını verebilir.
        self._reward_fn = reward_fn if reward_fn is not None else compute_reward

        # Injected collaborators. Resources are resolved through the
        # registry and route data is read through the data loader; both
        # default to the engine's own implementations but a host
        # application may substitute its own.
        # Enjekte edilen iş ortakları. Kaynaklar kayıt defteri üzerinden,
        # rota verisi veri yükleyici üzerinden çözülür; ikisi de varsayılan
        # olarak motorun kendi uygulamasıdır ancak bir uygulama kendi
        # uygulamasını koyabilir.
        self._reg = route_registry if route_registry is not None else registry
        self._loader = data_loader if data_loader is not None else _default_loader

        # --- Konfigurasyon ---
        self.vehicle: VehicleParams = self._reg.get_vehicle(vehicle_key)
        self.vehicle_key = vehicle_key
        self.route_key = route_key
        self.driver: DriverProfile = self._reg.get_driver(driver_key)
        self.load: LoadScenario = self._reg.get_load(load_key)
        self.weather: WeatherCondition = self._reg.get_weather(weather_key)
        self.departure_hour = departure_hour
        self.initial_soc = initial_soc

        # HVAC mode: load override veya driver default
        self.hvac_mode = self.load.hvac_override or self.driver.hvac_mode

        # --- Rota verileri ---
        route_spec = self._reg.get_route(route_key)

        # Gercek istasyonlari yukle (OCM verisi varsa), yoksa default kullan
        ocm_stations = self._loader.load_stations(route_key)
        self.stations = list(ocm_stations if ocm_stations
                             else route_spec.default_stations)

        # Rota mesafesini gercek veriye gore ayarla
        # Gercek rota geometrisi varsa ondan, yoksa kayit defteri varsayilani
        detected = self._loader.detect_route_distance(route_key)
        self.total_distance = detected if detected is not None else route_spec.distance_km

        # Tesla icin Supercharger ekle
        if vehicle_key == "tesla3" and route_spec.tesla_superchargers:
            self.stations.extend(route_spec.tesla_superchargers)
            self.stations.sort(key=lambda s: s["road_km"])

        # --- Model yukle ---
        if vehicle_key == "ioniq5":
            self.charge_curve = load_ioniq5_curve()
        elif vehicle_key == "tesla3":
            self.charge_curve = load_tesla3_curve()
        else:
            self.charge_curve = load_ioniq5_curve()

        self.elevation = self._loader.load_elevation(route_key)
        self.traffic = create_traffic_model(route_key, is_weekend, prefer_google=True)

        # --- Hava durumu etkileri ---
        self.weather_consumption_factor = combined_weather_factor(
            self.weather, 120  # Referans hiz
        )
        self.charging_temp_factor = 1.0 / charging_speed_factor(self.weather.temperature_c)
        self.battery_temp_factor = temperature_battery_factor(self.weather.temperature_c)

        # --- Gymnasium spaces ---
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(14,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(len(SPEEDS) * len(CHARGE_TARGETS))  # 25

        # --- State ---
        self._reset_state()

        if seed is not None:
            self.np_random = np.random.default_rng(seed)

    def _reset_state(self):
        self.soc = self.initial_soc
        self.current_km = 0.0
        self.current_hour = self.departure_hour
        self.station_idx = 0
        self.continuous_drive_min = 0.0
        self.total_drive_time_h = 0.0
        self.total_charge_time_min = 0.0
        self.total_wait_time_min = 0.0
        self.total_cost_tl = 0.0
        self.total_energy_kwh = 0.0
        self.total_detour_km = 0.0
        self.total_detour_time_h = 0.0
        self.history = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), self._get_info()

    def _get_obs(self) -> np.ndarray:
        """14 boyutlu normalize edilmis gozlem vektoru."""
        # Sonraki istasyon bilgileri
        if self.station_idx < len(self.stations):
            next_st = self.stations[self.station_idx]
            dist_to_next = max(0, next_st["road_km"] - self.current_km)
            next_power = next_st["power_kw"]
            station_occ = queue_wait_time(self.current_hour, next_st.get("slots", 2), 0.5) / 30.0
            price = get_electricity_price(next_st.get("type", "Generic_DC"), self.current_hour)
        else:
            dist_to_next = max(0, self.total_distance - self.current_km)
            next_power = 0
            station_occ = 0
            price = 0

        # Egim
        grade = self.elevation.grade_at_km(self.current_km)

        # Trafik
        traffic_factor = self.traffic.speed_factor_at(self.current_km, self.current_hour)

        # Saat kodlama (sin/cos)
        hour_rad = 2 * np.pi * (self.current_hour % 24) / 24.0

        obs = np.array([
            np.clip(self.soc, 0, 1),                                    # 0: SoC
            np.clip(self.current_km / self.total_distance, 0, 1),       # 1: Pozisyon
            np.clip(dist_to_next / self.total_distance, 0, 1),          # 2: Sonraki ist. mesafe
            np.clip(next_power / 300.0, 0, 1),                          # 3: Ist. gucu
            np.clip((self.total_distance - self.current_km) / self.total_distance, 0, 1),  # 4: Kalan mesafe
            np.clip((grade + 10) / 20.0, 0, 1),                         # 5: Egim ([-10,10] -> [0,1])
            (np.sin(hour_rad) + 1) / 2,                                  # 6: Saat sin
            (np.cos(hour_rad) + 1) / 2,                                  # 7: Saat cos
            np.clip((self.weather.temperature_c + 20) / 60.0, 0, 1),   # 8: Sicaklik ([-20,40] -> [0,1])
            np.clip((self.weather.wind_speed_ms) / 15.0, 0, 1),        # 9: Ruzgar
            np.clip(self.continuous_drive_min / 240.0, 0, 1),            # 10: Surus suresi
            np.clip(traffic_factor, 0, 1),                               # 11: Trafik
            np.clip(station_occ, 0, 1),                                  # 12: Ist. yogunlugu
            np.clip(price / 12.0, 0, 1),                                 # 13: Fiyat
        ], dtype=np.float32)

        return obs

    def _get_info(self) -> dict:
        return {
            "soc": self.soc,
            "km": self.current_km,
            "hour": self.current_hour,
            "station_idx": self.station_idx,
            "drive_time_h": self.total_drive_time_h,
            "charge_time_min": self.total_charge_time_min,
            "wait_time_min": self.total_wait_time_min,
            "cost_tl": self.total_cost_tl,
            "continuous_drive_min": self.continuous_drive_min,
        }

    @staticmethod
    def _path_hav_km(a, b) -> float:
        R = 6371.0
        p1, p2 = np.radians(a[0]), np.radians(b[0])
        dp = np.radians(b[0] - a[0])
        dl = np.radians(b[1] - a[1])
        x = np.sin(dp/2)**2 + np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
        return float(2 * R * np.arctan2(np.sqrt(x), np.sqrt(1-x)))

    def _detour_energy_kwh(self, path, speed_kmh: float = 55.0) -> float:
        """
        Sapma yolu ([lat,lng,elev] noktaları) boyunca gerçek eğimli
        enerji tüketimi (kWh). Tali yol hızı varsayılır.
        """
        if not path or len(path) < 2:
            return 0.0
        total = 0.0
        for i in range(len(path) - 1):
            d_km = self._path_hav_km(path[i], path[i + 1])
            if d_km <= 0:
                continue
            dz = (path[i + 1][2] - path[i][2]) if len(path[i]) > 2 else 0.0
            grade = (dz / (d_km * 1000.0)) * 100.0
            grade = max(-12.0, min(12.0, grade))
            e_per_km = energy_consumption_kwh_per_km(
                self.vehicle, speed_kmh,
                hvac_mode=self.hvac_mode,
                grade_percent=grade,
                extra_mass_kg=self.load.extra_mass_kg,
                temperature_c=self.weather.temperature_c,
                altitude_m=path[i][2] if len(path[i]) > 2 else 0.0,
                wet_road=self.weather.is_raining,
            )
            total += max(0.0, e_per_km) * d_km * self.battery_temp_factor
        return total

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Bir adim at: istasyona sur, (opsiyonel) sarj yap.

        Returns: (obs, reward, terminated, truncated, info)
        """
        # --- Aksiyon coz ---
        speed_idx = action // len(CHARGE_TARGETS)
        charge_idx = action % len(CHARGE_TARGETS)
        desired_speed = SPEEDS[speed_idx]
        charge_target = CHARGE_TARGETS[charge_idx]  # None veya 0.30-0.90

        # --- Hedef belirle ---
        if self.station_idx >= len(self.stations):
            # Tum istasyonlar gecildi, hedefe sur
            target_km = self.total_distance
        else:
            target_km = self.stations[self.station_idx]["road_km"]

        segment_distance = target_km - self.current_km
        if segment_distance <= 0:
            # Zaten istasyondayiz, sonrakine gec
            self.station_idx += 1
            if self.station_idx >= len(self.stations):
                target_km = self.total_distance
            else:
                target_km = self.stations[self.station_idx]["road_km"]
            segment_distance = target_km - self.current_km

        if segment_distance <= 0:
            # Hedefe vardik
            return self._get_obs(), 0.0, True, False, self._get_info()

        # --- Surus ---
        # Trafige gore efektif hiz (surucu toleransi dahil)
        eff_speed = self.traffic.effective_speed(
            self.current_km, self.current_hour, desired_speed,
            speed_tolerance=self.driver.speed_tolerance,
        )
        eff_speed = max(eff_speed, 30.0)  # Minimum 30 km/h

        # Surus suresi
        drive_time_h = segment_distance / eff_speed

        # --- Enerji: segmenti ~2 km alt-adimlara bolup GERCEK egim
        #     profili boyunca integre et (tek orta-nokta artefaktini kaldirir) ---
        SUB_KM = 2.0
        headwind_ms = self.weather.wind_speed_ms * np.cos(
            np.radians(self.weather.wind_direction_deg - 90)
        )
        n_sub = max(1, int(np.ceil(segment_distance / SUB_KM)))
        sub_len = segment_distance / n_sub
        energy_consumed = 0.0
        grade_acc = 0.0
        for k in range(n_sub):
            sub_mid_km = self.current_km + (k + 0.5) * sub_len
            g = self.elevation.grade_at_km(sub_mid_km)
            alt = self.elevation.altitude_at_km(sub_mid_km)
            grade_acc += g
            e_sub = energy_consumption_kwh_per_km(
                self.vehicle, eff_speed,
                hvac_mode=self.hvac_mode,
                grade_percent=g,
                extra_mass_kg=self.load.extra_mass_kg,
                headwind_ms=headwind_ms,
                temperature_c=self.weather.temperature_c,
                altitude_m=alt,
                wet_road=self.weather.is_raining,
            )
            energy_consumed += e_sub * sub_len
        # Hava durumu batarya etkisi
        energy_consumed *= self.battery_temp_factor
        avg_grade = grade_acc / n_sub
        avg_altitude = self.elevation.altitude_at_km(
            self.current_km + segment_distance / 2)

        soc_before = self.soc
        self.soc -= energy_consumed / self.vehicle.battery_usable_kwh

        # --- Batarya oldu mu? ---
        if self.soc <= 0:
            self.soc = 0.0
            reward_info = self._reward_fn(
                self.driver, drive_time_h, 0, 0, 0, "Generic_DC",
                self.current_hour, 0, 0, 0, self.continuous_drive_min,
                is_dead=True
            )
            self.history.append({
                "from_km": self.current_km, "to_km": target_km,
                "speed": eff_speed, "soc_before": soc_before, "soc_after": 0,
                "drive_time_h": drive_time_h, "charge_time_min": 0,
                "event": "DEAD",
            })
            return self._get_obs(), reward_info["total"], True, False, self._get_info()

        # --- State guncelle (surus) ---
        self.current_km = target_km
        self.current_hour += drive_time_h
        self.continuous_drive_min += drive_time_h * 60
        self.total_drive_time_h += drive_time_h
        self.total_energy_kwh += energy_consumed

        # --- Sarj ---
        charge_time_min = 0.0
        wait_time_min = 0.0
        energy_charged = 0.0
        station_type = "Generic_DC"
        station_power = 0
        target_soc_pct = self.soc * 100
        detour_km = 0.0
        detour_min = 0.0
        detour_out = []
        detour_back = []

        if (charge_target is not None
                and self.station_idx < len(self.stations)
                and self.soc < charge_target):

            station = self.stations[self.station_idx]
            station_power = station["power_kw"]
            station_type = station.get("type", "Generic_DC")
            station_slots = station.get("slots", 2)

            # --- SAPMA: otoyoldan çık -> istasyona git ---
            # (Yalnızca gerçekten şarj edilecekse sapılır; geçilirse yok.)
            # EN: Prefer the real detour cost from single-source Google
            #     A->C->B routing (extra_km / extra_min); fall back to the
            #     anchor path when Google extras are absent.
            # TR: Sapma maliyeti olarak tek-kaynak Google A->C->B'den
            #     gerçek değer (extra_km / extra_min) tercih edilir;
            #     Google verisi yoksa çapa yoluna düşülür.
            has_g = "extra_km" in station
            detour_out = [] if has_g else (station.get("detour_out_path") or [])
            detour_back = [] if has_g else (station.get("detour_back_path") or [])
            if has_g:
                detour_km = float(station.get("extra_km", 0.0) or 0.0)
                detour_min = float(station.get("extra_min", 0.0) or 0.0)
            else:
                detour_km = float(station.get("detour_km", 0.0) or 0.0)
                detour_min = float(station.get("detour_min", 0.0) or 0.0)

            # Sapma enerjisi: GERÇEK ekstra mesafe (extra_km) üzerinden,
            # eğim = istasyonun rota noktasına göre NET yükseklik farkı
            # (gerçek profil; 'dev' uzunluğu alternatif-yol içerebildiği
            # için mesafe ondan DEĞİL extra_km'den alınır — şişme yok).
            # Gidiş tırmanış, dönüş iniş (potansiyel enerji ~ telafi).
            g_dev = station.get("g_dev") if has_g else None
            if has_g and detour_km > 0:
                grade_up = 0.0
                if g_dev:
                    sc = (station.get("lat"), station.get("lng"))
                    di = (min(range(len(g_dev)),
                              key=lambda i: (g_dev[i][0] - sc[0]) ** 2
                              + (g_dev[i][1] - sc[1]) ** 2)
                          if sc[0] is not None else len(g_dev) // 2)
                    elev_st = g_dev[di][2] if len(g_dev[di]) > 2 else None
                    elev_rt = g_dev[0][2] if len(g_dev[0]) > 2 else None
                    half_m = max(1.0, (detour_km / 2.0) * 1000.0)
                    if elev_st is not None and elev_rt is not None:
                        grade_up = max(-12.0, min(
                            12.0, (elev_st - elev_rt) / half_m * 100.0))

                def _epk(gr):
                    e = energy_consumption_kwh_per_km(
                        self.vehicle, 55.0, hvac_mode=self.hvac_mode,
                        grade_percent=gr,
                        extra_mass_kg=self.load.extra_mass_kg,
                        temperature_c=self.weather.temperature_c,
                        wet_road=self.weather.is_raining)
                    return max(0.0, e) * self.battery_temp_factor

                e_out = _epk(grade_up) * (detour_km / 2.0)        # tırmanış
                e_back_g = _epk(-grade_up) * (detour_km / 2.0)     # iniş
            else:
                e_out = self._detour_energy_kwh(detour_out) if detour_out else 0.0
                e_back_g = None

            if e_out > 0:
                self.soc -= e_out / self.vehicle.battery_usable_kwh
                self.total_energy_kwh += e_out
                self.total_detour_km += detour_km
                # Sapmada batarya bitti mi? (istasyona ulaşamadan)
                if self.soc <= 0:
                    self.soc = 0.0
                    rinfo = self._reward_fn(
                        self.driver, drive_time_h, 0, 0, 0, "Generic_DC",
                        self.current_hour, 0, 0, 0, self.continuous_drive_min,
                        is_dead=True)
                    self.history.append({
                        "from_km": self.current_km, "to_km": self.current_km,
                        "speed": 0, "soc_before": soc_before, "soc_after": 0,
                        "drive_time_h": drive_time_h, "charge_time_min": 0,
                        "event": "DEAD_DETOUR",
                        "station": station["name"],
                        "detour_km": detour_km,
                    })
                    return (self._get_obs(), rinfo["total"],
                            True, False, self._get_info())

            # Kuyruk bekleme suresi
            wait_time_min = queue_wait_time(
                self.current_hour, station_slots, 0.5
            )

            target_soc_pct = charge_target * 100
            current_soc_pct = self.soc * 100

            # Sarj suresi (sicaklik etkisi dahil)
            charge_time_min = calculate_charge_time(
                self.charge_curve,
                current_soc_pct, target_soc_pct,
                station_power,
                self.vehicle.battery_usable_kwh,
                temperature_factor=self.charging_temp_factor,
            )

            # Enerji miktari
            energy_charged = energy_charged_kwh(
                current_soc_pct, target_soc_pct,
                self.vehicle.battery_usable_kwh
            )

            # Maliyet
            cost = energy_charged * get_electricity_price(station_type, self.current_hour)
            self.total_cost_tl += cost

            # SoC guncelle
            self.soc = charge_target

            # --- SAPMA dönüşü: istasyondan otoyola geri dön ---
            e_back = e_back_g if (has_g) else (
                self._detour_energy_kwh(detour_back) if detour_back else 0.0)
            if e_back:
                self.soc = max(0.0, self.soc - e_back / self.vehicle.battery_usable_kwh)
                self.total_energy_kwh += e_back

            # Zaman guncelle (sapmanın gerçek trafikli süresi dahil)
            detour_time_h = detour_min / 60.0
            self.total_detour_time_h += detour_time_h
            self.total_drive_time_h += detour_time_h
            self.continuous_drive_min += detour_min
            total_stop_min = wait_time_min + charge_time_min
            self.current_hour += total_stop_min / 60.0 + detour_time_h
            self.total_charge_time_min += charge_time_min
            self.total_wait_time_min += wait_time_min

        # --- Mola-sarj senkronizasyonu ---
        overlap = break_charge_overlap(
            self.continuous_drive_min,
            self.driver.max_continuous_drive_min,
            self.driver.preferred_break_min,
            charge_time_min,
        )

        # Eger mola alindiysa (sarjla birlikte veya tek basina), surus saati sifirla
        if overlap["needs_break"] and (charge_time_min > 0 or wait_time_min > 0):
            self.continuous_drive_min = 0.0
        elif overlap["needs_break"]:
            # Mola gerekli ama sarj yapilmadi -> mola suresi ekle
            break_time = self.driver.preferred_break_min
            self.current_hour += break_time / 60.0
            self.continuous_drive_min = 0.0

        # --- Sonraki istasyona gec ---
        self.station_idx += 1

        # --- Hedefe varis kontrolu ---
        terminated = self.current_km >= self.total_distance - 0.1

        # --- Reward hesapla ---
        reward_info = self._reward_fn(
            driver=self.driver,
            drive_time_h=drive_time_h,
            charge_time_min=charge_time_min,
            wait_time_min=wait_time_min,
            energy_kwh=energy_charged,
            station_type=station_type,
            hour=self.current_hour,
            current_soc=self.soc,
            target_soc_pct=target_soc_pct,
            station_power_kw=station_power,
            continuous_drive_min=self.continuous_drive_min,
        )

        # --- Gecmis kaydi ---
        self.history.append({
            "from_km": self.current_km - segment_distance,
            "to_km": self.current_km,
            "speed": eff_speed,
            "soc_before": soc_before,
            "soc_after": self.soc,
            "drive_time_h": drive_time_h,
            "charge_time_min": charge_time_min,
            "wait_time_min": wait_time_min,
            "cost_tl": self.total_cost_tl,
            "energy_charged": energy_charged,
            "overlap_saved_min": overlap["saved_time"],
            "station": self.stations[self.station_idx - 1]["name"] if self.station_idx <= len(self.stations) else "hedef",
            "action": (desired_speed, charge_target),
            "reward": reward_info["total"],
            "reward_detail": reward_info,
            "detour_km": detour_km,
            "detour_min": detour_min,
            "detour_out_path": detour_out if charge_time_min > 0 else [],
            "detour_back_path": detour_back if charge_time_min > 0 else [],
        })

        return self._get_obs(), reward_info["total"], terminated, False, self._get_info()

    def get_trip_summary(self) -> dict:
        """Yolculuk ozet istatistikleri."""
        return {
            "total_time_h": self.total_drive_time_h + (self.total_charge_time_min + self.total_wait_time_min) / 60,
            "drive_time_h": self.total_drive_time_h,
            "charge_time_min": self.total_charge_time_min,
            "wait_time_min": self.total_wait_time_min,
            "total_cost_tl": self.total_cost_tl,
            "total_energy_kwh": self.total_energy_kwh,
            "arrival_soc": self.soc,
            "num_charges": sum(1 for h in self.history if h.get("charge_time_min", 0) > 0),
            "total_overlap_saved_min": sum(h.get("overlap_saved_min", 0) for h in self.history),
            "distance_km": self.current_km,
            "total_detour_km": self.total_detour_km,
            "total_detour_time_min": self.total_detour_time_h * 60,
        }


# ``make_env`` lives in :mod:`evroute.env.factory` (factory split out so
# agents/baselines can import the env without pulling the factory).
# ``make_env`` :mod:`evroute.env.factory` içindedir (fabrika ayrıldı ki
# ajan/baseline'lar fabrikayı çekmeden ortamı içe aktarabilsin).
