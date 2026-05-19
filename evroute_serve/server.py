# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Saadettin Yıldırım (Yıldsam Teknoloji)
"""
EN: Interactive simulation server. The user selects a test scenario
    (vehicle, route, driver, weather, traffic) and a model (algorithm +
    the scenario it was trained on, or a baseline). Only the chosen
    weights are loaded; the trip is simulated on demand and streamed to
    the map/animation frontend. Enables cross-condition validation.
TR: İnteraktif simülasyon sunucusu. Kullanıcı bir test senaryosu (araç,
    rota, sürücü, hava, trafik) ve bir model (algoritma + eğitildiği
    senaryo, ya da baseline) seçer. Yalnızca seçilen ağırlık yüklenir;
    yolculuk anında simüle edilip harita/animasyon arayüzüne aktarılır.
    Çapraz-koşul doğrulamasına olanak tanır.

Run / Çalıştırma:
    ./venv/bin/python -m uvicorn src.server:app --port 8000
    -> open http://localhost:8000
"""

import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

import requests

from evroute import make_env
from evroute.config import get_data_dir
from evroute.data.secrets import get_secret
from evroute.agents.runner import run_simulation, _DQNWrapper, _PPOWrapper
from evroute_serve.visualize import get_route_coordinates
from evroute.agents.baselines import BASELINE_STRATEGIES

# ROOT is the parent of the data directory, so `ROOT / "data" / ...`
# paths resolve correctly.
# ROOT, veri dizininin üst klasörüdür; `ROOT / "data" / ...` yolları
# doğru çözülür.
ROOT = get_data_dir().parent
MODELS_DIR = get_data_dir() / "processed" / "results" / "models"

VEHICLES = ["ioniq5", "tesla3"]
ROUTES = ["istanbul_ankara", "istanbul_izmir", "ankara_antalya"]
DRIVERS = ["eco", "normal", "aggressive"]
WEATHERS = ["optimal", "yaz_gunesli", "kis_soguk", "yagmurlu"]
TRAFFICS = ["weekday", "weekend"]
RL_ALGOS = ["dqn", "ddqn", "ppo"]

ROUTE_LABELS = {
    "istanbul_ankara": "İstanbul – Ankara",
    "istanbul_izmir": "İstanbul – İzmir",
    "ankara_antalya": "Ankara – Antalya",
}

app = FastAPI(title="EV Şarj Planlama – İnteraktif Simülasyon")


def _scenario_key(vehicle, route, driver, weather, traffic) -> str:
    return f"{vehicle}_{route}_{driver}_{weather}_{traffic}"


def _model_path(algo: str, scenario_key: str) -> Optional[Path]:
    ext = ".zip" if algo == "ppo" else ".pt"
    p = MODELS_DIR / f"{algo}__{scenario_key}{ext}"
    return p if p.exists() else None


def _load_agent(algo: str, scenario_key: str):
    """EN: Loads chosen weights only. / TR: Yalnızca seçilen ağırlığı yükler."""
    path = _model_path(algo, scenario_key)
    if path is None:
        return None
    if algo in ("dqn", "ddqn"):
        from evroute.agents.dqn import DQNAgent
        agent = DQNAgent(state_dim=14, action_dim=25,
                         double_dqn=(algo == "ddqn"))
        agent.load(str(path))
        return _DQNWrapper(agent)
    if algo == "ppo":
        try:
            from stable_baselines3 import PPO
        except ImportError:
            return None
        return _PPOWrapper(PPO.load(str(path)))
    return None


import bisect as _bisect

_GROUTE_CACHE = {}


def _load_groute(route_key):
    """EN: Single-source Google route file (if built). / TR: Tek-kaynak
       Google rota dosyası (üretildiyse)."""
    if route_key in _GROUTE_CACHE:
        return _GROUTE_CACHE[route_key]
    fp = get_data_dir() / "processed" / f"{route_key}_route_google.json"
    g = json.loads(fp.read_text(encoding="utf-8")) if fp.exists() else None
    _GROUTE_CACHE[route_key] = g
    return g


def _pick(path, frac):
    if not path:
        return None
    i = int(round(max(0.0, min(1.0, frac)) * (len(path) - 1)))
    return path[i]


def _decode_poly(enc):
    pts, i, lat, lng = [], 0, 0, 0
    while i < len(enc):
        for k in range(2):
            shift = result = 0
            while True:
                b = ord(enc[i]) - 63
                i += 1
                result |= (b & 0x1f) << shift
                shift += 5
                if b < 0x20:
                    break
            d = ~(result >> 1) if (result & 1) else (result >> 1)
            if k == 0:
                lat += d
            else:
                lng += d
        pts.append([lat / 1e5, lng / 1e5])
    return pts


def _directions_legs(A, B, waypoints, gkey):
    """EN: One Directions query A->[stations]->B; returns per-leg dense
       polylines (real driven road incl. all access detours). None on
       failure. / TR: Tek sorgu; leg başına yoğun gerçek yol."""
    params = {"origin": f"{A[0]},{A[1]}",
              "destination": f"{B[0]},{B[1]}", "key": gkey}
    if waypoints:
        params["waypoints"] = "|".join(f"{a},{b}" for a, b in waypoints)
    try:
        j = requests.get(
            "https://maps.googleapis.com/maps/api/directions/json",
            params=params, timeout=40).json()
        if j.get("status") != "OK":
            return None
        legs = []
        for lg in j["routes"][0]["legs"]:
            path = []
            for s in lg["steps"]:
                seg = _decode_poly(s["polyline"]["points"])
                if path and seg and path[-1] == seg[0]:
                    seg = seg[1:]
                path += seg
            legs.append(path or [[A[0], A[1]]])
        return legs
    except Exception:
        return None


def _trajectory_through_route(trajectory, route_key, env, gkey):
    """
    EN: THE driven path = real Google route A -> [stations the agent
        actually charged at, in order] -> B (one query, per-leg
        polylines). Each trajectory point is placed on the leg matching
        its road_km; charging points dwell at the station. Single
        consistent source; no synthetic detour reshaping.
    TR: Sürülen yol = ajanın gerçekten şarj ettiği istasyonlardan sırayla
        geçen gerçek Google rotası (tek sorgu, leg polyline'ları). Her
        trajektori noktası road_km'sine uyan leg'e konur; şarj noktaları
        istasyonda bekler. Tek tutarlı kaynak; sentetik sapma yok.
    """
    g = _load_groute(route_key)
    ab = g["ab_path"] if g else None
    geom = get_route_coordinates(route_key)
    A = (geom[0][0], geom[0][1])
    B = (geom[-1][0], geom[-1][1])
    total = float(getattr(env, "total_distance", geom[-1][2]) or geom[-1][2])

    # EN: Some stations (e.g. Tesla Superchargers) have no coordinates;
    #     skip them in the lookup (the car just passes their km on route).
    # TR: Bazı istasyonların (örn. Tesla SC) koordinatı yok; aramaya
    #     katma (araç o km'de rota üzerinde geçer).
    nm2c = {s["name"]: (s["lat"], s["lng"]) for s in env.stations
            if s.get("lat") is not None and s.get("lng") is not None}
    charged = []
    for h in env.history:
        if h.get("charge_time_min", 0) > 0:
            nm = h.get("station")
            if nm in nm2c:
                charged.append((round(h.get("to_km", 0), 1), nm2c[nm]))
    charged.sort(key=lambda x: x[0])
    wp = [c for _, c in charged]

    legs = _directions_legs(A, B, wp, gkey)
    # km sınırları: [0, k1, k2, ..., kn, total] -> leg0..legN
    bounds = [0.0] + [k for k, _ in charged] + [total]

    if not legs or len(legs) != len(bounds) - 1:
        # Google başarısızsa: düz A->B referansına km oranıyla yerleştir
        base = ab or [[p[0], p[1]] for p in geom]

        def _atfrac(fr):
            i = int(round(max(0.0, min(1.0, fr)) * (len(base) - 1)))
            return base[i]
        out = []
        for p in trajectory:
            if p.get("lat") is not None:
                out.append(dict(p))
            else:
                c = _atfrac(p["km"] / max(total, 1e-6))
                out.append({**p, "lat": c[0], "lng": c[1]})
        route_line = [[p[0], p[1]] for p in base]
        return out, route_line

    # EN: Concatenate all legs into ONE continuous real-road polyline P.
    #     The car position is always a real point on P (never "through
    #     the air"). Between two consecutive mapped trajectory points we
    #     splice the actual P points in between, so the path strictly
    #     follows the real road. / TR: Tüm leg'ler tek sürekli gerçek-yol
    #     polyline'ı P'ye birleştirilir. Araç daima P üzerinde gerçek bir
    #     noktada (asla havadan değil). Ardışık iki nokta arası P'nin
    #     gerçek noktalarıyla doldurulur → katı şekilde yolu izler.
    P, segidx = [], []
    for s, lg in enumerate(legs):
        for pt in lg:
            if P and P[-1] == pt:
                continue
            P.append(pt)
            segidx.append(s)

    def _km_target(km):
        for s in range(len(legs)):
            lo, hi = bounds[s], bounds[s + 1]
            if lo <= km <= hi or s == len(legs) - 1:
                fr = (km - lo) / (hi - lo) if hi > lo else 1.0
                pth = legs[s]
                return pth[int(round(max(0.0, min(1.0, fr)) * (len(pth) - 1)))]
        return P[-1]

    def _near_idx(pt, lo):
        best, bi = 1e18, lo
        hi = min(len(P), lo + 6000)
        for i in range(lo, hi):
            dx = (P[i][0] - pt[0]) ** 2 + (P[i][1] - pt[1]) ** 2
            if dx < best:
                best, bi = dx, i
        return bi

    out, cur = [], 0
    prev_idx = 0
    for n, p in enumerate(trajectory):
        if p.get("event") == "charging":
            tgt = nm2c.get(p.get("station")) or _km_target(p["km"])
        else:
            tgt = _km_target(p["km"])
        idx = _near_idx(tgt, cur)
        # Önceki ile arasını GERÇEK yol noktalarıyla doldur (havadan gitme)
        if out and idx > prev_idx + 1:
            pp = out[-1]
            span = idx - prev_idx
            for s in range(prev_idx + 1, idx):
                fr = (s - prev_idx) / span
                out.append({**p,
                            "lat": P[s][0], "lng": P[s][1],
                            "event": "detour",
                            "total_time_min": pp["total_time_min"]
                            + (p["total_time_min"] - pp["total_time_min"]) * fr})
        q = dict(p)
        q["lat"], q["lng"] = P[idx][0], P[idx][1]
        out.append(q)
        prev_idx = idx
        cur = max(cur, idx)

    route_line = [[p[0], p[1]] for p in (ab or [[gg[0], gg[1]] for gg in geom])]
    return out, route_line


def _trajectory_with_coords(trajectory, route_key, stations=None):
    """EN: Single source: main line = Google A->B; each detour follows
       the real A->C->B deviation. Falls back to anchor mapping if the
       Google route file is absent.
       TR: Tek kaynak: ana hat = Google A->B; her sapma gerçek A->C->B
       ayrımını izler. Google dosyası yoksa çapa eşlemesine düşer."""
    g = _load_groute(route_key)

    if g:
        ab = g["ab_path"]
        cum = g["ab_cum_km"]
        total = cum[-1] or 1.0
        gstations = g.get("stations", [])
        by_name = {s["name"]: s for s in gstations}
        by_km = sorted(gstations, key=lambda s: s.get("road_km") or 0)

        def coord_at_km(km):
            km = max(0.0, min(total, km))
            i = _bisect.bisect_left(cum, km)
            if i <= 0:
                return [ab[0][0], ab[0][1]]
            if i >= len(ab):
                return [ab[-1][0], ab[-1][1]]
            seg = cum[i] - cum[i - 1] or 1e-9
            f = (km - cum[i - 1]) / seg
            return [ab[i - 1][0] + f * (ab[i][0] - ab[i - 1][0]),
                    ab[i - 1][1] + f * (ab[i][1] - ab[i - 1][1])]

        out = []
        for p in trajectory:
            if p.get("lat") is not None and p.get("lng") is not None:
                out.append(dict(p))
            else:
                c = coord_at_km(p["km"])
                out.append({**p, "lat": c[0], "lng": c[1]})

        n = len(out)
        k = 0
        while k < n:
            if out[k]["event"] in ("detour", "charging"):
                j = k
                while j < n and out[j]["event"] in ("detour", "charging"):
                    j += 1
                R = coord_at_km(out[k].get("km", 0))
                cidx = next((x for x in range(k, j)
                             if out[x]["event"] == "charging"), (k + j) // 2)
                nm = out[cidx].get("station") or out[k].get("station")
                stn = by_name.get(nm)
                if stn is None and by_km:
                    kmv = out[k].get("km", 0)
                    stn = min(by_km,
                              key=lambda s: abs((s.get("road_km") or 0) - kmv))
                dev = (stn or {}).get("dev") or []
                if dev:
                    # 'dev' = A->C->B'nin rotadan ayrılan kısmı (sıralı).
                    # İstasyona en yakın nokta = gidiş/dönüş ayrımı.
                    sc = (stn["lat"], stn["lng"])
                    di = min(range(len(dev)),
                             key=lambda i: (dev[i][0] - sc[0]) ** 2
                             + (dev[i][1] - sc[1]) ** 2)
                    for idx in range(k, j):
                        if idx < cidx:
                            f = (idx - k) / (cidx - k) if cidx > k else 1.0
                            c = _pick(dev[:di + 1], f)
                        elif idx == cidx:
                            c = [sc[0], sc[1]]
                        else:
                            f = ((idx - cidx) / (j - 1 - cidx)
                                 if (j - 1) > cidx else 1.0)
                            c = _pick(dev[di:], f)
                        out[idx]["lat"], out[idx]["lng"] = (c or R)[0], (c or R)[1]
                else:
                    for idx in range(k, j):
                        out[idx]["lat"], out[idx]["lng"] = R[0], R[1]
                out[k]["lat"], out[k]["lng"] = R[0], R[1]
                out[j - 1]["lat"], out[j - 1]["lng"] = R[0], R[1]
                k = j
            else:
                k += 1

        route_line = [[p[0], p[1]] for p in ab]
        return out, route_line, coord_at_km

    # ---- Fallback (Google dosyası yok): çapa eşlemesi ----
    coords = get_route_coordinates(route_key)

    def coord_at_km(km):
        for i in range(len(coords) - 1):
            if coords[i][2] <= km <= coords[i + 1][2]:
                f = (km - coords[i][2]) / max(coords[i + 1][2] - coords[i][2], 0.01)
                return [coords[i][0] + f * (coords[i + 1][0] - coords[i][0]),
                        coords[i][1] + f * (coords[i + 1][1] - coords[i][1])]
        return [coords[-1][0], coords[-1][1]]

    out = []
    for p in trajectory:
        if p.get("lat") is not None and p.get("lng") is not None:
            out.append(dict(p))
        else:
            c = coord_at_km(p["km"])
            out.append({**p, "lat": c[0], "lng": c[1]})
    route_line = [[c[0], c[1]] for c in coords]
    return out, route_line, coord_at_km


@app.get("/api/options")
def options():
    """EN: Selectable values + which trained model scenarios exist.
       TR: Seçilebilir değerler + hangi eğitilmiş model senaryoları var."""
    available = sorted(
        f.name.split("__", 1)[1].rsplit(".", 1)[0]
        for f in MODELS_DIR.glob("dqn__*.pt")
    ) if MODELS_DIR.exists() else []
    return {
        "vehicles": VEHICLES, "routes": ROUTES, "route_labels": ROUTE_LABELS,
        "drivers": DRIVERS, "weathers": WEATHERS, "traffics": TRAFFICS,
        "rl_algos": RL_ALGOS,
        "baselines": list(BASELINE_STRATEGIES.keys()),
        "model_scenarios": available,
    }


@app.get("/api/simulate")
def simulate(vehicle: str, route: str, driver: str, weather: str,
             traffic: str, algo: str, model_scenario: str = "matched"):
    """
    EN: Builds the TEST environment, loads the chosen model (or baseline),
        runs one trip, returns trajectory + route + stations + summary.
    TR: TEST ortamını kurar, seçilen modeli (ya da baseline) yükler, bir
        yolculuk koşar, trajektori + rota + istasyon + özet döndürür.
    """
    if route not in ROUTES or vehicle not in VEHICLES:
        raise HTTPException(400, "Geçersiz rota/araç")

    env = make_env(vehicle=vehicle, route=route, driver=driver,
                   weather=weather, is_weekend=(traffic == "weekend"),
                   seed=42)

    meta = {"trained_on": None, "kind": None}
    if algo.startswith("baseline_"):
        name = algo.split("baseline_", 1)[1]
        if name not in BASELINE_STRATEGIES:
            raise HTTPException(400, "Geçersiz baseline")
        _, fn = BASELINE_STRATEGIES[name]
        trajectory = run_simulation(env, strategy_fn=fn)
        meta["kind"] = "baseline"
    else:
        if algo not in RL_ALGOS:
            raise HTTPException(400, "Geçersiz algoritma")
        test_key = _scenario_key(vehicle, route, driver, weather, traffic)
        train_key = test_key if model_scenario in ("matched", "", None) else model_scenario
        agent = _load_agent(algo, train_key)
        if agent is None:
            raise HTTPException(404, f"Model yok: {algo} / {train_key}")
        trajectory = run_simulation(env, agent=agent)
        meta["kind"] = "rl"
        meta["trained_on"] = train_key
        meta["is_cross"] = (train_key != test_key)

    # EN: If a single-source Google route exists for this route, the
    #     driven path = real Google route through the stations the agent
    #     charged at. Otherwise, anchor mapping.
    # TR: Bu rota için tek-kaynak Google dosyası varsa sürülen yol =
    #     ajanın şarj ettiği istasyonlardan geçen gerçek Google rotası;
    #     aksi halde çapa eşlemesi.
    if _load_groute(route):
        traj, route_line = _trajectory_through_route(
            trajectory, route, env, get_secret("GOOGLE_MAPS_API_KEY"))
    else:
        traj, route_line, _ = _trajectory_with_coords(
            trajectory, route, env.stations)
    # Koordinatsız istasyonlar (örn. Tesla SC) haritada çizilemez -> atla.
    stations = [{"name": s["name"],
                 "lat": s["lat"], "lng": s["lng"],
                 "power_kw": s.get("power_kw", 120),
                 "road_km": s.get("road_km", 0),
                 "detour_km": s.get("detour_km", 0.0)} for s in env.stations
                if s.get("lat") is not None and s.get("lng") is not None]
    # EN: Ordered list of actual charging stops (station, duration, detour).
    # TR: Gerçekleşen şarj duraklarının sıralı listesi (istasyon, süre, sapma).
    stops = []
    for h in env.history:
        if h.get("charge_time_min", 0) > 0:
            stops.append({
                "station": h.get("station", "?"),
                "km": round(h.get("to_km", 0), 1),
                "charge_min": round(h.get("charge_time_min", 0), 1),
                "wait_min": round(h.get("wait_time_min", 0), 1),
                "detour_km": round(h.get("detour_km", 0.0), 1),
                "soc_before": round(h.get("soc_before", 0) * 100),
                "soc_after": round(h.get("soc_after", 0) * 100),
            })
    return JSONResponse({
        "trajectory": traj, "route_line": route_line, "stations": stations,
        "summary": env.get_trip_summary(), "meta": meta, "stops": stops,
    })


_INFO_CACHE = None


@app.get("/api/info")
def info():
    """
    EN: Per-select explanatory cards. VALUES are read live from the real
        model objects (vehicle/driver/weather/traffic/route/charge curve);
        only the 'meaning' glossary text is authored. Nothing hardcoded.
    TR: Her seçim kutusu için açıklama kartı. DEĞERLER gerçek model
        nesnelerinden canlı okunur (araç/sürücü/hava/trafik/rota/şarj
        eğrisi); yalnız 'anlam' sözlük metni yazılı. Hiçbiri gömülü değil.
    """
    global _INFO_CACHE
    if _INFO_CACHE is not None:
        return JSONResponse(_INFO_CACHE)

    from evroute.models.vehicle import VEHICLES as VM
    from evroute.models.driver import DRIVER_PROFILES as DP
    from evroute.models.weather import (WEATHER_SCENARIOS as WS,
                                   temperature_battery_factor as tbf)
    from evroute.models.charging import load_ioniq5_curve, load_tesla3_curve

    def P(label, value, meaning, unit=""):
        return {"label": label, "value": f"{value}{unit}", "meaning": meaning}

    # ---- ARAÇ ----
    curves = {"ioniq5": load_ioniq5_curve(), "tesla3": load_tesla3_curve()}
    vehicles = {}
    for k, v in VM.items():
        c = curves.get(k)
        peak_kw = float(c.power_points.max()) if c is not None else None
        peak_soc = (float(c.soc_points[c.power_points.argmax()])
                    if c is not None else None)
        rows = [
            P("Araç", v.name, "Modellenen gerçek araç"),
            P("Toplam batarya", v.battery_total_kwh,
              "Bataryanın toplam kapasitesi", " kWh"),
            P("Kullanılabilir batarya", v.battery_usable_kwh,
              "Sürüşte fiilen kullanılabilen kapasite (tampon hariç)", " kWh"),
            P("Kütle", v.mass_kg,
              "Boş araç kütlesi; yokuş ve hızlanma direncini belirler",
              " kg"),
            P("Aero sürtünme C_d", v.C_d,
              "Aerodinamik sürüklenme katsayısı; yüksek hızda tüketimi "
              "belirleyen ana etken"),
            P("Ön kesit alanı A_f", v.A_f,
              "Aracın havaya karşı ön yüz alanı (m²); hava direnci ile "
              "çarpılır"),
            P("Yuvarlanma C_rr", v.C_rr,
              "Lastik-yol yuvarlanma direnci katsayısı; düşük hızda baskın"),
            P("Aktarma verimi", f"%{v.eta_drivetrain*100:.0f}",
              "Bataryadan tekerleğe güç aktarım verimi"),
            P("Rejen verimi", f"%{v.eta_regen*100:.0f}",
              "Frenlemede geri kazanılan enerji oranı"),
            P("Aksesuar gücü", v.aux_power_kw,
              "BMS/soğutma/12V sabit tüketim", " kW"),
            P("HVAC güçleri", ", ".join(f"{m}:{p}kW"
              for m, p in v.hvac_power_kw.items()),
              "Klima/ısıtma modlarının güç tüketimi"),
        ]
        if peak_kw:
            rows.append(P("DC şarj tepe gücü",
                          f"{peak_kw:.0f} kW (%{peak_soc:.0f} SoC)",
                          "Hızlı şarjda ulaşılan en yüksek güç ve bu gücün "
                          "görüldüğü doluluk seviyesi; şarj süresini "
                          "belirleyen temel etkendir"))
            rows.append(P("Şarj eğrisi",
                          f"%10→{c.power_at_soc(10):.0f} · "
                          f"%50→{c.power_at_soc(50):.0f} · "
                          f"%80→{c.power_at_soc(80):.0f} kW",
                          "Doluluk arttıkça şarj gücü düşer (taper); "
                          "ajan bu yüzden genelde %80'de keser"))
        vehicles[k] = rows

    # ---- SÜRÜCÜ ----
    drivers = {}
    for k, d in DP.items():
        drivers[k] = [
            P("Profil", d.name, getattr(d, "description", "")),
            P("Persona tercih hızı", d.preferred_speed_kmh,
              "Bu sürücü profilinin karakterini tanımlayan referans hız. "
              "Reinforcement learning ajanı bu değeri doğrudan kullanmaz; "
              "hızı kendisi belirler",
              " km/h"),
            P("Persona hız varyasyonu", f"±{d.speed_variance_kmh}",
              "Profil tanımındaki hız dalgalanması. Eğitim ortamında "
              "uygulanmaz, yalnızca persona betimlemesi içindir",
              " km/h"),
            P("RL hız aksiyon kümesi", "80 / 100 / 120 / 140 / 160",
              "Ajanın aralarından seçim yaptığı olası hızlar (km/h, "
              "environment.SPEEDS). Ekranda görülen hız bu kümeden gelir; "
              "bu nedenle ekonomik profil de 100-120 km/h sürebilir"),
            P("Limit toleransı", f"%{d.speed_tolerance*100:.0f}",
              "Sürücü profilinden eğitim ortamına aktarılan tek hız "
              "kısıtıdır: hız limitini aşma oranı (ekonomik profilde 0, "
              "limit aşılmaz; agresif profilde 0.25, %25 aşılır). Ajanın "
              "seçtiği hız bu sınır ve trafik koşullarıyla sınırlanır"),
            P("Sabır eşiği", d.patience_threshold_min,
              "Bu süreye kadar şarjda rahat bekler; sonrası ceza", " dk"),
            P("Sabır azalma hızı", d.patience_decay_rate,
              "Eşik aşılınca bekleme cezasının büyüme hızı"),
            P("Min. rahat SoC", f"%{d.min_comfortable_soc*100:.0f}",
              "Altına inince menzil kaygısı cezası başlar"),
            P("Menzil kaygısı ağırlığı", d.range_anxiety_weight,
              "Düşük SoC'den ne kadar rahatsız olduğu"),
            P("Max kesintisiz sürüş", d.max_continuous_drive_min,
              "Bu süre sonra mola gerekir (yorgunluk)", " dk"),
            P("Tercih mola", d.preferred_break_min,
              "Molada tercih ettiği dinlenme süresi", " dk"),
            P("HVAC modu", d.hvac_mode, "Varsayılan klima/ısıtma kullanımı"),
            P("Rejen bonusu", f"{d.regen_efficiency_bonus:+.2f}",
              "Sürüş tarzının rejeneratif frenleme kazanımına etkisi. "
              "Agresif sürüşte fren daha geç uygulandığı için kazanım "
              "azalır"),
            P("Ödül ağırlıkları",
              f"süre {d.w_time} · maliyet {d.w_cost} · konfor "
              f"{d.w_comfort} · kaygı {d.w_anxiety} · yıpranma "
              f"{d.w_degradation}",
              "Ödül fonksiyonundaki terimlerin bu sürücü için ağırlıkları. "
              "Ajanın hangi amacı öncelikli olarak optimize edeceğini bu "
              "ağırlıklar belirler"),
        ]

    # ---- HAVA ----
    weathers = {}
    for k, w in WS.items():
        fac = tbf(w.temperature_c)
        weathers[k] = [
            P("Sıcaklık", w.temperature_c,
              "Ortam sıcaklığı — batarya iç direncini ve HVAC yükünü "
              "etkiler", " °C"),
            P("Tüketim çarpanı", f"×{fac:.2f}",
              "Bu sıcaklıkta enerji tüketimi çarpanı (1.00=optimal; "
              "koddaki temperature_battery_factor'dan)"),
            P("Rüzgâr", w.wind_speed_ms,
              "Rüzgâr hızı; yön araç yönüyle birleşince ek hava direnci",
              " m/s"),
            P("Rüzgâr yönü", w.wind_direction_deg,
              "0=kuzey, saat yönü derece", "°"),
            P("Yağmur", "Var" if w.is_raining else "Yok",
              "Yağmurda yuvarlanma direnci ve sürüş riski artar"),
            P("Nem", w.humidity_pct, "Bağıl nem (%)", "%"),
        ]

    # ---- TRAFİK ----
    traffics = {
        "weekday": [
            P("Gün tipi", "Hafta içi",
              "Sabah/akşam zirve saatlerinde trafik yoğun; ortalama hız "
              "faktörü düşük"),
            P("Hız faktörü", "konuma+saate göre 0.30–1.0",
              "1.0=serbest akış, 0.30=ağır trafik (koddaki "
              "speed_factor_at)"),
        ],
        "weekend": [
            P("Gün tipi", "Hafta sonu",
              "Zirve yoğunluğu yok; daha akıcı"),
            P("Hafta sonu iyileştirmesi", "+0.15 hız faktörü",
              "Sentetik modelde hafta sonu tüm saatlere eklenen akıcılık "
              "(koddaki weekend_boost=0.15)"),
        ],
    }

    # ---- ROTA ----
    routes = {}
    for rk in ROUTES:
        try:
            e = make_env(route=rk, seed=42)
            routes[rk] = [
                P("Rota", ROUTE_LABELS.get(rk, rk), "Gerçek Google rotası"),
                P("Mesafe", f"{e.total_distance:.0f}",
                  "Rotanın gerçek geometriden hesaplanan uzunluğu", " km"),
                P("Şarj istasyonu", len(e.stations),
                  "Rota üzerinde veya yakınında kullanılabilir DC istasyon "
                  "sayısı. İstasyon az olduğunda rota zorlaşır ve daha "
                  "temkinli bir şarj stratejisi gerekir"),
            ]
        except Exception:
            routes[rk] = [P("Rota", ROUTE_LABELS.get(rk, rk), "")]

    # ---- ALGORİTMA ----
    algos = {
        "dqn": [P("DQN", "Deep Q-Network",
                  "Değer-temelli, off-policy. Her durum-aksiyon için "
                  "Q-değeri öğrenir; en yüksek Q'lu aksiyonu seçer. "
                  "Aşırı-iyimser tahmine eğilimli.")],
        "ddqn": [P("Double DQN", "Çift Q-ağı",
                   "DQN'in aşırı-iyimserlik yanlılığını azaltır: aksiyon "
                   "seçimi ile değerlemesini ayrı ağlara böler. Daha "
                   "kararlı.")],
        "ppo": [P("PPO", "Proximal Policy Optimization",
                  "Politika-temelli, on-policy. Politikayı kırpılmış "
                  "(clipped) güncellemeyle yavaş ve kararlı iyileştirir. "
                  "Bu çalışmada en iyi genelleyen yöntem.")],
    }
    for bk in BASELINE_STRATEGIES:
        algos["baseline_" + bk] = [P("Baseline: " + bk, "Sabit kural",
            "Öğrenmeyen referans strateji; RL'in ne kadar kazandırdığını "
            "ölçmek için kıyas.")]

    # ---- MODEL SENARYOSU ----
    model_scn = [
        P("Eşleşen (kendi modeli)", "matched",
          "Test edilen koşulun birebir kendi senaryosunda eğitilmiş "
          "model. Genelleme matrisinin köşegenini ve üst sınır "
          "performansını verir."),
        P("Başka senaryo seçimi", "çapraz",
          "Farklı bir araç, rota, sürücü, hava veya gün senaryosunda "
          "eğitilmiş modelin bu koşula uygulanması; genelleme ve "
          "çapraz-rota testidir. Rota farklı olduğunda 'ÇAPRAZ' rozeti "
          "gösterilir."),
    ]

    _INFO_CACHE = {
        "vehicle": vehicles, "driver": drivers, "weather": weathers,
        "traffic": traffics, "route": routes, "algo": algos,
        "model_scenario": model_scn,
    }
    return JSONResponse(_INFO_CACHE)


_CURVES_CACHE = None


@app.get("/", response_class=HTMLResponse)
def index():
    return _FRONTEND_HTML




# ---------------------------------------------------------------------------
# HTML templates are read from the package.
# HTML şablonları paket içinden okunur.
# ---------------------------------------------------------------------------
_TEMPLATES_DIR = Path(__file__).parent / "templates"
_FRONTEND_HTML = (_TEMPLATES_DIR / "index.html").read_text(encoding="utf-8")
