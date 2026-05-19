# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Saadettin Yıldırım (Yıldsam Teknoloji)
"""
Canli Yolculuk Gorsellestirme (servis katmani)
==============================================

Google Maps tarzi animasyonlu harita. Cizim/HTML/Folium burada;
saf simulasyon mantigi cekirdekte (`evroute.agents.runner`).

Service layer: drawing/HTML/Folium only. Pure simulation logic
lives in the core (`evroute.agents.runner`).
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

from evroute import make_env
from evroute.env.spaces import SPEEDS, CHARGE_TARGETS
from evroute.env.charging_env import EVChargingEnv
from evroute.models.elevation import ROUTES
from evroute.agents.runner import (
    run_simulation, _load_rl_agent,
    _DQNWrapper, _PPOWrapper, _resolve_model_file,
)

def get_route_coordinates(route_key: str) -> List[List[float]]:
    """
    Rota koordinatlarini (lat, lng, km) listesi olarak dondurur.
    Oncelik: Google Directions gercek yol geometrisi -> elevation koordinatlari -> waypoint interpolasyonu
    """
    # 1. Google Directions API'den gercek yol geometrisi
    geom_file = Path(__file__).parent.parent / "data" / "raw" / f"{route_key}_route_geometry.json"
    if geom_file.exists():
        with open(geom_file) as f:
            data = json.load(f)
        raw_coords = data["coordinates"]  # [[lat, lng], ...]

        # Kumulatif mesafe hesapla (haversine)
        coords_with_km = []
        total_km = 0
        for i, (lat, lng) in enumerate(raw_coords):
            if i > 0:
                dlat = np.radians(lat - raw_coords[i-1][0])
                dlng = np.radians(lng - raw_coords[i-1][1])
                a = np.sin(dlat/2)**2 + np.cos(np.radians(raw_coords[i-1][0])) * np.cos(np.radians(lat)) * np.sin(dlng/2)**2
                total_km += 6371 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
            coords_with_km.append([lat, lng, total_km])

        # Scale yapmiyoruz - gercek mesafeyi kullan

        return coords_with_km

    # 2. Google Elevation verisinden
    elev_file = Path(__file__).parent.parent / "data" / "raw" / "elevation" / f"{route_key}_elevation_google.json"
    if elev_file.exists():
        with open(elev_file) as f:
            data = json.load(f)
        return list(zip(data["latitudes"], data["longitudes"], data["distances_km"]))

    # 3. Fallback: waypoint interpolasyonu
    from evroute.models.elevation import ROUTES as ELEV_ROUTES
    waypoints = ELEV_ROUTES.get(route_key, [])
    coords = []
    for i in range(len(waypoints) - 1):
        wp1, wp2 = waypoints[i], waypoints[i + 1]
        n = max(int((wp2["road_km"] - wp1["road_km"]) / 5), 2)
        for j in range(n):
            f = j / n
            coords.append([
                wp1["lat"] + f * (wp2["lat"] - wp1["lat"]),
                wp1["lng"] + f * (wp2["lng"] - wp1["lng"]),
                wp1["road_km"] + f * (wp2["road_km"] - wp1["road_km"]),
            ])
    coords.append([waypoints[-1]["lat"], waypoints[-1]["lng"], waypoints[-1]["road_km"]])
    return coords


def create_visualization_html(trajectory: List[Dict],
                              route_key: str,
                              stations: List[Dict],
                              title: str = "EV Yolculuk Simulasyonu",
                              output_path: str = "trip_visualization.html"):
    """Animasyonlu HTML harita olusturur."""

    coords = get_route_coordinates(route_key)

    # Koordinatlari km bazinda eslestir
    def coord_at_km(km):
        for i in range(len(coords) - 1):
            if coords[i][2] <= km <= coords[i + 1][2]:
                f = (km - coords[i][2]) / max(coords[i + 1][2] - coords[i][2], 0.01)
                lat = coords[i][0] + f * (coords[i + 1][0] - coords[i][0])
                lng = coords[i][1] + f * (coords[i + 1][1] - coords[i][1])
                return [lat, lng]
        return [coords[-1][0], coords[-1][1]]

    # Trajectory'ye koordinat ekle
    traj_with_coords = []
    for point in trajectory:
        c = coord_at_km(point["km"])
        traj_with_coords.append({**point, "lat": c[0], "lng": c[1]})

    # Rota cizgisi koordinatlari
    route_line = [[c[0], c[1]] for c in coords]

    # Istasyon koordinatlari
    station_coords = []
    for s in stations:
        c = coord_at_km(s["road_km"])
        station_coords.append({
            "name": s["name"],
            "lat": c[0], "lng": c[1],
            "power_kw": s["power_kw"],
            "road_km": s["road_km"],
        })

    # Harita merkezi
    center_lat = np.mean([c[0] for c in coords])
    center_lng = np.mean([c[1] for c in coords])

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #1a1a2e; }}
        #map {{ width: 100%; height: 100vh; }}

        .control-panel {{
            position: absolute; top: 15px; right: 15px; z-index: 1000;
            background: rgba(20, 20, 40, 0.95); border-radius: 16px;
            padding: 20px; width: 320px; color: #fff;
            box-shadow: 0 8px 32px rgba(0,0,0,0.4);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
        }}

        .panel-title {{
            font-size: 14px; color: #8b8fa3; text-transform: uppercase;
            letter-spacing: 1px; margin-bottom: 12px;
        }}

        .stat-row {{
            display: flex; justify-content: space-between;
            padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05);
        }}
        .stat-label {{ color: #8b8fa3; font-size: 13px; }}
        .stat-value {{ font-size: 15px; font-weight: 600; }}

        .soc-bar-container {{
            width: 100%; height: 24px; background: #2d2d4a;
            border-radius: 12px; margin: 10px 0; overflow: hidden;
            position: relative;
        }}
        .soc-bar {{
            height: 100%; border-radius: 12px;
            transition: width 0.3s ease, background 0.3s ease;
        }}
        .soc-text {{
            position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
            font-size: 12px; font-weight: 700; color: #fff; text-shadow: 0 1px 2px rgba(0,0,0,0.5);
        }}

        .speed-controls {{
            display: flex; gap: 6px; margin-top: 12px;
        }}
        .speed-btn {{
            flex: 1; padding: 10px 0; border: none; border-radius: 10px;
            font-size: 13px; font-weight: 700; cursor: pointer;
            transition: all 0.2s ease; color: #fff;
            background: #2d2d4a;
        }}
        .speed-btn:hover {{ background: #3d3d5a; transform: translateY(-1px); }}
        .speed-btn.active {{ background: #4361ee; box-shadow: 0 4px 15px rgba(67,97,238,0.4); }}

        .play-btn {{
            width: 100%; padding: 12px; border: none; border-radius: 12px;
            font-size: 15px; font-weight: 700; cursor: pointer;
            margin-top: 10px; transition: all 0.2s ease; color: #fff;
        }}
        .play-btn.play {{ background: #4361ee; }}
        .play-btn.play:hover {{ background: #3a56d4; }}
        .play-btn.pause {{ background: #e74c3c; }}

        .progress-bar {{
            width: 100%; height: 6px; background: #2d2d4a;
            border-radius: 3px; margin-top: 12px; cursor: pointer;
        }}
        .progress-fill {{
            height: 100%; background: #4361ee; border-radius: 3px;
            transition: width 0.1s linear;
        }}

        .event-popup {{
            position: absolute; bottom: 30px; left: 50%; transform: translateX(-50%);
            z-index: 1000; padding: 12px 24px; border-radius: 12px;
            font-size: 14px; font-weight: 600; color: #fff;
            opacity: 0; transition: opacity 0.3s ease;
            pointer-events: none;
        }}
        .event-popup.show {{ opacity: 1; }}
        .event-popup.charging {{ background: rgba(46, 204, 113, 0.9); }}
        .event-popup.warning {{ background: rgba(231, 76, 60, 0.9); }}

        .altitude-chart {{
            position: absolute; bottom: 15px; left: 15px; z-index: 1000;
            background: rgba(20, 20, 40, 0.9); border-radius: 12px;
            padding: 12px; width: 400px; height: 120px;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .altitude-chart canvas {{ width: 100%; height: 100%; }}
    </style>
</head>
<body>
    <div id="map"></div>

    <div class="control-panel">
        <div class="panel-title">EV Yolculuk Simulasyonu</div>

        <div class="stat-row">
            <span class="stat-label">Batarya (SoC)</span>
            <span class="stat-value" id="soc-value">80%</span>
        </div>
        <div class="soc-bar-container">
            <div class="soc-bar" id="soc-bar" style="width:80%; background: #2ecc71;"></div>
            <span class="soc-text" id="soc-bar-text">80%</span>
        </div>

        <div class="stat-row">
            <span class="stat-label">Hiz</span>
            <span class="stat-value" id="speed-value">0 km/h</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Mesafe</span>
            <span class="stat-value" id="distance-value">0 / {int(trajectory[-1]['km'])} km</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Sure</span>
            <span class="stat-value" id="time-value">0 dk</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Saat</span>
            <span class="stat-value" id="hour-value">08:00</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Yukseklik</span>
            <span class="stat-value" id="altitude-value">0 m</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Maliyet</span>
            <span class="stat-value" id="cost-value" style="color: #f39c12;">0 TL</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Egim</span>
            <span class="stat-value" id="grade-value">%0</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Tuketim</span>
            <span class="stat-value" id="consumption-value">0 kWh/100km</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Trafik</span>
            <span class="stat-value" id="traffic-value">Akici</span>
        </div>

        <div class="speed-controls">
            <button class="speed-btn" onclick="setSpeed(1)">1x</button>
            <button class="speed-btn" onclick="setSpeed(10)">10x</button>
            <button class="speed-btn active" onclick="setSpeed(40)">40x</button>
            <button class="speed-btn" onclick="setSpeed(80)">80x</button>
        </div>

        <button class="play-btn play" id="play-btn" onclick="togglePlay()">
            &#9654; Baslat
        </button>

        <div class="progress-bar" id="progress-bar" onclick="seekTo(event)">
            <div class="progress-fill" id="progress-fill" style="width: 0%;"></div>
        </div>
    </div>

    <div class="event-popup" id="event-popup"></div>

    <div class="altitude-chart">
        <canvas id="altitude-canvas"></canvas>
    </div>

    <script>
        // Veri
        const trajectory = {json.dumps(traj_with_coords)};
        const routeLine = {json.dumps(route_line)};
        const stations = {json.dumps(station_coords)};
        const totalKm = {trajectory[-1]['km']};

        // Harita
        const map = L.map('map', {{
            center: [{center_lat}, {center_lng}],
            zoom: 7,
            zoomControl: false,
        }});

        L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
            attribution: '&copy; OpenStreetMap &copy; CARTO',
            maxZoom: 19,
        }}).addTo(map);

        // Rota cizgisi
        L.polyline(routeLine, {{
            color: '#4361ee', weight: 4, opacity: 0.6,
        }}).addTo(map);

        // Gecilmis rota (canli guncellenir)
        const traveledLine = L.polyline([], {{
            color: '#2ecc71', weight: 5, opacity: 0.9,
        }}).addTo(map);

        // Istasyon ikonlari
        stations.forEach(s => {{
            const icon = L.divIcon({{
                html: `<div style="
                    background: #f39c12; width: 14px; height: 14px;
                    border-radius: 50%; border: 2px solid #fff;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.4);
                "></div>`,
                iconSize: [14, 14],
                iconAnchor: [7, 7],
            }});
            L.marker([s.lat, s.lng], {{ icon }})
                .addTo(map)
                .bindTooltip(`${{s.name}}<br>${{s.power_kw}} kW`, {{
                    className: 'station-tooltip',
                    direction: 'top',
                }});
        }});

        // Arac ikonu (ok seklinde)
        const carIcon = L.divIcon({{
            html: `<div id="car-arrow" style="
                width: 0; height: 0;
                border-left: 10px solid transparent;
                border-right: 10px solid transparent;
                border-bottom: 24px solid #4361ee;
                filter: drop-shadow(0 2px 4px rgba(0,0,0,0.5));
                transform-origin: center center;
                transition: transform 0.1s ease;
            "></div>`,
            iconSize: [20, 24],
            iconAnchor: [10, 12],
        }});

        const carMarker = L.marker([routeLine[0][0], routeLine[0][1]], {{
            icon: carIcon, zIndexOffset: 1000,
        }}).addTo(map);

        // Animasyon state
        let currentFrame = 0;
        let isPlaying = false;
        let speedMultiplier = 40;
        let animationId = null;
        let lastTimestamp = 0;

        function setSpeed(speed) {{
            speedMultiplier = speed;
            document.querySelectorAll('.speed-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
        }}

        function togglePlay() {{
            isPlaying = !isPlaying;
            const btn = document.getElementById('play-btn');
            if (isPlaying) {{
                btn.innerHTML = '&#9646;&#9646; Duraklat';
                btn.className = 'play-btn pause';
                lastTimestamp = performance.now();
                animate();
            }} else {{
                btn.innerHTML = '&#9654; Devam Et';
                btn.className = 'play-btn play';
                if (animationId) cancelAnimationFrame(animationId);
            }}
        }}

        function seekTo(e) {{
            const bar = document.getElementById('progress-bar');
            const rect = bar.getBoundingClientRect();
            const pct = (e.clientX - rect.left) / rect.width;
            currentFrame = Math.floor(pct * (trajectory.length - 1));
            updateDisplay(trajectory[currentFrame]);
        }}

        function getSocColor(soc) {{
            if (soc > 0.5) return '#2ecc71';
            if (soc > 0.25) return '#f39c12';
            return '#e74c3c';
        }}

        function updateDisplay(point) {{
            // SoC
            const socPct = Math.round(point.soc * 100);
            document.getElementById('soc-value').textContent = socPct + '%';
            document.getElementById('soc-bar').style.width = socPct + '%';
            document.getElementById('soc-bar').style.background = getSocColor(point.soc);
            document.getElementById('soc-bar-text').textContent = socPct + '%';
            document.getElementById('soc-value').style.color = getSocColor(point.soc);

            // Diger istatistikler
            document.getElementById('speed-value').textContent =
                point.event === 'charging' ? 'Sarj ediliyor...' : Math.round(point.speed) + ' km/h';
            document.getElementById('speed-value').style.color =
                point.event === 'charging' ? '#2ecc71' : '#fff';
            document.getElementById('distance-value').textContent =
                Math.round(point.km) + ' / ' + Math.round(totalKm) + ' km';
            document.getElementById('time-value').textContent =
                Math.round(point.total_time_min) + ' dk';

            const hour = Math.floor(point.hour % 24);
            const min = Math.floor((point.hour % 1) * 60);
            document.getElementById('hour-value').textContent =
                String(hour).padStart(2, '0') + ':' + String(min).padStart(2, '0');
            document.getElementById('altitude-value').textContent =
                Math.round(point.altitude) + ' m';
            document.getElementById('cost-value').textContent =
                Math.round(point.cost_tl) + ' TL';

            // Egim
            const grade = point.grade || 0;
            const gradeEl = document.getElementById('grade-value');
            gradeEl.textContent = (grade >= 0 ? '+' : '') + grade.toFixed(1) + '%';
            gradeEl.style.color = grade > 3 ? '#e74c3c' : grade < -3 ? '#2ecc71' : '#fff';

            // Tuketim
            const cons = point.consumption_kwh100 || 0;
            const consEl = document.getElementById('consumption-value');
            consEl.textContent = cons.toFixed(1) + ' kWh/100km';
            consEl.style.color = cons > 25 ? '#e74c3c' : cons > 18 ? '#f39c12' : '#2ecc71';

            // Trafik durumu
            const tf = point.traffic_factor || 1;
            const trafficEl = document.getElementById('traffic-value');
            if (tf > 0.85) {{ trafficEl.textContent = 'Akici'; trafficEl.style.color = '#2ecc71'; }}
            else if (tf > 0.65) {{ trafficEl.textContent = 'Normal'; trafficEl.style.color = '#f39c12'; }}
            else if (tf > 0.45) {{ trafficEl.textContent = 'Yogun'; trafficEl.style.color = '#e67e22'; }}
            else {{ trafficEl.textContent = 'Cok Yogun'; trafficEl.style.color = '#e74c3c'; }}

            // Progress bar
            const progress = currentFrame / (trajectory.length - 1) * 100;
            document.getElementById('progress-fill').style.width = progress + '%';

            // Arac pozisyonu
            carMarker.setLatLng([point.lat, point.lng]);

            // Arac yonu (ok dondurmesi)
            if (currentFrame > 0) {{
                const prev = trajectory[currentFrame - 1];
                const angle = Math.atan2(point.lng - prev.lng, point.lat - prev.lat) * 180 / Math.PI;
                const arrow = document.getElementById('car-arrow');
                if (arrow) arrow.style.transform = `rotate(${{-angle + 180}}deg)`;
            }}

            // Arac rengi (sarj/surus)
            const arrow = document.getElementById('car-arrow');
            if (arrow) {{
                arrow.style.borderBottomColor = point.event === 'charging' ? '#2ecc71' : '#4361ee';
            }}

            // Gecilmis rota
            const traveledCoords = trajectory.slice(0, currentFrame + 1).map(p => [p.lat, p.lng]);
            traveledLine.setLatLngs(traveledCoords);

            // Harita takip
            map.panTo([point.lat, point.lng], {{ animate: true, duration: 0.3 }});

            // Sarj event popup
            const popup = document.getElementById('event-popup');
            if (point.event === 'charging' && point.charge_time_min > 0) {{
                popup.textContent = `⚡ ${{point.station}} - ${{Math.round(point.charge_time_min)}} dk sarj`;
                popup.className = 'event-popup charging show';
            }} else if (point.soc < 0.15) {{
                popup.textContent = `⚠️ Dusuk batarya! %${{Math.round(point.soc * 100)}}`;
                popup.className = 'event-popup warning show';
            }} else {{
                popup.className = 'event-popup';
            }}

            // Yukseklik grafigi guncelle
            drawAltitudeChart(point.km);
        }}

        // Yukseklik grafigi
        function drawAltitudeChart(currentKm) {{
            const canvas = document.getElementById('altitude-canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = canvas.parentElement.clientWidth - 24;
            canvas.height = canvas.parentElement.clientHeight - 24;

            const w = canvas.width, h = canvas.height;
            ctx.clearRect(0, 0, w, h);

            // Tum yukseklikleri ciz
            const alts = trajectory.map(p => p.altitude);
            const kms = trajectory.map(p => p.km);
            const maxAlt = Math.max(...alts) * 1.1;
            const minAlt = Math.min(...alts.filter(a => a > 0)) * 0.8;

            // Arkaplan gradient
            const grad = ctx.createLinearGradient(0, 0, 0, h);
            grad.addColorStop(0, 'rgba(67, 97, 238, 0.3)');
            grad.addColorStop(1, 'rgba(67, 97, 238, 0.02)');

            ctx.beginPath();
            ctx.moveTo(0, h);
            for (let i = 0; i < trajectory.length; i++) {{
                const x = (kms[i] / totalKm) * w;
                const y = h - ((alts[i] - minAlt) / (maxAlt - minAlt)) * h;
                ctx.lineTo(x, y);
            }}
            ctx.lineTo(w, h);
            ctx.closePath();
            ctx.fillStyle = grad;
            ctx.fill();

            // Cizgi
            ctx.beginPath();
            for (let i = 0; i < trajectory.length; i++) {{
                const x = (kms[i] / totalKm) * w;
                const y = h - ((alts[i] - minAlt) / (maxAlt - minAlt)) * h;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }}
            ctx.strokeStyle = '#4361ee';
            ctx.lineWidth = 2;
            ctx.stroke();

            // Mevcut pozisyon
            const cx = (currentKm / totalKm) * w;
            const currentAlt = trajectory.find(p => p.km >= currentKm)?.altitude || 0;
            const cy = h - ((currentAlt - minAlt) / (maxAlt - minAlt)) * h;

            ctx.beginPath();
            ctx.arc(cx, cy, 5, 0, Math.PI * 2);
            ctx.fillStyle = '#2ecc71';
            ctx.fill();
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 2;
            ctx.stroke();

            // Etiketler
            ctx.fillStyle = '#8b8fa3';
            ctx.font = '10px sans-serif';
            ctx.fillText(Math.round(maxAlt) + 'm', 2, 12);
            ctx.fillText(Math.round(minAlt) + 'm', 2, h - 2);
            ctx.fillText('0 km', 2, h - 12);
            ctx.fillText(Math.round(totalKm) + ' km', w - 40, h - 12);
        }}

        function animate(timestamp) {{
            if (!isPlaying) return;

            if (!lastTimestamp) lastTimestamp = timestamp;
            const delta = timestamp - lastTimestamp;

            // Her frame'de speedMultiplier kadar ilerleme
            if (delta > 50) {{
                lastTimestamp = timestamp;
                currentFrame += Math.max(1, Math.floor(speedMultiplier / 10));

                if (currentFrame >= trajectory.length) {{
                    currentFrame = trajectory.length - 1;
                    isPlaying = false;
                    document.getElementById('play-btn').innerHTML = '&#9654; Tekrar';
                    document.getElementById('play-btn').className = 'play-btn play';
                }}

                updateDisplay(trajectory[currentFrame]);
            }}

            if (isPlaying) animationId = requestAnimationFrame(animate);
        }}

        // Baslangic
        updateDisplay(trajectory[0]);
        drawAltitudeChart(0);

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {{
            if (e.code === 'Space') {{ e.preventDefault(); togglePlay(); }}
            if (e.key === '1') setSpeed(1);
            if (e.key === '2') setSpeed(10);
            if (e.key === '3') setSpeed(40);
            if (e.key === '4') setSpeed(80);
        }});
    </script>
</body>
</html>"""

    output = Path(output_path)
    with open(output, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Gorselllestirme kaydedildi: {output.absolute()}")
    return str(output.absolute())


ROUTE_LABELS = {
    "istanbul_ankara": "İstanbul – Ankara",
    "istanbul_izmir":  "İstanbul – İzmir",
    "ankara_antalya":  "Ankara – Antalya",
}

DRIVER_DISPLAY_NAMES = {
    "eco":        "Eko Sürücü",
    "normal":     "Normal Sürücü",
    "aggressive": "Agresif Sürücü",
}

# Strateji etiketleri (panelde gösterilir)
STRATEGY_LABELS = {
    "always_70":      "Her İstasyonda %90 (120 km/h)",
    "minimum_charge": "Minimum Şarj (120 km/h)",
    "fast_drive":     "Hızlı Sürüş (140 km/h)",
    "eco":            "Eko Sürüş (80 km/h)",
    "dqn":            "DQN (Pekiştirmeli Öğrenme)",
    "double_dqn":     "Double DQN (Pekiştirmeli Öğrenme)",
    "ppo":            "PPO (Pekiştirmeli Öğrenme)",
}

STRATEGY_ORDER = ["always_70", "minimum_charge", "fast_drive", "eco",
                  "dqn", "double_dqn", "ppo"]


DRIVER_DESCRIPTIONS = {
    "eco":        "Yavaş ama verimli sürer; maliyet odaklı, sabırlı; batarya sağlığına dikkat eder.",
    "normal":     "Ortalama hız, makul sabır, dengeli tercihler.",
    "aggressive": "Hızlı sürer; sabırsız; süreyi minimize etmek ister.",
}


def _build_driver_info() -> Dict:
    """driver_profile.py'den 3 sürücü için tam parametre dökümü."""
    from evroute.models.driver import DRIVER_PROFILES
    out = {}
    for key, p in DRIVER_PROFILES.items():
        out[key] = {
            "name": DRIVER_DISPLAY_NAMES.get(key, p.name),
            "description": DRIVER_DESCRIPTIONS.get(key, p.description),
            "params": {
                "Tercih edilen hız":          f"{p.preferred_speed_kmh} km/h (±{p.speed_variance_kmh})",
                "Hız toleransı":              f"%{p.speed_tolerance*100:.0f} (hız sınırını aşma)",
                "Şarj sabrı eşiği":           f"{p.patience_threshold_min} dk",
                "Sabır azalma oranı":         f"{p.patience_decay_rate}",
                "Min. rahat SoC":             f"%{p.min_comfortable_soc*100:.0f}",
                "Menzil kaygısı ağırlığı":    f"{p.range_anxiety_weight}",
                "Maks. kesintisiz sürüş":     f"{p.max_continuous_drive_min} dk",
                "Tercih edilen mola":         f"{p.preferred_break_min} dk",
                "HVAC (iklimlendirme) modu":  p.hvac_mode,
                "Rejeneratif fren bonusu":    f"{p.regen_efficiency_bonus:+.2f} (0 = standart)",
            },
            "weights": {
                "w_time (süre)":              p.w_time,
                "w_cost (maliyet)":           p.w_cost,
                "w_comfort (konfor)":         p.w_comfort,
                "w_anxiety (menzil kaygısı)": p.w_anxiety,
                "w_degradation (yıpranma)":   p.w_degradation,
            },
            "refs": [
                "Franke & Krems (2013) — menzil konfor bölgesi",
                "Maister (2005), Antonides ve ark. (2002) — bekleme psikolojisi",
                "EU 561/2006 — sürücü dinlenme süresi",
            ],
        }
    return out


STRATEGY_INFO = {
    "always_70": {
        "kind": "baseline",
        "title": "Her İstasyonda %90 (Konservatif Baseline)",
        "desc": "Sabit politika: her şarj istasyonunda batarya %90'a kadar doldurulur, sürüş hızı sabit 120 km/h.",
        "params": {
            "Sürüş hızı":     "120 km/h (sabit)",
            "Şarj kuralı":    "Her istasyonda %90 hedef",
            "Karar mantığı":  "if at_station: charge_to(0.90); speed = 120",
        },
        "notes": "Güvende kalmaya odaklı. Toplam şarj süresi yüksek (CC→CV eğrisi yavaşlar), ancak varış SoC'si genelde %75 ve üzeri.",
    },
    "minimum_charge": {
        "kind": "baseline",
        "title": "Minimum Şarj (Süre Odaklı Baseline)",
        "desc": "Sabit politika: yalnızca SoC %50'nin altına düştüğünde %70'e şarj eder, aksi halde istasyonu atlar.",
        "params": {
            "Sürüş hızı":     "120 km/h (sabit)",
            "Şarj kuralı":    "Eğer SoC < %50 → %70'e şarj; değilse atla",
            "Karar mantığı":  "speed = 120; charge = 0.70 if soc < 0.5 else None",
        },
        "notes": "Hızlıdır ama risklidir. Uzun veya yokuşlu rotalarda batarya bitirebilir (örn. Ankara–Antalya).",
    },
    "fast_drive": {
        "kind": "baseline",
        "title": "Hızlı Sürüş (Agresif Baseline)",
        "desc": "Sabit politika: 140 km/h ile sürer; SoC %50'nin altında ise %90'a şarj eder.",
        "params": {
            "Sürüş hızı":     "140 km/h (sabit)",
            "Şarj kuralı":    "Eğer SoC < %50 → %90'a şarj",
            "Karar mantığı":  "speed = 140; charge = 0.90 if soc < 0.5 else None",
        },
        "notes": "En hızlı baseline'dır ancak tüketim yüksektir; rotaya göre batarya bitebilir.",
    },
    "eco": {
        "kind": "baseline",
        "title": "Eko Sürüş (Verimli Baseline)",
        "desc": "Sabit politika: 80 km/h ile sürer, her istasyonda %70'e şarj eder.",
        "params": {
            "Sürüş hızı":     "80 km/h (sabit)",
            "Şarj kuralı":    "Her istasyonda %70 hedef",
            "Karar mantığı":  "speed = 80; charge = 0.70",
        },
        "notes": "En düşük maliyet, en az batarya stresi. Toplam yolculuk süresi ise en uzunu.",
    },
    "dqn": {
        "kind": "rl",
        "title": "DQN (Deep Q-Network)",
        "desc": "Q-öğrenmenin derin ağ versiyonu (Mnih ve ark., 2015). Her durum–aksiyon çifti için Q değerini öğrenir; en yüksek Q'lu aksiyonu seçer.",
        "params": {
            "Durum boyutu":        "14 (SoC, mesafe, hız, saat, hava vb.)",
            "Aksiyon boyutu":      "25 (5 hız × 5 şarj hedefi)",
            "Gizli katman":        "128 nöron (2 katmanlı MLP)",
            "Optimizer":           "Adam, lr = 5×10⁻⁴",
            "İndirim (γ)":         "0.99",
            "ε-azalma":            "1.0 → 0.01 (her episodda ×0.995)",
            "Replay buffer":       "50.000",
            "Mini-batch":          "64",
            "Hedef ağ güncelleme": "soft, τ = 0.005",
            "Eğitim":              "2000 episode, İstanbul–Ankara rotası, IONIQ 5",
        },
        "notes": "Yalnızca İstanbul–Ankara rotasında eğitildi; diğer rotalarda genelleme sınırlıdır.",
    },
    "double_dqn": {
        "kind": "rl",
        "title": "Double DQN",
        "desc": "DQN'in aşırı tahmin (over-estimation) yanlılığını azaltmak için aksiyon seçimini online ağa, değerlendirmeyi hedef ağa bırakır (van Hasselt ve ark., 2016).",
        "params": {
            "Mimari": "DQN ile aynı (14 → 128 → 128 → 25)",
            "Fark":   "next_action = argmaxₐ Q_online(s′,a);  next_Q = Q_target(s′, next_action)",
            "Eğitim": "DQN ile aynı hiperparametreler",
        },
        "notes": "Standart DQN'e göre daha kararlı; bizim deneyde benzer ya da biraz daha muhafazakâr sonuçlar üretti.",
    },
    "ppo": {
        "kind": "rl",
        "title": "PPO (Proximal Policy Optimization)",
        "desc": "Politika gradyanı yöntemi (Schulman ve ark., 2017). Politika ve değer ağlarını birlikte eğitir; clip ile güvenli güncelleme yapar.",
        "params": {
            "Kütüphane":     "stable-baselines3 (MlpPolicy)",
            "Mimari":        "Actor–Critic, 2 × 64 MLP",
            "İndirim (γ)":   "0.99",
            "Clip aralığı":  "0.2",
            "Entropi kats.": "0.0",
            "Eğitim":        "stable-baselines3 varsayılanı, İstanbul–Ankara, IONIQ 5",
        },
        "notes": "DQN ailesi off-policy iken PPO on-policy'dir. Deneyimizde en kısa yolculuk süresini (3.4 sa) ve en az şarj süresini ürettiği rota mevcuttur.",
    },
}

REWARD_FORMULA = (
    "r_toplam = w_time·r_time + w_cost·r_cost + w_comfort·r_comfort "
    "+ w_anxiety·r_anxiety + w_degradation·r_degradation\n"
    "(SoC ≤ 0 ise yalnızca r_death = −100 uygulanır.) "
    "Ağırlıklar sürücü profilinden gelir; bu nedenle aynı RL ajanı farklı profillerde farklı bir amaç fonksiyonuna optimize edilmiş sayılır."
)


def _build_route_payload(route_key: str):
    """Tek bir rota icin tum senaryo + harita verisini uretir."""
    from evroute.agents.baselines import BASELINE_STRATEGIES

    drivers = ["eco", "normal", "aggressive"]
    baseline_keys = list(BASELINE_STRATEGIES.keys())
    rl_keys = ["dqn", "double_dqn", "ppo"]

    # Tum strateji secenekleri: baseline + RL (sadece yuklenebilenler)
    available_strategies = list(baseline_keys)
    for rk in rl_keys:
        if _load_rl_agent(rk, route_key) is not None:
            available_strategies.append(rk)
        else:
            print(f"  [uyari] {rk} modeli yuklenemedi (rota: {route_key}), atlanıyor.")

    all_data = {}
    total = len(drivers) * len(available_strategies)
    count = 0
    print(f"\n=== Rota: {route_key} ===")

    for driver in drivers:
        for strat_key in available_strategies:
            count += 1
            key = f"{driver}_{strat_key}"

            env = make_env(vehicle="ioniq5", route=route_key, driver=driver,
                           weather="optimal", seed=42)

            if strat_key in BASELINE_STRATEGIES:
                _, strategy_fn = BASELINE_STRATEGIES[strat_key]
                trajectory = run_simulation(env, strategy_fn=strategy_fn)
            else:
                agent = _load_rl_agent(strat_key, route_key)
                trajectory = run_simulation(env, agent=agent)
            summary = env.get_trip_summary()

            all_data[key] = {
                "trajectory": trajectory,
                "summary": summary,
                "stations": env.stations,
            }

            status = "OK" if summary["arrival_soc"] > 0.01 else "OLDU"
            print(f"  [{count}/{total}] {driver}/{strat_key}: "
                  f"{summary['total_time_h']:.1f}h, {summary['charge_time_min']:.0f}dk sarj, "
                  f"SoC={summary['arrival_soc']:.0%} {status}")

    coords = get_route_coordinates(route_key)
    route_line = [[c[0], c[1]] for c in coords]

    def coord_at_km(km):
        for i in range(len(coords) - 1):
            if coords[i][2] <= km <= coords[i + 1][2]:
                f = (km - coords[i][2]) / max(coords[i + 1][2] - coords[i][2], 0.01)
                return [coords[i][0] + f * (coords[i + 1][0] - coords[i][0]),
                        coords[i][1] + f * (coords[i + 1][1] - coords[i][1])]
        return [coords[-1][0], coords[-1][1]]

    scenarios_json = {}
    for key, data in all_data.items():
        traj_with_coords = []
        for point in data["trajectory"]:
            if point.get("lat") is not None and point.get("lng") is not None:
                # Sapma noktası: rota dışı, açık koordinat korunur
                traj_with_coords.append(dict(point))
            else:
                c = coord_at_km(point["km"])
                traj_with_coords.append({**point, "lat": c[0], "lng": c[1]})
        scenarios_json[key] = {
            "trajectory": traj_with_coords,
            "summary": data["summary"],
        }

    station_coords = []
    sample_stations = list(all_data.values())[0]["stations"]
    for s in sample_stations:
        c = coord_at_km(s["road_km"])
        station_coords.append({"name": s["name"], "lat": c[0], "lng": c[1],
                               "power_kw": s.get("power_kw", 120), "road_km": s["road_km"]})

    return {
        "label": ROUTE_LABELS.get(route_key, route_key),
        "scenarios": scenarios_json,
        "route_line": route_line,
        "stations": station_coords,
        "center_lat": float(np.mean([c[0] for c in coords])),
        "center_lng": float(np.mean([c[1] for c in coords])),
        "total_km": float(coords[-1][2]),
    }


def generate_all_scenarios(route=None, routes=None, output="trip_visualization.html"):
    """
    Tum surucu + strateji + rota kombinasyonlarini onceden hesaplar,
    tek interaktif HTML dosyasina gomer.
    """
    if routes is None:
        if route is not None:
            routes = [route]
        else:
            routes = ["istanbul_ankara", "istanbul_izmir", "ankara_antalya"]

    routes_data = {rk: _build_route_payload(rk) for rk in routes}

    default_route = routes[0]
    path = _write_interactive_html(routes_data, default_route, output)
    return path


def _write_interactive_html(routes_data, default_route, output):
    """Interaktif HTML dosyasi yazar (cok rotali)."""
    driver_info = _build_driver_info()
    default = routes_data[default_route]
    scenarios = default["scenarios"]
    route_line = default["route_line"]
    stations = default["stations"]
    center_lat = default["center_lat"]
    center_lng = default["center_lng"]
    total_km = default["total_km"]

    route_options_html = "".join(
        f'<option value="{rk}"{" selected" if rk == default_route else ""}>{rd["label"]}</option>'
        for rk, rd in routes_data.items()
    )

    # Strateji secenekleri JS tarafinda rotaya gore dinamik olusturulur.
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>EV Yolculuk Simülasyonu</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #1a1a2e; }}
        #map {{ width: 100%; height: 100vh; }}

        .control-panel {{
            position: absolute; top: 15px; right: 15px; z-index: 1000;
            background: rgba(20, 20, 40, 0.95); border-radius: 16px;
            padding: 20px; width: 340px; color: #fff;
            box-shadow: 0 8px 32px rgba(0,0,0,0.4);
            border: 1px solid rgba(255,255,255,0.1);
            max-height: 95vh; overflow-y: auto;
        }}
        .panel-title {{ font-size: 16px; font-weight: 700; margin-bottom: 15px; text-align: center; }}

        .selector-row {{ display: flex; gap: 6px; margin-bottom: 8px; }}
        .selector-row label {{ color: #8b8fa3; font-size: 11px; display: block; margin-bottom: 4px; }}
        .selector-row select {{
            width: 100%; padding: 8px; border-radius: 8px; border: 1px solid #3d3d5a;
            background: #2d2d4a; color: #fff; font-size: 13px; cursor: pointer;
        }}

        .stat-row {{ display: flex; justify-content: space-between; padding: 6px 0;
            border-bottom: 1px solid rgba(255,255,255,0.05); }}
        .stat-label {{ color: #8b8fa3; font-size: 12px; }}
        .stat-value {{ font-size: 14px; font-weight: 600; }}

        .soc-bar-container {{ width: 100%; height: 22px; background: #2d2d4a;
            border-radius: 11px; margin: 8px 0; overflow: hidden; position: relative; }}
        .soc-bar {{ height: 100%; border-radius: 11px; transition: width 0.3s, background 0.3s; }}
        .soc-text {{ position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
            font-size: 11px; font-weight: 700; color: #fff; text-shadow: 0 1px 2px rgba(0,0,0,0.5); }}

        .btn-row {{ display: flex; gap: 6px; margin-top: 10px; }}
        .speed-btn {{ flex: 1; padding: 8px 0; border: none; border-radius: 8px;
            font-size: 12px; font-weight: 700; cursor: pointer; color: #fff; background: #2d2d4a; }}
        .speed-btn:hover {{ background: #3d3d5a; }}
        .speed-btn.active {{ background: #4361ee; }}

        .play-btn {{ flex: 2; padding: 10px; border: none; border-radius: 10px;
            font-size: 14px; font-weight: 700; cursor: pointer; color: #fff; }}
        .play-btn.play {{ background: #4361ee; }}
        .play-btn.pause {{ background: #e74c3c; }}
        .reset-btn {{ flex: 1; padding: 10px; border: none; border-radius: 10px;
            font-size: 14px; font-weight: 700; cursor: pointer; color: #fff; background: #e67e22; }}

        .progress-bar {{ width: 100%; height: 6px; background: #2d2d4a;
            border-radius: 3px; margin-top: 10px; cursor: pointer; }}
        .progress-fill {{ height: 100%; background: #4361ee; border-radius: 3px; }}

        .event-popup {{ position: absolute; bottom: 30px; left: 50%; transform: translateX(-50%);
            z-index: 1000; padding: 12px 24px; border-radius: 12px;
            font-size: 14px; font-weight: 600; color: #fff;
            opacity: 0; transition: opacity 0.3s; pointer-events: none; }}
        .event-popup.show {{ opacity: 1; }}
        .event-popup.charging {{ background: rgba(46, 204, 113, 0.9); }}
        .event-popup.warning {{ background: rgba(231, 76, 60, 0.9); }}
        .event-popup.dead {{ background: rgba(231, 76, 60, 0.95); }}

        .altitude-chart {{ position: absolute; bottom: 15px; left: 15px; z-index: 1000;
            background: rgba(20, 20, 40, 0.9); border-radius: 12px;
            padding: 12px; width: 400px; height: 120px;
            border: 1px solid rgba(255,255,255,0.1); }}

        .summary-box {{
            background: linear-gradient(180deg, #232342 0%, #1c1c34 100%);
            border-radius: 14px; padding: 14px; margin-top: 12px;
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 4px 18px rgba(0,0,0,0.35);
            display: none;
        }}
        .summary-box.show {{ display: block; }}
        .summary-header {{
            display: flex; align-items: center; justify-content: space-between;
            margin-bottom: 10px;
        }}
        .summary-title {{
            font-size: 12px; font-weight: 700; letter-spacing: 0.5px;
            text-transform: uppercase; color: #8b8fa3;
        }}
        .status-pill {{
            padding: 3px 10px; border-radius: 999px; font-size: 11px;
            font-weight: 700; letter-spacing: 0.3px;
        }}
        .status-pill.ok   {{ background: rgba(46, 204, 113, 0.18); color: #2ecc71; border: 1px solid rgba(46,204,113,0.4); }}
        .status-pill.fail {{ background: rgba(231, 76, 60, 0.18);  color: #ff6b5b; border: 1px solid rgba(231,76,60,0.45); }}

        .metric-grid {{
            display: grid; grid-template-columns: 1fr 1fr; gap: 8px;
        }}
        .metric-card {{
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.05);
            border-radius: 10px; padding: 8px 10px;
        }}
        .metric-card .m-label {{
            font-size: 10px; color: #8b8fa3; text-transform: uppercase;
            letter-spacing: 0.5px; margin-bottom: 2px;
        }}
        .metric-card .m-value {{
            font-size: 16px; font-weight: 700; color: #fff;
            font-variant-numeric: tabular-nums;
        }}
        .metric-card .m-sub {{
            font-size: 10px; color: #8b8fa3; margin-top: 1px;
        }}
        .summary-footer {{
            margin-top: 10px; padding-top: 8px;
            border-top: 1px dashed rgba(255,255,255,0.08);
            font-size: 11px; color: #c8cad8; line-height: 1.45;
        }}
        .delta-pos {{ color: #2ecc71; font-weight: 700; }}
        .delta-neg {{ color: #ff6b5b; font-weight: 700; }}

        .untrained-note {{
            display: none; margin: 2px 0 4px;
            padding: 7px 9px; border-radius: 8px;
            background: rgba(243, 156, 18, 0.10);
            border: 1px solid rgba(243, 156, 18, 0.35);
            color: #f5c97a; font-size: 11px; line-height: 1.4;
        }}
        .untrained-note.show {{ display: block; }}

        .info-btn {{
            flex: 1; padding: 8px 6px; border: 1px solid #3d3d5a; border-radius: 8px;
            background: #2d2d4a; color: #cfd2e0; font-size: 11px; font-weight: 600;
            cursor: pointer; transition: background 0.2s, border-color 0.2s;
        }}
        .info-btn:hover {{ background: #3d3d5a; border-color: #4361ee; color: #fff; }}

        .info-modal {{
            display: none; position: fixed; inset: 0; z-index: 5000;
            background: rgba(10, 10, 25, 0.7); backdrop-filter: blur(4px);
            align-items: center; justify-content: center;
        }}
        .info-modal.show {{ display: flex; }}
        .info-card {{
            background: #1a1a2e; color: #e6e7ee; border-radius: 16px;
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: 0 16px 48px rgba(0,0,0,0.6);
            width: min(720px, 92vw); max-height: 88vh; overflow-y: auto;
            padding: 26px 28px; position: relative; line-height: 1.5;
        }}
        .info-card h2 {{ font-size: 19px; margin-bottom: 6px; color: #fff; }}
        .info-card .badge {{
            display: inline-block; padding: 2px 10px; border-radius: 999px;
            font-size: 11px; font-weight: 700; margin-bottom: 12px;
            letter-spacing: 0.4px; text-transform: uppercase;
        }}
        .info-card .badge.baseline {{ background: #e67e22; color: #fff; }}
        .info-card .badge.rl       {{ background: #4361ee; color: #fff; }}
        .info-card .badge.driver   {{ background: #2ecc71; color: #fff; }}
        .info-card h3 {{ font-size: 14px; margin: 18px 0 6px; color: #8ab4ff; }}
        .info-card p  {{ font-size: 13px; color: #c8cad8; margin-bottom: 8px; }}
        .info-card table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        .info-card table td {{ padding: 5px 8px; border-bottom: 1px solid rgba(255,255,255,0.06); }}
        .info-card table td:first-child {{ color: #8b8fa3; width: 46%; }}
        .info-card table td:last-child  {{ color: #fff; font-family: 'Consolas', monospace; }}
        .info-card ul {{ font-size: 12px; color: #8b8fa3; padding-left: 18px; margin-top: 4px; }}
        .info-card pre {{
            background: #0f0f1f; border: 1px solid rgba(255,255,255,0.06);
            border-radius: 8px; padding: 10px 12px; font-size: 12px;
            white-space: pre-wrap; word-break: break-word; color: #e6e7ee;
        }}
        .info-close {{
            position: absolute; top: 10px; right: 14px;
            background: none; border: none; color: #8b8fa3;
            font-size: 24px; cursor: pointer; line-height: 1;
        }}
        .info-close:hover {{ color: #fff; }}
    </style>
</head>
<body>
    <div id="map"></div>

    <div class="control-panel">
        <div class="panel-title">EV Yolculuk Simülasyonu</div>

        <div class="selector-row">
            <div style="flex:1">
                <label>Rota</label>
                <select id="sel-route" onchange="changeRoute()">
                    {route_options_html}
                </select>
            </div>
        </div>

        <div class="selector-row">
            <div style="flex:1">
                <label>Sürücü Profili</label>
                <select id="sel-driver" onchange="changeScenario()">
                    <option value="eco">Eko Sürücü</option>
                    <option value="normal" selected>Normal Sürücü</option>
                    <option value="aggressive">Agresif Sürücü</option>
                </select>
            </div>
            <div style="flex:1">
                <label>Strateji / Algoritma</label>
                <select id="sel-strategy" onchange="changeScenario()"></select>
            </div>
        </div>

        <div id="untrained-note" class="untrained-note"></div>

        <div class="btn-row" style="margin-top:6px">
            <button class="info-btn" onclick="showInfo('driver')">ℹ Sürücü Profili Detayı</button>
            <button class="info-btn" onclick="showInfo('strategy')">ℹ Strateji / Algoritma Detayı</button>
        </div>

        <div class="stat-row"><span class="stat-label">Batarya</span><span class="stat-value" id="soc-value">80%</span></div>
        <div class="soc-bar-container">
            <div class="soc-bar" id="soc-bar" style="width:80%; background:#2ecc71;"></div>
            <span class="soc-text" id="soc-bar-text">80%</span>
        </div>

        <div class="stat-row"><span class="stat-label">Hız</span><span class="stat-value" id="speed-value">0 km/h</span></div>
        <div class="stat-row"><span class="stat-label">Mesafe</span><span class="stat-value" id="distance-value">0 km</span></div>
        <div class="stat-row"><span class="stat-label">Süre</span><span class="stat-value" id="time-value">0 dk</span></div>
        <div class="stat-row"><span class="stat-label">Saat</span><span class="stat-value" id="hour-value">08:00</span></div>
        <div class="stat-row"><span class="stat-label">Yükseklik</span><span class="stat-value" id="altitude-value">0 m</span></div>
        <div class="stat-row"><span class="stat-label">Maliyet</span><span class="stat-value" id="cost-value" style="color:#f39c12">0 TL</span></div>
        <div class="stat-row"><span class="stat-label">Eğim</span><span class="stat-value" id="grade-value">%0</span></div>
        <div class="stat-row"><span class="stat-label">Tüketim</span><span class="stat-value" id="consumption-value">0 kWh/100km</span></div>
        <div class="stat-row"><span class="stat-label">Trafik</span><span class="stat-value" id="traffic-value">Akıcı</span></div>

        <div class="btn-row">
            <button class="speed-btn" onclick="setSpeed(1)">1x</button>
            <button class="speed-btn" onclick="setSpeed(10)">10x</button>
            <button class="speed-btn active" onclick="setSpeed(40)">40x</button>
            <button class="speed-btn" onclick="setSpeed(80)">80x</button>
        </div>

        <div class="btn-row">
            <button class="play-btn play" id="play-btn" onclick="togglePlay()">▶ Başlat</button>
            <button class="reset-btn" onclick="resetTrip()">↻ Sıfırla</button>
        </div>

        <div class="progress-bar" id="progress-bar" onclick="seekTo(event)">
            <div class="progress-fill" id="progress-fill" style="width:0%"></div>
        </div>

        <div class="summary-box" id="summary-box">
            <div class="summary-header">
                <span class="summary-title">Yolculuk Özeti</span>
                <span class="status-pill" id="summary-status">—</span>
            </div>
            <div class="metric-grid" id="summary-grid"></div>
            <div class="summary-footer" id="summary-footer"></div>
        </div>
    </div>

    <!-- Bilgi modali -->
    <div class="info-modal" id="info-modal" onclick="if(event.target===this)hideInfo()">
        <div class="info-card">
            <button class="info-close" onclick="hideInfo()">×</button>
            <div id="info-content"></div>
        </div>
    </div>

    <div class="event-popup" id="event-popup"></div>
    <div class="altitude-chart"><canvas id="altitude-canvas"></canvas></div>

    <script>
        const allRoutes = {json.dumps(routes_data, ensure_ascii=False)};
        const DRIVER_INFO = {json.dumps(driver_info, ensure_ascii=False)};
        const STRATEGY_INFO = {json.dumps(STRATEGY_INFO, ensure_ascii=False)};
        const STRATEGY_LABELS = {json.dumps(STRATEGY_LABELS, ensure_ascii=False)};
        const STRATEGY_ORDER = {json.dumps(STRATEGY_ORDER, ensure_ascii=False)};
        const REWARD_FORMULA = {json.dumps(REWARD_FORMULA, ensure_ascii=False)};
        let currentRouteKey = {json.dumps(default_route)};
        let allScenarios = allRoutes[currentRouteKey].scenarios;
        let routeLine = allRoutes[currentRouteKey].route_line;
        let stations = allRoutes[currentRouteKey].stations;
        let totalKm = allRoutes[currentRouteKey].total_km;

        let trajectory = [];
        let currentFrame = 0;
        let isPlaying = false;
        let speedMultiplier = 40;
        let animationId = null;
        let lastTimestamp = 0;

        // Harita
        const map = L.map('map', {{ center: [{center_lat}, {center_lng}], zoom: 7, zoomControl: false }});
        L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
            attribution: '&copy; OpenStreetMap', maxZoom: 19 }}).addTo(map);

        let routePath = L.polyline(routeLine, {{ color: '#4361ee', weight: 4, opacity: 0.5 }}).addTo(map);
        let stationMarkers = [];

        function drawStations() {{
            stationMarkers.forEach(m => map.removeLayer(m));
            stationMarkers = [];
            stations.forEach(s => {{
                const m = L.marker([s.lat, s.lng], {{
                    icon: L.divIcon({{
                        html: '<div style="background:#f39c12;width:12px;height:12px;border-radius:50%;border:2px solid #fff;box-shadow:0 2px 6px rgba(0,0,0,0.4)"></div>',
                        iconSize: [12, 12], iconAnchor: [6, 6] }})
                }}).addTo(map).bindTooltip(s.name + '<br>' + s.power_kw + ' kW', {{ direction: 'top' }});
                stationMarkers.push(m);
            }});
        }}
        drawStations();

        // Arac
        const carIcon = L.divIcon({{
            html: '<div id="car-arrow" style="width:0;height:0;border-left:10px solid transparent;border-right:10px solid transparent;border-bottom:24px solid #4361ee;filter:drop-shadow(0 2px 4px rgba(0,0,0,0.5));transition:transform 0.1s"></div>',
            iconSize: [20, 24], iconAnchor: [10, 12] }});
        const carMarker = L.marker([routeLine[0][0], routeLine[0][1]], {{ icon: carIcon, zIndexOffset: 1000 }}).addTo(map);

        function escapeHtml(s) {{
            return String(s).replace(/[&<>"']/g, c => ({{
                '&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'
            }})[c]);
        }}

        function renderRow(k, v) {{
            return '<tr><td>' + escapeHtml(k) + '</td><td>' + escapeHtml(v) + '</td></tr>';
        }}

        function buildDriverInfoHtml() {{
            const dk = document.getElementById('sel-driver').value;
            const d = DRIVER_INFO[dk];
            if (!d) return '<p>Bilgi bulunamadı.</p>';
            let html = '';
            html += '<span class="badge driver">Sürücü Profili</span>';
            html += '<h2>' + escapeHtml(d.name) + '</h2>';
            html += '<p>' + escapeHtml(d.description) + '</p>';

            html += '<h3>Davranış parametreleri</h3><table>';
            for (const [k, v] of Object.entries(d.params)) html += renderRow(k, v);
            html += '</table>';

            html += '<h3>Ödül fonksiyonu ağırlıkları</h3><table>';
            for (const [k, v] of Object.entries(d.weights)) html += renderRow(k, v);
            html += '</table>';

            html += '<h3>Ödül formülü</h3><pre>' + escapeHtml(REWARD_FORMULA) + '</pre>';

            html += '<h3>Literatür referansları</h3><ul>';
            d.refs.forEach(r => {{ html += '<li>' + escapeHtml(r) + '</li>'; }});
            html += '</ul>';
            return html;
        }}

        function buildStrategyInfoHtml() {{
            const sk = document.getElementById('sel-strategy').value;
            const s = STRATEGY_INFO[sk];
            if (!s) return '<p>Bilgi bulunamadı.</p>';
            const badgeCls = s.kind === 'rl' ? 'rl' : 'baseline';
            const badgeTxt = s.kind === 'rl' ? 'Pekiştirmeli Öğrenme Ajanı' : 'Sabit (Baseline) Politika';

            let html = '';
            html += '<span class="badge ' + badgeCls + '">' + badgeTxt + '</span>';
            html += '<h2>' + escapeHtml(s.title) + '</h2>';
            html += '<p>' + escapeHtml(s.desc) + '</p>';

            html += '<h3>Parametreler / Karar Mantığı</h3><table>';
            for (const [k, v] of Object.entries(s.params)) html += renderRow(k, v);
            html += '</table>';

            // Mevcut senaryonun gerçek sonuçlarını da göster
            const dk = document.getElementById('sel-driver').value;
            const key = dk + '_' + sk;
            const sum = (allScenarios[key] || {{}}).summary;
            if (sum) {{
                html += '<h3>Bu senaryodaki sonuç (' +
                        escapeHtml(DRIVER_INFO[dk].name) + ' · ' +
                        escapeHtml(allRoutes[currentRouteKey].label) + ')</h3><table>';
                html += renderRow('Toplam süre',  sum.total_time_h.toFixed(2) + ' saat');
                html += renderRow('Şarj süresi',  Math.round(sum.charge_time_min) + ' dk');
                html += renderRow('Maliyet',      Math.round(sum.total_cost_tl) + ' TL');
                html += renderRow('Varış SoC',    '%' + Math.round(sum.arrival_soc * 100));
                html += '</table>';
            }}

            html += '<h3>Notlar</h3><p>' + escapeHtml(s.notes) + '</p>';
            return html;
        }}

        function showInfo(kind) {{
            const c = document.getElementById('info-content');
            c.innerHTML = kind === 'driver' ? buildDriverInfoHtml() : buildStrategyInfoHtml();
            document.getElementById('info-modal').classList.add('show');
        }}

        function hideInfo() {{
            document.getElementById('info-modal').classList.remove('show');
        }}

        function _availableStrategies(rk) {{
            const sc = allRoutes[rk].scenarios;
            const keys = Object.keys(sc);
            const out = [];
            for (const sk of STRATEGY_ORDER) {{
                if (keys.some(k => k.endsWith('_' + sk))) out.push(sk);
            }}
            return out;
        }}

        function rebuildStrategyOptions(rk) {{
            const sel = document.getElementById('sel-strategy');
            const prev = sel.value;
            const avail = _availableStrategies(rk);

            sel.innerHTML = '';
            for (const sk of avail) {{
                const opt = document.createElement('option');
                opt.value = sk;
                opt.textContent = STRATEGY_LABELS[sk] || sk;
                sel.appendChild(opt);
            }}
            // Onceki secimi korumaya cali; yoksa 'eco' baseline'a duser
            if (avail.includes(prev))      sel.value = prev;
            else if (avail.includes('eco')) sel.value = 'eco';
            else                            sel.value = avail[0];

            // Eksik / egitilmemis stratejileri bildir
            const missing = STRATEGY_ORDER.filter(sk => !avail.includes(sk));
            const note = document.getElementById('untrained-note');
            if (missing.length === 0) {{
                note.className = 'untrained-note';
                note.innerHTML = '';
            }} else {{
                const names = missing.map(sk => (STRATEGY_LABELS[sk] || sk)).join(', ');
                note.className = 'untrained-note show';
                note.innerHTML = '<b>Eğitim yok:</b> ' + names +
                    ' — bu rota için henüz eğitilmedi (modeli eğittiğimizde otomatik eklenecek).';
            }}
        }}

        function changeRoute() {{
            const rk = document.getElementById('sel-route').value;
            if (!allRoutes[rk]) return;
            currentRouteKey = rk;
            const r = allRoutes[rk];
            allScenarios = r.scenarios;
            routeLine = r.route_line;
            stations = r.stations;
            totalKm = r.total_km;

            // Rota cizgisini yenile
            map.removeLayer(routePath);
            routePath = L.polyline(routeLine, {{ color: '#4361ee', weight: 4, opacity: 0.5 }}).addTo(map);
            drawStations();
            // Haritayi rotanin sinirlarina sigdir
            map.fitBounds(routePath.getBounds(), {{ padding: [40, 40] }});

            rebuildStrategyOptions(rk);
            changeScenario();
        }}

        function changeScenario() {{
            const driver = document.getElementById('sel-driver').value;
            const strat = document.getElementById('sel-strategy').value;
            const key = driver + '_' + strat;
            if (allScenarios[key]) {{
                isPlaying = false;
                if (animationId) cancelAnimationFrame(animationId);
                document.getElementById('play-btn').innerHTML = '▶ Başlat';
                document.getElementById('play-btn').className = 'play-btn play';
                trajectory = allScenarios[key].trajectory;
                currentFrame = 0;
                // reset
                document.getElementById('summary-box').className = 'summary-box';
                updateDisplay(trajectory[0]);
            }}
        }}

        function resetTrip() {{
            isPlaying = false;
            if (animationId) cancelAnimationFrame(animationId);
            currentFrame = 0;
            const pb = document.getElementById('play-btn');
            pb.innerHTML = '▶ Başlat'; pb.className = 'play-btn play';
            document.getElementById('summary-box').className = 'summary-box';
            if (trajectory.length) updateDisplay(trajectory[0]);
        }}

        function setSpeed(s) {{
            speedMultiplier = s;
            document.querySelectorAll('.speed-btn').forEach(b => b.classList.remove('active'));
            event.target.classList.add('active');
        }}

        function togglePlay() {{
            isPlaying = !isPlaying;
            const btn = document.getElementById('play-btn');
            if (isPlaying) {{
                if (currentFrame >= trajectory.length - 1) currentFrame = 0;
                btn.innerHTML = '❚❚ Duraklat'; btn.className = 'play-btn pause';
                lastTimestamp = performance.now(); animate();
            }} else {{
                btn.innerHTML = '▶ Devam'; btn.className = 'play-btn play';
                if (animationId) cancelAnimationFrame(animationId);
            }}
        }}

        function seekTo(e) {{
            const bar = document.getElementById('progress-bar');
            const pct = (e.clientX - bar.getBoundingClientRect().left) / bar.offsetWidth;
            currentFrame = Math.floor(pct * (trajectory.length - 1));
            updateDisplay(trajectory[currentFrame]);
        }}

        function getSocColor(soc) {{
            if (soc > 0.5) return '#2ecc71';
            if (soc > 0.25) return '#f39c12';
            return '#e74c3c';
        }}

        function _metric(label, value, sub) {{
            return '<div class="metric-card">' +
                '<div class="m-label">' + label + '</div>' +
                '<div class="m-value">' + value + '</div>' +
                (sub ? '<div class="m-sub">' + sub + '</div>' : '') +
                '</div>';
        }}

        // Aynı sürücü için tüm stratejilerin özetini çıkartır
        function _peerSummaries(driver) {{
            const out = {{}};
            for (const k of Object.keys(allScenarios)) {{
                if (!k.startsWith(driver + '_')) continue;
                const strat = k.substring(driver.length + 1);
                out[strat] = allScenarios[k].summary;
            }}
            return out;
        }}

        function showSummary() {{
            const driver = document.getElementById('sel-driver').value;
            const strat  = document.getElementById('sel-strategy').value;
            const key = driver + '_' + strat;
            const s = (allScenarios[key] || {{}}).summary;
            if (!s) return;
            const alive = s.arrival_soc > 0.01;

            // Durum rozeti
            const pill = document.getElementById('summary-status');
            pill.textContent = alive ? 'Varış Başarılı' : 'Batarya Bitti';
            pill.className = 'status-pill ' + (alive ? 'ok' : 'fail');

            // Karşılaştırma: aynı sürücüde en iyi baseline ile fark
            const baselines = ['always_70', 'minimum_charge', 'fast_drive', 'eco'];
            const peers = _peerSummaries(driver);
            let bestBaselineTime = Infinity, bestBaselineName = null;
            baselines.forEach(b => {{
                const ps = peers[b];
                if (ps && ps.arrival_soc > 0.01 && ps.total_time_h < bestBaselineTime) {{
                    bestBaselineTime = ps.total_time_h;
                    bestBaselineName = b;
                }}
            }});

            let timeSub = '';
            if (bestBaselineName && bestBaselineName !== strat) {{
                const delta = s.total_time_h - bestBaselineTime;
                const cls = delta < 0 ? 'delta-pos' : 'delta-neg';
                const sign = delta < 0 ? '−' : '+';
                timeSub = '<span class="' + cls + '">' + sign +
                          Math.abs(delta).toFixed(2) + ' sa</span> · en iyi baseline\\'a göre';
            }}

            // Metrik kartları
            const grid = document.getElementById('summary-grid');
            grid.innerHTML =
                _metric('Toplam Süre', s.total_time_h.toFixed(2) + ' sa', timeSub) +
                _metric('Şarj Süresi', Math.round(s.charge_time_min) + ' dk',
                        s.charge_time_min > 60 ? 'uzun mola' : 'kısa mola') +
                _metric('Maliyet', Math.round(s.total_cost_tl) + ' TL', null) +
                _metric('Varış SoC', '%' + Math.round(s.arrival_soc * 100),
                        s.arrival_soc < 0.15 ? 'rezerv düşük' : 'güvenli rezerv');

            // Alt bilgi: rota + sürücü + strateji
            const routeLabel = allRoutes[currentRouteKey].label;
            const driverName = (DRIVER_INFO[driver] || {{}}).name || driver;
            const stratLabel = (STRATEGY_INFO[strat] || {{}}).title || strat;
            document.getElementById('summary-footer').innerHTML =
                '<b>Rota:</b> ' + routeLabel + '<br>' +
                '<b>Sürücü:</b> ' + driverName + ' · <b>Strateji:</b> ' + stratLabel;

            document.getElementById('summary-box').className = 'summary-box show';
        }}

        function updateDisplay(point) {{
            const socPct = Math.round(point.soc * 100);
            document.getElementById('soc-value').textContent = socPct + '%';
            document.getElementById('soc-value').style.color = getSocColor(point.soc);
            document.getElementById('soc-bar').style.width = socPct + '%';
            document.getElementById('soc-bar').style.background = getSocColor(point.soc);
            document.getElementById('soc-bar-text').textContent = socPct + '%';

            const ev = point.event;
            document.getElementById('speed-value').textContent =
                ev === 'charging' ? 'Şarjda…' : ev === 'detour' ? 'Sapma' : Math.round(point.speed) + ' km/h';
            document.getElementById('speed-value').style.color =
                ev === 'charging' ? '#2ecc71' : ev === 'detour' ? '#e67e22' : '#fff';
            document.getElementById('distance-value').textContent = Math.round(point.km) + ' / ' + Math.round(totalKm) + ' km';
            document.getElementById('time-value').textContent = Math.round(point.total_time_min) + ' dk';
            const h = Math.floor(point.hour % 24), m = Math.floor((point.hour % 1) * 60);
            document.getElementById('hour-value').textContent = String(h).padStart(2,'0') + ':' + String(m).padStart(2,'0');
            document.getElementById('altitude-value').textContent = Math.round(point.altitude) + ' m';
            document.getElementById('cost-value').textContent = Math.round(point.cost_tl) + ' TL';

            const grade = point.grade || 0;
            const gEl = document.getElementById('grade-value');
            gEl.textContent = (grade >= 0 ? '+' : '') + grade.toFixed(1) + '%';
            gEl.style.color = grade > 3 ? '#e74c3c' : grade < -3 ? '#2ecc71' : '#fff';

            const cons = point.consumption_kwh100 || 0;
            const cEl = document.getElementById('consumption-value');
            cEl.textContent = cons.toFixed(1) + ' kWh/100km';
            cEl.style.color = cons > 25 ? '#e74c3c' : cons > 18 ? '#f39c12' : '#2ecc71';

            const tf = point.traffic_factor || 1;
            const tEl = document.getElementById('traffic-value');
            if (tf > 0.85) {{ tEl.textContent = 'Akıcı'; tEl.style.color = '#2ecc71'; }}
            else if (tf > 0.65) {{ tEl.textContent = 'Normal'; tEl.style.color = '#f39c12'; }}
            else {{ tEl.textContent = 'Yoğun'; tEl.style.color = '#e74c3c'; }}

            document.getElementById('progress-fill').style.width = (currentFrame / (trajectory.length - 1) * 100) + '%';

            carMarker.setLatLng([point.lat, point.lng]);
            if (currentFrame > 0) {{
                const prev = trajectory[currentFrame - 1];
                const angle = Math.atan2(point.lng - prev.lng, point.lat - prev.lat) * 180 / Math.PI;
                const arrow = document.getElementById('car-arrow');
                if (arrow) arrow.style.transform = 'rotate(' + (-angle + 180) + 'deg)';
            }}
            const arrow = document.getElementById('car-arrow');
            if (arrow) arrow.style.borderBottomColor = ev === 'charging' ? '#2ecc71' : ev === 'detour' ? '#e67e22' : point.soc < 0.15 ? '#e74c3c' : '#4361ee';

            // traveled line kaldirildi
            map.panTo([point.lat, point.lng], {{ animate: true, duration: 0.3 }});

            const popup = document.getElementById('event-popup');
            if (ev === 'detour') {{
                popup.textContent = '↪ Otoyoldan sapılıyor — ' + (point.station || 'istasyon');
                popup.className = 'event-popup charging show';
            }} else if (ev === 'charging' && point.charge_time_min > 0) {{
                popup.textContent = '⚡ ' + (point.station || '') + ' — ' + Math.round(point.charge_time_min) + ' dk şarj';
                popup.className = 'event-popup charging show';
            }} else if (point.soc <= 0.01) {{
                popup.textContent = '❌ Batarya bitti!';
                popup.className = 'event-popup dead show';
            }} else if (point.soc < 0.15) {{
                popup.textContent = '⚠ Düşük batarya! %' + socPct;
                popup.className = 'event-popup warning show';
            }} else popup.className = 'event-popup';

            drawAltitudeChart(point.km);
        }}

        function drawAltitudeChart(currentKm) {{
            const canvas = document.getElementById('altitude-canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = canvas.parentElement.clientWidth - 24;
            canvas.height = canvas.parentElement.clientHeight - 24;
            const w = canvas.width, h = canvas.height;
            ctx.clearRect(0, 0, w, h);
            const alts = trajectory.map(p => p.altitude);
            const kms = trajectory.map(p => p.km);
            const maxAlt = Math.max(...alts) * 1.1 || 100;
            const minAlt = Math.min(...alts.filter(a => a > -100)) * 0.8;
            const grad = ctx.createLinearGradient(0, 0, 0, h);
            grad.addColorStop(0, 'rgba(67,97,238,0.3)'); grad.addColorStop(1, 'rgba(67,97,238,0.02)');
            ctx.beginPath(); ctx.moveTo(0, h);
            for (let i = 0; i < trajectory.length; i++) {{
                ctx.lineTo((kms[i]/totalKm)*w, h-((alts[i]-minAlt)/(maxAlt-minAlt))*h);
            }}
            ctx.lineTo(w, h); ctx.closePath(); ctx.fillStyle = grad; ctx.fill();
            ctx.beginPath();
            for (let i = 0; i < trajectory.length; i++) {{
                const x = (kms[i]/totalKm)*w, y = h-((alts[i]-minAlt)/(maxAlt-minAlt))*h;
                if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
            }}
            ctx.strokeStyle = '#4361ee'; ctx.lineWidth = 2; ctx.stroke();
            const cx = (currentKm/totalKm)*w;
            const ca = trajectory.find(p => p.km >= currentKm)?.altitude || 0;
            const cy = h-((ca-minAlt)/(maxAlt-minAlt))*h;
            ctx.beginPath(); ctx.arc(cx, cy, 5, 0, Math.PI*2);
            ctx.fillStyle = '#2ecc71'; ctx.fill(); ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke();
            ctx.fillStyle = '#8b8fa3'; ctx.font = '10px sans-serif';
            ctx.fillText(Math.round(maxAlt)+'m', 2, 12);
            ctx.fillText(Math.round(minAlt)+'m', 2, h-2);
        }}

        function animate(timestamp) {{
            if (!isPlaying) return;
            if (!lastTimestamp) lastTimestamp = timestamp;
            if (timestamp - lastTimestamp > 50) {{
                lastTimestamp = timestamp;
                currentFrame += Math.max(1, Math.floor(speedMultiplier / 10));
                if (currentFrame >= trajectory.length - 1) {{
                    currentFrame = trajectory.length - 1;
                    updateDisplay(trajectory[currentFrame]);
                    isPlaying = false;
                    if (animationId) cancelAnimationFrame(animationId);
                    document.getElementById('play-btn').innerHTML = '▶ Yeniden Oynat';
                    document.getElementById('play-btn').className = 'play-btn play';
                    showSummary();
                    return;
                }}
                updateDisplay(trajectory[currentFrame]);
            }}
            if (isPlaying) animationId = requestAnimationFrame(animate);
        }}

        // Klavye kısayolları
        document.addEventListener('keydown', e => {{
            if (e.code === 'Space') {{ e.preventDefault(); togglePlay(); }}
            if (e.key === '1') setSpeed(1);
            if (e.key === '2') setSpeed(10);
            if (e.key === '3') setSpeed(40);
            if (e.key === '4') setSpeed(80);
            if (e.key === 'r' || e.key === 'R') resetTrip();
            if (e.key === 'Escape') hideInfo();
        }});

        // İlk yükleme
        rebuildStrategyOptions(currentRouteKey);
        changeScenario();
    </script>
</body>
</html>"""

    with open(output, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Kaydedildi: {Path(output).absolute()}")
    return str(Path(output).absolute())


def visualize_trip(vehicle="ioniq5", route="istanbul_ankara", driver="normal",
                   weather="optimal", strategy="always_70", output="trip_visualization.html"):
    """Convenience wrapper that calls generate_all_scenarios.

    generate_all_scenarios'u çağıran kısayol sarmalayıcı.
    """
    return generate_all_scenarios(route=route, output=output)


if __name__ == "__main__":
    path = generate_all_scenarios()
    print(f"\nTarayicide acin: file://{path}")
