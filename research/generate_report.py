# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Saadettin Yıldırım (Yıldsam Teknoloji)
"""
Akademik Rapor Olusturucu
==========================
Tum sonuclardan grafikler ve LaTeX/Markdown tablolar uretir.
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from evroute.config import get_results_dir

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

RESULTS_DIR = get_results_dir()
REPORT_DIR = RESULTS_DIR.parent / "report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RESULTS_DIR / "full_results.csv")

# ============================================================
# GRAFIK 1: Algoritma Karsilastirmasi (Bar Chart)
# ============================================================
print("[1/6] Algoritma karsilastirma grafigi...")
algo_stats = df.groupby("algorithm").agg({
    "avg_total_reward": "mean",
    "avg_total_time_h": "mean",
    "avg_charge_time_min": "mean",
    "avg_total_cost_tl": "mean",
    "avg_arrival_soc": "mean",
}).reset_index()

algo_stats = algo_stats.sort_values("avg_total_reward", ascending=False)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
colors_rl = "#4361ee"
colors_baseline = "#94a3b8"
algo_colors = [colors_rl if a in ["DQN", "Double DQN", "PPO"] else colors_baseline
               for a in algo_stats["algorithm"]]

# Reward
axes[0, 0].barh(algo_stats["algorithm"], algo_stats["avg_total_reward"], color=algo_colors)
axes[0, 0].set_title("Ortalama Reward (Yuksek = Iyi)")
axes[0, 0].set_xlabel("Reward")

# Sure
axes[0, 1].barh(algo_stats["algorithm"], algo_stats["avg_total_time_h"], color=algo_colors)
axes[0, 1].set_title("Ortalama Toplam Sure (saat)")
axes[0, 1].set_xlabel("Saat")

# Sarj suresi
axes[1, 0].barh(algo_stats["algorithm"], algo_stats["avg_charge_time_min"], color=algo_colors)
axes[1, 0].set_title("Ortalama Sarj Suresi (dakika)")
axes[1, 0].set_xlabel("Dakika")

# Maliyet
axes[1, 1].barh(algo_stats["algorithm"], algo_stats["avg_total_cost_tl"], color=algo_colors)
axes[1, 1].set_title("Ortalama Sarj Maliyeti (TL)")
axes[1, 1].set_xlabel("TL")

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=colors_rl, label="RL Algoritmalari"),
    Patch(facecolor=colors_baseline, label="Baseline Stratejiler"),
]
fig.legend(handles=legend_elements, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
plt.suptitle("Algoritma Karsilastirmasi - 144 Senaryo Ortalamasi", y=1.05, fontweight="bold")
plt.tight_layout()
plt.savefig(REPORT_DIR / "fig1_algorithm_comparison.png", dpi=200, bbox_inches="tight")
plt.close()

# ============================================================
# GRAFIK 2: Ogrenme Egrileri
# ============================================================
print("[2/6] Ogrenme egrileri...")
with open(RESULTS_DIR / "training_curves_all.json") as f:
    curves = json.load(f)

ALGOS = [("DQN", "DQN", "#4361ee"),
         ("DDQN", "Double DQN", "#10b981"),
         ("PPO", "PPO", "#f59e0b")]


def _smooth(series, window):
    """EN: Strong moving average. / TR: Güçlü hareketli ortalama."""
    return pd.Series(series).rolling(window=window, min_periods=1).mean()


def _algo_curves(algo_short):
    """
    EN: All learning curves of one algorithm across every scenario,
        truncated to the shortest common length.
    TR: Bir algoritmanın tüm senaryolardaki öğrenme eğrileri, en kısa
        ortak uzunluğa kırpılmış.
    """
    sel = []
    for key, vals in curves.items():
        # DDQN keys also end with 'DQN'; match the exact suffix.
        if key.endswith("_" + algo_short) and len(vals) > 10:
            sel.append(vals)
    if not sel:
        return None
    n = min(len(c) for c in sel)
    return np.array([c[:n] for c in sel])  # shape: (n_scenarios, n_episodes)


fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# --- Panel 1: representative scenario (illustrative, heavily smoothed) ---
# TR: Temsilî senaryo (illüstratif, güçlü yumuşatma).
sample_key = "ioniq5_istanbul_ankara_normal_optimal_weekday"
for algo_short, algo_label, color in ALGOS:
    rew = curves.get(f"{sample_key}_{algo_short}")
    if rew:
        w = max(20, len(rew) // 20)  # EN: window >= 20 episodes
        axes[0].plot(_smooth(rew, w), label=algo_label, color=color, linewidth=2)
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Reward (smoothed)")
axes[0].set_title("Temsili Senaryo Ogrenme Egrisi\n"
                  "(IONIQ 5 / Istanbul-Ankara / Normal / Optimal)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# --- Panel 2: ALL scenarios, robust aggregate per algorithm ---
# EN: Median + interquartile (25-75%) band — robust to death (-20)
#     outliers; far cleaner than mean +/- std.
# TR: Medyan + çeyrekler arası (25-75%) bant — ölüm (-20) aykırılarına
#     dayanıklı; ortalama ± std'den çok daha temiz.
for algo_short, algo_label, color in ALGOS:
    arr = _algo_curves(algo_short)
    if arr is None:
        continue
    w = max(20, arr.shape[1] // 20)
    med = _smooth(np.median(arr, axis=0), w)
    q25 = _smooth(np.percentile(arr, 25, axis=0), w)
    q75 = _smooth(np.percentile(arr, 75, axis=0), w)
    x = range(arr.shape[1])
    axes[1].plot(x, med, label=f"{algo_label} (medyan)", color=color, linewidth=2)
    axes[1].fill_between(x, q25, q75, alpha=0.18, color=color)
axes[1].set_xlabel("Episode")
axes[1].set_ylabel("Reward")
axes[1].set_title("Tum Senaryolarda Yakinsama\n(medyan + IQR 25-75%)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# --- Panel 3: success-rate proxy (reward above death band) ---
# EN: Fraction of scenarios whose smoothed reward exceeds -10, i.e.
#     not death-dominated. A monotone "is it learning?" signal.
# TR: Yumuşatılmış ödülü -10'un üzerinde (ölüm-baskın olmayan) senaryo
#     oranı. "Ogreniyor mu?" sorusuna neredeyse monoton yanit.
for algo_short, algo_label, color in ALGOS:
    arr = _algo_curves(algo_short)
    if arr is None:
        continue
    w = max(20, arr.shape[1] // 20)
    sm = np.vstack([_smooth(arr[i], w).to_numpy() for i in range(arr.shape[0])])
    success = (sm > -10.0).mean(axis=0)
    axes[2].plot(range(arr.shape[1]), success, label=algo_label,
                 color=color, linewidth=2)
axes[2].set_xlabel("Episode")
axes[2].set_ylabel("Basari orani (odul > -10)")
axes[2].set_ylim(-0.02, 1.02)
axes[2].set_title("Senaryolarin Basari Orani\n(olum-disi yumusak odul payi)")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(REPORT_DIR / "fig2_learning_curves.png", dpi=200, bbox_inches="tight")
plt.close()

# ============================================================
# GRAFIK 3: Hava Durumu Etkisi
# ============================================================
print("[3/6] Hava durumu etkisi...")
weather_pivot = df.pivot_table(values="avg_total_reward", index="algorithm",
                                columns="weather", aggfunc="mean")

# Sadece RL ve en iyi baseline
selected_algos = ["PPO", "DQN", "Double DQN", "Her Istasyonda %70", "Eko Surucu"]
weather_pivot = weather_pivot.loc[[a for a in selected_algos if a in weather_pivot.index]]

fig, ax = plt.subplots(figsize=(10, 6))
weather_pivot.plot(kind="bar", ax=ax, color=["#4361ee", "#10b981", "#f59e0b", "#ef4444"])
ax.set_title("Hava Durumunun Algoritma Performansina Etkisi")
ax.set_xlabel("Algoritma")
ax.set_ylabel("Ortalama Reward")
ax.legend(title="Hava", loc="lower left")
ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
plt.tight_layout()
plt.savefig(REPORT_DIR / "fig3_weather_impact.png", dpi=200, bbox_inches="tight")
plt.close()

# ============================================================
# GRAFIK 4: Surucu Profili Etkisi
# ============================================================
print("[4/6] Surucu profili etkisi...")
driver_pivot = df.pivot_table(values="avg_total_time_h", index="algorithm",
                               columns="driver", aggfunc="mean")
driver_pivot = driver_pivot.loc[[a for a in selected_algos if a in driver_pivot.index]]

fig, ax = plt.subplots(figsize=(10, 6))
driver_pivot.plot(kind="bar", ax=ax, color=["#10b981", "#4361ee", "#ef4444"])
ax.set_title("Surucu Profilinin Yolculuk Suresine Etkisi")
ax.set_xlabel("Algoritma")
ax.set_ylabel("Ortalama Sure (saat)")
ax.legend(title="Surucu", labels=["Agresif", "Eko", "Normal"])
ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
plt.tight_layout()
plt.savefig(REPORT_DIR / "fig4_driver_impact.png", dpi=200, bbox_inches="tight")
plt.close()

# ============================================================
# GRAFIK 5: Rota Karsilastirmasi
# ============================================================
print("[5/6] Rota karsilastirmasi...")
route_pivot = df.pivot_table(values="avg_total_reward", index="algorithm",
                              columns="route", aggfunc="mean")
route_pivot = route_pivot.loc[[a for a in selected_algos if a in route_pivot.index]]

fig, ax = plt.subplots(figsize=(10, 6))
route_pivot.plot(kind="bar", ax=ax, color=["#4361ee", "#10b981", "#f59e0b"])
ax.set_title("Rota Bazinda Algoritma Performansi")
ax.set_xlabel("Algoritma")
ax.set_ylabel("Ortalama Reward")
ax.legend(title="Rota", labels=["Ankara-Antalya", "Istanbul-Ankara", "Istanbul-Izmir"])
ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
plt.tight_layout()
plt.savefig(REPORT_DIR / "fig5_route_comparison.png", dpi=200, bbox_inches="tight")
plt.close()

# ============================================================
# GRAFIK 6: Arac Karsilastirmasi
# ============================================================
print("[6/6] Arac karsilastirmasi...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

vehicle_time = df[df["algorithm"].isin(["PPO", "DQN", "Double DQN"])].groupby(
    ["vehicle", "algorithm"])["avg_total_time_h"].mean().unstack()
vehicle_cost = df[df["algorithm"].isin(["PPO", "DQN", "Double DQN"])].groupby(
    ["vehicle", "algorithm"])["avg_total_cost_tl"].mean().unstack()

vehicle_time.plot(kind="bar", ax=axes[0], color=["#4361ee", "#10b981", "#f59e0b"])
axes[0].set_title("Ortalama Yolculuk Suresi - Arac Karsilastirmasi")
axes[0].set_ylabel("Saat")
axes[0].set_xticklabels(["IONIQ 5", "Tesla Model 3"], rotation=0)
axes[0].legend(title="Algoritma")

vehicle_cost.plot(kind="bar", ax=axes[1], color=["#4361ee", "#10b981", "#f59e0b"])
axes[1].set_title("Ortalama Sarj Maliyeti - Arac Karsilastirmasi")
axes[1].set_ylabel("TL")
axes[1].set_xticklabels(["IONIQ 5", "Tesla Model 3"], rotation=0)
axes[1].legend(title="Algoritma")

plt.tight_layout()
plt.savefig(REPORT_DIR / "fig6_vehicle_comparison.png", dpi=200, bbox_inches="tight")
plt.close()

# ============================================================
# AKADEMIK TABLOLAR (Markdown)
# ============================================================
print("\n[Markdown tablolari olusturuluyor...]")

md_lines = []
md_lines.append("# EV Sarj Optimizasyonu - Deney Sonuclari\n")
md_lines.append("**Deney Tarihi:** 2026-04-12 / 2026-04-13\n")
md_lines.append("**Toplam Senaryo:** 144 (3 rota x 2 arac x 3 surucu x 4 hava x 2 trafik)\n")
md_lines.append("**Toplam Degerlendirme:** 1008 (144 x 7 algoritma)\n")
md_lines.append("**Eğitim:** 1000 episode (DQN/Double DQN), 25000 timestep (PPO)\n")
md_lines.append("**Donanım:** Intel i5-12500H, RTX 3050 Ti 4GB, 32 GB RAM\n\n")

md_lines.append("## 1. Genel Algoritma Karsilastirmasi\n")
md_lines.append("Tum 144 senaryo ortalamasi:\n\n")

algo_table = pd.DataFrame({
    "Algoritma": algo_stats["algorithm"],
    "Ort. Sure (h)": algo_stats["avg_total_time_h"].round(2),
    "Ort. Sarj (dk)": algo_stats["avg_charge_time_min"].round(1),
    "Ort. Maliyet (TL)": algo_stats["avg_total_cost_tl"].round(0).astype(int),
    "Ort. Varis SoC": (algo_stats["avg_arrival_soc"] * 100).round(1).astype(str) + "%",
    "Ort. Reward": algo_stats["avg_total_reward"].round(3),
})

md_lines.append(algo_table.to_markdown(index=False))
md_lines.append("\n\n")

md_lines.append("## 2. Rota Bazinda PPO vs En Iyi Baseline\n\n")
route_summary = df.pivot_table(values="avg_total_time_h", index="route",
                                columns="algorithm", aggfunc="mean").round(2)
md_lines.append(route_summary.to_markdown())
md_lines.append("\n\n")

md_lines.append("## 3. Hava Durumu Etkisi\n\n")
weather_summary = df.pivot_table(values="avg_total_reward", index="weather",
                                  columns="algorithm", aggfunc="mean").round(3)
md_lines.append(weather_summary.to_markdown())
md_lines.append("\n\n")

md_lines.append("## 4. Arac Karsilastirmasi (RL Algoritmalar)\n\n")
v_summary = df[df["algorithm"].isin(["DQN", "Double DQN", "PPO"])].groupby(
    ["vehicle", "algorithm"]).agg({
    "avg_total_time_h": "mean",
    "avg_charge_time_min": "mean",
    "avg_total_cost_tl": "mean",
    "avg_total_reward": "mean",
}).round(3)
md_lines.append(v_summary.to_markdown())
md_lines.append("\n\n")

with open(REPORT_DIR / "results_summary.md", "w") as f:
    f.write("".join(md_lines))

# LaTeX tablo
latex_table = pd.DataFrame({
    "Algoritma": algo_stats["algorithm"],
    "Sure (h)": algo_stats["avg_total_time_h"].round(2),
    "Sarj (dk)": algo_stats["avg_charge_time_min"].round(1),
    "Maliyet (TL)": algo_stats["avg_total_cost_tl"].round(0).astype(int),
    "SoC": (algo_stats["avg_arrival_soc"] * 100).round(1).astype(str) + "\\%",
    "Reward": algo_stats["avg_total_reward"].round(3),
})

with open(REPORT_DIR / "table1_algorithm_comparison.tex", "w") as f:
    f.write(latex_table.to_latex(index=False, escape=False,
            caption="Algoritma Karsilastirmasi - 144 Senaryo Ortalamasi",
            label="tab:algorithm_comparison"))

print(f"\n[OK] Rapor klasoru: {REPORT_DIR}")
print("Olusturulan dosyalar:")
for f in sorted(REPORT_DIR.iterdir()):
    print(f"  - {f.name}")
