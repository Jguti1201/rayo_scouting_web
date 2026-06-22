"""
Análisis de patrones de fichajes del Rayo Vallecano (2016-2026)
Ejecutar con: python analisis_fichajes.py
Requiere: pip install pandas matplotlib seaborn
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ─── Configuración visual ──────────────────────────────────────────────────────
RAYO_RED   = "#D62828"
RAYO_WHITE = "#F5F5F5"
RAYO_DARK  = "#1A1A1A"
RAYO_GRAY  = "#4A4A4A"
ACCENT     = "#F7A12C"   # naranja para el acento

plt.rcParams.update({
    "figure.facecolor": RAYO_DARK,
    "axes.facecolor":   RAYO_DARK,
    "axes.edgecolor":   RAYO_GRAY,
    "axes.labelcolor":  RAYO_WHITE,
    "xtick.color":      RAYO_WHITE,
    "ytick.color":      RAYO_WHITE,
    "text.color":       RAYO_WHITE,
    "grid.color":       "#333333",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "font.family":      "DejaVu Sans",
})

RENDIMIENTO_ORDEN = ["Alto/Clave", "Funcional", "Deficiente (Flop)"]
RENDIMIENTO_COL   = {
    "Alto/Clave":        RAYO_RED,
    "Funcional":         ACCENT,
    "Deficiente (Flop)": "#888888",
}

# ─── Carga y limpieza ──────────────────────────────────────────────────────────
df = pd.read_csv("rayo_scouting_web\\notebooks\\fichajes\\fichajes_rayo.csv")
df["Rendimiento_clean"] = df["Descripción del rendimiento"].str.strip()

# Coste numérico: Libre=0, Cesión=NaN (no hay desembolso fijo), número=número
def parse_coste(val):
    if str(val).strip().lower() == "libre":
        return 0.0
    if str(val).strip().lower() == "cesión":
        return None
    try:
        return float(val)
    except:
        return None

df["Coste_num"] = df["Coste"].apply(parse_coste)
df["Temporada_start"] = df["Temporada"].str[:4].astype(int)
df["Es_Liga_Española"] = df["Liga del club de procedencia"].str.contains("España", na=False)
df["Tipo_Coste"] = df["Coste"].apply(
    lambda x: "Libre" if str(x).lower()=="libre"
    else ("Cesión" if str(x).lower()=="cesión" else "Traspaso")
)

# ─── Estadísticas generales ────────────────────────────────────────────────────
print("=" * 60)
print("  ANÁLISIS DE FICHAJES — RAYO VALLECANO 2016-2026")
print("=" * 60)
print(f"\nTotal fichajes analizados: {len(df)}")
print(f"Temporadas cubiertas:       {df['Temporada'].nunique()}")
print(f"Edad media en el fichaje:   {df['Edad en el fichaje'].mean():.1f} años")

print("\n--- Distribución por rendimiento ---")
vc = df["Rendimiento_clean"].value_counts()
for cat in RENDIMIENTO_ORDEN:
    n = vc.get(cat, 0)
    pct = n / len(df) * 100
    print(f"  {cat:<22} {n:>3}  ({pct:.1f}%)")

print("\n--- Tipo de operación ---")
print(df["Tipo_Coste"].value_counts().to_string())

print("\n--- Edad media por rendimiento ---")
print(df.groupby("Rendimiento_clean")["Edad en el fichaje"].mean().round(1).to_string())

print("\n--- TOP 5 fichajes más caros (traspasos) ---")
caros = df[df["Coste_num"].notna() & (df["Coste_num"] > 0)].sort_values("Coste_num", ascending=False)
for _, row in caros.head(5).iterrows():
    print(f"  {row['Jugador']:<22} {row['Coste_num']/1e6:.1f}M€  →  {row['Rendimiento_clean']}")

print("\n--- Rendimiento de fichajes 'Libre' ---")
libres = df[df["Tipo_Coste"] == "Libre"]
print(libres["Rendimiento_clean"].value_counts().to_string())

print("\n--- Rendimiento de fichajes 'Cesión' ---")
cesiones = df[df["Tipo_Coste"] == "Cesión"]
print(cesiones["Rendimiento_clean"].value_counts().to_string())

# ─── Gráficas ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Análisis de Fichajes · Rayo Vallecano 2016–2026",
             fontsize=16, fontweight="bold", color=RAYO_WHITE, y=1.01)
fig.patch.set_facecolor(RAYO_DARK)

# 1. Distribución de rendimiento (donut)
ax = axes[0, 0]
counts = [vc.get(c, 0) for c in RENDIMIENTO_ORDEN]
colors = [RENDIMIENTO_COL[c] for c in RENDIMIENTO_ORDEN]
wedges, texts, autotexts = ax.pie(
    counts, labels=RENDIMIENTO_ORDEN, colors=colors,
    autopct="%1.0f%%", startangle=90,
    wedgeprops=dict(width=0.55, edgecolor=RAYO_DARK, linewidth=2),
    pctdistance=0.75,
)
for t in texts:   t.set_color(RAYO_WHITE); t.set_fontsize(9)
for a in autotexts: a.set_color(RAYO_DARK); a.set_fontweight("bold")
ax.set_title("Distribución de Rendimiento", fontweight="bold")

# 2. Tipo de operación vs rendimiento (stacked bar)
ax = axes[0, 1]
cross = pd.crosstab(df["Tipo_Coste"], df["Rendimiento_clean"])
cross = cross[[c for c in RENDIMIENTO_ORDEN if c in cross.columns]]
bottom = [0] * len(cross)
for cat in RENDIMIENTO_ORDEN:
    if cat in cross.columns:
        vals = cross[cat].values
        bars = ax.bar(cross.index, vals, bottom=bottom,
                      color=RENDIMIENTO_COL[cat], label=cat, edgecolor=RAYO_DARK, linewidth=0.8)
        for b, v, bot in zip(bars, vals, bottom):
            if v > 0:
                ax.text(b.get_x() + b.get_width()/2, bot + v/2,
                        str(v), ha="center", va="center",
                        fontsize=9, fontweight="bold", color=RAYO_DARK)
        bottom = [b + v for b, v in zip(bottom, vals)]
ax.set_title("Tipo de Operación vs Rendimiento", fontweight="bold")
ax.set_xlabel("Tipo de coste")
ax.set_ylabel("Nº de fichajes")
ax.legend(fontsize=8, loc="upper right")
ax.yaxis.grid(True); ax.set_axisbelow(True)

# 3. Edad en el fichaje por rendimiento (violín)
ax = axes[0, 2]
for i, cat in enumerate(RENDIMIENTO_ORDEN):
    data = df[df["Rendimiento_clean"] == cat]["Edad en el fichaje"]
    parts = ax.violinplot(data, positions=[i], widths=0.6, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor(RENDIMIENTO_COL[cat])
        pc.set_alpha(0.7)
    parts["cmedians"].set_color(RAYO_WHITE)
    parts["cmins"].set_color(RAYO_GRAY)
    parts["cmaxes"].set_color(RAYO_GRAY)
    parts["cbars"].set_color(RAYO_GRAY)
ax.set_xticks(range(len(RENDIMIENTO_ORDEN)))
ax.set_xticklabels(RENDIMIENTO_ORDEN, fontsize=9)
ax.set_title("Distribución de Edad por Rendimiento", fontweight="bold")
ax.set_ylabel("Edad en el fichaje")
ax.yaxis.grid(True); ax.set_axisbelow(True)

# 4. Fichajes por temporada coloreados por rendimiento
ax = axes[1, 0]
temp_rend = pd.crosstab(df["Temporada"], df["Rendimiento_clean"])
temp_rend = temp_rend[[c for c in RENDIMIENTO_ORDEN if c in temp_rend.columns]]
bottom = [0] * len(temp_rend)
for cat in RENDIMIENTO_ORDEN:
    if cat in temp_rend.columns:
        vals = temp_rend[cat].values
        ax.bar(temp_rend.index, vals, bottom=bottom,
               color=RENDIMIENTO_COL[cat], label=cat, edgecolor=RAYO_DARK, linewidth=0.5)
        bottom = [b + v for b, v in zip(bottom, vals)]
ax.set_title("Fichajes por Temporada", fontweight="bold")
ax.set_xlabel("Temporada")
ax.set_ylabel("Nº de fichajes")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
ax.legend(fontsize=8)
ax.yaxis.grid(True); ax.set_axisbelow(True)

# 5. Coste de traspaso vs rendimiento (scatter con jitter)
ax = axes[1, 1]
traspasos = df[df["Coste_num"].notna() & (df["Coste_num"] > 0)].copy()
import numpy as np
np.random.seed(42)
rend_pos = {c: i for i, c in enumerate(RENDIMIENTO_ORDEN)}
for cat in RENDIMIENTO_ORDEN:
    sub = traspasos[traspasos["Rendimiento_clean"] == cat]
    jitter = np.random.uniform(-0.2, 0.2, len(sub))
    ax.scatter(
        [rend_pos[cat] + j for j in jitter],
        sub["Coste_num"] / 1e6,
        color=RENDIMIENTO_COL[cat], s=80, alpha=0.85,
        edgecolors=RAYO_WHITE, linewidths=0.5, zorder=3
    )
    for _, row in sub.iterrows():
        if row["Coste_num"] >= 4e6:
            ax.annotate(row["Jugador"].split()[-1],
                        xy=(rend_pos[cat], row["Coste_num"]/1e6),
                        xytext=(6, 0), textcoords="offset points",
                        fontsize=7.5, color=RAYO_WHITE, va="center")
ax.set_xticks(range(len(RENDIMIENTO_ORDEN)))
ax.set_xticklabels(RENDIMIENTO_ORDEN, fontsize=9)
ax.set_title("Coste de Traspaso vs Rendimiento", fontweight="bold")
ax.set_ylabel("Coste (M€)")
ax.yaxis.grid(True); ax.set_axisbelow(True)

# 6. Procedencia de liga vs tasa de éxito (Alto/Clave %)
ax = axes[1, 2]
liga_map = {
    "Segunda División (España)":    "2ª ESP",
    "Primera División (España)":    "1ª ESP",
    "Primera División (Argentina)": "1ª ARG",
    "Liga MX (México)":             "Liga MX",
    "Ligue 1 (Francia)":            "L1 FRA",
    "Première Liga (Portugal)":     "PL POR",
    "Primeira Liga (Portugal)":     "PL POR",
    "Segunda División B (España)":  "2ªB ESP",
    "Süper Lig (Turquía)":          "Turquía",
    "Superliga de China (China)":   "China",
    "Ligue 2 (Francia)":            "L2 FRA",
    "Serie A (Brasil)":             "SA BRA",
}
df["Liga_corta"] = df["Liga del club de procedencia"].map(liga_map).fillna("Otras")
liga_stats = df.groupby("Liga_corta").apply(
    lambda x: pd.Series({
        "total": len(x),
        "alto":  (x["Rendimiento_clean"] == "Alto/Clave").sum(),
    })
).reset_index()
liga_stats = liga_stats[liga_stats["total"] >= 3].copy()
liga_stats["tasa"] = liga_stats["alto"] / liga_stats["total"] * 100
liga_stats = liga_stats.sort_values("tasa", ascending=True)
bars = ax.barh(liga_stats["Liga_corta"], liga_stats["tasa"],
               color=RAYO_RED, edgecolor=RAYO_DARK, linewidth=0.5)
for bar, total in zip(bars, liga_stats["total"]):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
            f"n={total}", va="center", fontsize=8, color=RAYO_WHITE)
ax.set_title("% Éxito (Alto/Clave) por Liga de Procedencia\n(mín. 3 fichajes)", fontweight="bold")
ax.set_xlabel("% Alto/Clave")
ax.set_xlim(0, 100)
ax.xaxis.grid(True); ax.set_axisbelow(True)

plt.tight_layout(pad=2.5)
plt.savefig("analisis_fichajes.png", dpi=150, bbox_inches="tight",
            facecolor=RAYO_DARK, edgecolor="none")
plt.show()
print("\n✅ Gráfica guardada como 'analisis_fichajes.png'")