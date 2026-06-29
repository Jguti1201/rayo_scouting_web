"""
gemini_report_generation.py
============================
Generador de informes de scouting profesionales para el Rayo Vallecano.

v7 — Mejoras:
- Diagrama de estrella (5 dimensiones) en lugar del radar de percentiles:
    Shooting / Passing / Dribbling / Defending / Physical
    (adaptado para Portero: Paradas / Distribución / Salidas / Posicionamiento / Físico)
  Cada dimensión es un percentil compuesto calculado frente a jugadores de
  la misma posición con ≥ 450 minutos.
- Nueva estructura del PDF:
    01. Perfil del jugador + Diagrama de estrella
    02. Veredicto final
    03. Encaje con el Rayo Vallecano
    04. Análisis técnico
    05. Comparativa con plantilla
    06. Proyección de adaptación
- Reintentos a Claude si devuelve JSON inválido (hasta 2 reintentos)
- Keys de Streamlit 100% únicos
- API key SOLO desde variable de entorno
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import tempfile
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from PIL import Image

logger = logging.getLogger(__name__)


# ============================================================================
# COLORES Y CONFIG GENERAL
# ============================================================================

RAYO_RED   = "#E30613"
RAYO_DARK  = "#1A1A2E"
RAYO_GRAY  = "#6c757d"
RAYO_LIGHT = "#F8F9FA"
RAYO_GREEN = "#28a745"
RAYO_AMBER = "#C89607"

RAYO_ESCUDO_URL = (
    "https://upload.wikimedia.org/wikipedia/en/thumb/1/17/"
    "Rayo_Vallecano_logo.svg/180px-Rayo_Vallecano_logo.svg.png"
)

CLUB_NAME = "Rayo Vallecano"

MIN_MINUTES_PERCENTILE = 450   # mínimo de minutos para entrar en el cálculo percentil

CLUB_NEEDS = [
    {"need": "Extremo con gol",          "description": "Jugador de banda con capacidad goleadora (>5 goles/temporada)"},
    {"need": "Lateral derecho ofensivo", "description": "Lateral con proyección y centros (>2 centros p90)"},
    {"need": "Mediocentro recuperador",  "description": "Pivote con >3 intercepciones p90 y >85% pase"},
    {"need": "Delantero centro target",  "description": "9 con presencia aérea y >0.4 goles p90"},
    {"need": "Central con salida",       "description": "Defensa con >88% precisión de pase"},
]

ADAPTATION_CRITERIA = {
    "budget_max_millions": 5.0,
    "age_max": 30,
    "rating_min": 6.5,
    "minutes_min": 500,
    "preferred_leagues": [
        "laliga", "laliga2", "serie_a", "serie_b",
        "ligue1", "ligue2", "liga_argentina", "mls",
    ],
    "style_keywords": [
        "vertical", "transiciones", "pressing", "intensidad",
        "sacrificio defensivo", "juego directo",
    ],
}

CLUB_TRANSFER_PROFILE = """
PERFIL HISTÓRICO DE FICHAJES QUE HAN FUNCIONADO EN EL RAYO VALLECANO:

Vías de incorporación con buen rendimiento histórico:
- Cesiones de grandes clubes (Real Madrid, Atlético): jóvenes motivados por minutos.
- Agentes libres veteranos con mentalidad combativa a coste cero.
- Fichajes desde Segunda División con nivel de duelo físico alto.

Fortalezas en gestión deportiva:
- Gran acierto en porteros y laterales de bajo coste y alto rendimiento.
- Buen ojo para centrocampistas infravalorados en modelos de alta presión.

Debilidades a evitar:
- Fichajes caros o mediáticos que no encajan en plantilla "obrera".
- Bajo acierto en ligas exóticas fuera de España/Portugal/Sudamérica.

IMPLICACIÓN: valorar jóvenes con proyección, agentes libres consolidados,
jugadores de ligas físicas. Cautela con fichajes de alto coste mediático.
"""

CLUB_TACTICAL_MODEL = """
MODELO DE JUEGO DEL RAYO VALLECANO:

Construcción: salida en corto por carril central, ~79.8% precisión pase.
Ataque: ocupación de bandas, ~31.7 entradas al área/partido, 0.167 xG/remate.
PUNTO CRÍTICO: déficit de finalización (Goles - xG ≈ -1.3/partido).
Defensa: bloque medio-alto agresivo (PPDA ≈7.41), ~56.6% éxito en tackle.
Transiciones: buena recuperación pero dificultad para transitar verticalmente.

IMPLICACIÓN: valorar rematadores eficaces (alto ratio goles/xG), capacidad
de ruptura por conducción, solidez en duelo físico, seguridad en salida de
balón bajo presión.
"""

LIGA_DISPLAY = {
    "laliga": "LaLiga", "laliga2": "LaLiga 2", "premier": "Premier League",
    "championship": "Championship", "bundesliga": "Bundesliga",
    "bundesliga2": "Bundesliga 2", "serie_a": "Serie A", "serie_b": "Serie B",
    "ligue1": "Ligue 1", "ligue2": "Ligue 2", "liga_portuguesa": "Primeira Liga",
    "liga_argentina": "Liga Argentina", "mls": "MLS",
}


# ============================================================================
# STATS POR POSICIÓN (para tabla de stats clave en PDF)
# ============================================================================

POSITION_STATS = {
    "Delantero": {
        "key_stats": [
            "Goals", "Expected goals (xG)", "Total shots", "Goal conversion %",
            "Succ. dribbles", "Key passes", "Assists",
            "Big chances created", "Big chances missed",
            "aerial_duels_won_pct", "total_duels_won_pct",
        ],
    },
    "Centrocampista": {
        "key_stats": [
            "Goals", "Assists", "Key passes", "Accurate passes %",
            "Tackles", "Interceptions", "Big chances created",
            "ground_duels_won_pct", "aerial_duels_won_pct",
        ],
    },
    "Defensa": {
        "key_stats": [
            "Tackles", "Interceptions", "Clearances", "Blocked shots",
            "Accurate passes %", "aerial_duels_won_pct",
            "Errors leading to goal",
        ],
    },
    "Portero": {
        "key_stats": [
            "Total saves", "Saves from inside box", "Clean sheets",
            "Penalties saved", "Runs out", "Accurate passes %",
        ],
    },
}

P90_NAME_MAP = {
    "Goals":                  "Goals_p90",
    "Assists":                "Assists_p90",
    "Expected goals (xG)":    "xG_p90",
    "Succ. dribbles":         "Succ_dribbles_p90",
    "Total shots":            "Total_shots_p90",
    "Tackles":                "Tackles_p90",
    "Interceptions":          "Interceptions_p90",
    "Clearances":             "Clearances_p90",
    "Blocked shots":          "Blocked_shots_p90",
    "Big chances created":    "Big_chances_created_p90",
    "Key passes":             "Key_passes_p90",
    "Total saves":            "Total_saves_p90",
    "Saves from inside box":  "Saves_from_inside_box_p90",
    "Big chances missed":     "Big_chances_missed_p90",
    "Runs out":               "Runs_out_p90",
}

NON_P90_STATS = {
    "Goal conversion %", "Accurate passes %", "ground_duels_won_pct",
    "aerial_duels_won_pct", "total_duels_won_pct", "Average Sofascore Rating",
    "Clean sheets", "Penalties saved", "Errors leading to goal",
}


# ============================================================================
# HELPERS GENERALES
# ============================================================================

def _safe_float(v) -> float:
    if v is None:
        return 0.0
    try:
        if pd.isna(v):
            return 0.0
    except (TypeError, ValueError):
        pass
    try:
        return float(v)
    except Exception:
        return 0.0


def _safe_str(v) -> str:
    if v is None:
        return "N/D"
    try:
        if pd.isna(v):
            return "N/D"
    except (TypeError, ValueError):
        pass
    return str(v)


def _fmt_stat(v, decimals: int = 2) -> str:
    f = _safe_float(v)
    if f == 0.0 and (v is None or (isinstance(v, float) and pd.isna(v))):
        return "N/D"
    if f == int(f) and decimals == 0:
        return str(int(f))
    return f"{f:.{decimals}f}"


def _fmt_liga(liga: str) -> str:
    return LIGA_DISPLAY.get(str(liga), str(liga))


def _percentile_in_df(df: pd.DataFrame, col: str, value: float) -> float:
    if col not in df.columns:
        return 0.0
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if series.empty:
        return 0.0
    return round(float((series <= value).mean() * 100), 1)


def _download_image(url: str) -> bytes | None:
    if not url:
        return None
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            return resp.content
    except Exception as e:
        logger.warning("No se pudo descargar imagen: %s -> %s", url, e)
    return None


def _resize_image_bytes(img_bytes: bytes, max_size: int = 200) -> bytes:
    img = Image.open(io.BytesIO(img_bytes))
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _extract_json_object(text: str) -> str:
    cleaned = text.strip()
    if "```" in cleaned:
        lines, in_block = [], False
        for line in cleaned.splitlines():
            if line.strip().startswith("```"):
                in_block = not in_block
                continue
            lines.append(line)
        cleaned = "\n".join(lines).strip()

    start = cleaned.find("{")
    if start == -1:
        raise ValueError("No se encontró '{' en la respuesta de Claude")

    depth, in_string, escape = 0, False, False
    for idx in range(start, len(cleaned)):
        ch = cleaned[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return cleaned[start:idx + 1]

    raise ValueError("No se pudo cerrar el JSON (llaves desbalanceadas)")


def _make_key(base: str, context: str, idx: int) -> str:
    short_hash = hashlib.md5(base.encode()).hexdigest()[:6]
    return f"{base}_{context}_{idx}_{short_hash}"


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


# ============================================================================
# CÁLCULO DE COLUMNAS _p90
# ============================================================================

def add_p90_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "minutes_played" not in df.columns:
        return df

    minutes = pd.to_numeric(df["minutes_played"], errors="coerce")
    safe_minutes = minutes.where(minutes > 0)

    for raw_col, p90_col in P90_NAME_MAP.items():
        if raw_col not in df.columns or p90_col in df.columns:
            continue
        vals = pd.to_numeric(df[raw_col], errors="coerce")
        df[p90_col] = (vals / safe_minutes) * 90

    return df


# ============================================================================
# DIAGRAMA DE ESTRELLA — 5 DIMENSIONES
# ============================================================================

def compute_five_dimensions(
    player: pd.Series,
    df: pd.DataFrame,
    position: str,
) -> dict:
    """
    Calcula las 5 dimensiones del diagrama de estrella como percentil
    compuesto, comparando al jugador frente a todos los de su misma posición
    con ≥ MIN_MINUTES_PERCENTILE minutos.

    Outfield:
        Shooting   → Goals_p90 + xG_p90 + Total_shots_p90 + Goal_conversion_%
        Passing    → Accurate_passes_% + Key_passes_p90 + Assists_p90 + Big_chances_created_p90
        Dribbling  → Succ_dribbles_p90 + total_duels_won_pct
        Defending  → Tackles_p90 + Interceptions_p90 + Clearances_p90 + Blocked_shots_p90
        Physical   → aerial_duels_won_pct + ground_duels_won_pct + total_duels_won_pct

    Portero:
        Paradas        → Total_saves_p90 + Saves_from_inside_box_p90 + Clean_sheets
        Distribución   → Accurate_passes_%
        Salidas        → Runs_out_p90
        Posicionamiento→ aerial_duels_won_pct + ground_duels_won_pct
        Físico         → aerial_duels_won_pct + total_duels_won_pct
    """
    mins_col = pd.to_numeric(df["minutes_played"], errors="coerce")
    subset = df[
        (df["posicion"] == position) & (mins_col >= MIN_MINUTES_PERCENTILE)
    ].copy()
    sub_mins = pd.to_numeric(subset["minutes_played"], errors="coerce")

    p_mins_raw = pd.to_numeric(player.get("minutes_played", 1), errors="coerce")
    safe_p_mins = max(float(p_mins_raw) if pd.notna(p_mins_raw) else 1, 1)

    def _pp90(col: str) -> float:
        """Valor p90 del jugador para una columna acumulada."""
        v = pd.to_numeric(player.get(col, 0), errors="coerce")
        return float(v) / safe_p_mins * 90 if pd.notna(v) else 0.0

    def _pct(col: str) -> float:
        """Valor porcentual/ratio del jugador (sin normalizar por minutos)."""
        v = pd.to_numeric(player.get(col, 0), errors="coerce")
        return float(v) if pd.notna(v) else 0.0

    def _p90_series(col: str) -> pd.Series:
        """Serie p90 del subconjunto posicional."""
        if col not in subset.columns:
            return pd.Series(dtype=float)
        raw = pd.to_numeric(subset[col], errors="coerce")
        return (raw / sub_mins) * 90

    def _pctile(series: pd.Series, player_val: float) -> float:
        s = series.dropna()
        if s.empty:
            return 50.0
        return round(float((s <= player_val).mean() * 100), 1)

    def _pctile_col(col: str, player_val: float) -> float:
        s = pd.to_numeric(subset[col], errors="coerce").dropna() if col in subset.columns else pd.Series(dtype=float)
        if s.empty:
            return 50.0
        return round(float((s <= player_val).mean() * 100), 1)

    def _avg(*vals) -> float:
        valid = [v for v in vals if v is not None]
        return round(float(np.mean(valid)), 1) if valid else 0.0

    if position == "Portero":
        paradas = _avg(
            _pctile(_p90_series("Total saves"),           _pp90("Total saves")),
            _pctile(_p90_series("Saves from inside box"), _pp90("Saves from inside box")),
            _pctile_col("Clean sheets",                   _pct("Clean sheets")),
        )
        distribucion = _pctile_col("Accurate passes %", _pct("Accurate passes %"))
        salidas      = _pctile(_p90_series("Runs out"), _pp90("Runs out"))
        posicion_gk  = _avg(
            _pctile_col("aerial_duels_won_pct", _pct("aerial_duels_won_pct")),
            _pctile_col("ground_duels_won_pct", _pct("ground_duels_won_pct")),
        )
        fisico = _avg(
            _pctile_col("aerial_duels_won_pct", _pct("aerial_duels_won_pct")),
            _pctile_col("total_duels_won_pct",  _pct("total_duels_won_pct")),
        )
        labels = ["Paradas", "Distribución", "Salidas", "Posicionamiento", "Físico"]
        values = [paradas, distribucion, salidas, posicion_gk, fisico]

    else:
        shooting = _avg(
            _pctile(_p90_series("Goals"),               _pp90("Goals")),
            _pctile(_p90_series("Expected goals (xG)"), _pp90("Expected goals (xG)")),
            _pctile(_p90_series("Total shots"),         _pp90("Total shots")),
            _pctile_col("Goal conversion %",            _pct("Goal conversion %")),
        )
        passing = _avg(
            _pctile_col("Accurate passes %",            _pct("Accurate passes %")),
            _pctile(_p90_series("Key passes"),          _pp90("Key passes")),
            _pctile(_p90_series("Assists"),             _pp90("Assists")),
            _pctile(_p90_series("Big chances created"), _pp90("Big chances created")),
        )
        dribbling = _avg(
            _pctile(_p90_series("Succ. dribbles"),     _pp90("Succ. dribbles")),
            _pctile_col("total_duels_won_pct",         _pct("total_duels_won_pct")),
        )
        defending = _avg(
            _pctile(_p90_series("Tackles"),            _pp90("Tackles")),
            _pctile(_p90_series("Interceptions"),      _pp90("Interceptions")),
            _pctile(_p90_series("Clearances"),         _pp90("Clearances")),
            _pctile(_p90_series("Blocked shots"),      _pp90("Blocked shots")),
        )
        physical = _avg(
            _pctile_col("aerial_duels_won_pct",        _pct("aerial_duels_won_pct")),
            _pctile_col("ground_duels_won_pct",        _pct("ground_duels_won_pct")),
            _pctile_col("total_duels_won_pct",         _pct("total_duels_won_pct")),
        )
        labels = ["Shooting", "Passing", "Dribbling", "Defending", "Physical"]
        values = [shooting, passing, dribbling, defending, physical]

    return {
        "labels": labels,
        "values": values,
        "shooting":  values[0],
        "passing":   values[1],
        "dribbling": values[2],
        "defending": values[3],
        "physical":  values[4],
    }


def _generate_star_chart_matplotlib(dims: dict) -> bytes:
    """Fallback matplotlib para cuando kaleido/Chrome no está disponible."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = dims["labels"]
    values = dims["values"]
    n = len(labels)
    angles = [i * 2 * np.pi / n for i in range(n)] + [0]
    vals_closed = values + [values[0]]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # Referencia 50%
    ref_closed = [50] * n + [50]
    ax.fill(angles, ref_closed, alpha=0.08, color="#aaaaaa")
    ax.plot(angles, ref_closed, color="#aaaaaa", linewidth=1, linestyle="--")

    # Jugador
    ax.fill(angles, vals_closed, alpha=0.22, color=RAYO_RED)
    ax.plot(angles, vals_closed, color=RAYO_RED, linewidth=2.5)
    ax.scatter(angles[:-1], values, color=RAYO_RED, s=50, zorder=5)

    # Etiquetas con valores
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(
        [f"{lb}\n{int(v)}" for lb, v in zip(labels, values)],
        size=10, fontweight="bold", color=RAYO_DARK,
    )
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], size=7, color="#999999")
    ax.yaxis.grid(True, color="#dddddd", linestyle="--", linewidth=0.5)
    ax.xaxis.grid(True, color="#dddddd", linestyle="-", linewidth=0.5)
    ax.spines["polar"].set_color("#dddddd")

    plt.title("Perfil de rendimiento (percentil vs posición)", size=9, color=RAYO_GRAY, pad=15)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def generate_star_chart_image(
    player: pd.Series,
    df: pd.DataFrame,
    position: str,
) -> bytes:
    """
    Genera el diagrama de estrella (radar de 5 dimensiones) como imagen PNG.
    Fondo blanco, colores Rayo, con valores numéricos visibles en cada vértice.
    """
    dims = compute_five_dimensions(player, df, position)
    labels = dims["labels"]
    values = dims["values"]

    # Cerrar el polígono
    labels_closed = labels + [labels[0]]
    values_closed = values + [values[0]]

    # Capa de referencia al 50%
    ref = [50] * len(labels)
    ref_closed = ref + [ref[0]]

    fig = go.Figure()

    # Zona de referencia (50 pct)
    fig.add_trace(go.Scatterpolar(
        r=ref_closed,
        theta=labels_closed,
        fill="toself",
        fillcolor="rgba(200,200,200,0.15)",
        line=dict(color="rgba(150,150,150,0.4)", dash="dot", width=1),
        name="Mediana liga",
        showlegend=True,
    ))

    # Jugador
    fig.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=labels_closed,
        fill="toself",
        fillcolor="rgba(227,6,19,0.22)",
        line=dict(color=RAYO_RED, width=2.5),
        name=_safe_str(player.get("Name", "Jugador")),
        showlegend=True,
        mode="lines+markers+text",
        marker=dict(size=7, color=RAYO_RED),
        text=[f"<b>{int(v)}</b>" for v in values_closed],
        textposition="top center",
        textfont=dict(size=11, color=RAYO_DARK),
    ))

    fig.update_layout(
        polar=dict(
            bgcolor="white",
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[20, 40, 60, 80, 100],
                tickfont=dict(size=9, color=RAYO_GRAY),
                gridcolor="rgba(200,200,200,0.5)",
                linecolor="rgba(200,200,200,0.5)",
            ),
            angularaxis=dict(
                tickfont=dict(size=12, color=RAYO_DARK, family="Helvetica"),
                linecolor="rgba(200,200,200,0.5)",
                gridcolor="rgba(200,200,200,0.5)",
            ),
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.18,
            xanchor="center",
            x=0.5,
            font=dict(size=10),
        ),
        width=480,
        height=420,
        margin=dict(t=40, b=60, l=60, r=60),
        paper_bgcolor="white",
        plot_bgcolor="white",
        title=dict(
            text="Perfil de rendimiento (percentil vs posición)",
            font=dict(size=11, color=RAYO_GRAY),
            x=0.5,
        ),
    )

    try:
        return fig.to_image(format="png", scale=2)
    except Exception:
        # Fallback: matplotlib (no requiere Chrome/kaleido)
        return _generate_star_chart_matplotlib(dims)


# ============================================================================
# CARGA DEL CSV MASTER
# ============================================================================

def load_master_df(master_csv_path: str | None = None) -> pd.DataFrame | None:
    path = master_csv_path or os.environ.get("MASTER_CSV_PATH", "")
    if not path:
        default = (
            Path(__file__).resolve().parent
            / "rayo_scouting" / "app" / "data" / "all_leagues_master_v5.csv"
        )
        if default.exists():
            path = str(default)
        else:
            logger.warning("CSV master no encontrado")
            return None

    p = Path(path)
    if not p.exists():
        logger.warning("CSV master no encontrado en: %s", path)
        return None

    try:
        return add_p90_columns(pd.read_csv(p))
    except Exception as e:
        logger.warning("Error leyendo CSV master: %s", e)
        return None


# ============================================================================
# COMPARATIVA DE PLANTILLA
# ============================================================================

def get_squad_position_comparison(
    player: pd.Series,
    master_df: pd.DataFrame,
    position: str,
) -> dict:
    if master_df is None or master_df.empty:
        return {"squad_size": 0, "rows": [], "squad_players": []}

    squad = master_df[
        (master_df["tm_club"].astype(str).str.contains("Rayo", case=False, na=False))
        & (master_df["posicion"].astype(str).str.strip() == str(position).strip())
    ]

    config = POSITION_STATS.get(position, POSITION_STATS["Centrocampista"])
    radar_cols = [(c, c) for c in config["key_stats"][:6]]

    rows = []
    for col, label in radar_cols:
        player_val = _safe_float(player.get(col, 0))
        squad_vals = (
            pd.to_numeric(squad[col], errors="coerce").dropna()
            if col in squad.columns else pd.Series()
        )
        squad_mean = float(squad_vals.mean()) if not squad_vals.empty else None
        squad_max  = float(squad_vals.max())  if not squad_vals.empty else None

        diff_pct, mejora = None, None
        if squad_mean is not None and squad_mean != 0:
            diff_pct = round((player_val - squad_mean) / abs(squad_mean) * 100, 1)
            mejora   = player_val > squad_mean

        rows.append({
            "stat": label, "col": col,
            "player_val": round(player_val, 2),
            "squad_mean": round(squad_mean, 2) if squad_mean is not None else None,
            "squad_max":  round(squad_max,  2) if squad_max  is not None else None,
            "diff_pct": diff_pct, "mejora": mejora,
        })

    squad_players = []
    if not squad.empty:
        cols = [c for c in ["Name", "Average Sofascore Rating", "minutes_played"] if c in squad.columns]
        squad_players = (
            squad[cols].sort_values("Average Sofascore Rating", ascending=False)
            if "Average Sofascore Rating" in squad.columns
            else squad[cols]
        ).to_dict("records")

    return {"squad_size": int(len(squad)), "rows": rows, "squad_players": squad_players}


def build_comparison_text(comparison: dict, position: str) -> str:
    if comparison["squad_size"] == 0:
        return f"No se encontraron jugadores del {CLUB_NAME} en '{position}'."

    lines = [f"Jugadores del {CLUB_NAME} en '{position}': {comparison['squad_size']}."]
    for row in comparison["rows"]:
        if row["squad_mean"] is None:
            continue
        diff_txt = f"{row['diff_pct']:+.1f}%" if row["diff_pct"] is not None else "N/D"
        tag = "MEJORA" if row["mejora"] else "NO MEJORA"
        lines.append(
            f"- {row['stat']}: candidato={row['player_val']} | "
            f"media Rayo={row['squad_mean']} | máx={row['squad_max']} | {diff_txt} ({tag})"
        )
    if comparison["squad_players"]:
        lines.append("\nPlantilla Rayo en esa posición:")
        for p in comparison["squad_players"]:
            lines.append(
                f"  * {p.get('Name','N/D')}: "
                f"rating={p.get('Average Sofascore Rating','N/D')}, "
                f"min={p.get('minutes_played','N/D')}"
            )
    return "\n".join(lines)


# ============================================================================
# CLAUDE — análisis IA
# ============================================================================

def _build_claude_prompt(
    player: pd.Series,
    df: pd.DataFrame,
    position: str,
    comparison: dict,
    dims: dict,
    club_needs: list[dict],
    adaptation_criteria: dict,
) -> str:
    config = POSITION_STATS.get(position, POSITION_STATS["Centrocampista"])
    stats_lines = []
    for col in config["key_stats"]:
        if col in player.index and pd.notna(player[col]):
            raw  = _safe_float(player[col])
            pct  = _percentile_in_df(df, col, raw)
            p90_col = P90_NAME_MAP.get(col)
            p90_txt = ""
            if p90_col and p90_col in player.index:
                p90v = _fmt_stat(player.get(p90_col))
                if p90v != "N/D":
                    p90_txt = f" (p90: {p90v})"
            stats_lines.append(f"- {col}: {_fmt_stat(raw)}{p90_txt} (percentil {pct}%)")

    dims_text = "\n".join(
        f"  {label}: {int(val)}/100"
        for label, val in zip(dims["labels"], dims["values"])
    )

    return f"""Eres un analista de scouting profesional del Rayo Vallecano.
Genera un informe técnico detallado del siguiente jugador.

==================== CONTEXTO DEL CLUB ====================
NECESIDADES:
{chr(10).join(f"- {n['need']}: {n['description']}" for n in club_needs)}

{CLUB_TRANSFER_PROFILE}
{CLUB_TACTICAL_MODEL}
=============================================================

DATOS DEL JUGADOR:
- Nombre: {_safe_str(player.get('Name'))}
- Edad: {_safe_str(player.get('edad'))}
- Posición: {_safe_str(player.get('posicion'))}
- Club actual: {_safe_str(player.get('tm_club'))}
- Liga: {_fmt_liga(player.get('liga', ''))}
- Nacionalidad: {_safe_str(player.get('nacionalidades'))}
- Valor de mercado: {_safe_str(player.get('valor_mercado'))}
- Fin de contrato: {_safe_str(player.get('fin_contrato'))}
- Minutos jugados: {_safe_str(player.get('minutes_played'))}
- Rating medio: {_safe_str(player.get('Average Sofascore Rating'))}

PERFIL DE 5 DIMENSIONES (percentil vs posición):
{dims_text}

ESTADÍSTICAS CLAVE:
{chr(10).join(stats_lines) if stats_lines else "No disponibles."}

COMPARATIVA CON PLANTILLA ({position}):
{build_comparison_text(comparison, position)}

Criterios de adaptación:
- Presupuesto máx: {adaptation_criteria['budget_max_millions']}M EUR
- Edad máx: {adaptation_criteria['age_max']}
- Rating mín: {adaptation_criteria['rating_min']}
- Ligas preferidas: {', '.join(adaptation_criteria['preferred_leagues'])}

Responde EXCLUSIVAMENTE con un JSON válido (sin texto antes ni después):
{{
    "resumen_ejecutivo": "2-3 frases",
    "fortalezas": ["f1", "f2", "f3", "f4", "f5"],
    "debilidades": ["d1", "d2", "d3"],
    "encaje_tactico": "párrafo",
    "encaje_necesidades": [
        {{"need": "nombre", "status": "CUBIERTO|PARCIAL|NO CUBIERTO", "detail": "explicación"}}
    ],
    "comparativa_plantilla": "párrafo comparando con jugadores actuales del Rayo",
    "perfil_fichaje": "párrafo sobre si encaja con patrones históricos del club",
    "adaptacion_score": 75,
    "adaptacion_detalle": "párrafo",
    "veredicto": "MUY RECOMENDADO|RECOMENDADO|VALORAR|NO RECOMENDADO",
    "veredicto_detalle": "párrafo final",
    "estrellas": 4
}}"""


def _get_fallback_analysis() -> dict:
    return {
        "resumen_ejecutivo": "No se pudo generar el análisis automáticamente. Revisar ficha manualmente.",
        "fortalezas": ["Datos insuficientes para análisis automático"],
        "debilidades": ["Datos insuficientes para análisis automático"],
        "encaje_tactico": "Análisis no disponible.",
        "encaje_necesidades": [],
        "comparativa_plantilla": "Análisis no disponible.",
        "perfil_fichaje": "Análisis no disponible.",
        "adaptacion_score": 50,
        "adaptacion_detalle": "Análisis no disponible.",
        "veredicto": "VALORAR",
        "veredicto_detalle": "Se recomienda evaluación manual del jugador.",
        "estrellas": 3,
    }


def generate_analysis_with_claude(
    player: pd.Series,
    df: pd.DataFrame,
    position: str,
    comparison: dict,
    dims: dict,
    club_needs: list[dict] | None = None,
    adaptation_criteria: dict | None = None,
    max_retries: int = 2,
) -> dict:
    try:
        import anthropic
    except ImportError:
        raise ImportError("pip install anthropic")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.error("ANTHROPIC_API_KEY no configurada.")
        return _get_fallback_analysis()

    if club_needs is None:
        club_needs = CLUB_NEEDS
    if adaptation_criteria is None:
        adaptation_criteria = ADAPTATION_CRITERIA

    prompt = _build_claude_prompt(
        player, df, position, comparison, dims, club_needs, adaptation_criteria
    )
    client = anthropic.Anthropic(api_key=api_key)
    player_name = _safe_str(player.get("Name"))

    for attempt in range(1, max_retries + 2):
        logger.warning("[CLAUDE] Intento %d/%d para %s", attempt, max_retries + 1, player_name)
        try:
            message = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=3000,
                messages=[{"role": "user", "content": prompt}],
            )
            response_text = message.content[0].text.strip()
            json_str  = _extract_json_object(response_text)
            analysis  = json.loads(json_str)

            required = ["resumen_ejecutivo", "fortalezas", "veredicto"]
            missing  = [k for k in required if k not in analysis]
            if missing:
                if attempt <= max_retries:
                    continue
                fb = _get_fallback_analysis()
                for k in missing:
                    analysis[k] = fb[k]

            return analysis

        except json.JSONDecodeError:
            if attempt <= max_retries:
                prompt += "\n\nIMPORTANTE: responde SOLO con el JSON, sin texto antes o después."
                continue
        except Exception as e:
            logger.error("[CLAUDE] Error inesperado (intento %d): %s", attempt, e)
            if attempt <= max_retries:
                continue

    return _get_fallback_analysis()


# ============================================================================
# PDF — ScoutingReportPDF
# ============================================================================

class ScoutingReportPDF:

    def __init__(self):
        from fpdf import FPDF
        self.pdf = FPDF(orientation="P", unit="mm", format="A4")
        self.pdf.set_auto_page_break(auto=True, margin=15)
        self.font_name = "Helvetica"
        self._register_fonts()
        self._patch_text_methods()
        self.page_w = 210
        self.margin = 15

    def _register_fonts(self):
        font_dir = Path(__file__).parent / "fonts"
        candidates = []
        if font_dir.exists():
            candidates.append((
                font_dir / "DejaVuSans.ttf",
                font_dir / "DejaVuSans-Bold.ttf",
            ))
        system = [
            ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
            (
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            ),
        ]
        for reg, bold in candidates + system:
            r, b = Path(reg), Path(bold)
            if r.exists():
                self.pdf.add_font("DejaVu", "",  str(r), uni=True)
                self.pdf.add_font("DejaVu", "B", str(b if b.exists() else r), uni=True)
                self.font_name = "DejaVu"
                return

    def _sanitize(self, value) -> str:
        if value is None:
            return ""
        try:
            if pd.isna(value):
                return ""
        except Exception:
            pass
        text = str(value)
        for old, new in {"€": "EUR", "–": "-", "—": "-", "\xa0": " "}.items():
            text = text.replace(old, new)
        try:
            return unicodedata.normalize("NFKD", text)
        except Exception:
            return text

    def _patch_text_methods(self):
        orig_cell  = self.pdf.cell
        orig_multi = self.pdf.multi_cell

        def safe_cell(*a, **kw):
            if "txt" in kw:
                kw["txt"] = self._sanitize(kw["txt"])
            elif len(a) >= 3:
                a = list(a); a[2] = self._sanitize(a[2])
            return orig_cell(*a, **kw)

        def safe_multi(*a, **kw):
            if "txt" in kw:
                kw["txt"] = self._sanitize(kw["txt"])
            elif len(a) >= 3:
                a = list(a); a[2] = self._sanitize(a[2])
            return orig_multi(*a, **kw)

        self.pdf.cell      = safe_cell
        self.pdf.multi_cell = safe_multi

    def _set_font(self, style="", size=10):
        self.pdf.set_font(self.font_name, style, size)

    def _section(self, num: str, title: str):
        self.pdf.ln(5)
        self.pdf.set_fill_color(*_hex_to_rgb(RAYO_RED))
        self.pdf.set_text_color(255, 255, 255)
        self._set_font("B", 11)
        self.pdf.cell(
            self.page_w - 2 * self.margin, 8,
            f"  {num}   {title}",
            fill=True, ln=True,
        )
        self.pdf.set_text_color(0, 0, 0)
        self.pdf.ln(3)

    # ------------------------------------------------------------------
    # SECCIÓN 01: Perfil + Diagrama de estrella
    # ------------------------------------------------------------------

    def _section_01_profile_and_star(
        self,
        player: pd.Series,
        star_bytes: bytes,
        player_img_bytes: bytes | None,
        escudo_bytes: bytes | None,
    ):
        """
        Layout en dos columnas:
          Izquierda (~90 mm): datos del perfil del jugador
          Derecha  (~90 mm): diagrama de estrella
        """
        self._section("01", "PERFIL DEL JUGADOR")

        col_left  = 90
        col_right = self.page_w - 2 * self.margin - col_left
        y_start   = self.pdf.get_y()
        x_left    = self.margin

        # --- Columna izquierda: foto + datos ---
        if player_img_bytes:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
                f.write(player_img_bytes); p = f.name
            self.pdf.image(p, x=x_left, y=y_start, w=30)
            os.unlink(p)
            x_data = x_left + 33
        else:
            x_data = x_left

        self.pdf.set_xy(x_data, y_start)
        self._set_font("B", 13)
        self.pdf.cell(col_left - (x_data - x_left), 7, _safe_str(player.get("Name")), ln=True)

        fields = [
            ("Edad",          _fmt_stat(player.get("edad"), 0)),
            ("Posición",      _safe_str(player.get("posicion"))),
            ("Club",          _safe_str(player.get("tm_club"))),
            ("Liga",          _fmt_liga(player.get("liga", ""))),
            ("Nac.",          _safe_str(player.get("nacionalidades"))),
            ("Fin contrato",  _safe_str(player.get("fin_contrato"))),
            ("Valor",         _safe_str(player.get("valor_mercado"))),
            ("Rating",        _fmt_stat(player.get("Average Sofascore Rating"))),
            ("Minutos",       _fmt_stat(player.get("minutes_played"), 0)),
        ]
        self._set_font("", 8)
        for label, value in fields:
            self.pdf.set_x(x_data)
            self.pdf.cell(22, 5, f"{label}:")
            self._set_font("B", 8)
            self.pdf.cell(col_left - (x_data - x_left) - 22, 5, value, ln=True)
            self._set_font("", 8)

        y_after_profile = self.pdf.get_y()

        # --- Columna derecha: diagrama de estrella ---
        x_right = x_left + col_left + 5
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(star_bytes); p = f.name
        self.pdf.image(p, x=x_right, y=y_start, w=col_right)
        os.unlink(p)

        # Avanzar al final del bloque más alto
        self.pdf.set_y(max(y_after_profile, y_start + 75) + 5)

    # ------------------------------------------------------------------
    # SECCIÓN 02: Veredicto final
    # ------------------------------------------------------------------

    def _section_02_verdict(self, analysis: dict):
        self._section("02", "VEREDICTO FINAL")

        veredicto = analysis.get("veredicto", "VALORAR")
        estrellas = int(analysis.get("estrellas", 3))
        score     = int(analysis.get("adaptacion_score", 50))

        # Color del veredicto
        if veredicto in ("MUY RECOMENDADO", "RECOMENDADO"):
            rgb = _hex_to_rgb(RAYO_GREEN)
        elif veredicto == "VALORAR":
            rgb = _hex_to_rgb(RAYO_AMBER)
        else:
            rgb = _hex_to_rgb(RAYO_RED)

        # Caja de veredicto con fondo de color
        self.pdf.set_fill_color(*rgb)
        self.pdf.set_text_color(255, 255, 255)
        self._set_font("B", 14)
        stars_str = "★" * estrellas + "☆" * (5 - estrellas)
        self.pdf.cell(
            self.page_w - 2 * self.margin, 10,
            f"  {veredicto}   {stars_str}   Score adaptación: {score}%",
            fill=True, ln=True,
        )
        self.pdf.set_text_color(0, 0, 0)
        self.pdf.ln(2)

        self._set_font("", 9)
        self.pdf.multi_cell(0, 5, analysis.get("veredicto_detalle", ""))
        self.pdf.ln(2)

        self._set_font("B", 9)
        self.pdf.cell(0, 5, "Resumen ejecutivo:", ln=True)
        self._set_font("", 9)
        self.pdf.multi_cell(0, 5, analysis.get("resumen_ejecutivo", ""))

    # ------------------------------------------------------------------
    # SECCIÓN 03: Encaje con el Rayo
    # ------------------------------------------------------------------

    def _section_03_fit(self, analysis: dict):
        self.pdf.add_page()
        self._section("03", "ENCAJE CON EL RAYO VALLECANO")

        # Encaje táctico
        self._set_font("B", 10)
        self.pdf.cell(0, 6, "Encaje táctico:", ln=True)
        self._set_font("", 9)
        self.pdf.multi_cell(0, 5, analysis.get("encaje_tactico", ""))
        self.pdf.ln(3)

        # Perfil de fichaje
        self._set_font("B", 10)
        self.pdf.cell(0, 6, "Perfil de fichaje histórico:", ln=True)
        self._set_font("", 9)
        self.pdf.multi_cell(0, 5, analysis.get("perfil_fichaje", ""))
        self.pdf.ln(3)

        # Necesidades del club
        self._set_font("B", 10)
        self.pdf.cell(0, 6, "Necesidades del club:", ln=True)

        needs = analysis.get("encaje_necesidades", [])
        if needs:
            c1 = 50; c2 = 28
            c3 = self.page_w - 2 * self.margin - c1 - c2

            self.pdf.set_fill_color(*_hex_to_rgb(RAYO_LIGHT))
            self._set_font("B", 8)
            self.pdf.cell(c1, 6, "Necesidad",  border=1, fill=True)
            self.pdf.cell(c2, 6, "Estado",     border=1, fill=True)
            self.pdf.cell(c3, 6, "Detalle",    border=1, fill=True, ln=True)

            self._set_font("", 7)
            for item in needs:
                status = item.get("status", "")
                color  = (
                    _hex_to_rgb(RAYO_GREEN) if status == "CUBIERTO"
                    else _hex_to_rgb(RAYO_AMBER) if status == "PARCIAL"
                    else _hex_to_rgb(RAYO_RED)
                )
                self.pdf.set_text_color(*color)
                self.pdf.cell(c1, 5, item.get("need",   "")[:28], border=1)
                self.pdf.cell(c2, 5, status,                       border=1)
                self.pdf.set_text_color(0, 0, 0)
                self.pdf.cell(c3, 5, item.get("detail", "")[:55],  border=1, ln=True)
            self.pdf.set_text_color(0, 0, 0)
        else:
            self._set_font("", 9)
            self.pdf.cell(0, 5, "No se pudo evaluar el encaje con necesidades.", ln=True)

    # ------------------------------------------------------------------
    # SECCIÓN 04: Análisis técnico
    # ------------------------------------------------------------------

    def _section_04_analysis(self, analysis: dict):
        self._section("04", "ANÁLISIS TÉCNICO")

        self._set_font("B", 10)
        self.pdf.set_text_color(*_hex_to_rgb(RAYO_GREEN))
        self.pdf.cell(0, 6, "FORTALEZAS", ln=True)
        self.pdf.set_text_color(0, 0, 0)
        self._set_font("", 9)
        indent = self.margin + 4
        usable = self.page_w - indent - self.margin
        for f in analysis.get("fortalezas", []):
            self.pdf.set_x(indent)
            self.pdf.multi_cell(usable, 5, f"+ {f}")
        self.pdf.ln(3)

        self._set_font("B", 10)
        self.pdf.set_text_color(*_hex_to_rgb(RAYO_RED))
        self.pdf.cell(0, 6, "DEBILIDADES", ln=True)
        self.pdf.set_text_color(0, 0, 0)
        self._set_font("", 9)
        for d in analysis.get("debilidades", []):
            self.pdf.set_x(indent)
            self.pdf.multi_cell(usable, 5, f"- {d}")

    # ------------------------------------------------------------------
    # SECCIÓN 05: Comparativa con plantilla
    # ------------------------------------------------------------------

    def _section_05_squad(
        self, player: pd.Series, comparison: dict, analysis: dict, position: str,
    ):
        self.pdf.add_page()
        self._section("05", f"COMPARATIVA CON PLANTILLA — {position.upper()}")

        self._set_font("", 9)
        self.pdf.multi_cell(
            0, 5,
            f"Jugadores del Rayo Vallecano en esta posición: {comparison.get('squad_size', 0)}.",
        )
        self.pdf.ln(2)

        rows = comparison.get("rows", [])
        if rows:
            c1, c2, c3, c4, c5 = 42, 28, 28, 28, self.page_w - 2 * self.margin - 126

            self.pdf.set_fill_color(*_hex_to_rgb(RAYO_LIGHT))
            self._set_font("B", 7)
            self.pdf.cell(c1, 6, "Estadística",   border=1, fill=True)
            self.pdf.cell(c2, 6, "Candidato",     border=1, fill=True)
            self.pdf.cell(c3, 6, "Media Rayo",    border=1, fill=True)
            self.pdf.cell(c4, 6, "Máx Rayo",      border=1, fill=True)
            self.pdf.cell(c5, 6, "Dif. vs media", border=1, fill=True, ln=True)

            self._set_font("", 7)
            for row in rows:
                self.pdf.cell(c1, 5, str(row["stat"])[:23],                         border=1)
                self.pdf.cell(c2, 5, _fmt_stat(row["player_val"]),                   border=1)
                self.pdf.cell(c3, 5, _fmt_stat(row["squad_mean"]) if row["squad_mean"] is not None else "N/D", border=1)
                self.pdf.cell(c4, 5, _fmt_stat(row["squad_max"])  if row["squad_max"]  is not None else "N/D", border=1)

                if row["diff_pct"] is not None:
                    color = _hex_to_rgb(RAYO_GREEN) if row["mejora"] else _hex_to_rgb(RAYO_RED)
                    self.pdf.set_text_color(*color)
                    self.pdf.cell(c5, 5, f"{row['diff_pct']:+.1f}%", border=1, ln=True)
                    self.pdf.set_text_color(0, 0, 0)
                else:
                    self.pdf.cell(c5, 5, "N/D", border=1, ln=True)

        squad_players = comparison.get("squad_players", [])
        if squad_players:
            self.pdf.ln(3)
            self._set_font("B", 9)
            self.pdf.cell(0, 6, "Plantilla actual del Rayo en esta posición:", ln=True)
            self._set_font("", 8)
            for p in squad_players:
                self.pdf.cell(5, 5, "")
                self.pdf.cell(
                    0, 5,
                    f"- {p.get('Name','N/D')} | "
                    f"Rating: {_fmt_stat(p.get('Average Sofascore Rating'))} | "
                    f"Min: {_fmt_stat(p.get('minutes_played'), 0)}",
                    ln=True,
                )

        self.pdf.ln(3)
        self._set_font("B", 10)
        self.pdf.cell(0, 6, "Valoración: impacto en la plantilla", ln=True)
        self._set_font("", 9)
        self.pdf.multi_cell(0, 5, analysis.get("comparativa_plantilla", "No disponible."))

    # ------------------------------------------------------------------
    # SECCIÓN 06: Proyección de adaptación
    # ------------------------------------------------------------------

    def _section_06_adaptation(self, analysis: dict):
        self._section("06", "PROYECCIÓN DE ADAPTACIÓN")

        score = int(analysis.get("adaptacion_score", 0))
        self._set_font("B", 13)
        color = (
            _hex_to_rgb(RAYO_GREEN) if score >= 75
            else _hex_to_rgb(RAYO_AMBER) if score >= 50
            else _hex_to_rgb(RAYO_RED)
        )
        self.pdf.set_text_color(*color)
        self.pdf.cell(0, 8, f"Score de adaptación: {score}%", ln=True)
        self.pdf.set_text_color(0, 0, 0)
        self._set_font("", 9)
        self.pdf.multi_cell(0, 5, analysis.get("adaptacion_detalle", ""))

        # Pie de página confidencial
        self.pdf.ln(12)
        self._set_font("", 7)
        self.pdf.set_text_color(150, 150, 150)
        self.pdf.cell(
            0, 4,
            f"Informe generado el {datetime.now().strftime('%d/%m/%Y %H:%M')}",
            ln=True,
        )
        self.pdf.cell(
            0, 4,
            "Rayo Vallecano - Departamento de Scouting - CONFIDENCIAL",
            ln=True,
        )
        self.pdf.set_text_color(0, 0, 0)

    # ------------------------------------------------------------------
    # HEADER DE PÁGINA
    # ------------------------------------------------------------------

    def _header(self, escudo: bytes | None):
        self.pdf.set_fill_color(*_hex_to_rgb(RAYO_DARK))
        self.pdf.rect(0, 0, 210, 35, "F")
        self.pdf.set_fill_color(*_hex_to_rgb(RAYO_RED))
        self.pdf.rect(0, 35, 210, 3, "F")

        if escudo:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                f.write(escudo); p = f.name
            self.pdf.image(p, x=10, y=5, h=25)
            os.unlink(p)

        self.pdf.set_text_color(255, 255, 255)
        self._set_font("B", 16)
        self.pdf.set_xy(40, 8)
        self.pdf.cell(0, 8, "INFORME DE SCOUTING", ln=True)
        self._set_font("", 9)
        self.pdf.set_xy(40, 18)
        self.pdf.cell(0, 6, f"RAYO VALLECANO - DEPARTAMENTO DE SCOUTING", ln=True)
        self._set_font("", 8)
        self.pdf.set_xy(40, 25)
        self.pdf.cell(0, 5, f"Fecha: {datetime.now().strftime('%d/%m/%Y')}", ln=True)
        self.pdf.set_text_color(0, 0, 0)

    # ------------------------------------------------------------------
    # ORQUESTADOR DE GENERACIÓN
    # ------------------------------------------------------------------

    def generate(
        self,
        player: pd.Series,
        df: pd.DataFrame,
        analysis: dict,
        star_bytes: bytes,
        comparison: dict,
        escudo_bytes: bytes | None = None,
        player_img_bytes: bytes | None = None,
    ) -> bytes:
        position = _safe_str(player.get("posicion"))

        # ─── Página 1: Perfil + Estrella → Veredicto ───────────────
        self.pdf.add_page()
        self._header(escudo_bytes)
        self.pdf.set_y(42)
        self._section_01_profile_and_star(player, star_bytes, player_img_bytes, escudo_bytes)
        self._section_02_verdict(analysis)

        # ─── Página 2: Encaje + Análisis técnico ────────────────────
        self._section_03_fit(analysis)
        self._section_04_analysis(analysis)

        # ─── Página 3: Comparativa + Adaptación ─────────────────────
        self._section_05_squad(player, comparison, analysis, position)
        self._section_06_adaptation(analysis)

        return bytes(self.pdf.output())


# ============================================================================
# ORQUESTADOR PRINCIPAL
# ============================================================================

def generate_scouting_report(
    player: pd.Series,
    df: pd.DataFrame,
    player_image_url: str | None = None,
    team_logo_url: str | None = None,
    club_needs: list[dict] | None = None,
    adaptation_criteria: dict | None = None,
    master_csv_path: str | None = None,
) -> bytes:
    player_name = _safe_str(player.get("Name"))
    logger.warning("[REPORT] Generando informe para: %s", player_name)

    df = add_p90_columns(df)

    if "Name" in df.columns:
        match = df[df["Name"].astype(str) == player_name]
        if not match.empty:
            player = match.iloc[0]

    position = _safe_str(player.get("posicion"))

    master_df = load_master_df(master_csv_path)

    if master_df is not None:
        match_m = master_df[master_df["Name"].astype(str) == player_name]
        player_for_cmp = match_m.iloc[0] if not match_m.empty else player
    else:
        player_for_cmp = player

    comparison = get_squad_position_comparison(
        player=player_for_cmp,
        master_df=master_df if master_df is not None else df,
        position=position,
    )

    logger.warning("[REPORT] Calculando 5 dimensiones para %s (%s)...", player_name, position)
    dims = compute_five_dimensions(player, df, position)
    logger.warning("[REPORT] Dimensiones: %s", dict(zip(dims["labels"], [int(v) for v in dims["values"]])))

    logger.warning("[REPORT] Generando diagrama de estrella...")
    star_bytes = generate_star_chart_image(player, df, position)

    escudo_bytes = _download_image(RAYO_ESCUDO_URL)
    if escudo_bytes:
        escudo_bytes = _resize_image_bytes(escudo_bytes, max_size=100)

    player_img_bytes = None
    if player_image_url:
        raw = _download_image(player_image_url)
        if raw:
            player_img_bytes = _resize_image_bytes(raw, max_size=150)

    logger.warning("[REPORT] Generando análisis con Claude...")
    analysis = generate_analysis_with_claude(
        player=player,
        df=df,
        position=position,
        comparison=comparison,
        dims=dims,
        club_needs=club_needs,
        adaptation_criteria=adaptation_criteria,
    )

    logger.warning("[REPORT] Generando PDF...")
    pdf_gen = ScoutingReportPDF()
    pdf_bytes = pdf_gen.generate(
        player=player,
        df=df,
        analysis=analysis,
        star_bytes=star_bytes,
        comparison=comparison,
        escudo_bytes=escudo_bytes,
        player_img_bytes=player_img_bytes,
    )

    logger.warning("[REPORT] Informe completado para: %s", player_name)
    return pdf_bytes


# ============================================================================
# INTEGRACIÓN STREAMLIT
# ============================================================================

def render_report_download_button(
    player: pd.Series,
    df: pd.DataFrame,
    player_image_url: str | None = None,
    team_logo_url: str | None = None,
    key_suffix: str = "",
):
    import streamlit as st

    player_name   = _safe_str(player.get("Name"))
    pdf_state_key = f"_pdf_{key_suffix}"

    if pdf_state_key in st.session_state and st.session_state[pdf_state_key] is not None:
        st.download_button(
            label=f"⬇️ Descargar informe de {player_name} (PDF)",
            data=st.session_state[pdf_state_key],
            file_name=f"informe_{player_name.replace(' ', '_')}.pdf",
            mime="application/pdf",
            key=f"dl_{key_suffix}",
            use_container_width=True,
        )
        if st.button("🔄 Regenerar informe", key=f"regen_{key_suffix}", use_container_width=True):
            st.session_state[pdf_state_key] = None
            st.rerun()
        return

    if st.button(
        f"📄 Generar informe de {player_name}",
        key=f"report_btn_{key_suffix}",
        use_container_width=True,
    ):
        with st.spinner(f"Generando informe para {player_name}..."):
            try:
                pdf_bytes = generate_scouting_report(
                    player=player, df=df,
                    player_image_url=player_image_url,
                    team_logo_url=team_logo_url,
                )
                st.session_state[pdf_state_key] = pdf_bytes
                st.rerun()
            except Exception as e:
                logger.exception("Error generando informe: %s", e)
                st.error(f"Error al generar el informe: {e}")


def render_watchlist_reports(df: pd.DataFrame, context: str = "default"):
    import streamlit as st

    watchlist = st.session_state.get("watchlist", [])
    if not watchlist:
        st.info("No hay jugadores en la cartera.")
        return

    st.markdown("### Generar informes de scouting")
    st.caption("Selecciona un jugador para generar su informe completo con IA.")

    for idx, player_name in enumerate(watchlist):
        matches = df[df["Name"].astype(str) == str(player_name)]
        if matches.empty:
            st.warning(f"No se encontró a {player_name} en la base de datos.")
            continue

        if "minutes_played" in matches.columns:
            matches = matches.sort_values("minutes_played", ascending=False)

        player  = matches.iloc[0]
        img_url  = player.get("player_image_url")  if "player_image_url"  in player.index else None
        logo_url = player.get("team_logo_url")     if "team_logo_url"     in player.index else None

        key_suffix = _make_key(player_name, context, idx)

        with st.expander(
            f"{player_name} — "
            f"{_safe_str(player.get('tm_club'))} "
            f"({_safe_str(player.get('posicion'))})",
            key=f"expander_{key_suffix}",
        ):
            c1, c2 = st.columns([2, 1])
            with c1:
                st.write(f"**Liga:** {_fmt_liga(player.get('liga', ''))}")
                st.write(f"**Edad:** {_fmt_stat(player.get('edad'), 0)}")
                st.write(f"**Valor:** {_safe_str(player.get('valor_mercado'))}")
                st.write(f"**Rating:** {_fmt_stat(player.get('Average Sofascore Rating'))}")
            with c2:
                render_report_download_button(
                    player=player, df=df,
                    player_image_url=img_url,
                    team_logo_url=logo_url,
                    key_suffix=key_suffix,
                )