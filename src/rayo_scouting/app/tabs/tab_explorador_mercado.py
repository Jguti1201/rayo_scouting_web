"""
gemini_tab_explorador_mercado.py
================================
Explorador de mercado v3 con clustering integrado:

- Filtros generales + avanzados + numéricos dinámicos por posición
- Filtro por PERFIL TÁCTICO (cluster) según demarcación
- Score de encaje ponderado
- Fichas con perfil táctico, percentiles posicionales y similares
- Distribución de perfiles por posición
- Logs detallados
- Exportación CSV
"""

from __future__ import annotations

import logging
import re
import unicodedata

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tabs.tab_buscador_perfil import (
    render_buscador_perfil_tab as _render_buscador_perfil_tab,
)
from rayo_scouting.features.clustering import (
    CLUSTER_COLORS,
    POSITION_CLUSTER_DESCRIPTIONS,
    compute_positional_percentile,
    enrich_df_with_clusters,
    find_similar_players,
    get_cluster_distribution,
    get_cluster_for_player,
    get_or_build_clusters,
)
from rayo_scouting.features.feature_engineering import (
    CLUSTER_LABELS,
    N_CLUSTERS,
)

logger = logging.getLogger(__name__)


def render_buscador_perfil_tab(df: pd.DataFrame):
    return _render_buscador_perfil_tab(df)


# ============================================================================
# CONFIG
# ============================================================================

GRUPOS_POR_POSICION = {
    "Delantero": [
        {
            "id": "gol",
            "nombre": "Capacidad Goleadora",
            "icono": "⚽",
            "descripcion": "Producción de gol y amenaza ofensiva",
            "columnas": {
                "Goals_p90": "Goles p90",
                "xG_p90": "xG p90",
                "Total_shots_p90": "Disparos p90",
                "Goal conversion %": "Conversión (%)",
                "Big chances missed": "Grandes ocasiones falladas",
            },
            "invertir": ["Big chances missed"],
        },
        {
            "id": "creacion",
            "nombre": "Creación",
            "icono": "🎯",
            "descripcion": "Capacidad para asistir y generar ventajas",
            "columnas": {
                "Assists_p90": "Asistencias p90",
                "Big_chances_created_p90": "Grandes ocasiones creadas p90",
                "Key_passes_p90": "Pases clave p90",
                "Succ_dribbles_p90": "Regates p90",
            },
            "invertir": [],
        },
        {
            "id": "duelos",
            "nombre": "Duelos y Físico",
            "icono": "💪",
            "descripcion": "Impacto físico y disputa",
            "columnas": {
                "ground_duels_won_pct": "Duelos tierra (%)",
                "aerial_duels_won_pct": "Duelos aéreos (%)",
                "total_duels_won_pct": "Total duelos (%)",
            },
            "invertir": [],
        },
        {
            "id": "pase",
            "nombre": "Juego de Pase",
            "icono": "🔄",
            "descripcion": "Limpieza técnica y precisión",
            "columnas": {
                "Accurate passes %": "Precisión pase (%)",
                "Accurate_passes_p90": "Pases precisos p90",
            },
            "invertir": [],
        },
    ],
    "Centrocampista": [
        {
            "id": "creacion",
            "nombre": "Creación de Juego",
            "icono": "🎯",
            "descripcion": "Generación de ventajas y último pase",
            "columnas": {
                "Assists_p90": "Asistencias p90",
                "Big_chances_created_p90": "Grandes ocasiones creadas p90",
                "Key_passes_p90": "Pases clave p90",
            },
            "invertir": [],
        },
        {
            "id": "pase",
            "nombre": "Distribución",
            "icono": "🔄",
            "descripcion": "Control del juego y circulación",
            "columnas": {
                "Accurate passes %": "Precisión pase (%)",
                "Accurate_passes_p90": "Pases precisos p90",
            },
            "invertir": [],
        },
        {
            "id": "defensa",
            "nombre": "Trabajo Defensivo",
            "icono": "🛡️",
            "descripcion": "Recuperación e intervención defensiva",
            "columnas": {
                "Tackles_p90": "Tackles p90",
                "Interceptions_p90": "Intercepciones p90",
            },
            "invertir": [],
        },
        {
            "id": "duelos",
            "nombre": "Duelos",
            "icono": "💪",
            "descripcion": "Capacidad para imponerse",
            "columnas": {
                "ground_duels_won_pct": "Duelos tierra (%)",
                "aerial_duels_won_pct": "Duelos aéreos (%)",
                "total_duels_won_pct": "Total duelos (%)",
            },
            "invertir": [],
        },
        {
            "id": "amenaza",
            "nombre": "Amenaza Ofensiva",
            "icono": "🚀",
            "descripcion": "Llegada, regate y amenaza",
            "columnas": {
                "Goals_p90": "Goles p90",
                "Succ_dribbles_p90": "Regates p90",
                "xG_p90": "xG p90",
            },
            "invertir": [],
        },
    ],
    "Defensa": [
        {
            "id": "defensa",
            "nombre": "Solidez Defensiva",
            "icono": "🛡️",
            "descripcion": "Intervención y defensa del área",
            "columnas": {
                "Tackles_p90": "Tackles p90",
                "Interceptions_p90": "Intercepciones p90",
                "Clearances_p90": "Despejes p90",
                "Blocked_shots_p90": "Bloqueos p90",
            },
            "invertir": [],
        },
        {
            "id": "duelos",
            "nombre": "Duelos",
            "icono": "💪",
            "descripcion": "Disputa por abajo y por arriba",
            "columnas": {
                "ground_duels_won_pct": "Duelos tierra (%)",
                "aerial_duels_won_pct": "Duelos aéreos (%)",
                "total_duels_won_pct": "Total duelos (%)",
            },
            "invertir": [],
        },
        {
            "id": "pase",
            "nombre": "Salida de Balón",
            "icono": "🔄",
            "descripcion": "Capacidad para iniciar juego",
            "columnas": {
                "Accurate passes %": "Precisión pase (%)",
                "Accurate_passes_p90": "Pases precisos p90",
            },
            "invertir": [],
        },
        {
            "id": "disciplina",
            "nombre": "Disciplina",
            "icono": "📋",
            "descripcion": "Reducir faltas y errores graves",
            "columnas": {
                "fouls_p90": "Faltas p90",
                "Errors leading to goal": "Errores a gol",
            },
            "invertir": ["fouls_p90", "Errors leading to goal"],
        },
    ],
    "Portero": [
        {
            "id": "paradas",
            "nombre": "Capacidad de Parada",
            "icono": "🧤",
            "descripcion": "Paradas y rendimiento bajo palos",
            "columnas": {
                "Total_saves_p90": "Paradas p90",
                "Saves_from_inside_box_p90": "Paradas área p90",
                "Clean sheets": "Porterías a cero",
            },
            "invertir": [],
        },
        {
            "id": "salidas",
            "nombre": "Salidas y área",
            "icono": "✈️",
            "descripcion": "Dominio del área",
            "columnas": {
                "Runs out": "Salidas",
                "aerial_duels_won_pct": "Duelos aéreos (%)",
            },
            "invertir": [],
        },
        {
            "id": "pase",
            "nombre": "Juego con los Pies",
            "icono": "🦶",
            "descripcion": "Distribución y limpieza con balón",
            "columnas": {
                "Accurate passes %": "Precisión pase (%)",
                "Accurate_passes_p90": "Pases precisos p90",
            },
            "invertir": [],
        },
    ],
}

FILTROS_DINAMICOS_POR_POSICION = {
    "Delantero": [
        {"col": "Goals_p90", "label": "Goles p90 mínimos", "type": "min", "min": 0.0, "max": 2.0, "step": 0.05, "default": 0.0},
        {"col": "xG_p90", "label": "xG p90 mínimo", "type": "min", "min": 0.0, "max": 2.0, "step": 0.05, "default": 0.0},
        {"col": "Succ_dribbles_p90", "label": "Regates p90 mínimos", "type": "min", "min": 0.0, "max": 10.0, "step": 0.1, "default": 0.0},
        {"col": "Goal conversion %", "label": "Conversión mínima (%)", "type": "min", "min": 0.0, "max": 100.0, "step": 1.0, "default": 0.0},
    ],
    "Centrocampista": [
        {"col": "Key_passes_p90", "label": "Pases clave p90 mínimos", "type": "min", "min": 0.0, "max": 5.0, "step": 0.1, "default": 0.0},
        {"col": "Accurate passes %", "label": "Precisión pase mínima (%)", "type": "min", "min": 0.0, "max": 100.0, "step": 1.0, "default": 0.0},
        {"col": "Interceptions_p90", "label": "Intercepciones p90 mínimas", "type": "min", "min": 0.0, "max": 5.0, "step": 0.1, "default": 0.0},
        {"col": "ground_duels_won_pct", "label": "Duelos suelo mínimos (%)", "type": "min", "min": 0.0, "max": 100.0, "step": 1.0, "default": 0.0},
    ],
    "Defensa": [
        {"col": "Tackles_p90", "label": "Tackles p90 mínimos", "type": "min", "min": 0.0, "max": 6.0, "step": 0.1, "default": 0.0},
        {"col": "Interceptions_p90", "label": "Intercepciones p90 mínimas", "type": "min", "min": 0.0, "max": 6.0, "step": 0.1, "default": 0.0},
        {"col": "aerial_duels_won_pct", "label": "Duelos aéreos mínimos (%)", "type": "min", "min": 0.0, "max": 100.0, "step": 1.0, "default": 0.0},
        {"col": "fouls_p90", "label": "Faltas p90 máximas", "type": "max", "min": 0.0, "max": 5.0, "step": 0.1, "default": 5.0},
    ],
    "Portero": [
        {"col": "Total_saves_p90", "label": "Paradas p90 mínimas", "type": "min", "min": 0.0, "max": 10.0, "step": 0.1, "default": 0.0},
        {"col": "Clean sheets", "label": "Porterías a cero mínimas", "type": "min", "min": 0.0, "max": 30.0, "step": 1.0, "default": 0.0},
        {"col": "Accurate passes %", "label": "Precisión pase mínima (%)", "type": "min", "min": 0.0, "max": 100.0, "step": 1.0, "default": 0.0},
    ],
}

LIGA_DISPLAY = {
    "laliga": "LaLiga", "laliga2": "LaLiga 2", "premier": "Premier League",
    "championship": "Championship", "bundesliga": "Bundesliga",
    "bundesliga2": "Bundesliga 2", "serie_a": "Serie A", "serie_b": "Serie B",
    "ligue1": "Ligue 1", "ligue2": "Ligue 2", "liga_portuguesa": "Primeira Liga",
    "liga_argentina": "Liga Argentina", "mls": "MLS",
}

RADAR_BY_POSITION = {
    "Delantero": [
        ("Goals_p90", "Goles"), ("xG_p90", "xG"), ("Total_shots_p90", "Tiros"),
        ("Succ_dribbles_p90", "Regates"), ("Key_passes_p90", "P. clave"), ("Goal conversion %", "Conv."),
    ],
    "Centrocampista": [
        ("Key_passes_p90", "P. clave"), ("Accurate passes %", "Prec. pase"),
        ("Tackles_p90", "Tackles"), ("Interceptions_p90", "Intercep."),
        ("Goals_p90", "Goles"), ("ground_duels_won_pct", "Duelos"),
    ],
    "Defensa": [
        ("Tackles_p90", "Tackles"), ("Interceptions_p90", "Intercep."),
        ("Clearances_p90", "Despejes"), ("aerial_duels_won_pct", "Aéreos"),
        ("ground_duels_won_pct", "Suelo"), ("Accurate passes %", "Prec. pase"),
    ],
    "Portero": [
        ("Total_saves_p90", "Paradas"), ("Saves_from_inside_box_p90", "Paradas área"),
        ("Clean sheets", "Port. a 0"), ("Runs out", "Salidas"),
        ("Accurate passes %", "Prec. pase"), ("aerial_duels_won_pct", "Aéreos"),
    ],
}


# ============================================================================
# HELPERS
# ============================================================================

def _fix_text(value) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip()
    replacements = {
        "Ã¡": "á", "Ã©": "é", "Ã­": "í", "Ã³": "ó", "Ãº": "ú",
        "Ã±": "ñ", "Ã¼": "ü", "â‚¬": "€", "Â": "",
    }
    for bad, good in replacements.items():
        s = s.replace(bad, good)
    return s.strip()


def _normalize_text_key(value) -> str:
    s = _fix_text(value).lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", s).strip()


def _parse_contract_year(val) -> float:
    if pd.isna(val) or str(val).strip() in ["-", "", "nan"]:
        return np.nan
    s = _fix_text(str(val)).strip()
    year = s[-4:]
    return float(year) if year.isdigit() else np.nan


def _extract_nationalities(series: pd.Series) -> list[str]:
    nat_set = set()
    for val in series.dropna():
        parts = [p.strip() for p in re.split(r"[;,/]", _fix_text(val)) if p.strip()]
        nat_set.update(parts)
    return sorted(nat_set)


def _has_any_nationality(val, selected: list[str]) -> bool:
    if not selected:
        return True
    sel_norm = {_normalize_text_key(n) for n in selected}
    parts = [_normalize_text_key(x) for x in re.split(r"[;,/]", _fix_text(val)) if x.strip()]
    return any(p in sel_norm for p in parts)


def _normalise_col(series: pd.Series) -> pd.Series:
    mn, mx = series.min(), series.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)


def _compute_profile_score(df_pos: pd.DataFrame, grupos: list, pesos: dict) -> pd.Series:
    scores = pd.Series(0.0, index=df_pos.index)
    total_weight = 0.0

    for grupo in grupos:
        peso = pesos.get(grupo["id"], 0)
        if peso == 0:
            continue
        col_scores = []
        for col in grupo["columnas"]:
            if col not in df_pos.columns:
                continue
            col_data = pd.to_numeric(df_pos[col], errors="coerce").fillna(0)
            norm = _normalise_col(col_data)
            if col in grupo.get("invertir", []):
                norm = 1 - norm
            col_scores.append(norm)
        if not col_scores:
            continue
        grupo_score = pd.concat(col_scores, axis=1).mean(axis=1)
        scores += grupo_score * peso
        total_weight += peso

    if total_weight > 0:
        scores = scores / total_weight
    return (scores * 100).round(1)


def _add_watchlist_button(player_name: str, suffix: str = ""):
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = []
    key_base = f"{player_name}_{suffix}" if suffix else player_name

    if player_name in st.session_state.watchlist:
        if st.button("Quitar de cartera", key=f"rm_{key_base}"):
            st.session_state.watchlist.remove(player_name)
            st.rerun()
    else:
        if st.button("Añadir a cartera", key=f"add_{key_base}"):
            st.session_state.watchlist.append(player_name)
            st.rerun()


# ============================================================================
# MINI RADAR PARA FICHAS
# ============================================================================

def _build_mini_radar(player: pd.Series, df: pd.DataFrame, position: str) -> go.Figure | None:
    radar_cols = RADAR_BY_POSITION.get(position)
    if not radar_cols:
        return None

    labels, values = [], []
    for col, lbl in radar_cols:
        if col not in df.columns:
            continue
        raw = float(player.get(col, 0)) if pd.notna(player.get(col)) else 0.0
        pct = compute_positional_percentile(df, position, col, raw)
        labels.append(lbl)
        values.append(pct)

    if len(labels) < 3:
        return None

    fig = go.Figure(go.Scatterpolar(
        r=values, theta=labels, fill='toself',
        line_color='#E30613', fillcolor='rgba(227,6,19,0.2)',
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], showticklabels=False)),
        showlegend=False, height=280, margin=dict(t=20, b=20, l=30, r=30),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ============================================================================
# CLUSTER DISTRIBUTION CHART
# ============================================================================

def _chart_cluster_distribution(df: pd.DataFrame, position: str) -> go.Figure | None:
    dist = get_cluster_distribution(df, position)
    if dist.empty:
        return None

    colors = [list(CLUSTER_COLORS.values())[i % len(CLUSTER_COLORS)] for i in range(len(dist))]

    fig = go.Figure(go.Pie(
        labels=dist["Perfil"], values=dist["Jugadores"],
        marker_colors=colors, textinfo="label+percent", hole=0.4,
    ))
    fig.update_layout(
        title=f"Distribución de perfiles: {position}",
        height=350, showlegend=True,
    )
    return fig


# ============================================================================
# CORE SEARCH
# ============================================================================

def _apply_dynamic_filters(df_pos: pd.DataFrame, posicion: str, dynamic_filters: dict) -> pd.DataFrame:
    for fcfg in FILTROS_DINAMICOS_POR_POSICION.get(posicion, []):
        col = fcfg["col"]
        value = dynamic_filters.get(col)
        if col not in df_pos.columns or value is None:
            continue

        series = pd.to_numeric(df_pos[col], errors="coerce")
        before = len(df_pos)

        if fcfg["type"] == "min":
            df_pos = df_pos[series >= value]
        elif fcfg["type"] == "max":
            df_pos = df_pos[series <= value]

        logger.warning("[EXPLORADOR] Filtro %s (%s=%s): %d -> %d", col, fcfg["type"], value, before, len(df_pos))

    return df_pos


def _buscar_por_perfil(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    logger.warning("=" * 60)
    logger.warning("[EXPLORADOR] INICIO BÚSQUEDA")

    posicion = params["posicion"]
    df_pos = df.copy()

    # Posición
    if "posicion" in df_pos.columns:
        df_pos = df_pos[df_pos["posicion"] == posicion]
        logger.warning("[EXPLORADOR] Posición '%s': %d filas", posicion, len(df_pos))

    # Edad
    if "edad" in df_pos.columns:
        df_pos = df_pos[df_pos["edad"].between(params["edad_range"][0], params["edad_range"][1])]

    # Ligas
    if params["sel_ligas"] and "liga" in df_pos.columns:
        df_pos = df_pos[df_pos["liga"].isin(params["sel_ligas"])]

    # Fiabilidad
    if params["solo_fiables"] and "muestra_fiable" in df_pos.columns:
        df_pos = df_pos[df_pos["muestra_fiable"] == True]

    # Minutos
    if params["minutos_min"] > 0 and "minutes_played" in df_pos.columns:
        df_pos = df_pos[df_pos["minutes_played"] >= params["minutos_min"]]

    # Rating
    if params["rating_min"] > 0 and "Average Sofascore Rating" in df_pos.columns:
        df_pos = df_pos[pd.to_numeric(df_pos["Average Sofascore Rating"], errors="coerce") >= params["rating_min"]]

    # Valor mercado
    if "valor_mercado" in df_pos.columns:
        from rayo_scouting.features.data_loader import parse_market_value_to_millions
        df_pos["_vm"] = df_pos["valor_mercado"].apply(parse_market_value_to_millions)
        if params["presupuesto_min"] is not None:
            df_pos = df_pos[df_pos["_vm"].fillna(-1) >= params["presupuesto_min"]]
        if params["presupuesto_max"] is not None:
            df_pos = df_pos[df_pos["_vm"].fillna(np.inf) <= params["presupuesto_max"]]

    # Contrato
    if params["contract_year_max"] is not None and "fin_contrato" in df_pos.columns:
        df_pos["_cy"] = df_pos["fin_contrato"].apply(_parse_contract_year)
        df_pos = df_pos[df_pos["_cy"].fillna(np.inf) <= params["contract_year_max"]]

    # Nacionalidad
    if params["selected_nationalities"] and "nacionalidades" in df_pos.columns:
        df_pos = df_pos[df_pos["nacionalidades"].apply(lambda x: _has_any_nationality(x, params["selected_nationalities"]))]

    # Clubes
    if params["selected_clubs"] and "tm_club" in df_pos.columns:
        df_pos = df_pos[df_pos["tm_club"].isin(params["selected_clubs"])]
    if params["exclude_clubs"] and "tm_club" in df_pos.columns:
        df_pos = df_pos[~df_pos["tm_club"].isin(params["exclude_clubs"])]

    # Nombre
    if params["name_query"] and "Name" in df_pos.columns:
        q = _normalize_text_key(params["name_query"])
        df_pos = df_pos[df_pos["Name"].astype(str).apply(_normalize_text_key).str.contains(q, na=False)]

    # Filtro por perfil táctico / cluster
    selected_clusters = params.get("selected_clusters", [])
    if selected_clusters and "cluster_label" in df_pos.columns:
        before = len(df_pos)
        df_pos = df_pos[df_pos["cluster_label"].isin(selected_clusters)]
        logger.warning("[EXPLORADOR] Filtro clusters %s: %d -> %d", selected_clusters, before, len(df_pos))

    # Filtros dinámicos numéricos
    df_pos = _apply_dynamic_filters(df_pos, posicion, params.get("dynamic_filters", {}))

    if df_pos.empty:
        return df_pos

    # Score
    grupos = GRUPOS_POR_POSICION.get(posicion, [])
    pesos = params["pesos"]
    total_peso = sum(pesos.get(g["id"], 0) for g in grupos)

    if total_peso == 0:
        df_pos["perfil_score"] = 50.0
        if "Average Sofascore Rating" in df_pos.columns:
            df_pos["perfil_score"] = _normalise_col(
                pd.to_numeric(df_pos["Average Sofascore Rating"], errors="coerce").fillna(0)
            ) * 100
    else:
        df_pos["perfil_score"] = _compute_profile_score(df_pos, grupos, pesos)

    result = df_pos.sort_values("perfil_score", ascending=False).head(20).reset_index(drop=True)
    logger.warning("[EXPLORADOR] Resultados: %d", len(result))
    return result


# ============================================================================
# RENDER RESULTADOS
# ============================================================================

def _render_resultados(result: pd.DataFrame, params: dict, df_full: pd.DataFrame):
    posicion = params["posicion"]
    pesos = params["pesos"]
    grupos = GRUPOS_POR_POSICION.get(posicion, [])

    # Banner de perfil buscado
    rasgos = [
        f"{g['icono']} {g['nombre']} ({'★' * pesos.get(g['id'], 0)})"
        for g in grupos if pesos.get(g["id"], 0) > 0
    ]
    clusters_filtro = params.get("selected_clusters", [])

    if rasgos or clusters_filtro:
        parts = []
        if clusters_filtro:
            parts.append(f"Perfiles: {', '.join(clusters_filtro)}")
        if rasgos:
            parts.append(" · ".join(rasgos))

        st.markdown(f"""
        <div style="background:#F0FDF4;border-left:4px solid #16A34A;border-radius:8px;
                    padding:0.6rem 1rem;font-size:0.8rem;color:#15803d;margin-bottom:1rem;">
            <b>Búsqueda:</b> {' &nbsp;|&nbsp; '.join(parts)}
        </div>
        """, unsafe_allow_html=True)

    if result.empty:
        st.warning("Sin resultados. Amplía los filtros.")
        return

    st.markdown(f"#### TOP {len(result)} JUGADORES")

    for rank, (_, row) in enumerate(result.iterrows(), 1):
        score = row.get("perfil_score", 0.0)
        score_color = "#16A34A" if score >= 70 else "#D97706" if score >= 45 else "#DC2626"

        cluster = row.get("cluster_label", "")
        cluster_desc = POSITION_CLUSTER_DESCRIPTIONS.get(posicion, {}).get(cluster, "")
        liga_str = LIGA_DISPLAY.get(str(row.get("liga", "")), str(row.get("liga", "")))
        club = str(row.get("tm_club", "")) or "—"
        edad = f"{row.get('edad','?'):.0f}" if pd.notna(row.get("edad")) else "?"
        mins = f"{int(row.get('minutes_played',0)):,}'" if pd.notna(row.get("minutes_played")) else "?"
        rating = f"{row.get('Average Sofascore Rating',0):.2f}" if pd.notna(row.get("Average Sofascore Rating")) else "—"

        cluster_badge = ""
        if cluster:
            cluster_badge = f"""
            <span style="background:#EEF2FF;color:#3730A3;border-radius:999px;
                         padding:3px 10px;font-size:0.72rem;font-weight:600;">
                🏷️ {cluster}
            </span>
            """

        stat_lines = []
        for grupo in grupos:
            if pesos.get(grupo["id"], 0) == 0:
                continue
            vals = []
            for col, label in list(grupo["columnas"].items())[:2]:
                if col in row and pd.notna(row[col]):
                    try:
                        v = float(row[col])
                        vals.append(f"<b>{label}:</b> {v:.2f}")
                    except Exception:
                        vals.append(f"<b>{label}:</b> {row[col]}")
            if vals:
                stat_lines.append(f"{grupo['icono']} " + " · ".join(vals))

        stats_html = "<br>".join(stat_lines)

        st.markdown(f"""
        <div style="background:white;border:1px solid #E5E7EB;border-radius:12px;
                    padding:1rem;margin-bottom:0.8rem;">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:1rem;">
                <div>
                    <div style="font-size:0.72rem;color:#6B7280;font-weight:700;">#{rank}</div>
                    <div style="font-size:1.1rem;font-weight:700;color:#111827;">{row.get("Name","?")}</div>
                    <div style="font-size:0.82rem;color:#6B7280;">{liga_str} · {club} · {edad} años</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:1.6rem;font-weight:800;color:{score_color};line-height:1;">{score:.0f}</div>
                    <div style="font-size:0.62rem;color:#9CA3AF;text-transform:uppercase;">Encaje</div>
                </div>
            </div>
            <div style="margin-top:0.5rem;background:#F3F4F6;border-radius:999px;height:8px;overflow:hidden;">
                <div style="width:{int(score)}%;height:8px;background:{score_color};"></div>
            </div>
            <div style="margin-top:0.5rem;display:flex;gap:0.4rem;flex-wrap:wrap;">
                <span style="background:#EEF2FF;color:#3730A3;border-radius:999px;padding:3px 10px;font-size:0.72rem;">⏱ {mins}</span>
                <span style="background:#FEF3C7;color:#92400E;border-radius:999px;padding:3px 10px;font-size:0.72rem;">⭐ {rating}</span>
                <span style="background:#F3F4F6;color:#374151;border-radius:999px;padding:3px 10px;font-size:0.72rem;">💰 {row.get('valor_mercado','N/D')}</span>
                {cluster_badge}
            </div>
            {f'<div style="margin-top:0.5rem;font-size:0.75rem;color:#4B5563;line-height:1.7;">{stats_html}</div>' if stats_html else ''}
        </div>
        """, unsafe_allow_html=True)

        # Expander con radar + similares
        with st.expander(f"📊 Detalles de {row.get('Name', '?')}", expanded=False):
            det_left, det_right = st.columns([1, 1])

            with det_left:
                fig = _build_mini_radar(row, df_full, posicion)
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key=f"radar_{rank}_{posicion}")
                else:
                    st.caption("Radar no disponible.")

            with det_right:
                st.markdown("**Perfil táctico**")
                if cluster:
                    st.markdown(f"🏷️ **{cluster}**")
                    if cluster_desc:
                        st.caption(cluster_desc)
                else:
                    st.caption("Sin perfil asignado.")

                st.markdown("**Jugadores similares**")
                name = str(row.get("Name", ""))
                if name and posicion:
                    sim = find_similar_players(df_full, name, posicion, top_n=5)
                    if not sim.empty:
                        cols_sim = [c for c in ["Name", "tm_club", "similarity"] if c in sim.columns]
                        st.dataframe(sim[cols_sim], use_container_width=True, hide_index=True)
                    else:
                        st.caption("No se encontraron similares.")

        _add_watchlist_button(str(row.get("Name", f"p_{rank}")), suffix=f"exp_{rank}")

    # Exportar
    st.markdown("---")
    export_cols = [c for c in [
        "Name", "tm_club", "liga", "posicion", "edad", "minutes_played",
        "Average Sofascore Rating", "valor_mercado", "fin_contrato",
        "cluster_label", "perfil_score",
    ] if c in result.columns]

    export_df = result[export_cols].copy()
    export_df.columns = [c.replace("_", " ").title() for c in export_df.columns]

    st.download_button(
        "⬇️ Descargar CSV", export_df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"explorador_{params['posicion'].lower()}.csv",
        mime="text/csv", use_container_width=True,
        key=f"export_{posicion}",
    )