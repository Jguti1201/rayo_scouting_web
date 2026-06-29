"""
tab_comparador_perfiles.py
==========================
Comparador de perfiles v4 con:
- Selección de fuente: Plantilla Rayo / Cartera / Base completa
- Fichas con clustering y perfil táctico
- Radar General (5 ejes: Shooting / Passing / Dribbling / Defending / Physical)
  → percentiles GLOBALES respecto a todo el dataset
- Radares detallados por categoría seleccionable con todas las métricas
  → percentiles GLOBALES respecto a todo el dataset
- Lectura ejecutiva, tabla comparativa, jugadores similares
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from rayo_scouting.features.feature_engineering import validate_position
from rayo_scouting.scouting.dashboard_metrics import fmt_liga, get_rayo_df, get_watchlist_df
from rayo_scouting.features.clustering import (
    find_similar_players,
    get_cluster_for_player,
    POSITION_CLUSTER_DESCRIPTIONS,
)

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTES
# ============================================================================

SOURCE_OPTIONS = ["Plantilla Rayo", "Cartera", "Base completa"]

COLOR_A  = "#E30613"
COLOR_B  = "#1A1A2E"
FILL_A   = "rgba(227, 6, 19, 0.18)"
FILL_B   = "rgba(26, 26, 46, 0.18)"

# Métricas para la tabla comparativa (sin cambios respecto a v3)
COMPARISON_METRICS = [
    ("Goals",                    "Goles"),
    ("Assists",                  "Asistencias"),
    ("Expected goals (xG)",      "xG"),
    ("Key passes",               "Pases clave"),
    ("Succ. dribbles",           "Regates"),
    ("Tackles_p90",              "Tackles p90"),
    ("Interceptions_p90",        "Intercepciones p90"),
    ("Clearances_p90",           "Despejes p90"),
    ("Accurate passes %",        "Precisión pase (%)"),
    ("ground_duels_won_pct",     "Duelos suelo (%)"),
    ("aerial_duels_won_pct",     "Duelos aéreos (%)"),
    ("Average Sofascore Rating", "Rating"),
    ("minutes_played",           "Minutos"),
]

# ── Definición de categorías para los radares ─────────────────────────────
# Formato: (columna_df, etiqueta, invertir)
# invertir=True  → métrica negativa; se gira el percentil para que
#                  "hacia fuera" siempre signifique "mejor"
KPI_CATEGORIES: dict[str, list[tuple[str, str, bool]]] = {
    "Shooting": [
        ("Goals",               "Goles",               False),
        ("Expected goals (xG)", "xG",                  False),
        ("Total shots",         "Disparos tot.",        False),
        ("Goal conversion %",   "Conversión %",         False),
        ("Big chances missed",  "Gr. oc. falladas",     True),
        ("Blocked shots",       "Disparos bloq.",       False),
    ],
    "Passing": [
        ("Assists",             "Asistencias",          False),
        ("Accurate passes",     "Pases precisos",       False),
        ("Accurate passes %",   "Precisión pase %",     False),
        ("Key passes",          "Pases clave",          False),
        ("Big chances created", "Gr. oc. creadas",      False),
    ],
    "Dribbling": [
        ("Succ. dribbles",      "Regates exitosos",     False),
        ("ground_duels_won_pct","Duelos terrestres %",  False),
        ("total_duels_won_pct", "Duelos totales %",     False),
    ],
    "Defending": [
        ("Tackles",             "Entradas",             False),
        ("Interceptions",       "Intercepciones",       False),
        ("Clearances",          "Despejes",             False),
        ("aerial_duels_won_pct","Duelos aéreos %",      False),
        ("Errors leading to goal", "Errores -> Gol",    True),
        ("fouls",               "Faltas cometidas",     True),
    ],
    "Physical": [
        ("minutes_played",      "Minutos jugados",      False),
        ("total_duels_won_pct", "Duelos ganados %",     False),
        ("aerial_duels_won_pct","Duelos aéreos %",      False),
        ("ground_duels_won_pct","Duelos terr. %",       False),
        ("fouls",               "Faltas",               True),
    ],
}

CAT_ICONS = {
    "Shooting":  "Shooting",
    "Passing":   "Passing",
    "Dribbling": "Dribbling",
    "Defending": "Defending",
    "Physical":  "Physical",
}


# ============================================================================
# HELPERS
# ============================================================================

def _safe_float(v) -> float:
    if pd.isna(v):
        return 0.0
    try:
        return float(v)
    except Exception:
        return 0.0


def _global_percentile(df: pd.DataFrame, col: str, value: float, invert: bool) -> float:
    """Percentil global del value en df[col]. Si invert, se invierte (100 - pct)."""
    if col not in df.columns:
        return 0.0
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if series.empty or series.std() == 0:
        return 50.0
    pct = float((series <= value).mean() * 100)
    return round(100.0 - pct if invert else pct, 1)


def _get_source_df(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if source == "Plantilla Rayo":
        return get_rayo_df(df)
    elif source == "Cartera":
        return get_watchlist_df(df, st.session_state.get("watchlist", []))
    return df.copy()


def _get_player_options(df: pd.DataFrame, source: str, exclude: str | None = None) -> list[str]:
    source_df = _get_source_df(df, source)
    if source_df.empty or "Name" not in source_df.columns:
        return []
    options = source_df["Name"].dropna().astype(str).sort_values().unique().tolist()
    if exclude:
        options = [p for p in options if p != exclude]
    return options


def _get_player_row(df: pd.DataFrame, name: str) -> pd.Series | None:
    exact = df[df["Name"].astype(str) == str(name)]
    if exact.empty:
        exact = df[df["Name"].astype(str).str.contains(str(name), case=False, na=False)]
    if exact.empty:
        return None
    if "minutes_played" in exact.columns:
        exact = exact.sort_values("minutes_played", ascending=False)
    return exact.iloc[0]


# ============================================================================
# FICHA DE JUGADOR
# ============================================================================

def _render_player_card(row: pd.Series, title: str, color: str, df: pd.DataFrame):
    position = validate_position(row.get("posicion"))
    cluster = get_cluster_for_player(df, str(row.get("Name", "")), position) if position else None
    cluster_desc = ""
    if cluster and position:
        cluster_desc = POSITION_CLUSTER_DESCRIPTIONS.get(position, {}).get(cluster, "")

    st.markdown(f"""
    <div style="background:white;border:1px solid #dee2e6;border-top:4px solid {color};
                border-radius:8px;padding:16px;margin-bottom:14px;">
        <div style="font-size:0.7rem;color:#6c757d;font-weight:700;text-transform:uppercase;">{title}</div>
        <div style="font-size:1.25rem;font-weight:700;color:#212529;margin-top:6px;">{row.get('Name','?')}</div>
        <div style="font-size:0.85rem;color:#6c757d;margin-top:4px;">
            {row.get('tm_club','N/D')} · {fmt_liga(row.get('liga','N/D'))}
        </div>
        <div style="margin-top:10px;display:flex;gap:6px;flex-wrap:wrap;">
            <span style="background:#F3F4F6;border-radius:999px;padding:3px 10px;font-size:0.75rem;">{row.get('posicion','N/D')}</span>
            <span style="background:#F3F4F6;border-radius:999px;padding:3px 10px;font-size:0.75rem;">Edad: {row.get('edad','N/D')}</span>
            <span style="background:#F3F4F6;border-radius:999px;padding:3px 10px;font-size:0.75rem;">💰 {row.get('valor_mercado','N/D')}</span>
            <span style="background:#F3F4F6;border-radius:999px;padding:3px 10px;font-size:0.75rem;">📅 {row.get('fin_contrato','N/D')}</span>
            <span style="background:#F3F4F6;border-radius:999px;padding:3px 10px;font-size:0.75rem;">⏱ {row.get('minutes_played','N/D')}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if cluster:
        st.markdown(f"""
        <div style="margin-top:-8px;margin-bottom:14px;background:#EEF2FF;border-left:3px solid #3730A3;
                    border-radius:4px;padding:8px 12px;">
            <div style="font-size:0.7rem;color:#3730A3;font-weight:700;text-transform:uppercase;">Perfil táctico</div>
            <div style="font-size:0.95rem;font-weight:700;color:#1e1b4b;">{cluster}</div>
            <div style="font-size:0.75rem;color:#4338CA;margin-top:2px;">{cluster_desc}</div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# RADARES
# ============================================================================

def _build_general_radar(df: pd.DataFrame, pa: pd.Series, pb: pd.Series) -> go.Figure:
    """
    Radar de 5 ejes (uno por categoría).
    Cada eje = media de percentiles globales de sus métricas disponibles.
    """
    cat_names = list(KPI_CATEGORIES.keys())
    scores_a, scores_b = [], []

    for cat, metrics in KPI_CATEGORIES.items():
        sa, sb = [], []
        for col, _, inv in metrics:
            if col not in df.columns:
                continue
            va = _safe_float(pa.get(col, 0))
            vb = _safe_float(pb.get(col, 0))
            sa.append(_global_percentile(df, col, va, inv))
            sb.append(_global_percentile(df, col, vb, inv))
        scores_a.append(float(np.mean(sa)) if sa else 0.0)
        scores_b.append(float(np.mean(sb)) if sb else 0.0)

    name_a = str(pa.get("Name", "Jugador A"))
    name_b = str(pb.get("Name", "Jugador B"))

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=scores_a + [scores_a[0]],
        theta=cat_names + [cat_names[0]],
        fill="toself", name=name_a,
        line=dict(color=COLOR_A, width=2.5),
        fillcolor=FILL_A,
    ))
    fig.add_trace(go.Scatterpolar(
        r=scores_b + [scores_b[0]],
        theta=cat_names + [cat_names[0]],
        fill="toself", name=name_b,
        line=dict(color=COLOR_B, width=2.5),
        fillcolor=FILL_B,
    ))
    fig.update_layout(
        title=dict(text="Radar General · Percentil global por categoría",
                   font=dict(size=15, color="#212529"), x=0.5),
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100],
                            tickvals=[20, 40, 60, 80, 100],
                            tickfont=dict(size=9, color="#6c757d"),
                            gridcolor="#dee2e6"),
            angularaxis=dict(tickfont=dict(size=13, color="#212529")),
            bgcolor="rgba(248,249,250,0.6)",
        ),
        showlegend=True,
        legend=dict(orientation="h", y=-0.14, x=0.5, xanchor="center"),
        height=520,
        margin=dict(t=70, b=70, l=70, r=70),
        paper_bgcolor="white",
    )
    return fig


def _build_category_radar(
    df: pd.DataFrame,
    pa: pd.Series,
    pb: pd.Series,
    cat: str,
) -> go.Figure:
    """Radar detallado de una categoría con percentiles globales."""
    metrics = KPI_CATEGORIES[cat]
    labels, sa, sb = [], [], []

    for col, lbl, inv in metrics:
        if col not in df.columns:
            continue
        va = _safe_float(pa.get(col, 0))
        vb = _safe_float(pb.get(col, 0))
        labels.append(lbl)
        sa.append(_global_percentile(df, col, va, inv))
        sb.append(_global_percentile(df, col, vb, inv))

    if not labels:
        fig = go.Figure()
        fig.add_annotation(text="Sin datos disponibles", showarrow=False)
        return fig

    name_a = str(pa.get("Name", "Jugador A"))
    name_b = str(pb.get("Name", "Jugador B"))

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=sa + [sa[0]], theta=labels + [labels[0]],
        fill="toself", name=name_a,
        line=dict(color=COLOR_A, width=2), fillcolor=FILL_A,
    ))
    fig.add_trace(go.Scatterpolar(
        r=sb + [sb[0]], theta=labels + [labels[0]],
        fill="toself", name=name_b,
        line=dict(color=COLOR_B, width=2), fillcolor=FILL_B,
    ))
    fig.update_layout(
        title=dict(text=f"{cat} · Metricas detalladas (percentil global)",
                   font=dict(size=14, color="#212529"), x=0.5),
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100],
                            tickvals=[20, 40, 60, 80, 100],
                            tickfont=dict(size=9, color="#6c757d"),
                            gridcolor="#dee2e6"),
            angularaxis=dict(tickfont=dict(size=11, color="#212529")),
            bgcolor="rgba(248,249,250,0.6)",
        ),
        showlegend=True,
        legend=dict(orientation="h", y=-0.14, x=0.5, xanchor="center"),
        height=480,
        margin=dict(t=60, b=60, l=60, r=60),
        paper_bgcolor="white",
    )
    return fig


def _category_table(
    df: pd.DataFrame,
    pa: pd.Series,
    pb: pd.Series,
    cat: str,
) -> pd.DataFrame:
    """Tabla valor real + percentil global para la categoría seleccionada."""
    name_a = str(pa.get("Name", "Jugador A"))
    name_b = str(pb.get("Name", "Jugador B"))
    rows = []
    for col, lbl, inv in KPI_CATEGORIES[cat]:
        if col not in df.columns:
            continue
        va = _safe_float(pa.get(col, 0))
        vb = _safe_float(pb.get(col, 0))
        rows.append({
            "Metrica": lbl,
            f"{name_a}": round(va, 2),
            f"{name_a} pctl": int(_global_percentile(df, col, va, inv)),
            f"{name_b}": round(vb, 2),
            f"{name_b} pctl": int(_global_percentile(df, col, vb, inv)),
        })
    return pd.DataFrame(rows)


def _color_pct(val):
    try:
        v = int(val)
    except Exception:
        return ""
    if v >= 75:
        return "background-color:#d4edda;color:#155724;font-weight:600"
    elif v >= 50:
        return "background-color:#fff3cd;color:#856404;"
    return "background-color:#f8d7da;color:#721c24;"


# ============================================================================
# RENDER PRINCIPAL
# ============================================================================

def render_comparador_perfiles_tab(df: pd.DataFrame):

    st.markdown("""
    <div style="background:linear-gradient(135deg,#1A1A2E 0%,#16213E 50%,#0F3460 100%);
                padding:1.5rem 2rem;border-radius:14px;margin-bottom:1.5rem;border-left:5px solid #E30613;">
        <div style="font-size:2rem;color:white;font-weight:700;">COMPARADOR DE PERFILES</div>
        <div style="color:rgba(255,255,255,0.6);font-size:0.82rem;text-transform:uppercase;margin-top:0.25rem;">
            Analisis comparativo · Percentiles globales · Perfiles tacticos
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Selección de jugadores ────────────────────────────────────────────────
    comp1, comp2 = st.columns(2)

    with comp1:
        source_1 = st.radio("Fuente Jugador 1", SOURCE_OPTIONS, horizontal=True, key="cmp_src_1")
        options_1 = _get_player_options(df, source_1)
        if not options_1:
            st.warning(f"No hay jugadores en '{source_1}'.")
            return
        jugador_1 = st.selectbox("Jugador 1", options_1, key="cmp_p1")

    with comp2:
        source_2 = st.radio("Fuente Jugador 2", SOURCE_OPTIONS, horizontal=True, key="cmp_src_2")
        options_2 = _get_player_options(df, source_2, exclude=jugador_1)
        if not options_2:
            st.warning(f"No hay jugadores en '{source_2}' para comparar.")
            return
        jugador_2 = st.selectbox("Jugador 2", options_2, key="cmp_p2")

    if not jugador_1 or not jugador_2:
        st.info("Seleccione ambos jugadores.")
        return

    row_1 = _get_player_row(df, jugador_1)
    row_2 = _get_player_row(df, jugador_2)

    if row_1 is None or row_2 is None:
        st.warning("No se pudo localizar alguno de los jugadores.")
        return

    # ── Fichas ───────────────────────────────────────────────────────────────
    st.markdown("---")
    top_l, top_r = st.columns(2)
    with top_l:
        _render_player_card(row_1, "Jugador 1", COLOR_A, df)
    with top_r:
        _render_player_card(row_2, "Jugador 2", COLOR_B, df)

    # ── Radar General ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Radar General")
    st.markdown(
        "<p style='color:#6c757d;font-size:13px;margin-top:-6px;'>"
        "Vista global: cada eje representa la media de percentiles globales "
        "de todas las metricas de esa categoria.</p>",
        unsafe_allow_html=True,
    )

    col_gen, col_exec = st.columns([2, 1], gap="large")

    with col_gen:
        fig_gen = _build_general_radar(df, row_1, row_2)
        st.plotly_chart(fig_gen, use_container_width=True)
        st.caption("Percentiles globales (0-100) respecto a todo el dataset.")

    with col_exec:
        st.markdown("#### Lectura ejecutiva")

        if "Average Sofascore Rating" in df.columns:
            r1 = _safe_float(row_1.get("Average Sofascore Rating", 0))
            r2 = _safe_float(row_2.get("Average Sofascore Rating", 0))
            st.metric(jugador_1, f"{r1:.2f}")
            st.metric(jugador_2, f"{r2:.2f}", f"{r2 - r1:+.2f}")

        if "Accurate passes %" in df.columns:
            p1 = _safe_float(row_1.get("Accurate passes %", 0))
            p2 = _safe_float(row_2.get("Accurate passes %", 0))
            st.markdown("**Precision de pase**")
            st.progress(min(int(p1), 100), text=f"{jugador_1}: {p1:.1f}%")
            st.progress(min(int(p2), 100), text=f"{jugador_2}: {p2:.1f}%")

        st.markdown("---")
        if st.button("Añadir Jugador 2 a cartera", key="cmp_add_j2_v4", use_container_width=True):
            if "watchlist" not in st.session_state:
                st.session_state.watchlist = []
            if jugador_2 not in st.session_state.watchlist:
                st.session_state.watchlist.append(jugador_2)
                st.success(f"{jugador_2} añadido a cartera.")
            else:
                st.info(f"{jugador_2} ya estaba en cartera.")

    # ── Radares detallados por categoría ──────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Radar Detallado por Categoria")
    st.markdown(
        "<p style='color:#6c757d;font-size:13px;margin-top:-6px;'>"
        "Selecciona una categoria para ver el radar con todas sus metricas individuales.</p>",
        unsafe_allow_html=True,
    )

    selected_cat = st.radio(
        "Categoria",
        list(KPI_CATEGORIES.keys()),
        format_func=lambda c: CAT_ICONS.get(c, c),
        horizontal=True,
        key="comp_cat_radio",
    )

    col_det, col_tbl = st.columns([3, 2], gap="large")

    with col_det:
        fig_det = _build_category_radar(df, row_1, row_2, selected_cat)
        st.plotly_chart(fig_det, use_container_width=True)

    with col_tbl:
        st.markdown(f"**Metricas · {selected_cat}**")
        tbl = _category_table(df, row_1, row_2, selected_cat)
        if not tbl.empty:
            pct_cols = [c for c in tbl.columns if "pctl" in c]
            styled = tbl.style.map(_color_pct, subset=pct_cols)
            st.dataframe(styled, use_container_width=True, hide_index=True)
        else:
            st.info("Sin datos disponibles para esta categoria.")

    st.caption(
        "Verde >= 75 · Amarillo >= 50 · Rojo < 50  |  "
        "Metricas negativas (errores, faltas) invertidas: mayor percentil = mejor."
    )

    # ── Tabla comparativa completa ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Tabla comparativa general")

    rows = []
    for col, label in COMPARISON_METRICS:
        if col not in df.columns:
            continue
        va = row_1.get(col)
        vb = row_2.get(col)
        rows.append({
            "Metrica": label,
            jugador_1: round(_safe_float(va), 2) if pd.notna(va) else None,
            jugador_2: round(_safe_float(vb), 2) if pd.notna(vb) else None,
            "Diferencia": round(_safe_float(vb) - _safe_float(va), 2)
                          if (pd.notna(va) and pd.notna(vb)) else None,
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Jugadores similares ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Jugadores similares")

    sim_col1, sim_col2 = st.columns(2)

    with sim_col1:
        st.markdown(f"**Similares a {jugador_1}**")
        pos_1 = validate_position(row_1.get("posicion"))
        if pos_1:
            sim_1 = find_similar_players(df, jugador_1, pos_1, top_n=5)
            if not sim_1.empty:
                cols = [c for c in ["Name", "tm_club", "cluster_label", "similarity"] if c in sim_1.columns]
                st.dataframe(sim_1[cols], use_container_width=True, hide_index=True)
            else:
                st.caption("No se encontraron similares.")
        else:
            st.caption("Posicion no reconocida.")

    with sim_col2:
        st.markdown(f"**Similares a {jugador_2}**")
        pos_2 = validate_position(row_2.get("posicion"))
        if pos_2:
            sim_2 = find_similar_players(df, jugador_2, pos_2, top_n=5)
            if not sim_2.empty:
                cols = [c for c in ["Name", "tm_club", "cluster_label", "similarity"] if c in sim_2.columns]
                st.dataframe(sim_2[cols], use_container_width=True, hide_index=True)
            else:
                st.caption("No se encontraron similares.")
        else:
            st.caption("Posicion no reconocida.")