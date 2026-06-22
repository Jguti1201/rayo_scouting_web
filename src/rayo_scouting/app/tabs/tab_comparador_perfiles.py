"""
gemini_tab_comparador_perfiles.py
==================================
Comparador de perfiles v3 con:
- Clustering integrado (muestra perfil/cluster de cada jugador)
- Radar con percentiles POSICIONALES (no globales)
- Búsqueda de similares al jugador seleccionado
- Presentación ejecutiva mejorada
"""

from __future__ import annotations

import logging
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from rayo_scouting.scouting.dashboard_metrics import fmt_liga, get_rayo_df, get_watchlist_df
from rayo_scouting.features.clustering import (
    compute_positional_percentile,
    find_similar_players,
    get_cluster_for_player,
    POSITION_CLUSTER_DESCRIPTIONS,
    CLUSTER_COLORS,
)

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIG
# ============================================================================

COMPARISON_METRICS = [
    ("Goals", "Goles"),
    ("Assists", "Asistencias"),
    ("Expected goals (xG)", "xG"),
    ("Key passes", "Pases clave"),
    ("Succ. dribbles", "Regates"),
    ("Tackles_p90", "Tackles p90"),
    ("Interceptions_p90", "Intercepciones p90"),
    ("Clearances_p90", "Despejes p90"),
    ("Accurate passes %", "Precisión pase (%)"),
    ("ground_duels_won_pct", "Duelos suelo (%)"),
    ("aerial_duels_won_pct", "Duelos aéreos (%)"),
    ("Average Sofascore Rating", "Rating"),
    ("minutes_played", "Minutos"),
]

SOURCE_OPTIONS = ["Plantilla Rayo", "Cartera", "Base completa"]

POSITION_RADAR_MAP = {
    "portero": [
        ("Total_saves_p90", "Paradas p90"),
        ("Saves_from_inside_box_p90", "Paradas área"),
        ("Clean sheets", "Port. a 0"),
        ("Runs out", "Salidas"),
        ("Accurate passes %", "Prec. pase"),
        ("Average Sofascore Rating", "Rating"),
    ],
    "defensa": [
        ("Tackles_p90", "Tackles"),
        ("Interceptions_p90", "Intercep."),
        ("Clearances_p90", "Despejes"),
        ("aerial_duels_won_pct", "Aéreos %"),
        ("ground_duels_won_pct", "Suelo %"),
        ("Accurate passes %", "Prec. pase"),
    ],
    "medio": [
        ("Key_passes_p90", "P. clave"),
        ("Big_chances_created_p90", "Ocasiones"),
        ("Accurate passes %", "Prec. pase"),
        ("Tackles_p90", "Tackles"),
        ("Interceptions_p90", "Intercep."),
        ("Goals_p90", "Goles p90"),
    ],
    "delantero": [
        ("Goals_p90", "Goles p90"),
        ("xG_p90", "xG p90"),
        ("Total_shots_p90", "Tiros p90"),
        ("Succ_dribbles_p90", "Regates p90"),
        ("Key_passes_p90", "P. clave"),
        ("Goal conversion %", "Conv. gol"),
    ],
    "general": [
        ("Average Sofascore Rating", "Rating"),
        ("Accurate passes %", "Prec. pase"),
        ("ground_duels_won_pct", "Duelos suelo"),
        ("Goals_p90", "Goles p90"),
        ("Key_passes_p90", "P. clave"),
        ("Interceptions_p90", "Intercep."),
    ],
}


# ============================================================================
# HELPERS
# ============================================================================

def _safe_float(v):
    if pd.isna(v):
        return 0.0
    try:
        return float(v)
    except Exception:
        return 0.0


def _normalize_position(pos: str) -> str:
    pos = str(pos).strip().lower()
    if "portero" in pos:
        return "portero"
    if any(x in pos for x in ["central", "defensa", "lateral", "carrilero"]):
        return "defensa"
    if any(x in pos for x in ["mediocentro", "medio", "interior", "pivote", "mediapunta"]):
        return "medio"
    if any(x in pos for x in ["delantero", "extremo", "segundo punta"]):
        return "delantero"
    return "general"


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
# RADAR
# ============================================================================

def _get_radar_categories(player_a: pd.Series, player_b: pd.Series, df: pd.DataFrame):
    pos_a = _normalize_position(player_a.get("posicion", ""))
    pos_b = _normalize_position(player_b.get("posicion", ""))

    if pos_a == pos_b:
        selected = POSITION_RADAR_MAP.get(pos_a, POSITION_RADAR_MAP["general"])
        radar_type = f"Radar específico: {pos_a.capitalize()}"
        position_filter = player_a.get("posicion", "")
    else:
        selected = POSITION_RADAR_MAP["general"]
        radar_type = "Radar general (demarcaciones distintas)"
        position_filter = None

    selected = [(col, lbl) for col, lbl in selected if col in df.columns]
    return selected, radar_type, position_filter


def _build_radar(player_a: pd.Series, player_b: pd.Series, df: pd.DataFrame):
    categories, radar_type, pos_filter = _get_radar_categories(player_a, player_b, df)

    labels, vals_a, vals_b = [], [], []

    for col, lbl in categories:
        raw_a = _safe_float(player_a.get(col, 0))
        raw_b = _safe_float(player_b.get(col, 0))

        if pos_filter:
            pct_a = compute_positional_percentile(df, pos_filter, col, raw_a)
            pct_b = compute_positional_percentile(df, pos_filter, col, raw_b)
        else:
            series = pd.to_numeric(df[col], errors="coerce").dropna() if col in df.columns else pd.Series()
            pct_a = round(float((series <= raw_a).mean() * 100), 1) if not series.empty else 0
            pct_b = round(float((series <= raw_b).mean() * 100), 1) if not series.empty else 0

        labels.append(lbl)
        vals_a.append(pct_a)
        vals_b.append(pct_b)

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=vals_a, theta=labels, fill='toself',
        name=str(player_a.get("Name", "Jugador 1")),
        line_color='#E30613', fillcolor='rgba(227, 6, 19, 0.20)'
    ))
    fig.add_trace(go.Scatterpolar(
        r=vals_b, theta=labels, fill='toself',
        name=str(player_b.get("Name", "Jugador 2")),
        line_color='#1A1A2E', fillcolor='rgba(26, 26, 46, 0.20)'
    ))

    fig.update_layout(
        title=radar_type,
        polar=dict(radialaxis=dict(visible=True, range=[0, 100],
                                    tickvals=[20, 40, 60, 80, 100])),
        showlegend=True, height=540,
        margin=dict(t=80, b=40, l=40, r=40)
    )
    return fig


# ============================================================================
# FICHA DE JUGADOR
# ============================================================================

def _render_player_card(row: pd.Series, title: str, df: pd.DataFrame):
    position = str(row.get("posicion", ""))
    cluster = get_cluster_for_player(df, str(row.get("Name", "")), position)
    cluster_desc = ""
    if cluster:
        cluster_desc = POSITION_CLUSTER_DESCRIPTIONS.get(position, {}).get(cluster, "")

    st.markdown(f"""
    <div style="background:white;border:1px solid #dee2e6;border-top:4px solid #212529;
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
            <div style="font-size:0.7rem;color:#3730A3;font-weight:700;text-transform:uppercase;">
                Perfil táctico
            </div>
            <div style="font-size:0.95rem;font-weight:700;color:#1e1b4b;">{cluster}</div>
            <div style="font-size:0.75rem;color:#4338CA;margin-top:2px;">{cluster_desc}</div>
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# RENDER PRINCIPAL
# ============================================================================

def render_comparador_perfiles_tab(df: pd.DataFrame):
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1A1A2E 0%,#16213E 50%,#0F3460 100%);
                padding:1.5rem 2rem;border-radius:14px;margin-bottom:1.5rem;border-left:5px solid #E30613;">
        <div style="font-size:2rem;color:white;font-weight:700;">⚔️ COMPARADOR DE PERFILES</div>
        <div style="color:rgba(255,255,255,0.6);font-size:0.82rem;text-transform:uppercase;margin-top:0.25rem;">
            Análisis comparativo con percentiles posicionales y perfiles tácticos
        </div>
    </div>
    """, unsafe_allow_html=True)

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

    st.markdown("---")

    top_l, top_r = st.columns(2)
    with top_l:
        _render_player_card(row_1, "Jugador 1", df)
    with top_r:
        _render_player_card(row_2, "Jugador 2", df)

    st.markdown("---")

    fig = _build_radar(row_1, row_2, df)

    col_graph, col_expl = st.columns([2, 1], gap="large")

    with col_graph:
        st.plotly_chart(fig, width="stretch")
        st.caption("Percentiles posicionales (0-100) respecto a jugadores de la misma demarcación.")

    with col_expl:
        st.markdown("#### Lectura ejecutiva")

        if "Average Sofascore Rating" in df.columns:
            r1 = _safe_float(row_1.get("Average Sofascore Rating", 0))
            r2 = _safe_float(row_2.get("Average Sofascore Rating", 0))
            st.metric(jugador_1, f"{r1:.2f}", label_visibility="visible")
            st.metric(jugador_2, f"{r2:.2f}", f"{r2 - r1:+.2f}")

        if "Accurate passes %" in df.columns:
            p1 = _safe_float(row_1.get("Accurate passes %", 0))
            p2 = _safe_float(row_2.get("Accurate passes %", 0))
            st.markdown("**Precisión de pase**")
            st.progress(min(int(p1), 100), text=f"{jugador_1}: {p1:.1f}%")
            st.progress(min(int(p2), 100), text=f"{jugador_2}: {p2:.1f}%")

        st.markdown("---")
        if st.button("Añadir Jugador 2 a cartera", key="cmp_add_j2_v3", width="stretch"):
            if "watchlist" not in st.session_state:
                st.session_state.watchlist = []
            if jugador_2 not in st.session_state.watchlist:
                st.session_state.watchlist.append(jugador_2)
                st.success(f"{jugador_2} añadido a cartera.")
            else:
                st.info(f"{jugador_2} ya estaba en cartera.")

    # ── Tabla comparativa ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Tabla comparativa")

    rows = []
    for col, label in COMPARISON_METRICS:
        if col not in df.columns:
            continue
        va = row_1.get(col)
        vb = row_2.get(col)
        rows.append({
            "Métrica": label,
            "Jugador 1": round(_safe_float(va), 2) if pd.notna(va) else "N/D",
            "Jugador 2": round(_safe_float(vb), 2) if pd.notna(vb) else "N/D",
            "Diferencia": round(_safe_float(vb) - _safe_float(va), 2) if pd.notna(va) or pd.notna(vb) else "N/D",
        })

    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # ── Similares ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Jugadores similares")

    sim_col1, sim_col2 = st.columns(2)

    with sim_col1:
        st.markdown(f"**Similares a {jugador_1}**")
        pos_1 = str(row_1.get("posicion", ""))
        if pos_1:
            sim_1 = find_similar_players(df, jugador_1, pos_1, top_n=5)
            if not sim_1.empty:
                cols = [c for c in ["Name", "tm_club", "cluster_label", "similarity"] if c in sim_1.columns]
                st.dataframe(sim_1[cols], width="stretch", hide_index=True)
            else:
                st.caption("No se encontraron similares.")

    with sim_col2:
        st.markdown(f"**Similares a {jugador_2}**")
        pos_2 = str(row_2.get("posicion", ""))
        if pos_2:
            sim_2 = find_similar_players(df, jugador_2, pos_2, top_n=5)
            if not sim_2.empty:
                cols = [c for c in ["Name", "tm_club", "cluster_label", "similarity"] if c in sim_2.columns]
                st.dataframe(sim_2[cols], width="stretch", hide_index=True)
            else:
                st.caption("No se encontraron similares.")