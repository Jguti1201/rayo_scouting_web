"""
tab_analisis_plantilla.py
=========================
Pestaña de análisis completo de plantilla del Rayo Vallecano.
Diseñada para que el director deportivo tenga una visión ejecutiva
de la situación contractual, salarial, deportiva y de necesidades.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from rayo_scouting.scouting.dashboard_metrics import get_rayo_df, fmt_liga

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIG
# ============================================================================

RAYO_RED = "#E30613"
RAYO_DARK = "#1A1A2E"
RAYO_GOLD = "#D4A843"

SALARY_DATA_RAW = """Augusto Batalla,K,GK,29,Argentina,40000,2080000,520000,2600000,Jul 1 2025,Jun 30 2030,5,10400000,,Active,Starter
Florian Lejeune,D,CB,34,Francia,40000,2080000,520000,2600000,May 4 2026,Jun 30 2028,3,6240000,,Active,Starter
Isi Palazón,F,SS,31,España,40000,2080000,520000,2600000,May 8 2023,Jun 30 2028,3,6240000,50000000,Active,Starter
Álvaro García,F,LW,33,España,40000,2080000,520000,2600000,Feb 11 2025,Jun 30 2028,3,6240000,,Active,Reserve
Luiz Felipe,D,CB,28,Italia,40000,2080000,520000,2600000,Jul 7 2025,Jun 30 2026,1,2080000,,Active,Reserve
Óscar Valentín,M,CM,31,España,36154,1880000,480000,2360000,Sep 13 2023,Jun 30 2027,2,3760000,,Active,Starter
Andrei Rațiu,D,RB,27,Rumanía,36154,1880000,420000,2300000,Nov 21 2025,Jun 30 2030,5,9400000,,Active,Starter
Jorge de Frutos,F,RW,28,España,36154,1880000,480000,2360000,Aug 17 2023,Jun 30 2028,3,5640000,,Active,Starter
Unai López,M,CM,30,España,32115,1670000,420000,2090000,Jul 1 2024,Jun 30 2026,1,1670000,,Active,Reserve
Pep Chavarría,D,LB,27,España,32115,1670000,420000,2090000,Jun 16 2025,Jun 30 2030,5,8350000,,Active,Reserve
Pedro Díaz,M,CM,27,España,30000,1560000,400000,1960000,Aug 5 2024,Jun 30 2028,3,4680000,,Active,Reserve
Alexandre Alemão,F,CF,27,Brasil,28077,1460000,380000,1840000,Sep 1 2025,Jun 30 2030,5,7300000,,Active,Reserve
Randy Nteka,F,AM,28,Angola,24038,1250000,310000,1560000,Feb 13 2025,Jun 30 2028,3,3750000,,Active,Reserve
Óscar Trejo,F,AM,37,Argentina,20000,1040000,250000,1290000,Jul 8 2025,Jun 30 2026,1,1040000,,Active,Reserve
Iván Balliu,D,RB,34,Albania,18077,940000,230000,1170000,Feb 27 2025,Jun 30 2027,2,1880000,,Active,Reserve
Alfonso Espino,D,LB,34,Uruguay,18077,940000,230000,1170000,Jul 17 2023,Jun 30 2026,1,940000,,Active,Starter
Carlos Martín,F,LW,23,España,15962,830000,210000,1040000,Jan 2 2026,Jun 30 2026,1,830000,,Loan,Reserve
Ilias Akhomach,F,RW,21,Marruecos,15962,830000,210000,1040000,Jan 21 2026,Jun 30 2026,1,830000,,Loan,Starter
Gerard Gumbau,M,DM,31,España,15962,830000,210000,1040000,Jul 16 2025,Jun 30 2026,1,830000,,Active,Starter
Sergio Camello,F,CF,24,España,15962,830000,210000,1040000,Aug 17 2023,Jun 30 2027,2,1660000,,Active,Reserve
Fran Pérez,F,RW,23,España,14038,730000,190000,920000,Aug 15 2025,Jun 30 2029,4,2920000,,Active,Starter
Dani Cárdenas,K,GK,28,España,12115,630000,150000,780000,Aug 18 2023,Jun 30 2027,2,1260000,,Active,Reserve
Pathé Ciss,M,CM,31,Senegal,12115,630000,150000,780000,Jul 28 2021,Jun 30 2027,2,1260000,,Active,Reserve
Nobel Mendy,D,CB,21,Senegal,12115,630000,150000,780000,Aug 20 2025,Jun 30 2026,1,630000,,Active,Starter
Abdul Mumin,D,CB,27,Ghana,10000,520000,130000,650000,Sep 1 2022,Jun 30 2026,1,520000,,Active,Reserve
Jozhua Vertrouwd,D,CB,21,Países Bajos,5962,310000,80000,390000,Aug 12 2025,Jun 30 2026,1,310000,,Loan,Reserve
Samu Becerra,M,CM,19,España,3654,190000,42000,232000,Jul 1 2024,Jun 30 2027,2,380000,,Active,Reserve"""

SALARY_COLUMNS = [
    "Jugador", "Pos_code", "Pos_detail", "Edad_sal", "Pais",
    "Salario_semanal", "Salario_fijo_anual", "Bonus_anual", "Total_anual",
    "Fichado", "Vencimiento", "Anos_restantes", "Importe_restante",
    "Clausula", "Estado", "Rol",
]

POS_DETAIL_MAP = {
    "GK": "Portero",
    "CB": "Central",
    "RB": "Lateral Derecho",
    "LB": "Lateral Izquierdo",
    "DM": "Mediocentro Defensivo",
    "CM": "Centrocampista",
    "AM": "Mediapunta",
    "SS": "Segundo Punta",
    "CF": "Delantero Centro",
    "RW": "Extremo Derecho",
    "LW": "Extremo Izquierdo",
}

POS_CODE_MAP = {
    "K": "Portero",
    "D": "Defensa",
    "M": "Centrocampista",
    "F": "Delantero",
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
        "Ã±": "ñ", "Ã¼": "ü", "Ã": "Á", "Ã‰": "É", "Ã": "Í",
        "Ã“": "Ó", "Ãš": "Ú", "Ã'": "Ñ", "Ãœ": "Ü", "â‚¬": "€",
        "Â": "", "È›": "ț", "È™": "ș",
    }
    for bad, good in replacements.items():
        s = s.replace(bad, good)
    return s.strip()


def _fmt_eur(value) -> str:
    if pd.isna(value) or value == 0:
        return "-"
    try:
        v = float(value)
        if v >= 1_000_000:
            return f"{v / 1_000_000:.2f}M €"
        elif v >= 1_000:
            return f"{v / 1_000:.0f}K €"
        return f"{v:.0f} €"
    except Exception:
        return str(value)


def _load_salary_df() -> pd.DataFrame:
    rows = []
    for line in SALARY_DATA_RAW.strip().split("\n"):
        parts = line.split(",")
        if len(parts) == len(SALARY_COLUMNS):
            rows.append(parts)
        else:
            logger.warning("Línea con columnas incorrectas: %s", line[:50])

    df = pd.DataFrame(rows, columns=SALARY_COLUMNS)

    for col in ["Salario_semanal", "Salario_fijo_anual", "Bonus_anual", "Total_anual", "Importe_restante", "Clausula"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df["Anos_restantes"] = pd.to_numeric(df["Anos_restantes"], errors="coerce").fillna(0).astype(int)
    df["Edad_sal"] = pd.to_numeric(df["Edad_sal"], errors="coerce").fillna(0).astype(int)

    df["Jugador"] = df["Jugador"].apply(_fix_text)
    df["Pais"] = df["Pais"].apply(_fix_text)

    df["Pos_display"] = df["Pos_detail"].map(POS_DETAIL_MAP).fillna(df["Pos_detail"])
    df["Pos_grupo"] = df["Pos_code"].map(POS_CODE_MAP).fillna("Otro")

    return df


def _merge_salary_with_stats(salary_df: pd.DataFrame, rayo_df: pd.DataFrame) -> pd.DataFrame:
    if rayo_df.empty:
        return salary_df

    def _norm(name):
        s = _fix_text(name).lower()
        s = unicodedata.normalize("NFD", s)
        s = "".join(c for c in s if unicodedata.category(c) != "Mn")
        return re.sub(r"\s+", " ", s).strip()

    salary_df["_norm"] = salary_df["Jugador"].apply(_norm)
    rayo_df = rayo_df.copy()
    rayo_df["_norm"] = rayo_df["Name"].apply(_norm)

    stats_cols = [
        "Average Sofascore Rating", "minutes_played", "Goals", "Assists",
        "valor_mercado", "muestra_fiable",
    ]
    available = [c for c in stats_cols if c in rayo_df.columns]

    merged = salary_df.merge(
        rayo_df[["_norm"] + available],
        on="_norm",
        how="left",
        suffixes=("", "_stats"),
    )

    merged = merged.drop(columns=["_norm"], errors="ignore")
    return merged


# ============================================================================
# CHARTS
# ============================================================================

def _chart_salary_by_player(df: pd.DataFrame) -> go.Figure:
    df_sorted = df.sort_values("Total_anual", ascending=True)

    colors = [RAYO_RED if r == "Starter" else RAYO_DARK for r in df_sorted["Rol"]]

    fig = go.Figure(go.Bar(
        x=df_sorted["Total_anual"],
        y=df_sorted["Jugador"],
        orientation="h",
        marker_color=colors,
        text=df_sorted["Total_anual"].apply(lambda x: f"{x/1000:.0f}K"),
        textposition="outside",
    ))

    fig.update_layout(
        title="Masa salarial por jugador (total anual)",
        xaxis_title="Total anual (EUR)",
        yaxis_title="",
        height=max(500, len(df_sorted) * 22),
        margin=dict(l=140, r=60, t=50, b=40),
        plot_bgcolor="white",
    )
    return fig


def _chart_salary_by_position(df: pd.DataFrame) -> go.Figure:
    grouped = df.groupby("Pos_grupo")["Total_anual"].sum().sort_values(ascending=False).reset_index()

    fig = go.Figure(go.Pie(
        labels=grouped["Pos_grupo"],
        values=grouped["Total_anual"],
        marker_colors=[RAYO_RED, RAYO_DARK, RAYO_GOLD, "#6c757d"],
        textinfo="label+percent",
        hole=0.4,
    ))
    fig.update_layout(
        title="Distribución salarial por demarcación",
        height=400,
    )
    return fig


def _chart_contract_timeline(df: pd.DataFrame) -> go.Figure:
    grouped = df.groupby("Anos_restantes")["Jugador"].count().reset_index()
    grouped.columns = ["Años restantes", "Jugadores"]

    colors = []
    for y in grouped["Años restantes"]:
        if y <= 1:
            colors.append(RAYO_RED)
        elif y <= 2:
            colors.append(RAYO_GOLD)
        else:
            colors.append("#28a745")

    fig = go.Figure(go.Bar(
        x=grouped["Años restantes"],
        y=grouped["Jugadores"],
        marker_color=colors,
        text=grouped["Jugadores"],
        textposition="outside",
    ))
    fig.update_layout(
        title="Jugadores por años de contrato restante",
        xaxis_title="Años restantes",
        yaxis_title="Nº jugadores",
        height=350,
        plot_bgcolor="white",
    )
    return fig


def _chart_age_distribution(df: pd.DataFrame) -> go.Figure:
    bins = [0, 22, 25, 28, 31, 35, 50]
    labels = ["Sub-22", "23-25", "26-28", "29-31", "32-35", "36+"]
    df = df.copy()
    df["Rango_edad"] = pd.cut(df["Edad_sal"], bins=bins, labels=labels, right=True)

    grouped = df.groupby("Rango_edad", observed=True)["Jugador"].count().reset_index()
    grouped.columns = ["Rango", "Jugadores"]

    fig = go.Figure(go.Bar(
        x=grouped["Rango"],
        y=grouped["Jugadores"],
        marker_color=RAYO_DARK,
        text=grouped["Jugadores"],
        textposition="outside",
    ))
    fig.update_layout(
        title="Distribución por edad",
        xaxis_title="Rango de edad",
        yaxis_title="Jugadores",
        height=350,
        plot_bgcolor="white",
    )
    return fig


def _chart_starters_vs_reserves(df: pd.DataFrame) -> go.Figure:
    grouped = df.groupby(["Pos_grupo", "Rol"]).size().reset_index(name="Count")

    fig = go.Figure()
    for rol in ["Starter", "Reserve", "Loan"]:
        sub = grouped[grouped["Rol"] == rol]
        color = RAYO_RED if rol == "Starter" else (RAYO_GOLD if rol == "Loan" else RAYO_DARK)
        fig.add_trace(go.Bar(
            x=sub["Pos_grupo"],
            y=sub["Count"],
            name=rol,
            marker_color=color,
            text=sub["Count"],
            textposition="outside",
        ))

    fig.update_layout(
        title="Titulares vs Suplentes vs Cedidos por demarcación",
        barmode="group",
        height=400,
        plot_bgcolor="white",
    )
    return fig


# ============================================================================
# ANALYSIS
# ============================================================================

def _get_contract_risks(df: pd.DataFrame) -> pd.DataFrame:
    risks = df[df["Anos_restantes"] <= 1].copy()
    risks = risks.sort_values("Total_anual", ascending=False)
    return risks


def _get_high_salary_low_role(df: pd.DataFrame) -> pd.DataFrame:
    median_sal = df["Total_anual"].median()
    inefficient = df[
        (df["Total_anual"] > median_sal) & (df["Rol"] == "Reserve")
    ].sort_values("Total_anual", ascending=False)
    return inefficient


def _get_position_gaps(df: pd.DataFrame) -> list[dict]:
    pos_counts = df.groupby("Pos_detail").size().to_dict()
    gaps = []

    ideal = {
        "GK": {"min": 2, "label": "Porteros"},
        "CB": {"min": 4, "label": "Centrales"},
        "RB": {"min": 2, "label": "Laterales derechos"},
        "LB": {"min": 2, "label": "Laterales izquierdos"},
        "CM": {"min": 3, "label": "Centrocampistas"},
        "DM": {"min": 2, "label": "Pivotes"},
        "AM": {"min": 1, "label": "Mediapuntas"},
        "RW": {"min": 2, "label": "Extremos derechos"},
        "LW": {"min": 2, "label": "Extremos izquierdos"},
        "CF": {"min": 2, "label": "Delanteros centro"},
        "SS": {"min": 1, "label": "Segundos puntas"},
    }

    for pos, config in ideal.items():
        current = pos_counts.get(pos, 0)
        status = "ok"
        if current < config["min"]:
            status = "deficit"
        elif current > config["min"] + 1:
            status = "exceso"

        gaps.append({
            "Posición": config["label"],
            "Código": pos,
            "Actual": current,
            "Ideal mínimo": config["min"],
            "Estado": status,
        })

    return gaps


def _get_squad_kpis(df: pd.DataFrame, rayo_df: pd.DataFrame) -> dict:
    total_salary = df["Total_anual"].sum()
    avg_salary = df["Total_anual"].mean()
    median_salary = df["Total_anual"].median()
    avg_age = df["Edad_sal"].mean()
    n_expiring = len(df[df["Anos_restantes"] <= 1])
    n_loans = len(df[df["Estado"] == "Loan"])
    n_starters = len(df[df["Rol"] == "Starter"])
    n_total = len(df)
    n_spanish = len(df[df["Pais"].str.contains("España", case=False, na=False)])

    avg_rating = None
    if not rayo_df.empty and "Average Sofascore Rating" in rayo_df.columns:
        avg_rating = pd.to_numeric(rayo_df["Average Sofascore Rating"], errors="coerce").mean()

    return {
        "total_salary": total_salary,
        "avg_salary": avg_salary,
        "median_salary": median_salary,
        "avg_age": avg_age,
        "n_expiring": n_expiring,
        "n_loans": n_loans,
        "n_starters": n_starters,
        "n_total": n_total,
        "n_spanish": n_spanish,
        "pct_spanish": round(n_spanish / n_total * 100, 1) if n_total else 0,
        "avg_rating": avg_rating,
    }


# ============================================================================
# RENDER
# ============================================================================

def render_analisis_plantilla_tab(df: pd.DataFrame):
    logger.warning("[PLANTILLA] Iniciando render con %d filas en dataset global", len(df))

    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #1A1A2E 0%, #16213E 50%, #0F3460 100%);
        padding: 1.5rem 2rem;
        border-radius: 14px;
        margin-bottom: 1.5rem;
        border-left: 5px solid #E30613;
    ">
        <div style="font-size:2rem;color:white;letter-spacing:2px;font-weight:700;">
            📋 ANÁLISIS DE PLANTILLA
        </div>
        <div style="color:rgba(255,255,255,0.6);font-size:0.82rem;text-transform:uppercase;margin-top:0.25rem;">
            Rayo Vallecano · Situación contractual, salarial y deportiva
        </div>
    </div>
    """, unsafe_allow_html=True)

    salary_df = _load_salary_df()
    rayo_df = get_rayo_df(df)
    merged_df = _merge_salary_with_stats(salary_df, rayo_df)
    kpis = _get_squad_kpis(salary_df, rayo_df)

    logger.warning("[PLANTILLA] Jugadores en salary_df: %d", len(salary_df))
    logger.warning("[PLANTILLA] Jugadores en rayo_df: %d", len(rayo_df))

    # ── KPIs ──────────────────────────────────────────────────────────────
    st.markdown("#### Indicadores clave de plantilla")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Plantilla", f"{kpis['n_total']} jugadores", f"{kpis['n_starters']} titulares")
    k2.metric("Masa salarial", _fmt_eur(kpis['total_salary']), "Total anual")
    k3.metric("Salario medio", _fmt_eur(kpis['avg_salary']), f"Mediana: {_fmt_eur(kpis['median_salary'])}")
    k4.metric("Edad media", f"{kpis['avg_age']:.1f} años", f"{kpis['pct_spanish']:.0f}% españoles")

    if kpis["avg_rating"]:
        k5.metric("Rating medio", f"{kpis['avg_rating']:.2f}", "SofaScore")
    else:
        k5.metric("Cedidos", f"{kpis['n_loans']}", "Jugadores en préstamo")

    # ── Alertas ───────────────────────────────────────────────────────────
    st.markdown("---")

    a1, a2, a3 = st.columns(3)

    with a1:
        st.markdown(f"""
        <div style="background:#FFF3CD;border-left:4px solid #D4A843;border-radius:8px;
                    padding:1rem;margin-bottom:1rem;">
            <div style="font-size:0.75rem;color:#856404;font-weight:700;text-transform:uppercase;">
                ⚠️ Contratos expiran 2026
            </div>
            <div style="font-size:2rem;font-weight:800;color:#856404;">{kpis['n_expiring']}</div>
            <div style="font-size:0.8rem;color:#856404;">jugadores con ≤1 año de contrato</div>
        </div>
        """, unsafe_allow_html=True)

    with a2:
        inefficient = _get_high_salary_low_role(salary_df)
        st.markdown(f"""
        <div style="background:#F8D7DA;border-left:4px solid #E30613;border-radius:8px;
                    padding:1rem;margin-bottom:1rem;">
            <div style="font-size:0.75rem;color:#721c24;font-weight:700;text-transform:uppercase;">
                🔴 Salario alto + Suplente
            </div>
            <div style="font-size:2rem;font-weight:800;color:#721c24;">{len(inefficient)}</div>
            <div style="font-size:0.8rem;color:#721c24;">jugadores por encima de mediana sin titularidad</div>
        </div>
        """, unsafe_allow_html=True)

    with a3:
        st.markdown(f"""
        <div style="background:#D4EDDA;border-left:4px solid #28a745;border-radius:8px;
                    padding:1rem;margin-bottom:1rem;">
            <div style="font-size:0.75rem;color:#155724;font-weight:700;text-transform:uppercase;">
                ✅ Cedidos activos
            </div>
            <div style="font-size:2rem;font-weight:800;color:#155724;">{kpis['n_loans']}</div>
            <div style="font-size:0.8rem;color:#155724;">jugadores en situación de préstamo</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Tabla principal ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Plantilla completa · Situación contractual y salarial")

    display_df = merged_df.copy()

    display_cols = [
        "Jugador", "Pos_display", "Edad_sal", "Pais", "Rol", "Estado",
        "Total_anual", "Salario_semanal", "Bonus_anual",
        "Vencimiento", "Anos_restantes", "Importe_restante", "Clausula",
    ]

    if "Average Sofascore Rating" in display_df.columns:
        display_cols.append("Average Sofascore Rating")
    if "minutes_played" in display_df.columns:
        display_cols.append("minutes_played")
    if "Goals" in display_df.columns:
        display_cols.append("Goals")
    if "Assists" in display_df.columns:
        display_cols.append("Assists")

    available = [c for c in display_cols if c in display_df.columns]
    show_df = display_df[available].copy()

    rename = {
        "Jugador": "Jugador",
        "Pos_display": "Posición",
        "Edad_sal": "Edad",
        "Pais": "País",
        "Rol": "Rol",
        "Estado": "Estado",
        "Total_anual": "Total anual",
        "Salario_semanal": "Sal. semanal",
        "Bonus_anual": "Bonus",
        "Vencimiento": "Vencimiento",
        "Anos_restantes": "Años rest.",
        "Importe_restante": "Imp. restante",
        "Clausula": "Cláusula",
        "Average Sofascore Rating": "Rating",
        "minutes_played": "Minutos",
        "Goals": "Goles",
        "Assists": "Asist.",
    }
    show_df = show_df.rename(columns=rename)

    st.dataframe(
        show_df.sort_values("Total anual", ascending=False),
        use_container_width=True,
        hide_index=True,
        height=600,
    )

    # ── Gráficos ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Análisis visual")

    tab_sal, tab_pos, tab_contract, tab_age, tab_roles = st.tabs([
        "💰 Masa salarial",
        "📊 Por demarcación",
        "📅 Contratos",
        "👤 Edad",
        "⚽ Titulares vs Suplentes",
    ])

    with tab_sal:
        st.plotly_chart(_chart_salary_by_player(salary_df), use_container_width=True)

    with tab_pos:
        st.plotly_chart(_chart_salary_by_position(salary_df), use_container_width=True)

    with tab_contract:
        st.plotly_chart(_chart_contract_timeline(salary_df), use_container_width=True)

    with tab_age:
        st.plotly_chart(_chart_age_distribution(salary_df), use_container_width=True)

    with tab_roles:
        st.plotly_chart(_chart_starters_vs_reserves(salary_df), use_container_width=True)

    # ── Riesgos contractuales ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🔴 Riesgos contractuales (≤1 año)")

    risks = _get_contract_risks(salary_df)
    if not risks.empty:
        for _, row in risks.iterrows():
            rol_color = RAYO_RED if row["Rol"] == "Starter" else RAYO_DARK
            loan_tag = " · 🔄 CEDIDO" if row["Estado"] == "Loan" else ""

            st.markdown(f"""
            <div style="background:white;border:1px solid #dee2e6;border-left:4px solid {rol_color};
                        border-radius:4px;padding:12px 16px;margin-bottom:10px;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div>
                        <span style="font-weight:700;font-size:1rem;color:#212529;">{row['Jugador']}</span>
                        <span style="font-size:0.8rem;color:#6c757d;"> · {row['Pos_display']} · {row['Edad_sal']} años{loan_tag}</span>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:0.85rem;font-weight:700;color:{rol_color};">{_fmt_eur(row['Total_anual'])}/año</div>
                        <div style="font-size:0.7rem;color:#6c757d;">Vence: {row['Vencimiento']}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("No hay jugadores con contrato expirando próximamente.")

    # ── Ineficiencias salariales ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### ⚠️ Ineficiencias salariales (salario > mediana + suplente)")

    inefficient = _get_high_salary_low_role(salary_df)
    if not inefficient.empty:
        for _, row in inefficient.iterrows():
            st.markdown(f"""
            <div style="background:#FFF3CD;border-left:4px solid #D4A843;border-radius:4px;
                        padding:10px 14px;margin-bottom:8px;">
                <span style="font-weight:700;">{row['Jugador']}</span>
                <span style="color:#856404;"> · {row['Pos_display']} · {row['Edad_sal']} años ·
                Salario: {_fmt_eur(row['Total_anual'])}/año · Contrato hasta {row['Vencimiento']}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.success("No se detectan ineficiencias salariales evidentes.")

    # ── Gaps posicionales ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📋 Mapa de necesidades posicionales")

    gaps = _get_position_gaps(salary_df)
    gaps_df = pd.DataFrame(gaps)

    for _, row in gaps_df.iterrows():
        if row["Estado"] == "deficit":
            icon = "🔴"
            color = "#dc3545"
            bg = "#F8D7DA"
        elif row["Estado"] == "exceso":
            icon = "🟡"
            color = "#856404"
            bg = "#FFF3CD"
        else:
            icon = "✅"
            color = "#155724"
            bg = "#D4EDDA"

        st.markdown(f"""
        <div style="background:{bg};border-radius:6px;padding:8px 14px;margin-bottom:6px;
                    display:flex;justify-content:space-between;align-items:center;">
            <div>
                <span style="font-size:1.1rem;">{icon}</span>
                <span style="font-weight:700;color:{color};margin-left:8px;">{row['Posición']}</span>
                <span style="color:{color};font-size:0.85rem;"> ({row['Código']})</span>
            </div>
            <div style="text-align:right;">
                <span style="font-weight:700;color:{color};">{row['Actual']}</span>
                <span style="color:{color};font-size:0.8rem;"> / {row['Ideal mínimo']} mín.</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Resumen ejecutivo ─────────────────────────────────────────────────
    # ── Resumen ejecutivo ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📝 Resumen ejecutivo para dirección deportiva")

    deficit_positions = [g for g in gaps if g["Estado"] == "deficit"]
    expiring_starters = risks[risks["Rol"] == "Starter"]

    decisions = []
    for _, row in expiring_starters.iterrows():
        decisions.append(f"Renovar o reemplazar a **{row['Jugador']}** ({row['Pos_display']}, {row['Edad_sal']} años, {_fmt_eur(row['Total_anual'])}/año)")

    for g in deficit_positions:
        decisions.append(f"Fichar refuerzo en **{g['Posición']}** (actual: {g['Actual']}, mínimo: {g['Ideal mínimo']})")

    for _, row in inefficient.head(3).iterrows():
        decisions.append(f"Evaluar continuidad de **{row['Jugador']}** (suplente, {_fmt_eur(row['Total_anual'])}/año)")

    if not decisions:
        decisions.append("No se detectan decisiones urgentes inmediatas")

    with st.container(border=True):
        st.markdown("**INFORME DE SITUACIÓN · PLANTILLA 25/26**")
        st.markdown(
            f"**1. Masa salarial:** {_fmt_eur(kpis['total_salary'])} anuales "
            f"(media {_fmt_eur(kpis['avg_salary'])}, mediana {_fmt_eur(kpis['median_salary'])})"
        )
        st.markdown(
            f"**2. Perfil de plantilla:** {kpis['n_total']} jugadores, "
            f"edad media {kpis['avg_age']:.1f} años, "
            f"{kpis['pct_spanish']:.0f}% españoles, "
            f"{kpis['n_loans']} cedidos"
        )
        st.markdown(
            f"**3. Riesgo contractual:** {kpis['n_expiring']} jugadores con contrato ≤1 año, "
            f"de los cuales **{len(expiring_starters)} son titulares**"
        )
        st.markdown(
            f"**4. Ineficiencias:** {len(inefficient)} jugadores cobran por encima de la mediana "
            f"sin ser titulares"
        )
        deficit_names = ', '.join([g['Posición'] for g in deficit_positions]) if deficit_positions else 'Ninguna'
        st.markdown(f"**5. Necesidades posicionales:** {len(deficit_positions)} posiciones con déficit: {deficit_names}")
        st.markdown("**6. Decisiones urgentes:**")
        for d in decisions:
            st.markdown(f"- {d}")