"""
tab_resumen_general.py
======================
Pestaña de resumen ejecutivo para la plataforma de scouting.
"""

from __future__ import annotations
import pandas as pd
import streamlit as st

from rayo_scouting.scouting.dashboard_metrics import (
    get_summary_metrics,
    get_watchlist_df,
    get_contract_opportunities,
    fmt_liga,
)
from rayo_scouting.scouting.report_generation import render_watchlist_reports


def render_resumen_general_tab(df: pd.DataFrame):

    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #1A1A2E 0%, #16213E 50%, #0F3460 100%);
        padding: 1.5rem 2rem;
        border-radius: 14px;
        margin-bottom: 1.5rem;
        border-left: 5px solid #E30613;
    ">
        <div style="font-size:2rem;color:white;letter-spacing:2px;font-weight:700;">
            🏠 PANEL DE CONTROL
        </div>
        <div style="color:rgba(255,255,255,0.6);font-size:0.82rem;text-transform:uppercase;margin-top:0.25rem;">
            Rayo Vallecano · Plataforma de Scouting · Resumen ejecutivo
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Datos ─────────────────────────────────────────────────────────────────
    summary = get_summary_metrics(df)
    watchlist_df = get_watchlist_df(df, st.session_state.get("watchlist", []))
    contract_df = get_contract_opportunities(df)

    # ── KPIs ─────────────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Perfiles Activos", f"{summary['n_players']:,}", help="Total de jugadores en la base de datos")
    k2.metric("Oportunidades", f"{summary['n_contract_opportunities']:,}", help="Jugadores con contrato ≤ 2026")
    k3.metric("En Cartera", f"{summary['n_watchlist']}", help="Jugadores en seguimiento activo")
    k4.metric("Jugadores Rayo", f"{summary['n_rayo']}", help="Plantilla detectada en la base")

    st.markdown("---")

    # ── Bloque principal: cartera + alertas ───────────────────────────────────
    col_left, col_right = st.columns([1.15, 1], gap="large")

    with col_left:
        st.markdown("#### 👁 Jugadores en seguimiento")

        if not watchlist_df.empty:
            cols_show = [c for c in ["Name", "tm_club", "liga", "posicion", "edad", "valor_mercado", "fin_contrato"] if c in watchlist_df.columns]
            df_show = watchlist_df[cols_show].copy()
            if "liga" in df_show.columns:
                df_show["liga"] = df_show["liga"].map(fmt_liga)
            df_show = df_show.rename(columns={
                "Name": "Jugador", "tm_club": "Club", "liga": "Liga",
                "posicion": "Posición", "edad": "Edad",
                "valor_mercado": "Valor", "fin_contrato": "Fin contrato",
            })
            st.dataframe(df_show, use_container_width=True, hide_index=True)
        else:
            with st.container(border=True):
                st.info("Todavía no hay jugadores añadidos a cartera. Usa el buscador para añadir perfiles.", icon="ℹ️")

    with col_right:
        st.markdown("#### 🔔 Alertas contractuales")
        st.caption("Jugadores con contrato expirando próximamente.")

        if not contract_df.empty:
            cols_show = [c for c in ["Name", "tm_club", "posicion", "edad", "fin_contrato", "valor_mercado"] if c in contract_df.columns]
            df_show = contract_df[cols_show].head(12).copy()
            df_show = df_show.rename(columns={
                "Name": "Jugador", "tm_club": "Club", "posicion": "Posición",
                "edad": "Edad", "fin_contrato": "Fin contrato", "valor_mercado": "Valor",
            })
            st.dataframe(df_show, use_container_width=True, hide_index=True)
        else:
            with st.container(border=True):
                st.info("No se detectan alertas contractuales con la información disponible.", icon="ℹ️")

    st.markdown("---")

    # ── Informes de scouting ──────────────────────────────────────────────────
    st.markdown("#### 📄 Informes de scouting")
    st.caption("Genera informes PDF completos con análisis IA para los jugadores en cartera.")
    render_watchlist_reports(df)