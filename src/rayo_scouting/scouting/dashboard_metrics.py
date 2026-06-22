"""
dashboard_metrics.py
====================
Helpers para métricas de dashboard y formato ejecutivo.
"""

from __future__ import annotations
import pandas as pd

from rayo_scouting.features.data_loader import get_rayo_squad, get_contract_opportunities as dl_get_contract_opportunities

LIGA_DISPLAY = {
    "laliga": "LaLiga",
    "laliga2": "LaLiga 2",
    "premier": "Premier League",
    "championship": "Championship",
    "bundesliga": "Bundesliga",
    "bundesliga2": "Bundesliga 2",
    "serie_a": "Serie A",
    "serie_b": "Serie B",
    "ligue1": "Ligue 1",
    "ligue2": "Ligue 2",
    "liga_portuguesa": "Primeira Liga",
    "liga_argentina": "Liga Argentina",
    "mls": "MLS",
}

def fmt_liga(liga: str) -> str:
    return LIGA_DISPLAY.get(str(liga), str(liga))

def get_watchlist_df(df: pd.DataFrame, watchlist: list[str]) -> pd.DataFrame:
    if not watchlist:
        return pd.DataFrame(columns=df.columns)
    return df[df["Name"].isin(watchlist)].copy()

def get_rayo_df(df: pd.DataFrame) -> pd.DataFrame:
    return get_rayo_squad(df)

def get_summary_metrics(df: pd.DataFrame) -> dict:
    rayo_df = get_rayo_squad(df)
    contract_df = dl_get_contract_opportunities(df, max_year=2026)

    watchlist = []
    try:
        import streamlit as st
        watchlist = st.session_state.get("watchlist", [])
    except Exception:
        pass

    return {
        "n_players": len(df),
        "n_contract_opportunities": len(contract_df),
        "n_watchlist": len(watchlist),
        "n_rayo": len(rayo_df),
    }

def get_featured_recommendation(df: pd.DataFrame):
    tmp = df.copy()
    if "Average Sofascore Rating" in tmp.columns:
        tmp = tmp.sort_values("Average Sofascore Rating", ascending=False)
    elif "minutes_played" in tmp.columns:
        tmp = tmp.sort_values("minutes_played", ascending=False)

    if tmp.empty:
        return None
    return tmp.iloc[0]

def get_contract_opportunities(df: pd.DataFrame) -> pd.DataFrame:
    return dl_get_contract_opportunities(df, max_year=2026)