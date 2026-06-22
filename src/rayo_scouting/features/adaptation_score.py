"""
adaptation_score.py — Rayo Vallecano Scout IA (v2)
==================================================
Calcula el Adaptation Score para estimar la facilidad de integración de un
jugador en el contexto del Rayo Vallecano.

MEJORAS v2
----------
1. API pública más clara:
   - compute_adaptation_score
   - add_adaptation_scores
   - get_adaptation_summary

2. Más robustez ante datos faltantes.
3. Mantiene el diseño por 4 dimensiones.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)

SPANISH_COUNTRIES = frozenset({
    "España", "Argentina", "Uruguay", "Colombia", "México", "Chile",
    "Venezuela", "Ecuador", "Perú", "Paraguay", "Bolivia", "Honduras",
    "Costa Rica", "Panamá", "Guatemala", "Cuba", "República Dominicana",
    "El Salvador", "Nicaragua",
})

FRENCH_COUNTRIES = frozenset({
    "Francia", "Bélgica", "Senegal", "Costa de Marfil", "Camerún",
    "Mali", "Guinea", "Marruecos", "RD Congo", "Argelia", "Túnez",
    "Burkina Faso", "Gabón", "Congo", "Haití",
})

PORTUGUESE_COUNTRIES = frozenset({
    "Portugal", "Brasil", "Angola", "Mozambique", "Cabo Verde", "Guinea-Bisáu",
})

ITALIAN_COUNTRIES = frozenset({
    "Italia", "San Marino",
})

GERMAN_COUNTRIES = frozenset({
    "Alemania", "Austria", "Suiza",
})

IDIOMA_LALIGA_SCORE = {
    "Español": 100.0,
    "Portugués": 80.0,
    "Italiano": 65.0,
    "Francés": 50.0,
    "Inglés": 30.0,
    "Alemán": 25.0,
    "Otro": 15.0,
}

LIGA_EXPERIENCIA_SCORE = {
    "laliga": 100.0,
    "laliga2": 100.0,
    "liga_argentina": 60.0,
    "bundesliga": 55.0,
    "bundesliga2": 50.0,
    "premier": 55.0,
    "championship": 45.0,
    "serie_a": 55.0,
    "serie_b": 45.0,
    "ligue1": 55.0,
    "ligue2": 45.0,
    "liga_portuguesa": 70.0,
    "mls": 50.0,
}

PPDA_BY_LIGA = {
    "laliga": 8.5,
    "laliga2": 9.5,
    "premier": 9.0,
    "championship": 10.5,
    "bundesliga": 9.0,
    "bundesliga2": 10.0,
    "serie_a": 10.0,
    "serie_b": 11.0,
    "ligue1": 10.5,
    "ligue2": 11.5,
    "liga_portuguesa": 10.0,
    "liga_argentina": 12.5,
    "mls": 12.0,
}

RAYO_PPDA = 8.5
PPDA_PENALTY_FACTOR = 8.0

RAYO_SQUAD_LANG_COMPANIONS = {
    "Español": 15,
    "Francés": 2,
    "Portugués": 2,
    "Rumano": 1,
    "Albanés": 1,
    "Inglés": 0,
    "Alemán": 0,
    "Italiano": 0,
    "Otro": 0,
}

POINTS_PER_COMPANION = 15.0

@dataclass
class AdaptationBreakdown:
    player_name: str
    player_lang: str
    liga: str
    idioma_liga_score: float
    squad_lang_score: float
    liga_exp_score: float
    pressing_compat_score: float
    total: float

def detect_player_language(row: pd.Series) -> str:
    nac = str(row.get("nacionalidades", ""))

    for country in SPANISH_COUNTRIES:
        if country in nac:
            return "Español"

    for country in PORTUGUESE_COUNTRIES:
        if country in nac:
            return "Portugués"

    for country in ITALIAN_COUNTRIES:
        if country in nac:
            return "Italiano"

    for country in FRENCH_COUNTRIES:
        if country in nac:
            return "Francés"

    english_countries = {
        "Inglaterra", "Escocia", "Gales", "Irlanda", "Estados Unidos",
        "Jamaica", "Trinidad y Tobago", "Nigeria", "Ghana", "Sudáfrica",
        "Australia", "Canadá", "Kenia", "Uganda",
    }
    for country in english_countries:
        if country in nac:
            return "Inglés"

    for country in GERMAN_COUNTRIES:
        if country in nac:
            return "Alemán"

    return "Otro"

def _compute_pressing_compatibility(liga: str) -> float:
    ppda_liga = PPDA_BY_LIGA.get(liga, 11.0)
    diff = abs(ppda_liga - RAYO_PPDA)
    score = max(0.0, 100.0 - diff * PPDA_PENALTY_FACTOR)
    return round(score, 1)

def _compute_squad_lang_score(player_lang: str) -> float:
    n_companions = RAYO_SQUAD_LANG_COMPANIONS.get(player_lang, 0)
    score = min(n_companions * POINTS_PER_COMPANION, 100.0)
    return round(score, 1)

def compute_adaptation_score(row: pd.Series) -> AdaptationBreakdown:
    liga = str(row.get("liga", "")).lower().strip()
    player_lang = detect_player_language(row)

    idioma_liga = IDIOMA_LALIGA_SCORE.get(player_lang, 15.0)
    squad_lang = _compute_squad_lang_score(player_lang)
    liga_exp = LIGA_EXPERIENCIA_SCORE.get(liga, 40.0)
    pressing_compat = _compute_pressing_compatibility(liga)

    total = (
        0.35 * idioma_liga
        + 0.30 * squad_lang
        + 0.20 * liga_exp
        + 0.15 * pressing_compat
    )
    total = round(min(max(total, 0.0), 100.0), 1)

    return AdaptationBreakdown(
        player_name=str(row.get("Name", "Desconocido")),
        player_lang=player_lang,
        liga=liga,
        idioma_liga_score=idioma_liga,
        squad_lang_score=squad_lang,
        liga_exp_score=liga_exp,
        pressing_compat_score=pressing_compat,
        total=total,
    )

def add_adaptation_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    breakdowns = df.apply(compute_adaptation_score, axis=1)

    df["player_lang"] = [b.player_lang for b in breakdowns]
    df["adapt_idioma"] = [b.idioma_liga_score for b in breakdowns]
    df["adapt_squad"] = [b.squad_lang_score for b in breakdowns]
    df["adapt_liga"] = [b.liga_exp_score for b in breakdowns]
    df["adapt_pressing"] = [b.pressing_compat_score for b in breakdowns]
    df["adapt_score"] = [b.total for b in breakdowns]

    return df

def get_adaptation_summary(df: pd.DataFrame) -> pd.DataFrame:
    tmp = add_adaptation_scores(df)
    summary = (
        tmp.groupby("liga")["adapt_score"]
        .agg(["mean", "max", "min", "count"])
        .sort_values("mean", ascending=False)
        .round(1)
    )
    return summary