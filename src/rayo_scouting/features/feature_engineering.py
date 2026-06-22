"""
feature_engineering.py — Rayo Vallecano Scout IA (v2)
=====================================================
Define vectores de features por posición y construye matrices comparables
para clustering y similitud.

MEJORAS v2
----------
1. Mantiene la lógica de pesos por posición, pero simplifica la API pública.
2. Añade helper get_feature_vector_config(position).
3. Mejora trazabilidad para frontend ejecutivo.
4. Conserva:
   - imputación
   - escalado híbrido
   - inversión de variables negativas
   - custom_weights
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, StandardScaler

logger = logging.getLogger(__name__)

FEATURE_VECTORS: dict[str, list[tuple[str, float]]] = {
    "Defensa": [
        ("aerial_duels_won_pct", 0.20),
        ("Clearances_p90", 0.18),
        ("Interceptions_p90", 0.15),
        ("Accurate passes %", 0.14),
        ("Tackles_p90", 0.10),
        ("ground_duels_won_pct", 0.08),
        ("Assists_p90", 0.05),
        ("Accurate_passes_p90", 0.05),
        ("fouls_p90", 0.05),
        ("Blocked_shots_p90", 0.05),
        ("Errors leading to goal", 0.05),
    ],
    "Centrocampista": [
        ("Tackles_p90", 0.18),
        ("Key_passes_p90", 0.15),
        ("Interceptions_p90", 0.14),
        ("Big_chances_created_p90", 0.10),
        ("Goals_p90", 0.10),
        ("Assists_p90", 0.10),
        ("Accurate passes %", 0.09),
        ("Succ_dribbles_p90", 0.05),
        ("ground_duels_won_pct", 0.05),
        ("fouls_p90", 0.09),
        ("Blocked_shots_p90", 0.05),
    ],
    "Delantero": [
        ("aerial_duels_won_pct", 0.20),
        ("Succ_dribbles_p90", 0.15),
        ("Goals_p90", 0.15),
        ("Key_passes_p90", 0.11),
        ("Goal conversion %", 0.10),
        ("xG_p90", 0.10),
        ("ground_duels_won_pct", 0.11),
        ("Assists_p90", 0.05),
        ("Total_shots_p90", 0.03),
    ],
    "Portero": [
        ("Total_saves_p90", 0.28),
        ("Saves_from_inside_box_p90", 0.22),
        ("Runs out", 0.20),
        ("Clean sheets", 0.10),
        ("Penalties saved", 0.07),
        ("ground_duels_won_pct", 0.08),
        ("aerial_duels_won_pct", 0.05),
    ],
}

N_CLUSTERS: dict[str, int] = {
    "Defensa": 4,
    "Centrocampista": 4,
    "Delantero": 4,
    "Portero": 2,
}

CLUSTER_LABELS: dict[str, list[str]] = {
    "Defensa": ["Central libero", "Central marcaje", "Lateral ofensivo", "Lateral defensivo"],
    "Centrocampista": ["Pivote defensivo", "Box-to-Box", "Creativo / 10", "Organizador"],
    "Delantero": ["9 de área", "Fijador espaldas", "Combinativo / F9", "Rápido / Regate"],
    "Portero": ["Portero clásico", "Sweeper-Keeper"],
}

FEATURES_TO_INVERT = {
    "Errors leading to goal",
}

SKEWNESS_THRESHOLD = 1.5

@dataclass
class FeatureMatrix:
    position: str
    df_subset: pd.DataFrame
    features: list[str]
    weights: list[float]
    X_raw: np.ndarray
    X_scaled: np.ndarray
    X_weighted: np.ndarray
    scalers: dict[str, object]
    imputer: Optional[SimpleImputer]
    n_players: int = field(init=False)

    def __post_init__(self):
        self.n_players = len(self.df_subset)

def get_feature_vector_config(position: str) -> list[tuple[str, float]]:
    if position not in FEATURE_VECTORS:
        raise ValueError(f"Posición no soportada: {position}")
    return FEATURE_VECTORS[position]

def _resolve_features(position: str, df: pd.DataFrame) -> tuple[list[str], list[float]]:
    if position not in FEATURE_VECTORS:
        raise ValueError(f"[feature_engineering] Posición no soportada: {position}")

    features, weights = [], []
    for col, weight in FEATURE_VECTORS[position]:
        if col not in df.columns:
            logger.warning("[feature_engineering] Feature '%s' no encontrada para '%s'.", col, position)
            continue
        if df[col].isna().all():
            logger.warning("[feature_engineering] Feature '%s' completamente vacía para '%s'.", col, position)
            continue
        features.append(col)
        weights.append(weight)

    if not features:
        raise ValueError(f"[feature_engineering] No hay features válidas para {position}")

    total_abs = sum(abs(w) for w in weights)
    weights = [w / total_abs for w in weights]
    return features, weights

def _filter_muestra_fiable(df_subset: pd.DataFrame, position: str, min_players: int) -> pd.DataFrame:
    if "muestra_fiable" not in df_subset.columns:
        return df_subset

    reliable = df_subset[df_subset["muestra_fiable"] == True].copy()
    if len(reliable) < min_players:
        logger.warning(
            "[feature_engineering] Filtro muestra_fiable omitido para '%s' por tamaño insuficiente.",
            position
        )
        return df_subset

    return reliable

def _invert_negative_features(df_subset: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    df_subset = df_subset.copy()
    for feat in features:
        if feat in FEATURES_TO_INVERT and feat in df_subset.columns:
            df_subset[feat] = -df_subset[feat]
    return df_subset

def _impute_and_prepare(
    df_subset: pd.DataFrame,
    position: str,
    features: list[str],
) -> tuple[pd.DataFrame, np.ndarray, Optional[SimpleImputer]]:
    imputer = None

    if position == "Portero":
        df_subset = df_subset.copy()
        df_subset[features] = df_subset[features].fillna(0.0)
        X_raw = df_subset[features].values.astype(float)
    else:
        mask_all_nan = df_subset[features].isna().all(axis=1)
        if mask_all_nan.any():
            df_subset = df_subset[~mask_all_nan].copy()

        imputer = SimpleImputer(strategy="median")
        X_raw = imputer.fit_transform(df_subset[features].values.astype(float))

    df_subset = df_subset.reset_index(drop=True)
    return df_subset, X_raw, imputer

def _scale_by_skewness(X_raw: np.ndarray, features: list[str]) -> tuple[np.ndarray, dict[str, object]]:
    X_scaled = np.zeros_like(X_raw, dtype=float)
    scalers: dict[str, object] = {}

    for i, feat in enumerate(features):
        col = X_raw[:, i]
        skewness = abs(stats.skew(col))

        scaler = RobustScaler() if skewness > SKEWNESS_THRESHOLD else StandardScaler()
        X_scaled[:, i] = scaler.fit_transform(col.reshape(-1, 1)).ravel()
        scalers[feat] = scaler

    return X_scaled, scalers

def build_feature_matrix(
    df: pd.DataFrame,
    position: str,
    custom_weights: Optional[dict[str, float]] = None,
) -> FeatureMatrix:
    df_subset = df[df["posicion"] == position].copy()

    if len(df_subset) < 10:
        raise ValueError(f"[feature_engineering] Solo {len(df_subset)} jugadores para '{position}'.")

    df_subset = _filter_muestra_fiable(
        df_subset,
        position,
        min_players=N_CLUSTERS.get(position, 4) * 3,
    )

    features, weights = _resolve_features(position, df_subset)

    if custom_weights:
        for i, feat in enumerate(features):
            if feat in custom_weights:
                weights[i] = custom_weights[feat]
        total_abs = sum(abs(w) for w in weights)
        weights = [w / total_abs for w in weights]

    df_subset = _invert_negative_features(df_subset, features)
    df_subset, X_raw, imputer = _impute_and_prepare(df_subset, position, features)

    if len(df_subset) < N_CLUSTERS.get(position, 4):
        raise ValueError(
            f"[feature_engineering] Jugadores insuficientes tras limpieza para '{position}'."
        )

    X_scaled, scalers = _scale_by_skewness(X_raw, features)
    weights_array = np.array(weights, dtype=float)
    X_weighted = X_scaled * weights_array

    logger.info(
        "[feature_engineering] '%s': %d jugadores × %d features",
        position, X_weighted.shape[0], X_weighted.shape[1]
    )

    return FeatureMatrix(
        position=position,
        df_subset=df_subset,
        features=features,
        weights=weights,
        X_raw=X_raw,
        X_scaled=X_scaled,
        X_weighted=X_weighted,
        scalers=scalers,
        imputer=imputer,
    )

def get_player_vector(player_name: str, fm: FeatureMatrix) -> Optional[tuple[np.ndarray, pd.Series]]:
    mask = fm.df_subset["Name"] == player_name
    if not mask.any():
        mask = fm.df_subset["Name"].str.contains(player_name, case=False, na=False)
        if not mask.any():
            return None

    idx = mask.idxmax()
    return fm.X_weighted[idx].reshape(1, -1), fm.df_subset.loc[idx]

def transform_external_player(player_row: pd.Series, fm: FeatureMatrix) -> np.ndarray:
    values = player_row.reindex(fm.features).values.astype(float)

    for i, feat in enumerate(fm.features):
        if feat in FEATURES_TO_INVERT:
            values[i] = -values[i]

    values = values.reshape(1, -1)

    if fm.imputer is not None:
        values = fm.imputer.transform(values)
    else:
        values = np.nan_to_num(values, nan=0.0)

    values_scaled = np.zeros_like(values, dtype=float)
    for i, feat in enumerate(fm.features):
        scaler = fm.scalers[feat]
        values_scaled[0, i] = scaler.transform(values[0, i].reshape(-1, 1)).ravel()[0]

    return values_scaled * np.array(fm.weights, dtype=float)