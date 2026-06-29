"""
feature_engineering.py — Rayo Vallecano Scout IA (v3)
=====================================================
Define vectores de features por posición y construye matrices comparables
para clustering y similitud.

CAMBIOS v3 (basados en análisis del CSV all_leagues_master_v5)
--------------------------------------------------------------
1. CÁLCULO DE COLUMNAS _p90
   El CSV no tiene columnas _p90 precalculadas. Se calculan aquí desde
   los valores acumulados y minutes_played antes de construir la matriz.

2. FEATURES PROBLEMÁTICAS ELIMINADAS O TRATADAS
   - fouls_p90 / ground_duels_won_pct: 79% nulos en todo el CSV (no es
     problema de ligas concretas, es dato estructuralmente ausente).
     Se eliminan de Defensa y Centrocampista; se sustituyen por
     alternativas con cobertura real.
   - Expected goals (xG): 30% nulos. Se mantiene solo en Delantero con
     imputación segura; se excluye de otros vectores.

3. FILTRO DE MUESTRA MÍNIMA (MIN_MINUTES)
   Solo jugadores con ≥ 450 minutos entran en clustering/similitud,
   para evitar que jugadores con 1-100 min distorsionen los vectores.

4. VALIDACIÓN DE POSICIÓN
   Nuevo helper `validate_position` que acepta tanto el formato del CSV
   ("Defensa", "Centrocampista"...) como formas normalizadas del tab
   ("defensa", "medio"...) y devuelve siempre la clave canónica o None.

5. TRAZABILIDAD
   FeatureMatrix expone `coverage` (% jugadores por feature con dato real)
   para debug en el frontend.
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


# ============================================================================
# CONSTANTES
# ============================================================================

# Minutos mínimos para entrar en clustering / similitud.
# p25 de la distribución real por posición es ~400 min → 450 es conservador
# pero elimina jugadores con 1-100 min que distorsionan vectores.
MIN_MINUTES: dict[str, int] = {
    "Defensa": 450,
    "Centrocampista": 450,
    "Delantero": 450,
    "Portero": 450,
}

# Mínimo de jugadores necesarios para que el clustering sea válido
MIN_PLAYERS_FOR_CLUSTERING = 15

SKEWNESS_THRESHOLD = 1.5

# Features cuyo significado es "menor = mejor" → se invierten
FEATURES_TO_INVERT = {
    "Errors leading to goal",
}

# Mapa de normalización de posición (entradas del tab → clave canónica del CSV)
# El CSV usa: "Defensa", "Centrocampista", "Delantero", "Portero"
POSITION_ALIAS_MAP: dict[str, str] = {
    # Formato canónico del CSV
    "defensa": "Defensa",
    "centrocampista": "Centrocampista",
    "delantero": "Delantero",
    "portero": "Portero",
    # Alias del tab_comparador y otros tabs
    "central": "Defensa",
    "lateral": "Defensa",
    "carrilero": "Defensa",
    "mediocentro": "Centrocampista",
    "medio": "Centrocampista",
    "interior": "Centrocampista",
    "pivote": "Centrocampista",
    "mediapunta": "Centrocampista",
    "extremo": "Delantero",
    "segundo punta": "Delantero",
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

# ============================================================================
# VECTORES DE FEATURES
# Notas sobre cambios respecto a v2:
#   - fouls_p90 eliminado (79% nulos en CSV real)
#   - ground_duels_won_pct eliminado de Defensa (79% nulos); sustituido
#     por Blocked_shots_p90 con mayor cobertura real
#   - ground_duels_won_pct se mantiene en Delantero/Portero porque el
#     patrón de nulos coincide con posiciones donde hay datos parciales
#   - aerial_duels_won_pct se mantiene (solo 9.7% nulos, imputables)
#   - xG solo en Delantero (30% nulos, manejables con imputación)
#   - Los pesos se renormalizan internamente, no es necesario que sumen 1.0
# ============================================================================

FEATURE_VECTORS: dict[str, list[tuple[str, float]]] = {
    "Defensa": [
        ("aerial_duels_won_pct",    0.22),   # 9.7% nulos — feature más discriminante
        ("Clearances_p90",          0.20),   # 0.9% nulos — muy fiable
        ("Interceptions_p90",       0.18),   # 0.9% nulos
        ("Tackles_p90",             0.15),   # 0.9% nulos
        ("Accurate passes %",       0.13),   # 2.3% nulos
        ("Blocked_shots_p90",       0.07),   # 1.3% nulos
        ("Assists_p90",             0.05),   # 2.3% nulos — diferencia lateral/central
        # Eliminado en v3: fouls_p90 (79% nulos), ground_duels_won_pct (79% nulos)
        # Mantenido con peso reducido: Errors_leading_to_goal (invertido)
        ("Errors leading to goal",  0.00),   # se incluye como señal negativa si hay datos
    ],
    "Centrocampista": [
        ("Tackles_p90",             0.18),
        ("Key_passes_p90",          0.16),
        ("Interceptions_p90",       0.15),
        ("Big_chances_created_p90", 0.12),
        ("Goals_p90",               0.10),
        ("Assists_p90",             0.10),
        ("Accurate passes %",       0.10),
        ("Succ_dribbles_p90",       0.05),
        ("Blocked_shots_p90",       0.04),
        # Eliminado en v3: fouls_p90 (79% nulos), ground_duels_won_pct (79% nulos)
    ],
    "Delantero": [
        ("Goals_p90",               0.22),
        ("xG_p90",                  0.18),   # 30% nulos → imputar con mediana posicional
        ("aerial_duels_won_pct",    0.15),
        ("Succ_dribbles_p90",       0.13),
        ("Goal conversion %",       0.10),
        ("Key_passes_p90",          0.09),
        ("Total_shots_p90",         0.07),
        ("Assists_p90",             0.06),
    ],
    "Portero": [
        ("Total_saves_p90",         0.30),
        ("Saves_from_inside_box_p90", 0.22),
        ("Runs out",                0.20),   # acumulado, no p90 (SofaScore lo da por temporada)
        ("Clean sheets",            0.12),   # idem
        ("Penalties saved",         0.09),   # idem
        ("aerial_duels_won_pct",    0.07),
    ],
}


# ============================================================================
# DATACLASS
# ============================================================================

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
    coverage: dict[str, float] = field(default_factory=dict)  # nuevo en v3
    n_players: int = field(init=False)

    def __post_init__(self):
        self.n_players = len(self.df_subset)


# ============================================================================
# HELPERS PÚBLICOS
# ============================================================================

def validate_position(raw_position) -> Optional[str]:
    """
    Acepta cualquier forma de posición y devuelve la clave canónica
    que usa el CSV ("Defensa", "Centrocampista", "Delantero", "Portero"),
    o None si no es reconocible.

    Maneja NaN, None, strings vacíos, y alias del tab.
    """
    if raw_position is None:
        return None
    # Captura float NaN (lo que pandas devuelve para celdas vacías)
    try:
        if pd.isna(raw_position):
            return None
    except (TypeError, ValueError):
        pass

    pos_str = str(raw_position).strip().lower()

    if not pos_str or pos_str in ("nan", "none", ""):
        return None

    # Primero intenta match exacto insensible a mayúsculas
    for canonical in ("Defensa", "Centrocampista", "Delantero", "Portero"):
        if pos_str == canonical.lower():
            return canonical

    # Luego busca alias parciales (orden importa: más específicos primero)
    for alias, canonical in POSITION_ALIAS_MAP.items():
        if alias in pos_str:
            return canonical

    return None


def get_feature_vector_config(position: str) -> list[tuple[str, float]]:
    """Devuelve la config de features para una posición canónica."""
    canonical = validate_position(position)
    if canonical is None or canonical not in FEATURE_VECTORS:
        raise ValueError(f"Posición no soportada: {position!r}")
    return FEATURE_VECTORS[canonical]


# ============================================================================
# CÁLCULO DE COLUMNAS _p90
# El CSV no tiene columnas _p90 precalculadas. Esta función las añade
# al vuelo antes de construir la matriz de features.
# ============================================================================

_P90_MAP: dict[str, str] = {
    "Goals_p90":                  "Goals",
    "Assists_p90":                "Assists",
    "Key_passes_p90":             "Key passes",
    "Tackles_p90":                "Tackles",
    "Interceptions_p90":          "Interceptions",
    "Clearances_p90":             "Clearances",
    "Blocked_shots_p90":          "Blocked shots",
    "Big_chances_created_p90":    "Big chances created",
    "Succ_dribbles_p90":          "Succ. dribbles",
    "Total_shots_p90":            "Total shots",
    "Total_saves_p90":            "Total saves",
    "Saves_from_inside_box_p90":  "Saves from inside box",
    "Accurate_passes_p90":        "Accurate passes",
    "xG_p90":                     "Expected goals (xG)",
}


def compute_p90_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade columnas _p90 al dataframe si no existen ya.
    Solo modifica columnas ausentes; las que ya existan se respetan.
    Devuelve una copia del dataframe.
    """
    df = df.copy()
    minutes = pd.to_numeric(df.get("minutes_played"), errors="coerce")

    for p90_col, raw_col in _P90_MAP.items():
        if p90_col in df.columns:
            continue  # ya existe, no sobreescribir
        if raw_col not in df.columns:
            continue  # columna fuente no disponible

        raw = pd.to_numeric(df[raw_col], errors="coerce")
        # Evitar división por cero o minutos muy bajos (resultado infinito)
        safe_minutes = minutes.where(minutes >= 1, other=np.nan)
        df[p90_col] = (raw / safe_minutes) * 90.0

    return df


# ============================================================================
# FILTROS INTERNOS
# ============================================================================

def _filter_min_minutes(df_subset: pd.DataFrame, position: str) -> pd.DataFrame:
    """Elimina jugadores con pocos minutos antes de calcular p90."""
    if "minutes_played" not in df_subset.columns:
        return df_subset

    min_min = MIN_MINUTES.get(position, 450)
    minutes = pd.to_numeric(df_subset["minutes_played"], errors="coerce")
    mask = minutes >= min_min
    filtered = df_subset[mask].copy()

    dropped = len(df_subset) - len(filtered)
    if dropped > 0:
        logger.info(
            "[feature_engineering] '%s': %d jugadores descartados por < %d min",
            position, dropped, min_min,
        )
    return filtered


def _filter_muestra_fiable(df_subset: pd.DataFrame, position: str, min_players: int) -> pd.DataFrame:
    if "muestra_fiable" not in df_subset.columns:
        return df_subset

    reliable = df_subset[df_subset["muestra_fiable"] == True].copy()
    if len(reliable) < min_players:
        logger.warning(
            "[feature_engineering] Filtro muestra_fiable omitido para '%s' por tamaño insuficiente (%d < %d).",
            position, len(reliable), min_players,
        )
        return df_subset

    return reliable


def _resolve_features(
    position: str,
    df: pd.DataFrame,
    min_coverage: float = 0.15,
) -> tuple[list[str], list[float], dict[str, float]]:
    """
    Resuelve qué features del vector están disponibles en el dataframe.

    Parámetros
    ----------
    min_coverage : float
        Fracción mínima de filas con dato real para incluir una feature.
        Features por debajo de este umbral se descartan con warning.
        Default 0.15 (15%) — conservador para tolerar xG con 30% nulos.

    Devuelve features, weights (renormalizados), y coverage por feature.
    """
    if position not in FEATURE_VECTORS:
        raise ValueError(f"[feature_engineering] Posición no soportada: {position!r}")

    n = len(df)
    features, weights, coverage = [], [], {}

    for col, weight in FEATURE_VECTORS[position]:
        if weight == 0.0:
            # Feature marcada con peso 0 → incluir solo si hay datos, con peso mínimo
            # (sirve como señal negativa para Errors leading to goal)
            pass

        if col not in df.columns:
            logger.warning("[feature_engineering] Feature '%s' no encontrada en df para '%s'.", col, position)
            continue

        col_coverage = df[col].notna().mean()
        coverage[col] = round(col_coverage, 3)

        if col_coverage < min_coverage:
            logger.warning(
                "[feature_engineering] Feature '%s' descartada para '%s': cobertura %.1f%% < %.0f%%.",
                col, position, col_coverage * 100, min_coverage * 100,
            )
            continue

        if weight == 0.0 and col == "Errors leading to goal":
            # Incluir con peso pequeño si hay datos suficientes
            effective_weight = 0.04
        else:
            effective_weight = weight

        features.append(col)
        weights.append(effective_weight)

    if not features:
        raise ValueError(f"[feature_engineering] No hay features válidas para {position!r}")

    # Renormalizar pesos
    total = sum(abs(w) for w in weights)
    weights = [w / total for w in weights]

    logger.info(
        "[feature_engineering] '%s': %d features seleccionadas de %d definidas",
        position, len(features), len(FEATURE_VECTORS[position]),
    )
    return features, weights, coverage


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
    """
    Imputa NaN y devuelve array X_raw.
    Porteros: fillna(0) porque sus stats son acumuladas por temporada.
    Resto: mediana por columna (SimpleImputer), descartando filas all-NaN.
    """
    imputer = None

    if position == "Portero":
        df_subset = df_subset.copy()
        df_subset[features] = df_subset[features].fillna(0.0)
        X_raw = df_subset[features].values.astype(float)
    else:
        # Descartar jugadores sin ningún dato en ninguna feature
        mask_all_nan = df_subset[features].isna().all(axis=1)
        if mask_all_nan.any():
            n_drop = mask_all_nan.sum()
            logger.warning(
                "[feature_engineering] '%s': descartando %d jugadores sin ningún dato.",
                position, n_drop,
            )
            df_subset = df_subset[~mask_all_nan].copy()

        imputer = SimpleImputer(strategy="median")
        X_raw = imputer.fit_transform(df_subset[features].values.astype(float))

    df_subset = df_subset.reset_index(drop=True)
    return df_subset, X_raw, imputer


# ============================================================================
# ESCALADO
# ============================================================================

def _scale_by_skewness(
    X_raw: np.ndarray,
    features: list[str],
) -> tuple[np.ndarray, dict[str, object]]:
    """
    RobustScaler para features con skewness > threshold (outliers frecuentes
    en p90 de jugadores con pocos minutos); StandardScaler para el resto.
    """
    X_scaled = np.zeros_like(X_raw, dtype=float)
    scalers: dict[str, object] = {}

    for i, feat in enumerate(features):
        col = X_raw[:, i]
        skewness = abs(stats.skew(col))
        scaler = RobustScaler() if skewness > SKEWNESS_THRESHOLD else StandardScaler()
        X_scaled[:, i] = scaler.fit_transform(col.reshape(-1, 1)).ravel()
        scalers[feat] = scaler

    return X_scaled, scalers


# ============================================================================
# API PÚBLICA PRINCIPAL
# ============================================================================

def build_feature_matrix(
    df: pd.DataFrame,
    position: str,
    custom_weights: Optional[dict[str, float]] = None,
) -> FeatureMatrix:
    """
    Construye la FeatureMatrix para una posición.

    Parámetros
    ----------
    df : DataFrame completo (se filtra internamente por posición)
    position : clave canónica ("Defensa", "Centrocampista", "Delantero", "Portero")
               o cualquier alias aceptado por validate_position()
    custom_weights : pesos opcionales para sobreescribir los defaults

    Raises
    ------
    ValueError si la posición no es válida o hay pocos jugadores.
    """
    # Validar y normalizar posición
    canonical = validate_position(position)
    if canonical is None:
        raise ValueError(f"[feature_engineering] Posición no reconocida: {position!r}")
    position = canonical
    logger.warning("=" * 80)
    logger.warning("[DEBUG] build_feature_matrix(%s)", position)
    logger.warning(
        "[DEBUG] Posiciones en dataset:\n%s",
        df["posicion"].value_counts(dropna=False).to_string()
    )

    # Calcular columnas _p90 si no existen
    df = compute_p90_columns(df)

    # Filtrar por posición
    df_subset = df[df["posicion"] == position].copy()
    logger.warning(
        "[DEBUG] %s -> %d jugadores antes de filtros",
        position,
        len(df_subset),
    )

    if len(df_subset) < MIN_PLAYERS_FOR_CLUSTERING:
        raise ValueError(
            f"[feature_engineering] Solo {len(df_subset)} jugadores para '{position}' "
            f"(mínimo {MIN_PLAYERS_FOR_CLUSTERING})."
        )

    # Filtrar por minutos mínimos (evita que jugadores con 1 min distorsionen _p90)
    df_subset = _filter_min_minutes(df_subset, position)

    if len(df_subset) < MIN_PLAYERS_FOR_CLUSTERING:
        raise ValueError(
            f"[feature_engineering] Solo {len(df_subset)} jugadores para '{position}' "
            f"tras filtro de minutos mínimos."
        )

    # Filtro muestra_fiable (si existe la columna)
    df_subset = _filter_muestra_fiable(
        df_subset,
        position,
        min_players=N_CLUSTERS.get(position, 4) * 3,
    )

    # Resolver features disponibles con cobertura suficiente
    features, weights, coverage = _resolve_features(position, df_subset)

    # Custom weights (override)
    if custom_weights:
        for i, feat in enumerate(features):
            if feat in custom_weights:
                weights[i] = custom_weights[feat]
        total = sum(abs(w) for w in weights)
        weights = [w / total for w in weights]

    # Invertir features negativas
    df_subset = _invert_negative_features(df_subset, features)

    # Imputar y preparar X_raw
    df_subset, X_raw, imputer = _impute_and_prepare(df_subset, position, features)

    if len(df_subset) < N_CLUSTERS.get(position, 4):
        raise ValueError(
            f"[feature_engineering] Jugadores insuficientes tras limpieza para '{position}': "
            f"{len(df_subset)} < {N_CLUSTERS[position]} clusters."
        )

    # Escalar
    X_scaled, scalers = _scale_by_skewness(X_raw, features)

    # Aplicar pesos
    weights_array = np.array(weights, dtype=float)
    X_weighted = X_scaled * weights_array

    logger.info(
        "[feature_engineering] '%s': %d jugadores × %d features — listo",
        position, X_weighted.shape[0], X_weighted.shape[1],
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
        coverage=coverage,
    )


def get_player_vector(
    player_name: str,
    fm: FeatureMatrix,
) -> Optional[tuple[np.ndarray, pd.Series]]:
    """
    Devuelve (vector_ponderado, fila) para un jugador dentro de una FeatureMatrix.
    Busca primero por match exacto, luego por substring case-insensitive.
    """
    mask = fm.df_subset["Name"].astype(str) == str(player_name)
    if not mask.any():
        mask = fm.df_subset["Name"].astype(str).str.contains(
            str(player_name), case=False, na=False
        )
    if not mask.any():
        return None

    idx = mask.idxmax()
    return fm.X_weighted[idx].reshape(1, -1), fm.df_subset.loc[idx]


def transform_external_player(
    player_row: pd.Series,
    fm: FeatureMatrix,
) -> np.ndarray:
    """
    Proyecta un jugador externo (no en la FeatureMatrix) al mismo espacio vectorial.
    Útil para comparar un jugador nuevo contra el espacio de clustering existente.
    """
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
        values_scaled[0, i] = scaler.transform(
            values[0, i].reshape(-1, 1)
        ).ravel()[0]

    return values_scaled * np.array(fm.weights, dtype=float)