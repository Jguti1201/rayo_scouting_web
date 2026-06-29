"""
clustering_integration.py
=========================
Módulo puente entre feature_engineering.py y los tabs de Streamlit.

Proporciona:
- Cálculo y cacheo de clusters por posición
- Etiquetado automático de clusters
- Búsqueda de similares por cosine similarity
- Percentiles posicionales para radares
- Utilidades compartidas entre buscador, comparador y explorador

Se cachea en st.session_state para evitar recalcular en cada interacción.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# Importar desde tu módulo existente
from features.feature_engineering import (
    CLUSTER_LABELS,
    FEATURE_VECTORS,
    N_CLUSTERS,
    FeatureMatrix,
    build_feature_matrix,
    get_player_vector,
    validate_position,  
)


# ============================================================================
# CONFIG
# ============================================================================

CLUSTER_COLORS = {
    0: "#E30613",
    1: "#1A1A2E",
    2: "#D4A843",
    3: "#28a745",
    4: "#6c757d",
}

POSITION_CLUSTER_DESCRIPTIONS = {
    "Defensa": {
        "Central libero": "Central técnico que inicia juego desde atrás con pase limpio y lectura anticipativa.",
        "Central marcaje": "Central agresivo, dominante en duelos aéreos y terrestres, orientado a la destrucción.",
        "Lateral ofensivo": "Lateral con proyección, centros y participación en ataque.",
        "Lateral defensivo": "Lateral conservador, fiable en 1v1 y coberturas.",
    },
    "Centrocampista": {
        "Pivote defensivo": "Mediocentro de corte, recuperaciones y protección de la línea defensiva.",
        "Box-to-Box": "Centrocampista todoterreno con llegada, recuperación y presencia en ambas áreas.",
        "Creativo / 10": "Mediapunta o interior creativo con pases clave y generación de ocasiones.",
        "Organizador": "Centrocampista de circulación, alta precisión de pase y control del tempo.",
    },
    "Delantero": {
        "9 de área": "Delantero centro clásico, rematador de área con dominio aéreo.",
        "Fijador espaldas": "Delantero de referencia que fija centrales, gana duelos y asocia.",
        "Combinativo / F9": "Falso 9 o delantero técnico que baja a asociar y genera juego.",
        "Rápido / Regate": "Atacante veloz con desborde individual y capacidad para romper líneas.",
    },
    "Portero": {
        "Portero clásico": "Portero de línea, especialista en paradas y dominio del área.",
        "Sweeper-Keeper": "Portero proactivo con salidas, buena distribución y juego con los pies.",
    },
}


# ============================================================================
# CACHE & CLUSTERING
# ============================================================================

def _cache_key(position: str) -> str:
    return f"_cluster_fm_{position}"


def get_or_build_clusters(df: pd.DataFrame, position: str) -> tuple[FeatureMatrix, np.ndarray, list[str]]:
    """
    Construye o recupera de caché la FeatureMatrix + clusters para una posición.

    Retorna:
        - fm: FeatureMatrix
        - cluster_ids: array de ints con el cluster de cada jugador
        - cluster_names: lista de nombres legibles para cada jugador
    """
    # Normalizar y validar posición antes de cualquier operación
    canonical = validate_position(position)
    if canonical is None:
        raise ValueError(f"[CLUSTERING] Posición no reconocible: {position!r}")
    position = canonical  
    key = _cache_key(position)

    if key in st.session_state:
        cached = st.session_state[key]
        logger.info("[CLUSTERING] Recuperado de caché para '%s'", position)
        return cached["fm"], cached["cluster_ids"], cached["cluster_names"]

    logger.warning("[CLUSTERING] Construyendo clusters para '%s'...", position)

    fm = build_feature_matrix(df, position)

    n_clusters = N_CLUSTERS.get(position, 4)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(fm.X_weighted)

    cluster_names = _assign_cluster_labels(fm, cluster_ids, position)

    st.session_state[key] = {
        "fm": fm,
        "cluster_ids": cluster_ids,
        "cluster_names": cluster_names,
    }

    logger.warning(
        "[CLUSTERING] Clusters construidos para '%s': %d jugadores, %d clusters",
        position, fm.n_players, n_clusters
    )

    return fm, cluster_ids, cluster_names


def _assign_cluster_labels(fm: FeatureMatrix, cluster_ids: np.ndarray, position: str) -> list[str]:
    """
    Asigna etiquetas legibles a cada cluster basándose en los centroides.

    Estrategia: ordena los centroides por la feature más discriminante
    de cada posición y asigna las etiquetas predefinidas en CLUSTER_LABELS.
    """
    labels_pool = CLUSTER_LABELS.get(position, [f"Perfil {i+1}" for i in range(max(cluster_ids) + 1)])

    n_clusters = len(set(cluster_ids))
    if n_clusters != len(labels_pool):
        labels_pool = [f"Perfil {i+1}" for i in range(n_clusters)]

    centroids = np.zeros((n_clusters, fm.X_weighted.shape[1]))
    for c in range(n_clusters):
        mask = cluster_ids == c
        if mask.any():
            centroids[c] = fm.X_weighted[mask].mean(axis=0)

    if position == "Defensa" and "aerial_duels_won_pct" in fm.features:
        sort_idx = fm.features.index("aerial_duels_won_pct")
        order = np.argsort(-centroids[:, sort_idx])
    elif position == "Centrocampista" and "Tackles_p90" in fm.features:
        sort_idx = fm.features.index("Tackles_p90")
        order = np.argsort(-centroids[:, sort_idx])
    elif position == "Delantero" and "aerial_duels_won_pct" in fm.features:
        sort_idx = fm.features.index("aerial_duels_won_pct")
        order = np.argsort(-centroids[:, sort_idx])
    elif position == "Portero" and "Runs out" in fm.features:
        sort_idx = fm.features.index("Runs out")
        order = np.argsort(centroids[:, sort_idx])
    else:
        order = np.arange(n_clusters)

    label_map = {}
    for rank, cluster_id in enumerate(order):
        if rank < len(labels_pool):
            label_map[cluster_id] = labels_pool[rank]
        else:
            label_map[cluster_id] = f"Perfil {cluster_id + 1}"

    return [label_map[c] for c in cluster_ids]


def enrich_df_with_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade columnas 'cluster_label' y 'cluster_id' al dataframe completo.
    Solo para jugadores que pasan los filtros de feature_engineering.
    """
    df = df.copy()
    df["cluster_label"] = None
    df["cluster_id"] = -1

    for position in ["Defensa", "Centrocampista", "Delantero", "Portero"]:
        logger.warning("")
        logger.warning("=" * 70)
        logger.warning("[DEBUG] Procesando %s", position)
        logger.warning("=" * 70)

        try:
            fm, cluster_ids, cluster_names = get_or_build_clusters(df, position)

            for i, (_, row) in enumerate(fm.df_subset.iterrows()):
                name = row.get("Name")
                club = row.get("tm_club")

                mask = (df["Name"] == name)
                if "tm_club" in df.columns and pd.notna(club):
                    mask = mask & (df["tm_club"] == club)

                if mask.any():
                    idx = mask.idxmax()
                    df.loc[idx, "cluster_label"] = cluster_names[i]
                    df.loc[idx, "cluster_id"] = int(cluster_ids[i])

        except Exception:
            logger.exception(
                "[DEBUG] ERROR COMPLETO EN %s",
                position,
            )

    n_labeled = (df["cluster_label"].notna()).sum()
    logger.warning("[CLUSTERING] Jugadores etiquetados con cluster: %d / %d", n_labeled, len(df))

    return df


# ============================================================================
# SIMILITUD
# ============================================================================

def find_similar_players(
    df: pd.DataFrame,
    player_name: str,
    position: str,
    top_n: int = 10,
) -> pd.DataFrame:
    # Validar posición — captura NaN, "nan", None, ""
    if validate_position(position) is None:
        logger.warning("[SIMILITUD] Posición inválida para '%s': %r", player_name, position)
        return pd.DataFrame()
    try:
        fm, cluster_ids, cluster_names = get_or_build_clusters(df, position)
    except Exception as e:
        logger.error("[SIMILITUD] Error construyendo clusters: %s", e)
        return pd.DataFrame()

    result = get_player_vector(player_name, fm)
    if result is None:
        logger.warning("[SIMILITUD] Jugador '%s' no encontrado", player_name)
        return pd.DataFrame()

    player_vector, player_row = result

    similarities = cosine_similarity(player_vector, fm.X_weighted)[0]

    player_idx = fm.df_subset[fm.df_subset["Name"] == player_row["Name"]].index
    if len(player_idx) > 0:
        similarities[player_idx[0]] = -1

    top_indices = np.argsort(-similarities)[:top_n]

    results = []
    for idx in top_indices:
        row = fm.df_subset.iloc[idx]
        results.append({
            "Name": row.get("Name"),
            "tm_club": row.get("tm_club"),
            "liga": row.get("liga"),
            "posicion": row.get("posicion"),
            "edad": row.get("edad"),
            "valor_mercado": row.get("valor_mercado"),
            "fin_contrato": row.get("fin_contrato"),
            "minutes_played": row.get("minutes_played"),
            "Average Sofascore Rating": row.get("Average Sofascore Rating"),
            "cluster_label": cluster_names[idx],
            "similarity": round(float(similarities[idx]) * 100, 1),
        })

    return pd.DataFrame(results)


# ============================================================================
# PERCENTILES POSICIONALES
# ============================================================================

def compute_positional_percentile(
    df: pd.DataFrame,
    position: str,
    col: str,
    value: float,
) -> float:
    """
    Percentil de un valor respecto a jugadores de la MISMA posición.
    Más preciso que percentil global.
    """
    subset = df[df["posicion"] == position] if "posicion" in df.columns else df

    if col not in subset.columns:
        return 0.0

    series = pd.to_numeric(subset[col], errors="coerce").dropna()
    if series.empty:
        return 0.0

    return round(float((series <= value).mean() * 100), 1)


def get_cluster_distribution(df: pd.DataFrame, position: str) -> pd.DataFrame:
    """
    Distribución de jugadores por cluster para una posición.
    """
    try:
        fm, cluster_ids, cluster_names = get_or_build_clusters(df, position)
    except Exception:
        return pd.DataFrame()

    data = []
    for label in set(cluster_names):
        mask = [n == label for n in cluster_names]
        count = sum(mask)
        indices = [i for i, m in enumerate(mask) if m]

        avg_rating = None
        if "Average Sofascore Rating" in fm.df_subset.columns:
            vals = pd.to_numeric(
                fm.df_subset.iloc[indices]["Average Sofascore Rating"],
                errors="coerce"
            ).dropna()
            avg_rating = vals.mean() if not vals.empty else None

        avg_age = None
        if "edad" in fm.df_subset.columns:
            vals = fm.df_subset.iloc[indices]["edad"].dropna()
            avg_age = vals.mean() if not vals.empty else None

        desc = POSITION_CLUSTER_DESCRIPTIONS.get(position, {}).get(label, "")

        data.append({
            "Perfil": label,
            "Jugadores": count,
            "Edad media": round(avg_age, 1) if avg_age else "N/D",
            "Rating medio": round(avg_rating, 2) if avg_rating else "N/D",
            "Descripción": desc,
        })

    return pd.DataFrame(data).sort_values("Jugadores", ascending=False).reset_index(drop=True)


def get_cluster_for_player(df: pd.DataFrame, player_name: str, position: str) -> str | None:
    """
    Devuelve el nombre del cluster al que pertenece un jugador.
    """
    try:
        fm, cluster_ids, cluster_names = get_or_build_clusters(df, position)
    except Exception:
        return None

    mask = fm.df_subset["Name"].astype(str) == str(player_name)
    if not mask.any():
        mask = fm.df_subset["Name"].str.contains(str(player_name), case=False, na=False)

    if not mask.any():
        return None

    idx = mask.idxmax()
    return cluster_names[idx]

# En clustering.py — añadir estas dos funciones

_WARMUP_KEY = "_clusters_warmed_up"

def warmup_all_clusters(df: pd.DataFrame) -> None:
    """
    Preconstruye los clusters de las 4 posiciones y los guarda en caché.
    Debe llamarse UNA SOLA VEZ al inicio de la app, antes del primer render
    de cualquier tab que use clustering.

    Evita el bug de rerun: si cada tab construye sus clusters on-demand,
    cada escritura en st.session_state dispara un rerun de Streamlit,
    creando un loop donde Centrocampista y Portero nunca terminan.
    """
    if st.session_state.get(_WARMUP_KEY):
        return  # ya calentado, no hacer nada

    logger.warning("[CLUSTERING] Precalentando clusters para todas las posiciones...")

    for position in ["Defensa", "Centrocampista", "Delantero", "Portero"]:
        key = _cache_key(position)
        if key in st.session_state:
            continue  # este ya está, saltar
        try:
            # Construir sin usar get_or_build_clusters para controlar
            # exactamente cuándo escribimos en session_state
            from features.feature_engineering import build_feature_matrix
            from sklearn.cluster import KMeans

            fm = build_feature_matrix(df, position)
            n_clusters = N_CLUSTERS.get(position, 4)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_ids = kmeans.fit_predict(fm.X_weighted)
            cluster_names = _assign_cluster_labels(fm, cluster_ids, position)

            st.session_state[key] = {
                "fm": fm,
                "cluster_ids": cluster_ids,
                "cluster_names": cluster_names,
            }
            logger.warning(
                "[CLUSTERING] Precalentado '%s': %d jugadores, %d clusters",
                position, fm.n_players, n_clusters,
            )
        except Exception as e:
            logger.warning("[CLUSTERING] Error precalentando '%s': %s", position, e)

    # Marcar como completado — esta es la ÚNICA escritura que dispara rerun,
    # y ocurre después de haber guardado todos los clusters
    st.session_state[_WARMUP_KEY] = True
    logger.warning("[CLUSTERING] Warmup completado.")