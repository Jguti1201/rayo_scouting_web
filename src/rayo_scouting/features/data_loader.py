"""
data_loader.py — Rayo Vallecano Scout IA (v4)
=============================================
Carga el CSV maestro, corrige problemas comunes de codificación,
valida su estructura, aplica filtros mínimos de calidad y genera
features derivadas necesarias para la app y el modelado.

OBJETIVOS DE ESTA VERSIÓN
-------------------------
1. Detectar problemas de parseo del CSV:
   - filas desplazadas
   - delimitadores inconsistentes
   - quoting roto
   - columnas mal alineadas

2. Corregir problemas frecuentes de texto / encoding:
   - MbappÃ© -> Mbappé
   - CamerÃºn -> Camerún
   - â‚¬ -> €

3. Convertir de forma robusta las columnas numéricas.

4. Mantener columnas originales intactas y calcular columnas derivadas aparte.

5. Añadir diagnóstico explícito para depurar casos como Mbappé.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "data" / "all_leagues_master_v5.csv"
DEFAULT_MIN_MINUTES = 450

REQUIRED_COLUMNS = [
    "Name",
    "liga",
    "tm_club",
    "posicion",
    "edad",
    "minutes_played",
]

OPTIONAL_BUT_USEFUL_COLUMNS = [
    "nacionalidades",
    "fin_contrato",
    "valor_mercado",
    "Goals",
    "Assists",
    "Key passes",
    "Total shots",
    "Expected goals (xG)",
    "Succ. dribbles",
    "Big chances created",
    "Goal conversion %",
    "Accurate passes %",
    "Tackles",
    "Interceptions",
    "Clearances",
    "Blocked shots",
    "ground_duels_won_pct",
    "aerial_duels_won_pct",
    "Average Sofascore Rating",
    "Runs out",
    "Clean sheets",
    "Penalties saved",
    "Total saves",
    "Saves from inside box",
    "muestra_fiable",
]

RAW_NUMERIC_COLUMNS = [
    "edad",
    "minutes_played",
    "#",
    "Goals",
    "Assists",
    "Key passes",
    "Total shots",
    "Expected goals (xG)",
    "Succ. dribbles",
    "Big chances created",
    "Goal conversion %",
    "Accurate passes %",
    "Tackles",
    "Interceptions",
    "Clearances",
    "Errors leading to goal",
    "Blocked shots",
    "ground_duels_won_pct",
    "aerial_duels_won_pct",
    "total_duels_won_pct",
    "Accurate passes",
    "Total saves",
    "Clean sheets",
    "Penalties saved",
    "Saves from inside box",
    "Runs out",
    "Average Sofascore Rating",
    "Fouls",
]

P90_TO_COMPUTE = {
    "Goals_p90": "Goals",
    "Assists_p90": "Assists",
    "Key_passes_p90": "Key passes",
    "Total_shots_p90": "Total shots",
    "xG_p90": "Expected goals (xG)",
    "Succ_dribbles_p90": "Succ. dribbles",
    "Big_chances_created_p90": "Big chances created",
    "Tackles_p90": "Tackles",
    "Interceptions_p90": "Interceptions",
    "Clearances_p90": "Clearances",
    "Blocked_shots_p90": "Blocked shots",
    "Accurate_passes_p90": "Accurate passes",
    "Total_saves_p90": "Total saves",
    "Saves_from_inside_box_p90": "Saves from inside box",
    "fouls_p90": "Fouls",
}

MOJIBAKE_REPLACEMENTS = {
    "Ã¡": "á",
    "Ã©": "é",
    "Ã­": "í",
    "Ã³": "ó",
    "Ãº": "ú",
    "Ã": "Á",
    "Ã‰": "É",
    "Ã": "Í",
    "Ã“": "Ó",
    "Ãš": "Ú",
    "Ã±": "ñ",
    "Ã‘": "Ñ",
    "Ã¼": "ü",
    "Ãœ": "Ü",
    "â‚¬": "€",
    "â€“": "–",
    "â€”": "—",
    "â€˜": "'",
    "â€™": "'",
    "â€œ": '"',
    "â€": '"',
    "Â ": " ",
    "Â": "",
}

# ============================================================================
# UTILIDADES DE DEBUG / INTEGRIDAD CSV
# ============================================================================

def _debug_raw_file(data_path: Path, sample_lines: int = 3) -> None:
    print("\n" + "=" * 80)
    print("[DEBUG CSV] PRIMERAS LÍNEAS DEL ARCHIVO")
    print("=" * 80)
    try:
        with open(data_path, "r", encoding="utf-8", errors="replace") as f:
            for i in range(sample_lines):
                line = f.readline()
                print(f"[LINE {i}] {line.rstrip()}")
    except Exception as e:
        print(f"[DEBUG CSV] Error leyendo primeras líneas: {e}")

    print("\n" + "=" * 80)
    print("[DEBUG CSV] LÍNEA DE MBAPPÉ EN BRUTO")
    print("=" * 80)
    try:
        with open(data_path, "r", encoding="utf-8", errors="replace") as f:
            found = False
            for i, line in enumerate(f):
                if "Mbapp" in line or "MbappÃ" in line:
                    print(f"[RAW LINE {i}] {line.rstrip()}")
                    found = True
                    break
            if not found:
                print("[DEBUG CSV] No se encontró ninguna línea con 'Mbapp'.")
    except Exception as e:
        print(f"[DEBUG CSV] Error buscando línea de Mbappé: {e}")


def _validate_raw_csv_structure(data_path: Path, delimiter: str = ",", quotechar: str = '"', max_bad_lines: int = 20) -> None:
    """
    Valida que las filas del CSV tengan el mismo número de columnas
    que la cabecera. Imprime las líneas potencialmente corruptas.
    """
    print("\n" + "=" * 80)
    print("[DEBUG CSV] VALIDANDO ESTRUCTURA DEL CSV")
    print("=" * 80)

    try:
        with open(data_path, "r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
            rows = list(reader)

        if not rows:
            print("[DEBUG CSV] Archivo vacío.")
            return

        header = rows[0]
        expected_len = len(header)
        print(f"[DEBUG CSV] Columnas esperadas según cabecera: {expected_len}")
        print(f"[DEBUG CSV] Cabecera: {header}")

        bad_count = 0
        for i, row in enumerate(rows[1:], start=2):  # línea humana desde 2
            if len(row) != expected_len:
                print(f"[BAD LINE {i}] columnas={len(row)} esperado={expected_len}")
                print(row)
                bad_count += 1
                if bad_count >= max_bad_lines:
                    print(f"[DEBUG CSV] Se alcanzó el máximo de {max_bad_lines} líneas problemáticas mostradas.")
                    break

        if bad_count == 0:
            print("[DEBUG CSV] No se detectaron filas con anchura inconsistente.")
    except Exception as e:
        print(f"[DEBUG CSV] Error validando estructura: {e}")


# ============================================================================
# LIMPIEZA / VALIDACIÓN DE DATAFRAME
# ============================================================================

def _validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "[data_loader] Faltan columnas críticas en el CSV:\n"
            + "\n".join(f"  · {c}" for c in missing)
        )


def _warn_optional_columns(df: pd.DataFrame) -> None:
    missing = [c for c in OPTIONAL_BUT_USEFUL_COLUMNS if c not in df.columns]
    if missing:
        logger.warning(
            "[data_loader] Faltan %d columnas opcionales: %s",
            len(missing),
            ", ".join(missing)
        )


def _fix_mojibake_text(value):
    if not isinstance(value, str):
        return value
    s = value
    for bad, good in MOJIBAKE_REPLACEMENTS.items():
        s = s.replace(bad, good)
    return s.strip()


def _clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    obj_cols = df.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        df[col] = df[col].apply(_fix_mojibake_text)
    return df


def _safe_numeric(series: pd.Series) -> pd.Series:
    """
    Conversión robusta a numérico.
    No intenta parsear correctamente 'valor_mercado' porque esa se trata aparte.
    """
    if series.dtype == object:
        cleaned = (
            series.astype(str)
            .str.replace("%", "", regex=False)
            .str.replace("\xa0", "", regex=False)
            .str.strip()
            .str.replace(",", ".", regex=False)
        )
        return pd.to_numeric(cleaned, errors="coerce")
    return pd.to_numeric(series, errors="coerce")


def _convert_numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = _safe_numeric(df[col])
    return df


def _compute_p90_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    mp = pd.to_numeric(df["minutes_played"], errors="coerce").replace(0, np.nan)
    minutes_90 = mp / 90.0

    for new_col, source_col in P90_TO_COMPUTE.items():
        if source_col in df.columns:
            source = pd.to_numeric(df[source_col], errors="coerce")
            df[new_col] = source / minutes_90
        else:
            df[new_col] = np.nan

    return df


def _add_derived_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "es_portero" not in df.columns:
        if "posicion" in df.columns:
            df["es_portero"] = df["posicion"].astype(str).str.lower().eq("portero")
        else:
            df["es_portero"] = False

    if "muestra_fiable" not in df.columns:
        if "minutes_played" in df.columns:
            df["muestra_fiable"] = df["minutes_played"] >= 900
        else:
            df["muestra_fiable"] = False

    return df


def _deduplicate_players(df: pd.DataFrame) -> pd.DataFrame:
    """
    Si hay duplicados por Name, prioriza la fila con más minutos.
    """
    if "Name" not in df.columns:
        return df

    if df["Name"].duplicated().any():
        logger.warning("[data_loader] Detectados duplicados por Name. Se prioriza mayor minutes_played.")
        if "minutes_played" in df.columns:
            df = df.sort_values(["Name", "minutes_played"], ascending=[True, False])
        else:
            df = df.sort_values(["Name"], ascending=[True])

        df = df.drop_duplicates(subset=["Name"], keep="first").copy()

    return df


def _log_dataset_summary(df: pd.DataFrame) -> None:
    logger.info("=" * 60)
    logger.info("DATASET CARGADO — RESUMEN")
    logger.info("Jugadores: %d", len(df))

    if "liga" in df.columns:
        logger.info("Ligas: %s", sorted(df["liga"].dropna().unique().tolist()))

    if "posicion" in df.columns:
        for pos, cnt in df["posicion"].value_counts().items():
            logger.info("  %-18s %d", pos, cnt)

    logger.info("=" * 60)


def _debug_player_row(df: pd.DataFrame, player_name: str = "Mbapp", stage: str = "UNKNOWN") -> None:
    """
    Imprime en consola las filas que coincidan con el texto.
    """
    print("\n" + "=" * 80)
    print(f"[DEBUG PLAYER] BUSCANDO '{player_name}' EN ETAPA: {stage}")
    print("=" * 80)

    if "Name" not in df.columns:
        print("[DEBUG PLAYER] La columna Name no existe.")
        return

    subset = df[df["Name"].astype(str).str.contains(player_name, case=False, na=False)].copy()

    if subset.empty:
        print(f"[DEBUG PLAYER] No se encontró '{player_name}'.")
        return

    cols_interest = [c for c in [
        "Name", "tm_club", "liga", "posicion", "edad", "minutes_played",
        "Goals", "Assists", "Expected goals (xG)", "Key passes", "Succ. dribbles",
        "Tackles", "Interceptions", "Clearances", "Accurate passes %",
        "ground_duels_won_pct", "aerial_duels_won_pct", "Average Sofascore Rating",
        "Goals_p90", "Assists_p90", "xG_p90", "Key_passes_p90", "Succ_dribbles_p90"
    ] if c in subset.columns]

    print(subset[cols_interest].T)


# ============================================================================
# LECTURA PRINCIPAL
# ============================================================================

def load_data(
    path: Path | str | None = None,
    min_minutes: int = DEFAULT_MIN_MINUTES,
    require_position: bool = True,
    deduplicate_players: bool = True,
    debug_csv: bool = False,
) -> pd.DataFrame:
    data_path = Path(path) if path else DEFAULT_DATA_PATH

    if not data_path.exists():
        raise FileNotFoundError(f"[data_loader] CSV no encontrado en: {data_path}")

    logger.info("[data_loader] Leyendo CSV desde %s", data_path)

    if debug_csv:
        _debug_raw_file(data_path)
        _validate_raw_csv_structure(data_path)

    # Lectura explícita
    try:
        df = pd.read_csv(
            data_path,
            encoding="utf-8",
            sep=",",
            quotechar='"',
            engine="python",
            on_bad_lines="warn",
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            data_path,
            encoding="latin1",
            sep=",",
            quotechar='"',
            engine="python",
            on_bad_lines="warn",
        )

    _validate_columns(df)
    _warn_optional_columns(df)

    if debug_csv:
        _debug_player_row(df, player_name="Mbapp", stage="POST_READ_CSV_RAW")

    df = _clean_text_columns(df)

    if debug_csv:
        _debug_player_row(df, player_name="Mbapp", stage="POST_TEXT_CLEAN")

    df = _convert_numeric_columns(df, RAW_NUMERIC_COLUMNS)

    if debug_csv:
        _debug_player_row(df, player_name="Mbapp", stage="POST_NUMERIC_CONVERSION")

    before = len(df)
    df = df[df["minutes_played"] >= min_minutes].copy()
    logger.info(
        "[data_loader] Filtro minutes_played >= %d: %d → %d",
        min_minutes, before, len(df)
    )

    if require_position and "posicion" in df.columns:
        before = len(df)
        df = df[df["posicion"].notna()].copy()
        logger.info(
            "[data_loader] Filtro posicion notna: %d → %d",
            before, len(df)
        )

    if deduplicate_players:
        before = len(df)
        df = _deduplicate_players(df)
        logger.info(
            "[data_loader] Deduplicación por Name: %d → %d",
            before, len(df)
        )

    if df.empty:
        raise ValueError("[data_loader] El DataFrame está vacío tras aplicar filtros.")

    df = _compute_p90_features(df)
    df = _add_derived_flags(df)

    if debug_csv:
        _debug_player_row(df, player_name="Mbapp", stage="FINAL_AFTER_DERIVED_FEATURES")

    _log_dataset_summary(df)
    return df.reset_index(drop=True)


# ============================================================================
# HELPERS REUTILIZABLES PARA LA APP
# ============================================================================

def parse_market_value_to_millions(val_str) -> float:
    if pd.isna(val_str) or str(val_str).strip() in ["-", "", "nan"]:
        return np.nan

    s = _fix_mojibake_text(str(val_str)).lower()
    s = s.replace("€", "").replace("\xa0", " ").strip()

    # "200,00 mill. €" -> 200.00
    # Primero quitamos puntos de miles, luego cambiamos coma decimal
    s = s.replace(".", "").replace(",", ".")

    if "mill" in s:
        s = s.replace("mill.", "").replace("mill", "").strip()
        try:
            return float(s)
        except Exception:
            return np.nan

    if "mil" in s:
        s = s.replace("mil", "").strip()
        try:
            return float(s) / 1000
        except Exception:
            return np.nan

    try:
        return float(s)
    except Exception:
        return np.nan


def parse_contract_year(val) -> int | None:
    if pd.isna(val) or str(val).strip() in ["-", "", "nan"]:
        return None

    s = _fix_mojibake_text(str(val)).strip()
    year = s[-4:]
    if year.isdigit():
        return int(year)
    return None


def filter_by_market_value(df: pd.DataFrame, max_value_m: float) -> pd.DataFrame:
    if "valor_mercado" not in df.columns:
        return df.copy()

    tmp = df.copy()
    tmp["_market_value_m"] = tmp["valor_mercado"].apply(parse_market_value_to_millions)
    return tmp[tmp["_market_value_m"].fillna(np.inf) <= max_value_m].copy()


def filter_by_contract_year(df: pd.DataFrame, year_min: int, year_max: int) -> pd.DataFrame:
    if "fin_contrato" not in df.columns:
        return df.copy()

    tmp = df.copy()
    tmp["_contract_year"] = tmp["fin_contrato"].apply(parse_contract_year)
    return tmp[tmp["_contract_year"].apply(lambda y: y is not None and year_min <= y <= year_max)].copy()


def get_rayo_squad(df: pd.DataFrame) -> pd.DataFrame:
    if "tm_club" not in df.columns:
        return pd.DataFrame(columns=df.columns)

    return df[df["tm_club"].astype(str).str.contains("Rayo", case=False, na=False)].copy()


def get_contract_opportunities(df: pd.DataFrame, max_year: int = 2026) -> pd.DataFrame:
    if "fin_contrato" not in df.columns:
        return pd.DataFrame(columns=df.columns)

    tmp = df.copy()
    tmp["_contract_year"] = tmp["fin_contrato"].apply(parse_contract_year)
    tmp = tmp[tmp["_contract_year"].notna()]
    tmp = tmp[tmp["_contract_year"] <= max_year]

    if "edad" in tmp.columns:
        return tmp.sort_values(["_contract_year", "edad"], ascending=[True, True]).copy()
    return tmp.sort_values("_contract_year", ascending=True).copy()