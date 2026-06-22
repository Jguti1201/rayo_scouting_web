import json
import logging
import sys
from datetime import datetime

import boto3
import numpy as np
import pandas as pd


# =========================================================
# LOGGING
# =========================================================

log_filename = f"rayo_analisis_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)

log = logging.getLogger(__name__)


# =========================================================
# CONFIG S3
# =========================================================

BUCKET = "rayo-scout-data"
PREFIX = (
    "ligas_seleccionadas/"
    "testeo_ligas_europa/"
    "Spain_Primera_Division/"
    "2025-2026/"
    "partidos/"
)
REGION = "eu-west-1"


# =========================================================
# CARGA DE CATÁLOGOS
# =========================================================

def load_catalogs(event_types_csv: str, qualifier_types_csv: str):
    event_types_df = pd.read_csv(event_types_csv)
    qualifier_types_df = pd.read_csv(qualifier_types_csv)

    event_types_clean = (
        event_types_df
        .dropna(subset=["eventTypeId"])
        .drop_duplicates(subset=["eventTypeId"])
        .copy()
    )
    event_types_clean["eventTypeId"] = event_types_clean["eventTypeId"].astype(int)

    qualifier_types_clean = (
        qualifier_types_df
        .dropna(subset=["qualifierTypeId"])
        .drop_duplicates(subset=["qualifierTypeId"])
        .copy()
    )
    qualifier_types_clean["qualifierTypeId"] = qualifier_types_clean["qualifierTypeId"].astype(int)

    type_map = dict(zip(event_types_clean["eventTypeId"], event_types_clean["eventTypeName"]))
    qual_map = dict(zip(qualifier_types_clean["qualifierTypeId"], qualifier_types_clean["qualifierTypeName"]))

    return type_map, qual_map


# =========================================================
# FUNCIONES AUXILIARES
# =========================================================

def has_qualifier(dataframe: pd.DataFrame, q_id: int) -> pd.Series:
    col = f"q_{q_id}"
    if col in dataframe.columns:
        return dataframe[col].notna() & (dataframe[col] != "0") & (dataframe[col] != 0)
    return pd.Series(False, index=dataframe.index)


def safe_mean(series: pd.Series):
    return series.mean() if len(series) > 0 else np.nan


def safe_pct(num, den):
    return (num / den * 100) if den and den > 0 else np.nan


def detect_rayo_team_name(team_names):
    for t in team_names:
        if isinstance(t, str) and "Rayo" in t:
            return t
    return None


def classify_event_family(type_name: str) -> str:
    if type_name in ["Pass", "Offside Pass"]:
        return "Pase"
    elif type_name in ["Goal", "Miss", "Saved Shot", "Post", "Chance missed"]:
        return "Finalización"
    elif type_name in ["Take On", "Dispossessed"]:
        return "1v1 / Regate"
    elif type_name in ["Tackle", "Interception", "Ball recovery", "Blocked Pass", "Clearance"]:
        return "Acción defensiva"
    elif type_name in ["Aerial", "Challenge", "50/50"]:
        return "Duelo"
    elif type_name in ["Foul", "Card"]:
        return "Disciplina / Faltas"
    elif type_name in ["Save", "Claim", "Punch", "Keeper Sweeper", "Keeper pick-up", "Smother", "Penalty faced"]:
        return "Portero"
    elif type_name in ["Turnover", "Error", "Ball touch"]:
        return "Pérdida / Error"
    else:
        return "Otros"


# =========================================================
# S3
# =========================================================

def get_s3_client(region_name: str = REGION):
    return boto3.client("s3", region_name=region_name)


def list_json_keys_from_s3(bucket: str, prefix: str, region_name: str = REGION):
    s3 = get_s3_client(region_name=region_name)
    paginator = s3.get_paginator("list_objects_v2")

    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        contents = page.get("Contents", [])
        for obj in contents:
            key = obj["Key"]
            if key.endswith(".json"):
                keys.append(key)

    log.info("JSONs detectados en S3: %s", len(keys))
    return keys


def filter_rayo_keys(json_keys, team_keyword="Rayo"):
    filtered = []
    for key in json_keys:
        filename = key.split("/")[-1].lower()
        if team_keyword.lower() in filename:
            filtered.append(key)

    log.info("JSONs filtrados por nombre que contienen '%s': %s", team_keyword, len(filtered))
    return filtered


def read_json_from_s3(bucket: str, key: str, region_name: str = REGION):
    s3 = get_s3_client(region_name=region_name)
    response = s3.get_object(Bucket=bucket, Key=key)
    content = response["Body"].read().decode("utf-8")
    return json.loads(content)


# =========================================================
# ENRIQUECIMIENTO ESPACIAL
# =========================================================

def add_pitch_zones(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade:
    - tercio
    - carril de 5
    - zona combinada
    """
    df = df.copy()

    def third(x):
        if pd.isna(x):
            return "Unknown"
        if x < 33.33:
            return "Defensive Third"
        elif x < 66.67:
            return "Middle Third"
        return "Attacking Third"

    def lane_5(y):
        if pd.isna(y):
            return "Unknown"
        if y < 20:
            return "Left Wing"
        elif y < 40:
            return "Left Halfspace"
        elif y < 60:
            return "Central"
        elif y < 80:
            return "Right Halfspace"
        return "Right Wing"

    df["pitch_third"] = df["x"].apply(third)
    df["pitch_lane_5"] = df["y"].apply(lane_5)
    df["zone_15"] = df["pitch_third"] + " | " + df["pitch_lane_5"]

    return df


def add_pass_end_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrae coordenadas de final de pase desde qualifiers:
    - q_140 = Pass End X
    - q_141 = Pass End Y
    """
    df = df.copy()

    if "q_140" in df.columns:
        df["pass_end_x"] = pd.to_numeric(df["q_140"], errors="coerce")
    else:
        df["pass_end_x"] = np.nan

    if "q_141" in df.columns:
        df["pass_end_y"] = pd.to_numeric(df["q_141"], errors="coerce")
    else:
        df["pass_end_y"] = np.nan

    return df


def add_shot_zone(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clasificación simple de tiros por zona usando qualifiers o fallback espacial.
    """
    df = df.copy()

    def shot_zone(row):
        q = row.to_dict()

        # zonas Opta si vienen
        if pd.notna(q.get("q_16")) or pd.notna(q.get("q_60")) or pd.notna(q.get("q_61")):
            return "Six-yard box"
        if pd.notna(q.get("q_17")) or pd.notna(q.get("q_62")) or pd.notna(q.get("q_63")) or pd.notna(q.get("q_64")) or pd.notna(q.get("q_65")):
            return "Penalty box"
        if pd.notna(q.get("q_18")) or pd.notna(q.get("q_66")) or pd.notna(q.get("q_67")) or pd.notna(q.get("q_68")) or pd.notna(q.get("q_69")):
            return "Outside box"
        if pd.notna(q.get("q_19")) or pd.notna(q.get("q_70")) or pd.notna(q.get("q_71")):
            return "Long range"

        # fallback por coordenadas
        x = row.get("x")
        y = row.get("y")

        if pd.isna(x) or pd.isna(y):
            return "Unknown"

        if x >= 88 and 40 <= y <= 60:
            return "Six-yard box"
        elif x >= 80 and 18 <= y <= 82:
            return "Penalty box"
        elif x >= 66:
            return "Outside box"
        else:
            return "Long range"

    df["shot_zone"] = df.apply(shot_zone, axis=1)
    return df


# =========================================================
# XG PROXY
# =========================================================

def estimate_xg_proxy(shots_df: pd.DataFrame) -> pd.DataFrame:
    """
    xG estimado aproximado con reglas heurísticas.
    No es xG oficial.
    """
    shots_df = shots_df.copy()

    def shot_xg(row):
        zone = row.get("shot_zone", "Unknown")
        is_penalty = pd.notna(row.get("q_9"))
        is_big_chance = pd.notna(row.get("q_214"))
        is_header = pd.notna(row.get("q_15"))
        is_blocked = pd.notna(row.get("q_82"))

        if is_penalty:
            base = 0.76
        elif zone == "Six-yard box":
            base = 0.32
        elif zone == "Penalty box":
            base = 0.16
        elif zone == "Outside box":
            base = 0.06
        elif zone == "Long range":
            base = 0.03
        else:
            base = 0.05

        if is_big_chance:
            base += 0.10
        if is_header:
            base -= 0.03
        if is_blocked:
            base -= 0.01

        return max(min(base, 0.95), 0.01)

    shots_df["xg_proxy"] = shots_df.apply(shot_xg, axis=1)
    return shots_df


# =========================================================
# PARSEAR PARTIDO
# =========================================================

def parse_single_match(match_data: dict, type_map: dict, source_key: str | None = None) -> pd.DataFrame | None:
    match_info = match_data.get("matchInfo", {})
    live_data = match_data.get("liveData", {})

    fallback_match_id = source_key.split("/")[-1].replace(".json", "") if source_key else None
    match_id = match_info.get("id", match_data.get("matchId", match_data.get("id", fallback_match_id)))

    contestants = match_info.get("contestant", [])
    if len(contestants) < 2:
        log.warning("Partido sin contestants válidos. source_key=%s", source_key)
        return None

    home_team = None
    away_team = None

    for c in contestants:
        position = str(c.get("position", "")).lower()
        if position == "home":
            home_team = c
        elif position == "away":
            away_team = c

    if home_team is None or away_team is None:
        if len(contestants) >= 2:
            home_team = contestants[0]
            away_team = contestants[1]

    if home_team is None or away_team is None:
        log.warning("No se pudo identificar local/visitante. source_key=%s", source_key)
        return None

    home_id = home_team.get("id")
    away_id = away_team.get("id")
    home_name = home_team.get("name")
    away_name = away_team.get("name")

    rayo_name = detect_rayo_team_name([home_name, away_name])
    if rayo_name is None:
        log.info("Partido descartado: no participa el Rayo. source_key=%s", source_key)
        return None

    rayo_is_home = home_name == rayo_name
    rival_name = away_name if rayo_is_home else home_name

    competition = match_info.get("competition", {})
    competition_name = (
        competition.get("name")
        or competition.get("knownName")
        or match_data.get("competitionName")
        or "Unknown Competition"
    )

    match_date = (
        match_info.get("date")
        or match_info.get("localDate")
        or match_data.get("date")
        or None
    )

    # Resultado final
    match_details = live_data.get("matchDetails", {})
    scores = match_details.get("scores", {})
    ft_scores = scores.get("ft", scores.get("total", {}))
    ft_home = ft_scores.get("home")
    ft_away = ft_scores.get("away")

    raw_events = live_data.get("event", [])
    if len(raw_events) == 0:
        log.warning("Partido sin eventos. source_key=%s", source_key)
        return None

    rows = []
    for e in raw_events:
        row = {
            "source_key": source_key,
            "match_id": match_id,
            "match_date": match_date,
            "competition": competition_name,
            "home_team": home_name,
            "away_team": away_name,
            "is_home": rayo_is_home,
            "rayo_team": rayo_name,
            "rival_team": rival_name,
            "ft_home": ft_home,
            "ft_away": ft_away,
            "id": e.get("id"),
            "eventId": e.get("eventId"),
            "typeId": e.get("typeId"),
            "periodId": e.get("periodId"),
            "timeMin": e.get("timeMin"),
            "timeSec": e.get("timeSec"),
            "contestantId": e.get("contestantId"),
            "playerId": e.get("playerId"),
            "playerName": e.get("playerName"),
            "outcome": e.get("outcome"),
            "x": e.get("x"),
            "y": e.get("y"),
            "timeStamp": e.get("timeStamp"),
            "lastModified": e.get("lastModified"),
        }

        for q in e.get("qualifier", []):
            qid = q.get("qualifierId")
            if qid is not None:
                row[f"q_{qid}"] = q.get("value")

        rows.append(row)

    df_match = pd.DataFrame(rows)

    if df_match.empty:
        log.warning("DataFrame vacío tras parsear eventos. source_key=%s", source_key)
        return None

    df_match["typeName"] = df_match["typeId"].map(type_map)
    df_match["team"] = df_match["contestantId"].map({
        home_id: home_name,
        away_id: away_name
    })
    df_match["is_rayo_event"] = df_match["team"] == rayo_name
    df_match["timestamp"] = df_match["timeMin"].fillna(0) * 60 + df_match["timeSec"].fillna(0)

    df_match = df_match[df_match["periodId"].isin([1, 2])].copy()
    df_match = df_match.sort_values(["periodId", "timeMin", "timeSec", "id"]).reset_index(drop=True)

    df_match["event_family"] = df_match["typeName"].apply(classify_event_family)
    df_match = add_pitch_zones(df_match)
    df_match = add_pass_end_coordinates(df_match)

    log.info(
        "Parseado partido match_id=%s | rayo=%s | rival=%s | is_home=%s | eventos=%s",
        match_id, rayo_name, rival_name, rayo_is_home, len(df_match)
    )

    return df_match


# =========================================================
# KPIs AVANZADOS POR PARTIDO
# =========================================================

def compute_match_kpis(df_match: pd.DataFrame) -> dict | None:
    if df_match is None or df_match.empty:
        return None

    required_cols = ["rayo_team", "rival_team", "is_home", "team", "typeId", "outcome", "x", "y"]
    missing = [c for c in required_cols if c not in df_match.columns]
    if missing:
        log.warning("compute_match_kpis: faltan columnas requeridas: %s", missing)
        return None

    try:
        rayo_team = df_match["rayo_team"].dropna().iloc[0]
        rival_team = df_match["rival_team"].dropna().iloc[0]
        is_home = df_match["is_home"].iloc[0]
        competition = df_match["competition"].iloc[0] if "competition" in df_match.columns else None
        match_id = df_match["match_id"].iloc[0] if "match_id" in df_match.columns else None
        match_date = df_match["match_date"].iloc[0] if "match_date" in df_match.columns else None
        source_key = df_match["source_key"].iloc[0] if "source_key" in df_match.columns else None

        rayo_df = df_match[df_match["team"] == rayo_team].copy()
        rival_df = df_match[df_match["team"] == rival_team].copy()

        if rayo_df.empty or rival_df.empty:
            return None

        # ----------------------------------------------------
        # Resultado
        # ----------------------------------------------------
        ft_home = df_match["ft_home"].iloc[0] if "ft_home" in df_match.columns else np.nan
        ft_away = df_match["ft_away"].iloc[0] if "ft_away" in df_match.columns else np.nan

        goals_for = ft_home if is_home else ft_away
        goals_against = ft_away if is_home else ft_home
        goal_diff = goals_for - goals_against if pd.notna(goals_for) and pd.notna(goals_against) else np.nan

        if pd.isna(goal_diff):
            result_label = "Unknown"
            result_points = np.nan
        elif goal_diff > 0:
            result_label = "W"
            result_points = 3
        elif goal_diff == 0:
            result_label = "D"
            result_points = 1
        else:
            result_label = "L"
            result_points = 0

        # ----------------------------------------------------
        # Pases
        # ----------------------------------------------------
        passes = rayo_df[rayo_df["typeId"] == 1].copy()
        passes_success = passes[passes["outcome"] == 1].copy()

        pass_accuracy_pct = safe_pct(len(passes_success), len(passes))
        passes_total = len(passes)

        passes_def_third = len(passes[(passes["x"] < 33.33)])
        passes_mid_third = len(passes[(passes["x"] >= 33.33) & (passes["x"] < 66.67)])
        passes_att_third = len(passes[(passes["x"] >= 66.67)])

        if "pass_end_x" in passes.columns:
            progressive_passes = len(
                passes_success[
                    passes_success["pass_end_x"].notna() &
                    ((passes_success["pass_end_x"] - passes_success["x"]) >= 10)
                ]
            )
            backward_lateral_passes = len(
                passes_success[
                    passes_success["pass_end_x"].notna() &
                    ((passes_success["pass_end_x"] - passes_success["x"]) < 5)
                ]
            )
            entries_final_third_by_pass = len(
                passes_success[
                    passes_success["pass_end_x"].notna() &
                    (passes_success["pass_end_x"] >= 66.67) &
                    (passes_success["x"] < 66.67)
                ]
            )
            entries_penalty_area_by_pass = len(
                passes_success[
                    passes_success["pass_end_x"].notna() &
                    (passes_success["pass_end_x"] >= 83) &
                    (passes_success["pass_end_y"].between(21, 79, inclusive="both"))
                ]
            )
        else:
            progressive_passes = len(passes_success[(passes_success["x"] > 40) & (passes_success["x"] <= 70)])
            backward_lateral_passes = len(passes_success) - progressive_passes
            entries_final_third_by_pass = len(passes_success[passes_success["x"] >= 66.67])
            entries_penalty_area_by_pass = len(passes_success[(passes_success["x"] >= 83) & (passes_success["y"].between(21, 79, inclusive="both"))])

        verticality_ratio = progressive_passes / backward_lateral_passes if backward_lateral_passes > 0 else np.nan

        avg_circulation_height = safe_mean(passes["x"])
        field_tilt_pct = safe_pct(
            len(rayo_df[(rayo_df["typeId"] == 1) & (rayo_df["x"] >= 66.67)]),
            len(rayo_df[(rayo_df["typeId"] == 1) & (rayo_df["x"] >= 66.67)]) + len(rival_df[(rival_df["typeId"] == 1) & (rival_df["x"] >= 66.67)])
        )

        # centros
        crosses = passes[has_qualifier(passes, 2)].copy()
        crosses_completed = len(crosses[crosses["outcome"] == 1])
        crosses_attempted = len(crosses)

        crosses_from_left = len(crosses[crosses["y"] < 50])
        crosses_from_right = len(crosses[crosses["y"] >= 50])

        # key pass proxy = pase con qualifier assist (210) o relacionado con tiro
        key_passes_proxy = len(
            passes[
                has_qualifier(passes, 210) |
                has_qualifier(passes, 29) |
                has_qualifier(passes, 154) |
                has_qualifier(passes, 218)
            ]
        )

        # progressive carries proxy = take on exitoso en avance
        take_ons = rayo_df[rayo_df["typeId"] == 3].copy()
        progressive_carries_proxy = len(take_ons[(take_ons["outcome"] == 1) & (take_ons["x"] >= 40)])

        # ----------------------------------------------------
        # Tiros y finalización
        # ----------------------------------------------------
        shots = rayo_df[rayo_df["typeId"].isin([13, 14, 15, 16])].copy()
        shots = add_shot_zone(shots)
        shots = estimate_xg_proxy(shots)

        shots_total = len(shots)
        shots_six_yard_box = len(shots[shots["shot_zone"] == "Six-yard box"])
        shots_box = len(shots[shots["shot_zone"].isin(["Six-yard box", "Penalty box"])])
        shots_outside_box = len(shots[shots["shot_zone"].isin(["Outside box", "Long range"])])

        goals_from_events = len(shots[shots["typeId"] == 16])
        xg_proxy_total = shots["xg_proxy"].sum() if "xg_proxy" in shots.columns else np.nan
        xg_proxy_per_shot = shots["xg_proxy"].mean() if len(shots) > 0 else np.nan
        goals_minus_xg_proxy = goals_from_events - xg_proxy_total if pd.notna(xg_proxy_total) else np.nan

        # ----------------------------------------------------
        # Tipología de ataques
        # ----------------------------------------------------
        df_match = df_match.sort_values(["periodId", "timeMin", "timeSec", "id"]).reset_index(drop=True)
        df_match["prev_team"] = df_match["team"].shift(1)
        df_match["possession_change"] = (df_match["team"] != df_match["prev_team"]).fillna(True)
        df_match["possession_id"] = df_match["possession_change"].cumsum()

        possessions = (
            df_match.groupby("possession_id")
            .agg(
                team=("team", "first"),
                start_ts=("timestamp", "min"),
                end_ts=("timestamp", "max"),
                n_events=("id", "count"),
                n_passes=("typeId", lambda s: (s == 1).sum())
            )
            .reset_index()
        )
        possessions["duration"] = possessions["end_ts"] - possessions["start_ts"]

        rayo_possessions = possessions[possessions["team"] == rayo_team].copy()

        fast_attacks = len(rayo_possessions[(rayo_possessions["duration"] <= 10) & (rayo_possessions["n_passes"] <= 5)])
        positional_attacks = len(rayo_possessions[(rayo_possessions["n_passes"] >= 15)])

        # ----------------------------------------------------
        # Defensa y presión
        # ----------------------------------------------------
        recoveries = rayo_df[rayo_df["typeId"] == 49].copy()
        avg_recovery_x = safe_mean(recoveries["x"])
        high_recoveries = len(recoveries[recoveries["x"] >= 60])

        rival_passes_open = rival_df[
            (rival_df["typeId"] == 1) &
            (~(has_qualifier(rival_df, 5) | has_qualifier(rival_df, 6) | has_qualifier(rival_df, 107))) &
            (rival_df["outcome"] == 1)
        ].copy()

        actions_def_high = rayo_df[
            (rayo_df["typeId"].isin([4, 7, 8, 12, 74])) &
            (rayo_df["x"] >= 40)
        ].copy()

        ppda = (len(rival_passes_open[rival_passes_open["x"] <= 60]) / len(actions_def_high)) if len(actions_def_high) > 0 else np.nan

        actions_def_final_third = rayo_df[
            (rayo_df["typeId"].isin([4, 7, 8, 12, 74])) &
            (rayo_df["x"] >= 66.67)
        ].copy()
        rival_passes_final_third = rival_passes_open[rival_passes_open["x"] <= 66.67]
        ppda_final_third = (len(rival_passes_final_third) / len(actions_def_final_third)) if len(actions_def_final_third) > 0 else np.nan

        tackles = rayo_df[rayo_df["typeId"] == 7].copy()
        tackle_success_pct = safe_pct(len(tackles[tackles["outcome"] == 1]), len(tackles))
        interceptions = len(rayo_df[rayo_df["typeId"] == 8])

        # recoveries after loss within 5s
        recoveries_5s_after_loss = 0
        counterpress_5s_success = 0
        losses_total = 0
        losses_own_half = 0

        N = len(df_match)
        for i in range(N - 1):
            row = df_match.iloc[i]
            ts = row["timestamp"]

            es_perdida_rayo = (row["team"] == rayo_team) and (
                ((row["typeId"] == 1) and (row["outcome"] == 0)) or
                ((row["typeId"] == 3) and (row["outcome"] == 0)) or
                (row["typeName"] in ["Turnover", "Error", "Dispossessed", "Ball touch"])
            )

            if es_perdida_rayo:
                losses_total += 1
                if pd.notna(row["x"]) and row["x"] < 50:
                    losses_own_half += 1

                post_loss = df_match[
                    (df_match["timestamp"] > ts) &
                    (df_match["timestamp"] <= ts + 5)
                ].copy()

                recov_post = post_loss[
                    (post_loss["team"] == rayo_team) &
                    (post_loss["typeId"] == 49)
                ]
                actions_post = post_loss[
                    (post_loss["team"] == rayo_team) &
                    (post_loss["typeId"].isin([4, 7, 8, 12, 74]))
                ]

                if len(recov_post) > 0:
                    recoveries_5s_after_loss += 1
                if len(actions_post) > 0:
                    counterpress_5s_success += 1

        counterpress_5s_pct = safe_pct(counterpress_5s_success, losses_total)

        # ----------------------------------------------------
        # Set pieces
        # ----------------------------------------------------
        set_piece_shots = len(shots[
            has_qualifier(shots, 24) |  # set piece
            has_qualifier(shots, 25) |  # from corner
            has_qualifier(shots, 26) |  # direct free kick
            has_qualifier(shots, 160)   # throw in set piece
        ])

        set_piece_goals = len(shots[
            (shots["typeId"] == 16) &
            (has_qualifier(shots, 24) | has_qualifier(shots, 25) | has_qualifier(shots, 26) | has_qualifier(shots, 160))
        ])

        corners_for = len(rayo_df[rayo_df["typeId"] == 6])
        free_kick_shots = len(shots[has_qualifier(shots, 26)])

        # xg net proxy
        rival_shots = rival_df[rival_df["typeId"].isin([13, 14, 15, 16])].copy()
        rival_shots = add_shot_zone(rival_shots)
        rival_shots = estimate_xg_proxy(rival_shots)
        xg_proxy_against = rival_shots["xg_proxy"].sum() if "xg_proxy" in rival_shots.columns else np.nan
        xg_proxy_net = xg_proxy_total - xg_proxy_against if pd.notna(xg_proxy_total) and pd.notna(xg_proxy_against) else np.nan

        return {
            "source_key": source_key,
            "match_id": match_id,
            "match_date": match_date,
            "competition": competition,
            "is_home": is_home,
            "rayo_team": rayo_team,
            "rival_team": rival_team,

            "goals_for": goals_for,
            "goals_against": goals_against,
            "goal_diff": goal_diff,
            "result_label": result_label,
            "result_points": result_points,

            "passes_total": passes_total,
            "pass_accuracy_pct": pass_accuracy_pct,
            "passes_def_third": passes_def_third,
            "passes_mid_third": passes_mid_third,
            "passes_att_third": passes_att_third,
            "progressive_passes": progressive_passes,
            "backward_lateral_passes": backward_lateral_passes,
            "verticality_ratio": verticality_ratio,
            "avg_circulation_height": avg_circulation_height,
            "field_tilt_pct": field_tilt_pct,
            "entries_final_third_by_pass": entries_final_third_by_pass,
            "entries_penalty_area_by_pass": entries_penalty_area_by_pass,
            "key_passes_proxy": key_passes_proxy,
            "crosses_attempted": crosses_attempted,
            "crosses_completed": crosses_completed,
            "crosses_from_left": crosses_from_left,
            "crosses_from_right": crosses_from_right,
            "progressive_carries_proxy": progressive_carries_proxy,

            "shots_total": shots_total,
            "shots_six_yard_box": shots_six_yard_box,
            "shots_box": shots_box,
            "shots_outside_box": shots_outside_box,
            "goals_from_events": goals_from_events,
            "xg_proxy_total": xg_proxy_total,
            "xg_proxy_per_shot": xg_proxy_per_shot,
            "goals_minus_xg_proxy": goals_minus_xg_proxy,
            "fast_attacks": fast_attacks,
            "positional_attacks": positional_attacks,

            "avg_recovery_x": avg_recovery_x,
            "ppda": ppda,
            "ppda_final_third": ppda_final_third,
            "tackles": len(tackles),
            "tackle_success_pct": tackle_success_pct,
            "interceptions": interceptions,
            "recoveries_total": len(recoveries),
            "high_recoveries": high_recoveries,
            "recoveries_5s_after_loss": recoveries_5s_after_loss,

            "counterpress_5s_pct": counterpress_5s_pct,
            "losses_total": losses_total,
            "losses_own_half": losses_own_half,
            "set_piece_shots": set_piece_shots,
            "set_piece_goals": set_piece_goals,
            "corners_for": corners_for,
            "free_kick_shots": free_kick_shots,

            "xg_proxy_against": xg_proxy_against,
            "xg_proxy_net": xg_proxy_net,
        }

    except Exception as e:
        log.exception(
            "Error en compute_match_kpis para match_id=%s source_key=%s: %s",
            df_match["match_id"].iloc[0] if "match_id" in df_match.columns and not df_match.empty else None,
            df_match["source_key"].iloc[0] if "source_key" in df_match.columns and not df_match.empty else None,
            e
        )
        return None


# =========================================================
# PIPELINE PRINCIPAL
# =========================================================

def load_season_data_from_s3(
    event_types_csv: str,
    qualifier_types_csv: str,
    bucket: str = BUCKET,
    prefix: str = PREFIX,
    region_name: str = REGION,
):
    type_map, qual_map = load_catalogs(event_types_csv, qualifier_types_csv)

    json_keys = list_json_keys_from_s3(bucket=bucket, prefix=prefix, region_name=region_name)
    json_keys = filter_rayo_keys(json_keys, team_keyword="Rayo")

    if len(json_keys) == 0:
        raise ValueError("No se encontraron JSONs con 'Rayo' en el nombre dentro del prefix.")

    all_matches_dfs = []

    for key in json_keys:
        try:
            match_data = read_json_from_s3(bucket=bucket, key=key, region_name=region_name)
            parsed = parse_single_match(match_data, type_map=type_map, source_key=key)

            if parsed is not None and not parsed.empty:
                all_matches_dfs.append(parsed)
                log.info("Partido cargado correctamente: %s", key)
            else:
                log.warning("Partido vacío o descartado tras parseo: %s", key)

        except Exception as e:
            log.exception("Error procesando key %s: %s", key, e)

    if len(all_matches_dfs) == 0:
        raise ValueError("No se ha podido parsear ningún partido del Rayo desde S3.")

    events_season = pd.concat(all_matches_dfs, ignore_index=True)

    if events_season.empty:
        raise ValueError("events_season está vacío tras concatenar partidos.")

    log.info("Eventos totales cargados: %s", len(events_season))

    match_kpi_rows = []
    for match_id, df_m in events_season.groupby("match_id"):
        try:
            row = compute_match_kpis(df_m.copy())
            if row is not None:
                match_kpi_rows.append(row)
                log.info("KPIs calculados correctamente para match_id=%s", match_id)
            else:
                log.warning("compute_match_kpis devolvió None para match_id=%s", match_id)
        except Exception as e:
            log.exception("Error calculando KPIs para match_id=%s: %s", match_id, e)

    if len(match_kpi_rows) == 0:
        raise ValueError("No se pudieron calcular KPIs para ningún partido.")

    season_kpis = pd.DataFrame(match_kpi_rows)

    log.info("Partidos con KPIs calculados: %s", len(season_kpis))
    log.info("Columnas de season_kpis: %s", list(season_kpis.columns))

    return events_season, season_kpis


# =========================================================
# RESÚMENES
# =========================================================

def build_global_summary(season_kpis: pd.DataFrame) -> pd.DataFrame:
    if season_kpis is None or season_kpis.empty:
        raise ValueError("season_kpis está vacío.")
    return season_kpis.select_dtypes(include=[np.number]).mean().to_frame("media_temporada")


def build_home_away_summary(season_kpis: pd.DataFrame) -> pd.DataFrame:
    if season_kpis is None or season_kpis.empty:
        raise ValueError("season_kpis está vacío.")

    if "is_home" not in season_kpis.columns:
        raise KeyError(
            f"La columna 'is_home' no existe en season_kpis. "
            f"Columnas disponibles: {list(season_kpis.columns)}"
        )

    summary = season_kpis.groupby("is_home").mean(numeric_only=True).T
    summary.columns = ["Fuera" if c is False else "Casa" for c in summary.columns]
    return summary


def build_strengths_weaknesses(season_kpis: pd.DataFrame):
    """
    Construye fortalezas y debilidades usando umbrales internos de temporada.
    Versión robusta ante columnas ausentes.
    """
    if season_kpis is None or season_kpis.empty:
        return [], []

    avg = season_kpis.mean(numeric_only=True)

    strengths = []
    weaknesses = []

    # helpers
    def has_col(col):
        return col in season_kpis.columns

    def avg_val(col):
        return avg.get(col, np.nan)

    def median_val(col):
        if has_col(col):
            return np.nanmedian(season_kpis[col])
        return np.nan

    # -------------------------
    # Fortalezas
    # -------------------------
    if avg_val("field_tilt_pct") >= 55:
        strengths.append("Alta inclinación territorial: el equipo pasa mucho tiempo empujando al rival en su último tercio (Field Tilt alto).")

    if avg_val("ppda") <= 8:
        strengths.append("Presión activa y agresiva: el PPDA medio es bajo, señal de intervención temprana sobre la salida rival.")

    if avg_val("counterpress_5s_pct") >= 45:
        strengths.append("Buena reacción tras pérdida: activa contra-presión con frecuencia en los primeros 5 segundos.")

    if avg_val("xg_proxy_per_shot") >= 0.11:
        strengths.append("Consigue una calidad media de tiro razonable, señal de llegadas relativamente limpias.")

    if has_col("entries_penalty_area_by_pass"):
        if avg_val("entries_penalty_area_by_pass") >= median_val("entries_penalty_area_by_pass"):
            strengths.append("Capacidad para activar pases hacia el área penal con frecuencia.")

    if has_col("progressive_passes"):
        if avg_val("progressive_passes") >= median_val("progressive_passes"):
            strengths.append("Buena capacidad de progresión mediante pase vertical.")

    # -------------------------
    # Debilidades
    # -------------------------
    if has_col("losses_own_half"):
        if avg_val("losses_own_half") >= median_val("losses_own_half") + 1:
            weaknesses.append("Asume demasiado riesgo en salida: pérdidas frecuentes en campo propio.")

    if avg_val("goals_minus_xg_proxy") < -1:
        weaknesses.append("Problema de finalización: convierte por debajo de lo que su volumen/calidad aproximada de tiro sugiere.")

    if has_col("shots_outside_box") and has_col("shots_box"):
        if avg_val("shots_outside_box") > avg_val("shots_box"):
            weaknesses.append("Demasiada dependencia del tiro exterior respecto al remate dentro del área.")

    if avg_val("counterpress_5s_pct") < 30:
        weaknesses.append("Reacción tras pérdida insuficiente: la contra-presión no aparece con la frecuencia deseable.")

    if avg_val("tackle_success_pct") < 45:
        weaknesses.append("Eficiencia baja en el duelo directo: porcentaje de éxito en tackle mejorable.")

    return strengths, weaknesses


def build_tactical_identity(season_kpis: pd.DataFrame):
    """
    Asigna una etiqueta de ADN táctico basada en reglas simples.
    """
    if season_kpis is None or season_kpis.empty:
        return "ADN táctico no disponible"

    avg = season_kpis.mean(numeric_only=True)

    field_tilt = avg.get("field_tilt_pct", np.nan)
    ppda = avg.get("ppda", np.nan)
    verticality = avg.get("verticality_ratio", np.nan)
    fast_attacks = avg.get("fast_attacks", np.nan)
    positional_attacks = avg.get("positional_attacks", np.nan)
    counterpress = avg.get("counterpress_5s_pct", np.nan)

    if field_tilt >= 55 and ppda <= 8 and positional_attacks >= fast_attacks:
        return "Ataque Posicional de Alta Circulación con Presión Alta"
    if verticality >= 0.8 and fast_attacks > positional_attacks and counterpress >= 40:
        return "Bloque Reactivo de Transición Vertical con Contra-presión Activa"
    if field_tilt < 50 and fast_attacks >= positional_attacks:
        return "Modelo Reactivo de Progresión Vertical y Ataque Directo"
    if field_tilt >= 55 and counterpress < 30:
        return "Ataque Territorial con Fragilidad en Transición Defensiva"

    return "Modelo Mixto de Circulación Media y Presión Intermitente"


def build_correlation_summary(season_kpis: pd.DataFrame):
    """
    Calcula correlaciones simples contra:
    - result_points
    - xg_proxy_net

    Versión robusta: si faltan columnas, devuelve DataFrames vacíos
    en lugar de romper.
    """
    if season_kpis is None or season_kpis.empty:
        return pd.DataFrame(), pd.DataFrame()

    numeric = season_kpis.select_dtypes(include=[np.number]).copy()

    corr_points = pd.DataFrame()
    corr_xg_net = pd.DataFrame()

    if "result_points" in numeric.columns:
        corr_matrix = numeric.corr(numeric_only=True)
        if "result_points" in corr_matrix.columns:
            corr_points = corr_matrix[["result_points"]].sort_values("result_points", ascending=False)

    if "xg_proxy_net" in numeric.columns:
        corr_matrix = numeric.corr(numeric_only=True)
        if "xg_proxy_net" in corr_matrix.columns:
            corr_xg_net = corr_matrix[["xg_proxy_net"]].sort_values("xg_proxy_net", ascending=False)

    return corr_points, corr_xg_net


def build_expert_export_text(season_kpis: pd.DataFrame):
    """
    Construye un bloque de texto listo para copiar y pegar a ChatGPT / informe experto.
    Versión robusta ante columnas ausentes.
    """
    if season_kpis is None or season_kpis.empty:
        return "No hay datos suficientes."

    avg = season_kpis.mean(numeric_only=True)
    strengths, weaknesses = build_strengths_weaknesses(season_kpis)
    tactical_identity = build_tactical_identity(season_kpis)
    corr_points, corr_xg_net = build_correlation_summary(season_kpis)

    if not corr_points.empty:
        top_metrics_points = corr_points.drop(index=["result_points"], errors="ignore").head(3).index.tolist()
    else:
        top_metrics_points = []

    if not corr_xg_net.empty:
        top_metrics_xgnet = corr_xg_net.drop(index=["xg_proxy_net"], errors="ignore").head(3).index.tolist()
    else:
        top_metrics_xgnet = []

    def safe_avg(col):
        return avg.get(col, np.nan)

    text = f"""
Actúa como un Analista de Datos Tácticos de Élite y Científico de Datos de Fútbol Profesional.

A continuación te paso el resumen estructurado del equipo basado en datos eventing de toda la temporada. Quiero que determines:
1. El estilo de juego dominante del equipo.
2. Sus fortalezas estructurales.
3. Sus debilidades principales.
4. Qué métricas explican mejor sus victorias y derrotas.
5. Una etiqueta inequívoca de ADN táctico.

FASE OFENSIVA Y CONSTRUCCIÓN
- Passes total: {safe_avg('passes_total'):.2f}
- Pass accuracy %: {safe_avg('pass_accuracy_pct'):.2f}
- Passes own third: {safe_avg('passes_def_third'):.2f}
- Passes middle third: {safe_avg('passes_mid_third'):.2f}
- Passes attacking third: {safe_avg('passes_att_third'):.2f}
- Progressive passes: {safe_avg('progressive_passes'):.2f}
- Back/lateral passes: {safe_avg('backward_lateral_passes'):.2f}
- Verticality ratio: {safe_avg('verticality_ratio'):.2f}
- Avg circulation height: {safe_avg('avg_circulation_height'):.2f}
- Field tilt %: {safe_avg('field_tilt_pct'):.2f}
- Final third entries: {safe_avg('entries_final_third_by_pass'):.2f}
- Penalty area entries: {safe_avg('entries_penalty_area_by_pass'):.2f}
- Key passes proxy: {safe_avg('key_passes_proxy'):.2f}
- Crosses attempted: {safe_avg('crosses_attempted'):.2f}
- Crosses completed: {safe_avg('crosses_completed'):.2f}
- Progressive carries proxy: {safe_avg('progressive_carries_proxy'):.2f}

ÚLTIMO TERCIO Y FINALIZACIÓN
- Shots total: {safe_avg('shots_total'):.2f}
- Shots six-yard box: {safe_avg('shots_six_yard_box'):.2f}
- Shots in box: {safe_avg('shots_box'):.2f}
- Shots outside box: {safe_avg('shots_outside_box'):.2f}
- Goals: {safe_avg('goals_for'):.2f}
- xG proxy total: {safe_avg('xg_proxy_total'):.2f}
- xG proxy per shot: {safe_avg('xg_proxy_per_shot'):.3f}
- Goals - xG proxy: {safe_avg('goals_minus_xg_proxy'):.2f}
- Fast attacks: {safe_avg('fast_attacks'):.2f}
- Positional attacks: {safe_avg('positional_attacks'):.2f}

FASE DEFENSIVA Y PRESIÓN
- Avg recovery height: {safe_avg('avg_recovery_x'):.2f}
- PPDA: {safe_avg('ppda'):.2f}
- PPDA final third: {safe_avg('ppda_final_third'):.2f}
- Tackles: {safe_avg('tackles'):.2f}
- Tackle success %: {safe_avg('tackle_success_pct'):.2f}
- Interceptions: {safe_avg('interceptions'):.2f}
- Recoveries after loss in 5s: {safe_avg('recoveries_5s_after_loss'):.2f}

TRANSICIONES Y BALÓN PARADO
- Counterpress success 5s: {safe_avg('counterpress_5s_pct'):.2f}
- Losses own half: {safe_avg('losses_own_half'):.2f}
- Set-piece shots: {safe_avg('set_piece_shots'):.2f}
- Set-piece goals: {safe_avg('set_piece_goals'):.2f}
- Corners for: {safe_avg('corners_for'):.2f}
- Free-kick shots: {safe_avg('free_kick_shots'):.2f}

CORRELACIÓN CON RESULTADOS
- Top 3 metrics linked to result points: {top_metrics_points}
- Top 3 metrics linked to xG net: {top_metrics_xgnet}

ADN TÁCTICO PROPUESTO
- Tactical identity label: {tactical_identity}

FORTALEZAS DETECTADAS
- {"; ".join(strengths) if strengths else "No claramente dominantes"}

DEBILIDADES DETECTADAS
- {"; ".join(weaknesses) if weaknesses else "No claramente dominantes"}

Redacta el informe con tono profesional, crítico y de consultoría para un Director Técnico.
    """.strip()

    return text