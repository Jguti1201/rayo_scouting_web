"""
tab_buscador_perfil.py
======================
Explorador de Mercado / Buscador por Perfil mejorado.

Mejoras principales:
- Más filtros operativos:
    - club
    - excluir club
    - nacionalidad
    - fin de contrato
    - valor de mercado mínimo y máximo
    - rating mínimo
    - solo no porteros / solo porteros implícito por posición
- Configuración por grupos de rendimiento
- Tabla de resultados más ejecutiva
- Fichas rápidas
- Exportación CSV
- Watchlist / cartera

Mantiene compatibilidad con el estilo de tu versión actual.
"""

from __future__ import annotations

import re
import unicodedata
import pandas as pd
import numpy as np
import streamlit as st


# ============================================================================
# CONFIG
# ============================================================================

GRUPOS_POR_POSICION = {
    "Delantero": [
        {
            "id": "gol",
            "nombre": "Capacidad Goleadora",
            "icono": "⚽",
            "descripcion": "Producción de gol y amenaza ofensiva",
            "columnas": {
                "Goals_p90": "Goles p90",
                "xG_p90": "xG p90",
                "Total_shots_p90": "Disparos p90",
                "Goal conversion %": "Conversión (%)",
                "Big chances missed": "Grandes ocasiones falladas",
            },
            "invertir": ["Big chances missed"],
        },
        {
            "id": "creacion",
            "nombre": "Creación",
            "icono": "🎯",
            "descripcion": "Capacidad para asistir y generar ventajas",
            "columnas": {
                "Assists_p90": "Asistencias p90",
                "Big_chances_created_p90": "Grandes ocasiones creadas p90",
                "Key_passes_p90": "Pases clave p90",
                "Succ_dribbles_p90": "Regates completados p90",
            },
            "invertir": [],
        },
        {
            "id": "duelos",
            "nombre": "Duelos y Físico",
            "icono": "💪",
            "descripcion": "Dominio en disputa",
            "columnas": {
                "ground_duels_won_pct": "Duelos tierra (%)",
                "aerial_duels_won_pct": "Duelos aéreos (%)",
                "total_duels_won_pct": "Total duelos (%)",
            },
            "invertir": [],
        },
        {
            "id": "pase",
            "nombre": "Juego de Pase",
            "icono": "🔄",
            "descripcion": "Precisión y limpieza con balón",
            "columnas": {
                "Accurate passes %": "Precisión pase (%)",
                "Accurate_passes_p90": "Pases precisos p90",
            },
            "invertir": [],
        },
    ],
    "Centrocampista": [
        {
            "id": "creacion",
            "nombre": "Creación de Juego",
            "icono": "🎯",
            "descripcion": "Asistencia, pase clave y generación",
            "columnas": {
                "Assists_p90": "Asistencias p90",
                "Big_chances_created_p90": "Grandes ocasiones creadas p90",
                "Key_passes_p90": "Pases clave p90",
            },
            "invertir": [],
        },
        {
            "id": "pase",
            "nombre": "Distribución",
            "icono": "🔄",
            "descripcion": "Volumen y precisión en circulación",
            "columnas": {
                "Accurate passes %": "Precisión pase (%)",
                "Accurate_passes_p90": "Pases precisos p90",
            },
            "invertir": [],
        },
        {
            "id": "defensa",
            "nombre": "Trabajo Defensivo",
            "icono": "🛡️",
            "descripcion": "Recuperación, intercepción y presión",
            "columnas": {
                "Tackles_p90": "Tackles p90",
                "Interceptions_p90": "Intercepciones p90",
            },
            "invertir": [],
        },
        {
            "id": "duelos",
            "nombre": "Duelos",
            "icono": "💪",
            "descripcion": "Capacidad de imponerse físicamente",
            "columnas": {
                "ground_duels_won_pct": "Duelos tierra (%)",
                "aerial_duels_won_pct": "Duelos aéreos (%)",
                "total_duels_won_pct": "Total duelos (%)",
            },
            "invertir": [],
        },
        {
            "id": "amenaza",
            "nombre": "Amenaza Ofensiva",
            "icono": "🚀",
            "descripcion": "Capacidad de romper líneas y pisar área",
            "columnas": {
                "Goals_p90": "Goles p90",
                "Succ_dribbles_p90": "Regates p90",
                "xG_p90": "xG p90",
            },
            "invertir": [],
        },
    ],
    "Defensa": [
        {
            "id": "defensa",
            "nombre": "Solidez Defensiva",
            "icono": "🛡️",
            "descripcion": "Intervención defensiva y presencia en área",
            "columnas": {
                "Tackles_p90": "Tackles p90",
                "Interceptions_p90": "Intercepciones p90",
                "Clearances_p90": "Despejes p90",
                "Blocked_shots_p90": "Bloqueos p90",
            },
            "invertir": [],
        },
        {
            "id": "duelos",
            "nombre": "Duelos",
            "icono": "💪",
            "descripcion": "Juego aéreo y disputa",
            "columnas": {
                "ground_duels_won_pct": "Duelos tierra (%)",
                "aerial_duels_won_pct": "Duelos aéreos (%)",
                "total_duels_won_pct": "Total duelos (%)",
            },
            "invertir": [],
        },
        {
            "id": "pase",
            "nombre": "Juego con Balón",
            "icono": "🔄",
            "descripcion": "Salida de balón y distribución",
            "columnas": {
                "Accurate passes %": "Precisión pase (%)",
                "Accurate_passes_p90": "Pases precisos p90",
            },
            "invertir": [],
        },
        {
            "id": "disciplina",
            "nombre": "Disciplina",
            "icono": "📋",
            "descripcion": "Control del error y la falta",
            "columnas": {
                "fouls_p90": "Faltas p90",
                "Errors leading to goal": "Errores que llevan a gol",
            },
            "invertir": ["fouls_p90", "Errors leading to goal"],
        },
    ],
    "Portero": [
        {
            "id": "paradas",
            "nombre": "Capacidad de Parada",
            "icono": "🧤",
            "descripcion": "Paradas y fiabilidad bajo palos",
            "columnas": {
                "Total_saves_p90": "Paradas p90",
                "Saves_from_inside_box_p90": "Paradas área p90",
                "Clean sheets": "Porterías a cero",
            },
            "invertir": [],
        },
        {
            "id": "salidas",
            "nombre": "Salidas y área",
            "icono": "✈️",
            "descripcion": "Dominio del área",
            "columnas": {
                "Runs out": "Salidas",
                "aerial_duels_won_pct": "Duelos aéreos (%)",
            },
            "invertir": [],
        },
        {
            "id": "pase",
            "nombre": "Juego con los Pies",
            "icono": "🦶",
            "descripcion": "Distribución desde portería",
            "columnas": {
                "Accurate passes %": "Precisión pase (%)",
                "Accurate_passes_p90": "Pases precisos p90",
            },
            "invertir": [],
        },
    ],
}

LIGA_DISPLAY = {
    "laliga": "LaLiga", "laliga2": "LaLiga 2", "premier": "Premier League",
    "championship": "Championship", "bundesliga": "Bundesliga",
    "bundesliga2": "Bundesliga 2", "serie_a": "Serie A", "serie_b": "Serie B",
    "ligue1": "Ligue 1", "ligue2": "Ligue 2", "liga_portuguesa": "Primeira Liga",
    "liga_argentina": "Liga Argentina", "mls": "MLS",
}


# ============================================================================
# HELPERS
# ============================================================================

def _normalise_col(series: pd.Series) -> pd.Series:
    mn, mx = series.min(), series.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)


def _fix_text(value) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip()

    replacements = {
        "Ã¡": "á",
        "Ã©": "é",
        "Ã­": "í",
        "Ã³": "ó",
        "Ãº": "ú",
        "Ã±": "ñ",
        "Ã¼": "ü",
        "Ã": "Á",
        "Ã‰": "É",
        "Ã": "Í",
        "Ã“": "Ó",
        "Ãš": "Ú",
        "Ã‘": "Ñ",
        "Ãœ": "Ü",
        "â‚¬": "€",
        "Â": "",
    }
    for bad, good in replacements.items():
        s = s.replace(bad, good)
    return s.strip()


def _normalize_text_key(value) -> str:
    s = _fix_text(value).lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _parse_market_value(val_str) -> float:
    if pd.isna(val_str) or str(val_str).strip() in ["-", "", "nan"]:
        return np.nan

    s = _fix_text(str(val_str)).lower()
    s = s.replace("€", "").replace("\xa0", " ").strip()
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


def _parse_contract_year(val) -> float:
    if pd.isna(val) or str(val).strip() in ["-", "", "nan"]:
        return np.nan
    s = _fix_text(str(val)).strip()
    year = s[-4:]
    return float(year) if year.isdigit() else np.nan


def _compute_profile_score(df_pos: pd.DataFrame, grupos: list, pesos: dict) -> pd.Series:
    scores = pd.Series(0.0, index=df_pos.index)
    total_weight = 0.0

    for grupo in grupos:
        peso = pesos.get(grupo["id"], 0)
        if peso == 0:
            continue

        col_scores = []
        for col, _ in grupo["columnas"].items():
            if col not in df_pos.columns:
                continue
            col_data = pd.to_numeric(df_pos[col], errors="coerce").fillna(0)
            norm = _normalise_col(col_data)
            if col in grupo.get("invertir", []):
                norm = 1 - norm
            col_scores.append(norm)

        if not col_scores:
            continue

        grupo_score = pd.concat(col_scores, axis=1).mean(axis=1)
        scores += grupo_score * peso
        total_weight += peso

    if total_weight > 0:
        scores = scores / total_weight

    return (scores * 100).round(1)


def _add_watchlist_button(player_name: str):
    if "watchlist" not in st.session_state:
        st.session_state.watchlist = []

    if player_name in st.session_state.watchlist:
        if st.button("Quitar de cartera", key=f"rm_{player_name}"):
            st.session_state.watchlist.remove(player_name)
            st.rerun()
    else:
        if st.button("Añadir a cartera", key=f"add_{player_name}"):
            st.session_state.watchlist.append(player_name)
            st.rerun()


# ============================================================================
# CORE SEARCH
# ============================================================================

def _buscar_por_perfil(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    posicion = params["posicion"]
    edad_range = params["edad_range"]
    sel_ligas = params["sel_ligas"]
    solo_fiables = params["solo_fiables"]
    minutos_min = params["minutos_min"]
    presupuesto_min = params["presupuesto_min"]
    presupuesto_max = params["presupuesto_max"]
    rating_min = params["rating_min"]
    contract_year_max = params["contract_year_max"]
    selected_nationalities = params["selected_nationalities"]
    selected_clubs = params["selected_clubs"]
    exclude_clubs = params["exclude_clubs"]
    pesos = params["pesos"]

    df_pos = df.copy()

    # Posición
    df_pos = df_pos[df_pos["posicion"] == posicion]

    # Edad
    if "edad" in df_pos.columns:
        df_pos = df_pos[df_pos["edad"].between(edad_range[0], edad_range[1])]

    # Ligas
    if sel_ligas:
        df_pos = df_pos[df_pos["liga"].isin(sel_ligas)]

    # Fiabilidad
    if solo_fiables and "muestra_fiable" in df_pos.columns:
        df_pos = df_pos[df_pos["muestra_fiable"] == True]

    # Minutos
    if minutos_min > 0 and "minutes_played" in df_pos.columns:
        df_pos = df_pos[df_pos["minutes_played"] >= minutos_min]

    # Rating mínimo
    if rating_min > 0 and "Average Sofascore Rating" in df_pos.columns:
        df_pos = df_pos[pd.to_numeric(df_pos["Average Sofascore Rating"], errors="coerce") >= rating_min]

    # Valor mercado
    if "valor_mercado" in df_pos.columns:
        df_pos["_vm"] = df_pos["valor_mercado"].apply(_parse_market_value)

        if presupuesto_min is not None:
            df_pos = df_pos[df_pos["_vm"].fillna(-1) >= presupuesto_min]

        if presupuesto_max is not None:
            df_pos = df_pos[df_pos["_vm"].fillna(np.inf) <= presupuesto_max]

    # Fin de contrato
    if contract_year_max is not None and "fin_contrato" in df_pos.columns:
        df_pos["_contract_year"] = df_pos["fin_contrato"].apply(_parse_contract_year)
        df_pos = df_pos[df_pos["_contract_year"].fillna(np.inf) <= contract_year_max]

    # Nacionalidad
    if selected_nationalities and "nacionalidades" in df_pos.columns:
        selected_nat_norm = {_normalize_text_key(n) for n in selected_nationalities}

        def has_any_nationality(val):
            txt = _fix_text(val)
            parts = [_normalize_text_key(x) for x in re.split(r"[;,/]", txt) if str(x).strip()]
            return any(p in selected_nat_norm for p in parts)

        df_pos = df_pos[df_pos["nacionalidades"].apply(has_any_nationality)]

    # Clubes incluidos
    if selected_clubs and "tm_club" in df_pos.columns:
        df_pos = df_pos[df_pos["tm_club"].isin(selected_clubs)]

    # Clubes excluidos
    if exclude_clubs and "tm_club" in df_pos.columns:
        df_pos = df_pos[~df_pos["tm_club"].isin(exclude_clubs)]

    if df_pos.empty:
        return df_pos

    grupos = GRUPOS_POR_POSICION.get(posicion, [])
    total_peso = sum(pesos.get(g["id"], 0) for g in grupos)

    if total_peso == 0:
        df_pos["perfil_score"] = 50.0
        if "Average Sofascore Rating" in df_pos.columns:
            df_pos["perfil_score"] = _normalise_col(
                pd.to_numeric(df_pos["Average Sofascore Rating"], errors="coerce").fillna(0)
            ) * 100
        return df_pos.sort_values("perfil_score", ascending=False).head(20).reset_index(drop=True)

    df_pos["perfil_score"] = _compute_profile_score(df_pos, grupos, pesos)
    return df_pos.sort_values("perfil_score", ascending=False).head(20).reset_index(drop=True)


# ============================================================================
# RENDER
# ============================================================================

def render_buscador_perfil_tab(df: pd.DataFrame):
    st.markdown("### Explorador de Base de Datos")
    st.write("Ajusta filtros de mercado y prioridades de rendimiento para localizar perfiles compatibles.")

    col_config, col_results = st.columns([1.1, 1.4], gap="large")

    with col_config:
        posicion = st.selectbox(
            "Demarcación",
            options=["Defensa", "Centrocampista", "Delantero", "Portero"]
        )

        st.markdown("---")
        st.markdown("#### Filtros principales")

        edad_range = st.slider("Rango de edad", min_value=16, max_value=40, value=(18, 30))

        all_ligas = sorted(df["liga"].dropna().unique().tolist()) if "liga" in df.columns else []
        sel_ligas = st.multiselect(
            "Competición",
            options=all_ligas,
            default=all_ligas,
            format_func=lambda x: LIGA_DISPLAY.get(x, x)
        )

        solo_fiables = st.checkbox("Solo muestra fiable", value=True)
        minutos_min = st.slider("Minutos mínimos", min_value=0, max_value=3000, value=500, step=100)

        st.markdown("---")
        st.markdown("#### Filtros avanzados")

        presupuesto_cols = st.columns(2)
        with presupuesto_cols[0]:
            presupuesto_min = st.number_input("Valor mercado mín. (M€)", min_value=0.0, max_value=500.0, value=0.0, step=0.5)
        with presupuesto_cols[1]:
            presupuesto_max = st.number_input("Valor mercado máx. (M€)", min_value=0.0, max_value=500.0, value=20.0, step=0.5)

        rating_min = st.slider("Rating mínimo", min_value=0.0, max_value=10.0, value=0.0, step=0.1)

        contract_year_max = st.selectbox(
            "Contrato hasta...",
            options=[None, 2026, 2027, 2028, 2029, 2030],
            format_func=lambda x: "Sin filtro" if x is None else str(x)
        )

        all_clubs = sorted(df["tm_club"].dropna().astype(str).unique().tolist()) if "tm_club" in df.columns else []
        selected_clubs = st.multiselect("Incluir solo clubes", options=all_clubs, default=[])

        exclude_clubs = st.multiselect("Excluir clubes", options=all_clubs, default=[])

        all_nationalities = []
        if "nacionalidades" in df.columns:
            nat_set = set()
            for val in df["nacionalidades"].dropna():
                txt = _fix_text(val)
                parts = [p.strip() for p in re.split(r"[;,/]", txt) if p.strip()]
                nat_set.update(parts)
            all_nationalities = sorted(nat_set)

        selected_nationalities = st.multiselect("Nacionalidad", options=all_nationalities, default=[])

        st.markdown("---")
        st.markdown("#### Perfil de Rendimiento")

        grupos = GRUPOS_POR_POSICION.get(posicion, [])
        pesos = {}

        for grupo in grupos:
            with st.expander(f"{grupo['icono']} {grupo['nombre']}", expanded=True):
                st.caption(grupo["descripcion"])
                pesos[grupo["id"]] = st.slider(
                    "Importancia",
                    min_value=0,
                    max_value=5,
                    value=0,
                    key=f"peso_v3_{posicion}_{grupo['id']}"
                )

                show_stats = st.toggle("Ver métricas incluidas", key=f"metrics_{posicion}_{grupo['id']}")
                if show_stats:
                    for col, label in grupo["columnas"].items():
                        invertido = " ↓ menos es mejor" if col in grupo.get("invertir", []) else ""
                        st.markdown(f"- **{label}** (`{col}`){invertido}")

        buscar_btn = st.button("Ejecutar búsqueda", type="primary", use_container_width=True)

    with col_results:
        if buscar_btn or st.session_state.get("perfil_last_search_v3"):
            if buscar_btn:
                st.session_state["perfil_last_search_v3"] = {
                    "posicion": posicion,
                    "edad_range": edad_range,
                    "sel_ligas": sel_ligas,
                    "solo_fiables": solo_fiables,
                    "minutos_min": minutos_min,
                    "presupuesto_min": presupuesto_min,
                    "presupuesto_max": presupuesto_max,
                    "rating_min": rating_min,
                    "contract_year_max": contract_year_max,
                    "selected_nationalities": selected_nationalities,
                    "selected_clubs": selected_clubs,
                    "exclude_clubs": exclude_clubs,
                    "pesos": pesos.copy(),
                }

            params = st.session_state["perfil_last_search_v3"]
            resultados = _buscar_por_perfil(df, params)

            if resultados.empty:
                st.warning("No se encontraron jugadores con los filtros aplicados.")
                return

            st.markdown("#### Resultados de Búsqueda")

            # Resumen filtros activos
            filtros_activos = []
            if params["sel_ligas"]:
                filtros_activos.append(f"Ligas: {len(params['sel_ligas'])}")
            if params["selected_nationalities"]:
                filtros_activos.append(f"Nacionalidades: {', '.join(params['selected_nationalities'][:3])}")
            if params["selected_clubs"]:
                filtros_activos.append(f"Clubes incluidos: {len(params['selected_clubs'])}")
            if params["exclude_clubs"]:
                filtros_activos.append(f"Clubes excluidos: {len(params['exclude_clubs'])}")
            if params["contract_year_max"] is not None:
                filtros_activos.append(f"Contrato ≤ {params['contract_year_max']}")
            if params["rating_min"] > 0:
                filtros_activos.append(f"Rating ≥ {params['rating_min']:.1f}")

            if filtros_activos:
                st.caption(" | ".join(filtros_activos))

            table_df = resultados.copy()
            cols_show = [c for c in [
                "Name", "tm_club", "liga", "edad", "fin_contrato",
                "valor_mercado", "minutes_played", "perfil_score", "Average Sofascore Rating"
            ] if c in table_df.columns]
            table_df = table_df[cols_show].copy()

            if "liga" in table_df.columns:
                table_df["liga"] = table_df["liga"].map(lambda x: LIGA_DISPLAY.get(x, x))

            rename_map = {
                "Name": "Jugador",
                "tm_club": "Equipo",
                "liga": "Competición",
                "edad": "Edad",
                "fin_contrato": "Fin Contrato",
                "valor_mercado": "Valor Mercado",
                "minutes_played": "Minutos",
                "perfil_score": "Score Global",
                "Average Sofascore Rating": "Rating"
            }
            table_df = table_df.rename(columns=rename_map)

            st.dataframe(table_df, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown("#### Fichas rápidas")

            for i, (_, row) in enumerate(resultados.head(10).iterrows(), start=1):
                score = float(row.get("perfil_score", 0))
                score_color = "#28a745" if score >= 70 else "#ffc107" if score >= 45 else "#dc3545"

                st.markdown(f"""
                <div style="background:white;border:1px solid #dee2e6;border-top:4px solid {score_color};
                            border-radius:4px;padding:16px;margin-bottom:14px;">
                    <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                        <div>
                            <div style="font-size:12px;color:#6c757d;font-weight:700;">#{i}</div>
                            <div style="font-size:18px;font-weight:700;color:#212529;">{row.get('Name','?')}</div>
                            <div style="font-size:14px;color:#6c757d;">
                                {row.get('tm_club','?')} · {LIGA_DISPLAY.get(str(row.get('liga','')), str(row.get('liga','')))}
                            </div>
                        </div>
                        <div style="text-align:right;">
                            <div style="font-size:22px;font-weight:700;color:{score_color};">{score:.0f}</div>
                            <div style="font-size:11px;color:#6c757d;">SCORE</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                cols = st.columns([1, 1, 1, 1])
                cols[0].metric("Edad", f"{row.get('edad','N/D')}")
                cols[1].metric("Contrato", f"{row.get('fin_contrato','N/D')}")
                cols[2].metric("Valor", f"{row.get('valor_mercado','N/D')}")
                cols[3].metric("Rating", f"{row.get('Average Sofascore Rating','N/D')}")

                _add_watchlist_button(str(row.get("Name", f"player_{i}")))
                st.markdown("---")

            export_cols = [c for c in [
                "Name", "tm_club", "liga", "posicion", "edad",
                "fin_contrato", "valor_mercado", "minutes_played",
                "Average Sofascore Rating", "perfil_score"
            ] if c in resultados.columns]

            export_df = resultados[export_cols].copy()
            export_df.columns = [c.replace("_", " ").title() for c in export_df.columns]

            st.download_button(
                "Descargar resultados (CSV)",
                data=export_df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"explorador_{params['posicion'].lower()}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        else:
            st.info("Configure los filtros y pulse 'Ejecutar búsqueda' para explorar el mercado.")