"""
tab_scout_ia.py — Asistente IA v3
=================================
Versión adaptada a una interfaz más institucional / ejecutiva.
Mantiene la lógica de conversación, extracción de perfil y búsqueda.

v3 — Cambios:
- API key OpenAI cargada desde .env (variable OPENAI_API_KEY)
- Header ejecutivo con mismo formato que el resto de tabs
"""

from __future__ import annotations
import json
import os
import re
from typing import Any
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from pathlib import Path

# Buscar el .env subiendo desde el archivo actual (más robusto)
ENV_PATH = Path(__file__).resolve().parents[4] / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)


try:
    import openai
    OPENAI_OK = True
except ImportError:
    OPENAI_OK = False

LIGA_DISPLAY = {
    "laliga": "LaLiga", "laliga2": "LaLiga 2", "premier": "Premier League",
    "championship": "Championship", "bundesliga": "Bundesliga",
    "bundesliga2": "Bundesliga 2", "serie_a": "Serie A", "serie_b": "Serie B",
    "ligue1": "Ligue 1", "ligue2": "Ligue 2",
    "liga_argentina": "Liga Argentina", "mls": "MLS",
}

MIN_MINUTES = 450

KPI_MAP = {
    "tackles_p90": ("Tackles", "Tackles p90"),
    "interceptions_p90": ("Interceptions", "Intercepciones p90"),
    "clearances_p90": ("Clearances", "Despejes p90"),
    "duels_won_pct": ("ground_duels_won_pct", "Duelos ganados %"),
    "aerial_pct": ("aerial_duels_won_pct", "Duelos aéreos %"),
    "key_passes_p90": ("Key passes", "Pases clave p90"),
    "dribbles_p90": ("Succ. dribbles", "Regates p90"),
    "pass_accuracy": ("Accurate passes %", "Precisión pases %"),
    "passes_p90": ("Accurate_passes_p90", "Pases precisos p90"),
    "goals_p90": ("Goals", "Goles p90"),
    "assists_p90": ("Assists", "Asistencias p90"),
    "xg_p90": ("Expected goals (xG)", "xG p90"),
    "big_chances_p90": ("Big chances created", "Grandes ocasiones p90"),
    "rating": ("Average Sofascore Rating", "Rating global"),
}

DIMENSIONS = {
    "defense": ["tackles_p90", "interceptions_p90", "clearances_p90", "duels_won_pct"],
    "progression": ["key_passes_p90", "dribbles_p90", "pass_accuracy", "passes_p90"],
    "attack": ["goals_p90", "assists_p90", "xg_p90", "big_chances_p90"],
    "physical": ["rating", "aerial_pct"],
}

DEFAULT_WEIGHTS = {
    "defense": 0.30,
    "progression": 0.30,
    "attack": 0.20,
    "physical": 0.20
}

SYSTEM_SCOUT = """Eres un asistente de scouting profesional para el Rayo Vallecano.
Tu objetivo es conversar con el usuario para entender qué tipo de jugador busca.
Haz como máximo una pregunta por turno y cuando tengas suficiente información responde incluyendo [PERFIL_LISTO]."""

SYSTEM_EXTRACTOR = """Extrae la conversación a JSON válido.
Devuelve SOLO JSON válido con esta estructura:
{
  "position": "Centrocampista" | "Defensa" | "Delantero" | null,
  "age_min": int | null,
  "age_max": int | null,
  "ligas": ["laliga", "premier"] | null,
  "market_value_max_M": float | null,
  "weights": {
    "defense": float,
    "progression": float,
    "attack": float,
    "physical": float
  },
  "kpi_boosts": {
    "tackles_p90": float,
    "interceptions_p90": float,
    "clearances_p90": float,
    "duels_won_pct": float,
    "aerial_pct": float,
    "key_passes_p90": float,
    "dribbles_p90": float,
    "pass_accuracy": float,
    "passes_p90": float,
    "goals_p90": float,
    "assists_p90": float,
    "xg_p90": float,
    "big_chances_p90": float,
    "rating": float
  },
  "profile_description": "texto breve"
}
"""

SYSTEM_ANALYST = """Eres un analista de scouting senior.
Escribe una ficha breve y profesional del jugador candidato en markdown con estas secciones:
## Por qué encaja
## Fortalezas
## Puntos de atención
## Veredicto
"""


# ============================================================================
# API KEY desde .env
# ============================================================================

def _get_openai_api_key() -> str:
    import os
    print("=" * 60)
    print(f"📂 ENV_PATH existe: {ENV_PATH.exists()}")
    print(f"📂 ENV_PATH: {ENV_PATH}")
    key = os.getenv("OPENAI_API_KEY", "")
    print(f"🔑 Key cargada: {key[:10]}...{key[-4:] if key else 'VACÍA'}")
    print("=" * 60)
    key = os.getenv("OPENAI_API_KEY", "")
    # Limpieza defensiva: quita comillas, espacios y caracteres raros
    key = key.strip().strip('"').strip("'").strip()
    if key:
        return key

    try:
        secret = st.secrets.get("OPENAI_API_KEY", "")
        return secret.strip().strip('"').strip("'").strip()
    except Exception:
        return ""


# ============================================================================
# PREPARACIÓN DE DATOS
# ============================================================================

@st.cache_data(show_spinner=False)
def prepare_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    if "es_portero" in df.columns:
        df = df[df["es_portero"] != True]

    df = df[df["minutes_played"] >= MIN_MINUTES].copy()

    raw_to_p90 = {
        "Goals": "goals_p90",
        "Assists": "assists_p90",
        "Expected goals (xG)": "xg_p90",
        "Tackles": "tackles_p90",
        "Interceptions": "interceptions_p90",
        "Clearances": "clearances_p90",
        "Key passes": "key_passes_p90",
        "Succ. dribbles": "dribbles_p90",
        "Big chances created": "big_chances_p90",
    }

    for raw_col, new_col in raw_to_p90.items():
        if raw_col in df.columns:
            df[new_col] = (df[raw_col] / df["minutes_played"]) * 90
        else:
            df[new_col] = 0.0

    df["duels_won_pct"] = df.get("ground_duels_won_pct", pd.Series(0.0, index=df.index))
    df["aerial_pct"] = df.get("aerial_duels_won_pct", pd.Series(0.0, index=df.index))
    df["pass_accuracy"] = df.get("Accurate passes %", pd.Series(0.0, index=df.index))
    df["passes_p90"] = df.get("Accurate_passes_p90", pd.Series(0.0, index=df.index))
    df["rating"] = df.get("Average Sofascore Rating", pd.Series(0.0, index=df.index))

    kpi_cols = list(KPI_MAP.keys())

    for col in kpi_cols:
        if col in df.columns:
            cap = df[col].quantile(0.99)
            df[col] = df[col].clip(upper=cap)
            df[col] = df[col].fillna(df[col].median())

    scaler = MinMaxScaler()
    norm_data = scaler.fit_transform(df[kpi_cols])
    df["_vector"] = list(norm_data)
    df["liga_display"] = df["liga"].map(LIGA_DISPLAY).fillna(df.get("liga", ""))

    return df.reset_index(drop=True)


# ============================================================================
# LLAMADAS IA
# ============================================================================

def call_gpt(api_key: str, messages: list[dict], system: str,
             model: str = "gpt-4o-mini", temperature: float = 0.4,
             max_tokens: int = 700) -> str:
    client = openai.OpenAI(api_key=api_key)
    full_messages = [{"role": "system", "content": system}] + messages
    resp = client.chat.completions.create(
        model=model,
        messages=full_messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def extract_profile_from_conversation(api_key: str, conversation: list[dict]) -> dict | None:
    conv_text = "\n".join(
        f"{'ENTRENADOR' if m['role']=='user' else 'ASISTENTE'}: {m['content']}"
        for m in conversation
    )
    prompt = f"Extrae los parámetros de búsqueda de esta conversación:\n\n{conv_text}"

    raw = call_gpt(
        api_key,
        [{"role": "user", "content": prompt}],
        system=SYSTEM_EXTRACTOR,
        model="gpt-4o-mini",
        temperature=0.1,
        max_tokens=600,
    )

    raw = re.sub(r"```json|```", "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def generate_scout_report(api_key: str, profile_desc: str,
                          player_data: dict, rank: int,
                          similarity_score: float) -> str:
    stats_block = "\n".join(
        f"- {label}: {player_data.get(key, 'N/D')}"
        for key, (_, label) in KPI_MAP.items()
    )

    prompt = f"""
PERFIL BUSCADO:
{profile_desc}

CANDIDATO #{rank} — similitud: {similarity_score:.1%}
Nombre: {player_data.get('Name', '?')}
Club: {player_data.get('tm_club', '?')}
Liga: {player_data.get('liga_display', '?')}
Edad: {player_data.get('edad', '?')}
Valor de mercado: {player_data.get('valor_mercado', 'N/D')}
Posición: {player_data.get('posicion', '?')}

ESTADÍSTICAS:
{stats_block}
"""
    return call_gpt(
        api_key,
        [{"role": "user", "content": prompt}],
        system=SYSTEM_ANALYST,
        model="gpt-4o-mini",
        temperature=0.3,
        max_tokens=650,
    )


# ============================================================================
# UTILIDADES
# ============================================================================

def _coerce_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return default
        try:
            return float(s.replace(",", "."))
        except Exception:
            return default
    try:
        return float(value)
    except Exception:
        return default


def build_synthetic_vector(weights: dict, kpi_boosts: dict) -> np.ndarray:
    kpi_cols = list(KPI_MAP.keys())
    vector = np.zeros(len(kpi_cols))
    weights = weights or DEFAULT_WEIGHTS
    kpi_boosts = kpi_boosts or {}

    for i, kpi in enumerate(kpi_cols):
        weight = 0.25
        for dim, kpis in DIMENSIONS.items():
            if kpi in kpis:
                weight = _coerce_float(weights.get(dim, 0.25), default=0.25)
                break
        vector[i] = weight

    for i, kpi in enumerate(kpi_cols):
        boost = _coerce_float(kpi_boosts.get(kpi, 0.0), default=0.0)
        vector[i] = np.clip(vector[i] + boost, 0.0, 1.0)

    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm

    return vector


def _parse_vm(val_str):
    if pd.isna(val_str) or str(val_str) in ["-", "", "nan"]:
        return 9999.0
    s = str(val_str).lower().replace("€", "").replace(".", "").strip().replace(",", ".")
    if "mill" in s:
        return float(s.replace("mill.", "").replace("mill", "").strip())
    elif "mil" in s:
        return float(s.replace("mil", "").strip()) / 1000
    try:
        return float(s)
    except Exception:
        return 9999.0


def search_by_profile(df: pd.DataFrame, profile: dict, top_n: int = 10) -> pd.DataFrame:
    filtered = df.copy()

    if profile.get("position"):
        filtered = filtered[filtered["posicion"] == profile["position"]]

    if profile.get("age_min") is not None:
        filtered = filtered[filtered["edad"] >= profile["age_min"]]
    if profile.get("age_max") is not None:
        filtered = filtered[filtered["edad"] <= profile["age_max"]]

    if profile.get("ligas"):
        filtered = filtered[filtered["liga"].isin(profile["ligas"])]

    if profile.get("market_value_max_M") is not None:
        filtered["_vm"] = filtered["valor_mercado"].apply(_parse_vm)
        filtered = filtered[filtered["_vm"] <= profile["market_value_max_M"]]

    if filtered.empty:
        return pd.DataFrame()

    weights = profile.get("weights") or DEFAULT_WEIGHTS
    kpi_boost = profile.get("kpi_boosts") or {}
    synth_vec = build_synthetic_vector(weights, kpi_boost).reshape(1, -1)

    all_vecs = np.stack(filtered["_vector"].values)
    sims = cosine_similarity(synth_vec, all_vecs)[0]
    filtered = filtered.copy()
    filtered["similarity"] = sims
    filtered = filtered.sort_values("similarity", ascending=False).head(top_n)

    return filtered.reset_index(drop=True)


def render_profile_summary(profile: dict):
    st.markdown("#### Perfil extraído")
    cols = st.columns(4)
    w = profile.get("weights", DEFAULT_WEIGHTS)

    cols[0].metric("Defensa", f"{w.get('defense',0):.0%}")
    cols[1].metric("Progresión", f"{w.get('progression',0):.0%}")
    cols[2].metric("Ataque", f"{w.get('attack',0):.0%}")
    cols[3].metric("Físico", f"{w.get('physical',0):.0%}")

    details = []
    if profile.get("position"):
        details.append(f"**Posición:** {profile['position']}")
    if profile.get("age_min") or profile.get("age_max"):
        details.append(f"**Edad:** {profile.get('age_min', '—')} - {profile.get('age_max', '—')}")
    if profile.get("ligas"):
        details.append("**Ligas:** " + ", ".join(LIGA_DISPLAY.get(l, l) for l in profile["ligas"]))
    if profile.get("market_value_max_M"):
        details.append(f"**Valor máximo:** {profile['market_value_max_M']:.1f} M€")

    if details:
        st.markdown(" · ".join(details))


# ============================================================================
# RENDER PRINCIPAL
# ============================================================================

def render_scout_ia_tab(df_raw: pd.DataFrame):

    # ── Header ejecutivo (estilo unificado con resto de tabs) ──────────
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1A1A2E 0%,#16213E 50%,#0F3460 100%);
                padding:1.5rem 2rem;border-radius:14px;margin-bottom:1.5rem;border-left:5px solid #E30613;">
        <div style="font-size:2rem;color:white;font-weight:700;">🤖 ASISTENTE IA DE SCOUTING</div>
        <div style="color:rgba(255,255,255,0.6);font-size:0.82rem;text-transform:uppercase;margin-top:0.25rem;">
            Prospección avanzada mediante lenguaje natural y similitud vectorial
        </div>
    </div>
    """, unsafe_allow_html=True)

    df = prepare_df(df_raw)

    if not OPENAI_OK:
        st.error("El paquete `openai` no está instalado. Ejecuta: `pip install openai`")
        return

    # ── API key desde .env ─────────────────────────────────────────────
    api_key = _get_openai_api_key()

    if not api_key:
        st.error(
            "❌ No se ha encontrado la variable de entorno **OPENAI_API_KEY**.\n\n"
            "Añádela a tu archivo `.env` en la raíz del proyecto:\n\n"
            "```\nOPENAI_API_KEY=sk-xxxxxxxx...\n```"
        )
        return
    else:
        st.caption("🔐 API key cargada desde `.env`")

    # ── Estado de sesión ───────────────────────────────────────────────
    if "scout_messages" not in st.session_state:
        st.session_state.scout_messages = [
            {"role": "assistant", "content": "Sistema inicializado. Preparado para ejecutar consultas. Ejemplo: 'Buscar central zurdo en segundas divisiones con percentil aéreo superior a 80'."}
        ]
    if "scout_profile" not in st.session_state:
        st.session_state.scout_profile = None
    if "scout_results" not in st.session_state:
        st.session_state.scout_results = None
    if "scout_reports" not in st.session_state:
        st.session_state.scout_reports = {}
    if "scout_profile_ready" not in st.session_state:
        st.session_state.scout_profile_ready = False

    col_chat, col_results = st.columns([1, 1], gap="large")

    with col_chat:
        if st.button("Nueva conversación", use_container_width=True):
            st.session_state.scout_messages = [
                {"role": "assistant", "content": "Sistema reiniciado. Describe el perfil de jugador que buscas."}
            ]
            st.session_state.scout_profile = None
            st.session_state.scout_results = None
            st.session_state.scout_reports = {}
            st.session_state.scout_profile_ready = False
            st.rerun()

        for msg in st.session_state.scout_messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"].replace("[PERFIL_LISTO]", "").strip())

        if not st.session_state.scout_profile_ready:
            if prompt := st.chat_input("Introduzca su consulta técnica..."):
                st.session_state.scout_messages.append({"role": "user", "content": prompt})

                with st.spinner("Procesando consulta..."):
                    try:
                        reply = call_gpt(
                            api_key,
                            st.session_state.scout_messages,
                            system=SYSTEM_SCOUT,
                            model="gpt-4o-mini",
                            temperature=0.4,
                            max_tokens=350,
                        )
                    except Exception as e:
                        reply = f"Error al conectar con la IA: {e}"

                st.session_state.scout_messages.append({"role": "assistant", "content": reply})

                if "[PERFIL_LISTO]" in reply:
                    st.session_state.scout_profile_ready = True

                st.rerun()

        if st.session_state.scout_profile_ready:
            st.success("Perfil listo para búsqueda.")
            if st.button("Ejecutar búsqueda", type="primary", key="buscar_perfil", use_container_width=True):
                with st.spinner("Extrayendo perfil..."):
                    profile = extract_profile_from_conversation(api_key, st.session_state.scout_messages)

                if profile:
                    st.session_state.scout_profile = profile
                    with st.spinner("Buscando candidatos..."):
                        st.session_state.scout_results = search_by_profile(df, profile, top_n=10)
                    st.rerun()
                else:
                    st.error("No se pudo extraer el perfil correctamente.")

        if st.session_state.scout_profile:
            st.markdown("---")
            render_profile_summary(st.session_state.scout_profile)
            desc = st.session_state.scout_profile.get("profile_description", "")
            if desc:
                st.caption(desc)

    with col_results:
        st.markdown("#### Resultados")
        results = st.session_state.scout_results

        if results is None or results.empty:
            st.info("Los candidatos aparecerán aquí cuando completes la búsqueda.")
            return

        if st.button("Generar fichas IA para todos", use_container_width=True):
            progress = st.progress(0)
            for i, (_, row) in enumerate(results.iterrows(), start=1):
                player_name = row["Name"]
                if player_name not in st.session_state.scout_reports:
                    report = generate_scout_report(
                        api_key,
                        st.session_state.scout_profile.get("profile_description", "perfil buscado"),
                        row.to_dict(),
                        rank=i,
                        similarity_score=float(row["similarity"])
                    )
                    st.session_state.scout_reports[player_name] = report
                progress.progress(i / len(results))
            st.rerun()

        for rank, (_, row) in enumerate(results.iterrows(), start=1):
            similarity = float(row["similarity"])
            st.markdown(f"""
            <div style="background:white;border:1px solid #dee2e6;border-top:4px solid #E30613;
                        border-radius:4px;padding:16px;margin-bottom:14px;">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                    <div>
                        <div style="font-size:12px;color:#6c757d;font-weight:700;">#{rank}</div>
                        <div style="font-size:18px;font-weight:700;color:#212529;">{row.get('Name','?')}</div>
                        <div style="font-size:14px;color:#6c757d;">
                            {row.get('tm_club','?')} · {LIGA_DISPLAY.get(str(row.get('liga','')), str(row.get('liga','')))}
                        </div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:22px;font-weight:700;color:#28a745;">{similarity:.1%}</div>
                        <div style="font-size:11px;color:#6c757d;">SIMILITUD</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander(f"Ver ficha y métricas — {row['Name']}", expanded=(rank <= 3)):
                cols = st.columns(4)
                cols[0].metric("Edad", f"{row.get('edad','N/D')}")
                cols[1].metric("Valor mercado", f"{row.get('valor_mercado','N/D')}")
                cols[2].metric("Contrato", f"{row.get('fin_contrato','N/D')}")
                cols[3].metric("Rating", f"{row.get('Average Sofascore Rating','N/D')}")

                if row["Name"] in st.session_state.scout_reports:
                    st.markdown(st.session_state.scout_reports[row["Name"]])
                elif st.button(f"Generar ficha individual", key=f"gen_report_{rank}"):
                    with st.spinner("Generando ficha..."):
                        report = generate_scout_report(
                            api_key,
                            st.session_state.scout_profile.get("profile_description", "perfil buscado"),
                            row.to_dict(),
                            rank=rank,
                            similarity_score=similarity
                        )
                        st.session_state.scout_reports[row["Name"]] = report
                    st.rerun()

        export_df = results[[c for c in [
            "Name", "tm_club", "liga", "posicion", "edad", "valor_mercado", "similarity"
        ] if c in results.columns]].copy()
        export_df["similarity"] = (export_df["similarity"] * 100).round(1)

        st.download_button(
            "Descargar resultados (CSV)",
            data=export_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="candidatos_asistente_ia.csv",
            mime="text/csv",
            use_container_width=True
        )