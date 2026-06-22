"""
Rayo Vallecano | Dirección de Scouting
======================================
Nueva interfaz v2 orientada a dirección deportiva / scouting ejecutivo.

Pestañas:
- Resumen General
- Asistente IA
- Explorador de Mercado
- Comparador de Perfiles
- Análisis de Plantilla

Backend reutilizado:
- data_loader.py
- feature_engineering.py
- clustering.py
- adaptation_score.py
- ai_prompts.py
- tab_scout_ia.py
- tab_buscador_perfil.py
"""

import sys
from pathlib import Path
import logging
import base64

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from dotenv import load_dotenv
load_dotenv()  # carga el .env del directorio raíz

# ── Path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = ROOT.parent
PROJECT_ROOT = PACKAGE_ROOT.parent

for path in (ROOT, PACKAGE_ROOT, PROJECT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from tabs.tab_resumen_general import render_resumen_general_tab
from tabs.tab_buscador_perfil import render_buscador_perfil_tab
from tabs.tab_comparador_perfiles import render_comparador_perfiles_tab
from tabs.tab_analisis_plantilla import render_analisis_plantilla_tab
from tabs.tab_scout_ia import render_scout_ia_tab
from features.data_loader import load_data
from rayo_scouting.features.adaptation_score import compute_adaptation_score
from rayo_scouting.scouting.report_generation import render_watchlist_reports

logging.basicConfig(level=logging.WARNING)

if "watchlist" not in st.session_state:
    st.session_state.watchlist = []

# ── Config página ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Rayo Vallecano | Dirección de Scouting",
    page_icon="assets/img/Rayo_Vallecano.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Constantes ───────────────────────────────────────────────────────────────
DATA_PATH = ROOT / "data" / "all_leagues_master_v5.csv"
RAYO_IMG = ROOT / "assets" / "img" / "Rayo_Vallecano.png"

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

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    :root {
        --rojo-rayo: #E30613;
        --negro-carbon: #212529;
        --gris-claro: #F8F9FA;
        --blanco: #FFFFFF;
        --verde-top: #28a745;
        --amarillo-media: #ffc107;
        --rojo-oscuro: #8b0000;
        --gris-linea: #dee2e6;
        --gris-texto: #6c757d;
    }

    /* ─── OCULTAR BARRA SUPERIOR DE STREAMLIT ─────────────────── */
    header[data-testid="stHeader"] {
        display: none !important;
        height: 0 !important;
    }

    div[data-testid="stActionButtonIcon-deploy"],
    .stDeployButton {
        display: none !important;
    }

    button[title="View app menu"],
    #MainMenu {
        display: none !important;
    }

    button[title="Rerun"] {
        display: none !important;
    }

    div[data-testid="stToolbar"] {
        display: none !important;
        visibility: hidden !important;
        height: 0 !important;
    }

    footer {
        display: none !important;
    }

    div[data-testid="stDecoration"] {
        display: none !important;
    }

    div[data-testid="stStatusWidget"] {
        display: none !important;
    }

    /* ─── ELIMINAR HUECO SUPERIOR DEL CONTENIDO ───────────────── */
    /* Contenedor principal de la app */
    .stApp {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }

    /* Bloque principal (donde vive todo tu contenido) */
    .block-container,
    div[data-testid="stMainBlockContainer"],
    section[data-testid="stMain"] > div.block-container {
        padding-top: 0.5rem !important;
        padding-bottom: 2rem !important;
        margin-top: 0 !important;
        max-width: 95% !important;
    }

    /* AppView container (raíz de la app) */
    div[data-testid="stAppViewContainer"] {
        padding-top: 0 !important;
    }

    /* Vertical block (el que tú has identificado: stVerticalBlock) */
    div[data-testid="stVerticalBlock"]:first-child {
        gap: 0.5rem !important;
        padding-top: 0 !important;
        margin-top: 0 !important;
    }

    /* Sección main */
    section.main > div {
        padding-top: 0 !important;
    }

    /* Sidebar también pegado arriba */
    section[data-testid="stSidebar"] > div {
        padding-top: 1rem !important;
    }
    /* ─────────────────────────────────────────────────────────── */

    html, body, .stApp {
        background-color: var(--blanco) !important;
        color: var(--negro-carbon) !important;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }

    .main .block-container {
        padding-top: 0.5rem;
        padding-bottom: 2rem;
        max-width: 95%;
    }

    .header-franja {
        border-left: 12px solid var(--rojo-rayo);
        background-color: var(--negro-carbon);
        padding: 25px 30px;
        border-radius: 4px;
        color: var(--blanco);
        margin-bottom: 30px;
        margin-top: 0;
    }

    .header-franja h1 {
        margin: 0;
        color: var(--blanco);
        font-size: 26px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }

    .header-franja p {
        margin: 5px 0 0 0;
        color: #adb5bd;
        font-size: 14px;
        font-weight: 400;
    }

    .section-title {
        color: var(--negro-carbon);
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .subtle-text {
        color: var(--gris-texto);
        font-size: 14px;
    }

    .player-card {
        background-color: var(--blanco);
        border: 1px solid var(--gris-linea);
        border-top: 4px solid var(--negro-carbon);
        padding: 20px;
        border-radius: 4px;
        margin-bottom: 15px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    .player-card.destacado {
        border-top: 4px solid var(--rojo-rayo);
    }

    .afinidad-score {
        float: right;
        font-size: 18px;
        font-weight: 700;
        color: var(--verde-top);
    }

    .tag-football {
        background-color: var(--gris-claro);
        color: var(--negro-carbon);
        border: 1px solid #ced4da;
        padding: 4px 12px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        display: inline-block;
        margin-right: 8px;
        margin-top: 10px;
    }

    .alert-card {
        padding: 15px;
        border-radius: 4px;
        border-left: 5px solid;
        background-color: var(--gris-claro);
        min-height: 140px;
    }

    .alert-critical { border-left-color: var(--rojo-oscuro); }
    .alert-warning { border-left-color: var(--amarillo-media); }
    .alert-ok { border-left-color: var(--verde-top); }

    div.stButton > button:first-child,
    div.stDownloadButton > button:first-child {
        background-color: var(--rojo-rayo) !important;
        color: var(--blanco) !important;
        border: none !important;
        border-radius: 4px !important;
        font-weight: 600 !important;
    }

    div.stButton > button:first-child:hover,
    div.stDownloadButton > button:first-child:hover {
        background-color: #b3040f !important;
        color: white !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background: #f1f3f5;
        border-radius: 4px 4px 0 0;
        padding: 10px 16px;
        color: #343a40;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background: var(--negro-carbon);
        color: white !important;
    }

    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #dee2e6;
        border-radius: 4px;
        padding: 12px 16px;
    }
</style>
""", unsafe_allow_html=True)

# ── Helper escudo ────────────────────────────────────────────────────────────
@st.cache_data
def _img_to_base64(path: Path) -> str:
    if not path.exists():
        return ""
    return base64.b64encode(path.read_bytes()).decode("utf-8")

RAYO_LOGO_B64 = _img_to_base64(RAYO_IMG)

# ── Cache de datos ───────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Cargando base de datos de scouting...")
def get_data(debug_csv: bool = False):
    return load_data(DATA_PATH, debug_csv=debug_csv)

# ── Estado ───────────────────────────────────────────────────────────────────
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []

if "messages" not in st.session_state:
    st.session_state.messages = []

df = get_data(debug_csv=True)

# ── Helpers ──────────────────────────────────────────────────────────────────
def fmt_liga(liga: str) -> str:
    return LIGA_DISPLAY.get(str(liga), str(liga))

def parse_valor_mercado(val_str) -> float:
    if pd.isna(val_str) or str(val_str).strip() in ["-", "", "nan"]:
        return 9999.0
    s = str(val_str).lower().replace("€", "").replace(".", "").strip().replace(",", ".")
    if "mill" in s:
        return float(s.replace("mill.", "").replace("mill", "").strip())
    if "mil" in s:
        return float(s.replace("mil", "").strip()) / 1000
    try:
        return float(s)
    except Exception:
        return 9999.0

def add_to_watchlist(player_name: str):
    if player_name not in st.session_state.watchlist:
        st.session_state.watchlist.append(player_name)

def remove_from_watchlist(player_name: str):
    if player_name in st.session_state.watchlist:
        st.session_state.watchlist.remove(player_name)

def get_watchlist_df(df_: pd.DataFrame) -> pd.DataFrame:
    if not st.session_state.watchlist:
        return pd.DataFrame(columns=df_.columns)
    return df_[df_["Name"].isin(st.session_state.watchlist)].copy()

def get_rayo_df(df_: pd.DataFrame) -> pd.DataFrame:
    if "tm_club" not in df_.columns:
        return pd.DataFrame(columns=df_.columns)
    return df_[df_["tm_club"].astype(str).str.contains("Rayo", case=False, na=False)].copy()

def get_contract_opportunities(df_: pd.DataFrame) -> pd.DataFrame:
    if "fin_contrato" not in df_.columns:
        return pd.DataFrame(columns=df_.columns)

    tmp = df_.copy()
    tmp["fin_year"] = tmp["fin_contrato"].astype(str).str[-4:]
    tmp = tmp[tmp["fin_year"].str.isdigit()]
    tmp["fin_year"] = tmp["fin_year"].astype(int)
    tmp = tmp[tmp["fin_year"] <= 2026]
    return tmp.sort_values(["fin_year", "edad"], ascending=[True, True])

def get_top_candidates(df_: pd.DataFrame, n=5) -> pd.DataFrame:
    tmp = df_.copy()
    if "Average Sofascore Rating" in tmp.columns:
        tmp = tmp.sort_values("Average Sofascore Rating", ascending=False)
    return tmp.head(n)

def build_simple_radar(player_a: pd.Series, player_b: pd.Series):
    categories = [
        ("Goals", "Gol"),
        ("Assists", "Asist."),
        ("Key passes", "P. clave"),
        ("Tackles_p90", "Tackles"),
        ("Interceptions_p90", "Intercep."),
        ("Accurate passes %", "Prec. pase"),
    ]

    vals_a = []
    vals_b = []
    labels = []

    for col, lbl in categories:
        if col in df.columns:
            labels.append(lbl)
            vals_a.append(float(player_a.get(col, 0) if pd.notna(player_a.get(col, 0)) else 0))
            vals_b.append(float(player_b.get(col, 0) if pd.notna(player_b.get(col, 0)) else 0))

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=vals_a,
        theta=labels,
        fill='toself',
        name=str(player_a.get("Name", "Jugador A")),
        line_color='#E30613',
        fillcolor='rgba(227, 6, 19, 0.18)'
    ))

    fig.add_trace(go.Scatterpolar(
        r=vals_b,
        theta=labels,
        fill='toself',
        name=str(player_b.get("Name", "Jugador B")),
        line_color='#212529',
        fillcolor='rgba(33, 37, 41, 0.18)'
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True,
        height=500,
        margin=dict(t=40, b=40, l=40, r=40)
    )
    return fig

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h3 style='color: #E30613; font-weight: bold; letter-spacing: 1px;'>RAYO VALLECANO</h3>", unsafe_allow_html=True)
    st.markdown("<h4 style='color: #212529; font-size: 14px; margin-top: -10px;'>DIRECCIÓN DE SCOUTING</h4>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**Usuario:** Cuerpo Técnico")
    st.markdown("**Acceso:** Total (Nivel 1)")
    st.markdown("---")
    st.markdown(f"**Jugadores en Cartera:** {len(st.session_state.watchlist)}")
    st.markdown("---")
    st.markdown("**Base de Datos Activa:**\n- 5 Grandes Ligas\n- Segundas Divisiones")
    st.caption("Última actualización: Hoy, 06:00 AM")

# ── Header ───────────────────────────────────────────────────────────────────
logo_html = (
    f'<img src="data:image/png;base64,{RAYO_LOGO_B64}" '
    f'style="height:70px;width:auto;object-fit:contain;" alt="Rayo Vallecano">'
    if RAYO_LOGO_B64 else ""
)

st.markdown(f"""
<div class='header-franja' style="display:flex;align-items:center;justify-content:space-between;gap:20px;">
    <div>
        <h1>PLATAFORMA DE ANÁLISIS DE MERCADO</h1>
        <p>Sistema avanzado de evaluación de rendimiento y prospección de jugadores</p>
    </div>
    <div style="flex-shrink:0;">
        {logo_html}
    </div>
</div>
""", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Resumen General",
    "Asistente IA",
    "Explorador de Mercado",
    "Comparador de Perfiles",
    "Análisis de Plantilla"
])

with tab1:
    render_resumen_general_tab(df)
  

with tab2:
    render_scout_ia_tab(df)

with tab3:
    render_buscador_perfil_tab(df)

with tab4:
    render_comparador_perfiles_tab(df)

with tab5:
    render_analisis_plantilla_tab(df)