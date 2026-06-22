import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from matplotlib.patches import Arc, Circle, FancyArrowPatch
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection

from rayo_analisis_juego import (
    load_season_data_from_s3,
    build_global_summary,
    build_home_away_summary,
    build_strengths_weaknesses,
    build_tactical_identity,
    build_correlation_summary,
    build_expert_export_text,
)

# ============================================================
# CONFIG STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Rayo Vallecano - Tactical Intelligence Suite",
    page_icon="⚽",
    layout="wide"
)

sns.set_theme(style="whitegrid", context="talk")

RAYO_COLOR = "#D71920"
SECONDARY_COLOR = "#333333"
ACCENT_COLOR = "#F2C94C"
HOME_COLOR = "#2ca02c"
AWAY_COLOR = "#1f77b4"


# ============================================================
# FUNCIONES VISUALES
# ============================================================

def draw_pitch(ax=None, pitch_color="white", line_color="#222222", lw=2):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    ax.set_facecolor(pitch_color)

    ax.plot([0, 0], [0, 100], color=line_color, lw=lw)
    ax.plot([0, 100], [100, 100], color=line_color, lw=lw)
    ax.plot([100, 100], [100, 0], color=line_color, lw=lw)
    ax.plot([100, 0], [0, 0], color=line_color, lw=lw)

    ax.plot([50, 50], [0, 100], color=line_color, lw=lw)
    centre_circle = Circle((50, 50), 9.15, fill=False, color=line_color, lw=lw)
    ax.add_patch(centre_circle)
    ax.add_patch(Circle((50, 50), 0.8, color=line_color))

    ax.plot([0, 16.5], [79, 79], color=line_color, lw=lw)
    ax.plot([16.5, 16.5], [79, 21], color=line_color, lw=lw)
    ax.plot([16.5, 0], [21, 21], color=line_color, lw=lw)

    ax.plot([100, 83.5], [79, 79], color=line_color, lw=lw)
    ax.plot([83.5, 83.5], [79, 21], color=line_color, lw=lw)
    ax.plot([83.5, 100], [21, 21], color=line_color, lw=lw)

    ax.plot([0, 5.5], [63, 63], color=line_color, lw=lw)
    ax.plot([5.5, 5.5], [63, 37], color=line_color, lw=lw)
    ax.plot([5.5, 0], [37, 37], color=line_color, lw=lw)

    ax.plot([100, 94.5], [63, 63], color=line_color, lw=lw)
    ax.plot([94.5, 94.5], [63, 37], color=line_color, lw=lw)
    ax.plot([94.5, 100], [37, 37], color=line_color, lw=lw)

    ax.add_patch(Circle((11, 50), 0.8, color=line_color))
    ax.add_patch(Circle((89, 50), 0.8, color=line_color))

    left_arc = Arc((11, 50), height=18.3, width=18.3, angle=0, theta1=310, theta2=50, color=line_color, lw=lw)
    right_arc = Arc((89, 50), height=18.3, width=18.3, angle=0, theta1=130, theta2=230, color=line_color, lw=lw)
    ax.add_patch(left_arc)
    ax.add_patch(right_arc)

    # tercios
    ax.plot([33.33, 33.33], [0, 100], color="lightgrey", lw=1, ls="--")
    ax.plot([66.67, 66.67], [0, 100], color="lightgrey", lw=1, ls="--")

    # carriles 5
    for y in [20, 40, 60, 80]:
        ax.plot([0, 100], [y, y], color="lightgrey", lw=1, ls="--")

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


def radar_factory(num_vars, frame='circle'):
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):
        name = 'radar'
        RESOLUTION = 1

        def fill(self, *args, closed=True, **kwargs):
            return super().fill(*args, closed=closed, **kwargs)

        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                x, y = line.get_data()
                if x[0] != x[-1]:
                    x = np.append(x, x[0])
                    y = np.append(y, y[0])
                    line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

    register_projection(RadarAxes)
    return theta


def compute_percentile_profile(df, metrics):
    """
    Convierte las medias del equipo a percentiles internos respecto a su propia temporada.
    """
    result = {}
    for m in metrics:
        series = df[m].dropna()
        if len(series) == 0:
            result[m] = np.nan
            continue
        mean_value = series.mean()
        percentile = (series <= mean_value).mean() * 100
        result[m] = percentile
    return pd.Series(result)


# ============================================================
# CARGA DE DATOS
# ============================================================

@st.cache_data(show_spinner=True)
def load_data():
    return load_season_data_from_s3(
        event_types_csv="F1_opta_event_types.csv",
        qualifier_types_csv="F3_opta_qualifier_types.csv",
    )


try:
    events_season, season_kpis = load_data()
except Exception as e:
    st.error(f"Error cargando datos desde S3: {e}")
    st.stop()

if season_kpis.empty:
    st.error("No se pudieron calcular KPIs de temporada.")
    st.stop()

global_summary = build_global_summary(season_kpis)
try:
    home_away_summary = build_home_away_summary(season_kpis)
except Exception:
    home_away_summary = pd.DataFrame()

strengths, weaknesses = build_strengths_weaknesses(season_kpis)
tactical_identity = build_tactical_identity(season_kpis)
corr_points, corr_xg_net = build_correlation_summary(season_kpis)
expert_export_text = build_expert_export_text(season_kpis)

rayo_events = events_season[events_season["is_rayo_event"]].copy()

# ============================================================
# NAVEGACIÓN
# ============================================================

st.sidebar.title("⚽ Rayo Tactical Intelligence")
page = st.sidebar.radio(
    "Selecciona una vista",
    [
        "Resumen ejecutivo",
        "Build-up y fase ofensiva",
        "Último tercio y finalización",
        "Defensa y presión",
        "Transiciones y ABP",
        "Correlación y ADN táctico",
        "Informe táctico élite",
        "Exportar para análisis experto",
    ]
)

# ============================================================
# RESUMEN EJECUTIVO
# ============================================================

if page == "Resumen ejecutivo":
    st.title("Rayo Vallecano - Resumen ejecutivo de temporada")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Partidos", season_kpis["match_id"].nunique())
    col2.metric("PPDA medio", f"{season_kpis['ppda'].mean():.2f}")
    col3.metric("Field Tilt medio", f"{season_kpis['field_tilt_pct'].mean():.2f}%")
    col4.metric("xG net proxy", f"{season_kpis['xg_proxy_net'].mean():.2f}")

    st.subheader("ADN táctico propuesto")
    st.success(tactical_identity)

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Fortalezas")
        if strengths:
            for s in strengths:
                st.write(f"- {s}")
        else:
            st.write("- No se detectan fortalezas dominantes con los umbrales actuales.")

    with c2:
        st.subheader("Debilidades")
        if weaknesses:
            for w in weaknesses:
                st.write(f"- {w}")
        else:
            st.write("- No se detectan debilidades dominantes con los umbrales actuales.")

    st.subheader("Resumen numérico")
    st.dataframe(global_summary)


# ============================================================
# BUILD-UP Y FASE OFENSIVA
# ============================================================

elif page == "Build-up y fase ofensiva":
    st.title("Fase ofensiva y construcción")

    avg = season_kpis.mean(numeric_only=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precisión de pase", f"{avg['pass_accuracy_pct']:.2f}%")
    c2.metric("Verticality ratio", f"{avg['verticality_ratio']:.2f}")
    c3.metric("Field Tilt", f"{avg['field_tilt_pct']:.2f}%")
    c4.metric("Entradas a último tercio", f"{avg['entries_final_third_by_pass']:.2f}")

    # 1. barras por tercio
    thirds_df = pd.DataFrame({
        "third": ["Tercio propio", "Tercio medio", "Tercio rival"],
        "value": [avg["passes_def_third"], avg["passes_mid_third"], avg["passes_att_third"]]
    })

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=thirds_df, x="third", y="value", palette="Reds_r", ax=ax)
    ax.set_title("Volumen medio de pase por tercio")
    ax.set_xlabel("")
    ax.set_ylabel("Pases por partido")
    st.pyplot(fig)

    # 2. mapa de flechas de pases progresivos
    st.subheader("Mapa de flechas de pases progresivos")

    progressive_passes = rayo_events[
        (rayo_events["typeId"] == 1) &
        (rayo_events["outcome"] == 1) &
        (rayo_events["pass_end_x"].notna()) &
        ((rayo_events["pass_end_x"] - rayo_events["x"]) >= 10)
    ].copy()

    sample_prog = progressive_passes.sample(min(len(progressive_passes), 250), random_state=42) if len(progressive_passes) > 0 else progressive_passes

    fig, ax = plt.subplots(figsize=(14, 10))
    draw_pitch(ax=ax)

    for _, row in sample_prog.iterrows():
        arrow = FancyArrowPatch(
            (row["x"], row["y"]),
            (row["pass_end_x"], row["pass_end_y"]),
            arrowstyle="->",
            mutation_scale=8,
            color=RAYO_COLOR,
            alpha=0.25,
            linewidth=1.3
        )
        ax.add_patch(arrow)

    ax.set_title("Dirección y origen de los pases progresivos")
    st.pyplot(fig)

    # 3. heatmap zonal tercios/carriles
    st.subheader("Heatmap zonal por tercios y carriles")

    zone_counts = (
        rayo_events[rayo_events["typeId"] == 1]
        .groupby(["pitch_third", "pitch_lane_5"])
        .size()
        .reset_index(name="n")
    )

    third_order = ["Defensive Third", "Middle Third", "Attacking Third"]
    lane_order = ["Left Wing", "Left Halfspace", "Central", "Right Halfspace", "Right Wing"]

    pivot = zone_counts.pivot(index="pitch_lane_5", columns="pitch_third", values="n").reindex(index=lane_order, columns=third_order)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="Reds", ax=ax)
    ax.set_title("Circulación por zonas del campo")
    ax.set_xlabel("Tercio")
    ax.set_ylabel("Carril")
    st.pyplot(fig)


# ============================================================
# ÚLTIMO TERCIO Y FINALIZACIÓN
# ============================================================

elif page == "Último tercio y finalización":
    st.title("Comportamiento en último tercio y finalización")

    avg = season_kpis.mean(numeric_only=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Tiros", f"{avg['shots_total']:.2f}")
    c2.metric("xG proxy total", f"{avg['xg_proxy_total']:.2f}")
    c3.metric("xG por tiro", f"{avg['xg_proxy_per_shot']:.3f}")
    c4.metric("Goles - xG proxy", f"{avg['goals_minus_xg_proxy']:.2f}")

    # mapa de tiros
    shots = rayo_events[rayo_events["typeId"].isin([13, 14, 15, 16])].copy()

    fig, ax = plt.subplots(figsize=(14, 10))
    draw_pitch(ax=ax)
    sns.scatterplot(
        data=shots,
        x="x", y="y",
        hue="typeName",
        palette="Set1",
        s=80,
        ax=ax
    )
    ax.set_title("Mapa de tiros de la temporada")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    st.pyplot(fig)

    # distribución de tiro
    shot_zone_df = season_kpis[["shots_six_yard_box", "shots_box", "shots_outside_box"]].mean().reset_index()
    shot_zone_df.columns = ["zone", "value"]

    zone_label_map = {
        "shots_six_yard_box": "Área chica",
        "shots_box": "Área",
        "shots_outside_box": "Fuera del área"
    }
    shot_zone_df["zone"] = shot_zone_df["zone"].map(zone_label_map)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=shot_zone_df, x="zone", y="value", palette="rocket", ax=ax)
    ax.set_title("Distribución media de remates por zona")
    ax.set_xlabel("")
    ax.set_ylabel("Tiros por partido")
    st.pyplot(fig)

    # ataques rápidos vs posicionales
    attack_type_df = pd.DataFrame({
        "attack_type": ["Ataques rápidos", "Ataques posicionales"],
        "value": [avg["fast_attacks"], avg["positional_attacks"]]
    })

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=attack_type_df, x="attack_type", y="value", palette=["#ef4444", "#111827"], ax=ax)
    ax.set_title("Tipología de ataque")
    ax.set_xlabel("")
    ax.set_ylabel("Media por partido")
    st.pyplot(fig)


# ============================================================
# DEFENSA Y PRESIÓN
# ============================================================

elif page == "Defensa y presión":
    st.title("Fase defensiva y presión")

    avg = season_kpis.mean(numeric_only=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PPDA", f"{avg['ppda']:.2f}")
    c2.metric("PPDA tercio final", f"{avg['ppda_final_third']:.2f}")
    c3.metric("Altura media recuperación", f"{avg['avg_recovery_x']:.2f}")
    c4.metric("Tackle success", f"{avg['tackle_success_pct']:.2f}%")

    # mapa de recuperaciones
    recoveries = rayo_events[rayo_events["typeId"] == 49].copy()

    fig, ax = plt.subplots(figsize=(14, 10))
    draw_pitch(ax=ax)
    if not recoveries.empty:
        sns.kdeplot(
            data=recoveries,
            x="x", y="y",
            fill=True,
            thresh=0.05,
            levels=50,
            cmap="Greens",
            alpha=0.7,
            ax=ax
        )
    ax.set_title("Mapa de recuperaciones")
    st.pyplot(fig)

    # evolución PPDA
    temp = season_kpis.sort_values("match_date").copy()
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.lineplot(data=temp, x=np.arange(len(temp)), y="ppda", marker="o", color=RAYO_COLOR, linewidth=2.5, ax=ax)
    ax.set_title("Evolución del PPDA")
    ax.set_xlabel("Partido")
    ax.set_ylabel("PPDA")
    st.pyplot(fig)

    # heatmap zonal defensivo
    def_events = rayo_events[rayo_events["typeId"].isin([4, 7, 8, 12, 49, 74])].copy()
    zone_counts = def_events.groupby(["pitch_third", "pitch_lane_5"]).size().reset_index(name="n")

    third_order = ["Defensive Third", "Middle Third", "Attacking Third"]
    lane_order = ["Left Wing", "Left Halfspace", "Central", "Right Halfspace", "Right Wing"]
    pivot = zone_counts.pivot(index="pitch_lane_5", columns="pitch_third", values="n").reindex(index=lane_order, columns=third_order)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="Greens", ax=ax)
    ax.set_title("Actividad defensiva por zonas")
    ax.set_xlabel("Tercio")
    ax.set_ylabel("Carril")
    st.pyplot(fig)


# ============================================================
# TRANSICIONES Y ABP
# ============================================================

elif page == "Transiciones y ABP":
    st.title("Transiciones y balón parado")

    avg = season_kpis.mean(numeric_only=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Counterpress 5s", f"{avg['counterpress_5s_pct']:.2f}%")
    c2.metric("Recoveries <5s tras pérdida", f"{avg['recoveries_5s_after_loss']:.2f}")
    c3.metric("Set-piece shots", f"{avg['set_piece_shots']:.2f}")
    c4.metric("Set-piece goals", f"{avg['set_piece_goals']:.2f}")

    trans_df = pd.DataFrame({
        "metric": [
            "Counterpress 5s %",
            "Recoveries 5s",
            "Pérdidas campo propio",
            "Set-piece shots",
            "Corners for"
        ],
        "value": [
            avg["counterpress_5s_pct"],
            avg["recoveries_5s_after_loss"],
            avg["losses_own_half"],
            avg["set_piece_shots"],
            avg["corners_for"]
        ]
    })

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=trans_df, x="metric", y="value", palette="magma", ax=ax)
    ax.set_title("Indicadores medios de transición y ABP")
    ax.set_xlabel("")
    ax.set_ylabel("Valor medio por partido")
    ax.tick_params(axis="x", rotation=20)
    st.pyplot(fig)


# ============================================================
# CORRELACIÓN Y ADN TÁCTICO
# ============================================================

elif page == "Correlación y ADN táctico":
    st.title("Matriz de correlación con resultado y ADN táctico")

    st.subheader("ADN táctico")
    st.success(tactical_identity)

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Fortalezas")
        if strengths:
            for s in strengths:
                st.write(f"- {s}")
        else:
            st.write("- No claramente dominantes.")

    with c2:
        st.subheader("Debilidades")
        if weaknesses:
            for w in weaknesses:
                st.write(f"- {w}")
        else:
            st.write("- No claramente dominantes.")

    st.subheader("Correlación con puntos")
    if corr_points.empty:
        st.warning("No se pudo calcular la correlación con puntos porque falta la columna 'result_points' o no hay suficientes datos numéricos.")
    else:
        st.dataframe(corr_points.head(12))

    st.subheader("Correlación con xG net proxy")
    if corr_xg_net.empty:
        st.warning("No se pudo calcular la correlación con xG net proxy porque falta la columna 'xg_proxy_net' o no hay suficientes datos.")
    else:
        st.dataframe(corr_xg_net.head(12))

    # scatter
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    plotted = False

    if "field_tilt_pct" in season_kpis.columns and "result_points" in season_kpis.columns:
        sns.scatterplot(data=season_kpis, x="field_tilt_pct", y="result_points", ax=axes[0, 0], color=RAYO_COLOR)
        axes[0, 0].set_title("Field Tilt vs puntos")
        plotted = True
    else:
        axes[0, 0].set_title("Field Tilt vs puntos (no disponible)")
        axes[0, 0].axis("off")

    if "ppda" in season_kpis.columns and "result_points" in season_kpis.columns:
        sns.scatterplot(data=season_kpis, x="ppda", y="result_points", ax=axes[0, 1], color=SECONDARY_COLOR)
        axes[0, 1].set_title("PPDA vs puntos")
        plotted = True
    else:
        axes[0, 1].set_title("PPDA vs puntos (no disponible)")
        axes[0, 1].axis("off")

    if "xg_proxy_net" in season_kpis.columns and "result_points" in season_kpis.columns:
        sns.scatterplot(data=season_kpis, x="xg_proxy_net", y="result_points", ax=axes[1, 0], color=HOME_COLOR)
        axes[1, 0].set_title("xG net proxy vs puntos")
        plotted = True
    else:
        axes[1, 0].set_title("xG net proxy vs puntos (no disponible)")
        axes[1, 0].axis("off")

    if "counterpress_5s_pct" in season_kpis.columns and "result_points" in season_kpis.columns:
        sns.scatterplot(data=season_kpis, x="counterpress_5s_pct", y="result_points", ax=axes[1, 1], color=AWAY_COLOR)
        axes[1, 1].set_title("Counterpress 5s vs puntos")
        plotted = True
    else:
        axes[1, 1].set_title("Counterpress 5s vs puntos (no disponible)")
        axes[1, 1].axis("off")

    if plotted:
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("No hay suficientes columnas disponibles para dibujar los gráficos de correlación.")

    # radar con percentiles
    st.subheader("Radar con percentiles internos de temporada")

    radar_metrics = [
        "field_tilt_pct",
        "progressive_passes",
        "entries_penalty_area_by_pass",
        "xg_proxy_per_shot",
        "avg_recovery_x",
        "ppda",
        "counterpress_5s_pct"
    ]

    radar_labels = [
        "Field Tilt",
        "Pases progresivos",
        "Entradas área",
        "xG/tiro",
        "Altura recuperación",
        "PPDA",
        "Counterpress"
    ]

    percentile_profile = compute_percentile_profile(season_kpis, radar_metrics)

    # invertimos PPDA porque un valor bajo es mejor/agresivo
    if pd.notna(percentile_profile["ppda"]):
        percentile_profile["ppda"] = 100 - percentile_profile["ppda"]

    theta = radar_factory(len(radar_metrics), frame='circle')

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection='radar'))
    ax.set_title("Radar de percentiles internos", position=(0.5, 1.1), fontsize=18)
    ax.plot(theta, percentile_profile.values, color=RAYO_COLOR, linewidth=3)
    ax.fill(theta, percentile_profile.values, color=RAYO_COLOR, alpha=0.25)
    ax.set_varlabels(radar_labels)
    st.pyplot(fig)


# ============================================================
# INFORME TÁCTICO ÉLITE
# ============================================================

elif page == "Informe táctico élite":
    st.title("Informe táctico élite")
    st.markdown("""
### Mandato analítico
Este panel resume el comportamiento del equipo con lenguaje de analista táctico profesional, cruzando métricas estructurales, volumen ofensivo, presión, transición y resultado.
    """)

    avg = season_kpis.mean(numeric_only=True)

    st.subheader("1. Fase ofensiva y construcción")
    st.write(f"""
- El equipo promedia **{avg['passes_total']:.2f} pases** por partido, con una **precisión del {avg['pass_accuracy_pct']:.2f}%**.
- El reparto por tercios es: **{avg['passes_def_third']:.2f}** en tercio propio, **{avg['passes_mid_third']:.2f}** en tercio medio y **{avg['passes_att_third']:.2f}** en tercio rival.
- Su ratio de verticalidad es **{avg['verticality_ratio']:.2f}**, calculado como pases progresivos frente a pases laterales/atrás.
- La altura media de circulación se sitúa en **x={avg['avg_circulation_height']:.2f}**, útil para detectar si el equipo construye en bloque bajo, medio o alto.
- El **Field Tilt medio es {avg['field_tilt_pct']:.2f}%**, indicador de cuánto inclina el juego hacia el último tercio rival.
- Entra al último tercio por pase **{avg['entries_final_third_by_pass']:.2f}** veces por partido y al área penal **{avg['entries_penalty_area_by_pass']:.2f}** veces.
- Genera **{avg['key_passes_proxy']:.2f} key passes proxy** y **{avg['progressive_carries_proxy']:.2f} conducciones progresivas proxy** por partido.
    """)

    st.subheader("2. Comportamiento en último tercio y finalización")
    st.write(f"""
- El equipo remata **{avg['shots_total']:.2f} veces** por partido.
- La distribución es:
  - **{avg['shots_six_yard_box']:.2f}** tiros en área chica
  - **{avg['shots_box']:.2f}** tiros dentro del área
  - **{avg['shots_outside_box']:.2f}** tiros fuera del área
- El **xG proxy total** es **{avg['xg_proxy_total']:.2f}**, con **{avg['xg_proxy_per_shot']:.3f} xG proxy por tiro**.
- La diferencia entre goles y xG proxy es **{avg['goals_minus_xg_proxy']:.2f}**, útil para detectar sobre- o infra-finalización.
- En cuanto a tipología de ataque, el equipo registra **{avg['fast_attacks']:.2f} ataques rápidos** frente a **{avg['positional_attacks']:.2f} ataques posicionales** por partido.
    """)

    st.subheader("3. Fase defensiva y presión")
    st.write(f"""
- La altura media de recuperación está en **x={avg['avg_recovery_x']:.2f}**.
- El **PPDA medio es {avg['ppda']:.2f}**, y el **PPDA en tercio final {avg['ppda_final_third']:.2f}**.
- El equipo promedia **{avg['tackles']:.2f} tackles**, con un éxito del **{avg['tackle_success_pct']:.2f}%**.
- Además, suma **{avg['interceptions']:.2f} intercepciones** y **{avg['high_recoveries']:.2f} recuperaciones altas** por partido.
- Tras pérdida, logra **{avg['recoveries_5s_after_loss']:.2f} recuperaciones** en los primeros 5 segundos.
    """)

    st.subheader("4. Transiciones y balón parado")
    st.write(f"""
- El éxito de contra-presión en 5 segundos se sitúa en **{avg['counterpress_5s_pct']:.2f}%**.
- El equipo pierde **{avg['losses_own_half']:.2f} balones en campo propio** por partido.
- En acciones a balón parado genera **{avg['set_piece_shots']:.2f} tiros** y **{avg['set_piece_goals']:.2f} goles** por partido.
- Además, fuerza **{avg['corners_for']:.2f} córners** y **{avg['free_kick_shots']:.2f} tiros de libre directo/indirecto**.
    """)

    st.subheader("5. Métricas con mayor relación con el resultado")
    top_points = corr_points.drop(index=["result_points"], errors="ignore").head(3)
    top_xg = corr_xg_net.drop(index=["xg_proxy_net"], errors="ignore").head(3)

    st.write("**Top 3 métricas correlacionadas con puntos:**")
    st.dataframe(top_points)

    st.write("**Top 3 métricas correlacionadas con xG net proxy:**")
    st.dataframe(top_xg)

    st.subheader("6. Diagnóstico final")
    st.success(f"ADN táctico propuesto: **{tactical_identity}**")

    st.write("**Fortalezas detectadas**")
    for s in strengths:
        st.write(f"- {s}")

    st.write("**Debilidades detectadas**")
    for w in weaknesses:
        st.write(f"- {w}")


# ============================================================
# EXPORTAR PARA ANÁLISIS EXPERTO
# ============================================================

elif page == "Exportar para análisis experto":
    st.title("Exportar para análisis experto")
    st.markdown("""
Copia este bloque y pégamelo para que te devuelva una lectura táctica de alto nivel del estilo del equipo.
    """)
    st.text_area("Bloque listo para copiar", expert_export_text, height=700)