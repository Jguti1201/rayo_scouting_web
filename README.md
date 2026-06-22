# ⚡ Rayo Vallecano Scout IA — Plataforma de Scouting y Análisis de Mercado

Plataforma de análisis de rendimiento, scouting y soporte a decisiones para dirección deportiva, enfocada en el **Rayo Vallecano**.  
El proyecto integra scraping, normalización de datos, feature engineering, clustering táctico, scoring de adaptación y una app interactiva en Streamlit con asistentes IA y generación de informes PDF.

---

## 📌 Objetivo del proyecto

Construir un sistema integral que permita:

- Centralizar datos de jugadores de múltiples ligas.
- Analizar rendimiento individual y comparativo por posición.
- Detectar oportunidades de mercado (contratos, coste, ajuste táctico).
- Priorizar candidatos mediante scoring cuantitativo y análisis asistido por IA.
- Documentar decisiones de scouting con informes automatizados.

---

## 🧱 Arquitectura funcional (resumen)

El flujo del proyecto está dividido en 4 grandes bloques:

1. **Data Ingestion (Scraping + fuentes externas)**
   - Extracción de datos de plantilla y mercado desde Transfermarkt.
   - Extracción de estadísticas por liga desde SofaScore (multi-pestaña, paginación, checkpoints).

2. **Data Processing (ETL + Merge)**
   - Limpieza y estandarización de nombres/formatos.
   - Cálculo de métricas derivadas (p90, percentiles, etc.).
   - Unificación por liga y merge global (`all_leagues_master`).
   - Integración SofaScore + Transfermarkt con matching exacto/fuzzy.

3. **Analytics & Intelligence**
   - Segmentación táctica con clustering por posición.
   - Similaridad entre jugadores con cosine similarity.
   - Scoring de adaptación al contexto Rayo Vallecano.
   - Detección de riesgos de plantilla (contratos, salarios, gaps posicionales).

4. **Presentation Layer (App Streamlit)**
   - Dashboard ejecutivo.
   - Explorador de mercado.
   - Comparador de perfiles.
   - Asistente IA conversacional para búsqueda de perfiles.
   - Generación de informes PDF con análisis técnico.

---

## 🗂️ Estructura recomendada del repositorio

```bash
rayo-scouting-ia/
│
├── app/
│   ├── gemini_app.py
│   └── tabs/
│       ├── gemini_tab_resumen_general.py
│       ├── gemini_tab_scout_ia.py
│       ├── gemini_tab_explorador_mercado.py
│       ├── gemini_tab_comparador_perfiles.py
│       └── gemini_tab_analisis_plantilla.py
│
├── backend/
│   ├── utils/
│   │   ├── gemini_data_loader.py
│   │   ├── gemini_feature_engineering.py
│   │   ├── gemini_clustering.py
│   │   ├── gemini_adaptation_score.py
│   │   ├── gemini_dashboard_metrics.py
│   │   ├── gemini_report_generation.py
│   │   └── ...
│   └── data/
│       ├── raw/
│       ├── processed/
│       └── final/
│
├── notebooks/
│   ├── info_transfermarket.ipynb
│   ├── info_player_stats_big5_v3.ipynb
│   ├── rayo_vallecano_10_stats.ipynb
│   └── ...
│
├── scripts/
│   ├── analisis_fichajes.py
│   ├── merge_con_transfermarket.py
│   └── transfermarkt_links_scraper.py
│
├── reports/
│   ├── generated_pdfs/
│   └── figures/
│
├── tests/
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🔐 Variables de entorno (.env)

Crea un archivo `.env` en la raíz del proyecto:

```env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxxxxx
```

> ✅ Nunca subas `.env` al repositorio.

---

## 🚫 `.gitignore` recomendado (mínimo)

```gitignore
# Entorno
.venv/
venv/
__pycache__/
*.pyc

# Secrets
.env
.env.*

# Jupyter
.ipynb_checkpoints/

# Streamlit cache
.streamlit/secrets.toml

# Datos pesados / outputs
backend/data/raw/
backend/data/processed/
backend/data/final/
dataset_final/
sofascore_data/
reports/generated_pdfs/
*.log
```

---

## ⚙️ Instalación

```bash
# 1) Clonar repo
git clone <URL_DEL_REPO>
cd rayo-scouting-ia

# 2) Crear entorno virtual
python -m venv .venv

# 3) Activar entorno (Windows)
.venv\Scripts\activate

# 4) Instalar dependencias
pip install -r requirements.txt
```

---

## ▶️ Ejecución de la app

```bash
streamlit run app/gemini_app.py
```

---

## 🧪 Flujo de datos recomendado (orden de ejecución)

1. Ejecutar scraping de Transfermarkt.
2. Ejecutar scraping de SofaScore por ligas.
3. Unificar pestañas por liga (`*_players_unified.csv`).
4. Merge con datos de mercado Transfermarkt.
5. Generar `all_leagues_master`.
6. Lanzar app Streamlit.

---

## 🤖 Módulos IA

- **OpenAI**: extracción conversacional de perfil de scouting + generación de análisis por candidato.
- **Anthropic (Claude)**: redacción técnica avanzada para informes scouting PDF.
- **Sistema de fallback** en caso de error API o respuesta JSON inválida.

---

## 📈 Funcionalidades principales

- Filtros avanzados por posición, edad, liga, valor de mercado y minutos.
- Perfilado táctico por clusters (por demarcación).
- Comparación entre jugadores con radar por percentiles posicionales.
- Similares automáticos por cosine similarity.
- Watchlist operativa para seguimiento de jugadores.
- Análisis de plantilla (salarios, contratos, riesgos y necesidades).
- Informe PDF automático con recomendación final.

---

## ✅ Buenas prácticas aplicadas

- Separación por capas (ingesta, procesamiento, análisis, presentación).
- Reutilización modular de funciones.
- Logging en procesos críticos.
- Checkpointing y guardado incremental en scraping.
- Limpieza robusta de encoding y normalización de nombres.
- Configuración por variables de entorno para credenciales sensibles.

---

## 📚 Contexto académico (TFM)

Este repositorio forma parte de un **Trabajo de Fin de Máster** centrado en la aplicación de técnicas de **Big Data, analítica deportiva e IA aplicada al scouting profesional** para la toma de decisiones en dirección deportiva.

---

## 👤 Autor

**Jaime Gutiérrez**  
TFM — Scouting Inteligente aplicado al Rayo Vallecano

---

## 📄 Licencia

Uso académico / investigación.  
(Si quieres, puedes cambiarlo a MIT o la licencia que prefieras.)