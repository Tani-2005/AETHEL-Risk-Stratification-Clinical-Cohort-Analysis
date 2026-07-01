"""
dashboard.py
============
AETHEL Interactive Clinical Dashboard

Streamlit application for dynamic geospatial risk mapping and
individualised patient telemetry profiling across the 100-city
Pan-European cohort.

Launch
------
    streamlit run src/visualization/dashboard.py

The dashboard loads the analytical cohort from ``data/processed/``
and — if available — real VIMP values from ``outputs/metrics/vimp.csv``
(produced by the R survival model).  If the VIMP file does not yet exist,
it falls back gracefully to representative placeholder values.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src.utils.constants import Columns, EnvPollutants, Features, RiskLabels, SmokerLabels
from src.utils.paths import DataPaths, OutputPaths

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AETHEL Clinical Engine",
    layout="wide",
    page_icon="🧬",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    h1, h2, h3 { color: #2c3e50; font-family: 'Helvetica Neue', sans-serif; }
    [data-testid="stMetricValue"] { font-size: 2rem; color: #1f77b4; }
    </style>
""", unsafe_allow_html=True)

st.title("🧬 AETHEL: Pan-European Health Intelligence")
st.markdown("**Interactive Risk Stratification & Clinical Cohort Analysis**")
st.markdown("---")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@st.cache_data
def load_data() -> pd.DataFrame:
    """
    Load and enrich the analytical cohort for dashboard display.

    Computes a composite risk score from PM2.5 exposure, polygenic risk,
    and smoking status, then bins it into Low / Moderate / High categories.

    Returns
    -------
    pd.DataFrame
        Enriched cohort with ``calculated_risk_score`` and ``Risk Category``.
    """
    df = pd.read_csv(DataPaths.ANALYTICAL_COHORT)
    df[Columns.CALCULATED_RISK_SCORE] = (
        (df[Columns.AVG_PM25_EXPOSURE] * 0.4)
        + (df[Columns.GENOMIC_RISK_SCORE] * 15)
        + (df[Columns.IS_SMOKER] * 20)
    )
    score = df[Columns.CALCULATED_RISK_SCORE]
    df[Columns.CALCULATED_RISK_SCORE] = 100 * (score - score.min()) / (score.max() - score.min())
    df[Columns.RISK_CATEGORY] = pd.cut(
        df[Columns.CALCULATED_RISK_SCORE],
        bins=[-1, 33, 66, 100],
        labels=RiskLabels.ALL,
    )
    return df


@st.cache_data
def load_vimp() -> pd.DataFrame:
    """
    Load variable importance scores from the R model output.

    Attempts to read ``outputs/metrics/vimp.csv`` (produced by
    ``src/models/survival_model.R``).  Falls back gracefully to the
    representative placeholder values in ``EnvPollutants.FALLBACK_VIMP``
    if the file does not yet exist.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ``Feature``, ``Importance``.
    """
    if OutputPaths.VIMP_CSV.exists():
        vimp = pd.read_csv(OutputPaths.VIMP_CSV)
        return vimp.sort_values("Importance")

    # Graceful fallback — shown until R model has been run
    fallback = pd.DataFrame(
        list(EnvPollutants.FALLBACK_VIMP.items()),
        columns=["Feature", "Importance"],
    ).sort_values("Importance")
    return fallback


try:
    df = load_data()
except FileNotFoundError:
    st.error(
        f"Dataset missing. Ensure `{DataPaths.ANALYTICAL_COHORT}` exists.\n\n"
        "Run the full pipeline first: `python scripts/run_pipeline.py`"
    )
    st.stop()


# ---------------------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------------------

st.sidebar.header("🔬 Cohort Filters")
risk_filter = st.sidebar.multiselect(
    "Risk Category:", options=RiskLabels.ALL, default=RiskLabels.ALL
)
city_filter = st.sidebar.multiselect(
    "Region:", options=df[Columns.CITY].unique(), default=df[Columns.CITY].unique()
)
smoker_filter = st.sidebar.radio(
    "Smoking Status:", ["All", "Current Smoker", "Non-Smoker"]
)
age_range = st.sidebar.slider(
    "Age Range:",
    int(df[Columns.AGE].min()),
    int(df[Columns.AGE].max()),
    (int(df[Columns.AGE].min()), int(df[Columns.AGE].max())),
)

filtered_df = df[
    (df[Columns.RISK_CATEGORY].isin(risk_filter))
    & (df[Columns.CITY].isin(city_filter))
]
if smoker_filter == "Current Smoker":
    filtered_df = filtered_df[filtered_df[Columns.IS_SMOKER] == 1]
elif smoker_filter == "Non-Smoker":
    filtered_df = filtered_df[filtered_df[Columns.IS_SMOKER] == 0]
filtered_df = filtered_df[
    (filtered_df[Columns.AGE] >= age_range[0])
    & (filtered_df[Columns.AGE] <= age_range[1])
]


# ---------------------------------------------------------------------------
# Top-level KPIs
# ---------------------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)
col1.metric("Active Patients in View", f"{len(filtered_df):,}")
col2.metric("Critical Risk (>75%)", len(filtered_df[filtered_df[Columns.CALCULATED_RISK_SCORE] > 75]))
col3.metric(EnvPollutants.PM25_LABEL, f"{filtered_df[Columns.AVG_PM25_EXPOSURE].mean():.1f}")
col4.metric("Total Events", filtered_df[Columns.EVENT_OCCURRED].sum())
st.markdown("<br>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Core visualizations
# ---------------------------------------------------------------------------

map_col, chart_col = st.columns([1.5, 1])

with map_col:
    st.subheader("📍 Continental Hazard Distribution")
    fig_map = px.scatter_mapbox(
        filtered_df,
        lat=Columns.LAT,
        lon=Columns.LON,
        color=Columns.CALCULATED_RISK_SCORE,
        size=Columns.AVG_PM25_EXPOSURE,
        color_continuous_scale="RdYlBu_r",
        size_max=15,
        zoom=4,
        center={"lat": 51.0, "lon": 10.0},
        mapbox_style="carto-positron",
        hover_name=Columns.PATIENT_ID,
        hover_data={
            Columns.LAT: False,
            Columns.LON: False,
            Columns.CITY: True,
            Columns.AGE: True,
            Columns.RISK_CATEGORY: True,
        },
    )
    fig_map.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    st.plotly_chart(fig_map, use_container_width=True)

with chart_col:
    st.subheader("📊 Primary Risk Drivers")
    vimp_data = load_vimp()
    fig_bar = px.bar(
        vimp_data,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Blues",
    )
    fig_bar.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)


# ---------------------------------------------------------------------------
# Telemetry data table
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("📋 Multi-Regional Telemetry Registry")

display_df = filtered_df[Features.DASHBOARD_DISPLAY_COLUMNS].copy()
display_df[Columns.IS_SMOKER] = display_df[Columns.IS_SMOKER].map(SmokerLabels.MAPPING)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        Columns.PATIENT_ID: "Patient ID",
        Columns.CITY: "Location",
        Columns.AGE: st.column_config.NumberColumn("Age", format="%d"),
        Columns.IS_SMOKER: "Smoker",
        Columns.TOWNSEND_INDEX: st.column_config.NumberColumn("Deprivation Index", format="%.2f"),
        Columns.AVG_PM25_EXPOSURE: st.column_config.NumberColumn(EnvPollutants.PM25_LABEL, format="%.2f"),
        Columns.GENOMIC_RISK_SCORE: st.column_config.NumberColumn("Polygenic Score", format="%.3f"),
        Columns.CALCULATED_RISK_SCORE: st.column_config.ProgressColumn(
            "Overall Risk", format="%.1f", min_value=0, max_value=100
        ),
        Columns.RISK_CATEGORY: "Category",
    },
)
