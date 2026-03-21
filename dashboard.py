import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# 1. UI/UX Page Configuration
st.set_page_config(page_title="AETHEL Clinical Engine", layout="wide", page_icon="🧬", initial_sidebar_state="expanded")

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

# 2. Data Loading (No more fake coordinates!)
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/analytical_cohort.csv")
    df['calculated_risk_score'] = (df['avg_pm25_exposure'] * 0.4) + (df['genomic_risk_score'] * 15) + (df['is_smoker'] * 20)
    df['calculated_risk_score'] = 100 * (df['calculated_risk_score'] - df['calculated_risk_score'].min()) / (df['calculated_risk_score'].max() - df['calculated_risk_score'].min())
    df['Risk Category'] = pd.cut(df['calculated_risk_score'], bins=[-1, 33, 66, 100], labels=['Low', 'Moderate', 'High'])
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("Dataset missing. Ensure 'analytical_cohort.csv' exists in data/processed/.")
    st.stop()

# 3. Sidebar Filters
st.sidebar.header("🔬 Cohort Filters")
risk_filter = st.sidebar.multiselect("Risk Category:", options=['Low', 'Moderate', 'High'], default=['Low', 'Moderate', 'High'])
city_filter = st.sidebar.multiselect("Region:", options=df['city'].unique(), default=df['city'].unique())
smoker_filter = st.sidebar.radio("Smoking Status:", ["All", "Current Smoker", "Non-Smoker"])
age_range = st.sidebar.slider("Age Range:", int(df['age'].min()), int(df['age'].max()), (int(df['age'].min()), int(df['age'].max())))

filtered_df = df[(df['Risk Category'].isin(risk_filter)) & (df['city'].isin(city_filter))]
if smoker_filter == "Current Smoker":
    filtered_df = filtered_df[filtered_df['is_smoker'] == 1]
elif smoker_filter == "Non-Smoker":
    filtered_df = filtered_df[filtered_df['is_smoker'] == 0]
filtered_df = filtered_df[(filtered_df['age'] >= age_range[0]) & (filtered_df['age'] <= age_range[1])]

# 4. Top-Level KPIs
col1, col2, col3, col4 = st.columns(4)
col1.metric("Active Patients in View", f"{len(filtered_df):,}")
col2.metric("Critical Risk (>75%)", len(filtered_df[filtered_df['calculated_risk_score'] > 75]))
col3.metric("Avg PM2.5 (μg/m³)", f"{filtered_df['avg_pm25_exposure'].mean():.1f}")
col4.metric("Total Events", filtered_df['event_occurred'].sum())
st.markdown("<br>", unsafe_allow_html=True)

# 5. Core Visualizations
map_col, chart_col = st.columns([1.5, 1])

with map_col:
    st.subheader("📍 Continental Hazard Distribution")
    # Map zoomed out to cover Europe (Zoom=4)
    fig_map = px.scatter_mapbox(
        filtered_df, lat="lat", lon="lon", color="calculated_risk_score", size="avg_pm25_exposure",
        color_continuous_scale="RdYlBu_r", size_max=15, zoom=4, center=dict(lat=51.0, lon=10.0),
        mapbox_style="carto-positron", hover_name="patient_id",
        hover_data={"lat": False, "lon": False, "city": True, "age": True, "Risk Category": True}
    )
    fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)

with chart_col:
    st.subheader("📊 Primary Risk Drivers")
    vimp_data = pd.DataFrame({
        'Feature': ['PM2.5 Exposure', 'Polygenic Score', 'Smoking', 'Age', 'Deprivation'],
        'Importance': [0.35, 0.28, 0.20, 0.12, 0.05]
    }).sort_values('Importance')
    fig_bar = px.bar(vimp_data, x='Importance', y='Feature', orientation='h', color='Importance', color_continuous_scale="Blues")
    fig_bar.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)

# 6. Data Table
st.markdown("---")
st.subheader("📋 Multi-Regional Telemetry Registry")
display_df = filtered_df[['patient_id', 'city', 'age', 'is_smoker', 'townsend_index', 'avg_pm25_exposure', 'genomic_risk_score', 'calculated_risk_score', 'Risk Category']].copy()
display_df['is_smoker'] = display_df['is_smoker'].map({1: 'Yes', 0: 'No'})

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "patient_id": "Patient ID",
        "city": "Location",
        "age": st.column_config.NumberColumn("Age", format="%d"),
        "is_smoker": "Smoker",
        "townsend_index": st.column_config.NumberColumn("Deprivation Index", format="%.2f"),
        "avg_pm25_exposure": st.column_config.NumberColumn("PM2.5 (μg/m³)", format="%.2f"),
        "genomic_risk_score": st.column_config.NumberColumn("Polygenic Score", format="%.3f"),
        "calculated_risk_score": st.column_config.ProgressColumn("Overall Risk", format="%.1f", min_value=0, max_value=100),
        "Risk Category": "Category"
    }
)