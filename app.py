# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import pydeck as pdk
# pyrefly: ignore [missing-import]
from sqlalchemy import create_engine
import sys
from pathlib import Path
from typing import cast

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent))
import config
from src.ml.pipeline import get_inference_pipeline

# Page Configuration
st.set_page_config(
    page_title="Traffic Accident Severity Analytics",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Glassmorphism Theme)
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 6px;
        background-color: rgba(255, 255, 255, 0.05);
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)
def load_data() -> pd.DataFrame:
    """Load analytical dataset from database DW / Mart tables."""
    db_url = config.get_db_url()
    engine = create_engine(db_url)
    is_sqlite = config.USE_SQLITE_FALLBACK or "sqlite" in db_url

    try:
        if is_sqlite:
            query = """
            SELECT 
                f.accident_index,
                f.accident_timestamp,
                f.accident_hour,
                d.full_date,
                d.year,
                d.month,
                d.month_name,
                d.day_name,
                d.day_of_week,
                d.is_weekend,
                l.latitude,
                l.longitude,
                l.police_force_name,
                r.road_class,
                r.road_surface,
                r.speed_limit,
                r.light_conditions,
                w.weather_condition,
                w.temperature_celsius,
                w.precipitation_mm,
                s.accident_severity_label,
                s.number_of_vehicles,
                s.number_of_casualties
            FROM fact_accidents f
            JOIN dim_date d ON f.date_key = d.date_key
            JOIN dim_location l ON f.location_key = l.location_key
            JOIN dim_road r ON f.road_key = r.road_key
            LEFT JOIN dim_weather w ON f.weather_key = w.weather_key
            JOIN dim_severity s ON f.severity_key = s.severity_key
            """
        else:
            query = "SELECT * FROM mart.vw_accident_analytics"
        
        df = pd.read_sql(query, engine)
        df["accident_timestamp"] = pd.to_datetime(df["accident_timestamp"])
        return df

    except Exception as e:
        st.error(f"Error connecting to database: {e}")
        # Return fallback dummy dataset if database not initialized yet
        return pd.DataFrame({
            "accident_index": [f"ACC_{i}" for i in range(100)],
            "accident_timestamp": pd.date_range("2022-01-01", periods=100, freq="12h"),
            "accident_hour": np.random.randint(0, 24, 100),
            "year": 2022,
            "month_name": "January",
            "day_name": "Monday",
            "latitude": np.random.uniform(51.45, 51.55, 100),
            "longitude": np.random.uniform(-0.2, 0.05, 100),
            "police_force_name": "Metropolitan Police",
            "road_surface": "Dry",
            "weather_condition": "Mainly clear",
            "accident_severity_label": np.random.choice(["Slight", "Serious", "Fatal"], 100, p=[0.8, 0.15, 0.05]),
            "number_of_vehicles": np.random.randint(1, 4, 100),
            "number_of_casualties": np.random.randint(1, 3, 100)
        })

# Header & Title
st.title("🚗 Traffic Accident Severity Analytics")
st.caption("Data Engineering & MLOps Pipeline Dashboard | Part 1 Operational Data Mart")

df = load_data()

# Sidebar Filters
st.sidebar.header("🔍 Analytical Filters")
severity_filter = st.sidebar.multiselect(
    "Accident Severity:",
    options=df["accident_severity_label"].unique(),
    default=df["accident_severity_label"].unique()
)

road_surface_filter = st.sidebar.multiselect(
    "Road Surface Condition:",
    options=df["road_surface"].unique(),
    default=df["road_surface"].unique()
)

# Filter Dataframe
df_filtered = cast(
    pd.DataFrame,
    df[
        (df["accident_severity_label"].isin(severity_filter)) &
        (df["road_surface"].isin(road_surface_filter))
    ]
)

# Top Key Performance Indicators (KPIs)
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Total Accidents", f"{len(df_filtered):,}")
kpi2.metric("Total Casualties", f"{int(df_filtered['number_of_casualties'].sum()):,}")
kpi3.metric("Total Vehicles Involved", f"{int(df_filtered['number_of_vehicles'].sum()):,}")
serious_fatal_pct = (df_filtered['accident_severity_label'].isin(['Serious', 'Fatal']).sum() / max(len(df_filtered), 1)) * 100
kpi4.metric("Serious/Fatal Ratio", f"{serious_fatal_pct:.1f}%")

st.markdown("---")

# 6 Dashboard Views via Tabs (Including MLOps Severity Prediction)
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 1. Severity Distribution",
    "⏰ 2. Temporal Patterns (Hour & Day)",
    "🌧️ 3. Weather & Road Impact",
    "🗺️ 4. Geospatial Hotspots",
    "🚘 5. Vehicle & Casualty Analysis",
    "🔮 6. Real-Time Severity Predictor (MLOps)"
])

# -----------------------------------------------------------------------------
# TAB 1: SEVERITY DISTRIBUTION
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("Accident Severity Breakdown")
    col1, col2 = st.columns(2)

    with col1:
        sev_counts = df_filtered["accident_severity_label"].value_counts().reset_index()
        sev_counts.columns = ["Severity", "Accident Count"]
        st.write("### Severity Volume")
        st.bar_chart(sev_counts.set_index("Severity"))

    with col2:
        st.write("### Severity Proportions")
        st.dataframe(sev_counts, width="stretch")

# -----------------------------------------------------------------------------
# TAB 2: ACCIDENTS BY HOUR AND WEEKDAY
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("Accident Frequency by Time of Day & Weekday")

    hourly_counts = df_filtered.groupby("accident_hour")["accident_index"].count().reset_index()
    hourly_counts.columns = ["Hour of Day (0-23)", "Accidents"]
    st.line_chart(hourly_counts.set_index("Hour of Day (0-23)"))

    st.write("### Heatmap Matrix: Day of Week vs Hour")
    heatmap_df = df_filtered.groupby(["day_name", "accident_hour"])["accident_index"].count().unstack().fillna(0)
    st.dataframe(heatmap_df, width="stretch")

# -----------------------------------------------------------------------------
# TAB 3: WEATHER & ROAD CONDITION IMPACT
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("Weather & Road Surface Condition Analytics")
    col1, col2 = st.columns(2)

    with col1:
        st.write("### Accidents by Road Surface Condition")
        road_counts = df_filtered["road_surface"].value_counts()
        st.bar_chart(road_counts)

    with col2:
        st.write("### Accidents by Historical Weather Condition")
        weather_counts = df_filtered["weather_condition"].value_counts()
        st.bar_chart(weather_counts)

# -----------------------------------------------------------------------------
# TAB 4: LOCATION HOTSPOT MAP
# -----------------------------------------------------------------------------
with tab4:
    st.subheader("Geospatial Accident Hotspot Map (London Region)")
    
    map_df = df_filtered.dropna(subset=["latitude", "longitude"])
    
    view_state = pdk.ViewState(
        latitude=map_df["latitude"].mean() if not map_df.empty else 51.5074,
        longitude=map_df["longitude"].mean() if not map_df.empty else -0.1278,
        zoom=11,
        pitch=45
    )

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position=["longitude", "latitude"],
        get_color="[255, 0, 0, 160]",
        get_radius=80,
        pickable=True
    )

    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"text": "Severity: {accident_severity_label}\nRoad: {road_surface}\nWeather: {weather_condition}"}
    )

    st.pydeck_chart(r)

# -----------------------------------------------------------------------------
# TAB 5: VEHICLE AND CASUALTY ANALYSIS
# -----------------------------------------------------------------------------
with tab5:
    st.subheader("Vehicle Count & Casualty Distribution Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Casualty Distribution")
        cas_df = df_filtered["number_of_casualties"].value_counts().reset_index()
        cas_df.columns = ["Casualties per Accident", "Frequency"]
        st.bar_chart(cas_df.set_index("Casualties per Accident"))

    with col2:
        st.write("### Vehicles Involved Distribution")
        veh_df = df_filtered["number_of_vehicles"].value_counts().reset_index()
        veh_df.columns = ["Vehicles per Accident", "Frequency"]
        st.bar_chart(veh_df.set_index("Vehicles per Accident"))

# -----------------------------------------------------------------------------
# TAB 6: REAL-TIME SEVERITY PREDICTOR (MLOPS PIPELINE)
# -----------------------------------------------------------------------------
with tab6:
    st.subheader("🔮 End-to-End Severity Prediction Pipeline")
    st.markdown(
        "Simulate live accident scenarios and generate real-time severity predictions "
        "using the production-registered LightGBM pipeline trained on the Part 1 analytical mart."
    )

    # Preset Quick Buttons
    st.write("#### ⚡ Quick Scenario Presets")
    preset_col1, preset_col2, preset_col3 = st.columns(3)
    
    # Session state initialization for form values
    if "preset_hour" not in st.session_state:
        st.session_state["preset_hour"] = 18
        st.session_state["preset_speed"] = 30
        st.session_state["preset_surface"] = "Dry"
        st.session_state["preset_weather"] = "Mainly clear"
        st.session_state["preset_light"] = "Daylight"
        st.session_state["preset_temp"] = 12.0
        st.session_state["preset_precip"] = 0.0
        st.session_state["preset_vehicles"] = 1

    if preset_col1.button("☀️ Standard Urban Commute"):
        st.session_state["preset_hour"] = 17
        st.session_state["preset_speed"] = 30
        st.session_state["preset_surface"] = "Dry"
        st.session_state["preset_weather"] = "Mainly clear"
        st.session_state["preset_light"] = "Daylight"
        st.session_state["preset_temp"] = 15.0
        st.session_state["preset_precip"] = 0.0
        st.session_state["preset_vehicles"] = 2

    if preset_col2.button("🌧️ Wet Night Arterial"):
        st.session_state["preset_hour"] = 22
        st.session_state["preset_speed"] = 50
        st.session_state["preset_surface"] = "Wet or damp"
        st.session_state["preset_weather"] = "Light drizzle"
        st.session_state["preset_light"] = "Darkness - lights lit"
        st.session_state["preset_temp"] = 5.0
        st.session_state["preset_precip"] = 1.5
        st.session_state["preset_vehicles"] = 2

    if preset_col3.button("❄️ Hazardous Freezing Darkness"):
        st.session_state["preset_hour"] = 2
        st.session_state["preset_speed"] = 60
        st.session_state["preset_surface"] = "Frost or ice"
        st.session_state["preset_weather"] = "Light drizzle"
        st.session_state["preset_light"] = "Darkness - no lighting"
        st.session_state["preset_temp"] = -1.5
        st.session_state["preset_precip"] = 0.8
        st.session_state["preset_vehicles"] = 3

    st.markdown("---")

    with st.form("severity_prediction_form"):
        form_col1, form_col2 = st.columns(2)

        with form_col1:
            st.markdown("##### 🚗 Road & Environmental Factors")
            speed_limit = st.selectbox(
                "Speed Limit (mph)",
                options=[20, 30, 40, 50, 60, 70],
                index=[20, 30, 40, 50, 60, 70].index(st.session_state["preset_speed"])
            )
            road_surface = st.selectbox(
                "Road Surface Condition",
                options=["Dry", "Wet or damp", "Frost or ice", "Snow", "Flood over 3cm deep"],
                index=["Dry", "Wet or damp", "Frost or ice", "Snow", "Flood over 3cm deep"].index(st.session_state["preset_surface"])
            )
            light_conditions = st.selectbox(
                "Lighting Condition",
                options=[
                    "Daylight",
                    "Darkness - lights lit",
                    "Darkness - no lighting",
                    "Darkness - lights unlit",
                    "Darkness - lighting unknown"
                ],
                index=["Daylight", "Darkness - lights lit", "Darkness - no lighting", "Darkness - lights unlit", "Darkness - lighting unknown"].index(st.session_state["preset_light"])
            )
            weather_condition = st.selectbox(
                "Weather Condition",
                options=["Mainly clear", "Clear sky", "Overcast", "Light drizzle", "Partly cloudy", "Slight rain"],
                index=["Mainly clear", "Clear sky", "Overcast", "Light drizzle", "Partly cloudy", "Slight rain"].index(st.session_state["preset_weather"])
            )
            temp_celsius = st.number_input(
                "Air Temperature (°C)",
                value=float(st.session_state["preset_temp"]),
                min_value=-15.0,
                max_value=40.0,
                step=0.5
            )
            precip_mm = st.number_input(
                "Precipitation (mm/h)",
                value=float(st.session_state["preset_precip"]),
                min_value=0.0,
                max_value=50.0,
                step=0.2
            )

        with form_col2:
            st.markdown("##### ⏰ Time & Collision Context")
            accident_hour = st.slider(
                "Hour of Incident (0 = Midnight, 23 = 11 PM)",
                min_value=0,
                max_value=23,
                value=int(st.session_state["preset_hour"])
            )
            day_of_week = st.selectbox(
                "Day of Week",
                options=[
                    (1, "Monday"),
                    (2, "Tuesday"),
                    (3, "Wednesday"),
                    (4, "Thursday"),
                    (5, "Friday"),
                    (6, "Saturday"),
                    (7, "Sunday")
                ],
                format_func=lambda x: x[1],
                index=4
            )
            month = st.selectbox(
                "Month of Year",
                options=[
                    (1, "January"), (2, "February"), (3, "March"), (4, "April"),
                    (5, "May"), (6, "June"), (7, "July"), (8, "August"),
                    (9, "September"), (10, "October"), (11, "November"), (12, "December")
                ],
                format_func=lambda x: x[1],
                index=10
            )
            number_of_vehicles = st.number_input(
                "Number of Vehicles Involved",
                value=int(st.session_state["preset_vehicles"]),
                min_value=1,
                max_value=10,
                step=1
            )
            st.markdown("##### 📍 Location Coordinates")
            coord_preset = st.radio(
                "Location Context",
                options=["Central London (Congested/Low Speed)", "Outer London Arterial (Higher Speed)"],
                index=0
            )
            if "Central" in coord_preset:
                lat, lon = 51.5074, -0.1278
            else:
                lat, lon = 51.5850, -0.3200

        submit_pred = st.form_submit_button("⚡ Predict Accident Severity", type="primary", use_container_width=True)

    if submit_pred:
        try:
            pipeline = get_inference_pipeline()
            raw_payload = {
                "accident_hour": accident_hour,
                "day_of_week": day_of_week[0],
                "month": month[0],
                "is_weekend": 1 if day_of_week[0] >= 6 else 0,
                "latitude": lat,
                "longitude": lon,
                "speed_limit": speed_limit,
                "road_class": "A",
                "road_surface": road_surface,
                "light_conditions": light_conditions,
                "weather_condition": weather_condition,
                "temperature_celsius": temp_celsius,
                "precipitation_mm": precip_mm,
                "number_of_vehicles": number_of_vehicles
            }

            result = pipeline.predict(raw_payload)

            st.markdown("### 🎯 Inference Results")
            res_col1, res_col2, res_col3 = st.columns(3)

            sev = result["predicted_severity"]
            risk = result["risk_level"]
            conf = result["confidence"]
            high_sev_score = result["high_severity_combined_probability"]

            # Visual color schemes
            if sev == "Fatal":
                color_bg = "#B71C1C"
                badge = "🔴 FATAL"
            elif sev == "Serious":
                color_bg = "#E65100"
                badge = "🟠 SERIOUS"
            else:
                color_bg = "#1B5E20"
                badge = "🟢 SLIGHT"

            with res_col1:
                st.markdown(
                    f"""
                    <div style="background-color: {color_bg}; padding: 16px; border-radius: 8px; text-align: center; color: white;">
                        <h4 style="margin:0;">PREDICTED SEVERITY</h4>
                        <h2 style="margin:8px 0;">{badge}</h2>
                        <p style="margin:0;">Confidence: {conf*100:.1f}%</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with res_col2:
                risk_color = "#D32F2F" if risk == "High" else ("#F57C00" if risk == "Medium" else "#388E3C")
                st.markdown(
                    f"""
                    <div style="background-color: rgba(255,255,255,0.05); border: 2px solid {risk_color}; padding: 16px; border-radius: 8px; text-align: center;">
                        <h4 style="margin:0; color: #BBB;">EMERGENCY PRIORITY</h4>
                        <h2 style="margin:8px 0; color: {risk_color};">{risk.upper()} ALERT</h2>
                        <p style="margin:0; color: #DDD;">High-Severity Risk: {high_sev_score*100:.1f}%</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with res_col3:
                st.markdown(
                    """
                    <div style="background: rgba(255,255,255,0.05); padding: 16px; border-radius: 8px; text-align: center; border: 1px solid rgba(255,255,255,0.1);">
                        <h4 style="margin:0; color: #BBB;">PRODUCTION MODEL</h4>
                        <h3 style="margin:8px 0; color: #4FC3F7;">LightGBM Classifier</h3>
                        <p style="margin:0; color: #9E9E9E;">Registry v1 (Balanced Weights)</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.write("#### 📊 Predicted Class Probability Distribution")
            prob_df = pd.DataFrame({
                "Severity Class": ["Slight", "Serious", "Fatal"],
                "Probability (%)": [
                    result["probabilities"]["Slight"] * 100,
                    result["probabilities"]["Serious"] * 100,
                    result["probabilities"]["Fatal"] * 100
                ]
            })
            st.bar_chart(prob_df.set_index("Severity Class"))

            # Risk factor explanations
            st.markdown("##### 🔎 Identified Environmental Hazards")
            hazards = []
            if speed_limit >= 50:
                hazards.append(f"⚠️ **High Speed Corridor**: Posted speed limit of {speed_limit} mph exponentially increases collision kinetic energy.")
            if "dark" in light_conditions.lower():
                hazards.append("⚠️ **Reduced Visibility**: Dark conditions compromise driver reaction distance.")
            if road_surface != "Dry":
                hazards.append(f"⚠️ **Compromised Friction**: {road_surface} surface significantly extends vehicle braking distance.")
            if temp_celsius <= 0.0:
                hazards.append(f"⚠️ **Freezing Hazard**: Ambient temperature ({temp_celsius}°C) introduces black ice risk.")
            if number_of_vehicles > 1:
                hazards.append(f"⚠️ **Multi-Vehicle Dynamics**: Collision involves {number_of_vehicles} vehicles, increasing casualty likelihood.")
            
            if hazards:
                for h in hazards:
                    st.markdown(h)
            else:
                st.info("✅ Baseline conditions: Urban daylight dry road environment.")

        except Exception as e:
            st.error(f"Inference error: {e}")

