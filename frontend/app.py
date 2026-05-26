import streamlit as st
import requests
import folium
from streamlit_folium import st_folium
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Road Accident AI",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 AI-Powered Road Accident Prevention System")
st.markdown("Predict accident risk before it happens")

# Store prediction result
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

# Sidebar
st.sidebar.header("Road Conditions")

hour = st.sidebar.slider("Hour", 0, 23, 12)

month = st.sidebar.selectbox(
    "Month",
    range(1, 13)
)

day = st.sidebar.selectbox(
    "Day of Week",
    [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]
)

weather = st.sidebar.selectbox(
    "Weather",
    [
        "Clear",
        "Rain",
        "Fog",
        "Snow",
        "Wind"
    ]
)

speed = st.sidebar.slider(
    "Speed Limit",
    20,
    100,
    60
)

# Mappings
day_map = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}

weather_map = {
    "Clear": 0,
    "Rain": 1,
    "Fog": 2,
    "Snow": 3,
    "Wind": 4
}

# Predict Button
if st.sidebar.button("Predict Risk"):

    payload = {
        "hour": hour,
        "month": month,
        "day_of_week": day_map[day],
        "weather": weather_map[weather],
        "road_surface": 1,
        "light_condition": 1,
        "road_type": 0,
        "speed_limit": speed
    }

    try:

        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=payload
        )

        if response.status_code == 200:

            result = response.json()

            st.session_state.prediction_result = result

        else:
            st.error(f"Backend Error: {response.text}")

    except Exception as e:
        st.error(f"Frontend Error: {e}")

# Show prediction result permanently
if st.session_state.prediction_result is not None:

    result = st.session_state.prediction_result

    st.subheader("Prediction Result")

    col1, col2 = st.columns(2)

    col1.metric(
        "Risk Level",
        result["risk_level"]
    )

    col2.metric(
        "Confidence",
        result["confidence"]
    )

    st.warning(result["ai_warning"])

# Map
st.subheader("Accident Hotspot Map")

m = folium.Map(
    location=[11.0168, 76.9558],
    zoom_start=10
)

st_folium(m, width=1200, height=500)

# Charts
st.subheader("Accident Analysis")

try:

    df = pd.read_csv(
        "../data/processed/master_dataset.csv"
    )

    fig = px.histogram(
        df,
        x="Hour",
        title="Accidents by Hour"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

except:
    st.info(
        "Dataset visualization will appear after dataset setup."
    )