import streamlit as st
import pandas as pd
import numpy as np
import shap
import plotly.express as px

from sklearn.ensemble import RandomForestRegressor


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Delhi NCR Air Quality Prediction System",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #f4f7fb;
}

/* Main content */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* All normal text */
p, span, div, label {
    color: #000000;
}

/* Headings */
h1 {
    color: #17365d !important;
}

h2 {
    color: #17365d !important;
}

h3 {
    color: #17365d !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #e8f1f9;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div {
    color: #000000 !important;
}

/* Metric styling */
[data-testid="stMetric"] {
    background-color: #ffffff;
    border: 1px solid #d7e2ef;
    border-radius: 12px;
    padding: 15px;
    box-shadow: 0px 2px 8px rgba(0, 0, 0, 0.05);
}

[data-testid="stMetricLabel"] {
    color: #475569 !important;
}

[data-testid="stMetricValue"] {
    color: #000000 !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    background-color: #ffffff;
}

/* Selectbox */
div[data-baseweb="select"] {
    background-color: #ffffff;
}

/* Alert boxes */
.stAlert p {
    color: #000000 !important;
}

/* Caption */
.stCaption {
    color: #475569 !important;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# TITLE
# =========================================================

st.title("🌫️ Delhi NCR Air Quality Prediction System")

st.write(
    "AI-based AQI forecasting using weather, pollution and satellite fire hotspot data"
)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    data = pd.read_csv(
        "delhi_final_ml_dataset.csv"
    )

    data["timestamp"] = pd.to_datetime(
        data["timestamp"],
        errors="coerce"
    )

    data = data.dropna(
        subset=["timestamp"]
    )

    data = data.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    return data


data = load_data()


# =========================================================
# FEATURES
# =========================================================

features = [

    # Weather
    "temp_c",
    "humidity",
    "pressure_mb",
    "windspeed_kph",

    # Pollution
    "aqi_index",
    "pm2_5",
    "pm10",
    "co",
    "no2",

    # Fire activity
    "fire_count",
    "fire_frp_sum",
    "fire_brightness_sum",

    # Regional fire activity
    "punjab_fire_count",
    "haryana_fire_count",
    "uttar_pradesh_fire_count",
    "uttarakhand_fire_count",
    "delhi_ncr_fire_count"
]


# =========================================================
# TARGETS
# =========================================================

targets = [
    "aqi_1h",
    "aqi_24h",
    "aqi_72h"
]


# =========================================================
# CHECK FEATURES
# =========================================================

missing_features = [
    column
    for column in features
    if column not in data.columns
]

if missing_features:

    st.error(
        "Required features are missing from the dataset:"
    )

    st.write(missing_features)

    st.stop()


# =========================================================
# CHECK TARGETS
# =========================================================

missing_targets = [
    column
    for column in targets
    if column not in data.columns
]

if missing_targets:

    st.error(
        "Prediction target columns are missing:"
    )

    st.write(missing_targets)

    st.stop()


# =========================================================
# TRAINING DATA
# =========================================================

training_data = data.dropna(
    subset=features + targets
).reset_index(drop=True)


if len(training_data) < 20:

    st.error(
        "Not enough complete data available for model training."
    )

    st.stop()


# =========================================================
# TRAIN MODELS
# =========================================================

@st.cache_resource
def train_models(_data):

    models = {}

    X = _data[features]

    # -----------------------------------------------------
    # 1 HOUR
    # -----------------------------------------------------

    model_1h = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model_1h.fit(
        X,
        _data["aqi_1h"]
    )

    models["1 Hour"] = model_1h


    # -----------------------------------------------------
    # 24 HOURS
    # -----------------------------------------------------

    model_24h = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model_24h.fit(
        X,
        _data["aqi_24h"]
    )

    models["24 Hours"] = model_24h


    # -----------------------------------------------------
    # 72 HOURS
    # -----------------------------------------------------

    model_72h = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model_72h.fit(
        X,
        _data["aqi_72h"]
    )

    models["72 Hours"] = model_72h

    return models


with st.spinner("Training AQI prediction models..."):

    models = train_models(
        training_data
    )


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚙️ Prediction Settings")

st.sidebar.write(
    "Select location and forecast horizon."
)


# =========================================================
# LOCATION
# =========================================================

if "location" in data.columns:

    locations = sorted(
        data["location"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

else:

    locations = ["Delhi NCR"]


selected_location = st.sidebar.selectbox(
    "📍 Location",
    locations
)


# =========================================================
# FORECAST
# =========================================================

forecast = st.sidebar.selectbox(
    "⏱️ Forecast Horizon",
    [
        "1 Hour",
        "24 Hours",
        "72 Hours"
    ]
)


# =========================================================
# FILTER LOCATION
# =========================================================

if "location" in data.columns:

    location_data = data[
        data["location"].astype(str)
        == selected_location
    ].copy()

else:

    location_data = data.copy()


if len(location_data) == 0:

    st.error(
        "No data available for the selected location."
    )

    st.stop()


# =========================================================
# CURRENT ROW
# =========================================================

complete_location_data = location_data.dropna(
    subset=features
).copy()


if len(complete_location_data) == 0:

    st.error(
        "No complete feature data is available for this location."
    )

    st.stop()


current_row = (
    complete_location_data
    .sort_values("timestamp")
    .iloc[-1]
)


# =========================================================
# SELECT MODEL
# =========================================================

selected_model = models[forecast]


# =========================================================
# CURRENT INPUT
# =========================================================

X_current = pd.DataFrame(
    [current_row[features].values],
    columns=features
)


# =========================================================
# PREDICTION
# =========================================================

prediction = selected_model.predict(
    X_current
)[0]

prediction = max(
    0,
    round(float(prediction), 2)
)


# =========================================================
# AQI CATEGORY
# =========================================================

def get_aqi_category(aqi):

    if aqi <= 50:
        return "Good"

    elif aqi <= 100:
        return "Satisfactory"

    elif aqi <= 200:
        return "Moderate"

    elif aqi <= 300:
        return "Poor"

    elif aqi <= 400:
        return "Very Poor"

    else:
        return "Severe"


category = get_aqi_category(
    prediction
)


# =========================================================
# LOCATION / TIME
# =========================================================

st.subheader(
    f"📍 {selected_location}"
)

timestamp_text = current_row["timestamp"].strftime(
    "%d %b %Y, %I:%M %p"
)

st.caption(
    f"Latest available data: {timestamp_text}"
)


# =========================================================
# MAIN PREDICTION
# =========================================================

st.subheader(
    f"🌫️ Predicted AQI — {forecast}"
)


aqi_col1, aqi_col2, aqi_col3 = st.columns(3)


with aqi_col1:

    st.metric(
        "Predicted AQI",
        prediction
    )


with aqi_col2:

    st.metric(
        "AQI Category",
        category
    )


with aqi_col3:

    st.metric(
        "Forecast Horizon",
        forecast
    )


st.divider()


# =========================================================
# CURRENT CONDITIONS
# =========================================================

st.subheader(
    "📊 Current Conditions"
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "🌡️ Temperature",
        f"{current_row['temp_c']:.1f} °C"
    )


with col2:

    st.metric(
        "🫁 PM2.5",
        f"{current_row['pm2_5']:.1f} µg/m³"
    )


with col3:

    st.metric(
        "💨 PM10",
        f"{current_row['pm10']:.1f} µg/m³"
    )


with col4:

    st.metric(
        "📊 Current AQI",
        f"{current_row['aqi_index']:.0f}"
    )


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "💧 Humidity",
        f"{current_row['humidity']:.1f}%"
    )


with col2:

    st.metric(
        "🌬️ Wind Speed",
        f"{current_row['windspeed_kph']:.1f} km/h"
    )


with col3:

    st.metric(
        "🔥 Fire Hotspots",
        f"{int(current_row['fire_count'])}"
    )


with col4:

    st.metric(
        "🔥 Fire FRP",
        f"{current_row['fire_frp_sum']:.1f}"
    )


# =========================================================
# WIND DIRECTION
# =========================================================

st.subheader(
    "🧭 Wind Information"
)


wind_direction = float(
    current_row["Wind (From) Direction (Degrees)"]
)


def get_wind_direction(degrees):

    directions = [
        "North",
        "North-East",
        "East",
        "South-East",
        "South",
        "South-West",
        "West",
        "North-West"
    ]

    index = int(
        ((degrees + 22.5) % 360) // 45
    )

    return directions[index]


wind_direction_name = get_wind_direction(
    wind_direction
)


wind_col1, wind_col2 = st.columns(2)


with wind_col1:

    st.metric(
        "Wind Speed",
        f"{current_row['windspeed_kph']:.1f} km/h"
    )


with wind_col2:

    st.metric(
        "Wind From Direction",
        f"{wind_direction:.0f}° ({wind_direction_name})"
    )

# =========================================================
# SHAP EXPLANATION
# =========================================================

st.subheader(
    "🔍 Why is the AQI predicted to be this level?"
)


@st.cache_resource
def create_explainer(_model):

    return shap.TreeExplainer(
        _model
    )


explainer = create_explainer(
    selected_model
)


shap_values = explainer.shap_values(
    X_current
)


if isinstance(shap_values, list):

    values = shap_values[0]

else:

    values = np.asarray(
        shap_values
    )

    if values.ndim > 1:

        values = values[0]


explanation = pd.DataFrame({

    "Feature": features,

    "Impact": values,

    "Value": X_current.iloc[0].values

})


explanation["Importance"] = (
    explanation["Impact"].abs()
)


explanation = explanation.sort_values(
    "Importance",
    ascending=False
).reset_index(drop=True)


top_reasons = explanation.head(7)


# =========================================================
# SHAP REASONS
# =========================================================

for _, row in top_reasons.head(5).iterrows():

    feature = row["Feature"]

    impact = row["Impact"]

    value = row["Value"]


    if impact > 0:

        st.error(
            f"🔴 {feature}: {value:.2f} — Increasing AQI"
        )

    else:

        st.success(
            f"🟢 {feature}: {value:.2f} — Decreasing AQI"
        )


# =========================================================
# SHAP CHART
# =========================================================

st.subheader(
    "📊 Feature Importance (SHAP)"
)


shap_chart = top_reasons.copy()


shap_chart = shap_chart.sort_values(
    "Importance",
    ascending=True
)


shap_chart["Direction"] = shap_chart[
    "Impact"
].apply(
    lambda x:
    "Increases AQI"
    if x > 0
    else "Decreases AQI"
)


fig_shap = px.bar(
    shap_chart,
    x="Importance",
    y="Feature",
    orientation="h",
    color="Direction",
    text="Importance",
    title=f"Top Factors Affecting {forecast} Prediction"
)


fig_shap.update_traces(
    texttemplate="%{text:.3f}",
    textposition="outside"
)


fig_shap.update_layout(
    height=450,
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(
        color="black"
    )
)


st.plotly_chart(
    fig_shap,
    use_container_width=True
)


# =========================================================
# HISTORICAL AQI TREND
# =========================================================

st.subheader(
    "📈 Historical AQI Trend"
)


trend_data = location_data[
    [
        "timestamp",
        "aqi_index"
    ]
].dropna().tail(72)


if len(trend_data) > 0:

    fig_trend = px.line(
        trend_data,
        x="timestamp",
        y="aqi_index",
        markers=True,
        title="Recent AQI Trend"
    )

    fig_trend.update_layout(
        height=420,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(
            color="black"
        ),
        xaxis_title="Time",
        yaxis_title="AQI"
    )

    st.plotly_chart(
        fig_trend,
        use_container_width=True
    )


# =========================================================
# REGIONAL FIRE ACTIVITY
# =========================================================

st.subheader(
    "🔥 Regional Fire Activity"
)


region_columns = [

    "punjab_fire_count",
    "haryana_fire_count",
    "uttar_pradesh_fire_count",
    "uttarakhand_fire_count",
    "delhi_ncr_fire_count"

]


region_names = {

    "punjab_fire_count":
    "Punjab",

    "haryana_fire_count":
    "Haryana",

    "uttar_pradesh_fire_count":
    "Uttar Pradesh",

    "uttarakhand_fire_count":
    "Uttarakhand",

    "delhi_ncr_fire_count":
    "Delhi NCR"

}


region_values = []


for column in region_columns:

    region_values.append({

        "Region":
        region_names[column],

        "Fire Count":
        float(
            current_row.get(
                column,
                0
            )
        )

    })


region_data = pd.DataFrame(
    region_values
)


fig_fire = px.bar(
    region_data,
    x="Region",
    y="Fire Count",
    text="Fire Count",
    title="Current Regional Fire Hotspots"
)


fig_fire.update_traces(
    textposition="outside"
)


fig_fire.update_layout(
    height=420,
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(
        color="black"
    )
)


st.plotly_chart(
    fig_fire,
    use_container_width=True
)


# =========================================================
# FIRE SOURCE SUMMARY
# =========================================================

st.subheader(
    "🔥 Fire Source Summary"
)


fire_summary = pd.DataFrame({

    "Region": [

        "Punjab",
        "Haryana",
        "Uttar Pradesh",
        "Uttarakhand",
        "Delhi NCR"

    ],

    "Fire Count": [

        current_row.get(
            "punjab_fire_count",
            0
        ),

        current_row.get(
            "haryana_fire_count",
            0
        ),

        current_row.get(
            "uttar_pradesh_fire_count",
            0
        ),

        current_row.get(
            "uttarakhand_fire_count",
            0
        ),

        current_row.get(
            "delhi_ncr_fire_count",
            0
        )

    ]

})


st.dataframe(
    fire_summary,
    use_container_width=True,
    hide_index=True
)


# =========================================================
# WEATHER & POLLUTION INSIGHTS
# =========================================================

st.subheader(
    "🌡️ Weather & Pollution Insights"
)


temperature = float(
    current_row["temp_c"]
)

humidity = float(
    current_row["humidity"]
)

wind_speed = float(
    current_row["windspeed_kph"]
)

pm25 = float(
    current_row["pm2_5"]
)


insight_col1, insight_col2, insight_col3 = st.columns(3)


with insight_col1:

    st.markdown(
        "### 🌡️ Temperature"
    )

    st.metric(
        "Temperature",
        f"{temperature:.1f} °C"
    )

    if temperature < 15:

        st.write(
            "Cooler conditions may reduce atmospheric mixing "
            "and contribute to pollutant accumulation."
        )

    elif temperature < 20:

        st.write(
            "Moderately cool conditions may support pollutant "
            "persistence near the surface."
        )

    else:

        st.write(
            "Higher temperature generally supports stronger "
            "atmospheric mixing compared with colder conditions."
        )


with insight_col2:

    st.markdown(
        "### 🌬️ Wind Conditions"
    )

    st.metric(
        "Wind Speed",
        f"{wind_speed:.1f} km/h"
    )

    if wind_speed < 5:

        st.write(
            "Low wind speed can reduce pollutant dispersion "
            "and allow pollutants to accumulate."
        )

    elif wind_speed < 10:

        st.write(
            "Moderate wind provides some pollutant dispersion."
        )

    else:

        st.write(
            "Higher wind speed generally supports stronger "
            "pollutant dispersion."
        )


with insight_col3:

    st.markdown(
        "### 💧 Humidity"
    )

    st.metric(
        "Humidity",
        f"{humidity:.1f}%"
    )

    if humidity >= 70:

        st.write(
            "High humidity can be associated with conditions "
            "that favor particulate accumulation and growth."
        )

    else:

        st.write(
            "Humidity is not currently in the high range."
        )


# =========================================================
# NEXT AREA RISK ANALYSIS
# =========================================================

st.subheader(
    "🧭 Next Area Risk Analysis"
)


st.write(
    "Wind direction is used to identify the approximate downwind "
    "area where pollutants may move."
)


# ---------------------------------------------------------
# WIND FROM → WIND TO
# ---------------------------------------------------------

wind_from = float(
    current_row["Wind (From) Direction (Degrees)"]
)

wind_to = (
    wind_from + 180
) % 360


wind_to_name = get_wind_direction(
    wind_to
)


risk_col1, risk_col2, risk_col3 = st.columns(3)


with risk_col1:

    st.metric(
        "Wind From",
        f"{wind_from:.0f}°"
    )


with risk_col2:

    st.metric(
        "Wind Toward",
        f"{wind_to:.0f}°"
    )


with risk_col3:

    st.metric(
        "Downwind Direction",
        wind_to_name
    )


# ---------------------------------------------------------
# AREA RISK
# ---------------------------------------------------------

def get_downwind_areas(direction):

    direction = direction % 360

    if direction >= 337.5 or direction < 22.5:

        return [
            "Central Delhi",
            "North Delhi",
            "North-East Delhi"
        ]

    elif direction < 67.5:

        return [
            "East Delhi",
            "North-East Delhi",
            "Noida"
        ]

    elif direction < 112.5:

        return [
            "East Delhi",
            "Noida",
            "Ghaziabad"
        ]

    elif direction < 157.5:

        return [
            "South-East Delhi",
            "Noida",
            "Faridabad"
        ]

    elif direction < 202.5:

        return [
            "South Delhi",
            "Faridabad",
            "South-East Delhi"
        ]

    elif direction < 247.5:

        return [
            "West Delhi",
            "Dwarka",
            "Gurugram"
        ]

    elif direction < 292.5:

        return [
            "West Delhi",
            "North-West Delhi",
            "Gurugram"
        ]

    else:

        return [
            "North Delhi",
            "North-West Delhi",
            "Ghaziabad"
        ]


downwind_areas = get_downwind_areas(
    wind_to
)


st.write(
    "### ⚠️ Potential Downwind Areas"
)


for area in downwind_areas:

    st.warning(
        f"Potential pollution impact area: {area}"
    )


st.caption(
    "This is a directional screening analysis based on wind direction only. "
    "Actual pollution transport also depends on wind speed, atmospheric stability, "
    "emissions, terrain and other meteorological conditions."
)


# =========================================================
# AI SUMMARY
# =========================================================

st.subheader(
    "🤖 AI Explanation"
)


top_feature = top_reasons.iloc[0]["Feature"]

top_impact = top_reasons.iloc[0]["Impact"]


if top_impact > 0:

    ai_summary = (
        f"{top_feature} is currently the strongest model factor "
        f"pushing the {forecast} AQI prediction upward."
    )

else:

    ai_summary = (
        f"{top_feature} is currently the strongest model factor "
        f"reducing the {forecast} AQI prediction."
    )


fire_total = sum(
    [
        float(
            current_row.get(
                column,
                0
            )
        )
        for column in region_columns
    ]
)


st.info(
    f"""
**Model Interpretation**

{ai_summary}

Current PM2.5: **{pm25:.1f} µg/m³**

Regional fire hotspots: **{int(fire_total)}**

Temperature, humidity, wind speed and pollution variables
are included as model inputs.

SHAP explains how the available features contribute to
the selected AQI prediction.

Downwind screening indicates the potential direction of
pollution movement based on the current wind direction.
"""
)


# =========================================================
# MODEL INFORMATION
# =========================================================

st.subheader(
    "🤖 Model Information"
)


model_col1, model_col2, model_col3 = st.columns(3)


with model_col1:

    st.metric(
        "Algorithm",
        "Random Forest"
    )


with model_col2:

    st.metric(
        "Forecasts",
        "1h / 24h / 72h"
    )


with model_col3:

    st.metric(
        "Explainability",
        "SHAP"
    )


# =========================================================
# DATA INFORMATION
# =========================================================

st.subheader(
    "📁 Dataset Information"
)


data_col1, data_col2, data_col3 = st.columns(3)


with data_col1:

    st.metric(
        "Total Records",
        f"{len(data):,}"
    )


with data_col2:

    st.metric(
        "Training Records",
        f"{len(training_data):,}"
    )


with data_col3:

    st.metric(
        "Features Used",
        len(features)
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Delhi NCR Air Quality Prediction System | "
    "Random Forest + SHAP | "
    "Weather + Pollution + Satellite Fire Hotspots"
)