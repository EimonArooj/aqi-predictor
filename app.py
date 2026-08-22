import streamlit as st
import pandas as pd
import os
import joblib
from dotenv import load_dotenv
import hopsworks
from datetime import datetime
import shap
import matplotlib.pyplot as plt
load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")

st.set_page_config(page_title="Rawalpindi AQI Predictor", layout="centered")
st.title("🌫️ Rawalpindi AQI Predictor")
st.write("3-day Air Quality Index forecast, powered by a serverless ML pipeline.")

# ---- Connect to Hopsworks ----
@st.cache_resource
def connect_hopsworks():
    project = hopsworks.login(
        project=HOPSWORKS_PROJECT_NAME,
        api_key_value=HOPSWORKS_API_KEY
    )
    return project

project = connect_hopsworks()
st.success(f"Connected to Hopsworks project: {project.name}")

# ---- Load the trained model from Hopsworks Model Registry ----
@st.cache_resource
def load_model():
    mr = project.get_model_registry()
    model = mr.get_model("aqi_rf_model", version=1)
    model_dir = model.download()
    loaded_model = joblib.load(os.path.join(model_dir, "rf_model.pkl"))
    return loaded_model

with st.spinner("Loading model..."):
    rf_model = load_model()

st.success("Model loaded successfully!")

# ---- Load latest feature data from Hopsworks ----
@st.cache_data(ttl=300)  # refresh every 5 minutes
def load_latest_data():
    fs = project.get_feature_store()
    aqi_fg = fs.get_feature_group(name="aqi_features", version=4)
    df = aqi_fg.read()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

with st.spinner("Fetching latest data..."):
    df = load_latest_data()

latest_row = df.iloc[-1]
st.subheader("📍 Current Conditions (Rawalpindi)")
st.metric("Current AQI", int(latest_row["aqi"]))
st.write(f"Last updated: {latest_row['timestamp']}")

# ---- AQI Trend Chart (last 7 days) ----
st.subheader("📈 Recent AQI Trend")

recent_data = df.tail(24 * 7)  # last 7 days (hourly data)
chart_data = recent_data[["timestamp", "aqi"]].set_index("timestamp")
st.line_chart(chart_data)

# ---- Make prediction using latest row ----
feature_columns = ["hour", "day", "month", "weekday", "previous_aqi"]
X_latest = latest_row[feature_columns].values.reshape(1, -1)

prediction = rf_model.predict(X_latest)[0]

st.subheader("🔮 3-Day AQI Forecast")
st.metric("Predicted AQI (in 3 days)", round(prediction))

# ---- Hazardous AQI Alert ----
def get_aqi_category(aqi_value):
    if aqi_value <= 50:
        return "Good", "🟢"
    elif aqi_value <= 100:
        return "Moderate", "🟡"
    elif aqi_value <= 150:
        return "Unhealthy for Sensitive Groups", "🟠"
    elif aqi_value <= 200:
        return "Unhealthy", "🔴"
    elif aqi_value <= 300:
        return "Very Unhealthy", "🟣"
    else:
        return "Hazardous", "⚫"

category, emoji = get_aqi_category(prediction)

if prediction > 150:
    st.error(f"{emoji} **Warning: Predicted AQI is {category}** ({round(prediction)}). Consider limiting outdoor activity in 3 days.")
elif prediction > 100:
    st.warning(f"{emoji} **Predicted AQI: {category}** ({round(prediction)}). Sensitive groups should take precautions.")
else:
    st.success(f"{emoji} **Predicted AQI: {category}** ({round(prediction)}). Air quality looks acceptable.")
    # ---- SHAP Explanation ----
st.subheader("🔍 Why this prediction?")

@st.cache_resource
def get_shap_explainer(_model):
    return shap.TreeExplainer(_model)

explainer = get_shap_explainer(rf_model)
shap_values = explainer.shap_values(X_latest)

feature_names = feature_columns
shap_df = pd.DataFrame({
    "Feature": feature_names,
    "Impact": shap_values[0]
}).sort_values("Impact", key=abs, ascending=True)

fig, ax = plt.subplots(figsize=(8, 4))
colors = ["#ff4b4b" if x > 0 else "#4b8bff" for x in shap_df["Impact"]]
ax.barh(shap_df["Feature"], shap_df["Impact"], color=colors)
ax.set_xlabel("Impact on Predicted AQI")
ax.set_title("Feature Contributions to This Prediction")
st.pyplot(fig)

st.caption("🔴 Red bars increase the predicted AQI. 🔵 Blue bars decrease it.")