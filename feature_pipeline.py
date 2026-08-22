import requests
import pandas as pd
import os
import hopsworks
from dotenv import load_dotenv
from datetime import datetime

# Load variables from .env file into the program
load_dotenv()

# Read the API key safely
API_KEY = os.getenv("OPENWEATHER_API_KEY")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")

# Rawalpindi coordinates
LAT = 33.6
LON = 73.0

print("API key loaded:", API_KEY is not None)

# ---- Connect to Hopsworks ----
project = hopsworks.login(
    project=HOPSWORKS_PROJECT_NAME,
    api_key_value=HOPSWORKS_API_KEY
)

fs = project.get_feature_store()
print("Connected to Hopsworks project:", project.name)

# ---- STEP 6: AQI calculation function ----
def calculate_aqi(concentration, breakpoints):
    for bp_lo, bp_hi, aqi_lo, aqi_hi, conc_lo, conc_hi in breakpoints:
        if conc_lo <= concentration <= conc_hi:
            aqi = ((aqi_hi - aqi_lo) / (conc_hi - conc_lo)) * (concentration - conc_lo) + aqi_lo
            return round(aqi)
    return None

pm25_breakpoints = [
    (0, 50, 0, 50, 0.0, 12.0),
    (0, 50, 51, 100, 12.1, 35.4),
    (0, 50, 101, 150, 35.5, 55.4),
    (0, 50, 151, 200, 55.5, 150.4),
    (0, 50, 201, 300, 150.5, 250.4),
    (0, 50, 301, 500, 250.5, 500.4),
]

# ---- Call OpenWeather Air Pollution API ----
url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={API_KEY}"
response = requests.get(url)
data = response.json()

# ---- Call OpenWeather Current Weather API ----
weather_url = f"https://api.openweathermap.org/data/2.5/weather?lat={LAT}&lon={LON}&appid={API_KEY}&units=metric"
weather_response = requests.get(weather_url)
weather_data = weather_response.json()

print("Weather API raw response:", weather_data)  # TEMPORARY DEBUG LINE

temperature = weather_data["main"]["temp"]
humidity = weather_data["main"]["humidity"]
pressure = weather_data["main"]["pressure"]
wind_speed = weather_data["wind"]["speed"]

print("Temperature:", temperature)
print("Humidity:", humidity)
print("Wind Speed:", wind_speed)

# Extract pollutant components
components = data["list"][0]["components"]

pm25 = components["pm2_5"]
pm10 = components["pm10"]
co = components["co"]
no2 = components["no2"]
so2 = components["so2"]
o3 = components["o3"]

print("PM2.5:", pm25)
print("PM10:", pm10)

# ---- STEP 7: Calculate AQI ----
aqi_pm25 = calculate_aqi(pm25, pm25_breakpoints)
print("Calculated AQI (PM2.5-based):", aqi_pm25)

# ---- STEP 8: Time-based features ----
now = datetime.now()

hour = now.hour
day = now.day
month = now.month
weekday = now.weekday()  # Monday=0 ... Sunday=6

print("Hour:", hour)
print("Day:", day)
print("Month:", month)
print("Weekday:", weekday)

# File path
file_path = "data/aqi_data.csv"

# ---- STEP 11 (corrected): previous_aqi as a lag feature, NOT change rate ----
if os.path.exists(file_path):
    df_history = pd.read_csv(file_path)
    if len(df_history) > 0:
        previous_aqi = df_history.iloc[-1]["aqi"]
    else:
        previous_aqi = aqi_pm25
else:
    previous_aqi = aqi_pm25

print("Previous AQI:", previous_aqi)
# ---- END STEP 11 ----

# ---- STEP 9: Save as a row in CSV ----
row = {
    "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
    "hour": hour,
    "day": day,
    "month": month,
    "weekday": weekday,
    "pm2_5": pm25,
    "pm10": pm10,
    "co": co,
    "no2": no2,
    "so2": so2,
    "o3": o3,
    "aqi": aqi_pm25,
    "previous_aqi": previous_aqi,
    "temperature": float(temperature),
    "humidity": float(humidity),
    "pressure": float(pressure),
    "wind_speed": float(wind_speed)
}
df_new = pd.DataFrame([row])

if os.path.exists(file_path):
    df_new.to_csv(file_path, mode="a", header=False, index=False)
else:
    df_new.to_csv(file_path, mode="w", header=True, index=False)

print("Row saved successfully!")
print(df_new)
# ---- END STEP 9 ----

# ---- Push data to Hopsworks Feature Store (version 2, matching backfill) ----
aqi_fg = fs.get_or_create_feature_group(
    name="aqi_features",
    version=4,
    primary_key=["timestamp"],
    description="Hourly AQI, weather and time features for Rawalpindi (v3 - added weather)",
    time_travel_format="HUDI"
)

aqi_fg.insert(df_new)
print("Data pushed to Hopsworks Feature Store successfully!")