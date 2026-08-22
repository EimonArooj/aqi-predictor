import requests
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import hopsworks
import time

# Load environment variables
load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")

LAT = 33.6
LON = 73.0

print("Starting backfill...")

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

# ---- Calculate time range: past 30 days ----
end_time = datetime.now()
start_time = end_time - timedelta(days=30)

end_ts = int(end_time.timestamp())
start_ts = int(start_time.timestamp())

print(f"Fetching data from {start_time} to {end_time}")

# ---- Fetch historical data in one API call ----
url = f"https://api.openweathermap.org/data/2.5/air_pollution/history?lat={LAT}&lon={LON}&start={start_ts}&end={end_ts}&appid={API_KEY}"
response = requests.get(url)
data = response.json()

records = data["list"]
print(f"Total records fetched: {len(records)}")

# ---- Process each record into features ----
rows = []
previous_aqi = None

for record in records:
    components = record["components"]
    pm25 = components["pm2_5"]
    pm10 = components["pm10"]
    co = components["co"]
    no2 = components["no2"]
    so2 = components["so2"]
    o3 = components["o3"]

    aqi = calculate_aqi(pm25, pm25_breakpoints)

    if aqi is None:
        continue

    dt_obj = datetime.fromtimestamp(record["dt"])

    row_lag_aqi = previous_aqi if previous_aqi is not None else aqi
    previous_aqi = aqi

    row = {
        "timestamp": dt_obj.strftime("%Y-%m-%d %H:%M:%S"),
        "hour": dt_obj.hour,
        "day": dt_obj.day,
        "month": dt_obj.month,
        "weekday": dt_obj.weekday(),
        "pm2_5": pm25,
        "pm10": pm10,
        "co": co,
        "no2": no2,
        "so2": so2,
        "o3": o3,
        "aqi": aqi,
        "previous_aqi": row_lag_aqi,
        "temperature": np.nan,
        "humidity": np.nan,
        "pressure": np.nan,
        "wind_speed": np.nan
    }
    rows.append(row)

df_backfill = pd.DataFrame(rows)
# ---- Ensure weather columns are consistently float type ----
df_backfill["temperature"] = df_backfill["temperature"].astype(float)
df_backfill["humidity"] = df_backfill["humidity"].astype(float)
df_backfill["pressure"] = df_backfill["pressure"].astype(float)
df_backfill["wind_speed"] = df_backfill["wind_speed"].astype(float)
# ---- Save backfill locally too (backup) ----
df_backfill.to_csv("data/aqi_data.csv", mode="w", header=True, index=False)
print("Backfill also saved locally as backup.")
print(df_backfill.head())
print("Total rows processed:", len(df_backfill))

# ---- Connect to Hopsworks ----
project = hopsworks.login(
    project=HOPSWORKS_PROJECT_NAME,
    api_key_value=HOPSWORKS_API_KEY
)

fs = project.get_feature_store()

# ---- Get the feature group (v3 - includes weather columns) ----
aqi_fg = fs.get_or_create_feature_group(
    name="aqi_features",
    version=4,
    primary_key=["timestamp"],
    description="Hourly AQI, weather and time features for Rawalpindi (v3 - added weather)",
    time_travel_format="HUDI"
)

# ---- Insert entire batch at once ----
aqi_fg.insert(df_backfill)
print("Backfill data pushed to Hopsworks successfully!")