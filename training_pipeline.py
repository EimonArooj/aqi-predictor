import pandas as pd
import os
import time
from dotenv import load_dotenv
import hopsworks
#importing libraries from sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import joblib

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT_NAME")

# ---- Connect to Hopsworks ----
project = hopsworks.login(
    project=HOPSWORKS_PROJECT_NAME,
    api_key_value=HOPSWORKS_API_KEY
)

fs = project.get_feature_store()

# ---- Get the feature group ----
aqi_fg = fs.get_feature_group(name="aqi_features", version=4)
# ---- Try reading from Hopsworks, with retries; fall back to local CSV if it fails ----
df = None
max_attempts = 3

for attempt in range(1, max_attempts + 1):
    try:
        print(f"Attempt {attempt} to read from Hopsworks...")
        df = aqi_fg.read(read_options={"use_hive": True})
        print("Successfully read from Hopsworks!")
        break
    except Exception as e:
        print(f"Attempt {attempt} failed: {e}")
        time.sleep(5)

if df is None:
    print("Hopsworks read failed after retries. Falling back to local CSV.")
    df = pd.read_csv("data/aqi_data.csv")

print("Total rows fetched:", len(df))
print(df.head())
# ---- Sort by timestamp (critical for correct shifting) ----
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# ---- Create 3-day-ahead target ----
HOURS_AHEAD = 72  # 3 days

df["aqi_target_3day"] = df["aqi"].shift(-HOURS_AHEAD)

# Drop rows where we don't have a future value to predict (the last 72 rows)
df = df.dropna(subset=["aqi_target_3day"])

print("Rows after creating 3-day-ahead target:", len(df))
print(df[["timestamp", "aqi", "aqi_target_3day"]].head())
# ---- Prepare features (X) and target (y) ----
feature_columns = ["hour", "day", "month", "weekday", "previous_aqi"]
target_column = "aqi_target_3day"

X = df[feature_columns]
y = df[target_column]

# ---- Split into training and testing sets ----
# ---- Time-based split (NOT random) ----
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Train period:", df["timestamp"].iloc[0], "to", df["timestamp"].iloc[split_index-1])
print("Test period:", df["timestamp"].iloc[split_index], "to", df["timestamp"].iloc[-1])
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
# ---- Train Ridge Regression model ----
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

ridge_predictions = ridge_model.predict(X_test)

ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_predictions))
ridge_mae = mean_absolute_error(y_test, ridge_predictions)
ridge_r2 = r2_score(y_test, ridge_predictions)

print("\n--- Ridge Regression Results ---")
print("RMSE:", round(ridge_rmse, 2))
print("MAE:", round(ridge_mae, 2))
print("R²:", round(ridge_r2, 3))
# ---- Train Random Forest model ----
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)

print("\n--- Random Forest Results ---")
print("RMSE:", round(rf_rmse, 2))
print("MAE:", round(rf_mae, 2))
print("R²:", round(rf_r2, 3))

# ---- Save the trained model locally first ----
os.makedirs("model_output", exist_ok=True)
model_path = "model_output/rf_model.pkl"
joblib.dump(rf_model, model_path)
print("Model saved locally at:", model_path)

# ---- Push model to Hopsworks Model Registry ----
mr = project.get_model_registry()

rf_hopsworks_model = mr.python.create_model(
    name="aqi_rf_model",
    metrics={"rmse": rf_rmse, "mae": rf_mae, "r2": rf_r2},
    description="Random Forest model for 3-day-ahead AQI forecasting (Rawalpindi)"
)

rf_hopsworks_model.save(model_path)
print("Model pushed to Hopsworks Model Registry successfully!")