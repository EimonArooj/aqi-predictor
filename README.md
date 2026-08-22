# 🌫️ Rawalpindi AQI Predictor

A 100% serverless, end-to-end machine learning pipeline that predicts the Air Quality Index (AQI) for **Rawalpindi, Pakistan**, 3 days in advance — with fully automated hourly data collection, daily model retraining, and a live public dashboard.

## 🔗 Live Dashboard

**👉 [View the live app here](https://aqi-predictor-fix8ncwsax8utmmfudshmk.streamlit.app)**

The dashboard updates automatically as new data is collected — no manual refresh or redeployment needed.

---

## 📋 Project Overview

This project was built to forecast air quality for a region with limited direct sensor coverage, using a fully automated, cloud-native pipeline — no servers to manage, no manual data entry, no manual retraining.

**Core capabilities:**
- Fetches live weather and pollution data every hour
- Computes the real US EPA Air Quality Index (0–500 scale) from raw pollutant readings
- Engineers time-based and lag-based features for forecasting
- Trains and evaluates multiple ML models (Ridge Regression, Random Forest)
- Automatically retrains daily as new data accumulates
- Serves 3-day-ahead AQI predictions through a live web dashboard
- Explains individual predictions using SHAP feature importance
- Displays hazard alerts based on standard AQI severity categories

---

## 🏗️ Architecture

```
OpenWeather API  ──▶  feature_pipeline.py  ──▶  Hopsworks Feature Store
  (weather +              (hourly, via                  (aqi_features)
   pollution data)      GitHub Actions)                       │
                                                                ▼
                                                    training_pipeline.py
                                                    (daily, via GitHub Actions)
                                                                │
                                                                ▼
                                                    Hopsworks Model Registry
                                                        (aqi_rf_model)
                                                                │
                                                                ▼
                                                          app.py (Streamlit)
                                                       deployed on Streamlit
                                                          Community Cloud
```

---

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Data Source | [OpenWeather](https://openweathermap.org/) Air Pollution & Weather APIs |
| Feature Store & Model Registry | [Hopsworks](https://www.hopsworks.ai/) |
| ML Models | Scikit-learn (Ridge Regression, Random Forest) |
| Automation / CI-CD | GitHub Actions (hourly + daily scheduled workflows) |
| Dashboard | Streamlit, deployed on Streamlit Community Cloud |
| Explainability | SHAP |
| Language | Python 3.11 |

---

## 📁 Repository Structure

```
aqi-predictor/
│
├── feature_pipeline.py       # Fetches live data hourly, engineers features, pushes to Hopsworks
├── backfill.py                # One-time historical data backfill (30 days) for initial training data
├── training_pipeline.py       # Trains Ridge & Random Forest models, evaluates, saves best model
├── app.py                     # Streamlit dashboard: live predictions, trends, alerts, SHAP explanations
├── requirements.txt            # Python dependencies for local + cloud deployment
├── .github/
│   └── workflows/
│       ├── feature_pipeline.yml     # Runs feature_pipeline.py every hour
│       └── training_pipeline.yml    # Runs training_pipeline.py once daily
├── data/
│   └── aqi_data.csv           # Local backup of collected data (fallback if Hopsworks is unreachable)
└── .gitignore                 # Excludes .env and venv/ from version control
```

---

## ⚙️ How It Works

### 1. Feature Pipeline (`feature_pipeline.py`) — runs hourly
- Fetches current pollution data (PM2.5, PM10, CO, NO₂, SO₂, O₃) and weather data (temperature, humidity, pressure, wind speed) for Rawalpindi's coordinates
- Calculates the official **US EPA AQI** from PM2.5 concentration using standard breakpoint formulas
- Engineers time-based features (hour, day, month, weekday) and a `previous_aqi` lag feature
- Pushes the new row to the Hopsworks Feature Store (`aqi_features`, version 4)

### 2. Historical Backfill (`backfill.py`) — run once
- Pulls the past 30 days of hourly pollution history via OpenWeather's History API
- Applies the same feature engineering as the live pipeline
- Bulk-inserts ~718 historical rows into the Feature Store to bootstrap model training

### 3. Training Pipeline (`training_pipeline.py`) — runs daily
- Reads the full accumulated dataset from Hopsworks
- Builds a genuine **3-day-ahead forecasting target** by shifting the AQI column forward 72 hours
- Splits data **chronologically** (not randomly) — trains on older data, tests on more recent data, to properly simulate real-world forecasting
- Trains and evaluates Ridge Regression and Random Forest models using RMSE, MAE, and R²
- Saves the trained model to the Hopsworks Model Registry (`aqi_rf_model`)

### 4. Dashboard (`app.py`)
- Loads the latest model and feature data from Hopsworks
- Displays current AQI, a 7-day trend chart, and the 3-day-ahead prediction
- Shows a color-coded hazard alert based on EPA AQI categories
- Uses SHAP to explain which features drove each individual prediction

### 5. Automation (GitHub Actions)
- `feature_pipeline.yml` → runs `feature_pipeline.py` every hour, forever, independent of any local machine
- `training_pipeline.yml` → runs `training_pipeline.py` once daily to keep the model fresh as new data accumulates
- API keys are stored securely as GitHub Secrets, never committed to the repository

---

## 🐛 Notable Engineering Challenges Solved

Building this pipeline surfaced several real machine learning and infrastructure issues worth documenting:

- **Data leakage (round 1):** Raw pollutant concentrations (`pm2_5`, `pm10`) were initially included as model features, but since AQI is directly calculated from `pm2_5`, this let models "cheat" by reverse-engineering the formula (Random Forest scored a suspicious R² = 1.0). Fixed by removing raw pollutant concentrations from the feature set.
- **Data leakage (round 2):** An engineered `aqi_change_rate` feature was mathematically derived from the current target value itself. Replaced with a legitimate lag feature (`previous_aqi`) using only already-known historical values.
- **Temporal leakage in evaluation:** Random train/test splitting let models "peek" at data points very close in time to the test set, inflating scores. Fixed by switching to a strict chronological split — training only on older dates, testing only on newer ones.
- **Forecast horizon mismatch:** The model was initially predicting 1-hour-ahead AQI rather than the required 3-day-ahead forecast. Fixed by shifting the target column forward 72 hours.
- **Cross-platform dependency issues:** Several packages (`pyarrow`, `confluent-kafka`, `delta-spark`) required careful version pinning and Windows-specific build tools locally, and separate fixes for Streamlit Cloud's Linux build environment (removing unused packages, pinning minimum versions to avoid pip dependency backtracking into broken legacy releases).

---

## 🚀 Running Locally

```bash
# Clone the repository
git clone https://github.com/EimonArooj/aqi-predictor.git
cd aqi-predictor

# Create and activate a virtual environment (Python 3.11 recommended)
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Create a .env file with your API keys
# OPENWEATHER_API_KEY=your_key_here
# HOPSWORKS_API_KEY=your_key_here
# HOPSWORKS_PROJECT_NAME=your_project_name

# Run the feature pipeline once to collect live data
python feature_pipeline.py

# Run the dashboard
streamlit run app.py
```

---

## 📈 Future Improvements

- Incorporate weather features (temperature, humidity, wind) into training once sufficient historical weather data accumulates (currently collected live going forward, but unavailable retroactively due to API tier limits)
- Add a multi-model comparison view in the dashboard (Ridge vs. Random Forest side by side)
- Add a deep learning (TensorFlow) model as an additional forecasting option
- Expand historical backfill beyond 30 days as data availability allows

---

## 👤 Author

**Eimon Arooj Mazhar**
Aspiring Data Scientist | CS Undergraduate | Building ML & Data Science Projects

---

## 📄 License

This project was built for educational purposes as part of a machine learning systems course project.
