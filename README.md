# Rawalpindi AQI Predictor

A 100% serverless machine learning pipeline that predicts Air Quality Index (AQI) for Rawalpindi, Pakistan, 3 days in advance.

## 🔗 Live Dashboard

**[View the live app here](https://aqi-predictor-fix8ncwsax8utmmfudshmk.streamlit.app)**

## Project Overview

- **Data Source:** OpenWeather Air Pollution & Weather APIs
- **Feature Store & Model Registry:** Hopsworks
- **Automation:** GitHub Actions (hourly data collection)
- **Models:** Ridge Regression, Random Forest
- **Dashboard:** Streamlit

## Pipeline Components

- `feature_pipeline.py` — Fetches live weather/pollution data hourly, engineers features, pushes to Hopsworks
- `backfill.py` — Historical data backfill for model training
- `training_pipeline.py` — Trains and evaluates models, saves best model to Hopsworks Model Registry
- `app.py` — Streamlit dashboard displaying live predictions

## Author

Eimon Arooj Mazhar