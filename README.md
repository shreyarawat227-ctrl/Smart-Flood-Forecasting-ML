# 🌊 Smart Flood Forecasting & Early Warning using Machine Learning
BCA (AI & DS) project — Shreya Rawat, Pari Kulshrestha, Mahima Sharma

Predicts whether a river will cross its **danger level in the next 24 hours**, classifies the risk
(LOW / MODERATE / HIGH / SEVERE), issues alerts, and serves everything through a secured dashboard.

## Architecture (matches the proposal's 6 layers)
```
Data Acquisition → Preprocessing → Feature Engineering → Prediction (ML) → Risk Classification → Alerts → Dashboard
 data_generator     preprocess.clean   preprocess.engineer   train/predict      risk.classify      risk.dispatch  app/dashboard.py
```

| Folder | Contents |
|---|---|
| `src/` | config, data generator, cleaning + features, training, inference, risk/alerts, auth |
| `app/dashboard.py` | Streamlit UI: login, forecast, what-if simulator, model report, alert log, admin panel |
| `tests/` | 11 pytest tests (cleaning, causality of features, target, risk bands, alerts, auth, model quality) |
| `data/`, `models/`, `results/` | datasets, saved model, plots + metrics |

## Run
```bash
pip install -r requirements.txt
python run_all.py                    # data → clean → train → evaluate → demo alert
python -m pytest -q                  # tests
streamlit run app/dashboard.py       # login: admin/admin123 or viewer/viewer123
```

## What makes it advanced (not just "fit a model")
- **Messy-data pipeline**: duplicates, mixed date formats, sensor spikes (-9999, 85 °C), ~3 % missing values → validated ranges, interpolation, rain gaps = 0.
- **Hydrology-aware features**: lagged / rolling rainfall (1–14 d), upstream rain, soil saturation, level trend, distance-to-danger, soil×rain interaction, seasonality (sin/cos).
- **Leak-free evaluation**: chronological split (last 3 years held out), `TimeSeriesSplit` CV, causality unit-test.
- **Imbalanced problem handled**: ~5 % flood days → class weights, PR-AUC as selection metric, **F2 threshold tuning** (a missed flood is worse than a false alarm).
- **Model comparison**: Logistic Regression vs Random Forest vs Gradient Boosting, then **isotonic probability calibration** so "40 %" really means ~40 %.
- **Baseline comparison** against a naive "level > 5 m" rule; permutation feature importance; local explanation in the UI.
- **Secured app**: PBKDF2-salted passwords, role-based access (admin panel for users, data upload, one-click retrain).

## Results (held-out last 3 years)
ROC-AUC **0.991**, PR-AUC **0.883**, recall **0.95** (56/59 floods caught), precision 0.50, Brier 0.016.
Naive baseline: recall 0.85. See `results/evaluation.png`, `feature_importance.png`, `timeline.png`.

## Using real data
Replace `data/flood_raw.csv` with gauge/IMD data using the columns
`date, rainfall_mm, upstream_rainfall_mm, river_level_m, soil_moisture, temperature_c, humidity_pct, pressure_hpa`,
set `DANGER_LEVEL_M` in `src/config.py`, and re-run `python run_all.py`.
> The bundled dataset is **synthetic** (physics-based simulation), clearly stated so in the report; accuracy on real data will differ.

## Future work
Hourly data & LSTM/GRU, multi-day horizon, live IMD/CWC API ingestion, Twilio SMS / SMTP e-mail in `risk.dispatch`, GIS inundation map.

## Advanced monitoring features

This version adds a practical monitoring layer on top of the ML flood forecast:

- **Six sensor types:** rain gauge, river-level sensor, soil-moisture probe, temperature, humidity and barometric pressure.
- **SQLite database:** `data/flood_monitor.db` stores sensor readings and alert/incident history locally, so the project works without a cloud account or API key.
- **Live sensor simulator:** the Live Sensors tab generates realistic telemetry from the selected forecast conditions and saves each packet to the database.
- **Sensor history:** select any sensor to view its stored telemetry over time.
- **Overview control room:** current risk, river level, rainfall, soil moisture, sensor status and trend charts are shown together.
- **2030 data:** the bundled synthetic training/monitoring data runs through December 2030.
- **Future hardware integration:** replace the simulator writer with readings from ESP32/Arduino, LoRaWAN, MQTT or a weather/gauge API without changing the dashboard concept.

### Database tables

- `sensors` — sensor ID, type, location and operating range.
- `sensor_readings` — timestamped sensor telemetry and source.
- `alerts` — model/manual alert events and risk probability.

The sensor values in this college prototype are **simulated**, not measurements from physical field hardware. This is intentional so the project remains runnable on a normal laptop. For a real deployment, connect the sensor devices to the database ingestion layer.


## Added archive dataset (limited to 160 rows)

- File: `data/archive_flood_dataset_160.csv`
- Exactly **160 data rows** are included, sampled from the supplied `flood.csv` archive using a fixed random seed.
- It has 20 risk-factor input columns and `FloodProbability` as the target.
- This dataset is kept separate from `data/flood_raw.csv`: the main project uses daily sensor/time-series columns, while this archive dataset uses risk-factor scores. Keeping them separate avoids mixing incompatible columns or changing the existing dashboard's forecasting model.
- Train and evaluate a separate Random Forest regression model with:
  ```bash
  python -m src.train_archive_model
  ```
- Outputs: `models/archive_flood_model_160.joblib` and `results/archive_flood_metrics_160.json`.
- Because this is only a 160-row subset, its evaluation is for demonstration/academic use and should not be treated as validated real-world flood warning performance.
