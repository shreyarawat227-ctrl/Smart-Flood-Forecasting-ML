"""Train a separate regression model using the 160-row archive flood dataset.

This is separate from the main time-series river-level forecasting pipeline because
the archive dataset uses risk-factor scores and FloodProbability rather than daily sensor readings.
Run from the project root: python -m src.train_archive_model
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from .config import ROOT, RANDOM_STATE

DATASET = ROOT / "data" / "archive_flood_dataset_160.csv"
MODEL_OUT = ROOT / "models" / "archive_flood_model_160.joblib"
METRICS_OUT = ROOT / "results" / "archive_flood_metrics_160.json"
TARGET = "FloodProbability"


def main():
    df = pd.read_csv(DATASET)
    if len(df) != 160:
        raise ValueError(f"Expected exactly 160 dataset rows, found {len(df)}")
    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}")

    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    model = RandomForestRegressor(
        n_estimators=200, random_state=RANDOM_STATE, min_samples_leaf=2
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "dataset": DATASET.name,
        "dataset_rows": int(len(df)),
        "features": int(X.shape[1]),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "model": "RandomForestRegressor",
        "mae": float(mean_absolute_error(y_test, predictions)),
        "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
        "r2": float(r2_score(y_test, predictions)),
        "note": "Small 160-row subset; test metrics may vary and are not evidence of real-world flood warning accuracy."
    }
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": list(X.columns), "target": TARGET}, MODEL_OUT)
    METRICS_OUT.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"Saved model: {MODEL_OUT}")
    print(f"Saved metrics: {METRICS_OUT}")


if __name__ == "__main__":
    main()
