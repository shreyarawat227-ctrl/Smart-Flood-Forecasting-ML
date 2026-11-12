"""Central configuration for the Smart Flood Forecasting project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR, MODEL_DIR, RESULTS_DIR = ROOT / "data", ROOT / "models", ROOT / "results"
RAW_CSV = DATA_DIR / "flood_raw.csv"
CLEAN_CSV = DATA_DIR / "flood_clean.csv"
MODEL_PATH = MODEL_DIR / "flood_model.joblib"
ALERT_LOG = DATA_DIR / "alert_log.csv"
USERS_FILE = DATA_DIR / "users.json"

DANGER_LEVEL_M = 6.0          # river level above which the next-day label = flood
RANDOM_STATE = 42
TEST_YEARS = 3                # last N years held out (chronological split)

# Risk bands on predicted flood probability (inclusive lower bound)
RISK_BANDS = [(0.00, "LOW", "#2e7d32"), (0.15, "MODERATE", "#f9a825"),
              (0.40, "HIGH", "#ef6c00"), (0.70, "SEVERE", "#c62828")]

RAW_FEATURES = ["rainfall_mm", "upstream_rainfall_mm", "river_level_m",
                "soil_moisture", "temperature_c", "humidity_pct", "pressure_hpa"]
