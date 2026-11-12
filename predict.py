"""Inference API used by the dashboard and CLI."""
import joblib, pandas as pd
from .config import MODEL_PATH, RAW_FEATURES
from .preprocess import engineer

_cache = {}

def load_bundle():
    if "b" not in _cache: _cache["b"] = joblib.load(MODEL_PATH)
    return _cache["b"]

def reload_bundle(): _cache.clear(); return load_bundle()

def predict_latest(history: pd.DataFrame) -> float:
    """history: cleaned daily rows (date + RAW_FEATURES), >= 15 days, last row = today."""
    b = load_bundle(); feats = engineer(history[["date"] + RAW_FEATURES]).iloc[[-1]][b["features"]]
    return float(b["model"].predict_proba(feats)[0, 1])

def predict_series(df_features: pd.DataFrame):
    b = load_bundle(); return b["model"].predict_proba(df_features[b["features"]])[:, 1]

def explain(history: pd.DataFrame, top=5):
    """Local what-moved-the-needle: re-predict with each driver neutralised to a dry-day value."""
    b, base = load_bundle(), predict_latest(history); out = {}
    for col in ["rainfall_mm", "upstream_rainfall_mm", "river_level_m", "soil_moisture"]:
        h = history.copy(); h.loc[h.index[-1], col] = h[col].iloc[:-1].quantile(0.25)
        out[col] = base - predict_latest(h)
    return dict(sorted(out.items(), key=lambda kv: -abs(kv[1]))[:top])
