"""Data cleaning + feature engineering. Everything is causal (uses only past/present data)."""
import numpy as np, pandas as pd
from .config import DANGER_LEVEL_M, RAW_FEATURES, RAW_CSV, CLEAN_CSV

VALID = {"rainfall_mm": (0, 600), "upstream_rainfall_mm": (0, 600), "river_level_m": (0, 20),
         "soil_moisture": (0, 1), "temperature_c": (-10, 50),
         "humidity_pct": (0, 100), "pressure_hpa": (850, 1090)}


def _parse_dates(s):
    iso = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    return iso.fillna(pd.to_datetime(s, format="%d/%m/%Y", errors="coerce"))


def clean(df: pd.DataFrame, report: dict | None = None) -> pd.DataFrame:
    rep = {} if report is None else report
    df = df.copy(); rep["rows_in"] = len(df)
    df["date"] = _parse_dates(df["date"]); rep["bad_dates"] = int(df["date"].isna().sum())
    df = df.dropna(subset=["date"])
    before = len(df); df = df.drop_duplicates("date").sort_values("date"); rep["duplicates"] = before - len(df)
    bad = 0
    for c, (lo, hi) in VALID.items():                           # physically impossible -> NaN
        m = (df[c] < lo) | (df[c] > hi); bad += int(m.sum()); df.loc[m, c] = np.nan
    rep["out_of_range"] = bad; rep["missing_before_impute"] = int(df[RAW_FEATURES].isna().sum().sum())
    df = df.set_index("date").asfreq("D")                       # restore any missing days
    rain = ["rainfall_mm", "upstream_rainfall_mm"]
    df[rain] = df[rain].fillna(0)                               # no report ~ no rain
    others = [c for c in RAW_FEATURES if c not in rain]
    df[others] = df[others].interpolate(limit=5, limit_direction="both")
    df = df.ffill().bfill(); rep["rows_out"] = len(df)
    return df.reset_index()


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for k in (1, 2, 3): d[f"rain_lag{k}"] = d["rainfall_mm"].shift(k)
    d["rain_3d"] = d["rainfall_mm"].rolling(3).sum()
    d["rain_7d"] = d["rainfall_mm"].rolling(7).sum()
    d["rain_14d"] = d["rainfall_mm"].rolling(14).sum()
    d["upstream_3d"] = d["upstream_rainfall_mm"].rolling(3).sum()
    d["level_change_1d"] = d["river_level_m"].diff()
    d["level_max_3d"] = d["river_level_m"].rolling(3).max()
    d["level_to_danger"] = DANGER_LEVEL_M - d["river_level_m"]
    d["soil_x_rain"] = d["soil_moisture"] * d["rainfall_mm"]
    d["pressure_change_1d"] = d["pressure_hpa"].diff()
    d["month_sin"] = np.sin(2 * np.pi * d["date"].dt.month / 12)
    d["month_cos"] = np.cos(2 * np.pi * d["date"].dt.month / 12)
    d["is_monsoon"] = d["date"].dt.month.isin([6, 7, 8, 9]).astype(int)
    # target: does the river exceed the danger level TOMORROW? (24h forecast horizon)
    d["flood_next_day"] = (d["river_level_m"].shift(-1) > DANGER_LEVEL_M).astype(float)
    d.loc[d.index[-1], "flood_next_day"] = np.nan
    return d


FEATURES = (RAW_FEATURES + ["rain_lag1", "rain_lag2", "rain_lag3", "rain_3d", "rain_7d", "rain_14d",
            "upstream_3d", "level_change_1d", "level_max_3d", "level_to_danger", "soil_x_rain",
            "pressure_change_1d", "month_sin", "month_cos", "is_monsoon"])


def build_dataset():
    rep = {}; raw = pd.read_csv(RAW_CSV)
    out = engineer(clean(raw, rep)).dropna(subset=FEATURES + ["flood_next_day"])
    out.to_csv(CLEAN_CSV, index=False); rep["flood_rate"] = round(float(out["flood_next_day"].mean()), 4)
    return out, rep


if __name__ == "__main__":
    _, r = build_dataset(); print(r)
