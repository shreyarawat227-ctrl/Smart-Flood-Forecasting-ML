"""Synthetic but hydrologically-plausible dataset for a Himalayan-foothill river basin.

Real gauge data (CWC / IMD) can be dropped into data/flood_raw.csv with the same columns
and the rest of the pipeline works unchanged.
Physics used: monsoon-seasonal rainfall -> soil saturation (leaky bucket)
-> runoff -> river level (linear reservoir with 1-day upstream routing).
Deliberate defects (missing values, spikes, duplicates, mixed date formats) are injected so
the preprocessing stage has real work to do.
"""
import numpy as np, pandas as pd
from .config import RAW_CSV, RANDOM_STATE


def generate(years=17, start="2014-01-01", seed=RANDOM_STATE, inject_defects=True):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=int(365.25 * years), freq="D")
    n, doy = len(dates), dates.dayofyear.values
    monsoon = np.exp(-((doy - 205) / 38.0) ** 2)               # peak ~ late July
    p_rain = 0.08 + 0.75 * monsoon
    scale = 6 + 38 * monsoon

    def rain_series():
        wet = rng.random(n) < p_rain
        amt = rng.gamma(0.8, scale)
        amt[rng.random(n) < 0.03 * (monsoon > 0.3)] *= 3.5     # cloudburst days
        return np.where(wet, amt, 0.0)

    rain, up = rain_series(), rain_series()
    up = 0.6 * up + 0.4 * rain                                 # correlated catchments
    soil, level = np.zeros(n), np.zeros(n)
    s, lv = 0.3, 1.5
    for t in range(n):
        s = np.clip(0.88 * s + 0.0045 * rain[t], 0, 1)
        runoff = rain[t] * (0.15 + 0.85 * s)
        routed = up[t - 1] * (0.15 + 0.85 * s) if t else 0
        lv = 1.2 + 0.55 * (lv - 1.2) + 0.040 * runoff + 0.050 * routed + rng.normal(0, 0.05)
        soil[t], level[t] = s, max(lv, 0.3)
    temp = 22 + 9 * np.sin(2 * np.pi * (doy - 110) / 365) - 3.5 * monsoon + rng.normal(0, 1.8, n)
    hum = np.clip(55 + 35 * monsoon + 6 * (rain > 1) + rng.normal(0, 6, n), 20, 100)
    pres = 1008 - 5 * monsoon - 0.04 * np.minimum(rain, 120) + rng.normal(0, 1.5, n)
    df = pd.DataFrame({"date": dates, "rainfall_mm": rain.round(1),
                       "upstream_rainfall_mm": up.round(1), "river_level_m": level.round(2),
                       "soil_moisture": soil.round(3), "temperature_c": temp.round(1),
                       "humidity_pct": hum.round(1), "pressure_hpa": pres.round(1)})
    if inject_defects:
        for c in df.columns[1:]:                               # ~3% missing
            df.loc[rng.random(n) < 0.03, c] = np.nan
        df.loc[rng.choice(n, 8, replace=False), "rainfall_mm"] = rng.choice([-9999, 999, 850], 8)
        df.loc[rng.choice(n, 5, replace=False), "temperature_c"] = rng.choice([-99, 85], 5)
        df = pd.concat([df, df.sample(15, random_state=seed)]).sample(frac=1, random_state=seed)
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        mix = rng.random(len(df)) < 0.05                       # inconsistent date format
        df.loc[mix, "date"] = pd.to_datetime(df.loc[mix, "date"]).dt.strftime("%d/%m/%Y")
    return df


if __name__ == "__main__":
    RAW_CSV.parent.mkdir(exist_ok=True)
    d = generate(); d.to_csv(RAW_CSV, index=False); print(f"Wrote {len(d)} rows -> {RAW_CSV}")
