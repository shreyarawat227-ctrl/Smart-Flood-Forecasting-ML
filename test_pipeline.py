import numpy as np, pandas as pd, pytest
from src import risk, auth, config
from src.data_generator import generate
from src.preprocess import clean, engineer, FEATURES


@pytest.fixture(scope="module")
def raw(): return generate(years=3, seed=1)

def test_clean_removes_defects(raw):
    rep = {}; c = clean(raw, rep)
    assert c[config.RAW_FEATURES].isna().sum().sum() == 0
    assert c.date.is_unique and c.date.is_monotonic_increasing
    assert (c.rainfall_mm >= 0).all() and c.temperature_c.between(-10, 50).all()
    assert rep["duplicates"] == 15 and rep["out_of_range"] > 0

def test_features_are_causal(raw):
    c = clean(raw); full = engineer(c); cut = engineer(c.iloc[:600])
    row = 500   # features at t must not change if future data is removed
    pd.testing.assert_series_equal(full.loc[row, FEATURES].astype(float), cut.loc[row, FEATURES].astype(float))

def test_target_definition(raw):
    d = engineer(clean(raw)).dropna(subset=["flood_next_day"])
    nxt = (d.river_level_m.shift(-1) > config.DANGER_LEVEL_M).iloc[:-1]
    assert (d.flood_next_day.iloc[:-1].astype(bool) == nxt).all()

@pytest.mark.parametrize("p,lvl", [(0.0, "LOW"), (0.14, "LOW"), (0.15, "MODERATE"), (0.4, "HIGH"), (0.95, "SEVERE")])
def test_risk_bands(p, lvl): assert risk.classify(p)[0] == lvl

def test_alert_dispatch_threshold(tmp_path):
    log = tmp_path / "a.csv"
    assert not risk.dispatch(risk.build_alert(0.2), log_path=log)
    assert risk.dispatch(risk.build_alert(0.9), log_path=log) and log.exists()

def test_auth(tmp_path, monkeypatch):
    monkeypatch.setattr(auth, "USERS_FILE", tmp_path / "u.json")
    assert auth.verify("admin", "admin123") == "admin" and auth.verify("admin", "bad") is None
    auth.add_user("x", "pw", "viewer"); assert auth.verify("x", "pw") == "viewer"

def test_model_beats_baseline():
    import json; m = json.load(open(config.RESULTS_DIR / "metrics.json"))
    assert m["roc_auc"] > 0.9 and m["recall"] >= 0.85
