"""Local SQLite database for sensor monitoring, readings and alert history.

The dashboard uses this database as a small, self-contained monitoring backend.
For a real deployment, the sensor writer functions can be replaced by MQTT/API/serial inputs.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

from .config import ROOT

DB_PATH = ROOT / "data" / "flood_monitor.db"

SENSORS = [
    ("RG-01", "Rain Gauge", "Rainfall", "mm", "Upstream Catchment", 5.0, 300.0),
    ("WL-01", "River Level Sensor", "River Level", "m", "Main River", 0.5, 8.0),
    ("SM-01", "Soil Moisture Probe", "Soil Moisture", "%", "River Bank", 0.0, 100.0),
    ("TH-01", "Temperature Sensor", "Temperature", "°C", "Weather Station", -10.0, 50.0),
    ("HU-01", "Humidity Sensor", "Humidity", "%", "Weather Station", 20.0, 100.0),
    ("BP-01", "Barometric Sensor", "Pressure", "hPa", "Weather Station", 850.0, 1090.0),
]


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db() -> None:
    with connect() as con:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS sensors (
                sensor_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                metric TEXT NOT NULL,
                unit TEXT NOT NULL,
                location TEXT NOT NULL,
                min_value REAL,
                max_value REAL,
                active INTEGER NOT NULL DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS sensor_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sensor_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                value REAL NOT NULL,
                source TEXT NOT NULL DEFAULT 'simulator',
                FOREIGN KEY(sensor_id) REFERENCES sensors(sensor_id)
            );
            CREATE INDEX IF NOT EXISTS idx_readings_sensor_time
            ON sensor_readings(sensor_id, timestamp);
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                probability REAL NOT NULL,
                message TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'model'
            );
            """
        )
        for row in SENSORS:
            con.execute(
                "INSERT OR IGNORE INTO sensors(sensor_id,name,metric,unit,location,min_value,max_value) VALUES(?,?,?,?,?,?,?)",
                row,
            )


def seed_from_dataset(df: pd.DataFrame, days: int = 180) -> None:
    """Seed the local DB with recent historical observations once."""
    init_db()
    with connect() as con:
        count = con.execute("SELECT COUNT(*) FROM sensor_readings").fetchone()[0]
        if count:
            return
        d = df.sort_values("date").tail(days).copy()
        mapping = {
            "RG-01": "rainfall_mm",
            "WL-01": "river_level_m",
            "SM-01": "soil_moisture",
            "TH-01": "temperature_c",
            "HU-01": "humidity_pct",
            "BP-01": "pressure_hpa",
        }
        rows = []
        for _, r in d.iterrows():
            ts = pd.Timestamp(r.date).strftime("%Y-%m-%d 12:00:00")
            for sid, col in mapping.items():
                rows.append((sid, ts, float(r[col]), "historical"))
        con.executemany(
            "INSERT INTO sensor_readings(sensor_id,timestamp,value,source) VALUES(?,?,?,?)", rows
        )


def add_readings(values: dict[str, float], source: str = "simulator", timestamp: datetime | None = None) -> None:
    init_db()
    ts = (timestamp or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    with connect() as con:
        con.executemany(
            "INSERT INTO sensor_readings(sensor_id,timestamp,value,source) VALUES(?,?,?,?)",
            [(sid, ts, float(value), source) for sid, value in values.items() if sid],
        )


def latest_readings() -> pd.DataFrame:
    init_db()
    q = """
    SELECT s.sensor_id, s.name, s.metric, s.unit, s.location, s.active,
           r.timestamp, r.value, r.source
    FROM sensors s
    LEFT JOIN sensor_readings r ON r.id = (
        SELECT r2.id FROM sensor_readings r2
        WHERE r2.sensor_id = s.sensor_id ORDER BY r2.timestamp DESC, r2.id DESC LIMIT 1
    )
    ORDER BY s.sensor_id
    """
    with connect() as con:
        return pd.read_sql_query(q, con)


def history(sensor_id: str, hours: int = 168) -> pd.DataFrame:
    init_db()
    with connect() as con:
        return pd.read_sql_query(
            "SELECT timestamp, value, source FROM sensor_readings WHERE sensor_id=? ORDER BY timestamp DESC LIMIT ?",
            con,
            params=(sensor_id, hours * 2),
        ).sort_values("timestamp")


def log_alert(risk_level: str, probability: float, message: str, source: str = "model") -> None:
    init_db()
    with connect() as con:
        con.execute(
            "INSERT INTO alerts(timestamp,risk_level,probability,message,source) VALUES(?,?,?,?,?)",
            (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), risk_level, float(probability), message, source),
        )


def alerts(limit: int = 100) -> pd.DataFrame:
    init_db()
    with connect() as con:
        return pd.read_sql_query(
            "SELECT timestamp,risk_level,probability,message,source FROM alerts ORDER BY id DESC LIMIT ?",
            con,
            params=(limit,),
        )


def sensor_count() -> tuple[int, int]:
    d = latest_readings()
    if d.empty:
        return 0, 0
    online = int(d["timestamp"].notna().sum())
    return len(d), online


def simulate_from_conditions(base: pd.Series, rng: np.random.Generator | None = None) -> dict[str, float]:
    """Create realistic-looking sensor values from the model's latest conditions."""
    rng = rng or np.random.default_rng()
    return {
        "RG-01": max(0.0, float(base.rainfall_mm) + rng.normal(0, 2.5)),
        "WL-01": max(0.3, float(base.river_level_m) + rng.normal(0, 0.06)),
        "SM-01": float(np.clip(float(base.soil_moisture) * 100 + rng.normal(0, 1.5), 0, 100)),
        "TH-01": float(base.temperature_c) + rng.normal(0, 0.4),
        "HU-01": float(np.clip(float(base.humidity_pct) + rng.normal(0, 1.5), 20, 100)),
        "BP-01": float(base.pressure_hpa) + rng.normal(0, 0.7),
    }
