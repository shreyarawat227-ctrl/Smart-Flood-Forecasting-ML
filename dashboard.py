"""Advanced Streamlit dashboard for Smart Flood Forecasting & Early Warning."""
import sys, pathlib, subprocess
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from src import auth, predict, risk, database
from src.config import *
from src.preprocess import RAW_FEATURES

st.set_page_config(page_title="Smart Flood Monitoring", page_icon="🌊", layout="wide", initial_sidebar_state="expanded")

# ---------------- setup ----------------
if "role" not in st.session_state:
    st.session_state.role = None
if "last_sensor_values" not in st.session_state:
    st.session_state.last_sensor_values = None

def inject_css():
    st.markdown("""
    <style>
    .main-title {font-size: 2.2rem; font-weight: 800; margin-bottom: 0.1rem;}
    .subtitle {color: #6b7280; margin-bottom: 1rem;}
    .status {padding: 8px 12px; border-radius: 999px; font-weight: 700; display: inline-block; font-size: 0.85rem;}
    .online {background:#dcfce7; color:#166534;} .warning {background:#fef3c7; color:#92400e;}
    .sensor-card {border:1px solid #e5e7eb; border-radius:14px; padding:14px; background:#ffffff; box-shadow:0 1px 4px rgba(0,0,0,.05);}
    .small {font-size:.82rem; color:#6b7280;}
    </style>
    """, unsafe_allow_html=True)

inject_css()

# ---------------- login ----------------
if not st.session_state.role:
    st.markdown('<div class="main-title">🌊 Smart Flood Forecasting & Early Warning</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AI-powered flood risk monitoring with sensor telemetry and a local database</div>', unsafe_allow_html=True)
    c, _ = st.columns([1, 2])
    with c:
        st.subheader("Secure Login")
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Sign in", type="primary", use_container_width=True):
            r = auth.verify(u, p)
            if r:
                st.session_state.role, st.session_state.user = r, u
                st.rerun()
            else:
                st.error("Invalid credentials")
        st.caption("Demo accounts: admin / admin123  •  viewer / viewer123")
    st.stop()

if not MODEL_PATH.exists():
    st.error("Model not found. Run `python run_all.py` first.")
    st.stop()

@st.cache_data

def load_data():
    return pd.read_csv(CLEAN_CSV, parse_dates=["date"])

df = load_data()
bundle = predict.load_bundle()
database.seed_from_dataset(df)

# ---------------- sidebar ----------------
with st.sidebar:
    st.markdown("### 🌊 FloodWatch")
    st.caption("Smart Flood Forecasting & Early Warning System")
    st.write(f"👤 **{st.session_state.user}** · {st.session_state.role.title()}")
    if st.button("Sign out", use_container_width=True):
        st.session_state.role = None
        st.rerun()
    st.divider()
    date_min, date_max = df.date.min().date(), df.date.max().date()
    sel = st.date_input("Forecast date", date_max, min_value=date_min + pd.Timedelta(days=20), max_value=date_max)
    st.caption(f"Available model data: **{date_min} → {date_max}**")
    st.divider()
    st.markdown("**System status**")
    total, online = database.sensor_count()
    st.success(f"● {online}/{total} sensors online")
    st.caption("Local SQLite database connected")
    st.caption("Sensor mode: simulated telemetry")

# ---------------- common calculations ----------------
hist = df[df.date <= pd.Timestamp(sel)].tail(30)[["date"] + RAW_FEATURES].reset_index(drop=True)
prob = predict.predict_latest(hist)
alert = risk.build_alert(prob, pd.Timestamp(sel))
latest = hist.iloc[-1]

# Tabs
tabs = ["🏠 Overview", "📡 Live Sensors", "🔮 Forecast", "🧪 What-if", "📈 History", "🤖 Model", "🚨 Alerts"]
if st.session_state.role == "admin":
    tabs.append("⚙️ Admin")
T = st.tabs(tabs)

# ---------------- overview ----------------
with T[0]:
    st.markdown('<div class="main-title">Flood Monitoring Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">A practical control-room view combining sensor data, historical trends and machine-learning risk prediction.</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Flood probability", f"{prob:.1%}", "next 24 hours")
    c2.metric("River level", f"{latest.river_level_m:.2f} m", f"Danger: {DANGER_LEVEL_M:.1f} m")
    c3.metric("Rainfall", f"{latest.rainfall_mm:.1f} mm", "today")
    c4.metric("Soil moisture", f"{latest.soil_moisture*100:.0f}%", "saturation indicator")

    if alert["risk_level"] in ("HIGH", "SEVERE"):
        st.error(f"🚨 **{alert['risk_level']} RISK** — {alert['advice']}")
    elif alert["risk_level"] == "MODERATE":
        st.warning(f"⚠️ **MODERATE RISK** — {alert['advice']}")
    else:
        st.success(f"✅ **LOW RISK** — {alert['advice']}")

    st.subheader("Sensor network")
    sensors = database.latest_readings()
    cols = st.columns(3)
    for i, (_, r) in enumerate(sensors.iterrows()):
        with cols[i % 3]:
            value = "—" if pd.isna(r.value) else f"{r.value:.2f} {r.unit}"
            st.markdown(f'''<div class="sensor-card"><b>📡 {r.name}</b><br><span class="small">{r.sensor_id} · {r.location}</span><h3>{value}</h3><span class="status online">● ONLINE</span></div>''', unsafe_allow_html=True)

    st.subheader("River level & rainfall trend")
    chart = make_subplots(specs=[[{"secondary_y": True}]])
    chart.add_trace(go.Scatter(x=hist.date, y=hist.river_level_m, name="River level (m)", mode="lines+markers"), secondary_y=False)
    chart.add_trace(go.Bar(x=hist.date, y=hist.rainfall_mm, name="Rainfall (mm)", opacity=.35), secondary_y=True)
    chart.add_hline(y=DANGER_LEVEL_M, line_dash="dash", annotation_text="Danger level", secondary_y=False)
    chart.update_layout(height=380, margin=dict(t=30,l=10,r=10,b=10), legend=dict(orientation="h"))
    chart.update_yaxes(title_text="River level (m)", secondary_y=False)
    chart.update_yaxes(title_text="Rainfall (mm)", secondary_y=True)
    st.plotly_chart(chart, use_container_width=True)

# ---------------- live sensors ----------------
with T[1]:
    st.subheader("📡 Live Sensor Network")
    st.write("This prototype simulates field sensors and stores every reading in a local SQLite database. In a real deployment, the same database writer can receive MQTT, serial, LoRaWAN or API readings.")
    c1, c2, c3 = st.columns([1,1,2])
    if c1.button("🔄 Generate sensor reading", type="primary", use_container_width=True):
        vals = database.simulate_from_conditions(latest)
        database.add_readings(vals, source="simulator")
        st.session_state.last_sensor_values = vals
        st.success("New sensor packet received and saved to database.")
        st.rerun()
    c2.metric("Sensors online", f"{database.sensor_count()[1]}/{database.sensor_count()[0]}")
    c3.info("Demo mode: values are generated from the selected forecast day's conditions with small sensor noise.")

    sensors = database.latest_readings()
    table = sensors[["sensor_id","name","metric","value","unit","location","timestamp","source"]].copy()
    table["value"] = table["value"].round(2)
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.subheader("Sensor telemetry")
    selected_sensor = st.selectbox("Choose a sensor", sensors.sensor_id.tolist(), format_func=lambda x: f"{x} — {sensors.loc[sensors.sensor_id.eq(x),'name'].iloc[0]}")
    h = database.history(selected_sensor, 168)
    if not h.empty:
        sensor_name = sensors.loc[sensors.sensor_id.eq(selected_sensor), "metric"].iloc[0]
        f = go.Figure(go.Scatter(x=pd.to_datetime(h.timestamp), y=h.value, mode="lines", name=sensor_name))
        f.update_layout(height=330, xaxis_title="Time", yaxis_title=sensor_name, margin=dict(t=30,l=10,r=10,b=10))
        st.plotly_chart(f, use_container_width=True)

    st.caption(f"Database file: `data/flood_monitor.db` · historical seed + simulated live readings")

# ---------------- forecast ----------------
with T[2]:
    st.subheader("🔮 AI Flood Forecast")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='sensor-card'><span class='small'>RISK LEVEL</span><h2>{alert['risk_level']}</h2></div>", unsafe_allow_html=True)
    c2.metric("Probability", f"{prob:.1%}")
    c3.metric("Temperature", f"{latest.temperature_c:.1f} °C")
    c4.metric("Humidity", f"{latest.humidity_pct:.1f} %")
    st.info(f"**Decision support:** {alert['advice']}")
    if st.button("📨 Dispatch alert", type="secondary"):
        if risk.dispatch(alert):
            database.log_alert(alert["risk_level"], prob, alert["advice"], "manual dispatch")
            st.success("Alert recorded in the database and queued for notification.")
        else:
            st.info("Below HIGH threshold — notification not dispatched.")

    fig = go.Figure()
    fig.add_scatter(x=hist.date, y=hist.river_level_m, name="River level (m)", line=dict(width=3))
    fig.add_hline(y=DANGER_LEVEL_M, line_dash="dash", annotation_text="Danger level")
    fig.add_bar(x=hist.date, y=hist.rainfall_mm, name="Rainfall (mm)", yaxis="y2", opacity=.35)
    fig.update_layout(height=390, yaxis2=dict(overlaying="y", side="right", autorange="reversed"), yaxis_title="Level (m)", margin=dict(t=30), legend=dict(orientation="h"))
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("Why did the model give this risk?")
    st.bar_chart(pd.Series(predict.explain(hist), name="Probability change"))

# ---------------- what-if ----------------
with T[3]:
    st.subheader("🧪 What-if Risk Simulator")
    st.write("Change sensor conditions and see how the AI prediction changes. This is useful for emergency planning and viva demonstration.")
    base = hist.iloc[-1]
    c1, c2, c3 = st.columns(3)
    rain = c1.slider("Rainfall today (mm)", 0.0, 300.0, float(base.rainfall_mm), 1.0)
    up = c1.slider("Upstream rainfall (mm)", 0.0, 300.0, float(base.upstream_rainfall_mm), 1.0)
    lvl = c2.slider("River level (m)", 0.5, 8.0, float(base.river_level_m), 0.05)
    soil = c2.slider("Soil moisture (0–1)", 0.0, 1.0, float(base.soil_moisture), 0.01)
    hum = c3.slider("Humidity (%)", 20.0, 100.0, float(base.humidity_pct), 1.0)
    h2 = hist.copy()
    h2.loc[h2.index[-1], ["rainfall_mm","upstream_rainfall_mm","river_level_m","soil_moisture","humidity_pct"]] = [rain,up,lvl,soil,hum]
    p2 = predict.predict_latest(h2); lv, col = risk.classify(p2)
    st.markdown(f"### Scenario risk: <span style='color:{col}'>{lv}</span> — {p2:.1%}", unsafe_allow_html=True)
    sweep = [predict.predict_latest(h2.assign(rainfall_mm=h2.rainfall_mm.where(h2.index != h2.index[-1], r))) for r in range(0, 201, 10)]
    f = go.Figure(go.Scatter(x=list(range(0,201,10)), y=sweep, mode="lines+markers")); f.update_layout(height=300, xaxis_title="Rainfall today (mm)", yaxis_title="P(flood)", title="Sensitivity to rainfall")
    st.plotly_chart(f, use_container_width=True)

# ---------------- history ----------------
with T[4]:
    st.subheader("📈 Historical Monitoring")
    years = sorted(df.date.dt.year.unique())
    yr = st.selectbox("Select year", years, index=len(years)-1)
    yd = df[df.date.dt.year.eq(yr)]
    c1, c2, c3 = st.columns(3)
    c1.metric("Days monitored", len(yd)); c2.metric("Maximum river level", f"{yd.river_level_m.max():.2f} m"); c3.metric("Maximum rainfall", f"{yd.rainfall_mm.max():.1f} mm")
    f = go.Figure()
    f.add_trace(go.Scatter(x=yd.date, y=yd.river_level_m, name="River level"))
    f.add_hline(y=DANGER_LEVEL_M, line_dash="dash", annotation_text="Danger level")
    f.update_layout(height=380, xaxis_title="Date", yaxis_title="River level (m)")
    st.plotly_chart(f, use_container_width=True)
    st.dataframe(yd[["date"]+RAW_FEATURES].tail(100).iloc[::-1].round(3), use_container_width=True, hide_index=True)

# ---------------- model ----------------
with T[5]:
    st.subheader("🤖 Machine Learning Model")
    m = bundle["metrics"]
    cols = st.columns(5)
    for col, (k, lab) in zip(cols, [("roc_auc","ROC-AUC"),("pr_auc","PR-AUC"),("recall","Recall"),("precision","Precision"),("f2","F2")]):
        col.metric(lab, f"{m[k]:.3f}")
    st.caption(f"Model: **{m['model']}** · calibrated · threshold {m['threshold']:.2f} · trained through {m['train_end']}")
    st.info(f"Baseline rule (river level > 5 m): recall {m['baseline_recall']:.2f}, precision {m['baseline_precision']:.2f}")
    st.image(str(RESULTS_DIR / "evaluation.png"))
    a1, a2 = st.columns(2); a1.image(str(RESULTS_DIR / "feature_importance.png")); a2.image(str(RESULTS_DIR / "timeline.png"))
    st.dataframe(pd.DataFrame(m["cv"]).round(3), use_container_width=True, hide_index=True)

# ---------------- alerts ----------------
with T[6]:
    st.subheader("🚨 Alert & Incident Log")
    al = database.alerts()
    if ALERT_LOG.exists():
        old = pd.read_csv(ALERT_LOG)
        st.caption("Legacy CSV alert log")
        st.dataframe(old.iloc[::-1], use_container_width=True, hide_index=True)
    if al.empty:
        st.info("No database alerts yet. Dispatch a HIGH/SEVERE alert from the Forecast tab to create one.")
    else:
        st.caption("SQLite database alert history")
        st.dataframe(al, use_container_width=True, hide_index=True)

# ---------------- admin ----------------
if st.session_state.role == "admin":
    with T[7]:
        st.subheader("⚙️ Administration")
        st.markdown("### User management")
        st.table(pd.Series(auth.list_users(), name="role"))
        n, pw, ro = st.columns(3)
        nu = n.text_input("New username")
        np_ = pw.text_input("Password", type="password")
        nr = ro.selectbox("Role", ["viewer", "admin"])
        if st.button("Add user") and nu and np_:
            auth.add_user(nu, np_, nr); st.success("User added"); st.rerun()
        du = st.selectbox("Delete user", [""] + [u for u in auth.list_users() if u != st.session_state.user])
        if du and st.button("Delete user"):
            auth.delete_user(du); st.rerun()

        st.divider(); st.markdown("### Data & model")
        up_f = st.file_uploader("Upload new raw CSV (same columns as data/flood_raw.csv)", type="csv")
        if up_f:
            RAW_CSV.write_bytes(up_f.getvalue()); st.success("Raw data replaced. Rebuild the dataset and model.")
        if st.button("🔁 Rebuild dataset & retrain model"):
            with st.spinner("Training…"):
                r = subprocess.run([sys.executable, "-m", "src.train"], cwd=ROOT, capture_output=True, text=True)
            predict.reload_bundle(); load_data.clear(); st.code(r.stdout[-2500:] or r.stderr[-2500:])
        st.divider(); st.markdown("### Database")
        st.info(f"SQLite database: `{database.DB_PATH.relative_to(ROOT)}`")
        if st.button("🗃️ Seed database from current dataset"):
            database.seed_from_dataset(load_data()); st.success("Database checked/seeded.")
