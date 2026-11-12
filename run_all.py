"""One command: generate data -> clean -> train -> evaluate -> demo alert."""
from src import data_generator, preprocess, train, predict, risk
import pandas as pd
from src.config import RAW_CSV, CLEAN_CSV

if not RAW_CSV.exists():
    data_generator.generate().to_csv(RAW_CSV, index=False)
_, report = preprocess.build_dataset(); print("Cleaning report:", report)
train.main()
d = pd.read_csv(CLEAN_CSV, parse_dates=["date"]); today = d.iloc[d.river_level_m.tail(1000).idxmax()]  # a high-water day
h = d[d.date <= today.date].tail(30)[["date"] + preprocess.RAW_FEATURES].reset_index(drop=True)
a = risk.build_alert(predict.predict_latest(h), today.date); print("Demo alert:", a); risk.dispatch(a)
print("\nNow launch the dashboard:  streamlit run app/dashboard.py")
