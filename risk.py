"""Risk classification + alert generation."""
import csv, datetime as dt
from .config import RISK_BANDS, ALERT_LOG

ADVICE = {"LOW": "No action needed. Routine monitoring.",
          "MODERATE": "Stay alert. Monitor river gauge and weather bulletins.",
          "HIGH": "Prepare to move valuables and livestock. Avoid riverbanks and low-lying roads.",
          "SEVERE": "Evacuate low-lying areas now. Follow district control-room instructions."}


def classify(prob: float):
    name, color = RISK_BANDS[0][1], RISK_BANDS[0][2]
    for lo, n, c in RISK_BANDS:
        if prob >= lo: name, color = n, c
    return name, color


def build_alert(prob, date=None, location="Rispana River Basin, Dehradun"):
    level, color = classify(prob)
    return {"timestamp": (date or dt.datetime.now()).strftime("%Y-%m-%d %H:%M"), "location": location,
            "probability": round(float(prob), 3), "risk_level": level, "color": color,
            "advice": ADVICE[level]}


def dispatch(alert: dict, min_level="HIGH", log_path=ALERT_LOG):
    """Log every HIGH/SEVERE alert; SMS/e-mail hooks are stubbed (plug in Twilio/SMTP here)."""
    order = [b[1] for b in RISK_BANDS]
    if order.index(alert["risk_level"]) < order.index(min_level): return False
    new = not log_path.exists(); log_path.parent.mkdir(exist_ok=True)
    with open(log_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in alert if k != "color"])
        if new: w.writeheader()
        w.writerow({k: v for k, v in alert.items() if k != "color"})
    return True
