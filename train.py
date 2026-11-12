"""Train, compare, tune and evaluate flood models (leak-free, chronological)."""
import json, joblib, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_recall_curve,
                             roc_curve, confusion_matrix, fbeta_score, brier_score_loss,
                             classification_report, ConfusionMatrixDisplay)
from .config import *
from .preprocess import FEATURES, build_dataset

CANDIDATES = {
    "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced")),
    "RandomForest": RandomForestClassifier(n_estimators=120, min_samples_leaf=3, class_weight="balanced_subsample",
                                           n_jobs=-1, random_state=RANDOM_STATE),
    "GradientBoosting": HistGradientBoostingClassifier(learning_rate=0.05, max_depth=4, max_iter=120,
                                                       class_weight="balanced", random_state=RANDOM_STATE),
}


def best_threshold(y, p, beta=2.0):
    """F-beta (beta=2 favours recall: a missed flood costs more than a false alarm)."""
    ts = np.linspace(0.05, 0.95, 91)
    return float(ts[int(np.argmax([fbeta_score(y, p >= t, beta=beta, zero_division=0) for t in ts]))])


def main():
    RESULTS_DIR.mkdir(exist_ok=True); MODEL_DIR.mkdir(exist_ok=True)
    df, _ = build_dataset()
    cut = df["date"].max() - pd.DateOffset(years=TEST_YEARS)
    train, test = df[df.date <= cut], df[df.date > cut]
    X, y, Xt, yt = train[FEATURES], train.flood_next_day.astype(int), test[FEATURES], test.flood_next_day.astype(int)
    print(f"Train {len(train)} rows ({y.mean():.1%} flood) | Test {len(test)} rows ({yt.mean():.1%} flood)")

    cv, rows = TimeSeriesSplit(n_splits=3), []
    for name, m in CANDIDATES.items():                          # time-aware cross-validation
        s = cross_val_score(m, X, y, cv=cv, scoring="average_precision", n_jobs=1)
        rows.append({"model": name, "cv_pr_auc": s.mean(), "cv_std": s.std()}); print(f"  CV {name}: {s.mean():.3f} ± {s.std():.3f}")
    best = max(rows, key=lambda r: r["cv_pr_auc"])["model"]; print("Selected:", best)

    # hold last 20% of train for threshold selection, then refit on all train with calibration
    k = int(len(X) * 0.8)
    val_model = CANDIDATES[best].fit(X.iloc[:k], y.iloc[:k])
    thr = best_threshold(y.iloc[k:], val_model.predict_proba(X.iloc[k:])[:, 1])
    final = CalibratedClassifierCV(CANDIDATES[best], method="isotonic", cv=TimeSeriesSplit(n_splits=3)).fit(X, y)

    p = final.predict_proba(Xt)[:, 1]; pred = (p >= thr).astype(int); tn, fp, fn, tp = confusion_matrix(yt, pred).ravel()
    metrics = {"model": best, "threshold": thr, "roc_auc": roc_auc_score(yt, p), "pr_auc": average_precision_score(yt, p),
               "brier": brier_score_loss(yt, p), "recall": tp / (tp + fn), "precision": tp / max(tp + fp, 1),
               "f2": fbeta_score(yt, pred, beta=2), "false_alarms": int(fp), "missed_floods": int(fn),
               "tp": int(tp), "tn": int(tn), "train_end": str(cut.date()), "cv": rows}
    # baseline: "river already > 5 m => flood tomorrow" persistence rule
    base = (Xt["river_level_m"] > 5.0).astype(int); metrics["baseline_recall"] = float((base[yt == 1]).mean())
    metrics["baseline_precision"] = float(yt[base == 1].mean()) if base.sum() else 0.0
    print(classification_report(yt, pred, target_names=["No flood", "Flood"], digits=3))
    json.dump(metrics, open(RESULTS_DIR / "metrics.json", "w"), indent=2, default=float)
    joblib.dump({"model": final, "features": FEATURES, "threshold": thr, "metrics": metrics}, MODEL_PATH)
    test.assign(prob=p).to_csv(RESULTS_DIR / "test_predictions.csv", index=False)

    # ---------- plots ----------
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    fpr, tpr, _ = roc_curve(yt, p); ax[0].plot(fpr, tpr, lw=2); ax[0].plot([0, 1], [0, 1], "--", c="gray")
    ax[0].set(title=f"ROC (AUC={metrics['roc_auc']:.3f})", xlabel="False positive rate", ylabel="Recall")
    pr, rc, _ = precision_recall_curve(yt, p); ax[1].plot(rc, pr, lw=2); ax[1].axhline(yt.mean(), ls="--", c="gray")
    ax[1].set(title=f"Precision-Recall (AP={metrics['pr_auc']:.3f})", xlabel="Recall", ylabel="Precision")
    ConfusionMatrixDisplay(confusion_matrix(yt, pred), display_labels=["No flood", "Flood"]).plot(ax=ax[2], colorbar=False)
    ax[2].set_title(f"Confusion matrix @ threshold {thr:.2f}"); plt.tight_layout(); plt.savefig(RESULTS_DIR / "evaluation.png", dpi=130); plt.close()

    pi = permutation_importance(final, Xt, yt, scoring="average_precision", n_repeats=5, random_state=RANDOM_STATE, n_jobs=1)
    imp = pd.Series(pi.importances_mean, FEATURES).sort_values().tail(12)
    imp.plot.barh(figsize=(7, 5), color="#1565c0", title="Permutation importance (test PR-AUC drop)")
    plt.tight_layout(); plt.savefig(RESULTS_DIR / "feature_importance.png", dpi=130); plt.close()
    imp[::-1].to_csv(RESULTS_DIR / "feature_importance.csv", header=["importance"])

    t = test.assign(prob=p).tail(365); fig, a = plt.subplots(figsize=(14, 4)); a.plot(t.date, t.river_level_m, c="#1565c0")
    a.axhline(DANGER_LEVEL_M, c="red", ls="--", label="Danger level"); a2 = a.twinx(); a2.fill_between(t.date, t.prob, color="orange", alpha=.4)
    a.set(title="Last 365 days: river level (blue) vs predicted flood probability (orange)", ylabel="River level (m)"); a2.set_ylabel("P(flood tomorrow)")
    a.legend(loc="upper left"); plt.tight_layout(); plt.savefig(RESULTS_DIR / "timeline.png", dpi=130); plt.close()
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in metrics.items() if k != "cv"}, indent=1))


if __name__ == "__main__":
    main()
