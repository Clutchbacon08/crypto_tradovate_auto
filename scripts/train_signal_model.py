from __future__ import annotations
import os
import joblib
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier

IN_PATH = "artifacts/trades.parquet"
OUT_PATH = "models/signal_model.joblib"

def main():
    df = pd.read_parquet(IN_PATH)

    # Example feature columns you should store in trades.parquet:
    feature_cols = [
        "ret_1","ret_5","atr_14",
        "ema_fast_minus_slow","ema200_dist",
        "vol_zscore","trend_strength",
        "stop_atr","tp_atr","cooldown"
    ]
    y_col = "label_win"  # 1 if TP before SL else 0

    df = df.dropna(subset=feature_cols + [y_col]).copy()

    X = df[feature_cols]
    y = df[y_col].astype(int)

    tscv = TimeSeriesSplit(n_splits=5)
    model = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05)

    aucs = []
    for train_idx, test_idx in tscv.split(X):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = model.predict_proba(X.iloc[test_idx])[:, 1]
        aucs.append(roc_auc_score(y.iloc[test_idx], proba))

    model.fit(X, y)
    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": model, "feature_cols": feature_cols}, OUT_PATH)

    print("Saved signal model:", OUT_PATH)
    print("CV AUC:", sum(aucs)/len(aucs))

if __name__ == "__main__":
    main()
