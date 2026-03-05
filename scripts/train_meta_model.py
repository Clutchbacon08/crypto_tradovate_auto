from __future__ import annotations
import os
import joblib
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor

IN_PATH = "artifacts/trials_with_folds.parquet"
OUT_PATH = "models/meta_model.joblib"

def main():
    df = pd.read_parquet(IN_PATH)

    # Features = strategy params + stability metrics
    feature_cols = [
        "ema_fast","ema_slow","stop_atr","tp_atr","cooldown","use_ema200",
        "mean_test_pnl","std_test_pnl","worst_fold_pnl","mean_pf","mean_dd",
        "dll_breaches","eod_breaches","trades_mean"
    ]
    # Target = robustness score (what Optuna optimized)
    y_col = "robust_score"

    df = df.dropna(subset=feature_cols + [y_col]).copy()

    X = df[feature_cols]
    y = df[y_col]

    # TimeSeriesSplit is safer than random split if your trials are ordered by time windows
    tscv = TimeSeriesSplit(n_splits=5)
    model = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05)

    scores = []
    for train_idx, test_idx in tscv.split(X):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[test_idx])
        rmse = mean_squared_error(y.iloc[test_idx], pred, squared=False)
        scores.append(rmse)

    model.fit(X, y)
    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": model, "feature_cols": feature_cols}, OUT_PATH)

    print("Saved meta-model:", OUT_PATH)
    print("CV RMSE:", sum(scores)/len(scores))

if __name__ == "__main__":
    main()
