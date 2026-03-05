from __future__ import annotations

import os
import joblib
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier


IN_PATH = os.getenv("TRADES_PATH", "artifacts/trades.parquet")
OUT_PATH = os.getenv("SIGNAL_MODEL_OUT", "models/signal_model.joblib")

# Meta-labeling knobs
META_LABEL_MIN_R = float(os.getenv("META_LABEL_MIN_R", "0.2"))   # require at least +0.2R to count as "good"
N_SPLITS = int(os.getenv("ML_CV_SPLITS", "5"))


def _find_first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _compute_r_multiple(df: pd.DataFrame) -> pd.Series | None:
    """
    Try to compute/derive R-multiple for each trade using common column conventions.

    Priority:
    1) existing r-multiple columns (r_multiple, r_mult, pnl_r, R, r)
    2) pnl / risk columns
    3) entry/exit/stop + side

    Returns a Series of floats or None if not computable.
    """
    # 1) Direct columns
    direct = _find_first_col(df, ["r_multiple", "r_mult", "pnl_r", "R", "r"])
    if direct:
        s = pd.to_numeric(df[direct], errors="coerce")
        return s

    # 2) pnl / risk columns
    pnl_col = _find_first_col(df, ["pnl", "pnl_usd", "profit", "net_pnl", "realized_pnl"])
    risk_col = _find_first_col(df, ["risk_usd", "risk", "risk_amount"])
    if pnl_col and risk_col:
        pnl = pd.to_numeric(df[pnl_col], errors="coerce")
        risk = pd.to_numeric(df[risk_col], errors="coerce")
        r = pnl / risk.replace(0, pd.NA)
        return r.astype(float)

    # 3) entry/exit/stop + side
    entry_col = _find_first_col(df, ["entry_price", "entry", "open_price", "price_entry"])
    exit_col = _find_first_col(df, ["exit_price", "exit", "close_price", "price_exit"])
    stop_col = _find_first_col(df, ["stop_price", "stop_loss_price", "sl_price", "stop"])
    side_col = _find_first_col(df, ["side", "direction"])

    if entry_col and exit_col and stop_col and side_col:
        entry = pd.to_numeric(df[entry_col], errors="coerce")
        exitp = pd.to_numeric(df[exit_col], errors="coerce")
        stop = pd.to_numeric(df[stop_col], errors="coerce")
        side = df[side_col].astype(str).str.upper()

        # risk in price terms
        risk_pts_long = (entry - stop)
        risk_pts_short = (stop - entry)
        risk_pts = risk_pts_long.where(side.str.contains("BUY|LONG"), risk_pts_short)
        risk_pts = risk_pts.replace(0, pd.NA)

        # pnl in price terms
        pnl_pts_long = (exitp - entry)
        pnl_pts_short = (entry - exitp)
        pnl_pts = pnl_pts_long.where(side.str.contains("BUY|LONG"), pnl_pts_short)

        r = pnl_pts / risk_pts
        return r.astype(float)

    return None


def main():
    df = pd.read_parquet(IN_PATH)

    # MUST match infer.py
    feature_cols = [
        "ret_1", "ret_5", "atr_14",
        "ema_fast_minus_slow", "ema200_dist",
        "vol_zscore", "trend_strength",
        "stop_atr", "tp_atr", "cooldown"
    ]

    # Base label produced by your backtest loop: 1 if TP before SL else 0
    y_base_col = "label_win"
    if y_base_col not in df.columns:
        raise SystemExit(f"Expected '{y_base_col}' in {IN_PATH}. Add it in optimize_strategy trade rows.")

    # Clean rows
    df = df.dropna(subset=feature_cols + [y_base_col]).copy()

    # --- META LABELING ---
    # We only label "good trades" as positive:
    #   good = win AND (r_multiple >= META_LABEL_MIN_R) if r_multiple available
    r_mult = _compute_r_multiple(df)
    if r_mult is not None:
        df["r_multiple"] = r_mult
        df["label_meta"] = ((df[y_base_col].astype(int) == 1) & (df["r_multiple"] >= META_LABEL_MIN_R)).astype(int)
        label_used = f"label_meta(win & r>= {META_LABEL_MIN_R})"
    else:
        # fallback if we cannot compute R
        df["label_meta"] = (df[y_base_col].astype(int) == 1).astype(int)
        label_used = "label_meta(fallback=label_win)"

    # Optionally: drop rows with unknown r_multiple if we computed it
    if "r_multiple" in df.columns:
        df = df.dropna(subset=["r_multiple"]).copy()

    X = df[feature_cols]
    y = df["label_meta"].astype(int)

    # Basic sanity
    pos_rate = float(y.mean()) if len(y) else 0.0
    if pos_rate < 0.01 or pos_rate > 0.99:
        print(f"⚠️ label positive rate looks extreme: {pos_rate:.3f}. "
              f"Consider adjusting META_LABEL_MIN_R or your trade labeling.")

    # Model
    model = HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.05,
        max_bins=255,
        random_state=42,
    )

    # Time series CV
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    aucs = []
    for train_idx, test_idx in tscv.split(X):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = model.predict_proba(X.iloc[test_idx])[:, 1]
        try:
            auc = roc_auc_score(y.iloc[test_idx], proba)
            aucs.append(float(auc))
        except Exception:
            # Can happen if a fold has only one class
            pass

    # Fit full
    model.fit(X, y)

    os.makedirs("models", exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_cols": feature_cols,
            "label_used": label_used,
            "meta_label_min_r": META_LABEL_MIN_R,
        },
        OUT_PATH
    )

    print("Saved signal model:", OUT_PATH)
    if aucs:
        print("CV AUC:", sum(aucs) / len(aucs))
    else:
        print("CV AUC: n/a (some folds had single class)")
    print("Label used:", label_used)
    print("Rows:", len(df), "Positive rate:", pos_rate)


if __name__ == "__main__":
    main()
