# scripts/optimize_strategy.py
from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import optuna


# =========================================
# MBT (Micro Bitcoin Futures) backtest specs
# =========================================
# MBT tick size is $5; tick value is $0.50.
TICK_SIZE = float(os.getenv("TICK_SIZE", "5.0"))
TICK_VALUE = float(os.getenv("TICK_VALUE", "0.5"))
CONTRACT_MULTIPLIER = TICK_VALUE / TICK_SIZE  # $ per $1 BTC move per contract (0.10 for MBT)

FEE_PER_SIDE_USD = float(os.getenv("FEE_PER_SIDE_USD", "1.2"))  # entry or exit
SLIPPAGE_TICKS = float(os.getenv("SLIPPAGE_TICKS", "1.0"))       # in ticks
BAR_MINUTES = int(os.getenv("BAR_MINUTES", "15"))

# -------------------------
# Apex EOD Eval (50k) rules
# -------------------------
ACCOUNT_SIZE = float(os.getenv("ACCOUNT_SIZE", "50000"))
PROFIT_TARGET_USD = float(os.getenv("PROFIT_TARGET_USD", "3000"))
EOD_MAX_DRAWDOWN_USD = float(os.getenv("EOD_MAX_DRAWDOWN_USD", "2000"))
DAILY_LOSS_LIMIT_USD = float(os.getenv("DAILY_LOSS_LIMIT_USD", "1000"))

MAX_CONTRACTS = int(os.getenv("MAX_CONTRACTS", "6"))
CONTRACTS_PER_TRADE = int(os.getenv("CONTRACTS_PER_TRADE", "1"))  # fixed size during eval

STOP_TRADING_ON_TARGET = os.getenv("STOP_TRADING_ON_TARGET", "true").lower() == "true"

# Optimization runtime
DATA_CSV = os.getenv("HIST_CSV", "data/mbt_15m.csv")
N_TRIALS = int(os.getenv("N_TRIALS", "10000"))

# Walk-forward config (3–5 folds recommended)
N_FOLDS = int(os.getenv("N_FOLDS", "4"))
TRAIN_BARS = int(os.getenv("WF_TRAIN_BARS", "8000"))
TEST_BARS = int(os.getenv("WF_TEST_BARS", "2000"))
FOLD_STEP = int(os.getenv("WF_STEP_BARS", str(TEST_BARS)))  # step by one test window by default

# Dataset export options
SAVE_TRADE_SAMPLES = int(os.getenv("SAVE_TRADE_SAMPLES", "1"))  # 1 saves trades for ALL trials; set 0 to save only best trial later
MAX_TRADES_SAVED_PER_TRIAL = int(os.getenv("MAX_TRADES_SAVED_PER_TRIAL", "5000"))


# -------------------------
# Helpers
# -------------------------
def fee_roundtrip_usd() -> float:
    return 2.0 * FEE_PER_SIDE_USD

def slippage_price() -> float:
    return SLIPPAGE_TICKS * TICK_SIZE

def parse_ts_utc(ts: pd.Series) -> pd.DatetimeIndex:
    dt = pd.to_datetime(ts, utc=True, errors="coerce")
    if dt.isna().any():
        bad = dt.isna().sum()
        raise ValueError(f"Some timestamps could not be parsed ({bad}). Ensure ts_utc is valid ISO datetime.")
    return dt

def ema(arr: np.ndarray, span: int) -> np.ndarray:
    return pd.Series(arr).ewm(span=span, adjust=False).mean().to_numpy()

def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum.reduce([
        np.abs(high - low),
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])
    return pd.Series(tr).rolling(period).mean().bfill().to_numpy()

def zscore(x: np.ndarray, window: int = 96) -> np.ndarray:
    s = pd.Series(x)
    m = s.rolling(window).mean()
    sd = s.rolling(window).std().replace(0, np.nan)
    z = (s - m) / sd
    return z.fillna(0.0).to_numpy()


@dataclass
class FoldStats:
    net_pnl_usd: float
    max_dd_usd: float
    trades: int
    win_rate: float
    profit_factor: float
    dll_breaches: int
    eod_breaches: int

@dataclass
class TrialSummary:
    robust_score: float
    mean_test_pnl: float
    std_test_pnl: float
    worst_fold_pnl: float
    mean_pf: float
    mean_dd: float
    trades_mean: float
    dll_breaches: int
    eod_breaches: int


def build_walkforward_slices(n: int) -> List[Tuple[slice, slice]]:
    """
    Builds walk-forward folds like:
      Train: [start : start+TRAIN_BARS)
      Test:  [start+TRAIN_BARS : start+TRAIN_BARS+TEST_BARS)
    shifting start by FOLD_STEP each fold.
    """
    folds: List[Tuple[slice, slice]] = []
    start = 0
    for _ in range(N_FOLDS):
        train_end = start + TRAIN_BARS
        test_end = train_end + TEST_BARS
        if test_end > n:
            break
        folds.append((slice(start, train_end), slice(train_end, test_end)))
        start += FOLD_STEP
    if len(folds) < 2:
        raise ValueError(
            f"Not enough data for folds. n={n}, TRAIN_BARS={TRAIN_BARS}, TEST_BARS={TEST_BARS}, N_FOLDS={N_FOLDS}. "
            "Reduce TRAIN_BARS/TEST_BARS or provide more data."
        )
    return folds


def compute_features_arrays(close: np.ndarray, high: np.ndarray, low: np.ndarray, fast: int, slow: int, use_ema200: bool):
    ema_f = ema(close, fast)
    ema_s = ema(close, slow)
    ema_200 = ema(close, 200) if use_ema200 else None
    a = atr(high, low, close, 14)

    # extra ML features (signal-level) we can log:
    ret_1 = pd.Series(close).pct_change(1).fillna(0.0).to_numpy()
    ret_5 = pd.Series(close).pct_change(5).fillna(0.0).to_numpy()
    vol_z = zscore(pd.Series(ret_1).rolling(96).std().fillna(0.0).to_numpy(), window=96)  # rough vol regime
    trend_strength = (ema_f - ema_s) / np.maximum(a, 1e-9)

    return ema_f, ema_s, ema_200, a, ret_1, ret_5, vol_z, trend_strength


def simulate_fold(
    df: pd.DataFrame,
    dt: np.ndarray,
    ema_f: np.ndarray,
    ema_s: np.ndarray,
    ema_200: Optional[np.ndarray],
    a: np.ndarray,
    ret_1: np.ndarray,
    ret_5: np.ndarray,
    vol_z: np.ndarray,
    trend_strength: np.ndarray,
    stop_atr: float,
    tp_atr: float,
    cooldown: int,
    test_slice: slice,
    params_for_logging: Dict[str, Any],
    save_trades: bool,
) -> Tuple[FoldStats, List[Dict[str, Any]]]:
    """
    Run trading only inside test_slice. Indicators already computed on full series (ok for simplicity).
    Apex rules approximated using UTC day boundaries.
    """

    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)

    start_i = max(test_slice.start, 200)
    end_i = test_slice.stop

    position = 0  # 1 long, -1 short
    entry_price = 0.0
    stop_price = 0.0
    tp_price = 0.0
    cooldown_left = 0

    equity = ACCOUNT_SIZE
    peak_equity = equity
    max_dd = 0.0

    # Apex approximations by UTC date
    current_day = pd.Timestamp(dt[start_i]).date()
    day_start_equity = equity
    dll_floor = day_start_equity - DAILY_LOSS_LIMIT_USD

    eod_threshold_active = ACCOUNT_SIZE - EOD_MAX_DRAWDOWN_USD
    pending_next_eod_threshold: Optional[float] = None

    dll_breaches = 0
    eod_breaches = 0

    wins = 0
    losses = 0
    gross_win = 0.0
    gross_loss = 0.0
    trades = 0

    trades_rows: List[Dict[str, Any]] = []

    def update_drawdown():
        nonlocal peak_equity, max_dd
        peak_equity = max(peak_equity, equity)
        max_dd = max(max_dd, peak_equity - equity)

    for i in range(start_i, end_i):
        # day boundary handling (UTC approximation)
        this_day = pd.Timestamp(dt[i]).date()
        if this_day != current_day:
            # EOD close -> compute threshold for next session
            pending_next_eod_threshold = equity - EOD_MAX_DRAWDOWN_USD

            # new session
            current_day = this_day
            day_start_equity = equity
            dll_floor = day_start_equity - DAILY_LOSS_LIMIT_USD

            # activate new EOD threshold
            if pending_next_eod_threshold is not None:
                eod_threshold_active = pending_next_eod_threshold
                pending_next_eod_threshold = None

        update_drawdown()

        # optional stop on profit target
        if STOP_TRADING_ON_TARGET and equity >= ACCOUNT_SIZE + PROFIT_TARGET_USD:
            # stop trading: exit any open pos at close (worst-case)
            if position != 0:
                exit_price = close[i]
                slip = slippage_price()
                exit_price = exit_price - slip if position == 1 else exit_price + slip
                pnl_price = (exit_price - entry_price) * position
                pnl_usd = pnl_price * CONTRACT_MULTIPLIER * CONTRACTS_PER_TRADE
                pnl_usd -= fee_roundtrip_usd() * CONTRACTS_PER_TRADE
                equity += pnl_usd
                position = 0
            break

        # Apex enforcement
        if equity <= eod_threshold_active:
            eod_breaches += 1
            break

        if equity <= dll_floor:
            dll_breaches += 1
            break

        if cooldown_left > 0:
            cooldown_left -= 1

        # exit logic
        if position != 0:
            hit_stop = (low[i] <= stop_price) if position == 1 else (high[i] >= stop_price)
            hit_tp = (high[i] >= tp_price) if position == 1 else (low[i] <= tp_price)

            exit_price = None
            exit_reason = None
            if hit_stop and hit_tp:
                exit_price = stop_price
                exit_reason = "STOP_FIRST"
            elif hit_stop:
                exit_price = stop_price
                exit_reason = "STOP"
            elif hit_tp:
                exit_price = tp_price
                exit_reason = "TP"

            if exit_price is not None:
                slip = slippage_price()
                exit_price = exit_price - slip if position == 1 else exit_price + slip

                pnl_price = (exit_price - entry_price) * position
                pnl_usd = pnl_price * CONTRACT_MULTIPLIER * CONTRACTS_PER_TRADE
                pnl_usd -= fee_roundtrip_usd() * CONTRACTS_PER_TRADE

                equity += pnl_usd
                trades += 1

                if pnl_usd > 0:
                    wins += 1
                    gross_win += pnl_usd
                else:
                    losses += 1
                    gross_loss += abs(pnl_usd)

                # label for ML (win=1 if TP, else 0)
                if save_trades and len(trades_rows) < MAX_TRADES_SAVED_PER_TRIAL:
                    label_win = 1 if exit_reason == "TP" else 0
                    row = {
                        # features at entry time (stored when we entered)
                        **entry_feature_row,
                        # outcome
                        "label_win": label_win,
                        "pnl_usd": pnl_usd,
                        "exit_reason": exit_reason,
                    }
                    trades_rows.append(row)

                position = 0
                cooldown_left = cooldown

        # entry logic
        if position == 0 and cooldown_left == 0:
            cross_up = ema_f[i - 1] <= ema_s[i - 1] and ema_f[i] > ema_s[i]
            cross_dn = ema_f[i - 1] >= ema_s[i - 1] and ema_f[i] < ema_s[i]

            if ema_200 is not None:
                trend_ok_long = close[i] > ema_200[i]
                trend_ok_short = close[i] < ema_200[i]
                ema200_dist = close[i] - ema_200[i]
            else:
                trend_ok_long = True
                trend_ok_short = True
                ema200_dist = 0.0

            side = None
            if cross_up and trend_ok_long:
                side = 1
            elif cross_dn and trend_ok_short:
                side = -1

            if side is not None:
                position = side
                slip = slippage_price()
                entry_price = close[i] + slip if position == 1 else close[i] - slip

                dist = max(a[i], 1e-9)
                stop_dist = stop_atr * dist
                tp_dist = tp_atr * dist

                if position == 1:
                    stop_price = entry_price - stop_dist
                    tp_price = entry_price + tp_dist
                else:
                    stop_price = entry_price + stop_dist
                    tp_price = entry_price - tp_dist

                # store entry-time features for ML dataset
                entry_feature_row = {
                    **params_for_logging,
                    "ts_utc": str(df.iloc[i]["ts_utc"]),
                    "side": "BUY" if position == 1 else "SELL",
                    "ret_1": float(ret_1[i]),
                    "ret_5": float(ret_5[i]),
                    "atr_14": float(a[i]),
                    "ema_fast_minus_slow": float(ema_f[i] - ema_s[i]),
                    "ema200_dist": float(ema200_dist),
                    "vol_zscore": float(vol_z[i]),
                    "trend_strength": float(trend_strength[i]),
                }

    update_drawdown()
    net = equity - ACCOUNT_SIZE
    win_rate = (wins / trades) if trades else 0.0
    pf = (gross_win / gross_loss) if gross_loss > 0 else (gross_win if gross_win > 0 else 0.0)

    stats = FoldStats(
        net_pnl_usd=float(net),
        max_dd_usd=float(max_dd),
        trades=int(trades),
        win_rate=float(win_rate),
        profit_factor=float(pf),
        dll_breaches=int(dll_breaches),
        eod_breaches=int(eod_breaches),
    )
    return stats, trades_rows


def robust_score_from_folds(folds: List[FoldStats]) -> TrialSummary:
    test_pnls = np.array([f.net_pnl_usd for f in folds], dtype=float)
    test_dd = np.array([f.max_dd_usd for f in folds], dtype=float)
    pfs = np.array([f.profit_factor for f in folds], dtype=float)
    trades = np.array([f.trades for f in folds], dtype=float)

    dll_b = int(sum(f.dll_breaches for f in folds))
    eod_b = int(sum(f.eod_breaches for f in folds))

    mean_pnl = float(test_pnls.mean())
    std_pnl = float(test_pnls.std(ddof=0))
    worst_pnl = float(test_pnls.min())

    mean_pf = float(pfs.mean())
    mean_dd = float(test_dd.mean())
    trades_mean = float(trades.mean())

    # Hard rejects first:
    if trades_mean < 120:
        return TrialSummary(-1e9, mean_pnl, std_pnl, worst_pnl, mean_pf, mean_dd, trades_mean, dll_b, eod_b)

    if dll_b > 0 or eod_b > 0:
        return TrialSummary(-1e9, mean_pnl, std_pnl, worst_pnl, mean_pf, mean_dd, trades_mean, dll_b, eod_b)

    # Stability scoring (pro-style):
    # reward mean pnl and profit factor, penalize variability and drawdown and bad worst fold
    # You can tune weights later.
    dd_penalty = max(0.0, mean_dd - 0.75 * EOD_MAX_DRAWDOWN_USD) * 1.0
    var_penalty = std_pnl * 0.7
    worst_penalty = max(0.0, -worst_pnl) * 1.2
    pf_bonus = max(0.0, mean_pf - 1.0) * 400.0

    robust = mean_pnl + pf_bonus - dd_penalty - var_penalty - worst_penalty

    return TrialSummary(robust, mean_pnl, std_pnl, worst_pnl, mean_pf, mean_dd, trades_mean, dll_b, eod_b)


def main():
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    df = pd.read_csv(DATA_CSV).dropna().reset_index(drop=True)
    required = ["ts_utc", "open", "high", "low", "close"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"CSV must include column '{c}'")

    dt = parse_ts_utc(df["ts_utc"]).to_numpy()

    n = len(df)
    folds = build_walkforward_slices(n)

    # storage for ML datasets
    trial_rows: List[Dict[str, Any]] = []
    trade_rows_all: List[Dict[str, Any]] = []

    # --- objective
    def objective(trial: optuna.Trial) -> float:
        fast = trial.suggest_int("ema_fast", 5, 60)
        slow = trial.suggest_int("ema_slow", 20, 250)
        use_ema200 = trial.suggest_categorical("use_ema200", [True, False])
        stop_atr = trial.suggest_float("stop_atr", 0.8, 4.0)
        tp_atr = trial.suggest_float("tp_atr", 1.0, 8.0)
        cooldown = trial.suggest_int("cooldown", 0, 30)

        if slow <= fast + 5:
            return -1e9
        if CONTRACTS_PER_TRADE <= 0 or CONTRACTS_PER_TRADE > MAX_CONTRACTS:
            return -1e9

        close = df["close"].to_numpy(dtype=float)
        high = df["high"].to_numpy(dtype=float)
        low = df["low"].to_numpy(dtype=float)

        ema_f, ema_s, ema_200, a, ret_1, ret_5, vol_z, trend_strength = compute_features_arrays(
            close, high, low, fast, slow, use_ema200
        )

        params_for_logging = {
            "ema_fast": int(fast),
            "ema_slow": int(slow),
            "use_ema200": int(1 if use_ema200 else 0),
            "stop_atr": float(stop_atr),
            "tp_atr": float(tp_atr),
            "cooldown": int(cooldown),
        }

        fold_stats: List[FoldStats] = []
        local_trade_rows: List[Dict[str, Any]] = []

        for fold_id, (_train_slc, test_slc) in enumerate(folds):
            st, tr_rows = simulate_fold(
                df=df,
                dt=dt,
                ema_f=ema_f,
                ema_s=ema_s,
                ema_200=ema_200,
                a=a,
                ret_1=ret_1,
                ret_5=ret_5,
                vol_z=vol_z,
                trend_strength=trend_strength,
                stop_atr=float(stop_atr),
                tp_atr=float(tp_atr),
                cooldown=int(cooldown),
                test_slice=test_slc,
                params_for_logging={**params_for_logging, "fold_id": int(fold_id)},
                save_trades=(SAVE_TRADE_SAMPLES == 1),
            )
            fold_stats.append(st)
            if SAVE_TRADE_SAMPLES == 1 and tr_rows:
                local_trade_rows.extend(tr_rows)

        summary = robust_score_from_folds(fold_stats)

        # persist trial row for meta-model dataset
        row = {
            **params_for_logging,
            "robust_score": float(summary.robust_score),
            "mean_test_pnl": float(summary.mean_test_pnl),
            "std_test_pnl": float(summary.std_test_pnl),
            "worst_fold_pnl": float(summary.worst_fold_pnl),
            "mean_pf": float(summary.mean_pf),
            "mean_dd": float(summary.mean_dd),
            "trades_mean": float(summary.trades_mean),
            "dll_breaches": int(summary.dll_breaches),
            "eod_breaches": int(summary.eod_breaches),
        }

        # add fold metrics flattened
        for i, fs in enumerate(fold_stats):
            row[f"fold{i}_pnl"] = float(fs.net_pnl_usd)
            row[f"fold{i}_dd"] = float(fs.max_dd_usd)
            row[f"fold{i}_trades"] = int(fs.trades)
            row[f"fold{i}_pf"] = float(fs.profit_factor)

        trial_rows.append(row)

        if SAVE_TRADE_SAMPLES == 1 and local_trade_rows:
            # include trial number
            for tr in local_trade_rows:
                tr["trial_number"] = int(trial.number)
            trade_rows_all.extend(local_trade_rows)

        # prune bad trials early
        if summary.robust_score < -1e8:
            raise optuna.TrialPruned()

        return float(summary.robust_score)

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best = study.best_trial
    print("BEST SCORE:", best.value)
    print("BEST PARAMS:", best.params)

    # Export optuna summary table
    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    trials_df = trials_df.sort_values("value", ascending=False)
    trials_df.head(300).to_csv("artifacts/optuna_best.csv", index=False)

    # Export best params
    with open("artifacts/best_params.json", "w", encoding="utf-8") as f:
        json.dump(best.params, f, indent=2)

    # Export ML datasets
    trials_full = pd.DataFrame(trial_rows)
    trials_full.to_parquet("artifacts/trials_with_folds.parquet", index=False)
    print("Saved:", "artifacts/trials_with_folds.parquet")

    if SAVE_TRADE_SAMPLES == 1 and len(trade_rows_all) > 0:
        trades_df = pd.DataFrame(trade_rows_all)
        trades_df.to_parquet("artifacts/trades.parquet", index=False)
        print("Saved:", "artifacts/trades.parquet")
    else:
        print("Skipped saving trades.parquet (SAVE_TRADE_SAMPLES=0 or no trades).")

    # Export fold metrics for best trial (from trials_full lookup)
    # Find the row that matches best params (first match)
    best_row = None
    for r in trial_rows[::-1]:
        ok = True
        for k, v in best.params.items():
            if k == "use_ema200":
                vv = 1 if v else 0
                if int(r.get("use_ema200")) != int(vv):
                    ok = False
                    break
            else:
                if str(r.get(k)) != str(v):
                    ok = False
                    break
        if ok:
            best_row = r
            break

    if best_row is not None:
        with open("artifacts/fold_metrics_best.json", "w", encoding="utf-8") as f:
            json.dump(best_row, f, indent=2)
        print("Saved:", "artifacts/fold_metrics_best.json")

    print("Done.")


if __name__ == "__main__":
    main()
