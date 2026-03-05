# scripts/optimize_strategy.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import optuna


# =========================================
# MBT (Micro Bitcoin Futures) backtest specs
# =========================================
# MBT tick size is $5; tick value is $0.50.
TICK_SIZE = float(os.getenv("TICK_SIZE", "5.0"))
TICK_VALUE = float(os.getenv("TICK_VALUE", "0.5"))

# Convert BTCUSD price move ($1) -> PnL USD per contract
# $5 move => $0.50, so $1 move => $0.10
CONTRACT_MULTIPLIER = TICK_VALUE / TICK_SIZE  # 0.10 for MBT

# Costs (rough). You can tune later once you see real fills.
FEE_PER_SIDE_USD = float(os.getenv("FEE_PER_SIDE_USD", "1.2"))  # entry or exit
SLIPPAGE_TICKS = float(os.getenv("SLIPPAGE_TICKS", "1.0"))       # in ticks


# -------------------------
# Apex EOD Eval (50k) rules
# -------------------------
ACCOUNT_SIZE = float(os.getenv("ACCOUNT_SIZE", "50000"))
PROFIT_TARGET_USD = float(os.getenv("PROFIT_TARGET_USD", "3000"))
EOD_MAX_DRAWDOWN_USD = float(os.getenv("EOD_MAX_DRAWDOWN_USD", "2000"))
DAILY_LOSS_LIMIT_USD = float(os.getenv("DAILY_LOSS_LIMIT_USD", "1000"))

MAX_CONTRACTS = int(os.getenv("MAX_CONTRACTS", "6"))
CONTRACTS_PER_TRADE = int(os.getenv("CONTRACTS_PER_TRADE", "1"))  # fixed size during eval

# Optimization runtime
DATA_CSV = os.getenv("HIST_CSV", "data/mbt_15m.csv")
N_TRIALS = int(os.getenv("N_TRIALS", "10000"))

# Validation split
TRAIN_FRAC = float(os.getenv("TRAIN_FRAC", "0.6"))  # optimize on train, score on test

# We assume 15-minute bars
BAR_MINUTES = int(os.getenv("BAR_MINUTES", "15"))


# -------------------------
# Helpers
# -------------------------
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


def parse_ts_utc(ts: pd.Series) -> pd.DatetimeIndex:
    # expects ISO strings like 2025-01-01T12:00:00+00:00 or without tz
    dt = pd.to_datetime(ts, utc=True, errors="coerce")
    if dt.isna().any():
        raise ValueError("Some timestamps could not be parsed. Ensure ts_utc is valid ISO datetime.")
    return dt


def fee_roundtrip_usd() -> float:
    return 2.0 * FEE_PER_SIDE_USD


def slippage_price() -> float:
    return SLIPPAGE_TICKS * TICK_SIZE


@dataclass
class Stats:
    net_pnl_usd: float
    max_dd_usd: float
    trades: int
    win_rate: float
    profit_factor: float
    dll_breaches: int
    eod_breaches: int


def simulate_strategy_15m_apex(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    dt: np.ndarray,  # numpy datetime64[ns] or pandas datetime
    fast: int,
    slow: int,
    use_ema200: bool,
    stop_atr: float,
    tp_atr: float,
    cooldown: int,
) -> Stats:
    """
    Minimal but Apex-aware simulation:
    - One position at a time
    - ATR-based stop/TP
    - Fixed contracts per trade
    - Tracks:
        * max drawdown in USD
        * daily loss limit breaches (approx by "day" in UTC)
        * EOD drawdown breaches (approx by "day" in UTC; threshold updates daily)
    Notes:
    - Apex EOD threshold is set once/day at market close and enforced next session.
      Without an exchange calendar here, we approximate EOD using UTC date boundaries.
      On VPS we can refine with CME session calendar later.
    """

    n = len(close)
    if slow <= fast + 5:
        return Stats(-1e9, 1e9, 0, 0.0, 0.0, 0, 0)

    if CONTRACTS_PER_TRADE <= 0 or CONTRACTS_PER_TRADE > MAX_CONTRACTS:
        return Stats(-1e9, 1e9, 0, 0.0, 0.0, 0, 0)

    ema_f = ema(close, fast)
    ema_s = ema(close, slow)
    ema_200 = ema(close, 200) if use_ema200 else None
    a = atr(high, low, close, 14)

    position = 0  # 1 long, -1 short
    entry_price = 0.0
    stop_price = 0.0
    tp_price = 0.0
    cooldown_left = 0

    equity = ACCOUNT_SIZE
    peak_equity = equity
    max_dd = 0.0

    # Daily accounting (UTC-day approximation)
    current_day = pd.Timestamp(dt[0]).date()
    day_start_equity = equity
    dll_floor = day_start_equity - DAILY_LOSS_LIMIT_USD

    # EOD threshold accounting (EOD model)
    # threshold is set from prior day's EOD balance; for day 1 we seed from starting equity
    eod_threshold_active = ACCOUNT_SIZE - EOD_MAX_DRAWDOWN_USD
    # At each day close, we compute next day's threshold from today's EOD equity
    pending_next_eod_threshold: Optional[float] = None

    dll_breaches = 0
    eod_breaches = 0

    wins = 0
    losses = 0
    gross_win = 0.0
    gross_loss = 0.0
    trades = 0

    def update_drawdown():
        nonlocal peak_equity, max_dd
        peak_equity = max(peak_equity, equity)
        max_dd = max(max_dd, peak_equity - equity)

    for i in range(200, n):  # warmup
        # Handle day boundary (UTC approximation)
        this_day = pd.Timestamp(dt[i]).date()
        if this_day != current_day:
            # "EOD close": compute threshold for next session/day
            pending_next_eod_threshold = equity - EOD_MAX_DRAWDOWN_USD

            # start new day/session
            current_day = this_day
            day_start_equity = equity
            dll_floor = day_start_equity - DAILY_LOSS_LIMIT_USD

            # Activate pending EOD threshold for the new day
            if pending_next_eod_threshold is not None:
                eod_threshold_active = pending_next_eod_threshold
                pending_next_eod_threshold = None

        update_drawdown()

        # Apex enforcement (in-session)
        if equity <= eod_threshold_active:
            eod_breaches += 1
            # Fail hard: end simulation (worst-case)
            return Stats(equity - ACCOUNT_SIZE, max_dd, trades, (wins / trades) if trades else 0.0,
                         (gross_win / gross_loss) if gross_loss > 0 else (gross_win if gross_win > 0 else 0.0),
                         dll_breaches, eod_breaches)

        if equity <= dll_floor:
            dll_breaches += 1
            return Stats(equity - ACCOUNT_SIZE, max_dd, trades, (wins / trades) if trades else 0.0,
                         (gross_win / gross_loss) if gross_loss > 0 else (gross_win if gross_win > 0 else 0.0),
                         dll_breaches, eod_breaches)

        if cooldown_left > 0:
            cooldown_left -= 1

        # Exit logic
        if position != 0:
            hit_stop = (low[i] <= stop_price) if position == 1 else (high[i] >= stop_price)
            hit_tp = (high[i] >= tp_price) if position == 1 else (low[i] <= tp_price)

            exit_price = None
            if hit_stop and hit_tp:
                exit_price = stop_price  # worst-case
            elif hit_stop:
                exit_price = stop_price
            elif hit_tp:
                exit_price = tp_price

            if exit_price is not None:
                # slippage: worse fill
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

                position = 0
                cooldown_left = cooldown

        # Entry logic
        if position == 0 and cooldown_left == 0:
            cross_up = ema_f[i - 1] <= ema_s[i - 1] and ema_f[i] > ema_s[i]
            cross_dn = ema_f[i - 1] >= ema_s[i - 1] and ema_f[i] < ema_s[i]

            if use_ema200:
                trend_ok_long = close[i] > ema_200[i]
                trend_ok_short = close[i] < ema_200[i]
            else:
                trend_ok_long = True
                trend_ok_short = True

            if cross_up and trend_ok_long:
                position = 1
            elif cross_dn and trend_ok_short:
                position = -1

            if position != 0:
                slip = slippage_price()
                entry_price = close[i] + slip if position == 1 else close[i] - slip

                dist = max(a[i], 1e-6)
                stop_dist = stop_atr * dist
                tp_dist = tp_atr * dist

                if position == 1:
                    stop_price = entry_price - stop_dist
                    tp_price = entry_price + tp_dist
                else:
                    stop_price = entry_price + stop_dist
                    tp_price = entry_price - tp_dist

    update_drawdown()
    net = equity - ACCOUNT_SIZE
    win_rate = (wins / trades) if trades else 0.0
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (gross_win if gross_win > 0 else 0.0)

    return Stats(net, max_dd, trades, win_rate, profit_factor, dll_breaches, eod_breaches)


def score_apex(stats: Stats) -> float:
    """
    Optuna objective score:
    - hard reject: DLL/EOD breaches, too few trades
    - reward: net pnl + profit factor bonus
    - penalize: drawdown
    """
    if stats.trades < 150:
        return -1e9
    if stats.dll_breaches > 0 or stats.eod_breaches > 0:
        return -1e9

    # Penalize high DD (even if not breached). For Apex 50k EOD max DD is 2000.
    dd_penalty = 1.0 * max(0.0, stats.max_dd_usd - 0.75 * EOD_MAX_DRAWDOWN_USD)

    # Reward PF above 1.0
    pf_bonus = 300.0 * max(0.0, stats.profit_factor - 1.0)

    return stats.net_pnl_usd + pf_bonus - dd_penalty


def main():
    os.makedirs("artifacts", exist_ok=True)

    df = pd.read_csv(DATA_CSV).dropna().reset_index(drop=True)
    if "ts_utc" not in df.columns:
        raise ValueError("CSV must include ts_utc column.")
    for c in ["open", "high", "low", "close"]:
        if c not in df.columns:
            raise ValueError(f"CSV must include {c} column.")

    dt = parse_ts_utc(df["ts_utc"]).to_numpy()

    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)

    split = int(len(df) * TRAIN_FRAC)

    # Train (search) window
    close_tr, high_tr, low_tr, dt_tr = close[:split], high[:split], low[:split], dt[:split]
    # Test (score) window
    close_te, high_te, low_te, dt_te = close[split:], high[split:], low[split:], dt[split:]

    def objective(trial: optuna.Trial) -> float:
        fast = trial.suggest_int("ema_fast", 5, 60)
        slow = trial.suggest_int("ema_slow", 20, 250)
        use_ema200 = trial.suggest_categorical("use_ema200", [True, False])
        stop_atr = trial.suggest_float("stop_atr", 0.8, 4.0)
        tp_atr = trial.suggest_float("tp_atr", 1.0, 8.0)
        cooldown = trial.suggest_int("cooldown", 0, 30)

        if slow <= fast + 5:
            return -1e9

        # Fast pre-check on train (reject obvious bad ones)
        st_tr = simulate_strategy_15m_apex(close_tr, high_tr, low_tr, dt_tr, fast, slow, use_ema200, stop_atr, tp_atr, cooldown)
        s_tr = score_apex(st_tr)
        if s_tr < -1e8:
            return s_tr

        # Score on test (out-of-sample)
        st_te = simulate_strategy_15m_apex(close_te, high_te, low_te, dt_te, fast, slow, use_ema200, stop_atr, tp_atr, cooldown)
        return score_apex(st_te)

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    best = study.best_trial
    print("BEST SCORE:", best.value)
    print("BEST PARAMS:", best.params)

    trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    trials_df = trials_df.sort_values("value", ascending=False).head(300)
    out = "artifacts/optuna_best.csv"
    trials_df.to_csv(out, index=False)
    print(f"Saved top trials -> {out}")

    # Save best params as JSON too
    best_json = "artifacts/best_params.json"
    pd.Series(best.params).to_json(best_json)
    print(f"Saved best params -> {best_json}")


if __name__ == "__main__":
    main()
