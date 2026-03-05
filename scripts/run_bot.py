# scripts/run_bot.py
from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import pandas as pd

from src.common.settings import settings
from src.common.schemas import OrderCommand, Side
from src.data.feature_engineering import compute_features
from src.strategy.baseline_rules import propose_trade
from src.ml.infer import ml_filter
from src.risk.guardian import ApexEODState, evaluate_risk
from src.monitoring.health import heartbeat


def main():
    """
    True OFFLINE paper trader:
    - Reads candles from data/mbt_15m.csv
    - Computes features
    - Strategy proposes intents
    - ML filter approves/vetoes
    - Risk guard approves/blocks/kills
    - Simulates fills by logging OrderCommands (no broker calls)
    - Writes artifacts/paper_trades.csv
    """

    csv_path = Path(os.getenv("HIST_CSV", "data/mbt_15m.csv"))
    if not csv_path.exists():
        raise SystemExit(f"Missing {csv_path}. Run: python scripts/fetch_free_data.py")

    df_all = pd.read_csv(csv_path).dropna().reset_index(drop=True)
    required = {"ts_utc", "open", "high", "low", "close"}
    missing = required - set(df_all.columns)
    if missing:
        raise SystemExit(f"CSV missing columns: {missing}")

    # Apex state (offline sim: balance updates when we close a trade)
    state = ApexEODState(
        starting_balance=settings.ACCOUNT_SIZE,
        session_start_balance=settings.ACCOUNT_SIZE,
        current_balance=settings.ACCOUNT_SIZE,
        eod_threshold_active=settings.ACCOUNT_SIZE - settings.EOD_MAX_DRAWDOWN_USD,
        dll_floor_active=settings.ACCOUNT_SIZE - settings.DAILY_LOSS_LIMIT_USD,
        open_contracts=0,
    )

    # daily trade limiter
    max_trades_per_day = int(os.getenv("MAX_TRADES_PER_DAY", "20"))
    trade_day = None
    trades_today = 0

    # simple position sim (1 position at a time)
    position_side: Side | None = None
    entry_price = 0.0
    stop_price = 0.0
    tp_price = 0.0
    qty = 0
    entry_ts = None

    paper_rows = []

    # warmup window for indicators
    window = 300
    step_sleep = float(os.getenv("PAPER_STEP_SLEEP", "0"))  # set >0 if you want it slower

    for i in range(window, len(df_all)):
        chunk = df_all.iloc[i - window : i].copy()
        feats = compute_features(chunk, symbol=settings.BOT_SYMBOL, timeframe_min=settings.TIMEFRAME_MINUTES)

        # reset daily trade counter
        day = pd.to_datetime(feats.ts_utc).date()
        if trade_day != day:
            trade_day = day
            trades_today = 0

        last = float(feats.last_price)

        # --- manage open position: stop/tp check on latest bar ---
        # Use last bar high/low for intrabar hit approximation
        bar = df_all.iloc[i - 1]
        high = float(bar["high"])
        low = float(bar["low"])

        if position_side is not None:
            hit_stop = (low <= stop_price) if position_side == Side.BUY else (high >= stop_price)
            hit_tp = (high >= tp_price) if position_side == Side.BUY else (low <= tp_price)

            exit_reason = None
            exit_price = None

            if hit_stop and hit_tp:
                exit_reason = "STOP_FIRST"
                exit_price = stop_price
            elif hit_stop:
                exit_reason = "STOP"
                exit_price = stop_price
            elif hit_tp:
                exit_reason = "TP"
                exit_price = tp_price

            if exit_reason and exit_price is not None:
                # PnL in “price points”; convert to USD using MBT multiplier if present in settings
                # If your guardian/execution uses different economics, we keep it simple here.
                pnl_pts = (exit_price - entry_price) if position_side == Side.BUY else (entry_price - exit_price)
                pnl_usd = pnl_pts  # placeholder; you can plug CONTRACT_MULTIPLIER later if desired

                state.current_balance += float(pnl_usd)
                state.open_contracts = 0

                paper_rows.append(
                    {
                        "entry_ts": entry_ts,
                        "exit_ts": feats.ts_utc,
                        "side": position_side.value,
                        "qty": qty,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "stop_price": stop_price,
                        "tp_price": tp_price,
                        "exit_reason": exit_reason,
                        "pnl_usd": float(pnl_usd),
                        "balance": float(state.current_balance),
                    }
                )

                # flat
                position_side = None
                qty = 0

        # --- if flat, consider new trade ---
        if position_side is None:
            intent = propose_trade(feats)
            if not intent:
                heartbeat("paper_no_signal")
                if step_sleep:
                    time.sleep(step_sleep)
                continue

            # ML filter
            mld = ml_filter(intent, feats)
            if not mld.approved:
                heartbeat("paper_ml_veto")
                if step_sleep:
                    time.sleep(step_sleep)
                continue

            # risk guard
            rd = evaluate_risk(intent, state)
            if rd.action == "BLOCK":
                heartbeat("paper_risk_block")
                if step_sleep:
                    time.sleep(step_sleep)
                continue
            if rd.action == "KILL":
                print("[PAPER] KILL:", rd.reason)
                break

            # daily trade limiter
            if trades_today >= max_trades_per_day:
                heartbeat("paper_trade_limit")
                if step_sleep:
                    time.sleep(step_sleep)
                continue

            # create simulated market order + attach sl/tp
            price = last
            if intent.side == Side.BUY:
                sl = price - intent.stop_distance
                tp = price + intent.take_profit_distance
            else:
                sl = price + intent.stop_distance
                tp = price - intent.take_profit_distance

            cmd = OrderCommand(
                symbol=intent.symbol,
                side=intent.side,
                qty_contracts=rd.qty_contracts,
                order_type="MARKET",
                stop_loss_price=sl,
                take_profit_price=tp,
                client_order_id=str(uuid.uuid4()),
            )

            # "fill" immediately
            position_side = cmd.side
            entry_price = price
            stop_price = float(sl)
            tp_price = float(tp)
            qty = int(cmd.qty_contracts)
            entry_ts = feats.ts_utc
            trades_today += 1
            state.open_contracts = qty if position_side == Side.BUY else -qty

            print(f"[PAPER] ENTER {cmd.side.value} {qty} @ {entry_price:.2f} SL={stop_price:.2f} TP={tp_price:.2f} prob>={settings.ML_MIN_PROB}")

        heartbeat("paper_loop")
        if step_sleep:
            time.sleep(step_sleep)

    # Save results
    os.makedirs("artifacts", exist_ok=True)
    out = Path("artifacts/paper_trades.csv")
    pd.DataFrame(paper_rows).to_csv(out, index=False)
    print("Saved:", out, "rows=", len(paper_rows))


if __name__ == "__main__":
    main()
