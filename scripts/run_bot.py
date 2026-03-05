# scripts/run_bot.py
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.common.settings import settings
from src.common.schemas import OrderCommand, Side
from src.data.feature_engineering import compute_features
from src.strategy.baseline_rules import propose_trade
from src.ml.infer import ml_filter
from src.risk.guardian import ApexEODState, evaluate_risk
from src.monitoring.health import heartbeat


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return int(default)


def main():
    """
    True OFFLINE paper trader (economically correct for MBT):
    - Reads candles from data/mbt_15m.csv
    - Computes features
    - Strategy proposes intents
    - ML filter approves/vetoes
    - Risk guard approves/blocks/kills
    - Simulates fills with slippage + fees
    - Writes artifacts/paper_trades.csv + artifacts/paper_summary.json
    """

    # ----------- Economics (MBT defaults) -----------
    TICK_SIZE = _env_float("TICK_SIZE", 5.0)
    TICK_VALUE = _env_float("TICK_VALUE", 0.5)
    # $ per $1 BTC move per contract (MBT: 0.5/5 = 0.1)
    CONTRACT_MULTIPLIER = TICK_VALUE / max(TICK_SIZE, 1e-9)

    FEE_PER_SIDE_USD = _env_float("FEE_PER_SIDE_USD", 1.2)
    SLIPPAGE_TICKS = _env_float("SLIPPAGE_TICKS", 1.0)
    SLIPPAGE_USD_PER_CONTRACT = SLIPPAGE_TICKS * TICK_VALUE  # per side

    # ----------- Inputs -----------
    csv_path = Path(os.getenv("HIST_CSV", "data/mbt_15m.csv"))
    if not csv_path.exists():
        raise SystemExit(f"Missing {csv_path}. Run: python scripts/fetch_free_data.py")

    df_all = pd.read_csv(csv_path).dropna().reset_index(drop=True)
    required = {"ts_utc", "open", "high", "low", "close"}
    missing = required - set(df_all.columns)
    if missing:
        raise SystemExit(f"CSV missing columns: {missing}")

    # Ensure ts_utc parseable
    df_all["ts_utc"] = pd.to_datetime(df_all["ts_utc"], utc=True, errors="coerce")
    df_all = df_all.dropna(subset=["ts_utc"]).reset_index(drop=True)

    # ----------- Apex state (offline sim) -----------
    state = ApexEODState(
        starting_balance=settings.ACCOUNT_SIZE,
        session_start_balance=settings.ACCOUNT_SIZE,
        current_balance=settings.ACCOUNT_SIZE,
        eod_threshold_active=settings.ACCOUNT_SIZE - settings.EOD_MAX_DRAWDOWN_USD,
        dll_floor_active=settings.ACCOUNT_SIZE - settings.DAILY_LOSS_LIMIT_USD,
        open_contracts=0,
    )

    # ----------- Daily limiter -----------
    max_trades_per_day = _env_int("MAX_TRADES_PER_DAY", 20)
    trade_day = None
    trades_today = 0

    # ----------- Position state -----------
    position_side: Side | None = None
    entry_price = 0.0
    stop_price = 0.0
    tp_price = 0.0
    qty = 0
    entry_ts = None
    entry_id = None

    # For performance stats
    paper_rows: list[dict] = []
    veto_ml = 0
    veto_risk = 0
    no_signal = 0
    trade_limit_blocks = 0

    # Warmup window for indicators
    window = _env_int("PAPER_FEATURE_WINDOW", 300)
    step_sleep = _env_float("PAPER_STEP_SLEEP", 0.0)

    # Equity curve tracking
    peak_equity = float(state.current_balance)
    max_dd = 0.0

    def update_dd():
        nonlocal peak_equity, max_dd
        peak_equity = max(peak_equity, float(state.current_balance))
        max_dd = max(max_dd, peak_equity - float(state.current_balance))

    # ----------- Replay loop -----------
    for i in range(window, len(df_all)):
        chunk = df_all.iloc[i - window : i].copy()
        # compute_features expects ts_utc as string sometimes; give ISO for compatibility
        chunk2 = chunk.copy()
        chunk2["ts_utc"] = chunk2["ts_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        feats = compute_features(chunk2, symbol=settings.BOT_SYMBOL, timeframe_min=settings.TIMEFRAME_MINUTES)

        # reset daily trade counter
        cur_ts = pd.to_datetime(feats.ts_utc, utc=True, errors="coerce")
        if pd.isna(cur_ts):
            continue
        day = cur_ts.date()
        if trade_day != day:
            trade_day = day
            trades_today = 0

        last = float(feats.last_price)

        # Use last bar high/low for intrabar hit approximation
        bar = df_all.iloc[i - 1]
        high = float(bar["high"])
        low = float(bar["low"])

        # -------- Exit checks --------
        if position_side is not None:
            hit_stop = (low <= stop_price) if position_side == Side.BUY else (high >= stop_price)
            hit_tp = (high >= tp_price) if position_side == Side.BUY else (low <= tp_price)

            exit_reason = None
            exit_price = None

            # Conservative: if both hit, assume stop first
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
                # Apply slippage AGAIN on exit (per side)
                if position_side == Side.BUY:
                    exit_fill = float(exit_price) - (SLIPPAGE_TICKS * TICK_SIZE)
                else:
                    exit_fill = float(exit_price) + (SLIPPAGE_TICKS * TICK_SIZE)

                # Entry fill already includes slippage (see below)
                pnl_points = (exit_fill - entry_price) if position_side == Side.BUY else (entry_price - exit_fill)
                gross_usd = pnl_points * CONTRACT_MULTIPLIER * qty

                fees_usd = (2.0 * FEE_PER_SIDE_USD) * qty
                slip_usd = (2.0 * SLIPPAGE_USD_PER_CONTRACT) * qty
                net_usd = float(gross_usd - fees_usd - slip_usd)

                state.current_balance += net_usd
                state.open_contracts = 0
                update_dd()

                paper_rows.append(
                    {
                        "trade_id": entry_id,
                        "entry_ts": entry_ts,
                        "exit_ts": feats.ts_utc,
                        "side": position_side.value,
                        "qty": int(qty),
                        "entry_price": float(entry_price),
                        "exit_price": float(exit_fill),
                        "stop_price": float(stop_price),
                        "tp_price": float(tp_price),
                        "exit_reason": exit_reason,
                        "pnl_gross_usd": float(gross_usd),
                        "fees_usd": float(fees_usd),
                        "slippage_usd": float(slip_usd),
                        "pnl_usd": float(net_usd),
                        "balance": float(state.current_balance),
                        "max_dd_so_far": float(max_dd),
                    }
                )

                print(
                    f"[PAPER] EXIT  {position_side.value} {qty} id={entry_id} "
                    f"reason={exit_reason} pnl_usd={net_usd:.2f} bal={state.current_balance:.2f}"
                )

                # flat
                position_side = None
                qty = 0
                entry_ts = None
                entry_id = None

        # -------- Entry checks --------
        if position_side is None:
            intent = propose_trade(feats)
            if not intent:
                no_signal += 1
                heartbeat("paper_no_signal")
                if step_sleep:
                    time.sleep(step_sleep)
                continue

            # ML filter
            mld = ml_filter(intent, feats)
            if not mld.approved:
                veto_ml += 1
                heartbeat("paper_ml_veto")
                if step_sleep:
                    time.sleep(step_sleep)
                continue

            # risk guard
            rd = evaluate_risk(intent, state)
            if rd.action == "BLOCK":
                veto_risk += 1
                heartbeat("paper_risk_block")
                if step_sleep:
                    time.sleep(step_sleep)
                continue
            if rd.action == "KILL":
                print("[PAPER] KILL:", rd.reason)
                break

            # daily trade limiter
            if trades_today >= max_trades_per_day:
                trade_limit_blocks += 1
                heartbeat("paper_trade_limit")
                if step_sleep:
                    time.sleep(step_sleep)
                continue

            # create simulated market order + attach sl/tp
            # Apply slippage on entry
            if intent.side == Side.BUY:
                entry_fill = last + (SLIPPAGE_TICKS * TICK_SIZE)
                sl = last - intent.stop_distance
                tp = last + intent.take_profit_distance
            else:
                entry_fill = last - (SLIPPAGE_TICKS * TICK_SIZE)
                sl = last + intent.stop_distance
                tp = last - intent.take_profit_distance

            cmd = OrderCommand(
                symbol=intent.symbol,
                side=intent.side,
                qty_contracts=rd.qty_contracts,
                order_type="MARKET",
                stop_loss_price=float(sl),
                take_profit_price=float(tp),
                client_order_id=str(uuid.uuid4()),
            )

            # "fill" immediately
            position_side = cmd.side
            entry_price = float(entry_fill)
            stop_price = float(sl)
            tp_price = float(tp)
            qty = int(cmd.qty_contracts)
            entry_ts = feats.ts_utc
            entry_id = cmd.client_order_id

            trades_today += 1
            state.open_contracts = qty if position_side == Side.BUY else -qty

            print(
                f"[PAPER] ENTER {cmd.side.value} {qty} id={entry_id} @ {entry_price:.2f} "
                f"SL={stop_price:.2f} TP={tp_price:.2f} prob>={settings.ML_MIN_PROB}"
            )

        heartbeat("paper_loop")
        if step_sleep:
            time.sleep(step_sleep)

    # -------- Save results --------
    os.makedirs("artifacts", exist_ok=True)
    trades_out = Path("artifacts/paper_trades.csv")
    pd.DataFrame(paper_rows).to_csv(trades_out, index=False)

    # Summary
    summary_out = Path("artifacts/paper_summary.json")
    if paper_rows:
        tdf = pd.DataFrame(paper_rows)
        pnl = tdf["pnl_usd"].astype(float)
        wins = (pnl > 0).sum()
        losses = (pnl <= 0).sum()
        gross_win = pnl[pnl > 0].sum()
        gross_loss = (-pnl[pnl < 0]).sum()

        pf = float(gross_win / gross_loss) if gross_loss > 0 else float("inf") if gross_win > 0 else 0.0
        win_rate = float(wins / len(tdf)) if len(tdf) else 0.0

        summary = {
            "trades": int(len(tdf)),
            "win_rate": float(win_rate),
            "profit_factor": float(pf),
            "total_pnl_usd": float(pnl.sum()),
            "avg_pnl_usd": float(pnl.mean()),
            "median_pnl_usd": float(pnl.median()),
            "final_balance": float(tdf["balance"].iloc[-1]),
            "max_balance": float(tdf["balance"].max()),
            "min_balance": float(tdf["balance"].min()),
            "max_drawdown_usd": float(max_dd),
            "ml_vetos": int(veto_ml),
            "risk_blocks": int(veto_risk),
            "no_signal_bars": int(no_signal),
            "trade_limit_blocks": int(trade_limit_blocks),
            "economics": {
                "tick_size": float(TICK_SIZE),
                "tick_value": float(TICK_VALUE),
                "contract_multiplier": float(CONTRACT_MULTIPLIER),
                "fee_per_side_usd": float(FEE_PER_SIDE_USD),
                "slippage_ticks": float(SLIPPAGE_TICKS),
            },
        }
    else:
        summary = {
            "trades": 0,
            "message": "No completed trades in sample (check strategy/params).",
        }

    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Saved:", trades_out, "rows=", len(paper_rows))
    print("Saved:", summary_out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

    # --- Write summary (always) ---
    import json
    import numpy as np

    summary_out = Path("artifacts/paper_summary.json")
    tdf = pd.DataFrame(paper_rows)

    if len(tdf) == 0:
        summary = {"trades": 0, "message": "No completed trades."}
    else:
        pnl = tdf["pnl_usd"].astype(float)
        wins = int((pnl > 0).sum())
        losses = int((pnl <= 0).sum())
        gross_win = float(pnl[pnl > 0].sum())
        gross_loss = float((-pnl[pnl < 0]).sum())
        pf = float(gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)

        summary = {
            "trades": int(len(tdf)),
            "win_rate": float(wins / len(tdf)),
            "profit_factor": float(pf),
            "total_pnl_usd": float(pnl.sum()),
            "avg_pnl_usd": float(pnl.mean()),
            "median_pnl_usd": float(pnl.median()),
            "final_balance": float(tdf["balance"].iloc[-1]),
            "max_balance": float(tdf["balance"].max()),
            "min_balance": float(tdf["balance"].min()),
        }

    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Saved:", summary_out)
