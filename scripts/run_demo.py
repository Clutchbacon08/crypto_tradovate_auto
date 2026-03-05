
from __future__ import annotations

import os
import time
import uuid
from datetime import datetime, timezone

import pandas as pd

from src.common.settings import settings
from src.common.schemas import OrderCommand
from src.data.feature_engineering import compute_features
from src.strategy.baseline_rules import propose_trade
from src.ml.infer import ml_filter
from src.risk.guardian import ApexEODState, evaluate_risk
from src.execution.tradovate_executor import TradovateExecutor
from src.monitoring.health import heartbeat

from src.risk.safety_guard import SafetyGuard, SafetyState
from src.risk.reconcile import PositionReconciler


def fake_market_df(n: int = 300) -> pd.DataFrame:
    # TEMP: until full Tradovate candle/quote feed is wired
    rows = []
    price = 100.0
    for i in range(n):
        ts = datetime.now(timezone.utc) - pd.Timedelta(minutes=(n - i) * settings.TIMEFRAME_MINUTES)
        price *= (1.0 + (0.0005 if i % 20 < 10 else -0.0003))
        rows.append(
            {
                "ts_utc": ts.isoformat(),
                "open": price * 0.999,
                "high": price * 1.001,
                "low": price * 0.998,
                "close": price,
                "volume": 0.0,
            }
        )
    return pd.DataFrame(rows)


def main():
    if settings.TRADOVATE_ENV.lower() != "demo":
        raise SystemExit("Set TRADOVATE_ENV=demo in .env for demo trading.")

    execu = TradovateExecutor()

    # Safety guard (in demo we still want: block if ML model missing)
    safety = SafetyGuard()
    safety_state = SafetyState()

    # Position reconciliation (uses broker truth)
    reconciler = PositionReconciler(symbol=settings.TRADOVATE_SYMBOL, poll_seconds=5, max_errors=5)

    # NOTE: Until you wire real balance polling in demo, we still use starting balance.
    # SafetyGuard's balance check expects last_balance_ts; set a heartbeat timestamp each loop.
    state = ApexEODState(
        starting_balance=settings.ACCOUNT_SIZE,
        session_start_balance=settings.ACCOUNT_SIZE,
        current_balance=settings.ACCOUNT_SIZE,
        eod_threshold_active=settings.ACCOUNT_SIZE - settings.EOD_MAX_DRAWDOWN_USD,
        dll_floor_active=settings.ACCOUNT_SIZE - settings.DAILY_LOSS_LIMIT_USD,
        open_contracts=0,
    )

    while True:
        now = time.time()

        # DEMO: no real balance polling yet, but keep safety guard satisfied
        safety_state.last_balance_ts = now

        # Build fake market candles and compute features
        df = fake_market_df(300)
        feats = compute_features(df, symbol=settings.BOT_SYMBOL, timeframe_min=settings.TIMEFRAME_MINUTES)

        # DEMO: treat "quote fresh" as now (since we're generating the candles)
        safety_state.last_quote_ts = now

        # Kill-switch safety check (ML must be present; quote/balance "fresh")
        safe, reason = safety.check_all(safety_state)
        if not safe:
            print(f"[DEMO SAFETY BLOCK] {reason}")
            heartbeat("demo_safety_block")
            time.sleep(2)
            continue

        # Reconcile positions BEFORE proposing trades
        rr = reconciler.reconcile(local_open_contracts=state.open_contracts)
        if rr.action == "HALT":
            print("[DEMO RECONCILE HALT]", rr.reason)
            heartbeat("demo_reconcile_halt")
            time.sleep(2)
            continue

        if rr.action == "RESYNC" and rr.snapshot is not None:
            state.open_contracts = int(rr.snapshot.qty)
            print("[DEMO RECONCILE RESYNC]", rr.reason)
            heartbeat("demo_reconcile_resync")
            # Don't open a new trade on the same loop after resync
            time.sleep(1)
            continue

        intent = propose_trade(feats)
        if intent:
            mld = ml_filter(intent, feats)
            if not mld.approved:
                if mld.reason == "ml_model_missing":
                    safety_state.ml_model_loaded = False
                print(f"[{feats.ts_utc}] ML veto prob={mld.prob:.2f} reason={mld.reason}")
                heartbeat("demo_ml_veto")
                time.sleep(1)
                continue
            else:
                safety_state.ml_model_loaded = True

            rd = evaluate_risk(intent, state)
            if rd.action == "BLOCK":
                print(f"[{feats.ts_utc}] Risk block: {rd.reason}")
            elif rd.action == "KILL":
                print(f"[{feats.ts_utc}] KILL SWITCH: {rd.reason}")
                break
            else:
                price = feats.last_price
                if intent.side.value == "BUY":
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

                fill = execu.submit_market(cmd)
                print(
                    f"[{feats.ts_utc}] DEMO SUBMIT {cmd.side} {cmd.qty_contracts} {cmd.symbol} "
                    f"-> {fill.status} id={fill.client_order_id}"
                )

                # Best-effort local position update (reconciler is source of truth)
                try:
                    side = getattr(cmd.side, "value", str(cmd.side)).upper()
                    signed_qty = int(cmd.qty_contracts) * (1 if "BUY" in side else -1)
                    state.open_contracts = signed_qty
                except Exception:
                    pass

        heartbeat("demo_loop")
        time.sleep(5)


if __name__ == "__main__":
    main()
