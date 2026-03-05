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


def safety_check():
    # Hard gate so you can't accidentally go live
    if os.getenv("I_UNDERSTAND_LIVE_TRADING", "false").lower() != "true":
        raise SystemExit("Refusing to trade live. Set I_UNDERSTAND_LIVE_TRADING=true in .env")
    if settings.TRADOVATE_ENV.lower() != "live":
        raise SystemExit("MODE=live requires TRADOVATE_ENV=live")


def main():
    safety_check()
    execu = TradovateExecutor()

    # NOTE: until we wire real account balance polling, this state is static.
    # Next step is to poll Tradovate /account/balance and update state.current_balance.
    state = ApexEODState(
        starting_balance=settings.ACCOUNT_SIZE,
        session_start_balance=settings.ACCOUNT_SIZE,
        current_balance=settings.ACCOUNT_SIZE,
        eod_threshold_active=settings.ACCOUNT_SIZE - settings.EOD_MAX_DRAWDOWN_USD,
        dll_floor_active=settings.ACCOUNT_SIZE - settings.DAILY_LOSS_LIMIT_USD,
        open_contracts=0,
    )

    while True:
        # TEMP: uses fake candles until we wire quote data.
        # We'll replace with live data feed next.
        df = pd.DataFrame([{
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "volume": 0.0
        }] * 300)

        feats = compute_features(df, symbol=settings.BOT_SYMBOL, timeframe_min=settings.TIMEFRAME_MINUTES)
        intent = propose_trade(feats)

        if intent:
            mld = ml_filter(intent, feats)
            if mld.approved:
                rd = evaluate_risk(intent, state)
                if rd.action == "ALLOW":
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
                    print(f"[LIVE] submitted {cmd.side} {cmd.qty_contracts} {cmd.symbol} -> {fill.status}")
                elif rd.action == "KILL":
                    print(f"[LIVE] KILL SWITCH: {rd.reason}")
                    break

        heartbeat("live_loop")
        time.sleep(5)


if __name__ == "__main__":
    main()
