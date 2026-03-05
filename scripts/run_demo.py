from __future__ import annotations

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


def fake_market_df(n: int = 300) -> pd.DataFrame:
    # TEMP: until Tradovate quote stream is wired (Option A starts with REST)
    rows = []
    price = 100.0
    for i in range(n):
        ts = datetime.now(timezone.utc) - pd.Timedelta(minutes=(n - i) * settings.TIMEFRAME_MINUTES)
        price *= (1.0 + (0.0005 if i % 20 < 10 else -0.0003))
        rows.append({
            "ts_utc": ts.isoformat(),
            "open": price * 0.999,
            "high": price * 1.001,
            "low": price * 0.998,
            "close": price,
            "volume": 0.0
        })
    return pd.DataFrame(rows)


def main():
    if settings.TRADOVATE_ENV.lower() != "demo":
        raise SystemExit("Set TRADOVATE_ENV=demo in .env for demo trading.")

    execu = TradovateExecutor()

    # NOTE: Until we wire real balance polling, state uses starting balance only.
    state = ApexEODState(
        starting_balance=settings.ACCOUNT_SIZE,
        session_start_balance=settings.ACCOUNT_SIZE,
        current_balance=settings.ACCOUNT_SIZE,
        eod_threshold_active=settings.ACCOUNT_SIZE - settings.EOD_MAX_DRAWDOWN_USD,
        dll_floor_active=settings.ACCOUNT_SIZE - settings.DAILY_LOSS_LIMIT_USD,
        open_contracts=0,
    )

    while True:
        df = fake_market_df(300)
        feats = compute_features(df, symbol=settings.BOT_SYMBOL, timeframe_min=settings.TIMEFRAME_MINUTES)

        intent = propose_trade(feats)
        if intent:
            mld = ml_filter(intent, feats)
            if not mld.approved:
                print(f"[{feats.ts_utc}] ML veto prob={mld.prob:.2f} reason={mld.reason}")
            else:
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
                    print(f"[{feats.ts_utc}] DEMO SUBMIT {cmd.side} {cmd.qty_contracts} {cmd.symbol} -> {fill.status} id={fill.client_order_id}")

        heartbeat("demo_loop")
        time.sleep(5)


if __name__ == "__main__":
    main()
