from __future__ import annotations

import os
import time
import uuid

from src.common.settings import settings
from src.adapters.tradovate.rest_client import TradovateREST
from src.adapters.tradovate.md_ws import TradovateQuoteStream
from src.execution.tradovate_executor import TradovateExecutor

from src.strategy.baseline_rules import propose_trade
from src.ml.infer import ml_filter
from src.risk.guardian import ApexEODState, evaluate_risk
from src.monitoring.health import heartbeat


def _require_live_gate():
    if os.getenv("I_UNDERSTAND_LIVE_TRADING", "false").lower() != "true":
        raise SystemExit("Refusing to run live. Set I_UNDERSTAND_LIVE_TRADING=true in .env")


def main():
    _require_live_gate()

    tv_rest = TradovateREST()
    account_id = tv_rest.resolve_account_id()

    # start quote stream
    qs = TradovateQuoteStream()
    qs.start()

    execu = TradovateExecutor()

    quote_stale = int(os.getenv("QUOTE_STALE_SECONDS", "5"))
    bal_poll = int(os.getenv("BALANCE_POLL_SECONDS", "10"))

    # state initialized; we will update current_balance via cash balance snapshot
    state = ApexEODState(
        starting_balance=settings.ACCOUNT_SIZE,
        session_start_balance=settings.ACCOUNT_SIZE,
        current_balance=settings.ACCOUNT_SIZE,
        eod_threshold_active=settings.ACCOUNT_SIZE - settings.EOD_MAX_DRAWDOWN_USD,
        dll_floor_active=settings.ACCOUNT_SIZE - settings.DAILY_LOSS_LIMIT_USD,
        open_contracts=0,
    )

    last_bal_poll = 0.0

    while True:
        now = time.time()

        # 1) poll balance/PnL snapshot
        if now - last_bal_poll >= bal_poll:
            try:
                snap = tv_rest.cash_balance_snapshot(account_id)
                # best-effort fields; vary by account
                bal = snap.get("cashBalance") or snap.get("balance") or snap.get("netLiq") or snap.get("equity")
                if bal is not None:
                    state.current_balance = float(bal)
                last_bal_poll = now
            except Exception as e:
                print("[balance] poll failed:", e)

        # 2) get latest quote
        q = qs.get_last()
        if q is None or (now - q.ts) > quote_stale:
            heartbeat("live_waiting_quote")
            time.sleep(1)
            continue

        # 3) Build minimal Features object (your compute_features pipeline can replace this later)
        # We keep it simple: strategy uses price only right now.
        # If your propose_trade expects full Features, ensure it can accept price-only or update your feature engineering.
        feats = type("F", (), {})()
        feats.ts_utc = None
        feats.last_price = q.last
        feats.symbol = settings.BOT_SYMBOL
        feats.timeframe_min = settings.TIMEFRAME_MINUTES

        # placeholders used by ml_filter proxy features:
        feats.ret_1 = 0.0
        feats.ret_5 = 0.0
        feats.atr_14 = 1.0
        feats.ema_20 = q.last
        feats.ema_50 = q.last
        feats.ema_200 = q.last

        intent = propose_trade(feats)
        if intent:
            mld = ml_filter(intent, feats)
            if not mld.approved:
                print(f"[ml] veto prob={mld.prob:.2f}")
                heartbeat("live_ml_veto")
                time.sleep(1)
                continue

            rd = evaluate_risk(intent, state)
            if rd.action == "BLOCK":
                print("[risk] block:", rd.reason)
            elif rd.action == "KILL":
                print("[risk] KILL:", rd.reason)
                break
            else:
                cmd = intent.to_order_command(
                    qty_contracts=rd.qty_contracts,
                    client_order_id=str(uuid.uuid4()),
                )
                fill = execu.submit_market(cmd)
                print(f"[LIVE] {cmd.side} {cmd.qty_contracts} {cmd.symbol} -> {fill.status}")

        heartbeat("live_loop")
        time.sleep(1)


if __name__ == "__main__":
    main()
