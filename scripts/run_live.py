from __future__ import annotations

import os
import time
import uuid
from datetime import datetime

from src.common.settings import settings
from src.adapters.tradovate.rest_client import TradovateREST
from src.adapters.tradovate.md_ws import TradovateQuoteStream
from src.execution.tradovate_executor import TradovateExecutor

from src.strategy.baseline_rules import propose_trade
from src.ml.infer import ml_filter
from src.risk.guardian import ApexEODState, evaluate_risk
from src.monitoring.health import heartbeat

from src.risk.safety_guard import SafetyGuard, SafetyState
from src.risk.reconcile import PositionReconciler


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

    # Safety guard state
    safety = SafetyGuard()
    safety_state = SafetyState()

    # Position reconciler (prevents desync/double-positioning)
    reconciler = PositionReconciler(symbol=settings.TRADOVATE_SYMBOL, poll_seconds=5, max_errors=5)

    # Apex state (balance is updated via cash balance snapshot)
    state = ApexEODState(
        starting_balance=settings.ACCOUNT_SIZE,
        session_start_balance=settings.ACCOUNT_SIZE,
        current_balance=settings.ACCOUNT_SIZE,
        eod_threshold_active=settings.ACCOUNT_SIZE - settings.EOD_MAX_DRAWDOWN_USD,
        dll_floor_active=settings.ACCOUNT_SIZE - settings.DAILY_LOSS_LIMIT_USD,
        open_contracts=0,
    )

    # -----------------------------
    # MAX TRADES PER DAY LIMITER
    # -----------------------------
    max_trades_per_day = int(os.getenv("MAX_TRADES_PER_DAY", "20"))
    trade_day = datetime.now().date()
    trades_today = 0
    # -----------------------------

    last_bal_poll = 0.0

    while True:
        now = time.time()

        # -----------------------------
        # RESET TRADE COUNT EACH DAY
        # -----------------------------
        today = datetime.now().date()
        if today != trade_day:
            trade_day = today
            trades_today = 0
        # -----------------------------

        # 1) poll balance/PnL snapshot
        if now - last_bal_poll >= bal_poll:
            try:
                snap = tv_rest.cash_balance_snapshot(account_id)
                bal = (
                    snap.get("cashBalance")
                    or snap.get("balance")
                    or snap.get("netLiq")
                    or snap.get("equity")
                )
                if bal is not None:
                    state.current_balance = float(bal)

                last_bal_poll = now
                safety_state.last_balance_ts = now
            except Exception as e:
                print("[balance] poll failed:", e)

        # 2) get latest quote
        q = qs.get_last()
        if q is not None:
            safety_state.last_quote_ts = q.ts

        # 3) kill-switch safety check (quote + balance + ML presence)
        safe, reason = safety.check_all(safety_state)
        if not safe:
            print(f"[SAFETY BLOCK] {reason}")
            heartbeat("live_safety_block")
            time.sleep(1)
            continue

        # 4) quote freshness check
        if q is None or (now - q.ts) > quote_stale:
            heartbeat("live_waiting_quote")
            time.sleep(1)
            continue

        # 5) Build minimal Features object
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

        # 6) Reconcile local vs broker positions BEFORE proposing/placing trades
        rr = reconciler.reconcile(local_open_contracts=state.open_contracts)
        if rr.action == "HALT":
            print("[RECONCILE HALT]", rr.reason)
            heartbeat("live_reconcile_halt")
            time.sleep(2)
            continue

        if rr.action == "RESYNC" and rr.snapshot is not None:
            state.open_contracts = int(rr.snapshot.qty)
            print("[RECONCILE RESYNC]", rr.reason)
            heartbeat("live_reconcile_resync")
            time.sleep(1)
            continue

        # 7) Strategy proposes trade intent
        intent = propose_trade(feats)
        if intent:
            # 8) ML veto filter
            mld = ml_filter(intent, feats)
            if not mld.approved:
                if mld.reason == "ml_model_missing":
                    safety_state.ml_model_loaded = False
                print(f"[ml] veto prob={mld.prob:.2f} reason={mld.reason}")
                heartbeat("live_ml_veto")
                time.sleep(1)
                continue
            else:
                safety_state.ml_model_loaded = True

            # 9) Apex risk guard
            rd = evaluate_risk(intent, state)
            if rd.action == "BLOCK":
                print("[risk] block:", rd.reason)
            elif rd.action == "KILL":
                print("[risk] KILL:", rd.reason)
                break
            else:
                # -----------------------------
                # TRADE LIMIT CHECK (before submit)
                # -----------------------------
                if trades_today >= max_trades_per_day:
                    print(f"[LIMIT] MAX_TRADES_PER_DAY reached ({trades_today}/{max_trades_per_day}). Blocking entries.")
                    heartbeat("trade_limit_hit")
                    time.sleep(2)
                    continue
                # -----------------------------

                cmd = intent.to_order_command(
                    qty_contracts=rd.qty_contracts,
                    client_order_id=str(uuid.uuid4()),
                )

                fill = execu.submit_market(cmd)
                print(f"[LIVE] {cmd.side} {cmd.qty_contracts} {cmd.symbol} -> {fill.status}")

                # increment daily counter once per order submission
                trades_today += 1

                # Best-effort local position update (reconciler is source of truth)
                try:
                    side = getattr(cmd.side, "value", str(cmd.side)).upper()
                    signed_qty = int(cmd.qty_contracts) * (1 if "BUY" in side else -1)
                    state.open_contracts = signed_qty
                except Exception:
                    pass

        heartbeat("live_loop")
        time.sleep(1)


if __name__ == "__main__":
    main()
