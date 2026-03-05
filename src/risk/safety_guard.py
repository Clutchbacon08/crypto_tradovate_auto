from __future__ import annotations
import os
import time
from dataclasses import dataclass


@dataclass
class SafetyState:
    last_quote_ts: float | None = None
    last_balance_ts: float | None = None
    ml_model_loaded: bool = True


class SafetyGuard:
    """
    Global safety checks to prevent trading when core systems fail.
    """

    def __init__(self):
        self.quote_timeout = int(os.getenv("QUOTE_STALE_SECONDS", "5"))
        self.balance_timeout = int(os.getenv("BALANCE_STALE_SECONDS", "30"))

    def check_quote(self, state: SafetyState) -> tuple[bool, str]:
        if state.last_quote_ts is None:
            return False, "no_quote_received"

        if time.time() - state.last_quote_ts > self.quote_timeout:
            return False, "quote_stream_stale"

        return True, "ok"

    def check_balance(self, state: SafetyState) -> tuple[bool, str]:
        if state.last_balance_ts is None:
            return False, "balance_not_polled"

        if time.time() - state.last_balance_ts > self.balance_timeout:
            return False, "balance_data_stale"

        return True, "ok"

    def check_ml(self, state: SafetyState) -> tuple[bool, str]:
        if not state.ml_model_loaded:
            return False, "ml_model_missing"

        return True, "ok"

    def check_all(self, state: SafetyState) -> tuple[bool, str]:
        ok, msg = self.check_ml(state)
        if not ok:
            return False, msg

        ok, msg = self.check_quote(state)
        if not ok:
            return False, msg

        ok, msg = self.check_balance(state)
        if not ok:
            return False, msg

        return True, "safe"
