from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from src.adapters.tradovate.rest_client import TradovateREST


@dataclass
class FlattenResult:
    ok: bool
    reason: str
    actions: int = 0
    details: Optional[list] = None


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def flatten_all_positions(symbol: Optional[str] = None) -> FlattenResult:
    """
    Emergency: market-close any open positions.
    - If symbol is provided: only flatten that symbol.
    - If symbol is None: flatten everything (recommended for emergencies).
    """
    tv = TradovateREST()
    account_id = tv.resolve_account_id()

    try:
        positions = tv.positions(account_id)
    except Exception as e:
        return FlattenResult(ok=False, reason=f"positions_fetch_failed:{e}", actions=0)

    actions = 0
    details: list = []

    # Tradovate position/list fields vary; we try common ones.
    for p in positions or []:
        sym = (p.get("symbol") or p.get("contractName") or p.get("name") or "").strip()
        if symbol and sym != symbol:
            continue

        qty = _safe_int(
            p.get("netPos")
            or p.get("netPosition")
            or p.get("position")
            or p.get("qty")
            or p.get("quantity")
            or 0
        )

        if qty == 0:
            continue

        # If long qty>0 -> Sell qty; if short qty<0 -> Buy abs(qty)
        action = "Sell" if qty > 0 else "Buy"
        close_qty = abs(qty)

        try:
            res = tv.place_market_order(account_id=account_id, symbol=sym, action=action, qty=close_qty)
            actions += 1
            details.append({"symbol": sym, "qty": qty, "action": action, "close_qty": close_qty, "res": res})
        except Exception as e:
            details.append({"symbol": sym, "qty": qty, "action": action, "close_qty": close_qty, "error": str(e)})

    return FlattenResult(ok=True, reason="flatten_sent", actions=actions, details=details)
