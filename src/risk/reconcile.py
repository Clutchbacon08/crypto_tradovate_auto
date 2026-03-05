from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from src.adapters.tradovate.rest_client import TradovateREST


@dataclass
class PositionSnapshot:
    """
    Normalized view of Tradovate position for ONE symbol.
    qty > 0 => long, qty < 0 => short, qty == 0 => flat
    """
    symbol: str
    qty: int
    avg_price: Optional[float] = None
    raw: Optional[dict] = None


@dataclass
class ReconcileResult:
    action: str  # "OK", "RESYNC", "HALT"
    reason: str
    snapshot: Optional[PositionSnapshot] = None


class PositionReconciler:
    """
    Protects against:
      - VPS restarts
      - lost in-memory state
      - partial fills / unknown fills
      - desync between bot and broker

    Works by polling Tradovate positions and comparing to local state.
    If mismatch is detected, it can RESYNC your local state (recommended).
    If too many errors occur, it HALTs trading.
    """

    def __init__(self, symbol: str, poll_seconds: int = 5, max_errors: int = 5) -> None:
        self.symbol = symbol
        self.poll_seconds = poll_seconds
        self.max_errors = max_errors

        self._tv = TradovateREST()
        self._account_id: Optional[int] = None

        self._last_poll_ts: float = 0.0
        self._error_count: int = 0
        self._last_snapshot: Optional[PositionSnapshot] = None

    def _ensure_account(self) -> int:
        if self._account_id is None:
            self._account_id = self._tv.resolve_account_id()
        return self._account_id

    def _normalize_positions(self, positions: List[Dict[str, Any]]) -> PositionSnapshot:
        """
        Tradovate position/list field names can vary by environment.
        We normalize into a single PositionSnapshot for self.symbol.

        If multiple entries exist, we sum quantities.
        """
        qty_sum = 0
        avg_price = None
        raws = []

        for p in positions or []:
            sym = (p.get("symbol") or p.get("contractName") or p.get("name") or "").strip()
            if sym != self.symbol:
                continue

            raws.append(p)

            # Quantity fields can vary
            q = (
                p.get("netPos")
                or p.get("netPosition")
                or p.get("position")
                or p.get("qty")
                or p.get("quantity")
                or 0
            )
            try:
                q = int(float(q))
            except Exception:
                q = 0

            qty_sum += q

            ap = p.get("avgPrice") or p.get("averagePrice") or p.get("price")
            if ap is not None and avg_price is None:
                try:
                    avg_price = float(ap)
                except Exception:
                    pass

        return PositionSnapshot(symbol=self.symbol, qty=qty_sum, avg_price=avg_price, raw={"matches": raws})

    def poll(self, force: bool = False) -> ReconcileResult:
        """
        Poll broker positions at most every poll_seconds unless force=True.
        Stores last snapshot.
        """
        now = time.time()
        if not force and (now - self._last_poll_ts) < self.poll_seconds:
            return ReconcileResult(action="OK", reason="poll_skipped", snapshot=self._last_snapshot)

        try:
            account_id = self._ensure_account()
            positions = self._tv.positions(account_id)
            snap = self._normalize_positions(positions)
            self._last_snapshot = snap
            self._last_poll_ts = now
            self._error_count = 0
            return ReconcileResult(action="OK", reason="polled", snapshot=snap)
        except Exception as e:
            self._error_count += 1
            if self._error_count >= self.max_errors:
                return ReconcileResult(action="HALT", reason=f"positions_poll_failed:{e}", snapshot=self._last_snapshot)
            return ReconcileResult(action="OK", reason=f"positions_poll_error:{e}", snapshot=self._last_snapshot)

    def reconcile(self, local_open_contracts: int) -> ReconcileResult:
        """
        Compares local state (open contracts) to broker snapshot (qty).
        Returns:
          - OK if match
          - RESYNC if mismatch (with snapshot)
          - HALT if repeated API errors
        """
        pr = self.poll(force=False)
        if pr.action == "HALT":
            return pr

        snap = pr.snapshot
        if snap is None:
            # Can't verify; better to keep trading blocked at higher layer if desired
            return ReconcileResult(action="OK", reason="no_snapshot_yet", snapshot=None)

        broker_qty = int(snap.qty)
        local_qty = int(local_open_contracts)

        if broker_qty == local_qty:
            return ReconcileResult(action="OK", reason="in_sync", snapshot=snap)

        return ReconcileResult(
            action="RESYNC",
            reason=f"desync local={local_qty} broker={broker_qty}",
            snapshot=snap,
        )
