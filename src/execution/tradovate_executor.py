from __future__ import annotations

from src.common.schemas import OrderCommand, FillEvent, Side
from src.adapters.tradovate.rest_client import TradovateREST


class TradovateExecutor:
    def __init__(self) -> None:
        self.tv = TradovateREST()

    def submit_market(self, cmd: OrderCommand) -> FillEvent:
        account_id = self.tv.resolve_account_id()
        symbol = self.tv.resolve_symbol()

        action = "Buy" if cmd.side == Side.BUY else "Sell"
        res = self.tv.place_market_order(
            account_id=account_id,
            symbol=symbol,
            action=action,
            qty=cmd.qty_contracts,
        )

        # Tradovate returns orderId on success in common examples
        order_id = str(res.get("orderId", cmd.client_order_id))

        return FillEvent(
            client_order_id=order_id,
            status="SUBMITTED",
            fill_price=None,
            message="tradovate_submitted_market",
        )
