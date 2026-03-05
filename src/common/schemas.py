from __future__ import annotations

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class Mode(str, Enum):
    paper = "paper"
    live = "live"


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Features:
    ts_utc: Optional[str]
    symbol: str
    timeframe_min: int
    last_price: float

    ret_1: float
    ret_5: float
    atr_14: float

    ema_20: float
    ema_50: float
    ema_200: float


@dataclass
class TradeIntent:
    symbol: str
    side: Side
    stop_distance: float
    take_profit_distance: float

    def to_order_command(self, qty_contracts: int, client_order_id: str):
        from src.common.schemas import OrderCommand

        return OrderCommand(
            symbol=self.symbol,
            side=self.side,
            qty_contracts=qty_contracts,
            order_type="MARKET",
            stop_loss_price=None,
            take_profit_price=None,
            client_order_id=client_order_id,
        )


@dataclass
class MLDecision:
    approved: bool
    prob: float
    reason: str


@dataclass
class OrderCommand:
    symbol: str
    side: Side
    qty_contracts: int
    order_type: str
    stop_loss_price: Optional[float]
    take_profit_price: Optional[float]
    client_order_id: str
