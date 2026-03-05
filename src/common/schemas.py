from pathlib import Path

schemas_path = Path("/content/crypto_tradovate_auto/src/common/schemas.py")
schemas_path.write_text(
"""from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Mode(str, Enum):
    paper = "paper"
    demo = "demo"
    live = "live"


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Features:
    ts_utc: Optional[str] = None
    symbol: str = ""
    timeframe_min: int = 0
    last_price: float = 0.0

    ret_1: float = 0.0
    ret_5: float = 0.0
    atr_14: float = 0.0

    ema_20: float = 0.0
    ema_50: float = 0.0
    ema_200: float = 0.0


@dataclass
class TradeIntent:
    symbol: str
    side: Side
    stop_distance: float
    take_profit_distance: float

    def to_order_command(self, qty_contracts: int, client_order_id: str) -> "OrderCommand":
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
""",
    encoding="utf-8"
)

print("Overwrote:", schemas_path)
print("Contains 'class Mode':", "class Mode" in schemas_path.read_text(encoding="utf-8"))
