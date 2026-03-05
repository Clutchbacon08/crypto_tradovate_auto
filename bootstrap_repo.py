import os
from pathlib import Path

# ============================
# Apex EOD Eval (50k) defaults
# ============================
APEX_ACCOUNT_SIZE = 50000.0
APEX_PROFIT_TARGET_USD = 3000.0
APEX_EOD_MAX_DRAWDOWN_USD = 2000.0
APEX_DAILY_LOSS_LIMIT_USD = 1000.0
APEX_MAX_CONTRACTS = 6
APEX_CONTRACTS_PER_TRADE = 1  # fixed size during evaluation (can be <= max contracts)

FILES = {
    # Root files
    ".gitignore": """\
__pycache__/
*.pyc
*.pyo
*.pyd
*.log
.env
.venv/
venv/
data/
models/
logs/
mlruns/
.DS_Store
Thumbs.db
""",
    ".env.example": f"""\
# ===== Tradovate credentials (DO NOT COMMIT .env) =====
TRADOVATE_USERNAME=
TRADOVATE_PASSWORD=
TRADOVATE_APP_ID=
TRADOVATE_APP_VERSION=1.0
TRADOVATE_ENV=demo   # demo | live

# ===== Bot runtime =====
MODE=paper           # paper | demo | live
BOT_SYMBOL=MBT
TIMEFRAME_MINUTES=15

# ===== Apex EOD Eval (50k) =====
ACCOUNT_SIZE={int(APEX_ACCOUNT_SIZE)}
PROFIT_TARGET_USD={int(APEX_PROFIT_TARGET_USD)}
EOD_MAX_DRAWDOWN_USD={int(APEX_EOD_MAX_DRAWDOWN_USD)}
DAILY_LOSS_LIMIT_USD={int(APEX_DAILY_LOSS_LIMIT_USD)}

MAX_CONTRACTS={APEX_MAX_CONTRACTS}
CONTRACTS_PER_TRADE={APEX_CONTRACTS_PER_TRADE}

# Optional: stop trading once profit target is reached
STOP_TRADING_ON_TARGET=true

# ===== ML filter =====
ML_ENABLED=true
ML_MIN_PROB=0.55
""",
    "requirements.txt": """\
python-dotenv
pydantic
pydantic-settings
requests
websocket-client
pandas
numpy
scikit-learn
rq
rq-scheduler
redis
mlflow
optuna
evidently
prometheus-client
sentry-sdk
""",
    ".gitmodules": """\
[submodule "vendor/Tradovate-Python-Client"]
  path = vendor/Tradovate-Python-Client
  url = https://github.com/cullen-b/Tradovate-Python-Client

[submodule "vendor/mlflow"]
  path = vendor/mlflow
  url = https://github.com/mlflow/mlflow

[submodule "vendor/rq"]
  path = vendor/rq
  url = https://github.com/rq/rq

[submodule "vendor/rq-scheduler"]
  path = vendor/rq-scheduler
  url = https://github.com/rq/rq-scheduler

[submodule "vendor/tradovate-example-api-trading-strategy"]
  path = vendor/tradovate-example-api-trading-strategy
  url = https://github.com/tradovate/example-api-trading-strategy
""",
    # Common
    "src/common/schemas.py": """\
from __future__ import annotations
from enum import Enum
from typing import Optional, Literal
from pydantic import BaseModel, Field


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class Mode(str, Enum):
    PAPER = "paper"
    DEMO = "demo"
    LIVE = "live"


class Features(BaseModel):
    symbol: str
    timeframe_min: int
    ts_utc: str
    ema_20: float
    ema_50: float
    ema_200: float
    atr_14: float
    ret_1: float
    ret_5: float
    last_price: float


class TradeIntent(BaseModel):
    symbol: str
    side: Side
    ts_utc: str
    timeframe_min: int
    stop_distance: float
    take_profit_distance: float
    rule_confidence: float = Field(ge=0.0, le=1.0, default=0.5)


class MLDecision(BaseModel):
    approved: bool
    prob: float = Field(ge=0.0, le=1.0)
    reason: str = ""


class RiskDecision(BaseModel):
    action: Literal["ALLOW", "BLOCK", "KILL"]
    reason: str
    qty_contracts: int = 0


class OrderCommand(BaseModel):
    symbol: str
    side: Side
    qty_contracts: int
    order_type: Literal["MARKET"] = "MARKET"
    stop_loss_price: float
    take_profit_price: float
    client_order_id: str


class FillEvent(BaseModel):
    client_order_id: str
    status: Literal["SUBMITTED", "FILLED", "REJECTED", "CANCELLED"]
    fill_price: Optional[float] = None
    message: str = ""
""",
    "src/common/settings.py": """\
from __future__ import annotations
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from src.common.schemas import Mode


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Broker (wired later)
    TRADOVATE_USERNAME: str = ""
    TRADOVATE_PASSWORD: str = ""
    TRADOVATE_APP_ID: str = ""
    TRADOVATE_APP_VERSION: str = "1.0"
    TRADOVATE_ENV: str = "demo"

    # Bot
    MODE: Mode = Mode.PAPER
    BOT_SYMBOL: str = "MBT"
    TIMEFRAME_MINUTES: int = 15

    # ===== Apex EOD Eval rules (USD) =====
    ACCOUNT_SIZE: float = 50000.0

    PROFIT_TARGET_USD: float = 3000.0
    EOD_MAX_DRAWDOWN_USD: float = 2000.0
    DAILY_LOSS_LIMIT_USD: float = 1000.0

    MAX_CONTRACTS: int = 6
    CONTRACTS_PER_TRADE: int = 1

    STOP_TRADING_ON_TARGET: bool = True

    # ML
    ML_ENABLED: bool = True
    ML_MIN_PROB: float = 0.55


settings = Settings()
""",
    # Data/Indicators
    "src/data/indicators.py": """\
from __future__ import annotations
import pandas as pd


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1
    ).max(axis=1)

    return tr.rolling(period).mean().bfill()
""",
    "src/data/feature_engineering.py": """\
from __future__ import annotations
import pandas as pd
from src.data.indicators import ema, atr
from src.common.schemas import Features


def compute_features(df: pd.DataFrame, symbol: str, timeframe_min: int) -> Features:
    df = df.copy()
    close = df["close"]

    df["ema_20"] = ema(close, 20)
    df["ema_50"] = ema(close, 50)
    df["ema_200"] = ema(close, 200)
    df["atr_14"] = atr(df, 14)

    df["ret_1"] = close.pct_change(1).fillna(0.0)
    df["ret_5"] = close.pct_change(5).fillna(0.0)

    last = df.iloc[-1]
    return Features(
        symbol=symbol,
        timeframe_min=timeframe_min,
        ts_utc=str(last["ts_utc"]),
        ema_20=float(last["ema_20"]),
        ema_50=float(last["ema_50"]),
        ema_200=float(last["ema_200"]),
        atr_14=float(last["atr_14"]),
        ret_1=float(last["ret_1"]),
        ret_5=float(last["ret_5"]),
        last_price=float(last["close"]),
    )
""",
    # Strategy
    "src/strategy/baseline_rules.py": """\
from __future__ import annotations
from src.common.schemas import Features, TradeIntent, Side


def propose_trade(features: Features) -> TradeIntent | None:
    price = features.last_price
    atr = max(features.atr_14, 0.01)

    stop_dist = 1.5 * atr
    tp_dist = 2.5 * atr

    if features.ema_20 > features.ema_50 and price > features.ema_200:
        return TradeIntent(
            symbol=features.symbol,
            side=Side.BUY,
            ts_utc=features.ts_utc,
            timeframe_min=features.timeframe_min,
            stop_distance=stop_dist,
            take_profit_distance=tp_dist,
            rule_confidence=0.60,
        )

    if features.ema_20 < features.ema_50 and price < features.ema_200:
        return TradeIntent(
            symbol=features.symbol,
            side=Side.SELL,
            ts_utc=features.ts_utc,
            timeframe_min=features.timeframe_min,
            stop_distance=stop_dist,
            take_profit_distance=tp_dist,
            rule_confidence=0.60,
        )

    return None
""",
    # ML
    "src/ml/infer.py": """\
from __future__ import annotations
from src.common.schemas import Features, TradeIntent, MLDecision
from src.common.settings import settings


def ml_filter(intent: TradeIntent, features: Features) -> MLDecision:
    if not settings.ML_ENABLED:
        return MLDecision(approved=True, prob=1.0, reason="ml_disabled")

    # Placeholder until MLflow model is wired:
    prob = float(intent.rule_confidence)
    approved = prob >= settings.ML_MIN_PROB
    return MLDecision(approved=approved, prob=prob, reason="ml_placeholder_rule_conf")
""",
    "src/ml/train.py": """\
from __future__ import annotations

def main():
    print("Training pipeline placeholder. Later: train + log to MLflow + promote model.")

if __name__ == "__main__":
    main()
""",
    # Risk (Apex EOD Evaluation enforcement)
    "src/risk/guardian.py": """\
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from src.common.schemas import TradeIntent, RiskDecision
from src.common.settings import settings


@dataclass
class ApexEODState:
    # Session identification
    session_date: date | None = None

    # Balances (USD)
    starting_balance: float = settings.ACCOUNT_SIZE
    session_start_balance: float = settings.ACCOUNT_SIZE
    current_balance: float = settings.ACCOUNT_SIZE

    # Active thresholds during the trading session (USD)
    eod_threshold_active: float = settings.ACCOUNT_SIZE - settings.EOD_MAX_DRAWDOWN_USD
    dll_floor_active: float = settings.ACCOUNT_SIZE - settings.DAILY_LOSS_LIMIT_USD

    # Tracking
    trades_today: int = 0
    open_contracts: int = 0
    trading_disabled: bool = False
    disable_reason: str = ""

    # Profit target
    profit_target_balance: float = settings.ACCOUNT_SIZE + settings.PROFIT_TARGET_USD

    def start_new_session(self, today: date, current_balance: float) -> None:
        self.session_date = today
        self.session_start_balance = float(current_balance)
        self.current_balance = float(current_balance)

        # DLL is fixed during the session (based on session start balance)
        self.dll_floor_active = self.session_start_balance - settings.DAILY_LOSS_LIMIT_USD

        self.trades_today = 0
        self.trading_disabled = False
        self.disable_reason = ""

    def on_balance_update(self, current_balance: float) -> None:
        self.current_balance = float(current_balance)

    def on_eod_close(self, eod_balance: float) -> None:
        # EOD threshold is recalculated once per day at market close and enforced next session
        eod_balance = float(eod_balance)
        self.eod_threshold_active = eod_balance - settings.EOD_MAX_DRAWDOWN_USD


def _kill(reason: str) -> RiskDecision:
    return RiskDecision(action="KILL", reason=reason, qty_contracts=0)


def _block(reason: str) -> RiskDecision:
    return RiskDecision(action="BLOCK", reason=reason, qty_contracts=0)


def evaluate_risk(intent: TradeIntent, state: ApexEODState) -> RiskDecision:
    if state.trading_disabled:
        return _block(f"trading_disabled:{state.disable_reason}")

    # Optional: stop trading when profit target reached
    if settings.STOP_TRADING_ON_TARGET and state.current_balance >= state.profit_target_balance:
        state.trading_disabled = True
        state.disable_reason = "profit_target_reached"
        return _block("profit_target_reached_stop_trading")

    # Rule 1: never touch/breach EOD threshold during session
    if state.current_balance <= state.eod_threshold_active:
        state.trading_disabled = True
        state.disable_reason = "eod_threshold_breached"
        return _kill("EOD_THRESHOLD_BREACHED_FAIL_EVAL")

    # Rule 2: DLL fixed during session; do not breach
    if state.current_balance <= state.dll_floor_active:
        state.trading_disabled = True
        state.disable_reason = "daily_loss_limit_breached"
        return _kill("DAILY_LOSS_LIMIT_BREACHED_FAIL_EVAL")

    # Rule 3: fixed position size during evaluation + max contracts
    if settings.CONTRACTS_PER_TRADE <= 0:
        return _block("contracts_per_trade_invalid")

    if settings.CONTRACTS_PER_TRADE > settings.MAX_CONTRACTS:
        return _block("contracts_per_trade_exceeds_max_contracts")

    if state.open_contracts + settings.CONTRACTS_PER_TRADE > settings.MAX_CONTRACTS:
        return _block("max_contracts_exposure_limit")

    # Safety: optional trade count cap (not Apex rule)
    if state.trades_today >= 20:
        return _block("safety_trade_count_cap")

    return RiskDecision(action="ALLOW", reason="apex_eod_ok", qty_contracts=settings.CONTRACTS_PER_TRADE)
""",
    # Execution
    "src/execution/executor.py": """\
from __future__ import annotations
from src.common.schemas import OrderCommand, FillEvent


def execute_order_paper(cmd: OrderCommand) -> FillEvent:
    return FillEvent(
        client_order_id=cmd.client_order_id,
        status="FILLED",
        fill_price=None,
        message="paper_fill",
    )
""",
    # Tradovate adapter placeholder
    "src/adapters/tradovate/client.py": """\
from __future__ import annotations

class TradovateClient:
    def __init__(self) -> None:
        raise NotImplementedError(
            "Tradovate adapter not wired yet. Will use vendor/Tradovate-Python-Client after cloning on VPS."
        )
""",
    # Monitoring
    "src/monitoring/health.py": """\
from __future__ import annotations
import time

def heartbeat(msg: str = "alive") -> None:
    print(f"[HEARTBEAT] {msg} @ {time.time()}")
""",
    # Script
    "scripts/run_paper.py": """\
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
from src.execution.executor import execute_order_paper
from src.monitoring.health import heartbeat


def fake_market_df(n: int = 300) -> pd.DataFrame:
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
    print("=== PAPER MODE ===")
    print(f"Symbol={settings.BOT_SYMBOL} TF={settings.TIMEFRAME_MINUTES}m")

    # Paper balance starts at account size
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
                        stop_loss_price=sl,
                        take_profit_price=tp,
                        client_order_id=str(uuid.uuid4()),
                    )

                    fill = execute_order_paper(cmd)
                    state.trades_today += 1
                    state.open_contracts += rd.qty_contracts
                    print(f"[{feats.ts_utc}] PAPER {cmd.side} {cmd.qty_contracts} {cmd.symbol} SL={sl:.2f} TP={tp:.2f} -> {fill.status}")

        heartbeat("paper_loop")
        time.sleep(5)


if __name__ == "__main__":
    main()
""",
    "scripts/run_workers_local.py": """\
print("Workers placeholder. Later: Redis + RQ workers for data/strategy/ml/risk/execution.")
""",
}

DIRS = [
    "config",
    "scripts",
    "src/common",
    "src/adapters/tradovate",
    "src/data",
    "src/strategy",
    "src/ml",
    "src/risk",
    "src/execution",
    "src/monitoring",
    "workers",
    "vendor",
]


def write_file(path: str, content: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def main() -> None:
    for d in DIRS:
        Path(d).mkdir(parents=True, exist_ok=True)

    for path, content in FILES.items():
        write_file(path, content)

    print("✅ Repo scaffold generated WITH Apex EOD (50k) rules.")
    print("Next (on VPS):")
    print("  1) python -m venv .venv && .venv\\Scripts\\activate")
    print("  2) pip install -r requirements.txt")
    print("  3) git submodule update --init --recursive")
    print("  4) copy .env.example to .env and fill values")
    print("  5) python scripts/run_paper.py")


if __name__ == "__main__":
    main()
