from __future__ import annotations
from pydantic_settings import BaseSettings, SettingsConfigDict
from src.common.schemas import Mode


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    MODE: Mode = Mode.PAPER
    BOT_SYMBOL: str = "MBT"
    TIMEFRAME_MINUTES: int = 15

    # Apex EOD Eval (50k)
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
    SIGNAL_MODEL_PATH: str = "models/signal_model.joblib"

    # Tradovate REST Option A
    TRADOVATE_ENV: str = "demo"
    TRADOVATE_BASE_DEMO: str = "https://demo.tradovateapi.com/v1"
    TRADOVATE_BASE_LIVE: str = "https://live.tradovateapi.com/v1"

    TRADOVATE_USERNAME: str = ""
    TRADOVATE_PASSWORD: str = ""
    TRADOVATE_APP_ID: str = ""
    TRADOVATE_APP_VERSION: str = "1.0"
    TRADOVATE_CID: str = ""
    TRADOVATE_DEVICE_ID: str = ""
    TRADOVATE_API_SECRET: str = ""
    TRADOVATE_ACCOUNT_ID: str = ""
    TRADOVATE_SYMBOL: str = "MBT"


settings = Settings()
