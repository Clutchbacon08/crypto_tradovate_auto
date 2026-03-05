from __future__ import annotations

import os
import joblib
import pandas as pd

from src.common.settings import settings
from src.common.schemas import Features, TradeIntent, MLDecision

_MODEL_PACK = None
MODEL_PATH = os.getenv("SIGNAL_MODEL_PATH", "models/signal_model.joblib")


def _load_model():
    global _MODEL_PACK
    if _MODEL_PACK is None:
        if not os.path.exists(MODEL_PATH):
            _MODEL_PACK = {"model": None, "feature_cols": []}
        else:
            _MODEL_PACK = joblib.load(MODEL_PATH)
    return _MODEL_PACK


def ml_filter(intent: TradeIntent, features: Features) -> MLDecision:
    if not settings.ML_ENABLED:
        return MLDecision(approved=True, prob=1.0, reason="ml_disabled")

    pack = _load_model()
    model = pack.get("model")
    cols = pack.get("feature_cols", [])

    # No model yet -> allow (for early testing). Flip to block if you prefer.
    if model is None or not cols:
        return MLDecision(approved=True, prob=0.5, reason="model_missing_allow")

    # Feature row must match train_signal_model.py feature_cols.
    row = {
        "ret_1": features.ret_1,
        "ret_5": features.ret_5,
        "atr_14": features.atr_14,
        "ema_fast_minus_slow": features.ema_20 - features.ema_50,  # proxy until live computes chosen params
        "ema200_dist": features.last_price - features.ema_200,
        "vol_zscore": 0.0,
        "trend_strength": 0.0,
        "stop_atr": intent.stop_distance / max(features.atr_14, 1e-9),
        "tp_atr": intent.take_profit_distance / max(features.atr_14, 1e-9),
        "cooldown": 0,
    }

    X = pd.DataFrame([row])[cols]
    prob = float(model.predict_proba(X)[0, 1])
    approved = prob >= settings.ML_MIN_PROB
    return MLDecision(approved=approved, prob=prob, reason="signal_model")
