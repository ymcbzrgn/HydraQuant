"""Phase 30 B.3 — Long-horizon quantile 5th perception stage.

TimesFM-style long-horizon forecaster (4-24 hours ahead) producing quantile
distribution (q10, q50, q90). Plugged into triple_perception as 5th stage
between Conformal calibrator and DualAxis fusion.

Real model weights loaded out-of-session. Interface returns mock quantiles
based on EMA + empirical std when adapter is absent.
"""
from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)

DEFAULT_HORIZON_HOURS = 24


@dataclass
class QuantileForecast:
    horizon_hours: int
    q10: float
    q50: float
    q90: float
    confidence: float  # [0, 1]


class LongHorizonForecaster:
    def __init__(self, horizon_hours: int = DEFAULT_HORIZON_HOURS):
        self.horizon_hours = int(horizon_hours)
        self._adapter = None

    def _load(self):
        if self._adapter is not None:
            return self._adapter
        try:
            from foundation_models.timesfm_adapter import TimesFMAdapter  # type: ignore

            self._adapter = TimesFMAdapter()
        except Exception:
            self._adapter = None
        return self._adapter

    def forecast(
        self,
        ohlcv_1h_close: Sequence[float],
        regime: str = "unknown",
    ) -> QuantileForecast:
        adapter = self._load()
        if adapter is not None:
            try:
                q10, q50, q90, conf = adapter.forecast(ohlcv_1h_close, self.horizon_hours)
                return QuantileForecast(self.horizon_hours, q10, q50, q90, conf)
            except Exception as e:
                logger.warning(f"[B.3] adapter forecast failed: {e}")
        return self._fallback(ohlcv_1h_close)

    def _fallback(self, closes: Sequence[float]) -> QuantileForecast:
        if not closes or len(closes) < 24:
            return QuantileForecast(self.horizon_hours, 0.0, 0.0, 0.0, 0.0)
        ema = closes[-1]
        alpha = 0.2
        for c in reversed(closes[-48:]):
            ema = alpha * c + (1 - alpha) * ema
        returns = []
        for i in range(1, min(len(closes), 60)):
            prev = closes[-(i + 1)]
            curr = closes[-i]
            if prev:
                returns.append((curr - prev) / prev)
        std = statistics.stdev(returns) if len(returns) > 1 else 0.02
        q50 = (ema - closes[-1]) / closes[-1] if closes[-1] else 0.0
        return QuantileForecast(
            self.horizon_hours,
            q10=q50 - 1.28 * std,
            q50=q50,
            q90=q50 + 1.28 * std,
            confidence=0.3,
        )


_GLOBAL: Optional[LongHorizonForecaster] = None


def get_forecaster(horizon_hours: int = DEFAULT_HORIZON_HOURS) -> LongHorizonForecaster:
    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = LongHorizonForecaster(horizon_hours=horizon_hours)
    return _GLOBAL
