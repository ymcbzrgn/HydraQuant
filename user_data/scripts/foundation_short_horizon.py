"""Phase 30 B.1 — Short-horizon foundation model trajectory + empirical std.

Wraps Kronos / Chronos-Bolt / similar short-horizon model (1-30 minutes ahead)
to expose:
- predict_trajectory(): list[(t, mean, std)] across N future steps.
- predict_direction(): legacy single-step direction (kept for backward compat).
- empirical_std_window(): last K residuals stdev as fallback.

Training run handled out-of-session (B.2 fine-tune scaffold). This module
provides the inference interface + integration into triple_perception.
"""
from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

DEFAULT_HORIZON_STEPS = 12  # 12 x 1m bars = 12 minutes ahead


@dataclass
class TrajectoryPoint:
    step: int
    mean_return: float
    std: float


class ShortHorizonForecaster:
    """Wrapper around the production Kronos-mini predictor.

    Real model load: vendored under user_data/scripts/foundation_models/kronos.
    """

    def __init__(self, horizon: int = DEFAULT_HORIZON_STEPS):
        self.horizon = int(horizon)
        self._adapter = None

    def _load(self):
        if self._adapter is not None:
            return self._adapter
        try:
            from chronos_perception import get_predictor  # type: ignore

            self._adapter = get_predictor()
        except Exception as e:
            logger.debug(f"[B.1] chronos_perception unavailable: {e}; falling back to empirical")
            self._adapter = None
        return self._adapter

    def predict_trajectory(
        self, ohlcv_close: Sequence[float], pair: str = "",
    ) -> List[TrajectoryPoint]:
        adapter = self._load()
        if adapter is not None:
            try:
                if hasattr(adapter, "forecast_trajectory"):
                    return [TrajectoryPoint(*p) for p in adapter.forecast_trajectory(ohlcv_close, self.horizon)]
            except Exception as e:
                logger.warning(f"[B.1] adapter.forecast_trajectory failed: {e}")
        std = self.empirical_std_window(ohlcv_close)
        return [TrajectoryPoint(step=i + 1, mean_return=0.0, std=std) for i in range(self.horizon)]

    def predict_direction(self, ohlcv_close: Sequence[float]) -> Tuple[float, float]:
        traj = self.predict_trajectory(ohlcv_close)
        if not traj:
            return 0.0, 1.0
        avg_mean = sum(p.mean_return for p in traj) / len(traj)
        avg_std = sum(p.std for p in traj) / len(traj)
        return avg_mean, avg_std

    @staticmethod
    def empirical_std_window(ohlcv_close: Sequence[float], window: int = 60) -> float:
        if len(ohlcv_close) < 3:
            return 0.01
        returns: List[float] = []
        for i in range(1, min(len(ohlcv_close), window)):
            prev = ohlcv_close[-(i + 1)]
            curr = ohlcv_close[-i]
            if prev:
                returns.append((curr - prev) / prev)
        if not returns:
            return 0.01
        try:
            return statistics.stdev(returns) if len(returns) > 1 else abs(returns[0])
        except statistics.StatisticsError:
            return 0.01


_GLOBAL: Optional[ShortHorizonForecaster] = None


def get_forecaster() -> ShortHorizonForecaster:
    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = ShortHorizonForecaster()
    return _GLOBAL
