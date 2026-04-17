"""
Phase 26: Chronos-Bolt-Small Perception Module

Amazon Chronos-Bolt — Transformer-based probabilistic time series forecaster.
Input: Univariate close prices → Output: Quantile predictions P10/P50/P90.

Key properties:
  - 48M parameters, ~100ms inference, ~191MB RAM (mmap load)
  - Native quantile output: P10, P50, P90 (uncertainty distribution)
  - CPU-viable via bfloat16
  - TimesFM alternative (TimesFM NOT viable on CPU per arXiv 2602.10848)

Model: amazon/chronos-bolt-small
Library: chronos-forecasting
"""

import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# Lazy-loaded model singleton
_chronos_pipeline = None
_chronos_loaded = False


def _load_chronos():
    """Lazy-load Chronos-Bolt pipeline (singleton, first call only)."""
    global _chronos_pipeline, _chronos_loaded
    if _chronos_loaded:
        return _chronos_pipeline

    _chronos_loaded = True
    try:
        import torch
        from chronos import ChronosBoltPipeline

        model_id = "amazon/chronos-bolt-small"
        logger.info(f"[Chronos] Loading model {model_id}...")
        start = time.time()

        _chronos_pipeline = ChronosBoltPipeline.from_pretrained(
            model_id,
            device_map="cpu",
        )

        elapsed = time.time() - start
        logger.info(f"[Chronos] Loaded in {elapsed:.1f}s (CPU, bfloat16)")
        return _chronos_pipeline

    except Exception as e:
        logger.warning(f"[Chronos] Failed to load: {e}. Using fallback quantiles.")
        _chronos_pipeline = None
        return None


def compute_chronos_quantiles(
    df: pd.DataFrame,
    prediction_length: int = 6,
    context_length: int = 200,
) -> Dict[str, float]:
    """Compute quantile predictions using Chronos-Bolt.

    Args:
        df: DataFrame with 'close' column
        prediction_length: How many future candles to predict (default 6 = 6 hours for 1h TF)
        context_length: How many past candles to use as context

    Returns:
        {
            "p10": float — 10th percentile predicted change (pessimistic)
            "p50": float — median predicted change
            "p90": float — 90th percentile predicted change (optimistic)
            "interval_width": float — P90 - P10 (narrow = certain, wide = uncertain)
            "direction_confidence": float — how much the distribution leans one way
            "available": bool
        }
    """
    result = {
        "p10": 0.0, "p50": 0.0, "p90": 0.0,
        "interval_width": 0.1,  # default moderate uncertainty
        "direction_confidence": 0.0,
        "available": False,
    }

    pipeline = _load_chronos()

    if pipeline is None:
        return _fallback_quantiles(df, prediction_length)

    try:
        import torch

        # Prepare context: last N close prices
        n = min(len(df), context_length)
        if n < 20:
            return _fallback_quantiles(df, prediction_length)

        close_prices = df["close"].iloc[-n:].values.astype(np.float64)
        context = torch.tensor(close_prices, dtype=torch.float32)

        # Inference
        start = time.time()
        forecast = pipeline.predict(
            context,  # positional — ChronosBoltPipeline expects positional, not keyword
            prediction_length,
        )
        elapsed_ms = (time.time() - start) * 1000

        # forecast shape: (1, num_quantiles, prediction_length)
        # Extract quantiles — Chronos outputs multiple quantile levels
        forecast_np = forecast.numpy() if hasattr(forecast, 'numpy') else np.array(forecast)

        # Chronos-Bolt output: (batch, 9, prediction_length)
        # 9 quantiles: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        # Index: 0=P10, 1=P20, ..., 4=P50, ..., 8=P90
        if forecast_np.ndim == 3:
            quantile_forecasts = forecast_np[0]  # (9, prediction_length)
            n_quantiles = quantile_forecasts.shape[0]

            if n_quantiles == 9:
                p10_forecast = quantile_forecasts[0].mean()   # index 0 = P10
                p50_forecast = quantile_forecasts[4].mean()   # index 4 = P50 (median)
                p90_forecast = quantile_forecasts[8].mean()   # index 8 = P90
            elif n_quantiles >= 3:
                p10_forecast = quantile_forecasts[0].mean()
                p50_forecast = quantile_forecasts[n_quantiles // 2].mean()
                p90_forecast = quantile_forecasts[-1].mean()
            else:
                p50_forecast = quantile_forecasts[0].mean()
                p10_forecast = p50_forecast * 0.97
                p90_forecast = p50_forecast * 1.03
        elif forecast_np.ndim == 2:
            # Fallback: treat as (samples, prediction_length)
            all_means = forecast_np.mean(axis=1)
            p10_forecast = float(np.percentile(all_means, 10))
            p50_forecast = float(np.percentile(all_means, 50))
            p90_forecast = float(np.percentile(all_means, 90))
        else:
            return _fallback_quantiles(df, prediction_length)

        # Convert to percentage change relative to current price
        current_price = close_prices[-1]
        if current_price > 0:
            p10_pct = (p10_forecast - current_price) / current_price
            p50_pct = (p50_forecast - current_price) / current_price
            p90_pct = (p90_forecast - current_price) / current_price
        else:
            p10_pct = p50_pct = p90_pct = 0.0

        interval_width = p90_pct - p10_pct

        # Direction confidence: how skewed is the distribution?
        # Positive = bullish lean, negative = bearish lean
        if interval_width > 1e-8:
            direction_confidence = (p50_pct - (p10_pct + p90_pct) / 2) / interval_width
        else:
            direction_confidence = 0.0

        result["p10"] = round(float(p10_pct), 6)
        result["p50"] = round(float(p50_pct), 6)
        result["p90"] = round(float(p90_pct), 6)
        result["interval_width"] = round(float(max(interval_width, 1e-6)), 6)
        result["direction_confidence"] = round(float(np.clip(direction_confidence, -1, 1)), 4)
        result["available"] = True

        logger.debug(
            f"[Chronos] Inference: {elapsed_ms:.0f}ms, "
            f"P10={p10_pct:.4f} P50={p50_pct:.4f} P90={p90_pct:.4f} "
            f"width={interval_width:.4f}"
        )

    except Exception as e:
        logger.warning(f"[Chronos] Inference failed: {e}. Using fallback.")
        return _fallback_quantiles(df, prediction_length)

    return result


def _fallback_quantiles(df: pd.DataFrame, prediction_length: int) -> Dict[str, float]:
    """Statistical fallback when Chronos model is unavailable.
    Uses historical return distribution to estimate quantiles."""
    result = {
        "p10": 0.0, "p50": 0.0, "p90": 0.0,
        "interval_width": 0.1,
        "direction_confidence": 0.0,
        "available": False,
    }

    if len(df) < 30:
        return result

    try:
        close = df["close"].values
        # Calculate rolling returns over prediction_length candles
        returns = np.diff(np.log(close + 1e-10))

        if len(returns) < 20:
            return result

        # Use recent volatility to estimate future distribution
        recent_returns = returns[-50:] if len(returns) >= 50 else returns
        vol = np.std(recent_returns)
        drift = np.mean(recent_returns)

        # Scale to prediction horizon
        horizon_drift = drift * prediction_length
        horizon_vol = vol * np.sqrt(prediction_length)

        # Quantiles from normal approximation
        result["p10"] = round(float(horizon_drift - 1.28 * horizon_vol), 6)
        result["p50"] = round(float(horizon_drift), 6)
        result["p90"] = round(float(horizon_drift + 1.28 * horizon_vol), 6)
        result["interval_width"] = round(float(2.56 * horizon_vol), 6)

        if horizon_vol > 1e-8:
            result["direction_confidence"] = round(float(np.clip(horizon_drift / horizon_vol, -1, 1)), 4)

    except Exception as e:
        logger.warning(f"[Chronos] Fallback quantiles failed: {e}")

    return result


# Phase 27 Task 20 (C5 Ajani): Chronos-Bolt BitFit hook.
# Scheduler calls this weekly. Full BitFit bias-only update (768 params total
# per NeurIPS 2024 arXiv 2409.11302) is Sprint 3B task 20b — today we just
# record readiness so the pipeline surface exists without the risky half-
# trained weights in prod.
def bitfit_bias_update_if_available(min_samples: int = 50) -> Dict:
    """Chronos-Bolt BitFit bias-only update stub.

    BitFit (Zaken 2022) freezes all weights and updates ONLY the bias tensors,
    which in Chronos-Base is ~768 parameters (vs. 48M full fine-tune). That
    small a footprint is why BitFit beat LoRA for MSE on Chronos in the
    NeurIPS 2024 benchmark. Implementation pending Sprint 3B task 20b.
    """
    import os
    try:
        from ai_config import AI_DB_PATH
        from db import get_db_connection
        conn = get_db_connection(AI_DB_PATH)
        n = conn.execute(
            "SELECT COUNT(*) AS n FROM ai_decisions WHERE outcome_pnl IS NOT NULL"
        ).fetchone()["n"] or 0
        conn.close()
    except Exception as e:
        return {"status": "skipped", "reason": f"db: {e}"}

    if n < min_samples:
        return {"status": "skipped", "reason": f"insufficient samples (n={n})",
                "min_samples": min_samples}

    sidecar = os.path.join(
        os.path.dirname(__file__), "..", "models", "chronos_bitfit_pending.flag"
    )
    try:
        os.makedirs(os.path.dirname(sidecar), exist_ok=True)
        with open(sidecar, "w") as f:
            f.write(f"pending_trades={n}\n")
    except Exception:
        pass
    return {"status": "pending_impl", "samples_available": n,
            "message": "BitFit bias update planned Sprint 3B task 20b"}
