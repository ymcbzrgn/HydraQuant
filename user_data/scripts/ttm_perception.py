"""
Phase 26: TTM (Tiny Time Mixer) Perception Module

IBM Granite TTM — MLP-Mixer based time series model.
Input: Multivariate OHLCV + indicators → Output: 64-dim embedding + directional signal.

Key properties:
  - 1M parameters, <10ms inference, ~20MB RAM
  - Multivariate: captures cross-channel interactions (price-volume-RSI)
  - CPU-native, no GPU required
  - Embedding extracted from last hidden state

Model: ibm-granite/granite-timeseries-ttm-r2
Library: tsfm_public (IBM Toolkit) or transformers
"""

import logging
import time
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Lazy-loaded model singleton
_ttm_model = None
_ttm_loaded = False
_TTM_EMBEDDING_DIM = 64  # Target embedding dimension


def _load_ttm():
    """Lazy-load TTM model (singleton, first call only)."""
    global _ttm_model, _ttm_loaded
    if _ttm_loaded:
        return _ttm_model

    _ttm_loaded = True
    try:
        from tsfm_public.toolkit.get_model import get_model
        import torch

        model_id = "ibm-granite/granite-timeseries-ttm-r2"

        logger.info(f"[TTM] Loading model {model_id}...")
        start = time.time()

        _ttm_model = get_model(
            model_path=model_id,
            context_length=512,
            prediction_length=96,
            freq_prefix_tuning=False,
            prefer_longer_context=True,
        )
        _ttm_model.eval()

        elapsed = time.time() - start
        param_count = sum(p.numel() for p in _ttm_model.parameters())
        logger.info(f"[TTM] Loaded: {param_count/1e6:.1f}M params in {elapsed:.1f}s (granite-tsfm)")
        return _ttm_model

    except Exception as e:
        logger.warning(f"[TTM] Failed to load: {e}. Using fallback embedding.")
        _ttm_model = None
        return None


def compute_ttm_embedding(
    df: pd.DataFrame,
    context_length: int = 512,
    target_dim: int = _TTM_EMBEDDING_DIM,
) -> Dict[str, any]:
    """Compute TTM embedding from OHLCV DataFrame.

    Args:
        df: DataFrame with columns: open, high, low, close, volume (+ optional rsi, adx)
        context_length: Number of past candles to use (default 512 for TTM-B)
        target_dim: Target embedding dimension (default 64)

    Returns:
        {
            "embedding": np.ndarray of shape (target_dim,),
            "direction": float (-1.0 to 1.0, negative=bearish, positive=bullish),
            "magnitude": float (0.0 to 1.0, strength of signal),
            "available": bool
        }
    """
    result = {
        "embedding": np.zeros(target_dim),
        "direction": 0.0,
        "magnitude": 0.0,
        "available": False,
    }

    model = _load_ttm()

    if model is None:
        # Fallback: simple statistical embedding from raw data
        return _fallback_embedding(df, target_dim)

    try:
        import torch

        # Build multivariate input: select channels
        channels = _prepare_channels(df, context_length)
        if channels is None:
            return _fallback_embedding(df, target_dim)

        # Standard scale each channel independently (TTM requirement)
        scaled = _standard_scale(channels)

        # Shape: (1, context_length, num_channels)
        input_tensor = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)

        # Inference — TTM uses past_values= and output_hidden_states=True
        with torch.no_grad():
            start = time.time()
            outputs = model(past_values=input_tensor, output_hidden_states=True)
            elapsed_ms = (time.time() - start) * 1000

        # Extract embedding from backbone_hidden_state
        # Shape: (batch, num_channels, 8_patches, 192_hidden_dim)
        # Flatten patches × hidden → pool across channels → project to target_dim
        if hasattr(outputs, 'backbone_hidden_state') and outputs.backbone_hidden_state is not None:
            bhs = outputs.backbone_hidden_state  # (1, C, 8, 192)
            # Average across channels, flatten patches × hidden
            bhs_avg = bhs.mean(dim=1)  # (1, 8, 192)
            bhs_flat = bhs_avg.reshape(1, -1).numpy()[0]  # (1536,)
            # Project to target_dim via PCA-like truncation (take first target_dim components)
            # Better: learned linear projection, but for now truncate + normalize
            embedding = bhs_flat[:target_dim] if len(bhs_flat) >= target_dim else np.pad(bhs_flat, (0, target_dim - len(bhs_flat)))
        elif hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
            lhs = outputs.last_hidden_state[0, -1, :].numpy()
            embedding = lhs[:target_dim] if len(lhs) >= target_dim else np.pad(lhs, (0, target_dim - len(lhs)))
        else:
            embedding = np.zeros(target_dim)

        # Normalize embedding to unit vector
        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            embedding = embedding / norm

        # Directional signal from forecast
        direction, magnitude = _extract_direction(outputs, df)

        result["embedding"] = embedding
        result["direction"] = direction
        result["magnitude"] = magnitude
        result["available"] = True

        logger.debug(f"[TTM] Inference: {elapsed_ms:.1f}ms, dir={direction:.2f}, mag={magnitude:.2f}")

    except Exception as e:
        logger.warning(f"[TTM] Inference failed: {e}. Using fallback.")
        return _fallback_embedding(df, target_dim)

    return result


def _prepare_channels(df: pd.DataFrame, context_length: int) -> Optional[np.ndarray]:
    """Prepare multivariate input channels from DataFrame."""
    if len(df) < 50:  # minimum viable
        return None

    # Use up to context_length candles (pad if less)
    n = min(len(df), context_length)

    # Core channels: close, volume, + any available indicators
    channel_cols = ["close", "volume"]
    for col in ["rsi", "adx", "atr", "macd", "bb_upper", "bb_lower"]:
        if col in df.columns:
            channel_cols.append(col)

    data = df[channel_cols].iloc[-n:].values.astype(np.float64)

    # Handle NaN — forward fill then zero fill
    for j in range(data.shape[1]):
        col_data = data[:, j]
        mask = np.isnan(col_data)
        if mask.any():
            # Forward fill
            for i in range(1, len(col_data)):
                if mask[i]:
                    col_data[i] = col_data[i - 1]
            # Fill remaining leading NaN with first valid value
            first_valid = col_data[~np.isnan(col_data)]
            if len(first_valid) > 0:
                col_data[np.isnan(col_data)] = first_valid[0]
            else:
                col_data[:] = 0.0

    # Pad to context_length if needed (left-pad with first values)
    if data.shape[0] < context_length:
        pad_rows = context_length - data.shape[0]
        pad = np.tile(data[0:1, :], (pad_rows, 1))
        data = np.vstack([pad, data])

    return data


def _standard_scale(data: np.ndarray) -> np.ndarray:
    """Standard scale each channel independently (TTM requirement)."""
    scaled = np.zeros_like(data)
    for j in range(data.shape[1]):
        col = data[:, j]
        mean = col.mean()
        std = col.std()
        if std > 1e-10:
            scaled[:, j] = (col - mean) / std
        else:
            scaled[:, j] = 0.0
    return scaled


def _extract_direction(outputs, df: pd.DataFrame) -> Tuple[float, float]:
    """Extract directional signal from TTM output."""
    try:
        # TTM outputs prediction_outputs: (batch, prediction_length, channels)
        if hasattr(outputs, 'prediction_outputs') and outputs.prediction_outputs is not None:
            forecast = outputs.prediction_outputs[0, :, 0].detach().numpy()  # channel 0 = close
        elif isinstance(outputs, tuple) and len(outputs) > 0:
            out = outputs[0]
            forecast = out[0, :].detach().numpy() if hasattr(out, 'detach') else None
        else:
            forecast = None

        if forecast is not None and len(forecast) > 0:
            # Direction: is forecast going up or down relative to last close?
            current = df["close"].iloc[-1]
            forecast_mean = forecast.mean()
            if current > 0:
                direction = np.clip((forecast_mean - current) / current * 100, -1.0, 1.0)
                magnitude = min(abs(direction), 1.0)
                return float(direction), float(magnitude)
    except Exception:
        pass

    # Fallback: use simple momentum from last 10 candles
    if len(df) >= 10:
        returns = df["close"].pct_change().iloc[-10:]
        momentum = returns.mean()
        direction = np.clip(momentum * 100, -1.0, 1.0)
        return float(direction), float(min(abs(direction), 1.0))

    return 0.0, 0.0


def _fallback_embedding(df: pd.DataFrame, target_dim: int) -> Dict:
    """Statistical fallback when TTM model is unavailable.
    Creates a simple feature vector from raw price/volume statistics."""
    embedding = np.zeros(target_dim)

    if len(df) < 20:
        return {"embedding": embedding, "direction": 0.0, "magnitude": 0.0, "available": False}

    try:
        close = df["close"].values[-100:] if len(df) >= 100 else df["close"].values
        vol = df["volume"].values[-100:] if len(df) >= 100 else df["volume"].values

        # Basic statistical features
        features = [
            np.mean(close), np.std(close), np.median(close),
            np.mean(vol), np.std(vol),
            close[-1] / close[0] - 1 if close[0] > 0 else 0,  # total return
            np.corrcoef(close[:-1], close[1:])[0, 1] if len(close) > 1 else 0,  # autocorrelation
        ]

        # Returns statistics
        returns = np.diff(np.log(close + 1e-10))
        if len(returns) > 0:
            features.extend([
                np.mean(returns), np.std(returns),
                np.percentile(returns, 10), np.percentile(returns, 90),
            ])

        # Pad/truncate to target_dim
        features = np.array(features[:target_dim])
        if len(features) < target_dim:
            features = np.pad(features, (0, target_dim - len(features)))

        # Normalize
        norm = np.linalg.norm(features)
        if norm > 1e-8:
            features = features / norm

        # Direction from momentum
        direction = float(np.clip(np.mean(np.diff(close[-10:])) / (close[-1] + 1e-10) * 100, -1.0, 1.0))
        magnitude = float(min(abs(direction), 1.0))

        return {
            "embedding": features,
            "direction": direction,
            "magnitude": magnitude,
            "available": False,  # fallback, not real TTM
        }

    except Exception as e:
        logger.warning(f"[TTM] Fallback embedding failed: {e}")
        return {"embedding": np.zeros(target_dim), "direction": 0.0, "magnitude": 0.0, "available": False}


# Phase 27 Task 20 (C5 Ajani): foundation-model fine-tuning hook.
# The scheduler calls this weekly. Full head retrain + A/B swap is Sprint 3B
# (task 20b); today we just record the call so the pipeline surface exists
# and future wiring replaces the NotImplemented body without touching the
# scheduler contract.
def retrain_head_if_available(min_samples: int = 50) -> Dict:
    """TTM head retraining stub — intentionally conservative.

    Checks for ≥ `min_samples` labelled trades in ai_decisions and, if the
    IBM few-shot fine-tune package becomes available, trains the final
    classifier head on them (backbone frozen). Sidecar output is written to
    `user_data/models/ttm_head_ft.pt`; the current production pipeline keeps
    using the zero-shot TTM embedding until a manual A/B test swaps it in.
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

    # Fine-tune package is a Sprint 3B task — stay a no-op for now so the
    # scheduler can report "ran successfully" but we don't ship a risky
    # half-trained head to production. Future: wire into
    # granite-timeseries/TSTM BitFit + final-layer MSE loop here.
    sidecar = os.path.join(
        os.path.dirname(__file__), "..", "models", "ttm_head_ft_pending.flag"
    )
    try:
        os.makedirs(os.path.dirname(sidecar), exist_ok=True)
        with open(sidecar, "w") as f:
            f.write(f"pending_trades={n}\n")
    except Exception:
        pass
    return {"status": "pending_impl", "samples_available": n,
            "message": "head retrain planned Sprint 3B task 20b"}
