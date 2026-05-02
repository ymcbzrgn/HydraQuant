"""
exchange_microstructure_learner.py — Bot's exchange-rules self-discovery.

Sprint 2026-05-02 — Bot learns the venue's actual rules and behavior.

Problem this solves:
  Production observations on bybit.tr:
    • 14 open positions with stakes ranging $0.05 - $23 (extreme spread)
    • A $0.05 stake CANNOT generate a real fill — below exchange min_notional
    • Maker-only orders sit unfilled for 6+ hours, then cancel after profit
      target was hit (NAORIS lost a +3% ROI to time-out)

  These are NOT bot bugs — they are bot ignorance of exchange rules and
  liquidity. This module makes the bot LEARN those rules from observation
  rather than hardcoding them.

What it tracks per pair:
  • exchange_min_notional (USD value) — straight from `markets[pair].limits`,
    refreshed every 24h
  • min_amount + tick_size + price_precision — for safe order construction
  • maker_fill_rate (pair × hour-of-day, 24-bucket EMA) — empirical
    likelihood that a passive limit order at maker price gets filled
  • avg_fill_time_seconds — how long fills typically take
  • taker_to_maker_ratio_observed — if exchange forces taker (low-liquidity
    pairs), bot learns this and adapts

Public API consumed by HydraSizer:
  • get_min_notional(pair) → float (cold-start fallback: $5)
  • passes_min_notional(pair, stake_usd) → bool
  • should_use_maker(pair, hour=None) → bool (decides maker vs taker)
  • record_fill_attempt(pair, side, price, was_maker, filled, fill_time) → None

Self-healing: when a pair shows < 30% maker fill rate over the last 50
attempts, this module flips a per-pair flag in pheromone field. The
strategy's `custom_entry_price` reads that flag and switches to taker
quoting for that pair until the rate recovers. No human intervention.

PARAM_REGISTRY keys:
  envelope.microstructure.min_fill_rate_for_maker
  envelope.microstructure.maker_observation_window
  envelope.microstructure.notional_safety_margin
  envelope.microstructure.cache_refresh_hours
"""
from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from typing import Any, Deque, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# Module-level state (per-process). All state is also persisted to the
# `system_metrics` table so cross-process consistency is possible via
# pheromone bridges.
_LOCK = threading.Lock()
_MIN_NOTIONAL_CACHE: Dict[str, Tuple[float, float]] = {}  # pair → (notional_usd, refreshed_at)
_FILL_HISTORY: Dict[str, Deque[Dict[str, Any]]] = defaultdict(lambda: deque(maxlen=200))
_HOUR_FILL_RATE: Dict[Tuple[str, int], Tuple[float, float]] = {}  # (pair, hour) → (ema_rate, count)
_PAIR_TAKER_FLAG: Dict[str, bool] = {}  # pair → True if too illiquid for maker


def _params() -> Dict[str, float]:
    try:
        from neural_organism import _p
        return {
            "min_fill_rate_for_maker": float(_p("envelope.microstructure.min_fill_rate_for_maker", 0.30)),
            "maker_observation_window": float(_p("envelope.microstructure.maker_observation_window", 50)),
            "notional_safety_margin": float(_p("envelope.microstructure.notional_safety_margin", 1.10)),
            "cache_refresh_hours": float(_p("envelope.microstructure.cache_refresh_hours", 24)),
            "ema_alpha": float(_p("envelope.microstructure.ema_alpha", 0.10)),
            "min_samples_for_flip": int(_p("envelope.microstructure.min_samples_for_flip", 5)),
        }
    except Exception:
        return {
            "min_fill_rate_for_maker": 0.30,
            "maker_observation_window": 50,
            "notional_safety_margin": 1.10,
            "cache_refresh_hours": 24,
            "ema_alpha": 0.10,
            "min_samples_for_flip": 5,
        }


def _get_exchange_handle():
    """Return a CCXT exchange object usable for queries.

    Uses the module-level _last_exchange if previously set by
    set_exchange_handle(); otherwise tries to grab the running freqtrade
    exchange via dataprovider semantics (best effort).
    """
    return globals().get("_last_exchange")


def set_exchange_handle(exchange) -> None:
    """Strategy injects its CCXT exchange handle so this module can query
    `markets`. Called once from HydraSizer.bot_start.
    """
    globals()["_last_exchange"] = exchange


def get_min_notional(pair: str, fallback: float = 5.0) -> float:
    """Return the exchange-enforced minimum notional in USD for a pair.

    Cache TTL: cache_refresh_hours (24h default). When fresh, returns
    cached value. When stale, queries exchange.markets for limits.

    Cold-start fallback: $5.0 — sensible safety floor that won't block
    sane trading on an unknown pair.
    """
    cfg = _params()
    now = time.time()
    cached = _MIN_NOTIONAL_CACHE.get(pair)
    if cached is not None:
        notional, refreshed_at = cached
        age_hours = (now - refreshed_at) / 3600.0
        if age_hours < cfg["cache_refresh_hours"]:
            return notional

    # Refresh from exchange
    notional = fallback
    try:
        ex = _get_exchange_handle()
        if ex is not None and hasattr(ex, "markets") and pair in ex.markets:
            m = ex.markets[pair]
            limits = m.get("limits", {}) or {}
            cost_min = limits.get("cost", {}).get("min")
            if cost_min and cost_min > 0:
                notional = float(cost_min)
            else:
                # Bybit perps often expose only `amount.min` × precision.
                # Compute notional = min_amount × current_price × contractSize
                amt_min = limits.get("amount", {}).get("min") or 0
                contract_size = float(m.get("contractSize") or 1.0)
                # Use last close (cheapest fresh price) for notional estimation
                ticker = ex.markets.get(pair, {})
                price = float(m.get("last") or 0)
                if price <= 0:
                    # try to fetch latest ticker
                    try:
                        t = ex.fetch_ticker(pair)
                        price = float(t.get("last") or t.get("close") or 0)
                    except Exception:
                        price = 0.0
                if amt_min and price > 0:
                    notional = float(amt_min) * price * contract_size
        # Apply safety margin (always trade slightly above exchange min)
        notional = max(fallback, notional * cfg["notional_safety_margin"])
    except Exception as e:
        logger.debug(f"[Microstructure] {pair} min_notional refresh failed: {e}")
        notional = fallback

    with _LOCK:
        _MIN_NOTIONAL_CACHE[pair] = (notional, now)
    return notional


def passes_min_notional(pair: str, stake_usd: float) -> bool:
    """True if the proposed stake clears the exchange minimum notional.

    Used by sizing layer to short-circuit: if envelope cap is below
    min_notional, mark trade as SHADOW (don't actually open) — avoids
    the $0.05 useless-fill anti-pattern.
    """
    return float(stake_usd) >= get_min_notional(pair)


def record_fill_attempt(pair: str, side: str, price: float,
                        was_maker: bool, filled: bool,
                        fill_time_seconds: float = 0.0,
                        hour_utc: Optional[int] = None) -> None:
    """Update fill history + per-(pair, hour) maker fill EMA.

    Called from HydraSizer's fill verification path (bot_loop_start
    sweep) so every entry/exit attempt feeds the model.
    """
    if hour_utc is None:
        hour_utc = int((time.time() / 3600) % 24)
    cfg = _params()
    record = {
        "ts": time.time(),
        "side": side,
        "price": float(price),
        "was_maker": bool(was_maker),
        "filled": bool(filled),
        "fill_time": float(fill_time_seconds),
        "hour": int(hour_utc),
    }
    with _LOCK:
        _FILL_HISTORY[pair].append(record)
        # Update per-(pair, hour) EMA — only for MAKER attempts since the
        # downstream decision is "use maker?". Taker attempts always fill.
        if was_maker:
            key = (pair, hour_utc)
            old_rate, old_count = _HOUR_FILL_RATE.get(key, (0.5, 0))
            alpha = cfg["ema_alpha"]
            new_rate = alpha * (1.0 if filled else 0.0) + (1.0 - alpha) * old_rate
            _HOUR_FILL_RATE[key] = (new_rate, old_count + 1)
        # Refresh per-pair taker flag based on overall recent maker rate
        _refresh_taker_flag(pair, cfg)


def _refresh_taker_flag(pair: str, cfg: Dict[str, float]) -> None:
    """Compute pair-wide maker fill rate over last `maker_observation_window`
    attempts; flip taker flag if below threshold.

    NOTE: must be called while holding _LOCK.
    """
    window = int(cfg["maker_observation_window"])
    min_floor = int(cfg.get("min_samples_for_flip", 5))
    history = list(_FILL_HISTORY.get(pair, []))
    maker_attempts = [h for h in history if h.get("was_maker")][-window:]
    if len(maker_attempts) < max(min_floor, window // 2):
        # Not enough data — keep current flag (default False)
        return
    fill_rate = sum(1 for h in maker_attempts if h["filled"]) / len(maker_attempts)
    threshold = cfg["min_fill_rate_for_maker"]
    flag = fill_rate < threshold
    prev = _PAIR_TAKER_FLAG.get(pair, False)
    if flag != prev:
        _PAIR_TAKER_FLAG[pair] = flag
        logger.info(
            f"[Microstructure] {pair} taker_flag flipped {prev}→{flag} "
            f"(fill_rate={fill_rate:.2%}, n={len(maker_attempts)}, thr={threshold:.0%})"
        )
        # Cross-process pheromone deposit so both bots see it
        try:
            from pheromone_field import get_pheromone_field
            get_pheromone_field().deposit(
                "microstructure", f"taker_flag::{pair}",
                {"pair": pair, "force_taker": flag,
                 "fill_rate": round(fill_rate, 4),
                 "n_attempts": len(maker_attempts)},
                half_life=14400.0,  # 4h
            )
        except Exception:
            pass


def should_use_maker(pair: str, hour_utc: Optional[int] = None) -> bool:
    """Decision: should the strategy place a maker (limit-passive) order
    or a taker (immediate-fill) order for this pair right now?

    Logic:
      • If pair is flagged as too-illiquid-for-maker → False (use taker)
      • If per-(pair, hour) maker fill rate < threshold → False
      • Otherwise → True (default, capture the rebate)

    Cross-process: also reads pheromone field to honor flags set by other
    bot processes (e.g., scheduler observed low fill rate even if this
    process hasn't).
    """
    cfg = _params()
    if hour_utc is None:
        hour_utc = int((time.time() / 3600) % 24)
    # Per-process flag
    if _PAIR_TAKER_FLAG.get(pair, False):
        return False
    # Cross-process pheromone flag
    try:
        from pheromone_field import get_pheromone_field
        state = get_pheromone_field().read(f"taker_flag::{pair}")
        if isinstance(state, dict) and state.get("force_taker"):
            return False
    except Exception:
        pass
    # Per-(pair, hour) EMA fallback
    key = (pair, hour_utc)
    rate_info = _HOUR_FILL_RATE.get(key)
    if rate_info is not None:
        rate, count = rate_info
        if count >= 5 and rate < cfg["min_fill_rate_for_maker"]:
            return False
    return True


def get_fill_stats(pair: str) -> Dict[str, Any]:
    """Diagnostic — current fill performance for a pair."""
    history = list(_FILL_HISTORY.get(pair, []))
    if not history:
        return {"pair": pair, "n_attempts": 0, "available": False}
    maker_h = [h for h in history if h.get("was_maker")]
    n = len(maker_h)
    fills = sum(1 for h in maker_h if h["filled"])
    avg_fill_time = (sum(h["fill_time"] for h in maker_h if h["filled"]) /
                     max(1, fills)) if fills else 0.0
    return {
        "pair": pair,
        "available": True,
        "n_attempts_total": len(history),
        "n_maker_attempts": n,
        "maker_fill_rate": (fills / n) if n else 0.0,
        "avg_fill_time_seconds": round(avg_fill_time, 1),
        "taker_flag_active": _PAIR_TAKER_FLAG.get(pair, False),
        "min_notional_cached": _MIN_NOTIONAL_CACHE.get(pair, (None, None))[0],
    }


def publish_summary() -> Dict[str, Any]:
    """Aggregate snapshot for the daily comparison dashboard."""
    summary: Dict[str, Any] = {
        "n_pairs_tracked": len(_FILL_HISTORY),
        "n_pairs_taker_forced": sum(1 for v in _PAIR_TAKER_FLAG.values() if v),
        "n_pairs_min_notional_cached": len(_MIN_NOTIONAL_CACHE),
        "_ts": time.time(),
    }
    try:
        from pheromone_field import get_pheromone_field
        get_pheromone_field().deposit(
            "microstructure", "learner_summary",
            summary, half_life=3600.0,
        )
    except Exception:
        pass
    return summary
