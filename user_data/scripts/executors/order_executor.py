"""Phase 30 C.1 — OrderExecutor.

Thin wrapper around freqtrade trade-engine order placement. The real submit
path is freqtrade's IStrategy callback chain (custom_entry_price + amount);
this class formalises the interface so backtests + replay can substitute a
mock executor.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .base import BaseExecutor, OrderRequest, OrderResult

logger = logging.getLogger(__name__)


class OrderExecutor(BaseExecutor):
    name = "order"

    def __init__(self, freqtrade_strategy=None, dry_run: bool = True):
        self.strategy = freqtrade_strategy
        self.dry_run = bool(dry_run)

    def submit(self, req: OrderRequest) -> OrderResult:
        try:
            from assertions import (
                check_kelly_floor, check_kelly_ceiling, check_stop_loss_present,
                check_leverage_bound, check_single_position_cap,
                check_min_notional, check_slippage_tolerance,
            )

            kelly_floor = check_kelly_floor(req.metadata.get("kelly_fraction", 0.01))
            if not kelly_floor.passed and kelly_floor.severity == "error":
                return OrderResult(success=False, error=kelly_floor.reason)
            min_not = check_min_notional(req.stake_amount)
            if not min_not.passed:
                return OrderResult(success=False, error=min_not.reason)
        except Exception as e:
            logger.warning(f"[OrderExecutor] assert chain: {e}")

        try:
            from trade_event_emitter import emit

            emit("trade.opened", {
                "pair": req.pair,
                "side": req.side,
                "stake": req.stake_amount,
                "rate": req.rate,
                "dry_run": self.dry_run,
            })
        except Exception:
            pass

        if self.dry_run or self.strategy is None:
            return OrderResult(success=True, order_id="dry-run", filled_amount=req.stake_amount,
                               avg_price=req.rate, metadata={"dry_run": True})

        return OrderResult(success=True, order_id="strategy_callback",
                           filled_amount=req.stake_amount, avg_price=req.rate)
