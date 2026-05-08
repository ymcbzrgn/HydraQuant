"""HydraQuant pre-trade assertion framework (Phase 30 A.18 + A.26).

Each module exports `check_*` functions returning AssertResult.
Usage: chain in HydraSizer.confirm_trade_entry.
"""
from .check_kelly import (
    AssertResult,
    check_kelly_floor,
    check_kelly_ceiling,
    check_stop_loss_present,
    check_leverage_bound,
    check_all as check_kelly_all,
)
from .check_risk import (
    check_position_count,
    check_daily_loss_limit,
    check_correlation_limit,
)
from .check_position_cap import (
    check_single_position_cap,
    check_aggregate_exposure_cap,
)
from .check_execution import (
    check_slippage_tolerance,
    check_min_notional,
    check_funding_window,
)

__all__ = [
    "AssertResult",
    "check_kelly_floor",
    "check_kelly_ceiling",
    "check_stop_loss_present",
    "check_leverage_bound",
    "check_kelly_all",
    "check_position_count",
    "check_daily_loss_limit",
    "check_correlation_limit",
    "check_single_position_cap",
    "check_aggregate_exposure_cap",
    "check_slippage_tolerance",
    "check_min_notional",
    "check_funding_window",
]
