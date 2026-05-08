"""Single-position absolute cap (Phase 30 A.26 — LINK -1016 USDT cinayeti onlemi).

Trade #2187 LINK SHORT 884 USDT stake = ~%5 portfolio. fiyat +%100 -> liquidation.
Bu modul portfolio yuzdesi bazinda single-position ve aggregate exposure cap zorlar.
"""
from .check_kelly import AssertResult

DEFAULT_SINGLE_PCT = 0.025  # 2.5% — LINK ders
DEFAULT_AGGREGATE_PCT = 0.50


def check_single_position_cap(
    stake: float,
    portfolio_value: float,
    max_pct: float = DEFAULT_SINGLE_PCT,
) -> AssertResult:
    if portfolio_value is None or portfolio_value <= 0:
        return AssertResult(False, f"Portfolio value invalid: {portfolio_value}", "error")
    if stake is None or stake <= 0:
        return AssertResult(False, f"Stake invalid: {stake}", "error")
    pct = stake / portfolio_value
    if pct > max_pct:
        return AssertResult(
            False,
            f"Single-position {pct:.2%} > cap {max_pct:.2%} "
            f"(stake={stake:.2f}, portfolio={portfolio_value:.2f})",
            "error",
        )
    return AssertResult(True)


def check_aggregate_exposure_cap(
    open_positions_value: float,
    portfolio_value: float,
    max_pct: float = DEFAULT_AGGREGATE_PCT,
) -> AssertResult:
    if portfolio_value is None or portfolio_value <= 0:
        return AssertResult(False, "Portfolio value invalid", "error")
    if open_positions_value is None or open_positions_value < 0:
        return AssertResult(False, f"Open positions value invalid: {open_positions_value}", "error")
    pct = open_positions_value / portfolio_value
    if pct > max_pct:
        return AssertResult(
            False,
            f"Aggregate exposure {pct:.2%} > cap {max_pct:.2%}",
            "error",
        )
    return AssertResult(True)
