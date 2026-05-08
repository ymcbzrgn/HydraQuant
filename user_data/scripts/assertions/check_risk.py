"""Risk envelope pre-trade asserts (Phase 30 A.18)."""
from typing import List
from .check_kelly import AssertResult


def check_position_count(n_open: int, max_open: int = 30) -> AssertResult:
    if n_open is None:
        return AssertResult(False, "Open position count is None", "error")
    if n_open >= max_open:
        return AssertResult(False, f"Max positions reached: {n_open}/{max_open}", "error")
    return AssertResult(True)


def check_daily_loss_limit(today_pnl_pct: float, max_loss: float = -0.05) -> AssertResult:
    if today_pnl_pct is None:
        return AssertResult(True)
    if today_pnl_pct < max_loss:
        return AssertResult(
            False,
            f"Daily loss limit hit: {today_pnl_pct:.2%} < {max_loss:.2%}",
            "error",
        )
    return AssertResult(True)


def check_correlation_limit(
    pair: str,
    open_pairs: List[str],
    max_correlated: int = 3,
) -> AssertResult:
    btc_correlated = ("BTC", "ETH", "BNB")
    is_btc_corr = any(c in pair for c in btc_correlated)
    if not is_btc_corr:
        return AssertResult(True)
    n_corr = sum(1 for p in open_pairs if any(c in p for c in btc_correlated))
    if n_corr >= max_correlated:
        return AssertResult(
            False,
            f"Correlated positions: {n_corr}/{max_correlated} (BTC family)",
            "warn",
        )
    return AssertResult(True)
