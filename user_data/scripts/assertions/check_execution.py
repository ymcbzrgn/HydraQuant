"""Execution-time asserts (Phase 30 A.18)."""
from .check_kelly import AssertResult


def check_slippage_tolerance(
    expected_price: float,
    actual_price: float,
    max_slip_pct: float = 0.005,
) -> AssertResult:
    if expected_price is None or expected_price <= 0 or actual_price is None or actual_price <= 0:
        return AssertResult(False, "Invalid prices for slippage check", "error")
    slip = abs(actual_price - expected_price) / expected_price
    if slip > max_slip_pct:
        return AssertResult(
            False,
            f"Slippage {slip:.4%} > tolerance {max_slip_pct:.4%}",
            "warn",
        )
    return AssertResult(True)


def check_min_notional(stake: float, min_notional: float = 5.0) -> AssertResult:
    if stake is None:
        return AssertResult(False, "Stake is None", "error")
    if stake < min_notional:
        return AssertResult(False, f"Stake {stake:.2f} below min notional {min_notional}", "error")
    return AssertResult(True)


def check_funding_window(
    seconds_to_funding: float,
    avoid_window_seconds: int = 60,
) -> AssertResult:
    if seconds_to_funding is None:
        return AssertResult(True)
    if 0 <= seconds_to_funding < avoid_window_seconds:
        return AssertResult(
            False,
            f"Funding event in {seconds_to_funding:.0f}s — too close",
            "warn",
        )
    return AssertResult(True)
