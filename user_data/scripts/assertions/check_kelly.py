"""Kelly-specific pre-trade asserts (Phase 30 A.18)."""
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class AssertResult:
    passed: bool
    reason: str = ""
    severity: str = "error"

    def __bool__(self) -> bool:
        return self.passed


def check_kelly_floor(kelly_fraction: float, min_floor: float = 0.005) -> AssertResult:
    if kelly_fraction is None:
        return AssertResult(False, "Kelly fraction is None", "error")
    if kelly_fraction < min_floor:
        return AssertResult(
            False,
            f"Kelly fraction {kelly_fraction:.4f} below floor {min_floor}",
            "warn",
        )
    return AssertResult(True)


def check_kelly_ceiling(kelly_fraction: float, max_ceiling: float = 0.25) -> AssertResult:
    if kelly_fraction is None:
        return AssertResult(False, "Kelly fraction is None", "error")
    if kelly_fraction > max_ceiling:
        return AssertResult(
            False,
            f"Kelly fraction {kelly_fraction:.4f} exceeds ceiling {max_ceiling}",
            "error",
        )
    return AssertResult(True)


def check_stop_loss_present(stop_loss_pct: float) -> AssertResult:
    if stop_loss_pct is None:
        return AssertResult(False, "Stop loss missing", "error")
    if stop_loss_pct >= 0:
        return AssertResult(False, f"Stop loss must be negative, got {stop_loss_pct}", "error")
    if stop_loss_pct < -0.5:
        return AssertResult(False, f"Stop loss too aggressive {stop_loss_pct} < -0.5", "warn")
    return AssertResult(True)


def check_leverage_bound(leverage: float, max_leverage: float = 5.0) -> AssertResult:
    if leverage is None or leverage <= 0:
        return AssertResult(False, f"Leverage must be positive, got {leverage}", "error")
    if leverage > max_leverage:
        return AssertResult(False, f"Leverage {leverage}x exceeds max {max_leverage}x", "error")
    return AssertResult(True)


def check_all(decision: Dict[str, Any]) -> Dict[str, AssertResult]:
    return {
        "kelly_floor": check_kelly_floor(decision.get("kelly_fraction", 0.0)),
        "kelly_ceiling": check_kelly_ceiling(decision.get("kelly_fraction", 0.0)),
        "stop_loss": check_stop_loss_present(decision.get("stop_loss_pct", -0.05)),
        "leverage": check_leverage_bound(decision.get("leverage", 1.0)),
    }
