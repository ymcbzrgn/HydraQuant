"""
event_calendar.py — Macro-event sizing avoidance.

Sprint 2026-05-02 — Pre-FOMC / CPI / NFP defense per Kraken's report
that only 1 of 8 FOMC meetings in 2025 produced a BTC rally. The
typical pattern is "buy the rumor, sell the news" → vol crush + chop
in the 90-minute window around a major macro print. Trading INTO that
window historically loses 100-300 bps per event.

This module exposes:
  • is_event_window(now) → bool
  • current_event_factor() → float in [factor_during_event, 1.0]
  • next_event_seconds() → seconds until next event (for telemetry)

Event dates are explicit (FOMC + CPI + NFP for 2026-2027 from
Federal Reserve and BLS published schedules). When a date passes
without being updated the calendar simply reports "no event" — fail
safe to neutral, never blocks trading on stale data.

Tunable via PARAM_REGISTRY (envelope.event_calendar.*):
  • factor_during_event  — sizing multiplier inside window (default 0.5)
  • pre_event_minutes    — minutes BEFORE event to start dampening
  • post_event_minutes   — minutes AFTER event to keep dampening
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─── 2026-2027 macro event calendar ─────────────────────────────────────────
# All times UTC. Source: federalreserve.gov/monetarypolicy/fomccalendars.htm
# + bls.gov release calendar. Updated annually; if the calendar runs out,
# `is_event_window` returns False and `current_event_factor` returns 1.0.
#
# Event types: FOMC, CPI, NFP (each ~equally impactful for BTC)
_EVENTS_2026_2027: List[Tuple[str, str]] = [
    # 2026 FOMC (8 meetings, 14:00 UTC release)
    ("2026-01-28T19:00:00Z", "FOMC"),
    ("2026-03-18T18:00:00Z", "FOMC"),
    ("2026-04-29T18:00:00Z", "FOMC"),
    ("2026-06-17T18:00:00Z", "FOMC"),
    ("2026-07-29T18:00:00Z", "FOMC"),
    ("2026-09-16T18:00:00Z", "FOMC"),
    ("2026-11-04T19:00:00Z", "FOMC"),
    ("2026-12-16T19:00:00Z", "FOMC"),
    # 2026 CPI (12 monthly releases, 13:30 UTC second Tuesday-ish)
    ("2026-05-13T12:30:00Z", "CPI"),
    ("2026-06-10T12:30:00Z", "CPI"),
    ("2026-07-15T12:30:00Z", "CPI"),
    ("2026-08-12T12:30:00Z", "CPI"),
    ("2026-09-10T12:30:00Z", "CPI"),
    ("2026-10-15T12:30:00Z", "CPI"),
    ("2026-11-13T13:30:00Z", "CPI"),
    ("2026-12-10T13:30:00Z", "CPI"),
    # 2026 NFP (12 monthly releases, 13:30 UTC first Friday)
    ("2026-05-01T12:30:00Z", "NFP"),
    ("2026-06-05T12:30:00Z", "NFP"),
    ("2026-07-03T12:30:00Z", "NFP"),
    ("2026-08-07T12:30:00Z", "NFP"),
    ("2026-09-04T12:30:00Z", "NFP"),
    ("2026-10-02T12:30:00Z", "NFP"),
    ("2026-11-06T13:30:00Z", "NFP"),
    ("2026-12-04T13:30:00Z", "NFP"),
    # 2027 FOMC (placeholder, refresh when Fed publishes)
    ("2027-01-27T19:00:00Z", "FOMC"),
    ("2027-03-17T18:00:00Z", "FOMC"),
    ("2027-04-28T18:00:00Z", "FOMC"),
    ("2027-06-16T18:00:00Z", "FOMC"),
]


def _parse_events() -> List[Tuple[datetime, str]]:
    """Parse the static schedule once on first access; cache result."""
    parsed: List[Tuple[datetime, str]] = []
    for ts_str, kind in _EVENTS_2026_2027:
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            parsed.append((ts, kind))
        except Exception:
            continue
    return parsed


_PARSED_CACHE: Optional[List[Tuple[datetime, str]]] = None


def _get_events() -> List[Tuple[datetime, str]]:
    global _PARSED_CACHE
    if _PARSED_CACHE is None:
        _PARSED_CACHE = _parse_events()
    return _PARSED_CACHE


def _config_window() -> Tuple[float, int, int]:
    """Pull (factor_during_event, pre_minutes, post_minutes) from
    PARAM_REGISTRY. Falls back to (0.5, 60, 30) — 1h before, 30min after,
    half-sized inside.
    """
    try:
        from neural_organism import _p
        factor = float(_p("envelope.event_calendar.factor_during_event", 0.5))
        pre_min = int(_p("envelope.event_calendar.pre_event_minutes", 60))
        post_min = int(_p("envelope.event_calendar.post_event_minutes", 30))
        return factor, pre_min, post_min
    except Exception:
        return 0.5, 60, 30


def is_event_window(now: Optional[datetime] = None) -> Tuple[bool, Optional[str]]:
    """Returns (in_window, event_kind) for the given moment (default now)."""
    if now is None:
        now = datetime.now(tz=timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    _, pre_min, post_min = _config_window()
    window_start_delta = timedelta(minutes=pre_min)
    window_end_delta = timedelta(minutes=post_min)
    for ts, kind in _get_events():
        if ts - window_start_delta <= now <= ts + window_end_delta:
            return True, kind
    return False, None


def current_event_factor(now: Optional[datetime] = None) -> float:
    """Sizing factor right now. 1.0 outside event windows, env-tunable
    (default 0.5) inside any event window.
    """
    in_win, kind = is_event_window(now)
    if not in_win:
        return 1.0
    factor, _, _ = _config_window()
    if kind:
        logger.info(
            f"[EventCalendar] {kind} window active — sizing factor {factor:.2f}"
        )
    return float(factor)


def next_event_seconds(now: Optional[datetime] = None) -> Optional[float]:
    """Seconds until the next scheduled event (for telemetry / scheduling).
    Returns None when the static calendar has run out (signal to refresh)."""
    if now is None:
        now = datetime.now(tz=timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    future = [(ts, kind) for ts, kind in _get_events() if ts > now]
    if not future:
        return None
    future.sort(key=lambda x: x[0])
    return (future[0][0] - now).total_seconds()
