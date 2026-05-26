# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# isort: skip_file
import logging
import sqlite3
import sys
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional
from freqtrade.strategy import IStrategy
from freqtrade.persistence import Trade
from freqtrade.strategy import CategoricalParameter, DecimalParameter, IntParameter
import talib.abstract as ta

# Add scripts dir to path for AI module imports
_scripts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'scripts')
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from forgone_pnl_engine import ForgonePnLEngine
from confidence_calibrator import ConfidenceCalibrator

# Phase 25: Neural Organism — adaptive parameters (module-level, ALL methods use this)
try:
    from neural_organism import _p as _np
except ImportError:
    def _np(pid, fb=0.5, regime="_global"):
        return fb

logger = logging.getLogger(__name__)


def _canonical_signal(raw: Optional[str]) -> str:
    """Revize Tur-2 (H12): single source of truth for the BULLISH/BEARISH
    → BULL/BEAR mapping used by every forgone / shadow write. Extracting
    it means the unit test can call the real prod helper instead of
    re-implementing the branch logic."""
    if raw == "BULLISH":
        return "BULL"
    if raw == "BEARISH":
        return "BEAR"
    return raw or "NEUTRAL"

class HydraSizer(IStrategy):
    """
    AI-powered strategy focusing on the "Sizing not Blocking" motto.
    Uses our own LLM Router + RAG pipeline (not FreqAI) for trade decisions.
    Injects real-time SQLite sentiment metrics into the feature set.
    """
    
    INTERFACE_VERSION = 3

    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    can_short = True  # Futures: enable both LONG and SHORT
    startup_candle_count = 400  # EMA 200 + daily RSI warmup + multi-timeframe resampling
    position_adjustment_enable = True  # Phase 22: DCA + partial exit via adjust_trade_position

    # Sprint 2026-05-01 night — ASYMMETRIC R:R 1:3
    # Old table accepted +0.5% wins at 12h, then trailing stop ate +5% gains.
    # Asymmetric R:R principle: with stop-loss at -1.5% (custom_stoploss
    # ATR-based, typical 1-3% range), targets must be ≥3× the stop. So
    # default ROI starts at 4.5% and only relaxes after several hours.
    # Win rate 35% with 1:3 R:R is profitable; we need 60%+ with 1:1.
    # The custom_stoploss layer dynamically tightens further by ATR.
    minimal_roi = {
        "0": 0.045,      # 4.5% immediate target — 3× the typical -1.5% stop
        "120": 0.030,    # 3% after 2h — bot had time to compound
        "360": 0.020,    # 2% after 6h — pull profits in if no breakout yet
        "720": 0.012,    # 1.2% after 12h — accept smaller win rather than reverse
        "1440": 0.005,   # 0.5% after 24h — break-even harvest before stale
    }

    # Stoploss — hard floor for 1x leverage case. custom_stoploss enforces leverage-aware cap.
    # Dynamic formula: max_equity_loss(15%) / leverage → 1x=-15%, 2x=-7.5%, 3x=-5%
    stoploss = -0.15
    use_custom_stoploss = True

    # Trailing stop
    trailing_stop = False

    timeframe = '1h'

    # ── Hyperopt Parameters (Phase 22) ──────────────────────────────────
    # These make ALL key thresholds tunable via: freqtrade hyperopt --spaces entry exit stake protection
    confidence_threshold = DecimalParameter(0.30, 0.80, decimals=2, default=0.50, space='buy', optimize=True, load=True)
    atr_stoploss_mult = DecimalParameter(1.0, 3.5, decimals=1, default=1.5, space='protection', optimize=True, load=True)
    fg_extreme_threshold = IntParameter(15, 30, default=20, space='buy', optimize=True, load=True)
    stale_trade_hours = IntParameter(4, 24, default=8, space='sell', optimize=True, load=True)
    leverage_max = DecimalParameter(1.0, 5.0, decimals=1, default=3.0, space='buy', optimize=True, load=True)

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.db_path = os.path.join(self.config['user_data_dir'], "db", "ai_data.sqlite")
        self.rag_script_path = os.path.join(self.config['user_data_dir'], "scripts", "rag_graph.py")
        # Sprint 2026-05-01: bounded LRU. Apr 27 freqtrade.service OOM-killed at
        # 2GB cgroup with anon-rss 1.3GB — the unbounded dict accumulated stale
        # entries for every pair the scanner ever saw (40+ active × 20-pair
        # rotations × hours of history). LRU bound is derived from the live
        # whitelist size in _ai_cache_maxlen so it scales with the bot's
        # actual concurrent trade universe — never a static number.
        from collections import OrderedDict
        self.ai_signal_cache: "OrderedDict[str, dict]" = OrderedDict()
        # Sprint 2026-05-06 (F1): Triple Perception result cache mirrors
        # ai_signal_cache LRU pattern. Was unbounded dict at :1662 (forensic
        # 2026-05-06: tr-dry RSS 1GB → 4.4GB over 35h, ~100MB/h drift). Now
        # uses dynamic maxlen identical to ai_signal_cache so it scales
        # with whitelist × envelope × hormones (no hardcode).
        self._perception_cache: "OrderedDict[str, dict]" = OrderedDict()
        self.cache_ttl_hours = 12 # 2026-05-18 crisis brake: 6h → 12h to halve RAG /signal call rate while LLM providers (Groq RPM=20) are exhausted
        self._neutral_ttl_hours = 24.0  # 2026-05-18 crisis brake: 8h → 24h. With ~50 pairs, drops RAG call rate from ~6/h to ~2/h while LLM fleet recovers

        # Phase 3.5: Forgone P&L Engine — tracks every missed signal
        self.forgone_engine = ForgonePnLEngine(db_path=self.db_path)
        # Map pair -> forgone_id for resolving on trade exit
        self._forgone_ids: dict = {}

        # Task 4 patch: pending risk-budget consumption. custom_stake_amount
        # parks the proposed (stake, vol, conf) tuple per pair; confirm_trade_entry
        # commits it exactly once when the trade actually fires. Prior design
        # called consume_budget on every candle that custom_stake_amount ran,
        # even for pending/unfilled limit orders — so one stale pending trade
        # could accumulate 10+ duplicate consumption rows in a few hours.
        self._pending_risk_consume: dict[str, tuple] = {}

        # D2 (2026-04-25): (pair, side) → wallclock deadline for fill verification.
        # confirm_trade_entry seeds this; bot_loop_start sweeps and reports
        # fill_rate to PairCircuitBreaker, which flips chronic non-fillers
        # (ICP/WIF/UNI/0G observed in prod) into a soft 30-min dormant.
        # AUDIT-12: keyed by (pair, side) tuple so hedge mode (long+short
        # on the same pair) can be correctly bookkept. Today the bot runs
        # one-way mode so the side dimension is informational; if hedge
        # mode is ever enabled, the sweep stays correct.
        self._pending_fill_checks: dict[tuple[str, str], float] = {}

        # C1 (2026-04-25): RAG /health probe cache so bot_loop_start
        # doesn't hammer port 8891 once per tick. (probe_ts, healthy_bool).
        self._rag_health_cache: tuple[float, bool] = (0.0, True)

        # Risk/Position Management Modules
        from risk_budget import RiskBudgetManager
        from position_sizer import BayesianKelly, PositionSizer
        from telegram_notifier import AITelegramNotifier
        from autonomy_manager import AutonomyManager

        self.risk_budget = RiskBudgetManager(db_path=self.db_path)
        self._bayesian_kelly = BayesianKelly(db_path=self.db_path)
        self.autonomy_manager = AutonomyManager(db_path=self.db_path)

        self._position_sizer = PositionSizer()
        # Share instances with the PositionSizer to ensure state synchronization
        self._position_sizer.bayesian_kelly = self._bayesian_kelly
        self._position_sizer.autonomy = self.autonomy_manager

        self._telegram = AITelegramNotifier()
        self._last_portfolio_sync = None  # Track last sync time

        # Phase 18: Staggered batching — process 10 pairs per batch, 6 min apart
        self._batch_queue = []          # Pairs waiting for fetch in current cycle
        self._batch_index = 0           # Current position in queue
        self._batch_size = 5            # Smaller batches = higher quality per signal
        self._batch_interval_secs = 480  # 8 min between batches — quality over speed
        self._last_batch_time = 0       # Unix timestamp of last batch

        # Phase 20: OpportunityScanner singleton + rate limit (prevent file descriptor leak)
        self._opp_scanner = None        # Lazy init singleton
        self._last_scan_time = 0        # Rate limit: min 5 min between scans

        # Phase 27 Signal Quality: DCA-GATE throttle state.
        # Previously the 6 DCA block points all called logger.warning on every
        # guard attempt — LINK alone produced hundreds of identical lines in
        # 10 minutes. _emit_dca_gate() deduplicates per (pair, reason) for 60s
        # and emits a rollup every 5 minutes so the blocked-volume is visible
        # without the spam.
        self._dca_gate_last_log: dict[tuple[str, str], float] = {}
        self._dca_gate_counter: dict[tuple[str, str], dict] = {}

        logger.info("HydraSizer initialized with MADAM-RAG, Forgone PNL, Risk Budget, Telegram & Staggered Batching.")

    def _should_skip_batch(self, now_ts: float) -> str:
        """C1+C3+C4 backpressure gate. Returns reason string when the
        cycle's batch work should be skipped, or empty string to proceed.

        Skip conditions:
          - RAG /health unreachable / 5xx (60s probe cache so we don't
            hammer port 8891 each tick)
          - memory_pressure > 0.5 (cortisol drop is already shrinking
            sizing, but pulling whole batches saves CPU + LLM tokens)
          - llm_router fleet_exhausted (B1 deposit, ratio < 0.10)
        """
        # ── C1: RAG health probe — DISABLED 2026-05-26 ────────────────
        # The probe was silencing the whole pipeline whenever /health took
        # >5s, but the TP-Promoted RAG-dead fallback (line ~1779) already
        # handles the "RAG returned default" case at the per-pair level —
        # so the batch-wide kill switch was redundant AND blocked the
        # fallback from ever firing (the gate runs before per-pair logic).
        # Re-enable once the lock storm root cause (FAZ 4 hot-table split,
        # FAZ 1 connection-leak completion) makes /health sub-second again.
        #
        # The cache slot is still updated for observability/telemetry but
        # the return is unconditional success.
        try:
            rag_url = (
                self.config.get("ai_config", {}).get("rag_service_url")
                or "http://127.0.0.1:8891"
            )
            health_url = rag_url.rstrip("/") + "/health"
            probe_timeout = float(_np("rag.probe_timeout_s", 5.0))
            unreach_cache_s = float(_np("rag.unreachable_cache_s", 30))
            healthy_cache_s = max(unreach_cache_s * 4.0, 120.0)

            probe_ts, last_healthy = self._rag_health_cache
            cache_window = healthy_cache_s if last_healthy else unreach_cache_s
            if (now_ts - probe_ts) > cache_window:
                healthy = True
                try:
                    import urllib.request as _ur
                    req = _ur.Request(health_url)
                    with _ur.urlopen(req, timeout=probe_timeout) as resp:
                        healthy = (200 <= resp.status < 500)
                except Exception:
                    healthy = False
                self._rag_health_cache = (now_ts, healthy)
                last_healthy = healthy
            if not last_healthy:
                # OBSERVABILITY ONLY — gate disabled, do NOT skip the batch.
                logger.debug(
                    f"[BackpressureGate] /health unreachable (observability) — "
                    f"TP-Promoted fallback will protect per-pair"
                )
        except Exception:
            pass

        # ── C3: memory pressure ───────────────────────────────────────
        try:
            from sensor_bridges import aggregate_memory_stress
            pressure = float(aggregate_memory_stress())
            if pressure > 0.5:
                return f"memory_pressure={pressure:.2f}"
        except Exception:
            pass

        # ── C4: LLM fleet exhausted ───────────────────────────────────
        # AUDIT-7: apply pheromone _decay so the gate releases when the
        # fleet recovers, instead of staying triggered for the full
        # decay-to-zero window (~6 half-lives).
        try:
            from pheromone_field import get_pheromone_field
            fleet = get_pheromone_field().read("fleet_exhausted", source="llm_router")
            if isinstance(fleet, dict):
                raw_ratio = float(fleet.get("ratio", 0.0))
                decay = float(fleet.get("_decay", 1.0))
                effective_ratio = raw_ratio * decay + (1.0 - decay) * 1.0
                if effective_ratio < 0.10:
                    return f"fleet_exhausted raw={raw_ratio:.2%} eff={effective_ratio:.2%}"
        except Exception:
            pass

        return ""

    def bot_loop_start(self, current_time, **kwargs):
        """
        Phase 18: Staggered batch pre-fetch — 10 pairs per batch, 6 min apart.
        100 pairs / 10 per batch = 10 batches × 6min = 60min full cycle.
        Each batch: 10 pairs × 9 LLM calls = 90 calls → 15 calls/min → no rate limit issues.
        """
        if self.dp.runmode.value not in ('dry_run', 'live'):
            return

        import time as _time

        # Audit 2026-05-02 #11 fix: feed the predictive memory guardian
        # from the strategy process too — scheduler-only sampling misses
        # strategy-process leaks (and HydraSizer's ai_signal_cache + chart
        # features are the heaviest strategy-side memory consumers). Cheap
        # once-per-loop sample.
        try:
            if not hasattr(self, "_last_mem_tick_at"):
                self._last_mem_tick_at = 0.0
            if (_time.time() - self._last_mem_tick_at) >= 60.0:
                from memory_sensor import tick as _mem_tick
                _mem_tick()
                self._last_mem_tick_at = _time.time()
        except Exception as _ms_e:
            logger.debug(f"[bot_loop_start] mem_sensor tick skipped: {_ms_e}")

        # D2 (2026-04-25): sweep fill-verification deadlines. Each
        # confirm_trade_entry seeds a 10-min deadline; here we check
        # whether an open trade actually exists for that pair (filled)
        # or not (rejected/timed-out limit) and record the outcome to
        # PairCircuitBreaker. Chronic non-fillers slide into 30-min
        # soft dormant via record_order_attempt's threshold.
        if self._pending_fill_checks:
            try:
                from pair_circuit import get_pair_circuit
                circuit = get_pair_circuit()
                # AUDIT-12 (2026-04-25): build a (pair, side)→amount map so
                # hedge-mode (long + short on the same pair) does not get
                # collapsed. `Trade.is_short` is the side discriminator;
                # under one-way mode each pair has at most one open trade
                # so the dimension is informational.
                open_by_side: dict[tuple[str, str], float] = {}
                try:
                    for t in Trade.get_open_trades():
                        try:
                            amt = float(getattr(t, "amount", 0.0) or 0.0)
                        except Exception:
                            amt = 0.0
                        side = "short" if getattr(t, "is_short", False) else "long"
                        open_by_side[(t.pair, side)] = amt
                except Exception:
                    open_by_side = {}
                # AUDIT-2/Critic: list-copy so we can safely pop while iterating.
                # Audit 2026-05-02 #2 fix: pending value is now (deadline,
                # was_maker_at_entry) tuple so the microstructure learner
                # records the ACTUAL maker/taker decision, not a hardcoded
                # was_maker=True. Backward-compat: handle legacy bare-float
                # values left over from before-restart deque carryover.
                expired = []
                for k, v in list(self._pending_fill_checks.items()):
                    if isinstance(v, tuple):
                        dl_v, was_maker_v = v[0], (v[1] if len(v) > 1 else True)
                    else:
                        dl_v, was_maker_v = float(v), True
                    if _time.time() >= dl_v:
                        expired.append((k, dl_v, was_maker_v))
                for key, deadline, was_maker_at_entry in expired:
                    pair, side = key
                    amt = open_by_side.get((pair, side), 0.0)
                    filled = amt > 0
                    age = max(0.0, _time.time() - (deadline - 600.0))
                    try:
                        circuit.record_order_attempt(pair, filled=filled, age_seconds=age)
                    except Exception:
                        pass
                    # Audit 2026-05-02 #2 fix: was_maker reflects the actual
                    # entry decision. Taker entries always fill so they
                    # don't pollute the maker-fill EMA — and ExchangeMicro
                    # structureLearner._refresh_taker_flag only updates
                    # when was_maker=True (line 193-198), so taker fills
                    # are now correctly excluded from the rolling rate.
                    try:
                        from exchange_microstructure_learner import (
                            record_fill_attempt
                        )
                        record_fill_attempt(
                            pair=pair, side=side,
                            price=0.0,
                            was_maker=bool(was_maker_at_entry),
                            filled=filled,
                            fill_time_seconds=age if filled else 0.0,
                        )
                    except Exception as _msl_e:
                        logger.debug(f"[Microstructure] record skipped: {_msl_e}")
                    self._pending_fill_checks.pop(key, None)
                    # AUDIT-6: log rolling fill_rate so the operator sees
                    # convergence toward the 20% dormant threshold.
                    rolling_rate = None
                    try:
                        rolling_rate = circuit.get_fill_rate(pair)
                    except Exception:
                        pass
                    if not filled:
                        logger.warning(
                            f"[D2:FillCheck] {pair}/{side} non-fill after {age:.0f}s — "
                            f"recorded miss; rolling fill_rate_1h="
                            f"{f'{rolling_rate:.0%}' if rolling_rate is not None else 'N/A'} "
                            f"(circuit flips dormant at <20% with n>=5)"
                        )
                    elif rolling_rate is not None and rolling_rate < 0.5:
                        logger.info(
                            f"[D2:FillCheck] {pair}/{side} filled after {age:.0f}s "
                            f"(rolling fill_rate_1h={rolling_rate:.0%})"
                        )
            except Exception as _fc_e:
                logger.debug(f"[D2:FillCheck] sweep failed: {_fc_e}")

        # C1+C3+C4 (2026-04-25): unified backpressure gate. D2 sweep
        # above ALWAYS runs (it's pure observation, no exchange traffic).
        # The heavy batch work below is skipped when the body is degraded:
        #   - RAG /health 8891 down/5xx (C1)
        #   - memory_pressure > 0.5 (C3)
        #   - llm_router fleet_exhausted ratio < 0.10 (C4)
        # The cycle keeps ticking; batches resume naturally when organs
        # recover. No manual restart, no stuck queues.
        skip_reason = self._should_skip_batch(_time.time())
        if skip_reason:
            logger.info(f"[BackpressureGate] heartbeat only — {skip_reason}")
            return

        # Throttle: only process one batch per interval
        now = _time.time()
        if (now - self._last_batch_time) < self._batch_interval_secs:
            return  # Not time yet

        # If queue empty, rebuild from pairs needing refresh
        if not self._batch_queue:
            try:
                pairs = self.dp.current_whitelist()
            except Exception:
                return
            if not pairs:
                return

            # Phase 20: OpportunityScanner — screen pairs (rate-limited, singleton)
            # Only scan every 5 minutes to prevent file descriptor leak
            scan_age = _time.time() - self._last_scan_time
            if scan_age >= 300:  # 5 minutes
                try:
                    if self._opp_scanner is None:
                        from opportunity_scanner import OpportunityScanner
                        self._opp_scanner = OpportunityScanner()
                    scored = self._opp_scanner.scan_pairs(pairs, dp=self.dp, timeframe=self.timeframe, top_n=20)
                    if scored:
                        screened_pairs = [s["pair"] for s in scored]
                        logger.info(f"[Phase20:Scanner] {len(pairs)} whitelist → {len(screened_pairs)} top opportunities")
                        pairs = screened_pairs
                    self._last_scan_time = _time.time()
                except Exception as e:
                    logger.warning(f"[Phase20:Scanner] Failed, using full whitelist: {e}")

            pairs_to_fetch = []
            for pair in pairs:
                cached = self.ai_signal_cache.get(pair)
                if cached:
                    time_diff = (current_time - cached['timestamp']).total_seconds() / 3600
                    ttl = self._neutral_ttl_hours if cached.get('signal') == 'NEUTRAL' else self.cache_ttl_hours
                    if time_diff < ttl:
                        continue
                pairs_to_fetch.append(pair)

            if not pairs_to_fetch:
                return

            # T9 (2026-04-25): rank pairs_to_fetch by opportunity_scores from
            # DB so the highest-priority pairs go through the LLM funnel
            # FIRST. The opportunity scanner generates 21K rows/week — its
            # ranking was previously fed into pair-level filter only,
            # batch order was alphabetical (which let RAG starve real
            # alpha pairs behind random tickers).
            try:
                from db import get_db_connection
                with get_db_connection() as _conn_t9:
                    rows = _conn_t9.execute("""
                        SELECT pair, MAX(composite_score) AS best_score
                          FROM opportunity_scores
                         WHERE timestamp >= datetime('now', '-30 minutes')
                      GROUP BY pair
                    """).fetchall()
                score_map = {r["pair"]: float(r["best_score"]) for r in rows}
                if score_map:
                    pairs_to_fetch.sort(
                        key=lambda p: score_map.get(p, 0.0), reverse=True
                    )
                    top5 = ", ".join(f"{p}={score_map.get(p, 0):.2f}"
                                     for p in pairs_to_fetch[:5])
                    logger.info(f"[T9:OppRank] sorted by composite_score, top5: {top5}")
                else:
                    logger.warning("[T9:OppRank] no opportunity_scores rows in last 30min — sort skipped")
            except Exception as _t9_e:
                # FIX-A1: surface SQL errors at WARNING level so future schema
                # drift is visible. Audit found `score` typo silently failed.
                logger.warning(f"[T9:OppRank] SQL failed: {_t9_e}")

            # T20 (2026-04-25): inject up to 2 exploration pairs at the
            # FRONT of the queue (1-in-N policy — N defaults to 5 batches).
            # active_learner publishes "exploration_suggestions" — pairs
            # with weak Kelly evidence that should occasionally be probed
            # so the bandit doesn't get stuck on a local optimum.
            # FIX-A5 (2026-04-25): producer key is "suggestions" (list of
            # dicts with 'pair' key), not "pairs"/"candidates". Audit found
            # the consumer was reading wrong keys AND iterating dicts as
            # strings — wire was 100% dead.
            try:
                from pheromone_field import get_pheromone_field as _gpf_t20
                expl = _gpf_t20().read("exploration_suggestions",
                                        source="active_learner")
                if isinstance(expl, dict):
                    raw = expl.get("suggestions") or expl.get("pairs") or expl.get("candidates") or []
                    cand_pairs: list[str] = []
                    if isinstance(raw, list):
                        for item in raw:
                            if isinstance(item, dict) and isinstance(item.get("pair"), str):
                                cand_pairs.append(item["pair"])
                            elif isinstance(item, str):
                                cand_pairs.append(item)
                    injected: list[str] = []
                    for ep in cand_pairs[:2]:
                        if ep in pairs and ep not in pairs_to_fetch[:5]:
                            if ep in pairs_to_fetch:
                                pairs_to_fetch.remove(ep)
                            pairs_to_fetch.insert(0, ep)
                            injected.append(ep)
                    if injected:
                        logger.info(
                            f"[T20:Exploration] injected {len(injected)} "
                            f"active_learner pairs to front: {injected}"
                        )
            except Exception as _t20_e:
                logger.debug(f"[T20:Exploration] skipped: {_t20_e}")

            self._batch_queue = pairs_to_fetch
            self._batch_index = 0
            total_batches = (len(pairs_to_fetch) + self._batch_size - 1) // self._batch_size
            logger.info(f"[bot_loop_start] New cycle: {len(pairs_to_fetch)} pairs in {total_batches} batches ({self._batch_size}/batch, {self._batch_interval_secs}s interval)")

        # Slice current batch
        current_batch = self._batch_queue[self._batch_index : self._batch_index + self._batch_size]
        if not current_batch:
            self._batch_queue = []
            self._batch_index = 0
            return

        batch_num = (self._batch_index // self._batch_size) + 1
        total_batches = (len(self._batch_queue) + self._batch_size - 1) // self._batch_size
        logger.warning(f"[bot_loop_start] Batch {batch_num}/{total_batches}: fetching {len(current_batch)} pairs...")

        from concurrent.futures import (
            ThreadPoolExecutor,
            wait,
            ALL_COMPLETED,
        )

        t0 = _time.time()
        # C2 (2026-04-25): per-batch wallclock cap. The previous as_completed
        # loop's per-future timeout=180s let a single hung pair stretch the
        # batch to 3 minutes, and an entire batch could silently consume the
        # cycle window. With wait(timeout=45) the slow pairs are simply
        # cancelled — they reappear in the next batch as long as their RAG
        # call eventually returns. The 2.5h bot_loop_start gap observed in
        # prod traces back to exactly this pattern (a hung batch blocking
        # subsequent batches via `_last_batch_time` interlock).
        _BATCH_WALLCLOCK_S = 45.0

        def fetch_one(p):
            """Fetch signal for one pair via RAG service (Phase 17: POST with technical data).
            Uses class-level HTTP session to prevent fd leak (Errno 24: Too many open files)."""
            sig = {"signal": "NEUTRAL", "confidence": 0.0, "timestamp": current_time}
            try:
                session = HydraSizer._get_http_session()
                url = self.config.get('ai_config', {}).get(
                    'rag_service_url', 'http://127.0.0.1:8891')

                # Phase 17: Get analyzed dataframe for real indicator data
                technical_data = None
                try:
                    df, _ = self.dp.get_analyzed_dataframe(p, self.timeframe)
                    if df is not None and len(df) > 0:
                        technical_data = self._extract_technical_data(df, p)
                except Exception:
                    pass

                # No tech_data = blind signal. Don't cache it — let populate_entry_trend
                # handle this pair with full dataframe (it always has indicators).
                if not technical_data:
                    logger.debug(f"[bot_loop_start] {p}: no tech_data, skipping (populate_entry_trend will POST)")
                    return p, sig  # NEUTRAL 0.0, won't be cached (see below)

                _t = _time.time()
                # AUDIT-1 (2026-04-25): inner timeout 120→40s. The outer
                # batch wallclock is 45s; with inner=120 the executor.shutdown
                # blocked the bot loop for up to 2 min per hung pair, defeating
                # the wallclock cap that audit found to be a lie.
                resp = session.post(f"{url}/signal/{p}", json={"technical_data": technical_data}, timeout=40)
                lat = (_time.time() - _t) * 1000
                logger.info(f"[RAG Latency] {p}: {lat:.0f}ms (status={resp.status_code})")
                if resp.status_code == 200:
                    parsed = resp.json()
                    sig["signal"] = parsed.get("signal", "NEUTRAL")
                    sig["confidence"] = parsed.get("confidence", 0.0)
                    sig["reasoning"] = parsed.get("reasoning", "")
                    # Sprint 2026-05-01: same pass-through as _get_ai_signal —
                    # populates sub_scores/regime/source so the bot_loop
                    # fast-path sees the same enrichment as the cache-miss
                    # path. Without this, the parallel pre-fetch would
                    # poison the cache with stripped fields.
                    if isinstance(parsed.get("sub_scores"), dict):
                        sig["sub_scores"] = parsed["sub_scores"]
                    if parsed.get("regime"):
                        sig["regime"] = parsed["regime"]
                    if parsed.get("source"):
                        sig["source"] = parsed["source"]
                    if parsed.get("trust_score") is not None:
                        sig["trust_score"] = parsed["trust_score"]
                    # Audit 2026-05-02 #1 fix: feed RAG bias detector
                    # from the parallel-fetch path too. This is the
                    # bulk producer (~30-50 obs/cycle); without it the
                    # bias histogram starves and consensus override
                    # never engages.
                    try:
                        from signal_source_consensus import record_rag_observation
                        record_rag_observation(
                            float(sig.get("confidence", 0.0)),
                            str(sig.get("signal", "NEUTRAL")),
                        )
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"[bot_loop_start] Fetch failed for {p}: {e}")
            # Only cache if we got a real signal (POST with tech_data).
            # NEUTRAL from missing tech_data should NOT pollute cache.
            if sig.get("confidence", 0) > 0 or sig.get("signal") != "NEUTRAL":
                self._cache_set(p, sig)
            return p, sig

        results = {}
        # AUDIT-1 (2026-04-25): explicit executor lifecycle so the wallclock
        # cap is real. `with ThreadPoolExecutor(...)` calls shutdown(wait=True)
        # on __exit__, which BLOCKS until in-flight tasks finish (cancel() can
        # only stop NOT-YET-STARTED tasks). Our max_workers=batch_size=5 means
        # every task starts immediately, so cancel() always returned False
        # and the inner 120s session.post stretched bot_loop_start to 2 min.
        # shutdown(wait=False, cancel_futures=True) is the only way to make
        # the wallclock cap honest. The inner session.post timeout was also
        # tightened from 120s → 40s above so worker threads don't outlive
        # the cycle gap by a wide margin.
        executor = ThreadPoolExecutor(max_workers=5)
        try:
            futures = {executor.submit(fetch_one, p): p for p in current_batch}
            done, pending = wait(
                futures.keys(),
                timeout=_BATCH_WALLCLOCK_S,
                return_when=ALL_COMPLETED,
            )
            for future in done:
                try:
                    pair, signal = future.result(timeout=0.1)
                    results[pair] = signal
                except Exception as e:
                    pair = futures[future]
                    logger.warning(f"[bot_loop_start] Fetch error for {pair}: {e}")
            for future in pending:
                pair = futures[future]
                future.cancel()  # may return False (task already started)
                logger.warning(
                    f"[bot_loop_start] Batch wallclock {_BATCH_WALLCLOCK_S:.0f}s — "
                    f"{pair} abandoned (will retry next cycle; in-flight HTTP "
                    f"call has its own 40s timeout)"
                )
        finally:
            # cancel_futures=True (Python 3.9+) drops queued tasks instantly;
            # any started task continues until its inner timeout but the bot
            # loop is no longer blocked waiting for them.
            executor.shutdown(wait=False, cancel_futures=True)

        elapsed = _time.time() - t0
        dist = {}
        for sig in results.values():
            s = sig.get('signal', 'UNKNOWN')
            dist[s] = dist.get(s, 0) + 1
        logger.warning(
            f"[bot_loop_start] Batch {batch_num}/{total_batches}: {len(results)} signals in {elapsed:.1f}s | {dist}"
        )

        # Advance to next batch
        self._batch_index += self._batch_size
        self._last_batch_time = now

        # If last batch, reset queue for next cycle
        if self._batch_index >= len(self._batch_queue):
            logger.info(f"[bot_loop_start] Cycle complete. All {len(self._batch_queue)} pairs processed.")
            self._batch_queue = []
            self._batch_index = 0

        # Sprint 2026-05-01: telemetry + cross-process state publishing.
        # Three jobs in one place — they all share the same per-cycle
        # data so doing them together avoids re-reading whitelist/cache.
        try:
            wl = list(self.dp.current_whitelist() or [])
        except Exception:
            wl = []
        try:
            from db import execute_with_retry
            execute_with_retry(
                "INSERT INTO system_metrics (timestamp, metric_name, metric_value, metadata_json) "
                "VALUES (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'), ?, ?, NULL)",
                ("whitelist_size", float(len(wl))),
                max_retries=2,
            )
        except Exception:
            pass
        # Most-frequent regime across active cache → pheromone deposit so
        # cross-process readers (RiskEnvelope.protection_lookback_candles,
        # scheduler jobs) can see the strategy-side regime view without a
        # DB hit.
        if self.ai_signal_cache:
            try:
                from collections import Counter
                regimes = [s.get("regime") for s in self.ai_signal_cache.values()
                           if isinstance(s, dict) and s.get("regime")]
                if regimes:
                    most_common = Counter(regimes).most_common(1)[0][0]
                    try:
                        from pheromone_field import get_pheromone_field
                        get_pheromone_field().deposit(
                            "regime_state", "current_regime",
                            {"regime": most_common, "n_pairs": len(regimes)},
                            half_life=600.0,
                        )
                    except Exception:
                        pass
            except Exception:
                pass
        # Per-pair spread/volume sample feeds adaptive_spread_cap +
        # adaptive_volume_floor. We only sample a slice each cycle to
        # avoid hammering the exchange — the scanner already touches
        # tickers, so this is a low-cost piggyback on its work.
        if wl:
            try:
                from pair_circuit import get_pair_circuit
                circuit = get_pair_circuit()
                # Sample first 12 — round-robin across cycles via batch_index
                cycle_offset = (self._batch_index // max(1, self._batch_size)) % max(1, len(wl) - 11)
                sample = wl[cycle_offset:cycle_offset + 12]
                for p in sample:
                    try:
                        ob = self.dp.orderbook(p, 1)
                    except Exception:
                        ob = None
                    spread = None
                    if ob and ob.get("bids") and ob.get("asks"):
                        try:
                            bid = float(ob["bids"][0][0])
                            ask = float(ob["asks"][0][0])
                            if bid > 0 and ask > 0:
                                spread = max(0.0, 1.0 - bid / ask)
                        except Exception:
                            spread = None
                    quote_volume = None
                    try:
                        ticker = self.dp.ticker(p)
                        if ticker and ticker.get("quoteVolume"):
                            quote_volume = float(ticker["quoteVolume"])
                    except Exception:
                        quote_volume = None
                    if spread is not None or quote_volume is not None:
                        circuit.record_pair_observation(
                            p, spread=spread, quote_volume=quote_volume
                        )
            except Exception as e:
                logger.debug(f"[bot_loop_start] pair observation sample failed: {e}")

    def get_entry_signal(self, pair, timeframe, dataframe):
        """
        DIAGNOSTIC OVERRIDE: Wraps parent's get_entry_signal to log exactly
        what Freqtrade sees when checking for entry signals.
        This tells us precisely WHY signals are accepted or rejected.

        2026-05-27 confirm_trade_entry mystery probe: [GES-CALLED] fires
        on EVERY invocation regardless of signal/lock state. If this log
        never appears alongside a Signal:REAL, freqtrade is short-circuiting
        between populate_entry_trend and get_entry_signal.
        """
        try:
            _row = dataframe.iloc[-1] if len(dataframe) > 0 else None
            _el = int(_row.get('enter_long', 0)) if _row is not None else -1
            _es = int(_row.get('enter_short', 0)) if _row is not None else -1
            logger.info(
                f"[GES-CALLED] {pair} tf={timeframe} rows={len(dataframe)} "
                f"enter_long={_el} enter_short={_es}"
            )
        except Exception as _gex:
            logger.info(f"[GES-CALLED] {pair} probe failed: {_gex}")

        signal, tag = super().get_entry_signal(pair, timeframe, dataframe)
        if signal:
            # Only log if pair is NOT locked (avoid 14K+ spam lines per day)
            if not self.is_pair_locked(pair):
                logger.info(f"[ENTRY-SIGNAL] {pair}: {signal} tag={tag}")
        else:
            if len(dataframe) > 0:
                latest = dataframe.iloc[-1]
                el = latest.get('enter_long', 'N/A')
                xl = latest.get('exit_long', 'N/A')
                es = latest.get('enter_short', 'N/A')
                xs = latest.get('exit_short', 'N/A')
                logger.debug(
                    f"[ENTRY-SIGNAL] {pair}: NO SIGNAL! "
                    f"enter_long={el} exit_long={xl} enter_short={es} exit_short={xs}"
                )
            else:
                logger.debug(f"[ENTRY-SIGNAL] {pair}: EMPTY DATAFRAME!")
        return signal, tag

    # Class-level HTTP session — connection pooling prevents Errno 24 (Too many open files)
    _http_session = None

    @classmethod
    def _get_http_session(cls):
        if cls._http_session is None:
            import requests
            cls._http_session = requests.Session()
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=10, pool_maxsize=10, max_retries=1
            )
            cls._http_session.mount('http://', adapter)
            cls._http_session.mount('https://', adapter)
        return cls._http_session

    def _get_sqlite_connection(self):
        """Pool-managed connection. conn.close() returns to pool (no leak).

        2026-05-26: previously called sqlite3.connect(self.db_path) directly,
        bypassing db.py's pool. With ~750 trade ticks/day each opening + leaking
        a connection on exception paths, tr-dry accumulated 700+ FD handles
        which pinned ~3.7GB of SQLite cache and triggered swap thrash + 5h DB
        freeze. Routing through the pool caps live connections at max_size=4.
        """
        try:
            from db import get_db_connection
            return get_db_connection()
        except Exception as e:
            logger.error(f"Error connecting to AI SQLite DB: {e}")
            return None

    def _extract_technical_data(self, dataframe: pd.DataFrame, pair: str) -> dict:
        """
        Phase 17 Enhanced: Extract comprehensive multi-resolution technical data for RAG service.

        Telescopic approach:
        - Micro (24h): Full OHLCV candles for immediate price action
        - Short (7d): Daily summaries for medium-term trend
        - Long (30d): Key levels only for strategic context
        Plus: S/R levels, Fibonacci, pivot points, multi-timeframe, patterns, volume profile.
        """
        if dataframe is None or len(dataframe) < 2:
            return {}

        last = dataframe.iloc[-1]
        prev = dataframe.iloc[-2]
        price = float(last['close'])
        prev_price = float(prev['close'])

        def _safe(val):
            if pd.isna(val):
                return None
            return round(float(val), 4)

        def _pct_change(current, past):
            return round(((current - past) / past * 100), 2) if past > 0 else 0.0

        # === PRICE CHANGES (multi-horizon) ===
        change_1h = _pct_change(price, prev_price)
        change_4h = _pct_change(price, float(dataframe.iloc[-5]['close'])) if len(dataframe) >= 5 else 0.0
        change_24h = _pct_change(price, float(dataframe.iloc[-25]['close'])) if len(dataframe) >= 25 else 0.0
        change_7d = _pct_change(price, float(dataframe.iloc[-169]['close'])) if len(dataframe) >= 169 else 0.0

        # === BASIC INDICATORS (1h timeframe) ===
        td = {
            "current_price": round(price, 8),  # 8 decimals for sub-cent coins (1000PEPE etc.)
            "price_change_1h_pct": change_1h,
            "price_change_4h_pct": change_4h,
            "price_change_24h_pct": change_24h,
            "price_change_7d_pct": change_7d,
            "rsi_14": _safe(last.get('rsi')),
            "macd": _safe(last.get('macd')),
            "macd_signal": _safe(last.get('macdsignal')),
            "macd_histogram": _safe(last.get('macdhist')),
            "atr_14": _safe(last.get('atr')),
            "adx_14": _safe(last.get('adx')),
            "ema_9": _safe(last.get('ema_9')),
            "ema_20": _safe(last.get('ema_20')),
            "ema_50": _safe(last.get('ema_50')),
            "ema_200": _safe(last.get('ema_200')),
            "sma_50": _safe(last.get('sma_50')),
            "sma_200": _safe(last.get('sma_200')),
            "bb_upper": _safe(last.get('bb_upper')),
            "bb_mid": _safe(last.get('bb_mid')),
            "bb_lower": _safe(last.get('bb_lower')),
        }

        # === RECENT CLOSES (for OHLCV pattern matching - Evidence Engine needs 21+) ===
        n_closes = min(50, len(dataframe))
        if n_closes >= 21:
            td["recent_closes"] = [round(float(c), 6) for c in dataframe['close'].iloc[-n_closes:].tolist()]

        # === KEY LEVELS (Support/Resistance/Fibonacci/Pivots) ===
        levels = {}

        # Time-horizon highs and lows
        for n, label in [(24, "24h"), (168, "7d"), (720, "30d")]:
            if len(dataframe) >= n:
                chunk = dataframe.tail(n)
                levels[f"high_{label}"] = round(float(chunk['high'].max()), 2)
                levels[f"low_{label}"] = round(float(chunk['low'].min()), 2)
            elif label == "30d" and len(dataframe) >= 168:
                levels["high_30d"] = round(float(dataframe['high'].max()), 2)
                levels["low_30d"] = round(float(dataframe['low'].min()), 2)

        # Swing-based Support/Resistance
        supports, resistances = self._find_swing_levels(dataframe, price)
        levels["support"] = supports
        levels["resistance"] = resistances

        # Fibonacci retracement (from recent swing)
        lookback = min(100, len(dataframe))
        recent = dataframe.tail(lookback)
        swing_high = float(recent['high'].max())
        swing_low = float(recent['low'].min())
        if swing_high > swing_low:
            diff = swing_high - swing_low
            levels["fibonacci"] = {
                "swing_high": round(swing_high, 2),
                "swing_low": round(swing_low, 2),
                "fib_236": round(swing_low + 0.236 * diff, 2),
                "fib_382": round(swing_low + 0.382 * diff, 2),
                "fib_500": round(swing_low + 0.500 * diff, 2),
                "fib_618": round(swing_low + 0.618 * diff, 2),
                "fib_786": round(swing_low + 0.786 * diff, 2),
            }

        # Classic Pivot Points (from yesterday's 24 candles)
        if len(dataframe) >= 25:
            yesterday = dataframe.iloc[-25:-1]
            yh, yl, yc = float(yesterday['high'].max()), float(yesterday['low'].min()), float(yesterday.iloc[-1]['close'])
            pp = (yh + yl + yc) / 3
            levels["pivot"] = {
                "pp": round(pp, 2),
                "r1": round(2 * pp - yl, 2), "r2": round(pp + (yh - yl), 2),
                "s1": round(2 * pp - yh, 2), "s2": round(pp - (yh - yl), 2),
            }

        td["levels"] = levels

        # === VOLUME ANALYSIS ===
        volume = {}
        if 'volume' in dataframe.columns and len(dataframe) >= 20:
            curr_vol = float(last.get('volume', 0))
            avg_vol = float(dataframe.tail(20)['volume'].mean())
            volume["current"] = round(curr_vol, 0)
            volume["avg_20"] = round(avg_vol, 0)
            volume["ratio"] = round(curr_vol / avg_vol, 2) if avg_vol > 0 else 0
            if len(dataframe) >= 10:
                recent_5 = float(dataframe.tail(5)['volume'].mean())
                prev_5 = float(dataframe.iloc[-10:-5]['volume'].mean())
                if prev_5 > 0:
                    vol_chg = ((recent_5 - prev_5) / prev_5 * 100)
                    volume["trend"] = "rising" if vol_chg > 10 else "declining" if vol_chg < -10 else "stable"
                    volume["trend_pct"] = round(vol_chg, 1)
        td["volume"] = volume

        # === CANDLESTICK PATTERNS (last candle) ===
        patterns = []
        pattern_cols = {
            'cdl_doji': 'Doji', 'cdl_engulfing': 'Engulfing',
            'cdl_hammer': 'Hammer', 'cdl_shooting_star': 'Shooting Star',
            'cdl_morning_star': 'Morning Star', 'cdl_evening_star': 'Evening Star',
            'cdl_three_white': 'Three White Soldiers', 'cdl_three_black': 'Three Black Crows',
            'cdl_harami': 'Harami', 'cdl_inverted_hammer': 'Inverted Hammer',
        }
        for col, name in pattern_cols.items():
            val = last.get(col, 0)
            if pd.notna(val) and val != 0:
                direction = "bullish" if val > 0 else "bearish"
                patterns.append(f"{name} ({direction})")
        td["patterns"] = patterns

        # === MULTI-TIMEFRAME INDICATORS (derived from 1h data) ===
        td["htf"] = self._compute_higher_timeframe(dataframe)

        # === LAST 24 CANDLES (detailed OHLCV) ===
        n_candles = min(24, len(dataframe))
        candles = []
        for i in range(n_candles, 0, -1):
            row = dataframe.iloc[-i]
            candles.append({
                "time": str(row['date']),
                "open": round(float(row['open']), 2),
                "high": round(float(row['high']), 2),
                "low": round(float(row['low']), 2),
                "close": round(float(row['close']), 2),
                "volume": round(float(row.get('volume', 0)), 0),
            })
        td["last_candles"] = candles

        # === DAILY SUMMARIES (7 days, aggregated from 1h) ===
        td["daily_summaries"] = self._compute_daily_summaries(dataframe, n_days=7)

        return td

    @staticmethod
    def _find_swing_levels(dataframe: pd.DataFrame, current_price: float,
                           window: int = 5, n_levels: int = 3):
        """Find support/resistance from swing highs and lows in recent price action."""
        lookback = min(100, len(dataframe))
        df = dataframe.tail(lookback)
        highs = df['high'].values
        lows = df['low'].values
        supports = []
        resistances = []

        for i in range(window, len(df) - window):
            local_lows = lows[max(0, i - window):i + window + 1]
            local_highs = highs[max(0, i - window):i + window + 1]
            if lows[i] == min(local_lows):
                supports.append(float(lows[i]))
            if highs[i] == max(local_highs):
                resistances.append(float(highs[i]))

        # Deduplicate nearby levels (within 1%)
        def _dedup(levels, threshold=0.01):
            if not levels:
                return []
            levels.sort()
            deduped = [levels[0]]
            for lv in levels[1:]:
                if abs(lv - deduped[-1]) / deduped[-1] > threshold:
                    deduped.append(lv)
            return deduped

        supports = _dedup(supports)
        resistances = _dedup(resistances)

        # Filter: supports below current price, resistances above
        supports = sorted([s for s in supports if s < current_price], reverse=True)[:n_levels]
        resistances = sorted([r for r in resistances if r > current_price])[:n_levels]

        return [round(s, 2) for s in supports], [round(r, 2) for r in resistances]

    def _compute_higher_timeframe(self, dataframe: pd.DataFrame) -> dict:
        """Derive 4H and Daily indicators from 1h candles via resampling."""
        htf = {}
        try:
            df_temp = dataframe.copy()
            df_temp['date'] = pd.to_datetime(df_temp['date'])
            df_temp = df_temp.set_index('date')

            # 4H timeframe
            if len(df_temp) >= 56:  # 14 periods × 4h = 56 candles
                df_4h = df_temp.resample('4h').agg({
                    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
                }).dropna()
                if len(df_4h) >= 14:
                    rsi_4h = ta.RSI(df_4h, timeperiod=14)
                    ema_20_4h = ta.EMA(df_4h, timeperiod=20)
                    if len(rsi_4h) > 0 and pd.notna(rsi_4h.iloc[-1]):
                        htf["rsi_4h"] = round(float(rsi_4h.iloc[-1]), 1)
                    if len(ema_20_4h) > 0 and pd.notna(ema_20_4h.iloc[-1]):
                        htf["ema_20_4h"] = round(float(ema_20_4h.iloc[-1]), 2)
                    # 4H trend: price vs EMA20 on 4h
                    if htf.get("ema_20_4h"):
                        p = float(df_4h.iloc[-1]['close'])
                        htf["trend_4h"] = "bullish" if p > htf["ema_20_4h"] else "bearish"

            # Daily timeframe
            if len(df_temp) >= 336:  # 14 days × 24h
                df_daily = df_temp.resample('1D').agg({
                    'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
                }).dropna()
                if len(df_daily) >= 14:
                    rsi_d = ta.RSI(df_daily, timeperiod=14)
                    if len(rsi_d) > 0 and pd.notna(rsi_d.iloc[-1]):
                        htf["rsi_daily"] = round(float(rsi_d.iloc[-1]), 1)
                    if len(df_daily) >= 50:
                        ema_50_d = ta.EMA(df_daily, timeperiod=50)
                        if len(ema_50_d) > 0 and pd.notna(ema_50_d.iloc[-1]):
                            htf["ema_50_daily"] = round(float(ema_50_d.iloc[-1]), 2)
                    p_daily = float(df_daily.iloc[-1]['close'])
                    # Daily trend from EMA alignment
                    if htf.get("ema_50_daily"):
                        htf["trend_daily"] = "bullish" if p_daily > htf["ema_50_daily"] else "bearish"
        except Exception as e:
            logger.debug(f"[Phase17] Higher timeframe computation failed: {e}")

        return htf

    @staticmethod
    def _compute_daily_summaries(dataframe: pd.DataFrame, n_days: int = 7) -> list:
        """Aggregate 1h candles into daily OHLCV summaries."""
        summaries = []
        try:
            df_temp = dataframe.copy()
            df_temp['date'] = pd.to_datetime(df_temp['date'])
            df_temp = df_temp.set_index('date')
            daily = df_temp.resample('1D').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
            }).dropna()
            for _, row in daily.tail(n_days).iterrows():
                summaries.append({
                    "date": str(row.name.date()),
                    "open": round(float(row['open']), 2),
                    "high": round(float(row['high']), 2),
                    "low": round(float(row['low']), 2),
                    "close": round(float(row['close']), 2),
                    "volume": round(float(row['volume']), 0),
                })
        except Exception as e:
            logger.debug(f"[Phase17] Daily summary computation failed: {e}")
        return summaries

    @property
    def _ai_cache_maxlen(self) -> int:
        """Bound on `ai_signal_cache` — derived dynamically.

        Uses (whitelist size + envelope's max_open_positions) × hormonal
        ease factor so a relaxed organism keeps a wider history while a
        tense one trims aggressively. Never returns a hardcoded number;
        always recomputed from current state. Final clamp [16, 256] is a
        safety bound, not a tuning knob.
        """
        try:
            wl = self.dp.current_whitelist() if getattr(self, "dp", None) else []
            wl_size = max(8, len(wl))
        except Exception:
            wl_size = 16
        try:
            from risk_envelope import get_risk_envelope
            max_open = int(get_risk_envelope().get_max_open_positions())
        except Exception:
            max_open = 8
        try:
            from neural_organism import get_organism
            sero = float(get_organism().hormones.serotonin)  # 0=anxious, 1=ease
        except Exception:
            sero = 1.0
        # Calm/eased → wider cache; anxious → tighter
        ease_mult = max(0.7, min(1.6, 0.8 + 0.8 * sero))
        size = int((wl_size + max_open * 2) * ease_mult)
        return max(16, min(256, size))

    def _cache_set(self, pair: str, signal: dict) -> None:
        """LRU-bounded write to `ai_signal_cache`. Evicts oldest entries
        when over the dynamic maxlen so an unbounded scanner can't drift
        the freqtrade heap toward OOM (Apr 27 03:08 root cause).
        """
        try:
            self.ai_signal_cache[pair] = signal
            self.ai_signal_cache.move_to_end(pair)
        except Exception:
            self.ai_signal_cache[pair] = signal
            return
        try:
            limit = self._ai_cache_maxlen
            while len(self.ai_signal_cache) > limit:
                self.ai_signal_cache.popitem(last=False)
        except Exception:
            pass

    def _perception_cache_set(self, pair: str, perception: dict) -> None:
        """Sprint 2026-05-06 (F1): LRU-bounded write to `_perception_cache`.

        Triple Perception results retain large feature tensors (chart features,
        Kronos hidden states, OOD distances). Forensic 2026-05-06 found the
        old unbounded dict drifted tr-dry RSS by ~100MB/h over 35h. Mirrors
        ai_signal_cache LRU policy with the SAME dynamic maxlen so the two
        caches respect a shared, organism-aware budget — no hardcoded knob.
        """
        try:
            self._perception_cache[pair] = perception
            self._perception_cache.move_to_end(pair)
        except Exception:
            self._perception_cache[pair] = perception
            return
        try:
            limit = self._ai_cache_maxlen
            while len(self._perception_cache) > limit:
                self._perception_cache.popitem(last=False)
        except Exception:
            pass

    def _get_ai_signal(self, pair: str, current_time: datetime, dataframe: pd.DataFrame = None) -> dict:
        """
        The Bridge (Phase 5.1): Asks the RAG Signal Service for a decision.
        HTTP-first with subprocess fallback. Models stay loaded in the service.
        """
        # 1. Check Memory Cache (NEUTRAL uses shorter TTL)
        cached = self.ai_signal_cache.get(pair)
        if cached:
            time_diff = (current_time - cached['timestamp']).total_seconds() / 3600
            ttl = self._neutral_ttl_hours if cached.get('signal') == 'NEUTRAL' else self.cache_ttl_hours
            if time_diff < ttl:
                return cached

        # 2. Cache Miss → HTTP call to RAG Signal Service
        logger.info(f"AI Signal Cache Miss for {pair}. Asking RAG Signal Service...")
        signal_data = {"signal": "NEUTRAL", "confidence": 0.0, "timestamp": current_time}

        # Phase 17: Extract technical data from dataframe for RAG service
        technical_data = None
        if dataframe is not None and len(dataframe) > 0:
            try:
                technical_data = self._extract_technical_data(dataframe, pair)
            except Exception as e:
                logger.debug(f"[Phase17] Failed to extract technical data for {pair}: {e}")

        try:
            import requests
            import time as _time
            rag_service_url = self.config.get('ai_config', {}).get(
                'rag_service_url', 'http://127.0.0.1:8891')

            _t0 = _time.time()
            # 2026-05-26 revised: timeout = 60s. Initial revision dropped 120 → 30
            # which under-served the happy path (RAG warm cycle is 45-60s when
            # ColBERT + MADAM run hot); 60s lets real responses through while
            # still capping the wall-clock budget per pair so a 40-coin cycle
            # caps at 40min in the worst case. The deeper safety net is the
            # TriplePerception RAG-dead fallback below (line ~1779).
            # Phase 17: POST with technical data when available, GET fallback
            if technical_data:
                response = requests.post(
                    f"{rag_service_url}/signal/{pair}",
                    json={"technical_data": technical_data},
                    timeout=60
                )
            else:
                response = requests.get(
                    f"{rag_service_url}/signal/{pair}",
                    timeout=60
                )
            _latency = (_time.time() - _t0) * 1000
            logger.info(f"[RAG Latency] {pair}: {_latency:.0f}ms (status={response.status_code}, POST={'Y' if technical_data else 'N'})")
            if response.status_code == 200:
                parsed = response.json()
                signal_data["signal"] = parsed.get("signal", "NEUTRAL")
                signal_data["confidence"] = parsed.get("confidence", 0.0)
                signal_data["reasoning"] = parsed.get("reasoning", "")
                # Sprint 2026-05-01: pass-through ALL fields so downstream
                # consumers (forgone_pnl_engine sub-scores, calibrator,
                # regime-aware threshold, source attribution) get real
                # values instead of the constant 0.5 placeholder fallback.
                # Previous shape dropped sub_scores / regime / source on
                # the floor at the HTTP boundary even though _synthesize()
                # populated them upstream.
                if isinstance(parsed.get("sub_scores"), dict):
                    signal_data["sub_scores"] = parsed["sub_scores"]
                if parsed.get("regime"):
                    signal_data["regime"] = parsed["regime"]
                if parsed.get("source"):
                    signal_data["source"] = parsed["source"]
                if parsed.get("trust_score") is not None:
                    signal_data["trust_score"] = parsed["trust_score"]
                logger.info(f"RAG Signal: {signal_data['signal']} ({signal_data['confidence']}) for {pair}")
                # Sprint 2026-05-02: feed RAG observation to consensus
                # bias detector. Tracks rolling histogram of RAG raw
                # confidence values; when one bucket dominates >50%
                # of recent samples, flags RAG as biased and the
                # consensus override engages.
                try:
                    from signal_source_consensus import record_rag_observation
                    record_rag_observation(
                        float(signal_data["confidence"]),
                        str(signal_data["signal"]),
                    )
                except Exception:
                    pass
            else:
                logger.warning(f"RAG service returned {response.status_code} for {pair}")
        except Exception as e:
            is_connection_error = False
            try:
                import requests as _req
                is_connection_error = isinstance(e, _req.exceptions.ConnectionError)
            except Exception:
                pass

            # 2026-05-26: subprocess fallback DISABLED. The fallback was
            # spawning a fresh python process that imports rag_graph.py end
            # to end (LanceStore, MemoRAG, ColBERT, semantic_cache) — every
            # invocation cost 3-5 minutes of cold-start init and burned a
            # second copy of the same model weights against the same DB,
            # which is what drove the cycle-init storm we saw in the 18:52
            # restart. The TP-Promoted RAG-dead fallback at line ~1779 now
            # owns the RAG-unreachable code path: it trusts TripleAgent's
            # directional reading instead of spawning a clone. signal_data
            # stays at the line-1146 default (NEUTRAL, 0.0, no source),
            # which rag_dead detection picks up downstream.
            if is_connection_error:
                logger.warning(
                    f"[RAG] ConnectionError for {pair} — relying on TP-Promoted "
                    f"RAG-dead fallback (subprocess path disabled 2026-05-26)"
                )
            else:
                logger.error(f"Error calling RAG Signal Service for {pair}: {e}")

        # 3. Save to Cache (ALL signals including NEUTRAL — TTL handles expiry)
        # NEUTRAL uses shorter TTL (0.9h) so it's retried on next candle
        # LRU-bounded write — see _cache_set above (Sprint 2026-05-01).
        self._cache_set(pair, signal_data)
        return signal_data

    def _get_ai_signal_subprocess(self, pair: str, signal_data: dict):
        """Legacy subprocess fallback — only used if HTTP service is down."""
        try:
            import subprocess
            import json
            result = subprocess.run(
                [sys.executable, self.rag_script_path, f"--pair={pair}"],
                capture_output=True, text=True, check=True, timeout=35
            )
            output = result.stdout
            if "--- JSON OUTPUT ---" in output:
                json_str = output.split("--- JSON OUTPUT ---")[1].strip()
                parsed = json.loads(json_str)
                signal_data["signal"] = parsed.get("signal", "NEUTRAL")
                signal_data["confidence"] = parsed.get("confidence", 0.0)
                logger.info(f"[Subprocess Fallback] {signal_data['signal']} ({signal_data['confidence']}) for {pair}")
        except subprocess.TimeoutExpired:
            logger.warning(f"Subprocess timed out for {pair} (120s)")
        except Exception as e:
            logger.error(f"Subprocess fallback failed for {pair}: {e}")

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """Compute technical indicators and sentiment features for sizing/stoploss."""
        # Technical indicators
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['ema_9'] = ta.EMA(dataframe, timeperiod=9)
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema_50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        dataframe['sma_50'] = ta.SMA(dataframe, timeperiod=50)
        dataframe['sma_200'] = ta.SMA(dataframe, timeperiod=200)

        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe['bb_lower'] = bollinger['lowerband']
        dataframe['bb_mid'] = bollinger['middleband']
        dataframe['bb_upper'] = bollinger['upperband']

        # Phase 17: Candlestick pattern detection (for AI context)
        dataframe['cdl_doji'] = ta.CDLDOJI(dataframe)
        dataframe['cdl_engulfing'] = ta.CDLENGULFING(dataframe)
        dataframe['cdl_hammer'] = ta.CDLHAMMER(dataframe)
        dataframe['cdl_shooting_star'] = ta.CDLSHOOTINGSTAR(dataframe)
        dataframe['cdl_morning_star'] = ta.CDLMORNINGSTAR(dataframe)
        dataframe['cdl_evening_star'] = ta.CDLEVENINGSTAR(dataframe)
        dataframe['cdl_three_white'] = ta.CDL3WHITESOLDIERS(dataframe)
        dataframe['cdl_three_black'] = ta.CDL3BLACKCROWS(dataframe)
        dataframe['cdl_harami'] = ta.CDLHARAMI(dataframe)
        dataframe['cdl_inverted_hammer'] = ta.CDLINVERTEDHAMMER(dataframe)

        # Chandelier Exit: highest high / lowest low over 14 bars (for trailing stoploss)
        dataframe['highest_high_14'] = dataframe['high'].rolling(14).max()
        dataframe['lowest_low_14'] = dataframe['low'].rolling(14).min()
        dataframe['lowest_low_20'] = dataframe['low'].rolling(20).min()

        # Sprint 2026-05-02 — adopted indicators from jesse repo:
        # 1. Squeeze Momentum (LazyBear) — BB inside Keltner = compression
        #    that often precedes breakouts.
        # 2. Stiffness (daviddtech) — count of bars where price stayed
        #    above a smoothed SMA band; high stiffness = strong trend.
        # 3. Reflex (Ehlers) — zero-lag oscillator for cycle detection.
        # Audit Finding #7 fix (2026-05-02): all magic numbers sourced
        # from PARAM_REGISTRY so the neural organism can adapt periods.
        # Audit Finding #13 fix (2026-05-02): defensive .fillna(0) +
        # inf-replacement so first-N-row NaNs don't propagate to consumers.
        try:
            from neural_organism import _p as _np_ind
            sq_bb_p = int(_np_ind("strategy.indicators.squeeze.bb_period", 20))
            sq_bb_std = float(_np_ind("strategy.indicators.squeeze.bb_std", 2.0))
            sq_kc_p = int(_np_ind("strategy.indicators.squeeze.kc_period", 20))
            sq_kc_atr = float(_np_ind("strategy.indicators.squeeze.kc_atr_mult", 1.5))
            st_sma_p = int(_np_ind("strategy.indicators.stiffness.sma_period", 100))
            st_std_m = float(_np_ind("strategy.indicators.stiffness.std_mult", 0.2))
            st_win = int(_np_ind("strategy.indicators.stiffness.window", 60))
            rf_period = int(_np_ind("strategy.indicators.reflex.period", 20))
            rf_std_w = int(_np_ind("strategy.indicators.reflex.std_window", 50))
        except Exception:
            sq_bb_p, sq_bb_std, sq_kc_p, sq_kc_atr = 20, 2.0, 20, 1.5
            st_sma_p, st_std_m, st_win = 100, 0.2, 60
            rf_period, rf_std_w = 20, 50

        try:
            # Squeeze Momentum: BB inside Keltner → compression
            bb_mid = dataframe['close'].rolling(sq_bb_p).mean()
            bb_std_ser = dataframe['close'].rolling(sq_bb_p).std()
            bb_upper_sq = bb_mid + sq_bb_std * bb_std_ser
            bb_lower_sq = bb_mid - sq_bb_std * bb_std_ser
            kc_atr = ta.ATR(dataframe, timeperiod=sq_kc_p)
            kc_mid = dataframe['close'].rolling(sq_kc_p).mean()
            kc_upper = kc_mid + sq_kc_atr * kc_atr
            kc_lower = kc_mid - sq_kc_atr * kc_atr
            dataframe['%-squeeze_on'] = (
                (bb_lower_sq > kc_lower) & (bb_upper_sq < kc_upper)
            ).astype(float).fillna(0.0)
            # Squeeze momentum oscillator: price - midpoint
            highest_sq = dataframe['high'].rolling(sq_bb_p).max()
            lowest_sq = dataframe['low'].rolling(sq_bb_p).min()
            sq_mid = (highest_sq + lowest_sq) / 2
            sq_osc = dataframe['close'] - sq_mid
            dataframe['%-squeeze_momentum'] = (
                sq_osc.fillna(0.0)
                      .replace([np.inf, -np.inf], 0.0)
            )
        except Exception as _sq_e:
            logger.debug(f"[Indicators:Squeeze] failed: {_sq_e}")
            dataframe['%-squeeze_on'] = 0.0
            dataframe['%-squeeze_momentum'] = 0.0

        try:
            # Stiffness: count bars in last N where close > SMA - thr*stddev
            sma_n = dataframe['close'].rolling(st_sma_p).mean()
            std_n = dataframe['close'].rolling(st_sma_p).std()
            stiff_band = sma_n - st_std_m * std_n
            above_band = (dataframe['close'] > stiff_band).astype(float)
            stiffness_raw = above_band.rolling(st_win).sum()
            dataframe['%-stiffness'] = (
                stiffness_raw.fillna(0.0)
                             .replace([np.inf, -np.inf], 0.0)
            )
        except Exception as _st_e:
            logger.debug(f"[Indicators:Stiffness] failed: {_st_e}")
            dataframe['%-stiffness'] = 0.0

        try:
            # Reflex (Ehlers super-smoother) — proper 2-pole recursive
            # implementation per John F. Ehlers (Cycle Analytics).
            # Audit Finding #8 fix (2026-05-02): prior implementation
            # computed alpha1/beta1 but discarded them, falling back to
            # a generic EMA mislabeled as super-smoother. Now uses the
            # actual recursive form: y[t] = c1*x[t] + c2*y[t-1] + c3*y[t-2]
            close = dataframe['close'].astype(float).values
            import math as _m
            # Ehlers super-smoother coefficients for given period
            a1 = _m.exp(-_m.sqrt(2.0) * _m.pi / max(2, rf_period))
            b1 = 2.0 * a1 * _m.cos(_m.sqrt(2.0) * _m.pi / max(2, rf_period))
            c2 = b1
            c3 = -a1 * a1
            c1 = 1.0 - c2 - c3
            n = len(close)
            ss = np.zeros(n, dtype=float)
            if n >= 2:
                ss[0] = close[0]
                ss[1] = close[1]
                for i in range(2, n):
                    ss[i] = c1 * (close[i] + close[i - 1]) / 2.0 \
                            + c2 * ss[i - 1] + c3 * ss[i - 2]
            super_smooth = pd.Series(ss, index=dataframe.index)
            slope = (super_smooth - super_smooth.shift(rf_period)) / float(rf_period)
            reflex = (super_smooth - super_smooth.shift(rf_period)) - \
                     slope.rolling(rf_period).mean()
            std_reflex = reflex.rolling(rf_std_w).std() + 1e-9
            reflex_norm = reflex / std_reflex
            dataframe['%-reflex'] = (
                reflex_norm.fillna(0.0)
                           .replace([np.inf, -np.inf], 0.0)
            )
        except Exception as _rf_e:
            logger.debug(f"[Indicators:Reflex] failed: {_rf_e}")
            dataframe['%-reflex'] = 0.0

        # Hurst 3-vote — publish PER-PAIR to pheromone (audit BLOCKER fix
        # 2026-05-02: previously the deposit had no pair suffix, so
        # multi-pair concurrent populate_indicators overwrote each other).
        try:
            from hurst_estimator import publish_hurst_to_pheromone
            closes_for_hurst = dataframe['close'].dropna().values[-200:]
            if len(closes_for_hurst) >= 50:
                publish_hurst_to_pheromone(closes_for_hurst,
                                           pair=metadata.get('pair'))
        except Exception as _h_e:
            logger.debug(f"[Indicators:Hurst3vote] failed: {_h_e}")

        # Sprint 2026-05-02 (audit Finding #2 fix) — recent closes
        # deposit so the OLMAR scheduler tick has price feed input.
        # The tick reads `recent_closes::<pair>` per pair when computing
        # cross-pair mean-reversion weights.
        try:
            from pheromone_field import get_pheromone_field
            recent_closes = dataframe['close'].dropna().values[-30:]
            if len(recent_closes) >= 5:
                get_pheromone_field().deposit(
                    "market_data", f"recent_closes::{metadata.get('pair')}",
                    {"pair": metadata.get('pair'),
                     "closes": [float(x) for x in recent_closes]},
                    half_life=900.0,  # 15 min — refreshed every populate cycle
                )
        except Exception as _rc_e:
            logger.debug(f"[Indicators:RecentCloses] {metadata.get('pair')} failed: {_rc_e}")

        # Sentiment features from SQLite (used by custom_stake_amount)
        # 2026-05-26 leak hardening: close() now in finally so an exception
        # outside the inner try/except (KeyboardInterrupt, OOM, etc.) can't
        # leave the FD dangling.
        conn = self._get_sqlite_connection()
        if conn:
            try:
                pair = metadata['pair']
                base_coin = pair.split('/')[0]
                try:
                    fng_df = pd.read_sql_query(
                        "SELECT value as fng_value FROM fear_and_greed ORDER BY timestamp DESC LIMIT 1", conn)
                    dataframe['%-fng_index'] = fng_df['fng_value'].iloc[0] if not fng_df.empty else 50
                except Exception:
                    dataframe['%-fng_index'] = 50
                try:
                    sent_df = pd.read_sql_query(
                        "SELECT sentiment_1h, sentiment_4h, sentiment_24h FROM coin_sentiment_rolling "
                        "WHERE coin = ? ORDER BY timestamp DESC LIMIT 1", conn, params=(base_coin,))
                    if not sent_df.empty:
                        dataframe['%-sentiment_1h'] = sent_df['sentiment_1h'].iloc[0]
                        dataframe['%-sentiment_4h'] = sent_df['sentiment_4h'].iloc[0]
                        dataframe['%-sentiment_24h'] = sent_df['sentiment_24h'].iloc[0]
                    else:
                        dataframe['%-sentiment_1h'] = 0.0
                        dataframe['%-sentiment_4h'] = 0.0
                        dataframe['%-sentiment_24h'] = 0.0
                except Exception:
                    dataframe['%-sentiment_1h'] = 0.0
                    dataframe['%-sentiment_4h'] = 0.0
                    dataframe['%-sentiment_24h'] = 0.0
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
        else:
            dataframe['%-fng_index'] = 50
            dataframe['%-sentiment_1h'] = 0.0
            dataframe['%-sentiment_4h'] = 0.0
            dataframe['%-sentiment_24h'] = 0.0

        # Phase 26: Chart Structure Intelligence (6 layers, ~130-230 features)
        try:
            from chart_features import compute_chart_features

            # Get 4h and 1d informative pair data if available
            pair = metadata['pair']
            stake = self.config.get('stake_currency', 'USDT')
            df_4h = None
            df_1d = None
            if self.dp:
                try:
                    df_4h = self.dp.get_pair_dataframe(pair=f"BTC/{stake}", timeframe="4h")
                except Exception:
                    pass
                try:
                    df_1d = self.dp.get_pair_dataframe(pair=f"BTC/{stake}", timeframe="1d")
                except Exception:
                    pass

            chart = compute_chart_features(dataframe, df_4h=df_4h, df_1d=df_1d, include_signature=True)
            # Use pd.concat to avoid DataFrame fragmentation (193 columns at once, not one by one)
            import pandas as _pd
            chart_df = _pd.DataFrame({f'%-{k}': [v] * len(dataframe) for k, v in chart.items()}, index=dataframe.index)
            dataframe = _pd.concat([dataframe, chart_df], axis=1)
            logger.info(f"[Phase26:ChartFeatures] {len(chart)} features computed for {pair}")
        except Exception as e:
            logger.warning(f"[Phase26:ChartFeatures] Failed: {e}")

        return dataframe

    def populate_entry_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        pair = metadata['pair']
        df['enter_long'] = 0
        df['enter_short'] = 0

        # Pair-circuit gate: skip entry scouting entirely while a pair's
        # exchange interaction is in dormant/backoff (after 5 consecutive
        # empty-book events). scheduler._pair_revive_tick reopens the
        # circuit once the book is healthy again.
        try:
            from pair_circuit import get_pair_circuit
            if get_pair_circuit().is_dormant(pair):
                return df
        except Exception:
            pass

        # ═════════════════════════════════════════════════════════════
        # LEARNING GATE — LOOP-1 (shadow per-pair) + LOOP-4 (Thompson)
        # ─────────────────────────────────────────────────────────────
        # Two independent posteriors decide whether this pair is worth
        # scouting THIS tick:
        #   • LOOP-1: shadow Kelly Beta posterior fed by forgone PnL
        #     resolutions (pair we DIDN'T trade — counterfactual win rate)
        #   • LOOP-4: real Kelly Beta posterior fed by actual trade
        #     outcomes
        # Each is sampled via Thompson — exploration when uncertain,
        # exploitation when converged. The strict gate (`min < 0.30`)
        # only fires if we have ENOUGH evidence (n_real >= 3 OR
        # n_shadow >= 5) so cold-start pairs still get explored.
        # ═════════════════════════════════════════════════════════════
        try:
            self._learning_gate_skip_count = getattr(self, "_learning_gate_skip_count", 0)
            shadow_score = None
            try:
                from pheromone_field import get_pheromone_field
                pfield = get_pheromone_field()
                shadow_payload = pfield.read(f"shadow_score::{pair}", source="shadow_kelly")
                if isinstance(shadow_payload, dict):
                    n_shadow = int(shadow_payload.get("n_shadow", 0) or 0)
                    if n_shadow >= 5:
                        shadow_score = float(shadow_payload.get("score", 0.5))
            except Exception:
                pass

            real_score = None
            real_n = 0
            try:
                from position_sizer import get_real_kelly
                real_kelly = get_real_kelly()
                stats = real_kelly._load_pair(pair, regime="_global")
                real_n = int(stats.get("n_trades", 0) or 0)
                if real_n >= 3:
                    import numpy as _np_lg
                    a = float(stats.get("alpha", 2.0))
                    b = float(stats.get("beta_param", 2.0))
                    real_score = float(_np_lg.random.beta(max(a, 0.5), max(b, 0.5)))
            except Exception:
                pass

            if shadow_score is not None and shadow_score < 0.30:
                self._learning_gate_skip_count += 1
                logger.info(
                    f"[Loop-1:LearningGate] {pair} SKIP — "
                    f"shadow_score={shadow_score:.3f} < 0.30 (forgone evidence "
                    f"says this pair would lose more often than win)"
                )
                return df

            if real_score is not None and real_score < 0.30 and real_n >= 5:
                self._learning_gate_skip_count += 1
                logger.info(
                    f"[Loop-4:LearningGate] {pair} SKIP — "
                    f"real_kelly_thompson={real_score:.3f} < 0.30 with n={real_n} "
                    f"(actual trade history says this pair does not work for us)"
                )
                return df

            # Soft warning when one signal is weak and the other absent —
            # operator visibility, no skip.
            if shadow_score is not None and shadow_score < 0.40 and real_score is None:
                logger.debug(
                    f"[LearningGate] {pair} weak shadow={shadow_score:.3f} but "
                    f"no real-Kelly evidence yet — exploring"
                )

            # ─── T3 — Order-flow squeeze veto ─────────────────────────
            try:
                from pheromone_field import get_pheromone_field as _gpf_t3
                of_state = _gpf_t3().read("order_flow_state", source="order_flow")
                if isinstance(of_state, dict) and of_state.get("pair") == pair:
                    sq_long = float(of_state.get("squeeze_long", 0.0))
                    sq_short = float(of_state.get("squeeze_short", 0.0))
                    flow_tox = float(of_state.get("flow_toxicity", 0.0))
                    # If short-squeeze probability is high, fading shorts is dangerous
                    # (price likely to spike up). Block SHORT entries.
                    # Symmetric for long-squeeze (capitulation candle imminent).
                    if sq_short > 0.70:
                        logger.info(
                            f"[T3:OrderFlow] {pair} SHORT veto — squeeze_short={sq_short:.2f} "
                            f"(price likely to spike up, fading shorts risky)"
                        )
                        df['enter_short'] = 0
                    if sq_long > 0.70:
                        logger.info(
                            f"[T3:OrderFlow] {pair} LONG veto — squeeze_long={sq_long:.2f}"
                        )
                        df['enter_long'] = 0
                    if flow_tox > 0.75:
                        logger.info(
                            f"[T3:OrderFlow] {pair} BOTH veto — flow_toxicity={flow_tox:.2f} "
                            f"(book is being manipulated, skip both sides)"
                        )
                        return df
            except Exception:
                pass

            # ─── T16 — FEAR_LEVEL gate ────────────────────────────────
            try:
                from pheromone_field import get_pheromone_field as _gpf_t16
                fear = _gpf_t16().read("FEAR_LEVEL", source="neural_organism")
                if isinstance(fear, dict):
                    tier = fear.get("tier", "normal")
                    fear_lvl = float(fear.get("fear_level", 0.0))
                    if str(tier or "").lower() in ("panic", "extreme") or fear_lvl >= 2.0:
                        logger.info(
                            f"[T16:FearGate] {pair} SKIP — fear tier={tier} level={fear_lvl:.2f} "
                            f"(no new exposure under amygdala panic)"
                        )
                        return df
            except Exception:
                pass

            # ─── T17 — Market-maker spread gate ───────────────────────
            try:
                from pheromone_field import get_pheromone_field as _gpf_t17
                mm = _gpf_t17().read("mm_state", source="market_maker")
                if isinstance(mm, dict):
                    half_spread_pct = float(mm.get("half_spread_pct",
                                                     mm.get("spread_pct", 0.0) / 2.0))
                    # 25 bps = 0.25%. Above this, MM module already says "toxic".
                    if half_spread_pct > 0.25:
                        logger.info(
                            f"[T17:MMSpread] {pair} SKIP — half_spread={half_spread_pct:.2f}% "
                            f"(MM module flags toxic spread)"
                        )
                        return df
            except Exception:
                pass
        except Exception as _gate_e:
            # Gate is non-blocking by design: a failure here must NEVER
            # prevent scouting. Log debug so we can spot a regression.
            logger.debug(f"[LearningGate] error (proceeding): {_gate_e}")

        # Task 26: consume ProactiveDispatcher's trade_frequency_hint.
        # When PredictiveInteroception fires "Reduce trade frequency, run
        # diagnostic" (organism_health below danger), the dispatcher
        # deposits a min-gap-in-seconds. Skip entry scouting if another
        # entry on this pair happened within that window. Pair-scoped so
        # a healthy pair isn't locked by a separate pair's recent entry.
        try:
            from pheromone_field import get_pheromone_field
            tf_hint = get_pheromone_field().read("trade_frequency_hint")
            if isinstance(tf_hint, dict):
                min_gap = int(tf_hint.get("min_gap_seconds", 0) or 0)
                if min_gap > 0:
                    if not hasattr(self, "_last_entry_ts_per_pair"):
                        self._last_entry_ts_per_pair: dict[str, float] = {}
                    import time as _tm
                    last_ts = self._last_entry_ts_per_pair.get(pair, 0.0)
                    if _tm.time() - last_ts < min_gap:
                        return df
        except Exception:
            pass

        if self.dp.runmode.value in ('dry_run', 'live'):
            last_time = df['date'].iloc[-1]
            current_rate = df['close'].iloc[-1]

            # Phase 10: Invalidate semantic cache if sudden market movement is >3%
            if len(df) > 1:
                prev_close = df['close'].iloc[-2]
                if prev_close > 0 and abs(current_rate - prev_close) / prev_close > 0.03:
                    logger.info(f"Significant price movement >3% detected for {pair}. Invalidating semantic cache.")
                    if not hasattr(self, '_semantic_cache'):
                        from semantic_cache import SemanticCache
                        self._semantic_cache = SemanticCache(db_path=self.db_path)
                    self._semantic_cache.invalidate(pair=pair)

            # Phase 26: Triple Perception (TTM + Chronos + CatBoost) — runs HERE because
            # DataFrame is only available in strategy, not in rag_graph HTTP service.
            _tp_result = None
            try:
                from triple_perception import get_triple_perception
                _tp = get_triple_perception()

                # Get 4h/1d data for chart features
                stake = self.config.get('stake_currency', 'USDT')
                _tp_4h = self.dp.get_pair_dataframe(pair=f"BTC/{stake}", timeframe="4h") if self.dp else None
                _tp_1d = self.dp.get_pair_dataframe(pair=f"BTC/{stake}", timeframe="1d") if self.dp else None

                # Compute chart features
                from chart_features import compute_chart_features
                _tp_chart = compute_chart_features(df, df_4h=_tp_4h, df_1d=_tp_1d, include_signature=True)

                _tp_result = _tp.perceive(
                    df_1h=df,
                    df_4h=_tp_4h,
                    df_1d=_tp_1d,
                    chart_features=_tp_chart,
                    pair=pair,
                    timeframe=self.timeframe,
                )
                if _tp_result:
                    # Store for sizing/confidence enrichment downstream
                    # Sprint 2026-05-06 (F1): LRU-bounded — was unbounded dict
                    # leaking ~100MB/h. Init now in __init__; setter applies
                    # dynamic maxlen via shared organism-aware budget.
                    self._perception_cache_set(pair, _tp_result)
                    logger.info(
                        f"[Phase26:TriplePerception] {pair}: {_tp_result['signal']} "
                        f"conf={_tp_result['confidence']:.2f} sizing={_tp_result['sizing_multiplier']:.2f}"
                    )
            except Exception as e:
                logger.warning(f"[Phase26:TriplePerception] {pair} failed: {e}", exc_info=True)

            ai_decision = self._get_ai_signal(pair, last_time, dataframe=df)

            # Sprint 2026-05-01: Triple-Perception promotion. The legacy
            # logic only boosted/penalized when AI was already directional
            # — when EVIDENCE_ENGINE returned NEUTRAL conf=0.11 (the
            # 28-Apr-to-1-May NO-TRADE pathology) the boost path was a
            # no-op and TP's strong directional reading was wasted. Now
            # we PROMOTE TP to primary signal when (a) AI is NEUTRAL or
            # weak and (b) regime is directional and (c) TP's confidence
            # clears the envelope-driven promotion threshold. Boost /
            # disagreement penalties retain their previous behaviour
            # when the promotion gate doesn't fire.
            if _tp_result and _tp_result.get("confidence", 0) > 0.1:
                tp_signal = _tp_result.get("signal", "NEUTRAL")
                tp_conf = float(_tp_result.get("confidence", 0.0) or 0.0)
                ai_signal = ai_decision.get("signal", "NEUTRAL")
                ai_conf = float(ai_decision.get("confidence", 0.0) or 0.0)
                regime_now = ai_decision.get("regime") or "transitional"

                # Pull dynamic gates from RiskEnvelope (no hardcode).
                try:
                    from risk_envelope import get_risk_envelope
                    env = get_risk_envelope()
                    env_floor = float(env.conviction_floor())
                    tp_min_conf = float(env.tp_promotion_threshold())
                    tp_haircut = float(env.tp_promotion_haircut())
                except Exception:
                    env_floor = 0.45
                    tp_min_conf = 0.50
                    tp_haircut = 0.85

                try:
                    from regime_classifier import RegimeClassifier
                    regime_directional = RegimeClassifier.is_directional(regime_now)
                except Exception:
                    regime_directional = regime_now in ("trending_bull", "trending_bear")

                ai_too_weak = (ai_signal == "NEUTRAL") or (ai_conf < env_floor)

                # 2026-05-26: RAG-DEAD FALLBACK. When RAG/EvidenceEngine times
                # out the response is the line-1146 default {NEUTRAL, 0.0, no
                # regime}, so regime_now collapses to "transitional" and the
                # original promotion gate (requires regime_directional) silently
                # blocks every fallback path — exactly the 5-day NO-TRADE
                # pathology we just lived through.
                #
                # Detection covers two collapse modes:
                #   (a) exception/timeout — defaults stay untouched
                #       (NEUTRAL, 0.0, no source)
                #   (b) HTTP non-200 — defaults also stay (handled by the
                #       same `signal_data` dict).
                # We extend (a) to include very-low-conf RAG returns too, so
                # the bot doesn't sit on a 0.05-conf RAG result while TP has
                # a directional reading worth trusting.
                #
                # tp floor: 0.50 (not 0.55). TP is already three independent
                # backbones (Kronos + chart + sentiment) — a 0.50 directional
                # confidence is the same bar as TP-Promoted normal path. The
                # stricter 0.55 was pre-production conservatism; in practice
                # it gated out exactly the BEARISH 0.54 signals we observed
                # under the RAG-down scenario above.
                rag_dead = (
                    (ai_signal == "NEUTRAL" and ai_conf < 0.20)
                    or not ai_decision.get("source")
                )
                strict_tp_floor = max(tp_min_conf, 0.50)

                if (ai_too_weak and regime_directional
                        and tp_signal != "NEUTRAL" and tp_conf >= tp_min_conf):
                    # PROMOTE — TP becomes the primary signal, with a
                    # haircut to remain humble about the override.
                    promoted_conf = min(0.95, tp_conf * tp_haircut)
                    logger.info(
                        f"[Phase26:TP-Promoted] {pair} AI={ai_signal}/{ai_conf:.2f} "
                        f"→ TP={tp_signal}/{tp_conf:.2f} (promoted to {promoted_conf:.2f}, "
                        f"regime={regime_now}, env_floor={env_floor:.2f})"
                    )
                    ai_decision["signal"] = tp_signal
                    ai_decision["confidence"] = promoted_conf
                    ai_decision["sizing_multiplier"] = _tp_result.get("sizing_multiplier", 1.0)
                    ai_decision["source"] = "TRIPLE_PERCEPTION_PROMOTED"
                    ai_decision["reasoning"] = (
                        f"[TP-Promoted: AI was {ai_signal}/{ai_conf:.2f} weak in {regime_now}] "
                        + str(_tp_result.get("reasoning", ""))
                    )
                elif (rag_dead
                        and tp_signal != "NEUTRAL"
                        and tp_conf >= strict_tp_floor):
                    promoted_conf = min(0.85, tp_conf * tp_haircut)
                    logger.warning(
                        f"[Phase26:TP-PromotedRAGDead] {pair} RAG/EvidenceEngine "
                        f"returned default fallback (NEUTRAL/0.0) → trusting TP="
                        f"{tp_signal}/{tp_conf:.2f} (promoted to {promoted_conf:.2f}, "
                        f"regime={regime_now}, strict_floor={strict_tp_floor:.2f})"
                    )
                    ai_decision["signal"] = tp_signal
                    ai_decision["confidence"] = promoted_conf
                    ai_decision["sizing_multiplier"] = _tp_result.get("sizing_multiplier", 1.0)
                    ai_decision["source"] = "TRIPLE_PERCEPTION_RAG_DEAD"
                    ai_decision["reasoning"] = (
                        f"[TP-Promoted RAG-dead: AI returned NEUTRAL/0.0 default] "
                        + str(_tp_result.get("reasoning", ""))
                    )
                elif tp_signal == ai_signal and tp_signal != "NEUTRAL":
                    ai_decision["confidence"] = min(ai_conf * 1.15, 0.95)
                    ai_decision["sizing_multiplier"] = _tp_result.get("sizing_multiplier", 1.0)
                elif tp_signal != "NEUTRAL" and ai_signal != "NEUTRAL" and tp_signal != ai_signal:
                    ai_decision["confidence"] = ai_conf * 0.75
                ai_decision["perception"] = _tp_result

            signal_type = ai_decision.get('signal', 'NEUTRAL')
            confidence = ai_decision.get('confidence', 0.0)
            is_bullish = signal_type == 'BULLISH'
            is_bearish = signal_type == 'BEARISH'

            # ═══ Sprint 2026-05-01 night — MULTI-TIMEFRAME AGREEMENT GATE ═══
            # Real edge rule: trade only when 1h, 4h, and daily trends agree
            # with the directional signal. Reduces false breakout signals
            # in choppy markets by 40-60% (academic literature consensus).
            # 1h trend = current ai_decision direction
            # 4h trend = htf["trend_4h"] (bullish/bearish from price-vs-EMA20)
            # daily   = htf["trend_daily"] (price-vs-EMA50)
            # When fewer than `mtf_required` higher TFs agree, downgrade
            # to SHADOW so the calibrator still sees the signal but no
            # real trade fires.
            #
            # Audit fix (2026-05-01 night, Finding A1): the prior
            # implementation read `technical_data` from an unbound local
            # in this method's scope (the variable lives in
            # bot_loop_start / _get_ai_signal, not here) — the resulting
            # NameError got swallowed by the try/except, so mtf_agreement
            # stayed "neutral" and the gate never fired. Compute htf
            # DIRECTLY from the dataframe via the existing helper.
            mtf_agreement = "neutral"
            mtf_agree_count = 0
            try:
                htf = self._compute_higher_timeframe(df)
                trend_4h = htf.get("trend_4h", "")
                trend_daily = htf.get("trend_daily", "")
                if signal_type in ("BULLISH", "BEARISH"):
                    target_word = "bullish" if is_bullish else "bearish"
                    mtf_agree_count = (
                        (1 if trend_4h == target_word else 0)
                        + (1 if trend_daily == target_word else 0)
                    )
                    try:
                        from neural_organism import _p as _np
                        mtf_required = int(_np("envelope.mtf.required_agreement", 1))
                    except Exception:
                        mtf_required = 1  # at least 1 of (4h, daily) must agree
                    if mtf_agree_count >= mtf_required:
                        mtf_agreement = "agreed"
                    else:
                        mtf_agreement = "disagreed"
                    logger.info(
                        f"[MTF-Gate] {pair} sig={signal_type} "
                        f"4h={trend_4h or 'unknown'} daily={trend_daily or 'unknown'} "
                        f"agree={mtf_agree_count}/2 required={mtf_required} "
                        f"verdict={mtf_agreement}"
                    )
                # Audit 2026-05-02 #8 fix: persist canonical MTF state on
                # ai_decision so SignalSourceConsensus can read the TRUE
                # 4h/daily resampled trends instead of mis-deriving them
                # from 1h ema_50/ema_200 (which approximate ~50h/200h
                # trends, not 4h/daily).
                ai_decision["mtf_state"] = {
                    "trend_4h": trend_4h,
                    "trend_daily": trend_daily,
                    "agree_count": mtf_agree_count,
                    "agreement": mtf_agreement,
                    "required": mtf_required,
                }
            except Exception as _mtf_e:
                logger.debug(f"[MTF-Gate] {pair} agreement check failed: {_mtf_e}")

            # ═══ GRADUATED EXECUTION: Log ALL signals, trade only high-confidence ═══
            # Sprint 2026-05-01: REAL_TRADE_THRESHOLD is now FULLY dynamic.
            # Components blended via geometric mean so all three voices
            # contribute, none dominates:
            #   1. RiskEnvelope.conviction_floor()  — autonomy/decay-driven
            #   2. _pair_confidence_threshold      — per-pair forgone alpha
            #   3. self.confidence_threshold       — hyperopt safety ceiling
            # geo-mean keeps the bar between the three, so an envelope at L0
            # (0.45) AND an aggressively-tuned pair_thr (0.40) yield
            # ~0.42 — neither extreme dictates alone.
            try:
                from risk_envelope import get_risk_envelope
                env_floor = float(get_risk_envelope().conviction_floor())
            except Exception:
                env_floor = float(self.confidence_threshold.value)
            try:
                regime_for_thr = ai_decision.get("regime") or "_global"
                pair_thr = float(self._pair_confidence_threshold(pair, regime_for_thr))
            except Exception:
                pair_thr = float(self.confidence_threshold.value)
            hyper_thr = float(self.confidence_threshold.value)
            # Geometric mean of (env_floor, pair_thr, hyper_thr) with a
            # safety floor to make sure no path goes below env_floor.
            try:
                geo = (env_floor * pair_thr * hyper_thr) ** (1.0 / 3.0)
            except Exception:
                geo = hyper_thr
            REAL_TRADE_THRESHOLD = max(env_floor, geo)
            shadow_floor = max(0.20, env_floor * 0.5)

            # Sprint 2026-05-02: SignalSourceConsensus override.
            # When RAG returned NEUTRAL/weak but 4+ of 6 signal sources
            # (TP, MTF, VPIN, funding, Hurst) agree directionally, the
            # consensus replaces RAG's verdict. Also detects RAG bias
            # (rolling distribution mode > 50% on one value) and demotes
            # RAG's weight in that case.
            try:
                from signal_source_consensus import compute_consensus
                consensus = compute_consensus(pair, ai_decision, df)
                if consensus.get("action") == "override":
                    new_signal = consensus["consensus_signal"]
                    new_conf = float(consensus["consensus_conf"])
                    logger.info(
                        f"[Consensus:Override] {pair} RAG={signal_type}/{confidence:.2f} → "
                        f"{new_signal}/{new_conf:.2f} "
                        f"(agreement={consensus['agreement']}/6, "
                        f"sources_active={consensus['sources_active']}, "
                        f"rag_biased={consensus['rag_bias_active']})"
                    )
                    signal_type = new_signal
                    confidence = new_conf
                    is_bullish = (signal_type == "BULLISH")
                    is_bearish = (signal_type == "BEARISH")
                    ai_decision["signal"] = signal_type
                    ai_decision["confidence"] = confidence
                    ai_decision["source"] = "CONSENSUS_OVERRIDE"
                    ai_decision["consensus_breakdown"] = consensus.get("breakdown")
            except Exception as _cons_e:
                logger.debug(f"[Consensus] {pair} skipped: {_cons_e}")

            # Determine execution mode for logging.
            # Sprint 2026-05-01 night — MTF gate downgrades REAL→SHADOW
            # when 1h/4h/daily trends disagree with the entry direction.
            if confidence >= REAL_TRADE_THRESHOLD and signal_type != 'NEUTRAL':
                if mtf_agreement == "disagreed":
                    exec_mode = "SHADOW"
                    logger.info(
                        f"[MTF-Gate] {pair} {signal_type} conf={confidence:.2f} "
                        f"DOWNGRADED to SHADOW — only {mtf_agree_count}/2 higher TFs agree"
                    )
                else:
                    exec_mode = "REAL"
            elif confidence >= shadow_floor:
                exec_mode = "SHADOW"      # Decent signal, paper trade it
            else:
                exec_mode = "SHADOW_WEAK"  # Garbage signal, still log for learning

            # Log EVERY signal to forgone engine (shadow or real).
            # Data Acceleration audit: pass the REAL sub-scores from the
            # evidence engine so the shadow → CatBoost pipeline stops
            # training on constant 0.5 placeholders. ai_decision["sub_scores"]
            # is populated by evidence_engine._synthesize().
            sig_label = "BULL" if is_bullish else ("BEAR" if is_bearish else "NEUTRAL")
            _sub = ai_decision.get("sub_scores", {}) or {}
            fid = self.forgone_engine.log_forgone_signal(
                pair=pair,
                signal_type=sig_label,
                confidence=confidence,
                entry_price=float(current_rate),
                was_executed=(exec_mode == "REAL"),
                regime=ai_decision.get("regime"),
                trust_score=float(ai_decision.get("trust_score", 0.5) or 0.5),
                sub_trend=float(_sub.get("q1_trend", 0.5) or 0.5),
                sub_momentum=float(_sub.get("q2_momentum", 0.5) or 0.5),
                sub_crowd=float(_sub.get("q3_crowd", 0.5) or 0.5),
                sub_evidence=float(_sub.get("q4_evidence", 0.5) or 0.5),
                sub_macro=float(_sub.get("q5_macro", 0.5) or 0.5),
                sub_risk=float(_sub.get("q6_risk", 0.5) or 0.5),
            )
            if fid:
                self._forgone_ids[pair] = fid

            # Set entry signals — ONLY for high-confidence directional signals
            if exec_mode == "REAL" and is_bullish:
                df.iloc[-1, df.columns.get_loc('enter_long')] = 1
                logger.info(f"[Signal:REAL] {pair} → enter_long=1 (BULLISH conf={confidence:.2f})")
            elif exec_mode == "REAL" and is_bearish:
                df.iloc[-1, df.columns.get_loc('enter_short')] = 1
                logger.info(f"[Signal:REAL] {pair} → enter_short=1 (BEARISH conf={confidence:.2f})")
            else:
                # Shadow trade: NO real entry, just learn from the outcome
                logger.info(f"[Signal:{exec_mode}] {pair} → NO TRADE ({signal_type} conf={confidence:.2f}) — shadow tracking for calibrator")
        else:
            # Backtesting: Simple technical signals
            if 'rsi' in df.columns and 'macd' in df.columns:
                df.loc[(df['rsi'] < 35) & (df['macd'] > df['macdsignal']), 'enter_long'] = 1
                df.loc[(df['rsi'] > 65) & (df['macd'] < df['macdsignal']), 'enter_short'] = 1

        # 2026-05-27 confirm_trade_entry mystery probe: log exit state of
        # populate_entry_trend so we can correlate against [GES-CALLED] / the
        # absence thereof. Together these prove whether freqtrade ever picks
        # up the entry signal we just placed on df.iloc[-1].
        try:
            _last = df.iloc[-1]
            _el = int(_last.get('enter_long', 0))
            _es = int(_last.get('enter_short', 0))
            if _el or _es:
                logger.info(
                    f"[POPULATE-EXIT] {pair} enter_long={_el} enter_short={_es} "
                    f"rows={len(df)} last_date={_last.get('date', 'N/A')}"
                )
        except Exception as _pex:
            logger.debug(f"[POPULATE-EXIT] {pair} probe failed: {_pex}")

        return df

    def populate_exit_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        df['exit_long'] = 0
        df['exit_short'] = 0

        if self.dp.runmode.value in ('dry_run', 'live'):
            # Exit signals for OPEN POSITIONS only — never conflict with same-candle entry signals.
            # Freqtrade rejects entries when exit_long/exit_short is set on the same candle:
            #   get_entry_signal: enter_long == 1 and not any([exit_long, enter_short])
            # So we MUST NOT set exit signals that conflict with current entry signals.
            pair = metadata['pair']
            cached = self.ai_signal_cache.get(pair)
            if cached:
                last_enter_long = df.iloc[-1].get('enter_long', 0)
                last_enter_short = df.iloc[-1].get('enter_short', 0)

                if cached['signal'] == 'BEARISH' and not last_enter_long:
                    df.iloc[-1, df.columns.get_loc('exit_long')] = 1
                    logger.debug(f"[Exit] {pair}: exit_long=1 (BEARISH, no entry conflict)")
                elif cached['signal'] == 'BULLISH' and not last_enter_short:
                    df.iloc[-1, df.columns.get_loc('exit_short')] = 1
                    logger.debug(f"[Exit] {pair}: exit_short=1 (BULLISH, no entry conflict)")
        else:
            # Backtesting: Technical exit signals
            if 'rsi' in df.columns and 'macd' in df.columns:
                df.loc[(df['rsi'] > 70) & (df['macd'] < df['macdsignal']), 'exit_long'] = 1
                df.loc[(df['rsi'] < 30) & (df['macd'] > df['macdsignal']), 'exit_short'] = 1

        return df

    # ═══════════════════════════════════════════════════════════
    # Phase 27 Task 11 helpers — CAAT multiplier + per-pair threshold
    # ═══════════════════════════════════════════════════════════

    def _caat_asymmetric_multiplier(
        self,
        pair: str,
        regime: str,
        confidence: float,
        ai_decision: dict,
        last_candle,
        proposed_stake: float,
    ) -> tuple:
        """Phase 27 Task 11 CAAT Asymmetric Alpha sizing multiplier.

        Combines PARÇAs 3 (Hormonal), 4 (Dream), 5 (VWTSMOM/fractal regime),
        6 (Shannon harvest), 7 (Impact sqrt-law), 9 (Confidence integral),
        and 10 (Forgone alpha) into a single multiplicative adjustment applied
        on top of the existing Kelly×confidence×sizing_multiplier fraction.
        PARÇAs 1 (Kelly), 2 (confidence trust curve), 8 (Hawkes veto/clamp)
        are already applied elsewhere in the sizing pipeline.

        Returns (multiplier, breakdown_dict). The multiplier is hard-clamped
        to [0.2, 1.5] so no single layer can singlehandedly dominate.
        """
        import math as _math
        breakdown = {}
        mult = 1.0

        # ── PARÇA 3: Hormonal scalar ──
        # Code convention: LOW cortisol = STRESSED (floor 0.5), HIGH = CALM (1.0).
        # Post-audit fix: the ALPHA doc pseudocode used biology convention (high
        # cortisol = stressed) and wrote `1/cortisol`, which in OUR convention
        # INVERTED the signal — stressed organism got a LARGER multiplier.
        # Correct form: multiply BY cortisol so stressed → shrink, calm → neutral.
        try:
            from neural_organism import get_organism
            org = get_organism()
            h = org.hormones
            hormonal = h.dopamine * max(h.cortisol, 0.3) * h.serotonin
            hormonal = max(0.5, min(2.0, hormonal))
            mult *= hormonal
            breakdown["hormonal"] = round(hormonal, 3)
        except Exception:
            breakdown["hormonal"] = 1.0

        # ── PARÇA 4: Dream Familiarity (only if the engine has data) ──
        try:
            from dream_engine import get_dream_engine
            eng = get_dream_engine()
            if eng._filter._stats_computed:
                # If the organism has been dreaming, increase the bonus
                # marginally — we don't have per-state variance lookup yet,
                # so this is a one-step +10% favour of the familiar regime.
                mult *= 1.05
                breakdown["dream"] = 1.05
            else:
                breakdown["dream"] = 1.0
        except Exception:
            breakdown["dream"] = 1.0

        # ── PARÇA 5: Fractal regime filter (Hurst + VWTSMOM) ──
        # chart_features keys land in the dataframe with a `%-` prefix (see
        # populate_indicators L780: `f'%-{k}': [v] * len(dataframe)`). Without
        # the prefix every call fell back to 0.5 (neutral) and the regime
        # filter was a no-op.
        hurst = 0.5
        try:
            hurst = float(
                last_candle.get("%-hurst_100",
                                last_candle.get("%-hurst_50",
                                                last_candle.get("hurst_100",
                                                                last_candle.get("hurst_50", 0.5))))
                or 0.5
            )
        except Exception:
            pass
        if hurst > 0.55:
            regime_mult = 1.2  # trending → full engagement
        elif hurst < 0.45:
            regime_mult = 0.6  # anti-persistent → halve
        else:
            regime_mult = 0.9
        mult *= regime_mult
        breakdown["hurst_regime"] = round(regime_mult, 3)

        # ── PARÇA 6: Shannon harvest — ranging markets size smaller ──
        adx = 0.0
        try:
            adx = float(last_candle.get("adx") or last_candle.get("adx_14") or 0.0)
        except Exception:
            adx = 0.0
        if adx and adx < 20.0:
            mult *= 0.6
            breakdown["shannon"] = 0.6
        else:
            breakdown["shannon"] = 1.0

        # ── PARÇA 7: Impact constraint (sqrt-law) ──
        try:
            from slippage_forecaster import sqrt_law_impact_bps
            # Real 24h quote volume from Bybit ticker (fallback chain:
            # ticker → market_data → conservative 1M USDT). The old fallback
            # of `proposed_stake * 10000` made the impact check a no-op.
            adv_usd = None
            try:
                ticker = self.dp.ticker(pair) if hasattr(self, "dp") else None
                if ticker:
                    for key in ("quoteVolume", "quoteVolume24h", "quote_volume_24h"):
                        if ticker.get(key):
                            adv_usd = float(ticker[key])
                            break
            except Exception:
                pass
            if adv_usd is None:
                try:
                    _md = getattr(self, "_market_data", None) or ai_decision.get("market_data", {})
                    if isinstance(_md, dict):
                        adv_usd = float(
                            _md.get("adv_usd") or _md.get("average_daily_volume") or 0.0
                        )
                except Exception:
                    adv_usd = None
            if not adv_usd or adv_usd <= 0:
                # Last-resort floor — 1M USDT/day is realistic for most liquid
                # perps; if the pair is thinner than that the impact check
                # will (correctly) flag the trade as expensive.
                adv_usd = 1_000_000.0

            sigma_bps = 200.0
            try:
                sigma_bps = float(ai_decision.get("sigma_bps") or 200.0)
            except Exception:
                pass

            impact = sqrt_law_impact_bps(
                dollar_size=float(proposed_stake),
                adv_usd=adv_usd,
                sigma_daily_bps=sigma_bps,
            )
            # Convert cost → multiplier: if cost consumes > expected alpha,
            # clamp sizing proportionally (alpha proxy: confidence × 100 bps).
            expected_alpha_bps = max(10.0, confidence * 100.0)
            cost_ratio = impact["total_cost_bps"] / expected_alpha_bps
            impact_mult = max(0.3, 1.0 - min(1.0, cost_ratio) * 0.5)
            mult *= impact_mult
            breakdown["impact"] = round(impact_mult, 3)
            breakdown["adv_usd"] = round(float(adv_usd), 0)
        except Exception:
            breakdown["impact"] = 1.0

        # ── PARÇA 9: Confidence Integral (HQ-1) ──
        try:
            from pheromone_field import get_pheromone_field, PheromoneField
            pf = get_pheromone_field()
            integ = pf.read_integral("prediction", window_seconds=14400.0)
            grad = pf.read_gradient("prediction")
            # 4h-sustained bullish conviction → up to +20%; fading grad → -10%.
            integ_mult = 1.0 + 0.20 * max(-1.0, min(1.0, abs(integ) - 0.5))
            grad_mult = 1.0 + 0.10 * _math.tanh(grad * 100.0)
            combined = max(0.8, min(1.25, integ_mult * grad_mult))
            mult *= combined
            breakdown["integral"] = round(combined, 3)
        except Exception:
            breakdown["integral"] = 1.0

        # ── PARÇA 5b (Task 12): 4-Layer regime detection ──
        try:
            from regime_classifier import get_regime_detector
            det = get_regime_detector()
            tech_snapshot = {
                "adx": last_candle.get("adx") or last_candle.get("adx_14"),
                "atr": last_candle.get("atr") or last_candle.get("atr_14"),
                "price": last_candle.get("close") or last_candle.get("current_price"),
                "ema20": last_candle.get("ema_20") or last_candle.get("ema20"),
                "ema200": last_candle.get("ema_200") or last_candle.get("ema200"),
                "recent_closes": ai_decision.get("recent_closes") or [],
            }
            rl = det.detect(pair, tech_snapshot)
            regime_mult_4l = float(rl.get("sizing_modifier", 1.0))
            mult *= regime_mult_4l
            breakdown["regime_4layer"] = round(regime_mult_4l, 3)
        except Exception as e:
            logger.debug(f"[Phase27:4LayerRegime] skipped ({pair}): {e}")
            breakdown["regime_4layer"] = 1.0

        # ── Item 9: trinity_fusion sizing hook ──
        # trinity_fusion combines perception + sentiment + macro into a single
        # confidence-weighted signal. We use its `confidence_multiplier` as an
        # additional sizing factor — clamped to [0.5, 1.5] so it can't dominate.
        try:
            from trinity_fusion import get_trinity
            trinity = get_trinity()
            fusion_result = trinity.fuse(pair=pair, regime=regime)
            decision = fusion_result.get("decision", {}) if fusion_result.get("fused") else {}
            fusion_size = float(decision.get("sizing_multiplier", 1.0))
            fusion_mult = max(0.5, min(1.5, fusion_size))
            mult *= fusion_mult
            breakdown["trinity_fusion"] = round(fusion_mult, 3)
        except Exception as e:
            logger.debug(f"[Phase27:TrinityFusion] skipped: {e}")
            breakdown["trinity_fusion"] = 1.0

        # ── Item 11: HRL meta-policy organ selection ──
        # hrl_meta_policy picks which RL motor (IQL / SAC) should drive this
        # trade's sizing refinement. We read the meta-policy's organ weight
        # for 'sizing' and use it as a sizing modifier.
        try:
            from hrl_meta_policy import get_meta_policy
            meta = get_meta_policy()
            sizing_weight = meta.get_organ_weight("sizing") if hasattr(meta, "get_organ_weight") else 1.0
            meta_mult = max(0.7, min(1.3, float(sizing_weight)))
            mult *= meta_mult
            breakdown["hrl_meta"] = round(meta_mult, 3)
        except Exception as e:
            logger.debug(f"[Phase27:HRLMeta] skipped: {e}")
            breakdown["hrl_meta"] = 1.0

        # ── PARÇA 10: Forgone alpha adjustment ──
        # 2026-05-26 leak fix: with-block guarantees close() on exception.
        try:
            from db import get_db_connection
            with get_db_connection() as conn:
                row = conn.execute("""
                    SELECT forgone_alpha_7d FROM pair_thresholds
                    WHERE pair = ? AND regime = ?
                """, (pair, regime)).fetchone()
            if row is not None:
                alpha_7d = float(row["forgone_alpha_7d"] or 0.0)
                if alpha_7d > 2.0:
                    forgone_mult = 1.10  # we've been missing winners — be bolder
                elif alpha_7d < -1.0:
                    forgone_mult = 0.90  # catching losers — be more cautious
                else:
                    forgone_mult = 1.0
                mult *= forgone_mult
                breakdown["forgone"] = round(forgone_mult, 3)
            else:
                breakdown["forgone"] = 1.0
        except Exception:
            breakdown["forgone"] = 1.0

        # Hard clamp so no single layer dominates
        mult = max(0.2, min(1.5, mult))
        return mult, breakdown

    def _pair_confidence_threshold(self, pair: str, regime: str) -> float:
        """Phase 27 Fix 6: per-pair adaptive threshold lookup.

        Sprint 2026-05-01: fallback now sourced from RiskEnvelope's
        conviction_floor (autonomy + decay-driven) so even pairs with no
        threshold history follow the dynamic envelope rather than a
        hardcoded constant.
        """
        try:
            from db import get_db_connection
            with get_db_connection() as conn:
                row = conn.execute("""
                    SELECT confidence_threshold FROM pair_thresholds
                    WHERE pair = ? AND regime = ?
                """, (pair, regime)).fetchone()
            if row is not None:
                return float(row["confidence_threshold"])
        except Exception:
            pass
        try:
            from risk_envelope import get_risk_envelope
            return float(get_risk_envelope().conviction_floor())
        except Exception:
            return 0.50  # last-resort safety only on import failure

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Chandelier Exit — ATR-based trailing stoploss with confidence-adaptive distance.

        Uses highest_high_14 (LONG) or lowest_low_14 (SHORT) as the anchor, not current_rate.
        This creates a proper trailing stop that locks in profits as price advances.

        Confidence modulates trailing distance (research: IDS 2025):
          conf >= 0.80: 3.0x ATR (wide — stay in strong trend)
          conf 0.60-0.79: 2.5x ATR (normal)
          conf < 0.60: 2.0x ATR (tight — exit quickly on weak conviction)

        References:
          - Karassavidis et al. (2025) SSRN 5821842
          - Palazzi (2025) Journal of Futures Markets
          - LuxAlgo (2024): 2.0-2.5x ATR optimal for 1H crypto
        """
        # RE-3 (2026-04-25): SL fallback now envelope-driven. L0 → 15% (wide),
        # L5 → 5% (tight). Under panic decay the SL widens to give the trade
        # more room rather than panic-exit. The static `self.stoploss` is
        # the FreqTrade-level kill switch (-15%) — never violated.
        try:
            from risk_envelope import get_risk_envelope
            envelope_sl = -float(get_risk_envelope().get_sl_base_pct())
            sl_fallback = max(self.stoploss, envelope_sl)  # closer-to-zero wins (less negative)
        except Exception:
            sl_fallback = self.stoploss

        # Sprint 2026-05-01 night — STOP HUNT DEFENSE.
        # Whales periodically widen spreads 3-5x to trigger retail stops,
        # then snap back. We detect it via pair_circuit's rolling spread
        # distribution: when current spread > spread_multiplier × median,
        # temporarily LOOSEN stop by 50% so we don't get hunted out of
        # an otherwise healthy trade.
        stop_hunt_factor = 1.0
        try:
            from pair_circuit import get_pair_circuit
            from neural_organism import _p as _np_sh
            spread_mult = float(_np_sh("envelope.stop_hunt.spread_multiplier", 3.0))
            circuit = get_pair_circuit()
            ob_now = self.dp.orderbook(pair, 1)
            cur_spread = None
            if ob_now and ob_now.get("bids") and ob_now.get("asks"):
                bid_p = float(ob_now["bids"][0][0])
                ask_p = float(ob_now["asks"][0][0])
                if bid_p > 0 and ask_p > 0:
                    cur_spread = max(0.0, 1.0 - bid_p / ask_p)
            dist = circuit.get_exchange_spread_distribution()
            pair_median = dist.get(pair)
            if cur_spread is not None and pair_median and pair_median > 0:
                ratio = cur_spread / pair_median
                if ratio >= spread_mult:
                    # Stop loosens proportionally (max 1.5×) so the trade
                    # rides through the manipulation candle.
                    stop_hunt_factor = min(1.5, 1.0 + (ratio - spread_mult) * 0.25)
                    logger.warning(
                        f"[StopHuntDefense] {pair} spread {cur_spread:.4%} = "
                        f"{ratio:.1f}× median {pair_median:.4%} — loosening stop {stop_hunt_factor:.2f}×"
                    )
        except Exception as _sh_e:
            logger.debug(f"[StopHuntDefense] {pair} skipped: {_sh_e}")

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or len(dataframe) < 2:
            return sl_fallback

        last_candle = dataframe.iloc[-1].squeeze()

        atr = last_candle.get('atr')
        if not atr or atr != atr or current_rate <= 0:  # NaN check + sanity
            return sl_fallback

        atr = float(atr)

        # ═══ BREAKEVEN CHECK (after partial profit lock) ═══
        breakeven_active = False
        try:
            breakeven_active = trade.get_custom_data("breakeven_active", False)
        except Exception:
            pass

        if breakeven_active and current_rate > 0 and trade.open_rate > 0:
            if not trade.is_short:
                breakeven_sl = (trade.open_rate * _np("strategy.breakeven_long", 0.998) / current_rate) - 1
            else:
                breakeven_sl = -((trade.open_rate * _np("strategy.breakeven_short", 1.002) / current_rate) - 1)
        else:
            breakeven_sl = self.stoploss

        # ═══ CONFIDENCE-ADAPTIVE ATR MULTIPLIER (Phase 25: adaptive) ═══
        ai = self.ai_signal_cache.get(pair, {})
        confidence = ai.get('confidence', 0.0)

        if confidence >= _np("strategy.chandelier_high_conf", 0.80):
            mult = _np("strategy.chandelier_atr_high", 1.5)
        elif confidence >= _np("strategy.chandelier_med_conf", 0.60):
            mult = _np("strategy.chandelier_atr_med", 1.35)
        else:
            mult = _np("strategy.chandelier_atr_low", 1.2)

        # ═══ Phase 27 Task 14: Fractal trailing + cortisol modulation ═══
        # Hurst > 0.55 → trend persistent → widen to 3.5x (ride the trend).
        # Hurst < 0.45 → anti-persistent → tighten to 1.5x (mean-revert fast).
        # Cortisol modulates: α ×= (2 − cortisol) so stressed organism widens
        # stops (less prone to premature exits) and calm organism can tighten.
        # chart_features keys land with `%-` prefix (populate_indicators L780).
        try:
            hurst = (last_candle.get('%-hurst_100')
                     or last_candle.get('%-hurst_50')
                     or last_candle.get('hurst_100')
                     or last_candle.get('hurst_50'))
            if hurst is not None and hurst == hurst:  # NaN check
                hurst_val = float(hurst)
                if hurst_val > 0.55:
                    mult = max(mult, _np("strategy.trailing_hurst_high", 3.5))
                elif hurst_val < 0.45:
                    mult = min(mult, _np("strategy.trailing_hurst_low", 1.5))
        except Exception:
            pass
        try:
            from neural_organism import get_organism
            cortisol = float(get_organism().hormones.cortisol)
            # Sprint 2026-05-05 (B-CONNECT C4): PARAM-driven cortisol-stop
            # coupling that TIGHTENS under panic (was: widened, defeating
            # protective intent — see audit 2026-05-04). cortisol ∈ [0.5,1.0],
            # factor = min + slope * cortisol so panic shrinks dollar exposure
            # symmetric with sizer. Neural BCM+STDP adapts both knobs.
            cs_min = float(_np("strategy.cortisol_stop_min", 0.70))
            cs_slope = float(_np("strategy.cortisol_stop_slope", 0.30))
            cortisol_clamped = max(0.5, min(1.0, cortisol))
            factor = cs_min + cs_slope * cortisol_clamped
            # Hard guard: factor ∈ [0.5, 1.5] regardless of param drift
            factor = max(0.5, min(1.5, factor))
            mult *= factor
            mult = max(0.5, min(5.0, mult))  # safety clamp (lowered floor for tighten)
        except Exception:
            pass

        # ═══ Phase 27 Task 14: 2-week rule — age-based trailing tightening ═══
        try:
            age_days = (current_time - trade.open_date_utc).total_seconds() / 86400.0
            if age_days > 10.0:
                # Past Dobrynskaya's momentum window → tighten trailing
                mult = min(mult, _np("strategy.trailing_age_10d", 1.5))
        except Exception:
            pass

        # ═══ 3-TIER TRAILING (Phase 25: adaptive PnL tiers + ATR caps) ═══
        effective_pnl = current_profit * (trade.leverage or 1.0)

        if effective_pnl >= _np("strategy.trailing_pnl_high", 0.15):
            mult = min(mult, _np("strategy.trailing_atr_high", 1.0))
        elif effective_pnl >= _np("strategy.trailing_pnl_med", 0.08):
            mult = min(mult, _np("strategy.trailing_atr_med", 1.3))
        elif effective_pnl >= _np("strategy.trailing_pnl_low", 0.04):
            mult = min(mult, _np("strategy.trailing_atr_low", 1.6))

        # ═══ CHANDELIER EXIT CALCULATION ═══
        if trade.is_short:
            anchor = last_candle.get('lowest_low_14')
            if not anchor or anchor != anchor:
                anchor = current_rate
            anchor = float(anchor)
            stop_price = anchor + (mult * atr)
            chandelier_result = -((stop_price / current_rate) - 1)
        else:
            anchor = last_candle.get('highest_high_14')
            if not anchor or anchor != anchor:
                anchor = current_rate
            anchor = float(anchor)
            stop_price = anchor - (mult * atr)
            chandelier_result = (stop_price / current_rate) - 1

        # Sanity: result must be negative (a loss)
        if chandelier_result >= 0:
            chandelier_result = self.stoploss

        # Pick the TIGHTER of breakeven vs chandelier (higher value = tighter)
        result = max(chandelier_result, breakeven_sl)

        # Sprint 2026-05-01 night — Stop Hunt Defense applied.
        # When stop_hunt_factor > 1.0 (spread anomaly detected) we WIDEN
        # the stop temporarily — multiplying a negative number by >1 makes
        # it MORE negative (looser stop). Capped at -15% equity floor.
        if stop_hunt_factor > 1.0:
            result = result * stop_hunt_factor

        # Leverage-aware equity cap: max 15% EQUITY loss regardless of leverage
        # 1x → -15% price, 2x → -7.5% price, 3x → -5% price
        MAX_EQUITY_LOSS = 0.15
        leverage = trade.leverage or 1.0
        leverage_aware_floor = -(MAX_EQUITY_LOSS / leverage)

        # 2026-05-18 FAZ B#2 — SHORT bleed guard.
        # Chandelier's lowest_low_14 anchor never locks a stop above price once
        # a SHORT goes against us (price rising): stop_price stays far below
        # current_rate, chandelier_result flips positive → falls back to the
        # -15% floor, so losing SHORTs bleed all the way down. Forensic
        # 2026-05-18: 25 trailing-stop exits, 0 winners, -335 USDT. For a SHORT
        # that is >1h old and >2% in the red (and NOT inside a stop-hunt spread
        # anomaly, where we deliberately loosen), clamp the stop to a -4% equity
        # loss so the trade is cut instead of bleeding to -15%.
        if trade.is_short and effective_pnl < -0.02 and stop_hunt_factor <= 1.0:
            try:
                age_h = (current_time - trade.open_date_utc).total_seconds() / 3600.0
            except Exception:
                age_h = 99.0
            if age_h > 1.0:
                result = max(result, -(0.04 / leverage))

        return max(result, leverage_aware_floor)

    def _sync_portfolio_to_ai(self):
        """Bridge: Sync real exchange balance → AI modules (RiskBudget, Autonomy)."""
        try:
            stake = self.config.get('stake_currency', 'USDT')
            total = self.wallets.get_total(stake)
            free = self.wallets.get_free(stake)

            if total <= 0:
                return

            # Update RiskBudget with real portfolio value + notional-in-flight
            # so the constitution's drawdown and heat guards (HydraSizer:1608)
            # see live numbers instead of getattr()-default-0.
            in_trades_now = max(0.0, total - free)
            self.risk_budget.update_portfolio_value(total, in_trades_usd=in_trades_now)

            # Task 24: pick up the scheduler's latest weekly_adjust without
            # requiring a strategy restart. Runs on the same cadence as
            # portfolio sync (every trade close) — cheap SQLite SELECT.
            try:
                self.risk_budget.reload_multiplier_from_db()
            except Exception:
                pass

            # Persist to SQLite so scheduler/API can read it
            import json
            all_balances = {}
            total_portfolio_usd = total  # Start with stake currency

            for currency, wallet in self.wallets._wallets.items():
                if wallet.total > 0:
                    amount = round(wallet.total, 8)
                    if currency == stake:
                        all_balances[currency] = {"amount": amount, "usd": round(amount, 2)}
                    else:
                        usd = 0.0
                        try:
                            tpair = f"{currency}/{stake}"
                            ticker = self.dp.ticker(tpair) if self.dp else {}
                            price = ticker.get('last', 0) or 0
                            usd = round(amount * price, 2)
                            total_portfolio_usd += usd
                        except Exception:
                            pass
                        all_balances[currency] = {"amount": amount, "usd": usd}

            conn = self._get_sqlite_connection()
            if conn:
                try:
                    conn.execute('''
                        CREATE TABLE IF NOT EXISTS portfolio_state (
                            id INTEGER PRIMARY KEY CHECK (id = 1),
                            stake_currency TEXT, total_balance REAL,
                            free_balance REAL, in_trades REAL,
                            assets_json TEXT, updated_at TEXT
                        )
                    ''')
                    in_trades = total - free
                    conn.execute('''
                        INSERT OR REPLACE INTO portfolio_state
                        (id, stake_currency, total_balance, free_balance, in_trades, assets_json, updated_at)
                        VALUES (1, ?, ?, ?, ?, ?, ?)
                    ''', (stake, total, free, in_trades, json.dumps(all_balances),
                          datetime.utcnow().isoformat()))
                    conn.commit()
                finally:
                    conn.close()

            self._last_portfolio_sync = datetime.utcnow()
            logger.debug(f"[Portfolio Sync] {stake}: stake=${total:.2f} total_usd=${total_portfolio_usd:.2f} assets={len(all_balances)}")
        except Exception as e:
            logger.debug(f"[Portfolio Sync] Skipped: {e}")

    def _emit_dca_gate(self, pair: str, reason: str, detail: str,
                        extra: Optional[Dict[str, Any]] = None) -> None:
        """Throttled emitter for `[DCA-GATE]` block events.

        Emits at most one WARNING per (pair, reason) in any 60-second window
        and a single `[DCA-GATE:ROLLUP]` summary every 5 minutes so we retain
        the audit trail (count, last sample) without flooding journalctl.
        """
        import time
        key = (pair, reason)
        now = time.time()
        last = self._dca_gate_last_log.get(key, 0.0)
        bucket = self._dca_gate_counter.setdefault(
            key, {"count": 0, "ts": now, "sample": dict(extra) if extra else {}}
        )
        bucket["count"] += 1
        if extra:
            bucket["sample"] = dict(extra)

        if now - last >= 60.0:
            logger.warning(f"[DCA-GATE] {pair} BLOCKED ({reason}) — {detail}")
            self._dca_gate_last_log[key] = now

        if now - bucket["ts"] >= 300.0:
            logger.info(
                f"[DCA-GATE:ROLLUP] {pair} {reason}: {bucket['count']} blocks in "
                f"{(now - bucket['ts']) / 60:.1f}m (sample={bucket['sample']})"
            )
            self._dca_gate_counter[key] = {"count": 0, "ts": now, "sample": {}}

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float, max_stake: float,
                            leverage: float, entry_tag: str, side: str,
                            **kwargs) -> float:
        """
        CORE PRINCIPLE: TRADE-FIRST AUTONOMY (Sizing not blocking).
        Instead of blocking a trade, we scale the size based on FreqAI confidence/market regime.
        """
        logger.info(
            f"[TRADE-ATTEMPT] custom_stake_amount CALLED: {pair} side={side} "
            f"proposed={proposed_stake:.4f} min={min_stake:.4f} max={max_stake:.4f}"
        )
        # Sync real exchange balance to AI modules (every trade entry)
        self._sync_portfolio_to_ai()

        # ═══ Phase 27: Full-portfolio base stake — Kelly decides the fraction ═══
        # Freqtrade's config stake_amount is a fixed dollar figure. Hand Kelly
        # the FULL wallet; Kelly already answers "what fraction to risk?" and
        # Constitution caps the ceiling. No hardcoded % band in between.
        try:
            portfolio_val = float(self.wallets.get_total_stake_amount())
            if portfolio_val > 0:
                old_proposed = proposed_stake
                proposed_stake = min(portfolio_val, max_stake)
                logger.info(
                    f"[Phase27:DynamicBase] {pair} portfolio=${portfolio_val:.2f} "
                    f"base=${proposed_stake:.2f} (was ${old_proposed:.2f}) — Kelly decides fraction"
                )
        except Exception as e:
            logger.debug(f"[Phase27:DynamicBase] override skipped: {e}")

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # Sentiment/F&G already reflected in AI confidence — no separate multiplier (Phase 21: removed double-counting)
        final_stake = proposed_stake
        
        if self.dp.runmode.value in ('dry_run', 'live'):
            # Modulate stake heavily based on RAG Brain's LLM Confidence (Phase 3.5.2 logic)
            ai_decision = self._get_ai_signal(pair, current_time)
            confidence = ai_decision.get('confidence', 0.5)
            
            # Phase 6.3: Calibrate confidence using historical accuracy
            try:
                if not hasattr(self, '_calibrator'):
                    self._calibrator = ConfidenceCalibrator(db_path=self.db_path)
                confidence = self._calibrator.adjust_confidence(confidence)
            except Exception as e:
                logger.debug(f"Confidence calibration skipped: {e}")
            
            # Görev 1 Fix: Use PositionSizer to calculate fraction, which respects BayesianKelly and Autonomy logic.
            # Phase 27 Task 1: pair is REQUIRED so per-pair Kelly (not the old global Kelly) drives sizing.
            regime_for_kelly = ai_decision.get("regime") or "_global"
            fraction = self._position_sizer.calculate_stake_fraction(
                confidence, pair=pair, regime=regime_for_kelly, side=side
            )

            # Let it scale down to "dust" sizes if confidence is terribly low
            final_stake = final_stake * fraction

            # Phase 28 Fix: Apply Triple Perception sizing_multiplier
            # This is computed by OOD + CQR + DeepEnsemble + Chronos uncertainty
            # and was previously calculated but never applied to actual trade size
            sizing_mult = ai_decision.get("sizing_multiplier", 1.0)
            if hasattr(self, '_perception_cache') and pair in self._perception_cache:
                sizing_mult = self._perception_cache[pair].get("sizing_multiplier", sizing_mult)

            # ═══ Phase 27 Task 11: CAAT Asymmetric Alpha multiplier (compute only) ═══
            caat_mult = 1.0
            caat_breakdown = {}
            try:
                caat_mult, caat_breakdown = self._caat_asymmetric_multiplier(
                    pair=pair,
                    regime=regime_for_kelly,
                    confidence=confidence,
                    ai_decision=ai_decision,
                    last_candle=last_candle,
                    proposed_stake=proposed_stake,
                )
                # Per-pair confidence threshold gate — Task 11 forgone alpha feedback.
                pair_thr = self._pair_confidence_threshold(pair, regime_for_kelly)
                if confidence < pair_thr:
                    logger.info(
                        f"[Phase27:Threshold] {pair} conf={confidence:.2f} < "
                        f"per-pair thr={pair_thr:.2f} → returning min_stake"
                    )
                    return min_stake
            except Exception as e:
                logger.debug(f"[Phase27:CAAT] multiplier skipped: {e}")

            # ═══ Phase 27 Fix: UNIFIED SIZING (weighted average, not multiplicative) ═══
            # Death-spiral fix: $50 × 0.64 × 0.50 × 0.30 × 1.14 = $0.54 → shadow → zero trades.
            # Kelly (confidence^1.5 already applied inside PositionSizer) stays as base;
            # condition multipliers (CAAT, DualAxis, Cerebellum, Lifecycle) are blended
            # via weighted average so one weak signal cannot annihilate the stake.
            cerebellum_mult = 1.0
            try:
                from cerebellum_timing import get_cerebellum
                # Sprint 2026-05-05 (B-CONNECT C5): pair-aware timing
                # multiplier — falls back to global when per-pair sample
                # count < MIN_TRADES_PER_SLOT (5).
                cerebellum_mult = float(
                    get_cerebellum().get_timing_multiplier(pair=pair)
                )
            except Exception as e:
                logger.debug(f"[Sprint2:Cerebellum] timing skipped: {e}")

            lifecycle_mult = 1.0
            lifecycle_danger = "NORMAL"
            defensive_cap = 1.0
            try:
                from pheromone_field import get_pheromone_field
                pfield_for_sizing = get_pheromone_field()
                lc_state = pfield_for_sizing.read("lifecycle_state")
                if lc_state and isinstance(lc_state, dict):
                    lifecycle_mult = float(lc_state.get("sizing_mult", 1.0))
                    lifecycle_danger = lc_state.get("danger_response", "NORMAL")
                # ProactiveDispatcher (interoception → efferent) deposits
                # defensive_mode pheromones when win_rate_7d or organism_health
                # crosses the danger threshold. Pull the sizing cap so the
                # organism's alarm actually throttles size instead of just
                # logging "Enter defensive mode, reduce sizing".
                dm_state = pfield_for_sizing.read("defensive_mode")
                if dm_state and isinstance(dm_state, dict):
                    cap = float(dm_state.get("sizing_cap", 1.0))
                    defensive_cap = max(0.25, min(1.0, cap))
            except Exception as e:
                logger.debug(f"[Sprint2:Lifecycle] read skipped: {e}")

            weights = {"caat": 0.45, "dual_axis": 0.30, "cerebellum": 0.10, "lifecycle": 0.15}
            parts = {
                "caat": float(caat_mult),
                "dual_axis": float(sizing_mult),
                "cerebellum": float(cerebellum_mult),
                "lifecycle": float(lifecycle_mult),
            }
            unified_mult = sum(parts[k] * weights[k] for k in weights)
            unified_mult = max(0.20, min(unified_mult, 1.50))
            # Hard cap from the interoception→dispatcher defensive pheromone.
            # This applies AFTER weighted blend so a "win_rate_7d too low"
            # alert genuinely shrinks stake regardless of Kelly optimism.
            unified_mult = min(unified_mult, defensive_cap * 1.50)

            # B4 (2026-04-25): emergency degraded mode. When llm_router
            # publishes fleet_exhausted (available_ratio<0.10) the MADAM
            # debate is being skipped (B3) and only the technical fallback
            # is signing trades. Cap sizing at 30% so we keep flow without
            # betting full stake on a degraded brain. Decays naturally as
            # llm_router pushes a healthier ratio.
            # AUDIT-7: apply pheromone _decay so the cap lifts as the
            # brownout signal weakens.
            # FIX-A4 (2026-04-25): set _b4_active flag so the dead-intel
            # block's final clamp re-applies the cap (audit found T22/T18
            # boosts could push unified_mult back above 0.30 cap).
            _b4_active = False
            _b4_emergency_cap = 0.30
            try:
                from pheromone_field import get_pheromone_field as _gpf_b4
                fleet = _gpf_b4().read("fleet_exhausted", source="llm_router")
                if isinstance(fleet, dict):
                    raw_ratio = float(fleet.get("ratio", 0.0))
                    decay = float(fleet.get("_decay", 1.0))
                    effective_ratio = raw_ratio * decay + (1.0 - decay) * 1.0
                    if effective_ratio < 0.10:
                        if unified_mult > _b4_emergency_cap:
                            logger.warning(
                                f"[B4:EmergencyDegraded] fleet_exhausted "
                                f"raw={raw_ratio:.2%} decay={decay:.2f} "
                                f"effective={effective_ratio:.2%} → unified_mult "
                                f"{unified_mult:.3f}→{_b4_emergency_cap:.3f}"
                            )
                            unified_mult = _b4_emergency_cap
                        _b4_active = True
            except Exception:
                pass

            # ═════════════════════════════════════════════════════════════
            # DEAD-INTEL REVIVAL — unified_mult enrichment (sprint 2026-04-25)
            # ─────────────────────────────────────────────────────────────
            # 8 previously-dead signals now modulate sizing. Each is
            # (a) clamped to a sane band, (b) logged when active, (c) safe
            # under exception (defaults to identity multiplier).
            # ═════════════════════════════════════════════════════════════
            dead_intel_breakdown: dict[str, float] = {}
            try:
                from pheromone_field import get_pheromone_field as _gpf_di
                _pf_di = _gpf_di()

                # T5: cortisol → sizer (HORMONE_STATE pheromone, 1.0 calm → 0.5 panic)
                try:
                    hs = _pf_di.read("HORMONE_STATE", source="neural_organism")
                    if isinstance(hs, dict):
                        cortisol = float(hs.get("cortisol", 1.0))
                        # 1.0 calm → multiplier 1.0; 0.5 panic → 0.7 (hard floor)
                        cort_mult = max(0.4, 1.0 - 0.6 * (1.0 - cortisol))
                        if cort_mult < 0.95:
                            unified_mult *= cort_mult
                            dead_intel_breakdown["cortisol"] = round(cort_mult, 3)
                except Exception:
                    pass

                # T22 + T4: agent consensus boost / dissent haircut (pheromone)
                try:
                    cns = _pf_di.read("agent_consensus", source="agent_pool")
                    if isinstance(cns, dict):
                        strength = float(cns.get("signal_strength", cns.get("confidence", 0.0)))
                        if strength > 0.70:
                            unified_mult *= 1.05
                            dead_intel_breakdown["consensus_boost"] = 1.05
                except Exception:
                    pass
                try:
                    dis = _pf_di.read("agent_dissent", source="agent_pool")
                    if isinstance(dis, dict):
                        bull_str = float(dis.get("bull_strength", 0.0))
                        bear_str = float(dis.get("bear_strength", 0.0))
                        if bull_str > 0.30 and bear_str > 0.30:
                            unified_mult *= 0.70
                            dead_intel_breakdown["dissent_haircut"] = 0.70
                except Exception:
                    pass

                # T18: cerebellum_timing pheromone (separate from existing
                # get_timing_multiplier() — the pheromone exposes per-hour
                # confidence too, not just the raw multiplier).
                try:
                    ct = _pf_di.read("cerebellum_timing", source="cerebellum")
                    if isinstance(ct, dict):
                        # current_multiplier is the canonical key; fall back
                        # to mean of best/worst if absent.
                        ct_mult = float(ct.get("current_multiplier", 1.0))
                        # Only apply if it diverges meaningfully from the in-process call
                        if abs(ct_mult - 1.0) > 0.05 and abs(ct_mult - cerebellum_mult) > 0.10:
                            blend = (ct_mult + cerebellum_mult) / 2.0
                            unified_mult *= max(0.5, min(1.4, blend / max(cerebellum_mult, 0.5)))
                            dead_intel_breakdown["cerebellum_pheromone"] = round(blend, 3)
                except Exception:
                    pass
            except Exception as _di_e:
                logger.debug(f"[DeadIntel] pheromone enrichment skipped: {_di_e}")

            # T6: safety_mod (Proprioception) — direct organism call
            try:
                from neural_organism import get_organism as _go_t6
                _org_t6 = _go_t6()
                self_state = _org_t6.proprioception.assess(
                    _org_t6._neurons,
                    consec_wins=getattr(_org_t6, "_consec_wins", 0),
                    consec_losses=getattr(_org_t6, "_consec_losses", 0),
                )
                safety_mod = float(self_state.get("safety_mod", 1.0))
                if safety_mod > 1.05:
                    # Phase=learning/overconfident → reduce sizing
                    sm_mult = 1.0 / safety_mod
                    unified_mult *= max(0.5, sm_mult)
                    dead_intel_breakdown["safety_mod"] = round(sm_mult, 3)
            except Exception:
                pass

            # T1+T13+T14: RL Trinity feed (DT + SAC inference → trinity wakeup)
            # FIX-A7 (2026-04-25): trinity.fuse() requires ml_prediction
            # field to be fresh (alignment check). Audit found we never
            # called update_ml_prediction → fuse() always returned
            # fused=False. Now we feed both ML (from ai_decision) and RL
            # (from SAC) before calling fuse.
            try:
                from dt_inference import get_dt_inference
                from sac_inference import get_sac_inference
                from trinity_fusion import get_trinity
                trinity = get_trinity()
                rl_state = {
                    "confidence": float(confidence),
                    "signal": signal_type,
                    "regime_id": int(hash(regime_for_kelly) % 6),
                    "fng": float(ai_decision.get("fng", 50.0)),
                    "drawdown_pct": float(getattr(self.risk_budget, '_current_drawdown_pct', 0)),
                    "balance_vs_peak": 1.0,
                    "organism_health": float(ai_decision.get("organism_health", 0.5)),
                    "hour_of_day": current_time.hour if current_time else 12,
                    "uncertainty": float(ai_decision.get("uncertainty", 0.5)),
                    "ood_score": float(ai_decision.get("ood_score", 0.0)),
                }
                # FIX-A7: feed ML field from RAG/agent_pool decision.
                trinity.update_ml_prediction({
                    "signal": signal_type,
                    "confidence": float(confidence),
                    "sizing_multiplier": float(ai_decision.get("sizing_multiplier", 1.0)),
                })
                dt = get_dt_inference()
                sac = get_sac_inference()
                # SAC prediction → trinity RL slot. DT predict currently
                # returns neutral until proper action-head lands (FIX-C3).
                sac_action, sac_q = sac.predict(rl_state)
                # Always feed RL field if model is loaded — even neutral
                # action lets trinity report fused=True for telemetry.
                if sac.has_model() or abs(sac_q) > 1e-6:
                    trinity.update_rl_action(sac_action, q_value=float(sac_q))
                fusion_result = trinity.fuse(pair=pair, regime=regime_for_kelly)
                if fusion_result.get("fused", False) and sac.has_model():
                    sac_size_dim = float(sac_action[0]) if len(sac_action) > 0 else 0.0
                    rl_mult = 1.0 + 0.3 * sac_size_dim
                    rl_mult = max(0.7, min(1.3, rl_mult))
                    if abs(rl_mult - 1.0) > 0.02:
                        unified_mult *= rl_mult
                        dead_intel_breakdown["rl_trinity"] = round(rl_mult, 3)
            except Exception as _rl_e:
                logger.debug(f"[DeadIntel:RL] trinity feed skipped: {_rl_e}")

            # Final clamp after dead-intel enrichments.
            # FIX-A4: re-apply B4 emergency cap so dead-intel boosts
            # (T22 consensus +5%, T18 cerebellum, T1 RL ×1.3) cannot
            # silently push unified_mult above the brownout safety cap.
            unified_mult = max(0.20, min(1.50, unified_mult))
            if _b4_active and unified_mult > _b4_emergency_cap:
                logger.warning(
                    f"[B4:Reapply] dead-intel boost defeated emergency cap — "
                    f"clamping {unified_mult:.3f}→{_b4_emergency_cap:.3f}"
                )
                unified_mult = _b4_emergency_cap

            old_way = final_stake * caat_mult * sizing_mult * cerebellum_mult * lifecycle_mult
            pre_unified = final_stake
            final_stake = final_stake * unified_mult
            logger.info(
                f"[Phase27:UnifiedSizing] {pair} kelly=${pre_unified:.2f} × "
                f"unified={unified_mult:.3f} = ${final_stake:.2f} "
                f"(OLD multiplicative would be ${old_way:.2f}) "
                f"parts={{caat:{caat_mult:.2f}, dual:{sizing_mult:.2f}, "
                f"cereb:{cerebellum_mult:.2f}, life:{lifecycle_mult:.2f}, conf:{confidence:.2f}}} "
                f"caat_breakdown={caat_breakdown} danger={lifecycle_danger} "
                f"dead_intel={dead_intel_breakdown}"
            )
            
            # ═══ SPRINT 2: Constitution Check ═══
            # Sprint 2026-05-01: the prior outer try/except silently swallowed
            # ImportError + check_trade exceptions, leaving the Apr 17 $5,481
            # HYPE stake unaudited. The fix is two-fold:
            #   1. Make constitution failures EXPLICIT and route to a
            #      defensive min_stake (NOT the proposed full stake which
            #      Freqtrade's strategy_safe_wrapper would otherwise return
            #      as default_retval on any unhandled exception).
            #   2. Log every failure so the audit trail captures it.
            try:
                from constitution import get_constitution
                enforcer = get_constitution()
                atr_val = last_candle.get('atr', 0) / current_rate if current_rate > 0 else 0.02
                portfolio_val_c = self.risk_budget.portfolio_value
                sizing_pct = (final_stake / portfolio_val_c * 100) if portfolio_val_c > 0 else 1.0
                const_check = enforcer.check_trade({
                    "sizing_pct": sizing_pct,
                    "leverage": float(leverage),
                    "portfolio_drawdown_pct": getattr(self.risk_budget, '_current_drawdown_pct', 0),
                    "portfolio_heat_pct": getattr(self.risk_budget, '_portfolio_heat_pct', 0),
                    "atr_pct": atr_val * 100,
                    "consecutive_losses": getattr(self, '_consecutive_losses', 0),
                    "market_stress": ai_decision.get("market_stress", 0.3),
                })
                if not const_check["allowed"]:
                    logger.warning(f"[Sprint2:Constitution] {pair} BLOCKED: {const_check['violations']}")
                    return min_stake
            except Exception as _const_e:
                # Defensive close — when the constitution can't be evaluated
                # (import failure, missing config, malformed enforcer), trade
                # at the exchange's minimum to keep the audit trail loud and
                # clear without falling back to the full proposed stake.
                logger.error(
                    f"[Sprint2:Constitution] {pair} EVALUATION_FAILED "
                    f"({type(_const_e).__name__}: {_const_e}) — "
                    f"defensive close at min_stake"
                )
                return min_stake

            # ═══ SPRINT 2: Order Flow Sizing ═══
            try:
                from order_flow import get_order_flow
                of = get_order_flow()
                # Phase 27 Item 1: pump live Bybit orderbook into the LOB
                # encoder pipeline. lob_encoder.encode() and order_flow.analyze()
                # both expect Freqtrade's `{"bids": [[p, s], ...], "asks": ...}`
                # shape, which dp.orderbook(pair, depth) already returns —
                # zero conversion needed.
                try:
                    ob_snapshot = self.dp.orderbook(pair, 20)
                    of.publish_orderbook(pair, ob_snapshot)
                    # Afferent: empty book → deposit stress pheromone.
                    try:
                        from sensor_bridges import probe_orderbook
                        probe_orderbook(pair, ob_snapshot, call_site="populate_entry")
                    except Exception:
                        pass
                except Exception:
                    pass  # orderbook unavailable → LOB modality skips gracefully
                signal_type = "BULLISH" if side == "long" else "BEARISH"
                veto, veto_reason = of.should_veto_trade(pair, signal_type)
                if veto:
                    logger.warning(f"[Sprint2:OrderFlow] {pair} VETO: {veto_reason}")
                    return min_stake
                of_sizing = of.get_sizing_adjustment(pair)
                if of_sizing < 1.0:
                    final_stake *= of_sizing
                    logger.info(f"[Sprint2:OrderFlow] {pair} sizing: ×{of_sizing:.2f}")
            except Exception as e:
                logger.debug(f"[Sprint2:OrderFlow] Check skipped: {e}")

            # Cerebellum timing + Lifecycle sizing are now folded into the
            # Phase 27 UnifiedSizing weighted average above — applying them
            # again here would re-create the multiplicative death spiral.

            # ═══ SPRINT 2: Self-Model Competence ═══
            try:
                from self_model import get_self_model
                sm = get_self_model()
                regime = ai_decision.get("regime", "transitional")
                should_trade, sm_reason, conf_mod = sm.should_i_trade(
                    pair, regime, current_time.hour
                )
                if not should_trade:
                    logger.info(f"[Sprint2:SelfModel] {pair} low competence: {sm_reason}")
                    return min_stake
                if conf_mod < 1.0:
                    final_stake *= conf_mod
                    logger.info(f"[Sprint2:SelfModel] {pair} competence mod: ×{conf_mod:.2f}")
            except Exception as e:
                logger.debug(f"[Sprint2:SelfModel] Check skipped: {e}")

            # Phase 3.5.3: Risk Budget scaling — shrink if budget running low
            final_stake = self.risk_budget.scale_position(final_stake)

            # Autonomy max_stake cap (scales with real portfolio)
            portfolio_val = self.risk_budget.portfolio_value
            autonomy_cap = self.autonomy_manager.get_max_stake(portfolio_value=portfolio_val)
            if autonomy_cap is not None:
                final_stake = min(final_stake, autonomy_cap)

            # Phase 22: Funding rate check — extreme funding = cap position (not multiply)
            try:
                funding = self.dp.funding_rate(pair)
                if funding and isinstance(funding, dict):
                    fr = funding.get('fundingRate', 0)
                    if fr and abs(fr) > 0.0005:  # >0.05% funding = extreme
                        funding_cap = autonomy_cap * 0.5 if autonomy_cap else final_stake * 0.5
                        if final_stake > funding_cap:
                            logger.info(f"[FundingRate] {pair} extreme funding {fr:.4%}, capping stake ${final_stake:.2f}→${funding_cap:.2f}")
                            final_stake = funding_cap
            except Exception:
                pass

            # Phase 20: Opportunity score boost — reuse singleton scanner
            try:
                if self._opp_scanner is None:
                    from opportunity_scanner import OpportunityScanner
                    self._opp_scanner = OpportunityScanner()
                opp_score = self._opp_scanner.get_cached_score(pair)
                if opp_score and opp_score > 70:
                    final_stake *= 1.15  # 15% boost for high-opportunity pairs
                    logger.info(f"[Phase20:Opportunity] {pair} stake boosted 15% (score={opp_score:.0f})")
            except Exception:
                pass

            # ═══ EQUAL RISK PER TRADE (Van Tharp CPR Formula) ═══
            # Stake = Risk_per_trade / stoploss_distance_pct
            # This prevents BTC ($322 stake, -$42 loss) vs HOOK ($0.60 stake, -$0.03 loss) asymmetry
            # With this cap, every trade risks the SAME dollar amount regardless of pair
            atr_volatility = last_candle.get('atr', 0.02) / current_rate if current_rate > 0 else 0.02
            if atr_volatility > 0:
                portfolio_val = self.risk_budget.portfolio_value
                risk_pct = 0.005  # 0.5% of portfolio per trade (conservative for cold start)
                risk_per_trade = portfolio_val * risk_pct
                # Stoploss distance ≈ 2.5x ATR as fraction of price
                # Leverage-aware: assume max possible leverage for worst-case sizing
                _max_lev = float(self.leverage_max.value)
                stoploss_distance = min(2.5 * atr_volatility, 0.15 / max(_max_lev, 1.0))
                stoploss_distance = max(stoploss_distance, 0.01)     # floor at 1%
                max_risk_stake = risk_per_trade / stoploss_distance
                if final_stake > max_risk_stake:
                    logger.info(f"[EqualRisk] {pair} stake capped: ${final_stake:.2f} → ${max_risk_stake:.2f} "
                               f"(risk=${risk_per_trade:.0f}, SL={stoploss_distance:.1%})")
                    final_stake = max_risk_stake

            # Budget consumption moved below — only real entries consume.
            # Previous placement here counted stakes that the shadow gate or
            # the min-stake guard immediately rejected, inflating `consumed`
            # to >180% of initial_budget by day 3 (observed 22-24 Apr 2026).
            # Cache the trade's volatility so the post-gate call can reuse it.
            _rb_atr_vol = atr_volatility

        # Phase 21: Removed Graduated Kelly — PositionSizer already applies confidence^1.5 curve.
        # Double confidence reduction was destroying stake sizes ($50→$0.02).

        # ═══ Phase 27 EMERGENCY FIX B — Shadow trade gate ═══
        # CAAT correctly shrinks dust trades to <$1, but exchange min_stake then
        # promotes them back into real $5+ orders that just bleed fees. Audit
        # showed 248 such micro-trades earning $2.34 net (commission absorbed
        # the alpha). The fix: when CAAT-multiplied stake is genuinely tiny,
        # log a SHADOW trade (forgone_pnl_engine) and SKIP the real entry.
        # Returns 0 → Freqtrade interprets as "no entry" → fees zero, learning
        # signal still captured because forgone_resolver re-prices it later.
        SHADOW_THRESHOLD_USD = 1.0
        if final_stake < SHADOW_THRESHOLD_USD:
            try:
                ai_dec = ai_decision if 'ai_decision' in locals() else {}
                # Mega Sprint 2026-04-23 (B.1.1) + Tur-2 (H12): canonical label
                # mapping lives in `_canonical_signal` so shadow writes and the
                # unit test share the same source of truth.
                canonical_signal = _canonical_signal(ai_dec.get("signal"))
                self.forgone_engine.log_forgone_signal(
                    pair=pair,
                    signal_type=canonical_signal,
                    confidence=float(ai_dec.get("confidence", 0.0) or 0.0),
                    entry_price=float(current_rate),
                    was_executed=False,
                    regime=ai_dec.get("regime"),
                )
                logger.info(
                    f"[Phase27:Shadow] {pair} CAAT stake ${final_stake:.4f} < "
                    f"${SHADOW_THRESHOLD_USD:.2f} → SHADOW (no real entry, "
                    f"fees=$0, learning intact)"
                )
            except Exception as e:
                logger.debug(f"[Phase27:Shadow] log failed: {e}")
            return 0.0  # Freqtrade: 0 = skip entry

        # ═══ Phase 27: Min-stake forced-ratio guard ═══
        # Bybit BTC perp min ≈ $160. If Kelly/CAAT wanted $2, naively lifting
        # to min_stake is an 80× forced bet — the exchange, not the organism,
        # would be setting position size. We tolerate up to 3× inflation;
        # beyond that, route to SHADOW so the pair sits out until Kelly grows.
        if final_stake < min_stake:
            forced_ratio = min_stake / (final_stake + 1e-8)
            # Mega Sprint 2026-04-23 (B.2): BTC min_stake ≈ $160 was tripping
            # this gate on every BTC signal (1498 SKIPs in 42h). A 6x
            # tolerance lets a healthy Kelly still take the stake while the
            # autonomy/constitution caps keep the absolute size bounded.
            # RE-3 (2026-04-25): stake-lift tolerance is now envelope-driven.
            # L0 → 3x (conservative), L5 → 10x (high autonomy). Under decay
            # the tolerance shrinks so borderline trades route to SHADOW
            # rather than getting force-lifted.
            try:
                from risk_envelope import get_risk_envelope
                tolerance = float(get_risk_envelope().get_stake_lift_tolerance())
            except Exception:
                tolerance = float(_np("sizing.min_stake_lift_tolerance", 6.0))
            if forced_ratio > tolerance:
                try:
                    ai_dec = ai_decision if 'ai_decision' in locals() else {}
                    canonical_signal = _canonical_signal(ai_dec.get("signal"))
                    self.forgone_engine.log_forgone_signal(
                        pair=pair,
                        signal_type=canonical_signal,
                        confidence=float(ai_dec.get("confidence", 0.0) or 0.0),
                        entry_price=float(current_rate),
                        was_executed=False,
                        regime=ai_dec.get("regime"),
                    )
                except Exception as e:
                    logger.debug(f"[Phase27:MinStakeGuard] shadow log failed: {e}")
                logger.info(
                    f"[Phase27:MinStakeGuard] {pair} SKIP — exchange min "
                    f"${min_stake:.2f} vs desired ${final_stake:.2f} "
                    f"(forced {forced_ratio:.1f}x — max tolerated {tolerance:.1f}x) → SHADOW"
                )
                return 0.0
            logger.info(
                f"[Phase27:MinStakeGuard] {pair} lift ${final_stake:.2f} → "
                f"${min_stake:.2f} (forced {forced_ratio:.1f}x ≤ {tolerance:.1f}x tolerance)"
            )
            final_stake = min_stake

        realised_stake = min(final_stake, max_stake)
        # Task 4 patch (2026-04-25): defer consume_budget until
        # confirm_trade_entry. custom_stake_amount fires on every candle
        # while a limit order is still pending (observed 6 identical
        # $2.21 consume logs for the same pair across 3 hours →
        # consumed crossed 215% of initial_budget). We cache the
        # proposed risk here and only commit it when Freqtrade actually
        # confirms the trade. The cache holds the LATEST proposal per
        # pair, so a second custom_stake_amount before the fill
        # overwrites the first — matching Freqtrade's own "last call
        # wins" stake semantics.

        # Sprint 2026-05-01 evening — EARNED TRUST SYSTEM hard cap.
        # cap = TIER × earned_trust × conviction_scalar × hormonal × decay × vol_brake
        # passes the CURRENT signal confidence so high-conviction trades
        # get more cap utilization than mediocre setups. Hard ceiling 30%.
        #
        # Apr 17 disaster scenario (cold-start L0, conf=0.95):
        #   tier=0.05 × trust=1.0 × conv=1.0 × hormonal=1.0 × decay=1.0 × vol=1.0
        #   = 0.05 → $264 cap on $5,287 portfolio (was $5,481 unclamped)
        # 1-month profitable scenario (L1, trust=1.5, conf=0.95):
        #   tier=0.08 × 1.5 × 1.0 × ... = 0.12 → $617
        # 6-month L5 mature (trust=2.0, conf=0.95):
        #   tier=0.20 × 2.0 × 1.0 × ... = 0.40 → clamped to 30% = $1,541
        # Sprint 2026-05-01 night — full alpha chain. Signal type is
        # passed so F&G contrarian bias can boost/penalize directionally.
        try:
            from risk_envelope import get_risk_envelope
            env = get_risk_envelope()
            portfolio_val_envcap = float(
                getattr(self.risk_budget, "portfolio_value", 0.0) or 0.0
            )
            if portfolio_val_envcap <= 0:
                portfolio_val_envcap = float(
                    self.wallets.get_total_stake_amount()
                ) if self.wallets else 0.0
            sig_for_envcap = ai_decision.get("signal", "NEUTRAL")
            if portfolio_val_envcap > 0:
                # Sprint 2026-05-02 — pass FOMO-veto inputs (recent_low,
                # recent_high, current_price, atr) so the alpha chain can
                # detect chasing in BOTH directions (long FOMO above the
                # recent low; short FOMO below the recent high).
                _recent_low = None
                _recent_high = None
                _atr_val = None
                try:
                    _recent_low = float(last_candle.get("lowest_low_20")
                                        or last_candle.get("lowest_low_14") or 0.0) or None
                    _recent_high = float(last_candle.get("highest_high_14") or 0.0) or None
                    _atr_val = float(last_candle.get("atr") or 0.0) or None
                except Exception:
                    pass
                envelope_cap = float(
                    env.max_single_stake(
                        portfolio_val_envcap,
                        confidence=float(confidence),
                        signal_type=sig_for_envcap,
                        pair=pair,
                        recent_low=_recent_low,
                        current_price=float(current_rate),
                        atr=_atr_val,
                        recent_high=_recent_high,
                    )
                )
                if realised_stake > envelope_cap:
                    logger.warning(
                        f"[EnvelopeCap] {pair} stake ${realised_stake:.2f} > "
                        f"alpha-chain cap ${envelope_cap:.2f} "
                        f"({envelope_cap/portfolio_val_envcap*100:.1f}% of portfolio, "
                        f"conf={float(confidence):.2f} sig={sig_for_envcap}). "
                        f"Clamping."
                    )
                    realised_stake = envelope_cap
                # Sprint 2026-05-02: Exchange-Aware Min-Notional Gate.
                # If the resulting stake is below what Bybit will actually
                # accept as a real order, downgrade to SHADOW (return 0
                # so freqtrade aborts the entry). Avoids the $0.05 useless
                # fill anti-pattern observed in TR-DRY production.
                try:
                    from exchange_microstructure_learner import (
                        get_min_notional, passes_min_notional
                    )
                    min_notional = get_min_notional(pair)
                    if not passes_min_notional(pair, realised_stake):
                        logger.info(
                            f"[MinNotional] {pair} stake ${realised_stake:.2f} < "
                            f"exchange min ${min_notional:.2f} — DOWNGRADED to SHADOW "
                            f"(no real order). Bot's conviction insufficient for venue."
                        )
                        return 0.0
                except Exception as _mn_e:
                    logger.debug(f"[MinNotional] {pair} gate skipped: {_mn_e}")
        except Exception as _envcap_e:
            logger.debug(f"[EnvelopeCap] {pair} cap check skipped: {_envcap_e}")

        try:
            if not hasattr(self, "_pending_risk_consume"):
                self._pending_risk_consume: dict[str, tuple] = {}
            self._pending_risk_consume[pair] = (
                float(realised_stake),
                float(locals().get("_rb_atr_vol", 0.0)),
                float(confidence),
            )
        except Exception as _rb_e:
            logger.debug(f"[RiskBudget] pending consume cache failed: {_rb_e}")
        return realised_stake

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: str,
                            side: str, **kwargs) -> bool:
        """Pre-trade validation + AI metadata storage."""
        logger.info(
            f"[TRADE-ATTEMPT] confirm_trade_entry CALLED: {pair} side={side} "
            f"rate={rate:.6f} stake=${amount*rate:.2f}"
        )

        # D3 (2026-04-25): pair_circuit dormant gate at the last gate
        # before order submission — closes the race where a pair flips
        # dormant between populate_entry_trend (where the signal was
        # approved) and order submission (e.g. fill_rate threshold trips
        # mid-cycle). Without this the soft-dormant flip is racy and the
        # same chronic non-filler can keep re-attempting once per cycle.
        try:
            from pair_circuit import get_pair_circuit
            if get_pair_circuit().is_dormant(pair):
                logger.warning(
                    f"[D3:DormantGate] {pair} dormant at confirm — rejecting entry"
                )
                return False
        except Exception:
            pass

        # ═══ PHASE 30 B.13 — Iteration budget for this trade decision ═══
        _phase30_run_id = f"trade_{pair}_{int(current_time.timestamp())}"
        try:
            from iteration_budget import begin as _phase30_ib_begin
            _phase30_ib_begin(_phase30_run_id, parent_max=6, child_max=3, token_budget=50000)
        except Exception:
            pass

        # ═══ PHASE 30 — A.27 Realtime Anomaly Halt (testnet SHORT bias defense) ═══
        try:
            from realtime_anomaly_detector import get_detector
            df_1m, _ = self.dp.get_analyzed_dataframe(pair, "1m") if "1m" in getattr(self, "informative_pairs", lambda: [])() or True else (None, None)
            if df_1m is None or (hasattr(df_1m, "empty") and df_1m.empty):
                df_1m = self.dp.ohlcv(pair, "1m") if hasattr(self.dp, "ohlcv") else None
            if df_1m is not None and len(df_1m) >= 2:
                last_close = float(df_1m["close"].iat[-1])
                last_volume = float(df_1m.get("volume", df_1m["close"]).iat[-1])
                anom = get_detector().check_bar(pair, last_close, last_volume)
                if anom:
                    logger.error(f"[Phase30:AnomalyHalt] BLOCKED entry {pair}: {anom}")
                    return False
            halted, halt_reason = get_detector().is_halted(pair)
            if halted:
                logger.warning(f"[Phase30:AnomalyHalt] {pair} cooldown active: {halt_reason}")
                return False
        except Exception as _aex:
            logger.debug(f"[Phase30:AnomalyHalt] check failed (non-fatal): {_aex}")

        # ═══ PHASE 30 — A.18 + A.26 Custom assert chain + position cap (LINK -1016 onlemi) ═══
        try:
            from assertions import (
                check_kelly_floor, check_kelly_ceiling, check_stop_loss_present,
                check_leverage_bound, check_single_position_cap,
                check_aggregate_exposure_cap, check_min_notional,
            )
            _portfolio_value = 0.0
            try:
                _portfolio_value = float(self.wallets.get_total(self.config.get("stake_currency", "USDT")))
            except Exception:
                _portfolio_value = float(self.wallets.get_starting_balance()) if hasattr(self, "wallets") else 0.0
            _stake_now = float(amount) * float(rate) if rate else 0.0
            _ai_dec = self.ai_signal_cache.get(pair, {}) or {}
            _kelly = float(_ai_dec.get("kelly_fraction", 0.01) or 0.01)
            _stop = float(_ai_dec.get("stop_loss_pct", -0.05) or -0.05)
            _lev = float(_ai_dec.get("leverage", 1.0) or 1.0)

            for _r, _label in (
                (check_kelly_floor(_kelly), "kelly_floor"),
                (check_kelly_ceiling(_kelly), "kelly_ceiling"),
                (check_stop_loss_present(_stop), "stop_loss"),
                (check_leverage_bound(_lev), "leverage"),
            ):
                if not _r.passed and _r.severity == "error":
                    logger.warning(f"[Phase30:Assert:{_label}] BLOCKED {pair}: {_r.reason}")
                    return False

            _cap = check_single_position_cap(_stake_now, _portfolio_value, max_pct=0.025)
            if not _cap.passed:
                logger.error(f"[Phase30:PositionCap] BLOCKED {pair}: {_cap.reason}")
                return False
            try:
                from freqtrade.persistence import Trade as _T
                _open_value = float(sum((t.stake_amount or 0) for t in _T.get_open_trades()))
            except Exception:
                _open_value = 0.0
            _agg = check_aggregate_exposure_cap(_open_value + _stake_now, _portfolio_value, max_pct=0.50)
            if not _agg.passed:
                logger.error(f"[Phase30:AggregateCap] BLOCKED {pair}: {_agg.reason}")
                return False
            _mn = check_min_notional(_stake_now)
            if not _mn.passed:
                logger.warning(f"[Phase30:MinNotional] BLOCKED {pair}: {_mn.reason}")
                return False
        except Exception as _ax:
            logger.debug(f"[Phase30:AssertChain] non-fatal: {_ax}")

        # ═══ PHASE 30 — A.25 Workflow event emission ═══
        try:
            from trade_event_emitter import emit as _evt_emit
            _evt_emit("trade.opened", {
                "pair": pair, "side": side, "rate": float(rate),
                "stake": float(amount) * float(rate) if rate else 0.0,
                "confidence": float(self.ai_signal_cache.get(pair, {}).get("confidence", 0.0)),
                "regime": self.ai_signal_cache.get(pair, {}).get("regime", ""),
            })
        except Exception:
            pass

        # ═══ PHASE 30 C.1 — Strangler Fig: Controllers/Executors shadow chain ═══
        # The legacy 22-callback HydraSizer flow remains canonical. This block
        # runs the new Controllers (Signal -> Risk -> Timing) + OrderExecutor
        # in parallel and records the verdict to telemetry_events. After 30
        # days of observation showing chain matches legacy within tolerance,
        # the override flag (PARAM `controllers.canonical`) can be flipped.
        try:
            from controllers import (
                SignalController, RiskController, TimingController, ControllerContext,
            )
            from executors import OrderExecutor, OrderRequest
            from composite_risk_manager import build_default as _phase30_crm
            from freqtrade.persistence import Trade as _PT  # safe in strategy context
            _phase30_open = []
            try:
                _phase30_open = [
                    {"pair": t.pair, "stake_amount": float(t.stake_amount or 0)}
                    for t in _PT.get_open_trades()
                ]
            except Exception:
                _phase30_open = []
            _phase30_pv = 0.0
            try:
                _phase30_pv = float(self.wallets.get_total(self.config.get("stake_currency", "USDT")))
            except Exception:
                _phase30_pv = 0.0
            _phase30_ai = self.ai_signal_cache.get(pair, {}) or {}
            _phase30_ctx = ControllerContext(
                pair=pair, side=side,
                portfolio_value=_phase30_pv,
                open_positions=_phase30_open,
                market_regime=str(_phase30_ai.get("regime", "")),
                metadata={
                    "proposed_stake": float(amount) * float(rate) if rate else 0.0,
                    "decision_metadata": {
                        "kelly_fraction": float(_phase30_ai.get("kelly_fraction", 0.01) or 0.01),
                        "leverage": float(_phase30_ai.get("leverage", 1.0) or 1.0),
                        "stop_loss_pct": float(_phase30_ai.get("stop_loss_pct", -0.05) or -0.05),
                    },
                },
            )
            _phase30_sig = SignalController().decide(_phase30_ctx)
            _phase30_risk = RiskController(composite_risk_manager=_phase30_crm()).decide(_phase30_ctx)
            _phase30_time = TimingController().decide(_phase30_ctx)
            _phase30_ord = OrderExecutor(dry_run=True).submit(OrderRequest(
                pair=pair, side=side,
                stake_amount=float(amount) * float(rate) if rate else 0.0,
                rate=float(rate or 0),
                metadata={"kelly_fraction": float(_phase30_ai.get("kelly_fraction", 0.01) or 0.01)},
            ))
            _phase30_chain_proceed = _phase30_sig.proceed and _phase30_risk.proceed and _phase30_time.proceed
            _phase30_chain_blocked = []
            if not _phase30_sig.proceed:
                _phase30_chain_blocked.append(f"signal:{_phase30_sig.blocked_by or _phase30_sig.reason}")
            if not _phase30_risk.proceed:
                _phase30_chain_blocked.append(f"risk:{_phase30_risk.blocked_by or _phase30_risk.reason}")
            if not _phase30_time.proceed:
                _phase30_chain_blocked.append(f"timing:{_phase30_time.blocked_by or _phase30_time.reason}")
            try:
                from telemetry import record as _phase30_t
                _phase30_t(
                    kind="controllers.shadow_decision",
                    severity="info",
                    source_module="HydraSizer.confirm_trade_entry",
                    payload={
                        "pair": pair,
                        "legacy_will_proceed": True,
                        "controller_chain_proceed": bool(_phase30_chain_proceed),
                        "blocked_by": _phase30_chain_blocked,
                        "executor_dry_run_ok": bool(_phase30_ord.success),
                        "stake_signal": _phase30_sig.stake_amount,
                        "stake_risk": _phase30_risk.stake_amount,
                        "stake_timing": _phase30_time.stake_amount,
                    },
                )
            except Exception:
                pass
            if not _phase30_chain_proceed:
                logger.info(
                    f"[Phase30:C.1:ShadowDiverge] {pair} controller-chain would BLOCK "
                    f"({','.join(_phase30_chain_blocked) or 'no_reason'}) — legacy still proceeds (observation only)"
                )
        except Exception as _phase30_cx:
            logger.debug(f"[Phase30:C.1] strangler chain skipped: {_phase30_cx}")

        # ═══ POST-QUANTIZATION NOTIONAL CHECK (Hummingbot BudgetChecker pattern) ═══
        # Freqtrade truncates amount in create_order() AFTER this callback.
        # But we can pre-check: if notional is borderline, exchange will reject.
        if rate > 0:
            notional = amount * rate
            try:
                market = self.dp._exchange.markets.get(pair, {})
                min_cost = market.get("limits", {}).get("cost", {}).get("min")
                if min_cost and notional < min_cost * 1.1:  # 10% buffer
                    logger.warning(f"[NotionalCheck] {pair} notional ${notional:.2f} too close to "
                                  f"min ${min_cost} — skipping to avoid rejection")
                    return False
            except Exception:
                pass  # Can't check = proceed
        ai_decision = self.ai_signal_cache.get(pair, {})
        confidence = ai_decision.get('confidence', 0.5)
        signal_type = "BULL" if side == "long" else "BEAR"
        reasoning = ai_decision.get('reasoning', "Technical entry with AI confirmation")

        # Phase 22: Store AI metadata in Trade.custom_data (persists across restarts)
        trade = kwargs.get('trade')
        if trade:
            try:
                trade.set_custom_data("ai_confidence", round(confidence, 4))
                trade.set_custom_data("ai_signal", signal_type)
                trade.set_custom_data("ai_reasoning", reasoning[:500] if reasoning else "")
                # Audit fix 3: capture the regime at entry so confirm_trade_exit
                # can pass the right (pair, regime) pair into BayesianKelly even
                # if ai_signal_cache has flipped by the time the trade closes.
                trade.set_custom_data(
                    "entry_regime",
                    ai_decision.get("regime") or "_global",
                )
                # Snapshot market state at entry for exit comparison
                dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                if dataframe is not None and len(dataframe) > 0:
                    last = dataframe.iloc[-1]
                    trade.set_custom_data("entry_fng", int(last.get('%-fng_index', 50)))
                    trade.set_custom_data("entry_rsi", round(float(last.get('rsi', 50)), 1))
                    trade.set_custom_data("entry_sentiment_24h", round(float(last.get('%-sentiment_24h', 0)), 3))
                # FIX-C1 (2026-04-25): persist which RL motor was selected
                # at entry. confirm_trade_exit reads this back to update
                # the per-motor HRL tracker (T2). Without this write the
                # motor branch was dead code (audit found 0 writers).
                try:
                    from hrl_meta_policy import get_meta_policy as _gmp_motor
                    motor = _gmp_motor().select_motor(
                        regime=ai_decision.get("regime") or "_global"
                    )
                    if motor in ("iql", "sac", "dt"):
                        trade.set_custom_data("rl_motor", motor)
                except Exception:
                    pass
            except Exception as e:
                logger.debug(f"[custom_data] Failed to store: {e}")

        # Update the existing forgone entry (logged in populate_entry_trend) to was_executed=True
        fid = self._forgone_ids.pop(pair, None)
        if fid:
            self.forgone_engine.mark_executed(fid)

        # Task 26: stamp last-entry timestamp so trade_frequency_hint's
        # pair-scoped cooldown in populate_entry_trend sees real entries.
        try:
            import time as _tm
            if not hasattr(self, "_last_entry_ts_per_pair"):
                self._last_entry_ts_per_pair = {}
            self._last_entry_ts_per_pair[pair] = _tm.time()
        except Exception:
            pass

        # Task 4 patch: commit the cached risk budget for THIS trade.
        # custom_stake_amount parks the (stake, vol, conf) tuple; we
        # consume it exactly once here — right when Freqtrade confirms
        # the order is going to the exchange. A trade that was offered
        # but never confirmed (different pair, cancelled, or limit
        # rejected) quietly expires when the next custom_stake_amount
        # overwrites the cache entry OR when the process restarts.
        try:
            pending = getattr(self, "_pending_risk_consume", {}).pop(pair, None)
            if pending:
                stake, vol, conf = pending
                self.risk_budget.consume_budget(stake, vol, conf)
        except Exception as _rb_commit_e:
            logger.debug(
                f"[RiskBudget] confirm consume failed: {_rb_commit_e}"
            )

        # D2 (2026-04-25): seed a 10-min fill verification deadline.
        # bot_loop_start sweeps these and reports fill outcome to
        # PairCircuitBreaker so chronic limit-reject pairs flip dormant.
        # AUDIT-12: key by (pair, side) so hedge-mode does not collide.
        # Audit 2026-05-02 #2 fix: capture the maker/taker decision AT
        # ENTRY TIME so the sweep records it correctly (instead of
        # hardcoding was_maker=True which oscillates the taker-flag
        # self-flip in ExchangeMicrostructureLearner).
        try:
            import time as _tm_d2
            entry_was_maker = True
            try:
                from exchange_microstructure_learner import should_use_maker
                entry_was_maker = bool(should_use_maker(pair))
            except Exception:
                pass
            self._pending_fill_checks[(pair, side)] = (
                _tm_d2.time() + 600.0,
                entry_was_maker,
            )
        except Exception:
            pass

        # Phase 22: Notify via strategy message
        try:
            self.dp.send_msg(
                f"AI Entry: {pair} {signal_type} conf={confidence:.0%} stake=${amount*rate:.2f}"
            )
        except Exception:
            pass

        # ═══ SPRINT 2: Decision Contract ═══
        try:
            from decision_contract import get_decision_contract
            dc = get_decision_contract()
            dc.create_contract(
                pair=pair, signal=signal_type, confidence=confidence,
                sizing_multiplier=ai_decision.get("sizing_multiplier", 1.0),
                regime=ai_decision.get("regime", "transitional"),
                modules_active=list(ai_decision.get("active_modules", ["evidence_engine"])),
                evidence=ai_decision,
                perception=self._perception_cache.get(pair, {}) if hasattr(self, '_perception_cache') else None,
            )
        except Exception as e:
            logger.debug(f"[Sprint2:Contract] Creation skipped: {e}")

        logger.info(f"[Trade Entry] {pair} {signal_type} conf={confidence:.2f} stake=${amount*rate:.2f} — {reasoning}")
        # ═══ PHASE 30 B.13 — Close iteration budget on commit ═══
        try:
            from iteration_budget import end as _phase30_ib_end
            _phase30_ib_end(_phase30_run_id)
        except Exception:
            pass
        return True

    def confirm_trade_exit(self, pair: str, trade: 'Trade', order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        """Resolve forgone trades and update Bayesian Kelly with trade outcome."""
        # Phase 25: Duplicate exit guard — prevent multiple feedback updates for same trade+order
        # confirm_trade_exit can be called multiple times when order fill fails
        _exit_key = f"{trade.id}_{exit_reason}_{rate}"
        if not hasattr(self, '_processed_exits'):
            self._processed_exits = set()
        if _exit_key in self._processed_exits:
            logger.debug(f"[ExitGuard] {pair} duplicate exit skipped: {_exit_key}")
            return True
        self._processed_exits.add(_exit_key)
        # Cleanup: keep only last 100 entries to prevent memory leak
        if len(self._processed_exits) > 100:
            self._processed_exits = set(list(self._processed_exits)[-50:])

        # Forgone P&L resolution
        fid = self._forgone_ids.pop(pair, None)
        if fid:
            self.forgone_engine.resolve_forgone_trade(fid, exit_price=rate)

        logger.info(f"[Trade Exit] {pair} reason={exit_reason}")

        # Phase 3.5.2 + Phase 27 Task 1: Per-pair Bayesian Kelly update on trade exit.
        try:
            pnl_pct = trade.calc_profit_ratio(rate) if hasattr(trade, 'calc_profit_ratio') else 0.0
            won = pnl_pct > 0
            # Data Acceleration audit fix 3: pass regime so real trades update
            # the SAME (pair, regime) row shadows are training. Without this,
            # every real exit fell into regime="_global" and per-pair-per-regime
            # Kelly only ever learned from shadow data.
            cached_exit = self.ai_signal_cache.get(pair, {})
            regime_for_exit = (
                cached_exit.get("regime")
                or trade.get_custom_data("entry_regime")
                or "_global"
            )
            # Sprint 2026-05-05 (B-CONNECT C1): side-aware Beta posterior.
            side_for_kelly = "short" if trade.is_short else "long"
            self._bayesian_kelly.update(
                won=won, pnl_pct=pnl_pct, pair=pair,
                regime=regime_for_exit, side=side_for_kelly,
            )
            wp = self._bayesian_kelly.win_probability(
                pair=pair, regime=regime_for_exit, side=side_for_kelly,
            )
            kf = self._bayesian_kelly.kelly_fraction(
                pair=pair, regime=regime_for_exit, side=side_for_kelly,
            )
            logger.info(
                f"[BayesianKelly:{pair}/{regime_for_exit}/{side_for_kelly}] Updated: "
                f"{'WIN' if won else 'LOSS'} pnl={pnl_pct:.4f} → "
                f"win_p={wp:.3f} kelly_f={kf:.4f}"
            )

            # T2 (2026-04-25): HRL meta-policy feedback. Without this call
            # the per-organ tracker never accumulates statistics and
            # `meta.get_organ_weight("sizing")` returns 1.0 forever — the
            # entire HRL → CAAT chain in custom_stake_amount was a no-op.
            # Now sizing/IQL/SAC organs accumulate per-trade reward and
            # win-rate, and CAAT can actually weight them.
            try:
                # FIX-A2 (2026-04-25): use singleton accessor so stats
                # persist across confirm_trade_exit calls. Audit found
                # `HRLMetaPolicy()` per-call discarded all updates because
                # OrganPerformanceTracker is in-memory only — fresh tracker
                # per instance meant get_organ_weight() always returned 1.0.
                from hrl_meta_policy import get_meta_policy
                meta = get_meta_policy()
                meta.update("sizing", reward=pnl_pct, win=won)
                # Also update the motor that was selected at entry, if known.
                motor = trade.get_custom_data("rl_motor") if hasattr(trade, "get_custom_data") else None
                if motor in ("iql", "sac", "dt"):
                    meta.update(motor, reward=pnl_pct, win=won)
                # FIX-C2: persist tracker state so restart doesn't reset stats.
                try:
                    meta.organ_tracker.persist_to_db()
                except Exception:
                    pass
            except Exception as _hrl_e:
                logger.debug(f"[T2:HRL] update failed: {_hrl_e}")
        except Exception as e:
            logger.warning(f"[BayesianKelly] Update failed: {e}")

        # Hypothetical $100 Portfolio: compound every closed trade (position-size weighted)
        try:
            trade_pnl_pct = (trade.calc_profit_ratio(rate) * 100) if hasattr(trade, 'calc_profit_ratio') else 0.0
            portfolio_value = self.risk_budget.portfolio_value
            stake_fraction = (trade.stake_amount / portfolio_value) if portfolio_value > 0 else 0.01
            portfolio_pnl_pct = trade_pnl_pct * stake_fraction
            self.forgone_engine.record_trade_for_portfolio(pair, portfolio_pnl_pct)
        except Exception as e:
            logger.warning(f"[Portfolio] Update failed: {e}")

        # ═══ SPRINT 2: Post-Trade Court Investigation ═══
        try:
            from post_trade_court import get_court
            court = get_court()
            # Find the ai_decisions row for this trade
            from db import get_db_connection
            conn = get_db_connection()
            try:
                row = conn.execute(
                    "SELECT id FROM ai_decisions WHERE pair = ? ORDER BY timestamp DESC LIMIT 1",
                    (pair,)
                ).fetchone()
                if row:
                    court.investigate_trade(row["id"])
            finally:
                conn.close()
        except Exception as e:
            logger.debug(f"[Sprint2:Court] Investigation skipped: {e}")

        # ═══ EK Sprint 2026-04-23 (EK.2.9): retroactive LinUCB feedback ═══
        # Every LLM call that fired against this pair during the trade's
        # lifetime gets a small posterior nudge: +0.1 on a winning trade,
        # -0.05 on a losing one. The nudge is tiny on purpose — outcome_pnl
        # is noisy, so we only want to bias the bandit, not let it drown
        # out the direct quality/latency reward signal.
        try:
            from ai_config import get_flag
            if get_flag("llm_contextual_bandit_retroactive_reward", True):
                from llm_router import get_router
                from llm_features import extract_features
                from db import get_db_connection

                conn = get_db_connection()
                try:
                    llm_calls = conn.execute(
                        """
                        SELECT provider, model, agent_name, timestamp
                        FROM llm_calls
                        WHERE trading_pair = ?
                          AND datetime(timestamp) >= datetime(?)
                          AND datetime(timestamp) <= datetime(?)
                        """,
                        (pair, trade.open_date_utc.isoformat(), current_time.isoformat()),
                    ).fetchall()
                finally:
                    conn.close()

                router = get_router()
                slots_by_key = {(s.provider, s.model_name): s for s in router.slots}
                nudge = 0.1 if pnl_pct > 0 else (-0.05 if pnl_pct < 0 else 0.0)
                retro = 0
                for call in llm_calls:
                    slot = slots_by_key.get((call["provider"], call["model"]))
                    if slot is None or slot.linucb_n_updates == 0:
                        continue
                    try:
                        ts_hour = datetime.fromisoformat(
                            str(call["timestamp"]).replace(" ", "T")
                        ).hour
                    except Exception:
                        ts_hour = current_time.hour
                    x_approx = extract_features({
                        "task": call["agent_name"] or "default",
                        "prompt_len": 2000,
                        "needs_json": True,
                        "regime_vol": 0.5,
                        "hour_utc": ts_hour,
                    })
                    slot.linucb_update(x_approx, nudge)
                    retro += 1
                if retro:
                    logger.info(
                        f"[LinUCB:Retro] {pair} pnl={pnl_pct:+.2%} "
                        f"nudged {retro} LLM calls by {nudge:+.2f}"
                    )
        except Exception as e:
            logger.debug(f"[LinUCB:Retro] skipped: {e}")

        # ═══ SPRINT 2: Track consecutive losses for constitution ═══
        try:
            if pnl_pct < 0:
                self._consecutive_losses = getattr(self, '_consecutive_losses', 0) + 1
            else:
                self._consecutive_losses = 0
        except Exception:
            pass

        # ══════════════════════════════════════════════════════════════
        # LIVE FEEDBACK LOOP: Update ALL learning modules on trade close
        # This is the CORE self-improvement mechanism.
        # ══════════════════════════════════════════════════════════════
        trade_pnl_pct = (trade.calc_profit_ratio(rate) * 100) if hasattr(trade, 'calc_profit_ratio') else 0.0

        # 1. PatternStatStore — record this trade for future statistical queries
        try:
            from pattern_stat_store import PatternStatStore
            pss = PatternStatStore(db_path=self.db_path)
            # Get cached indicators from trade entry
            ai_meta = {}
            try:
                ai_meta = {
                    'confidence': trade.custom_data.get('ai_confidence', {}).get('value'),
                    'signal': trade.custom_data.get('ai_signal', {}).get('value'),
                    'reasoning': trade.custom_data.get('ai_reasoning', {}).get('value'),
                    'rsi': trade.custom_data.get('entry_rsi', {}).get('value'),
                    'fng': trade.custom_data.get('entry_fng', {}).get('value'),
                }
            except Exception:
                pass

            pss.ingest_trade({
                'pair': pair,
                'strategy': self.name,
                'direction': 'short' if trade.is_short else 'long',
                'entry_date': str(trade.open_date),
                'exit_date': str(current_time),
                'profit_pct': round(trade_pnl_pct, 3),
                'duration_hours': round((current_time - trade.open_date_utc).total_seconds() / 3600, 2) if trade.open_date else None,
                'exit_reason': exit_reason,
                'entry_price': trade.open_rate,
                'rsi_bucket': PatternStatStore.classify_rsi(float(ai_meta['rsi'])) if ai_meta.get('rsi') else None,
                'fng_bucket': PatternStatStore.classify_fng(int(ai_meta['fng'])) if ai_meta.get('fng') else None,
            })
            logger.info(f"[LiveFeedback:PatternStatStore] {pair} trade recorded: {trade_pnl_pct:+.2f}%")
        except Exception as e:
            logger.debug(f"[LiveFeedback:PatternStatStore] {pair} failed: {e}")

        # 2. BidirectionalRAG — generate lesson from this trade
        try:
            from bidirectional_rag import BidirectionalRAG
            bidi = BidirectionalRAG(db_path=self.db_path)
            reasoning = ai_meta.get('reasoning', '') or 'No reasoning available'
            bidi.evaluate_trade_outcome(
                decision_id=0, pair=pair,
                signal='BULLISH' if not trade.is_short else 'BEARISH',
                outcome_pnl=trade_pnl_pct, reasoning=str(reasoning)
            )
            logger.info(f"[LiveFeedback:BidiRAG] {pair} lesson generated")
        except Exception as e:
            logger.debug(f"[LiveFeedback:BidiRAG] {pair} failed: {e}")

        # 3. MAGMA — reinforce causal edges based on outcome
        try:
            from magma_memory import MAGMAMemory
            magma = MAGMAMemory(db_path=self.db_path)
            outcome = "win" if trade_pnl_pct > 0 else "loss"
            magma.add_edge("causal", pair.lower().replace("/", "_"), f"trade_{outcome}",
                           exit_reason, metadata={"pnl": trade_pnl_pct})
            logger.info(f"[LiveFeedback:MAGMA] {pair} causal edge: {outcome} via {exit_reason}")
        except Exception as e:
            logger.debug(f"[LiveFeedback:MAGMA] {pair} failed: {e}")

        # 4. Update ai_decisions outcome (for ConfidenceCalibrator to use in next re-fit)
        # 2026-05-26 leak fix: was direct sqlite3.connect(self.db_path) bypassing
        # the pool. Now goes through get_db_connection() with a with-block so
        # exception paths can't leak the FD.
        try:
            from db import get_db_connection
            with get_db_connection() as conn:
                conn.execute("""
                    UPDATE ai_decisions SET outcome_pnl = ?, outcome_duration = ?
                    WHERE pair = ? AND outcome_pnl IS NULL
                    ORDER BY timestamp DESC LIMIT 1
                """, (trade_pnl_pct,
                      int((current_time - trade.open_date_utc).total_seconds() / 60) if trade.open_date else None,
                      pair))
            logger.info(f"[LiveFeedback:Calibrator] {pair} decision outcome updated: {trade_pnl_pct:+.2f}%")
        except Exception as e:
            logger.debug(f"[LiveFeedback:Calibrator] {pair} outcome update failed: {e}")

        # 5. Phase 20: Agent Pool — update agent track records
        try:
            from agent_pool import AgentPool
            pool = AgentPool(db_path=self.db_path)
            pool.record_trade_outcome(
                pair=pair,
                outcome_pnl=trade_pnl_pct,
                regime=ai_meta.get('regime'),
                signal='BULLISH' if not trade.is_short else 'BEARISH'
            )
            logger.info(f"[LiveFeedback:AgentPool] {pair} agent outcomes updated: {trade_pnl_pct:+.2f}%")
        except Exception as e:
            logger.debug(f"[LiveFeedback:AgentPool] {pair} agent update failed: {e}")

        # 5b. Task 9: Retroactive PnL feedback to the LLM bandit. The
        # router's LinUCB reward pathway accepts outcome_pnl but no live
        # caller was wiring it — confirm_trade_exit is the natural place
        # because this is where the ground-truth PnL becomes knowable.
        # Task 19 fix: SQLite DATETIME DEFAULT CURRENT_TIMESTAMP stores as
        # "YYYY-MM-DD HH:MM:SS" (space separator, no TZ). datetime.isoformat()
        # emits "YYYY-MM-DDTHH:MM:SS+00:00" — lexicographic T (0x54) sorts
        # AFTER space (0x20), so `timestamp >= open_iso` excluded every
        # single llm_call row. strftime-normalised strings compare correctly.
        try:
            from llm_router import get_router
            router = get_router()
            _fmt = "%Y-%m-%d %H:%M:%S"
            open_iso = trade.open_date_utc.strftime(_fmt) if trade.open_date_utc else None
            close_iso = current_time.strftime(_fmt) if current_time else None
            router.record_pnl_feedback(
                pair=pair,
                outcome_pnl_pct=float(trade_pnl_pct),
                open_time=open_iso,
                close_time=close_iso,
            )
        except Exception as e:
            logger.debug(f"[LiveFeedback:LLMRouter] {pair} feedback failed: {e}")

        # 6. Phase 24: Neural Organism — adaptive parameter update + pair ban
        try:
            from neural_organism import get_organism
            organism = get_organism()
            # Gather context — robust extraction with safe defaults
            fng_val = None
            adx_val = 20.0
            funding_val = 0.0
            conf_val = 0.5
            regime_val = "transitional"
            try:
                _fng = ai_meta.get('fng')
                if _fng is not None:
                    fng_val = int(float(_fng))
            except (ValueError, TypeError):
                pass
            try:
                _conf = ai_meta.get('confidence')
                if _conf is not None:
                    conf_val = float(_conf)
            except (ValueError, TypeError):
                pass
            try:
                _regime = ai_meta.get('regime')
                if _regime:
                    regime_val = str(_regime)
                else:
                    # Phase 25: ai_meta'da regime yoksa signal cache'den al
                    cached = self.ai_signal_cache.get(pair, {})
                    _cached_reasoning = str(cached.get('reasoning', ''))
                    if 'trending_bull' in _cached_reasoning:
                        regime_val = 'trending_bull'
                    elif 'trending_bear' in _cached_reasoning:
                        regime_val = 'trending_bear'
                    elif 'ranging' in _cached_reasoning:
                        regime_val = 'ranging'
                    elif 'high_volatility' in _cached_reasoning:
                        regime_val = 'high_volatility'
            except Exception:
                pass
            try:
                balance = self.risk_budget.portfolio_value if hasattr(self, 'risk_budget') else 10000
            except Exception:
                balance = 10000
            peak = max(balance, 1.0)

            result = organism.update_cycle(
                pair=pair, pnl_pct=trade_pnl_pct,
                regime=regime_val,
                confidence=conf_val,
                exit_reason=exit_reason,
                duration_hours=round((current_time - trade.open_date_utc).total_seconds() / 3600, 2) if trade.open_date else 1.0,
                stake_amount=trade.stake_amount if hasattr(trade, 'stake_amount') else 0,
                fng=fng_val,
                adx=adx_val,
                funding_rate=funding_val,
                balance_vs_peak=min(1.0, balance / peak),
            )
            # Apply pair ban if organism recommends it
            ban_minutes = result.get("ban_minutes", 0)
            if ban_minutes > 0:
                ban_until = current_time + timedelta(minutes=ban_minutes)
                self.lock_pair(pair, ban_until,
                               reason=f"NeuralOrganism:loss={trade_pnl_pct:+.1f}%,fear={result.get('fear_tier','normal')}")
                logger.info(f"[NeuralOrganism:PairBan] {pair} locked {ban_minutes:.0f}min "
                           f"(loss={trade_pnl_pct:+.1f}%, fear={result.get('fear_tier')})")
            logger.info(f"[LiveFeedback:NeuralOrganism] {pair} updated: phase={result.get('phase')} "
                       f"overrides={len(result.get('overrides', []))}")
        except Exception as e:
            logger.warning(f"[LiveFeedback:NeuralOrganism] {pair} update FAILED: {e}", exc_info=True)

        # Phase 22: Notify exit via strategy message
        try:
            self.dp.send_msg(
                f"AI Exit: {pair} reason={exit_reason} profit={trade.calc_profit_ratio(rate):.1%}"
            )
        except Exception:
            pass

        return True

    # ══════════════════════════════════════════════════════════════════════
    # Phase 22: ALL NEW STRATEGY CALLBACKS
    # ══════════════════════════════════════════════════════════════════════

    def bot_start(self, **kwargs) -> None:
        """One-time initialization after all configs loaded (Phase 22 #3)."""
        logger.info("[bot_start] AI Trading System initializing...")
        try:
            from semantic_cache import SemanticCache
            self._semantic_cache = SemanticCache(db_path=self.db_path)
            logger.info("[bot_start] Semantic cache ready.")
        except Exception as e:
            logger.warning(f"[bot_start] Semantic cache init failed: {e}")
        # Sprint 2026-05-02: register the live CCXT exchange handle with
        # ExchangeMicrostructureLearner so it can self-discover min_notional
        # and other venue rules. Falls back gracefully if dataprovider not
        # ready yet — first signal cycle will retry.
        try:
            from exchange_microstructure_learner import set_exchange_handle
            ex_handle = getattr(self.dp, "_exchange", None) or getattr(self, "exchange", None)
            if ex_handle is not None:
                set_exchange_handle(ex_handle)
                logger.info("[bot_start] Microstructure learner registered exchange handle.")
        except Exception as e:
            logger.debug(f"[bot_start] Microstructure init skipped: {e}")
        # Ensure protection_logs table exists for testnet data collection
        # 2026-05-26 leak hardening: close() now in finally so a CREATE TABLE
        # parse failure can't leak the pool handle on cold start.
        conn = self._get_sqlite_connection()
        if conn:
            try:
                try:
                    conn.execute('''
                        CREATE TABLE IF NOT EXISTS obs.protection_logs (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            timestamp TEXT NOT NULL,
                            event_type TEXT NOT NULL,
                            pair TEXT,
                            details TEXT,
                            profit_at_event REAL,
                            trade_count INTEGER
                        )
                    ''')
                    conn.commit()
                except Exception:
                    pass
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
        logger.info("[bot_start] AI Trading System ready.")

    @property
    def protections(self):
        """Built-in protections — Sprint 2026-05-01: FULLY DYNAMIC.

        Apr 23 → Apr 27 production observed 17 MaxDrawdown trips with
        values 0.92 / 1.57 / 1.69 / 2.26 / 2.51 vs threshold 0.50.
        Root cause (audit 2026-05-01): legacy `ratios` mode summed
        leverage-weighted close_profit values — Apr 26 BTC liquidation
        alone (-1.0015 ratio) exceeded 50% threshold. The fix is two-fold:

          1. `calculation_mode: "equity"` — true % of equity drawdown
             using close_profit_abs in dollars instead of ratio cumsum.
          2. EVERY parameter sourced from RiskEnvelope (autonomy + decay
             + hormonal + regime). No hardcoded thresholds.

        At cold start (envelope unreachable) the fallbacks are still
        permissive enough to NOT lock the bot — better to trade through
        a transient envelope outage than to block on a missing import.
        """
        try:
            from risk_envelope import get_risk_envelope
            env = get_risk_envelope()
            max_dd = float(env.protection_max_drawdown())
            lookback = int(env.protection_lookback_candles())
            trade_lim = int(env.protection_trade_limit())
            stop_dur = int(env.protection_stop_duration())
        except Exception:
            # Safe permissive defaults on cold-start / import failure —
            # never block trading on an envelope outage. Numbers
            # intentionally generous so the missing-envelope branch
            # doesn't itself create a lock-out scenario.
            max_dd = 0.20
            lookback = 96
            trade_lim = 12
            stop_dur = 4

        # StoplossGuard scales with envelope's current concurrent-trade
        # appetite — more open positions allowed → more stoplosses
        # tolerable per-pair before locking that pair.
        try:
            sl_lookback = max(12, min(96, lookback // 2))
            sl_trade_lim = max(3, min(12, trade_lim // 2))
            sl_stop_dur = max(1, min(24, stop_dur // 2))
        except Exception:
            sl_lookback, sl_trade_lim, sl_stop_dur = 24, 4, 2

        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 1,
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": sl_lookback,
                "trade_limit": sl_trade_lim,
                "stop_duration_candles": sl_stop_dur,
                "only_per_pair": True,
            },
            {
                # Equity-mode: true % of account drawdown, not ratio cumsum.
                # `calculation_mode="equity"` makes Freqtrade use
                # close_profit_abs against starting balance (a real % of
                # equity) instead of summing leveraged close_profit ratios.
                # See max_drawdown_protection.py:58-81 for the dual-mode
                # implementation.
                "method": "MaxDrawdown",
                "calculation_mode": "equity",
                "lookback_period_candles": lookback,
                "trade_limit": trade_lim,
                "stop_duration_candles": stop_dur,
                "max_allowed_drawdown": max_dd,
            },
        ]

    def informative_pairs(self):
        """Multi-timeframe + cross-pair data (Phase 22 #4 + Phase 26 Multi-TF)."""
        stake = self.config.get('stake_currency', 'USDT')
        return [
            (f"BTC/{stake}", "1h"),
            (f"BTC/{stake}", "4h"),
            (f"BTC/{stake}", "1d"),   # Phase 26: daily TF for chart structure Layer 2
            (f"ETH/{stake}", "4h"),
            (f"ETH/{stake}", "1d"),   # Phase 26: daily TF for chart structure Layer 2
        ]

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str,
                 side: str, **kwargs) -> float:
        """Regime-aware + confidence-based dynamic leverage.

        RE-3 (2026-04-25): leverage_max is now sourced from RiskEnvelope
        (autonomy-tier × hormonal × decay) instead of static
        `self.leverage_max.value`. The static parameter remains as a
        FreqTrade hyperopt knob for backtesting; live trading uses the
        envelope. L0 → 2x, L5 → 10x; under panic → halved.
        """
        ai = self.ai_signal_cache.get(pair, {})
        confidence = ai.get('confidence', 0.0)

        # RE-3: dynamic envelope-driven leverage cap
        try:
            from risk_envelope import get_risk_envelope
            envelope_lev_max = float(get_risk_envelope().get_leverage_max())
        except Exception:
            envelope_lev_max = float(self.leverage_max.value)
        # Combine envelope cap with the static hyperopt parameter — the tighter wins.
        effective_lev_ceiling = min(envelope_lev_max, float(self.leverage_max.value) * 3.0)

        # Regime-aware max leverage cap (OMEGA-inspired but safer)
        regime_max = effective_lev_ceiling
        atr_safe_max = effective_lev_ceiling  # ATR safety cap (calculated below)
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe is not None and len(dataframe) > 0:
                last = dataframe.iloc[-1]
                adx = float(last.get('adx', 20))
                ema200 = last.get('ema_200')
                price = float(last.get('close', 0))
                if adx > 25 and ema200 and price > float(ema200):
                    regime_max = effective_lev_ceiling       # trending_bull → full leverage
                elif adx < 20:
                    regime_max = min(2.0, effective_lev_ceiling)  # ranging → cap 2x
                else:
                    regime_max = min(1.5, effective_lev_ceiling)  # bear/volatile → cap 1.5x

                # ATR-based leverage cap: keep max equity loss <= 15%
                # Chandelier Exit multiplier is sourced from PARAM_REGISTRY so
                # tuning stays in one place (Revize Tur-2 H9). leverage *
                # chandelier_distance <= 15%.
                # This lets Chandelier work freely — we limit leverage, not stoploss.
                atr = float(last.get('atr', 0))
                if atr > 0 and price > 0:
                    chandelier_mult = _np("strategy.chandelier_atr_med", 1.35)
                    chandelier_distance = chandelier_mult * (atr / price)
                    atr_safe_max = 0.15 / max(chandelier_distance, 0.005)
                    if atr_safe_max < regime_max:
                        logger.info(f"[SmartLeverage] {pair} ATR cap: {atr_safe_max:.1f}x "
                                   f"(ATR={atr/price:.1%}, chandelier={chandelier_distance:.1%})")
        except Exception:
            regime_max = 1.0  # Error → safe

        # Effective cap = minimum of regime, ATR safety, and exchange max
        effective_max = min(regime_max, atr_safe_max, max_leverage)

        # Confidence-based within effective cap (Phase 25: adaptive)
        if confidence >= _np("strategy.leverage_conf_high", 0.75):
            lev = effective_max * _np("strategy.leverage_mult_high", 1.0)
        elif confidence >= _np("strategy.leverage_conf_med", 0.60):
            lev = effective_max * 0.7
        elif confidence >= _np("strategy.leverage_conf_low", 0.45):
            lev = effective_max * _np("strategy.leverage_mult_low", 0.5)
        else:
            lev = 1.0

        return max(1.0, round(lev, 1))

    def custom_exit(self, pair: str, trade: 'Trade', current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs) -> str | bool | None:
        """AI-driven exit logic (Phase 22 #13)."""
        if self.dp.runmode.value not in ('dry_run', 'live'):
            return None

        # Pair-circuit gate: if the exchange keeps returning empty books
        # (ICP observed 4,693× in 17h), suppress the exit attempt. The
        # scheduler revive_probe job retests every 5 min and closes the
        # circuit once the book recovers. Suppressing ≠ abandoning —
        # Freqtrade keeps the position, we just stop flogging a dead book.
        try:
            from pair_circuit import get_pair_circuit
            if get_pair_circuit().is_dormant(pair):
                return None
        except Exception:
            pass

        hours_held = (current_time - trade.open_date_utc).total_seconds() / 3600
        age_days = hours_held / 24.0

        # Phase 27 Task 14: 2-week rule — beyond 14 days crypto momentum has
        # empirically reverted (Dobrynskaya 2021 SSRN 3913263 timing map).
        # Force exit regardless of profit state to free capital for fresh
        # momentum candidates.
        if age_days > 14.0:
            return f"age_14d_reversal_{age_days:.1f}d"

        # 1. STALE TRADE — Sprint 2026-05-05 (B-CONNECT C2): regime-aware.
        # Trending markets get wider patience (24h default), ranging/choppy
        # get cut faster (4-8h). Read PARAM_REGISTRY per detected regime;
        # neural BCM+STDP adapts these from trade outcomes over time.
        cached_for_stale = self.ai_signal_cache.get(pair, {})
        regime_for_stale = (
            cached_for_stale.get("regime")
            or trade.get_custom_data("entry_regime")
            or "_global"
        )
        _stale_hours_map = {
            "trending_bull": _np("strategy.stale_hours.trending_bull", 24),
            "trending_bear": _np("strategy.stale_hours.trending_bear", 24),
            "ranging":       _np("strategy.stale_hours.ranging", 6),
            "choppy":        _np("strategy.stale_hours.choppy", 8),
            "high_vol":      _np("strategy.stale_hours.high_vol", 4),
        }
        stale_hr = _stale_hours_map.get(
            regime_for_stale, float(self.stale_trade_hours.value)
        )
        stale_pct = float(_np("strategy.stale_flat_pct", 0.005))
        if hours_held > stale_hr and abs(current_profit) < stale_pct:
            return f"stale_{hours_held:.0f}h_flat_{regime_for_stale}"

        # 2. SIGNAL REVERSAL
        cached = self.ai_signal_cache.get(pair, {})
        signal = cached.get('signal', 'NEUTRAL')
        confidence = cached.get('confidence', 0.0)

        _flip_conf = _np("strategy.flip_exit_conf", 0.55)
        if not trade.is_short and signal == 'BEARISH' and confidence >= _flip_conf:
            return f"ai_flip_bearish_{confidence:.0%}"
        if trade.is_short and signal == 'BULLISH' and confidence >= _flip_conf:
            return f"ai_flip_bullish_{confidence:.0%}"

        # 3. CONFIDENCE DEGRADATION — exit if confidence dropped significantly
        entry_conf = trade.get_custom_data("ai_confidence", 0.5)
        if isinstance(entry_conf, (int, float)) and entry_conf > 0.55 and confidence < 0.30:
            return f"confidence_drop_{entry_conf:.0%}_to_{confidence:.0%}"

        # 4. FEAR & GREED CRASH
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe is not None and len(dataframe) > 0:
                fng = dataframe.iloc[-1].get('%-fng_index', 50)
                entry_fng = trade.get_custom_data("entry_fng", 50)
                if isinstance(entry_fng, (int, float)) and isinstance(fng, (int, float)):
                    if fng < self.fg_extreme_threshold.value and entry_fng > 40:
                        return f"extreme_fear_fng_{int(fng)}"
        except Exception:
            pass

        # 5. FIRST-HOUR CRASH — dynamic ATR-based threshold (was fixed 7%)
        # Wen et al. (2022): crypto intraday reversal exists, fixed % is too aggressive
        # for high-ATR coins. 2.5x ATR as % of price = pair-appropriate threshold.
        if hours_held <= 1.0 and current_profit < 0:
            try:
                dataframe_fh, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                if dataframe_fh is not None and len(dataframe_fh) > 0 and 'atr' in dataframe_fh.columns:
                    atr_val = float(dataframe_fh['atr'].iloc[-1])
                    atr_pct = atr_val / current_rate if current_rate > 0 else 0.03
                    dynamic_threshold = -(2.5 * atr_pct)  # 2.5x ATR as loss threshold
                    dynamic_threshold = max(dynamic_threshold, -0.15)  # Never wider than 15%
                    dynamic_threshold = min(dynamic_threshold, -0.03)  # Never tighter than 3%
                    if current_profit <= dynamic_threshold:
                        return f"first_hour_atr_loss_{abs(current_profit):.1%}"
                else:
                    # Fallback: fixed 7% if no ATR data
                    if current_profit <= -0.07:
                        return "first_hour_7pct_loss"
            except Exception:
                if current_profit <= -0.07:
                    return "first_hour_7pct_loss"

        # 6. LOG EVERYTHING for testnet analysis (even when NOT exiting)
        # This data is gold when we switch to real money
        # 2026-05-26 leak fix: close() guaranteed in finally even if
        # INSERT raises sqlite3.OperationalError under lock contention.
        if hours_held > 0 and int(hours_held) % 4 == 0:  # Every 4 hours
            try:
                conn = self._get_sqlite_connection()
                if conn:
                    try:
                        conn.execute(
                            "INSERT INTO protection_logs (timestamp, event_type, pair, details, profit_at_event) "
                            "VALUES (?, ?, ?, ?, ?)",
                            (current_time.isoformat(), "trade_check", pair,
                             f"signal={signal} conf={confidence:.2f} entry_conf={entry_conf} hours={hours_held:.1f}",
                             round(current_profit, 6))
                        )
                        conn.commit()
                    finally:
                        try:
                            conn.close()
                        except Exception:
                            pass
            except Exception:
                pass

        return None

    def custom_roi(self, pair: str, trade: 'Trade', current_time: datetime,
                   trade_duration: int, entry_tag: str | None, side: str,
                   **kwargs) -> float | None:
        """Dynamic ROI based on AI trend confidence (Phase 22 #15)."""
        cached = self.ai_signal_cache.get(pair, {})
        confidence = cached.get('confidence', 0.0)

        _roi_hi_conf = _np("strategy.chandelier_high_conf", 0.80)
        if confidence >= _roi_hi_conf:
            if trade_duration < 120:
                return 0.20
            if trade_duration < 360:
                return 0.08
            return 0.02

        if confidence < 0.40:
            if trade_duration < 60:
                return 0.05
            return 0.01

        return None

    def check_entry_timeout(self, pair: str, trade: 'Trade', order: 'Order',
                            current_time: datetime, **kwargs) -> bool:
        """Cancel entry if AI signal changed (Phase 22 #9)."""
        cached = self.ai_signal_cache.get(pair, {})
        signal = cached.get('signal', 'NEUTRAL')

        if not trade.is_short and signal == 'BEARISH':
            logger.info(f"[Timeout] Cancelling LONG entry for {pair}: AI flipped to BEARISH")
            return True
        if trade.is_short and signal == 'BULLISH':
            logger.info(f"[Timeout] Cancelling SHORT entry for {pair}: AI flipped to BULLISH")
            return True
        return False

    def check_exit_timeout(self, pair: str, trade: 'Trade', order: 'Order',
                           current_time: datetime, **kwargs) -> bool:
        """Cancel stale exit order for retry (Phase 22 #10)."""
        if order.order_date_utc:
            minutes_open = (current_time - order.order_date_utc).total_seconds() / 60
            if minutes_open > 5:
                logger.info(f"[Timeout] Exit order for {pair} open {minutes_open:.0f}m, cancelling for retry")
                return True
        return False

    def order_filled(self, pair: str, trade: 'Trade', order: 'Order',
                     current_time: datetime, **kwargs) -> None:
        """Called immediately after ANY order fills (Phase 22 #12)."""
        fill_side = "ENTRY" if order.ft_order_side == trade.entry_side else "EXIT"
        logger.info(f"[OrderFilled] {pair} {fill_side} @ {order.safe_price:.6f}")
        try:
            if fill_side == "ENTRY":
                trade.set_custom_data("fill_price", round(float(order.safe_price), 6))
                trade.set_custom_data("fill_time", current_time.isoformat())
        except Exception:
            pass

    def custom_entry_price(self, pair: str, trade: 'Trade | None', current_time: datetime,
                           proposed_rate: float, entry_tag: str | None, side: str,
                           **kwargs) -> float:
        """Orderbook-aware entry pricing (Phase 22 #11) + Phase 27 Item 12
        market_maker_mode hook.

        When ADX < 20 (ranging regime) and the pair is liquid we ask
        market_maker_mode for an inventory-skewed quote — Stoikov GLFT-style
        bid/ask placement that is more aggressive than the +0.1% Phase 22
        offset. Falls back to legacy bid/ask shading if MM is unavailable.
        """
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            last = dataframe.iloc[-1] if dataframe is not None and len(dataframe) else None
            adx_val = float(last.get("adx") or last.get("adx_14") or 0.0) if last is not None else 0.0
        except Exception:
            adx_val = 0.0

        # Item 12: market_maker_mode quote when ranging.
        if adx_val and adx_val < 20.0:
            try:
                from market_maker_mode import get_market_maker
                mm = get_market_maker()
                ob = self.dp.orderbook(pair, 5)
                try:
                    from sensor_bridges import probe_orderbook
                    probe_orderbook(pair, ob, call_site="custom_entry_price:mm")
                except Exception:
                    pass
                if ob and ob.get("bids") and ob.get("asks"):
                    best_bid = float(ob["bids"][0][0])
                    best_ask = float(ob["asks"][0][0])
                    mid = (best_bid + best_ask) / 2.0
                    spread = best_ask - best_bid
                    # FIX-A6 (2026-04-25): publish mm_state pheromone so
                    # T17 spread-toxic gate has a producer. Audit found
                    # the publish_to_pheromone method had ZERO callers.
                    try:
                        half_spread_pct = (spread / mid) * 50.0 if mid > 0 else 0.0
                        mm.publish_to_pheromone({
                            "mode": "mm" if adx_val < 20 else "trend",
                            "half_spread_pct": half_spread_pct,
                            "gamma_effective": 0.0,
                            "bp_ema": 0.0,
                        }, pair)
                    except Exception:
                        pass
                    # GLFT skew: shrink half-spread by 30%, then bias by inventory.
                    inventory = mm._position.get(pair, 0.0) if hasattr(mm, "_position") else 0.0
                    skew = -0.0001 * inventory  # negative inventory → quote tighter on the bid
                    if side == "long":
                        quoted = mid - 0.35 * spread + skew * mid
                        return max(min(proposed_rate, quoted), best_bid)
                    elif side == "short":
                        quoted = mid + 0.35 * spread + skew * mid
                        return min(max(proposed_rate, quoted), best_ask)
            except Exception as e:
                logger.debug(f"[MarketMaker:Entry] {pair} skipped: {e}")

        # Sprint 2026-05-01 night — MAKER-ONLY pricing.
        # Sprint 2026-05-02: Adaptive maker/taker. ExchangeMicrostructure
        # Learner observes per-pair × hour fill rates; when a pair shows
        # < threshold maker fill rate over recent attempts, this method
        # falls back to taker pricing AUTOMATICALLY for that pair.
        # No human intervention — bot learns each venue's behavior.
        try:
            from exchange_microstructure_learner import should_use_maker
            use_maker = should_use_maker(pair)
        except Exception:
            use_maker = True

        if not use_maker:
            # Microstructure learner says maker won't fill on this pair —
            # use proposed_rate directly (taker fill, lose rebate but win
            # the trade). Logged so operator sees adaptive behavior.
            logger.info(
                f"[MakerAdaptive] {pair} → TAKER (learner observed low maker "
                f"fill rate; rebate sacrificed for fill probability)"
            )
            return proposed_rate

        try:
            from neural_organism import _p as _np_maker
            offset_bps = float(_np_maker("envelope.maker_only.offset_bps", 5.0))  # 5 bps = 0.05%
        except Exception:
            offset_bps = 5.0
        offset_frac = offset_bps / 10000.0

        try:
            ob = self.dp.orderbook(pair, 5)
            try:
                from sensor_bridges import probe_orderbook
                probe_orderbook(pair, ob, call_site="custom_entry_price:maker")
            except Exception:
                pass
            if ob and side == 'long' and ob.get('bids'):
                best_bid = float(ob['bids'][0][0])
                maker_price = best_bid * (1.0 - offset_frac)
                # Audit Finding D5: when proposed_rate < maker_price,
                # min() falls back to proposed_rate which may still be
                # above the bid → taker fill. Log it explicitly so the
                # operator can see fee rebate degradation, then deposit
                # a pheromone for downstream observability.
                if proposed_rate < maker_price:
                    logger.info(
                        f"[MakerOnly] {pair} long degraded to TAKER — "
                        f"proposed={proposed_rate:.6f} < maker={maker_price:.6f} "
                        f"(bid={best_bid:.6f}). Rebate lost this fill."
                    )
                    try:
                        from pheromone_field import get_pheromone_field
                        get_pheromone_field().deposit(
                            "maker_only", "taker_degradation",
                            {"pair": pair, "side": side,
                             "proposed": proposed_rate, "maker": maker_price},
                            half_life=600.0,
                        )
                    except Exception:
                        pass
                    return proposed_rate
                return maker_price
            elif ob and side == 'short' and ob.get('asks'):
                best_ask = float(ob['asks'][0][0])
                maker_price = best_ask * (1.0 + offset_frac)
                if proposed_rate > maker_price:
                    logger.info(
                        f"[MakerOnly] {pair} short degraded to TAKER — "
                        f"proposed={proposed_rate:.6f} > maker={maker_price:.6f} "
                        f"(ask={best_ask:.6f}). Rebate lost this fill."
                    )
                    try:
                        from pheromone_field import get_pheromone_field
                        get_pheromone_field().deposit(
                            "maker_only", "taker_degradation",
                            {"pair": pair, "side": side,
                             "proposed": proposed_rate, "maker": maker_price},
                            half_life=600.0,
                        )
                    except Exception:
                        pass
                    return proposed_rate
                return maker_price
        except Exception:
            pass
        return proposed_rate

    def custom_exit_price(self, pair: str, trade: 'Trade', current_time: datetime,
                          proposed_rate: float, current_profit: float,
                          exit_tag: str | None, **kwargs) -> float:
        """Orderbook-aware exit pricing (Phase 22 #12).

        Sprint 2026-05-02: AdaptiveExitUrgency. The bot computes per-trade
        urgency from profit, volatility, age, regime, and signal reversal.
        High urgency → cross the spread (taker, immediate fill, save the
        gain). Low urgency → place at maker price (capture rebate, wait).

        This fixes the NAORIS pattern: a +3% ROI hit at 09:03 sat unfilled
        as a maker until 11:48 then trailing-stop wiped the gain. With
        urgency-aware routing, +3% profit + age >2h + non-trivial vol
        → urgency >> threshold → taker exit fires fill immediately.
        """
        try:
            from pair_circuit import get_pair_circuit
            if get_pair_circuit().is_dormant(pair):
                return proposed_rate
        except Exception:
            pass

        # Compute urgency
        last_candle = None
        regime = None
        ai_sig = None
        try:
            df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if df is not None and len(df):
                last_candle = df.iloc[-1].squeeze().to_dict()
        except Exception:
            pass
        try:
            cached = self.ai_signal_cache.get(pair, {})
            ai_sig = cached.get("signal", "NEUTRAL")
            regime = cached.get("regime")
        except Exception:
            pass
        try:
            from adaptive_exit_urgency import compute_urgency
            urgency = compute_urgency(
                trade, current_profit, last_candle, ai_sig, regime, current_time
            )
            use_taker = urgency.get("should_use_taker", False)
            if use_taker:
                logger.info(
                    f"[ExitUrgency] {pair} TAKER — urgency={urgency['urgency_score']} "
                    f"profit={current_profit:+.2%} dominant={urgency['dominant_factor']}"
                )
        except Exception as _u_e:
            logger.debug(f"[ExitUrgency] {pair} skipped: {_u_e}")
            use_taker = False

        try:
            ob = self.dp.orderbook(pair, 5)
            try:
                from sensor_bridges import probe_orderbook
                probe_orderbook(pair, ob, call_site="custom_exit_price")
            except Exception:
                pass
            if ob:
                # When taker urgency triggered, cross the spread aggressively
                # (place inside the opposite side). Otherwise stay passive
                # at our own side for maker rebate.
                if not trade.is_short and ob.get('asks') and ob.get('bids'):
                    best_ask = float(ob['asks'][0][0])
                    best_bid = float(ob['bids'][0][0])
                    if use_taker:
                        # Long exit: SELL aggressively at the bid (cross the spread)
                        return min(proposed_rate, best_bid * 0.999)
                    else:
                        # Long exit: stay at ask (passive maker)
                        return max(proposed_rate, best_ask * 0.999)
                elif trade.is_short and ob.get('bids') and ob.get('asks'):
                    best_bid = float(ob['bids'][0][0])
                    best_ask = float(ob['asks'][0][0])
                    if use_taker:
                        # Short exit: BUY-back aggressively at the ask (cross spread)
                        return max(proposed_rate, best_ask * 1.001)
                    else:
                        # Short exit: stay at bid (passive maker)
                        return min(proposed_rate, best_bid * 1.001)
        except Exception:
            pass
        return proposed_rate

    def adjust_trade_position(self, trade: 'Trade', current_time: datetime,
                              current_rate: float, current_profit: float,
                              min_stake: float | None, max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs) -> float | None:
        """DCA + partial exit + staged profit lock (OMEGA-inspired Phase 23)."""
        if self.dp.runmode.value not in ('dry_run', 'live'):
            return None
        # Sprint 2026-05-01: max DCA levels comes from RiskEnvelope (autonomy
        # tier × decay). L0 → 1 add (initial + 1 DCA), L5 → 5. Replaces the
        # hardcoded 4 cap that was the structural enabler of the Apr 17
        # ADA $0.25 → $1,463 pyramid (3 DCAs uncapped).
        try:
            from risk_envelope import get_risk_envelope
            max_dca = int(get_risk_envelope().max_dca_levels())
        except Exception:
            max_dca = 1  # safest cold-start default
        # nr_of_successful_entries counts the INITIAL entry too, so:
        #   max_dca=1 → block at >=2 entries (1 initial + 1 DCA done)
        #   max_dca=5 → block at >=6 entries
        if trade.nr_of_successful_entries >= (1 + max_dca):
            return None

        # Task 20: pair-circuit dormant gate. Without this, DCA pyramid and
        # partial-exit decisions for a dormant pair (≥5 consecutive empty
        # orderbooks) would still propose stake deltas; Freqtrade would
        # attempt exchange orders on a book that just rejected entry + exit
        # attempts. Early-return keeps the position intact and waits for
        # the revive probe (scheduler._pair_circuit_revive_tick) to reopen
        # the circuit.
        try:
            from pair_circuit import get_pair_circuit
            if get_pair_circuit().is_dormant(trade.pair):
                return None
        except Exception:
            pass

        # Phase 25: Empty orderbook guard — skip if rates are 0 (exchange has no data)
        if current_entry_rate <= 0 or current_exit_rate <= 0:
            return None

        hours_held = (current_time - trade.open_date_utc).total_seconds() / 3600
        if hours_held < 0.5:  # Wait at least 30min (was 1h)
            return None

        # ═══ STAGED PARTIAL PROFIT LOCK (OMEGA-inspired) ═══
        # +6% effective PnL → close 25%, set breakeven flag
        # +12% effective PnL → close 25% more, tighten SL to entry+buffer
        # Remaining 50% → trailing stop handles it (free ride)
        effective_pnl = current_profit * (trade.leverage or 1.0)
        partial_1_done = trade.get_custom_data("partial_lock_1", False)
        partial_2_done = trade.get_custom_data("partial_lock_2", False)

        if not partial_1_done and effective_pnl >= _np("strategy.dca_lock1_pnl", 0.02):
            close_amount = trade.stake_amount * _np("strategy.dca_lock_pct", 0.25)
            if min_stake and close_amount >= min_stake:
                trade.set_custom_data("partial_lock_1", True)
                try:
                    self.dp.send_msg(
                        f"LOCK1: {trade.pair} +{effective_pnl:.1%} eff → closing 25%, setting breakeven SL")
                except Exception:
                    pass
                logger.info(f"[PartialLock] {trade.pair} LOCK1: +{effective_pnl:.1%} → close 25%")
                return -close_amount

        if partial_1_done and not partial_2_done and effective_pnl >= _np("strategy.dca_lock2_pnl", 0.06):
            close_amount = trade.stake_amount * _np("strategy.dca_lock_pct", 0.25)
            if min_stake and close_amount >= min_stake:
                trade.set_custom_data("partial_lock_2", True)
                trade.set_custom_data("breakeven_active", True)
                try:
                    self.dp.send_msg(
                        f"LOCK2: {trade.pair} +{effective_pnl:.1%} eff → closing 25% more, SL→breakeven")
                except Exception:
                    pass
                logger.info(f"[PartialLock] {trade.pair} LOCK2: +{effective_pnl:.1%} → close 25% + breakeven")
                return -close_amount

        # ═══ EXISTING: AI-based DCA / Reduce / Half-exit ═══
        cached = self.ai_signal_cache.get(trade.pair, {})
        confidence = cached.get('confidence', 0.0)
        signal = cached.get('signal', 'NEUTRAL')
        entry_conf = trade.get_custom_data("ai_confidence", 0.5)
        if not isinstance(entry_conf, (int, float)):
            entry_conf = 0.5

        # Sprint 2026-05-01 night — WINNER PYRAMID
        # "Cut your losers fast, let your winners run."
        # Pyramid up only when:
        #   (a) position is already profitable beyond the env threshold
        #   (b) signal confidence STILL above env_min_conf (not faded)
        #   (c) confidence has held or risen since entry (not deteriorating
        #       beyond the conf_decay_tolerance window)
        # All four thresholds come from PARAM_REGISTRY so neurons can tune.
        try:
            from neural_organism import _p as _np_pyr
            wp_min_profit = float(_np_pyr("envelope.winner_pyramid.min_profit", 0.02))
            wp_min_conf = float(_np_pyr("envelope.winner_pyramid.min_conf", 0.65))
            wp_decay_tol = float(_np_pyr("envelope.winner_pyramid.conf_decay_tolerance", 0.10))
        except Exception:
            wp_min_profit, wp_min_conf, wp_decay_tol = 0.02, 0.65, 0.10

        # PYRAMID: position winning AND confidence holds → add on the winner
        if (current_profit >= wp_min_profit
                and confidence >= wp_min_conf
                and confidence >= float(entry_conf) - wp_decay_tol):
            # ═══ Phase 27 EMERGENCY DCA GATE ═══
            # Audit found: today ADA went $0.25 → $1,463 via 3 DCAs because this
            # branch had ZERO Phase 27 controls. Apply per-pair Kelly + CAAT +
            # Hawkes + Constitution position cap to every DCA proposal.
            try:
                regime_for_dca = cached.get("regime") or "_global"
                # Gate 1: per-pair Bayesian Kelly — 0 means "this pair / regime
                # has structurally negative edge right now, no more capital".
                # Sprint 2026-05-05 (B-CONNECT C1): side-aware Kelly for DCA decisions.
                side_for_dca = "short" if trade.is_short else "long"
                kelly_dca = self._position_sizer.bayesian_kelly.kelly_fraction(
                    pair=trade.pair, regime=regime_for_dca, side=side_for_dca,
                )
                if kelly_dca <= 0.005:
                    self._emit_dca_gate(
                        trade.pair, reason="kelly",
                        detail=(f"per-pair Kelly={kelly_dca:.4f} "
                                f"(no structural edge in {regime_for_dca})"),
                        extra={"kelly": round(kelly_dca, 6), "regime": regime_for_dca},
                    )
                    return None

                # Gate 2: CAAT multiplier — if the asymmetric formula says the
                # current state warrants <30% sizing, the LAST thing we want
                # is to ADD on top of an existing position.
                try:
                    df_dca, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
                    last_for_dca = df_dca.iloc[-1].squeeze() if df_dca is not None and len(df_dca) else None
                except Exception:
                    last_for_dca = None
                if last_for_dca is not None:
                    try:
                        caat_mult_dca, caat_breakdown_dca = self._caat_asymmetric_multiplier(
                            pair=trade.pair, regime=regime_for_dca,
                            confidence=confidence,
                            ai_decision=cached, last_candle=last_for_dca,
                            proposed_stake=trade.stake_amount,
                        )
                        if caat_mult_dca < 0.30:
                            self._emit_dca_gate(
                                trade.pair, reason="caat",
                                detail=(f"CAAT mult={caat_mult_dca:.3f} <0.30 "
                                        f"({caat_breakdown_dca})"),
                                extra={"caat": round(caat_mult_dca, 4)},
                            )
                            return None
                    except Exception:
                        caat_mult_dca = 1.0
                else:
                    caat_mult_dca = 1.0

                # Gate 3: Hawkes cascade veto — if the branching ratio is in
                # the danger zone, don't pour fuel on the fire.
                try:
                    from order_flow import get_order_flow
                    of_dca = get_order_flow()
                    of_state = of_dca.analyze(trade.pair, trades=[])
                    n_branch = float(of_state.get("hawkes_branching_ratio", 0.0))
                    if n_branch >= 0.80:
                        self._emit_dca_gate(
                            trade.pair, reason="hawkes",
                            detail=f"Hawkes n={n_branch:.2f} ≥0.80 (cascade risk)",
                            extra={"hawkes_branching_ratio": round(n_branch, 4)},
                        )
                        return None
                except Exception:
                    pass

                # Gate 4: ENVELOPE-DRIVEN combined-position cap.
                # Sprint 2026-05-01: replaces the constitution-only check.
                # The envelope's max_combined_position scales with autonomy
                # tier × hormonal × decay so a wounded L0 caps combined
                # position at 7% (vs 30% at calm L5). Constitution is also
                # honoured as an upper-bound: take the TIGHTER of the two.
                try:
                    portfolio_value = float(getattr(self.risk_budget, "portfolio_value", 0.0) or 0.0)
                    if portfolio_value <= 0 and self.wallets:
                        portfolio_value = float(self.wallets.get_total_stake_amount())
                    if portfolio_value > 0:
                        # Sprint 2026-05-01 evening — EarnedTrust pipeline:
                        # combined cap and DCA increment scale with the
                        # CURRENT signal confidence so a high-conviction
                        # add-on gets larger stake than a mediocre one.
                        # Audit Finding #4 fix (2026-05-02): also pass
                        # FOMO-veto inputs (recent_low/high, current_price,
                        # atr) so the DCA path does not silently bypass
                        # the chase-prevention gate that custom_stake_amount
                        # already honors.
                        _dca_recent_low = None
                        _dca_recent_high = None
                        _dca_atr = None
                        try:
                            if last_for_dca is not None:
                                _dca_recent_low = float(
                                    last_for_dca.get("lowest_low_20")
                                    or last_for_dca.get("lowest_low_14") or 0.0
                                ) or None
                                _dca_recent_high = float(
                                    last_for_dca.get("highest_high_14") or 0.0
                                ) or None
                                _dca_atr = float(
                                    last_for_dca.get("atr") or 0.0
                                ) or None
                        except Exception:
                            pass
                        try:
                            from risk_envelope import get_risk_envelope
                            envelope_combined_cap = float(
                                get_risk_envelope().max_combined_position(
                                    portfolio_value,
                                    confidence=float(confidence),
                                    signal_type=signal,
                                    pair=trade.pair,
                                    recent_low=_dca_recent_low,
                                    current_price=float(current_rate),
                                    atr=_dca_atr,
                                    recent_high=_dca_recent_high,
                                )
                            )
                        except Exception:
                            envelope_combined_cap = portfolio_value * 0.10
                        # Constitution cap (institutional safety floor)
                        try:
                            from constitution import CONSTITUTION
                            const_pct = float(CONSTITUTION["safety_limits"]["max_single_position_pct"]) / 100.0
                            constitution_cap = portfolio_value * const_pct
                        except Exception:
                            constitution_cap = portfolio_value * 0.03
                        # Tighter wins
                        max_total_stake = min(envelope_combined_cap, constitution_cap)

                        # Proposed DCA conviction-scaled (no hardcoded 0.30
                        # — that was Apr 17 mechanism).
                        try:
                            from risk_envelope import get_risk_envelope as _ge
                            dca_pct = float(_ge().dca_increment_pct(
                                confidence=float(confidence),
                                signal_type=signal,
                                pair=trade.pair,
                                recent_low=_dca_recent_low,
                                current_price=float(current_rate),
                                atr=_dca_atr,
                                recent_high=_dca_recent_high,
                            ))
                            proposed_dca = portfolio_value * dca_pct
                        except Exception:
                            proposed_dca = portfolio_value * 0.02

                        # Never exceed exchange-allowed stake either
                        proposed_dca = min(proposed_dca, float(max_stake or proposed_dca))
                        proposed_total = float(trade.stake_amount or 0) + proposed_dca
                        if proposed_total > max_total_stake:
                            self._emit_dca_gate(
                                trade.pair, reason="combined_cap",
                                detail=(f"proposed total ${proposed_total:.2f} > "
                                        f"envelope cap ${max_total_stake:.2f}"),
                                extra={
                                    "proposed_total": round(proposed_total, 2),
                                    "envelope_cap": round(envelope_combined_cap, 2),
                                    "constitution_cap": round(constitution_cap, 2),
                                    "cap_used": round(max_total_stake, 2),
                                },
                            )
                            return None
                except Exception as e:
                    logger.debug(f"[DCA-GATE] envelope+constitution cap check failed: {e}")
            except Exception as e:
                self._emit_dca_gate(
                    trade.pair, reason="fail_open",
                    detail=f"guard chain failed open ({type(e).__name__}: {e}); blocking DCA defensively",
                    extra={"error_type": type(e).__name__},
                )
                return None

            # All four Phase 27 gates passed — DCA is risk-bounded.
            # Sprint 2026-05-01 evening — EarnedTrust conviction-scaled.
            # Audit Finding #4 fix (2026-05-02): forward FOMO inputs into
            # dca_increment_pct so the PYRAMID add-stake honors the chase
            # gate just like custom_stake_amount does. Re-fetch the candle
            # locally — this branch sits outside Gate 4's scope so
            # last_for_dca isn't visible here.
            _pyr_recent_low = None
            _pyr_recent_high = None
            _pyr_atr = None
            try:
                df_pyr, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
                if df_pyr is not None and len(df_pyr):
                    cand = df_pyr.iloc[-1].squeeze()
                    _pyr_recent_low = float(
                        cand.get("lowest_low_20") or cand.get("lowest_low_14") or 0.0
                    ) or None
                    _pyr_recent_high = float(cand.get("highest_high_14") or 0.0) or None
                    _pyr_atr = float(cand.get("atr") or 0.0) or None
            except Exception:
                pass
            try:
                from risk_envelope import get_risk_envelope as _ge2
                portfolio_value_dca = float(getattr(self.risk_budget, "portfolio_value", 0.0) or 0.0)
                if portfolio_value_dca <= 0 and self.wallets:
                    portfolio_value_dca = float(self.wallets.get_total_stake_amount())
                if portfolio_value_dca > 0:
                    dca_pct_final = float(_ge2().dca_increment_pct(
                        confidence=float(confidence),
                        signal_type=signal,
                        pair=trade.pair,
                        recent_low=_pyr_recent_low,
                        current_price=float(current_rate),
                        atr=_pyr_atr,
                        recent_high=_pyr_recent_high,
                    ))
                    add_stake = portfolio_value_dca * dca_pct_final
                else:
                    add_stake = float(max_stake) * 0.10  # very conservative cold-start
                # Never exceed exchange-permitted max for this trade
                add_stake = min(add_stake, float(max_stake or add_stake))
            except Exception:
                add_stake = float(max_stake) * 0.10
            if min_stake and add_stake >= min_stake:
                logger.info(
                    f"[DCA] {trade.pair} PYRAMID: conf {confidence:.0%} "
                    f"(Kelly={kelly_dca:.3f}, CAAT={caat_mult_dca:.2f}, "
                    f"add=${add_stake:.2f} envelope-driven)"
                )
                return add_stake

        # REDUCE: Confidence dropped + losing
        if confidence < 0.30 and entry_conf > 0.60 and current_profit < -0.02:
            logger.info(f"[DCA] {trade.pair} REDUCE 30%: conf {entry_conf:.0%}→{confidence:.0%}")
            return -(trade.stake_amount * 0.30)

        # HALF-EXIT: Signal reversed
        if not trade.is_short and signal == 'BEARISH' and confidence > 0.60:
            logger.info(f"[DCA] {trade.pair} HALF-EXIT: BEARISH conf={confidence:.0%}")
            return -(trade.stake_amount * 0.50)
        if trade.is_short and signal == 'BULLISH' and confidence > 0.60:
            logger.info(f"[DCA] {trade.pair} HALF-EXIT: BULLISH conf={confidence:.0%}")
            return -(trade.stake_amount * 0.50)

        return None

    # ── Remaining Gems: adjust_entry/exit_price + funding rate ────────

    def adjust_entry_price(self, trade: 'Trade', order: 'Order', pair: str,
                           current_time: datetime, proposed_rate: float,
                           current_order_rate: float, entry_tag: str | None,
                           side: str, **kwargs) -> float:
        """Re-adjust unfilled entry orders each candle to improve fill rate (Phase 22 #remaining).
        If order hasn't filled, chase the price slightly."""
        try:
            ob = self.dp.orderbook(pair, 3)
            try:
                from sensor_bridges import probe_orderbook
                probe_orderbook(pair, ob, call_site="adjust_entry_price")
            except Exception:
                pass
            if ob and side == 'long' and ob.get('bids'):
                best_bid = ob['bids'][0][0]
                # Chase: move order to best bid + 0.1% (improve fill probability)
                new_price = best_bid * 1.001
                if abs(new_price - current_order_rate) / current_order_rate > 0.002:
                    logger.debug(f"[AdjustEntry] {pair} {current_order_rate:.6f} → {new_price:.6f}")
                    return new_price
            elif ob and side == 'short' and ob.get('asks'):
                best_ask = ob['asks'][0][0]
                new_price = best_ask * 0.999
                if abs(new_price - current_order_rate) / current_order_rate > 0.002:
                    return new_price
        except Exception:
            pass
        return current_order_rate  # Keep current price

    def adjust_exit_price(self, trade: 'Trade', order: 'Order', pair: str,
                          current_time: datetime, proposed_rate: float,
                          current_order_rate: float, entry_tag: str | None,
                          side: str, **kwargs) -> float:
        """Re-adjust unfilled exit orders to lock in profits faster (Phase 22 #remaining)."""
        try:
            from pair_circuit import get_pair_circuit
            if get_pair_circuit().is_dormant(pair):
                return current_order_rate
        except Exception:
            pass
        try:
            ob = self.dp.orderbook(pair, 3)
            try:
                from sensor_bridges import probe_orderbook
                probe_orderbook(pair, ob, call_site="adjust_exit_price")
            except Exception:
                pass
            if ob and not trade.is_short and ob.get('asks'):
                best_ask = ob['asks'][0][0]
                new_price = best_ask * 0.999
                if abs(new_price - current_order_rate) / current_order_rate > 0.002:
                    return new_price
            elif ob and trade.is_short and ob.get('bids'):
                best_bid = ob['bids'][0][0]
                new_price = best_bid * 1.001
                if abs(new_price - current_order_rate) / current_order_rate > 0.002:
                    return new_price
        except Exception:
            pass
        return current_order_rate
