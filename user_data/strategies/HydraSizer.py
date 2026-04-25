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

    # Minimal ROI — let winners run longer (Dobrynskaya 2021: crypto momentum lasts weeks)
    # Old "240": 0 was killing +0.5% winners at 4h. Extended to give trailing stop time to work.
    minimal_roi = {
        "0": 0.15,       # 15% immediate (unlikely, but protects windfall)
        "60": 0.05,      # 5% after 1h
        "120": 0.03,     # 3% after 2h (was 2% — slightly more room)
        "360": 0.015,    # 1.5% after 6h (new — was 0% at 4h)
        "720": 0.005,    # 0.5% after 12h (new — let it breathe)
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
        self.ai_signal_cache = {} # Memory cache: { "BTC/USDT": {"signal": "BULLISH", "confidence": 0.8, "timestamp": datetime} }
        self.cache_ttl_hours = 6 # Non-NEUTRAL signals valid for 6 hours (Phase 22: increased from 4h)
        self._neutral_ttl_hours = 8.0  # NEUTRAL signals retried after 8h (reduced LLM calls for free tier)

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
        # ── C1: RAG health probe (cached 60s) ─────────────────────────
        try:
            probe_ts, healthy = self._rag_health_cache
            if (now_ts - probe_ts) > 60.0:
                healthy = True
                try:
                    import urllib.request as _ur
                    req = _ur.Request("http://127.0.0.1:8891/health")
                    with _ur.urlopen(req, timeout=5) as resp:
                        healthy = (200 <= resp.status < 500)
                except Exception:
                    healthy = False
                self._rag_health_cache = (now_ts, healthy)
            if not healthy:
                return "rag_health_unreachable"
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
                expired = [(k, dl) for k, dl in list(self._pending_fill_checks.items())
                           if _time.time() >= dl]
                for key, deadline in expired:
                    pair, side = key
                    amt = open_by_side.get((pair, side), 0.0)
                    filled = amt > 0
                    age = max(0.0, _time.time() - (deadline - 600.0))
                    try:
                        circuit.record_order_attempt(pair, filled=filled, age_seconds=age)
                    except Exception:
                        pass
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
            except Exception as e:
                logger.warning(f"[bot_loop_start] Fetch failed for {p}: {e}")
            # Only cache if we got a real signal (POST with tech_data).
            # NEUTRAL from missing tech_data should NOT pollute cache.
            if sig.get("confidence", 0) > 0 or sig.get("signal") != "NEUTRAL":
                self.ai_signal_cache[p] = sig
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

    def get_entry_signal(self, pair, timeframe, dataframe):
        """
        DIAGNOSTIC OVERRIDE: Wraps parent's get_entry_signal to log exactly
        what Freqtrade sees when checking for entry signals.
        This tells us precisely WHY signals are accepted or rejected.
        """
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
        try:
            conn = sqlite3.connect(self.db_path)
            return conn
        except sqlite3.Error as e:
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
            # Phase 17: POST with technical data when available, GET fallback
            if technical_data:
                response = requests.post(
                    f"{rag_service_url}/signal/{pair}",
                    json={"technical_data": technical_data},
                    timeout=120
                )
            else:
                response = requests.get(
                    f"{rag_service_url}/signal/{pair}",
                    timeout=120  # Quality-first: give ColBERT+MADAM time to complete
                )
            _latency = (_time.time() - _t0) * 1000
            logger.info(f"[RAG Latency] {pair}: {_latency:.0f}ms (status={response.status_code}, POST={'Y' if technical_data else 'N'})")
            if response.status_code == 200:
                parsed = response.json()
                signal_data["signal"] = parsed.get("signal", "NEUTRAL")
                signal_data["confidence"] = parsed.get("confidence", 0.0)
                signal_data["reasoning"] = parsed.get("reasoning", "")
                logger.info(f"RAG Signal: {signal_data['signal']} ({signal_data['confidence']}) for {pair}")
            else:
                logger.warning(f"RAG service returned {response.status_code} for {pair}")
        except Exception as e:
            is_connection_error = False
            try:
                import requests as _req
                is_connection_error = isinstance(e, _req.exceptions.ConnectionError)
            except Exception:
                pass

            if is_connection_error:
                logger.warning(f"RAG service not running. Falling back to subprocess for {pair}")
                self._get_ai_signal_subprocess(pair, signal_data)
            else:
                logger.error(f"Error calling RAG Signal Service for {pair}: {e}")

        # 3. Save to Cache (ALL signals including NEUTRAL — TTL handles expiry)
        # NEUTRAL uses shorter TTL (0.9h) so it's retried on next candle
        self.ai_signal_cache[pair] = signal_data
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

        # Sentiment features from SQLite (used by custom_stake_amount)
        conn = self._get_sqlite_connection()
        if conn:
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
            conn.close()
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
                    if tier in ("PANIC", "EXTREME") or fear_lvl >= 2.0:
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
                    if not hasattr(self, '_perception_cache'):
                        self._perception_cache = {}
                    self._perception_cache[pair] = _tp_result
                    logger.info(
                        f"[Phase26:TriplePerception] {pair}: {_tp_result['signal']} "
                        f"conf={_tp_result['confidence']:.2f} sizing={_tp_result['sizing_multiplier']:.2f}"
                    )
            except Exception as e:
                logger.warning(f"[Phase26:TriplePerception] {pair} failed: {e}", exc_info=True)

            ai_decision = self._get_ai_signal(pair, last_time, dataframe=df)

            # Phase 26: Enrich AI decision with Triple Perception
            if _tp_result and _tp_result.get("confidence", 0) > 0.1:
                # If perception and RAG agree → boost confidence
                # If they disagree → reduce confidence (disagreement penalty)
                tp_signal = _tp_result.get("signal", "NEUTRAL")
                ai_signal = ai_decision.get("signal", "NEUTRAL")
                if tp_signal == ai_signal and tp_signal != "NEUTRAL":
                    ai_decision["confidence"] = min(ai_decision.get("confidence", 0) * 1.15, 0.95)
                    ai_decision["sizing_multiplier"] = _tp_result.get("sizing_multiplier", 1.0)
                elif tp_signal != "NEUTRAL" and ai_signal != "NEUTRAL" and tp_signal != ai_signal:
                    ai_decision["confidence"] *= 0.75  # disagreement penalty
                ai_decision["perception"] = _tp_result

            signal_type = ai_decision.get('signal', 'NEUTRAL')
            confidence = ai_decision.get('confidence', 0.0)
            is_bullish = signal_type == 'BULLISH'
            is_bearish = signal_type == 'BEARISH'

            # ═══ GRADUATED EXECUTION: Log ALL signals, trade only high-confidence ═══
            # Philosophy: LOG EVERYTHING → TRADE SELECTIVELY
            # Every signal (even conf=0.05) is shadow-logged for calibrator learning.
            # Only conf>=0.55 directional signals become real trades (fee break-even ~0.55).
            # The calibrator needs BOTH positive and negative examples to learn properly.
            # Logging <0.30 = "look how bad this was" is just as valuable as "look how good".
            # Shadow data proves: conf 0.30+ = +7.88% avg PnL, %83 win rate
            # Lowered from 0.55 to 0.40 — sweet spot between fee break-even and missed alpha
            REAL_TRADE_THRESHOLD = float(self.confidence_threshold.value)

            # Determine execution mode for logging
            if confidence >= REAL_TRADE_THRESHOLD and signal_type != 'NEUTRAL':
                exec_mode = "REAL"
            elif confidence >= _np("strategy.stoploss_floor", 0.30):
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
        try:
            from db import get_db_connection
            conn = get_db_connection()
            row = conn.execute("""
                SELECT forgone_alpha_7d FROM pair_thresholds
                WHERE pair = ? AND regime = ?
            """, (pair, regime)).fetchone()
            conn.close()
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
        """Phase 27 Fix 6: per-pair adaptive threshold lookup (fallback 0.50)."""
        try:
            from db import get_db_connection
            conn = get_db_connection()
            row = conn.execute("""
                SELECT confidence_threshold FROM pair_thresholds
                WHERE pair = ? AND regime = ?
            """, (pair, regime)).fetchone()
            conn.close()
            if row is not None:
                return float(row["confidence_threshold"])
        except Exception:
            pass
        return 0.50

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
            # cortisol ∈ [0.5, 1.0] in code convention (1.0 calm / 0.5 stressed)
            # factor = (2 − cortisol): calm → 1.0, stressed → 1.5 (widen).
            mult *= (2.0 - max(0.5, min(1.0, cortisol)))
            mult = max(1.0, min(5.0, mult))  # safety clamp
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

        # Leverage-aware equity cap: max 15% EQUITY loss regardless of leverage
        # 1x → -15% price, 2x → -7.5% price, 3x → -5% price
        MAX_EQUITY_LOSS = 0.15
        leverage = trade.leverage or 1.0
        leverage_aware_floor = -(MAX_EQUITY_LOSS / leverage)
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
                confidence, pair=pair, regime=regime_for_kelly
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
                cerebellum_mult = float(get_cerebellum().get_timing_multiplier())
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
            except Exception as e:
                logger.debug(f"[Sprint2:Constitution] Check skipped: {e}")

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
        try:
            import time as _tm_d2
            self._pending_fill_checks[(pair, side)] = _tm_d2.time() + 600.0
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
            self._bayesian_kelly.update(
                won=won, pnl_pct=pnl_pct, pair=pair, regime=regime_for_exit
            )
            wp = self._bayesian_kelly.win_probability(pair=pair, regime=regime_for_exit)
            kf = self._bayesian_kelly.kelly_fraction(pair=pair, regime=regime_for_exit)
            logger.info(
                f"[BayesianKelly:{pair}/{regime_for_exit}] Updated: "
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
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path, timeout=10)
            conn.execute("""
                UPDATE ai_decisions SET outcome_pnl = ?, outcome_duration = ?
                WHERE pair = ? AND outcome_pnl IS NULL
                ORDER BY timestamp DESC LIMIT 1
            """, (trade_pnl_pct,
                  int((current_time - trade.open_date_utc).total_seconds() / 60) if trade.open_date else None,
                  pair))
            conn.commit()
            conn.close()
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
        # Ensure protection_logs table exists for testnet data collection
        conn = self._get_sqlite_connection()
        if conn:
            try:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS protection_logs (
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
            conn.close()
        logger.info("[bot_start] AI Trading System ready.")

    @property
    def protections(self):
        """Built-in protections — TESTNET MODE: Very loose, log everything.
        Trade-First: NEVER block trades aggressively. Just brief cooldowns.
        All trade data logged to DB for analysis when switching to real money."""
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 1,  # Just 1 candle cooldown (not 2)
            },
            {
                # Only trigger after 6 consecutive stoplosses on same pair (very loose)
                "method": "StoplossGuard",
                "lookback_period_candles": 48,
                "trade_limit": 6,  # 6 stoplosses before lock (was 4)
                "stop_duration_candles": 2,  # Lock only 2 candles (was 4)
                "only_per_pair": True,
            },
            {
                # Nuclear option: only if account drawdown >50%
                # Testnet: Equal Risk sizing limits per-trade loss to 0.5% of portfolio
                # so 50% drawdown would require ~100 consecutive losses — nearly impossible
                "method": "MaxDrawdown",
                "lookback_period_candles": 168,  # 7 days window (was 3 days — too short)
                "trade_limit": 20,
                "stop_duration_candles": 2,  # Brief pause, resume quickly
                "max_allowed_drawdown": 0.50,  # 50% (was 25% — kept locking bot)
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

        # 1. STALE TRADE
        if hours_held > self.stale_trade_hours.value and abs(current_profit) < 0.005:
            return f"stale_{hours_held:.0f}h_flat"

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
        if hours_held > 0 and int(hours_held) % 4 == 0:  # Every 4 hours
            try:
                conn = self._get_sqlite_connection()
                if conn:
                    conn.execute(
                        "INSERT INTO protection_logs (timestamp, event_type, pair, details, profit_at_event) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (current_time.isoformat(), "trade_check", pair,
                         f"signal={signal} conf={confidence:.2f} entry_conf={entry_conf} hours={hours_held:.1f}",
                         round(current_profit, 6))
                    )
                    conn.commit()
                    conn.close()
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

        # Legacy Phase 22 path.
        try:
            ob = self.dp.orderbook(pair, 5)
            try:
                from sensor_bridges import probe_orderbook
                probe_orderbook(pair, ob, call_site="custom_entry_price:legacy")
            except Exception:
                pass
            if ob and side == 'long' and ob.get('bids'):
                best_bid = ob['bids'][0][0]
                return min(proposed_rate, best_bid * 1.001)
            elif ob and side == 'short' and ob.get('asks'):
                best_ask = ob['asks'][0][0]
                return max(proposed_rate, best_ask * 0.999)
        except Exception:
            pass
        return proposed_rate

    def custom_exit_price(self, pair: str, trade: 'Trade', current_time: datetime,
                          proposed_rate: float, current_profit: float,
                          exit_tag: str | None, **kwargs) -> float:
        """Orderbook-aware exit pricing (Phase 22 #12)."""
        try:
            from pair_circuit import get_pair_circuit
            if get_pair_circuit().is_dormant(pair):
                return proposed_rate
        except Exception:
            pass
        try:
            ob = self.dp.orderbook(pair, 5)
            try:
                from sensor_bridges import probe_orderbook
                probe_orderbook(pair, ob, call_site="custom_exit_price")
            except Exception:
                pass
            if ob:
                if not trade.is_short and ob.get('asks'):
                    best_ask = ob['asks'][0][0]
                    return max(proposed_rate, best_ask * 0.999)
                elif trade.is_short and ob.get('bids'):
                    best_bid = ob['bids'][0][0]
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
        if trade.nr_of_successful_entries >= 4:
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

        # PYRAMID: Confidence up + profitable
        if confidence > 0.80 and current_profit > 0.01 and confidence > entry_conf + 0.1:
            # ═══ Phase 27 EMERGENCY DCA GATE ═══
            # Audit found: today ADA went $0.25 → $1,463 via 3 DCAs because this
            # branch had ZERO Phase 27 controls. Apply per-pair Kelly + CAAT +
            # Hawkes + Constitution position cap to every DCA proposal.
            try:
                regime_for_dca = cached.get("regime") or "_global"
                # Gate 1: per-pair Bayesian Kelly — 0 means "this pair / regime
                # has structurally negative edge right now, no more capital".
                kelly_dca = self._position_sizer.bayesian_kelly.kelly_fraction(
                    pair=trade.pair, regime=regime_for_dca
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

                # Gate 4: Constitution-style absolute position cap. Per ALPHA
                # PRENSİP 0 + constitution.max_single_position_pct=3%, the
                # combined (current_stake + proposed DCA) can NEVER exceed
                # 3% of portfolio value. This is the hard ceiling that would
                # have stopped today's MNT $6,826 monster.
                try:
                    from constitution import IDENTITY_LIMITS, CONSTITUTION
                    portfolio_value = float(getattr(self.risk_budget, "portfolio_value", 0.0) or 0.0)
                    max_pos_pct = float(CONSTITUTION["safety_limits"]["max_single_position_pct"]) / 100.0
                    if portfolio_value > 0:
                        max_total_stake = portfolio_value * max_pos_pct
                        proposed_dca = max_stake * 0.3
                        proposed_total = float(trade.stake_amount or 0) + proposed_dca
                        if proposed_total > max_total_stake:
                            self._emit_dca_gate(
                                trade.pair, reason="portfolio_cap",
                                detail=(f"proposed total ${proposed_total:.2f} > "
                                        f"{max_pos_pct*100:.1f}% portfolio cap "
                                        f"(${max_total_stake:.2f})"),
                                extra={
                                    "proposed_total": round(proposed_total, 2),
                                    "cap": round(max_total_stake, 2),
                                    "cap_pct": max_pos_pct,
                                },
                            )
                            return None
                except Exception as e:
                    logger.debug(f"[DCA-GATE] constitution cap check failed: {e}")
            except Exception as e:
                self._emit_dca_gate(
                    trade.pair, reason="fail_open",
                    detail=f"guard chain failed open ({type(e).__name__}: {e}); blocking DCA defensively",
                    extra={"error_type": type(e).__name__},
                )
                return None

            # All four Phase 27 gates passed — DCA is risk-bounded.
            add_stake = max_stake * 0.3
            if min_stake and add_stake >= min_stake:
                logger.info(
                    f"[DCA] {trade.pair} PYRAMID: conf {confidence:.0%} "
                    f"(Kelly={kelly_dca:.3f}, CAAT={caat_mult_dca:.2f})"
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
