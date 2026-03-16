# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# isort: skip_file
import logging
import sqlite3
import sys
import pandas as pd
import numpy as np
import os
from datetime import datetime
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

logger = logging.getLogger(__name__)

class AIFreqtradeSizer(IStrategy):
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

    # Minimal ROI (handled mostly by AI and custom stoploss)
    minimal_roi = {
        "0": 0.15,
        "60": 0.05,
        "120": 0.02,
        "240": 0
    }

    # Stoploss (Wide, rely on dynamic trailing/custom stoploss)
    stoploss = -0.20
    use_custom_stoploss = True

    # Trailing stop
    trailing_stop = False

    timeframe = '1h'

    # ── Hyperopt Parameters (Phase 22) ──────────────────────────────────
    # These make ALL key thresholds tunable via: freqtrade hyperopt --spaces entry exit stake protection
    confidence_threshold = DecimalParameter(0.30, 0.80, decimals=2, default=0.50, space='buy', optimize=True, load=True)
    atr_stoploss_mult = DecimalParameter(1.5, 5.0, decimals=1, default=3.0, space='protection', optimize=True, load=True)
    fg_extreme_threshold = IntParameter(15, 30, default=20, space='buy', optimize=True, load=True)
    stale_trade_hours = IntParameter(4, 24, default=8, space='sell', optimize=True, load=True)
    leverage_max = DecimalParameter(1.0, 5.0, decimals=1, default=2.0, space='buy', optimize=True, load=True)

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.db_path = os.path.join(self.config['user_data_dir'], "db", "ai_data.sqlite")
        self.rag_script_path = os.path.join(self.config['user_data_dir'], "scripts", "rag_graph.py")
        self.ai_signal_cache = {} # Memory cache: { "BTC/USDT": {"signal": "BULLISH", "confidence": 0.8, "timestamp": datetime} }
        self.cache_ttl_hours = 6 # Non-NEUTRAL signals valid for 6 hours (Phase 22: increased from 4h)
        self._neutral_ttl_hours = 1.5  # NEUTRAL signals retried after 1.5h (Phase 22: increased from 0.9h)

        # Phase 3.5: Forgone P&L Engine — tracks every missed signal
        self.forgone_engine = ForgonePnLEngine(db_path=self.db_path)
        # Map pair -> forgone_id for resolving on trade exit
        self._forgone_ids: dict = {}

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
        self._batch_size = 10           # Pairs per batch
        self._batch_interval_secs = 360  # 6 minutes between batches
        self._last_batch_time = 0       # Unix timestamp of last batch

        logger.info("AIFreqtradeSizer initialized with MADAM-RAG, Forgone PNL, Risk Budget, Telegram & Staggered Batching.")

    def bot_loop_start(self, current_time, **kwargs):
        """
        Phase 18: Staggered batch pre-fetch — 10 pairs per batch, 6 min apart.
        100 pairs / 10 per batch = 10 batches × 6min = 60min full cycle.
        Each batch: 10 pairs × 9 LLM calls = 90 calls → 15 calls/min → no rate limit issues.
        """
        if self.dp.runmode.value not in ('dry_run', 'live'):
            return

        import time as _time

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

        from concurrent.futures import ThreadPoolExecutor, as_completed

        t0 = _time.time()

        def fetch_one(p):
            """Fetch signal for one pair via RAG service (Phase 17: POST with technical data).
            Uses class-level HTTP session to prevent fd leak (Errno 24: Too many open files)."""
            sig = {"signal": "NEUTRAL", "confidence": 0.0, "timestamp": current_time}
            try:
                session = AIFreqtradeSizer._get_http_session()
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

                _t = _time.time()
                if technical_data:
                    resp = session.post(f"{url}/signal/{p}", json={"technical_data": technical_data}, timeout=30)
                else:
                    resp = session.get(f"{url}/signal/{p}", timeout=30)
                lat = (_time.time() - _t) * 1000
                logger.info(f"[RAG Latency] {p}: {lat:.0f}ms (status={resp.status_code}, POST={'Y' if technical_data else 'N'})")
                if resp.status_code == 200:
                    parsed = resp.json()
                    sig["signal"] = parsed.get("signal", "NEUTRAL")
                    sig["confidence"] = parsed.get("confidence", 0.0)
                    sig["reasoning"] = parsed.get("reasoning", "")
            except Exception as e:
                logger.warning(f"[bot_loop_start] Fetch failed for {p}: {e}")
            # Cache ALL results (including NEUTRAL) — populate_entry_trend reads from here
            self.ai_signal_cache[p] = sig
            return p, sig

        results = {}
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(fetch_one, p): p for p in current_batch}
            for future in as_completed(futures):
                try:
                    pair, signal = future.result(timeout=45)
                    results[pair] = signal
                except Exception as e:
                    pair = futures[future]
                    logger.warning(f"[bot_loop_start] Timeout for {pair}: {e}")

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
            logger.warning(f"[ENTRY-SIGNAL] {pair}: DETECTED {signal} tag={tag}")
        else:
            if len(dataframe) > 0:
                latest = dataframe.iloc[-1]
                el = latest.get('enter_long', 'N/A')
                xl = latest.get('exit_long', 'N/A')
                es = latest.get('enter_short', 'N/A')
                xs = latest.get('exit_short', 'N/A')
                logger.warning(
                    f"[ENTRY-SIGNAL] {pair}: NO SIGNAL! "
                    f"enter_long={el} exit_long={xl} enter_short={es} exit_short={xs}"
                )
            else:
                logger.warning(f"[ENTRY-SIGNAL] {pair}: EMPTY DATAFRAME!")
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
            "current_price": round(price, 2),
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
                    timeout=30
                )
            else:
                response = requests.get(
                    f"{rag_service_url}/signal/{pair}",
                    timeout=30  # Fast-fail: 20 pairs × 30s = 10min << 60min candle
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

        return dataframe

    def populate_entry_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        pair = metadata['pair']
        df['enter_long'] = 0
        df['enter_short'] = 0

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

            ai_decision = self._get_ai_signal(pair, last_time, dataframe=df)
            signal_type = ai_decision.get('signal', 'NEUTRAL')
            confidence = ai_decision.get('confidence', 0.0)
            is_bullish = signal_type == 'BULLISH'
            is_bearish = signal_type == 'BEARISH'

            # Forgone P&L: Log signal as NOT executed here.
            # Actual execution is confirmed in confirm_trade_entry().
            if signal_type != 'NEUTRAL':
                fid = self.forgone_engine.log_forgone_signal(
                    pair=pair,
                    signal_type="BULL" if is_bullish else "BEAR",
                    confidence=confidence,
                    entry_price=float(current_rate),
                    was_executed=False  # Will be updated in confirm_trade_entry
                )
                if fid:
                    self._forgone_ids[pair] = fid

            # Set entry signals based on AI decision (only last candle)
            # Trade-First: NEUTRAL defaults to enter_long=1 (min_stake sizing handles risk)
            if is_bullish:
                df.iloc[-1, df.columns.get_loc('enter_long')] = 1
                logger.info(f"[Signal] {pair} → enter_long=1 (BULLISH conf={confidence:.2f})")
            elif is_bearish:
                df.iloc[-1, df.columns.get_loc('enter_short')] = 1
                logger.info(f"[Signal] {pair} → enter_short=1 (BEARISH conf={confidence:.2f})")
            else:
                # NEUTRAL: Trade-First philosophy — confidence modulates SIZE not PERMISSION
                df.iloc[-1, df.columns.get_loc('enter_long')] = 1
                logger.info(f"[Signal] {pair} → enter_long=1 (NEUTRAL default, min_stake sizing)")
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

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Dynamic ATR-based stoploss. Sizing manages risk, so we allow wide breathing room
        but cut if trend drastically reverses (e.g. 3x ATR).
        Handles both LONG and SHORT positions correctly.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        if 'atr' in last_candle and current_rate > 0:
            atr = last_candle['atr']

            mult = self.atr_stoploss_mult.value

            if trade.is_short:
                # SHORT: stoploss is ABOVE current price
                stop_price = current_rate + (mult * atr)
                result = -((stop_price / current_rate) - 1)
            else:
                # LONG: stoploss is BELOW current price
                stop_price = current_rate - (mult * atr)
                result = (stop_price / current_rate) - 1

            # Sanity: result must be negative (a loss). If ATR is stale/huge, fall back.
            if result >= 0:
                logger.warning(f"[Stoploss] {pair} ATR-based SL would be >= current price "
                               f"(atr={atr:.4f}, rate={current_rate:.4f}). Using hard stoploss.")
                return self.stoploss

            # Never exceed hard stop of -0.20
            return max(result, self.stoploss)

        return self.stoploss

    def _sync_portfolio_to_ai(self):
        """Bridge: Sync real exchange balance → AI modules (RiskBudget, Autonomy)."""
        try:
            stake = self.config.get('stake_currency', 'USDT')
            total = self.wallets.get_total(stake)
            free = self.wallets.get_free(stake)

            if total <= 0:
                return

            # Update RiskBudget with real portfolio value
            self.risk_budget.update_portfolio_value(total)

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

            self._last_portfolio_sync = current_time if hasattr(self, '_last_portfolio_sync') else datetime.utcnow()
            logger.debug(f"[Portfolio Sync] {stake}: stake=${total:.2f} total_usd=${total_portfolio_usd:.2f} assets={len(all_balances)}")
        except Exception as e:
            logger.debug(f"[Portfolio Sync] Skipped: {e}")

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float, max_stake: float,
                            leverage: float, entry_tag: str, side: str,
                            **kwargs) -> float:
        """
        CORE PRINCIPLE: TRADE-FIRST AUTONOMY (Sizing not blocking).
        Instead of blocking a trade, we scale the size based on FreqAI confidence/market regime.
        """
        logger.warning(
            f"[TRADE-ATTEMPT] custom_stake_amount CALLED: {pair} side={side} "
            f"proposed={proposed_stake:.4f} min={min_stake:.4f} max={max_stake:.4f}"
        )
        # Sync real exchange balance to AI modules (every trade entry)
        self._sync_portfolio_to_ai()

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()

        # Base multiplier
        multiplier = 1.0
        
        # Scale based on sentiment & F&G
        if '%-fng_index' in last_candle:
            fng = last_candle['%-fng_index']
            # If Extreme Greed (> 80) or Extreme Fear (< 20), we reduce stake (contrarian caution)
            if fng > 80 or fng < 20:
                multiplier *= 0.5
                
        if '%-sentiment_24h' in last_candle:
            sent_24h = last_candle['%-sentiment_24h']
            # Positive sentiment gives slight sizing boost
            if sent_24h > 0.5:
                multiplier *= 1.2
            elif sent_24h < -0.5:
                # Still trade, but 70% smaller
                multiplier *= 0.3
                
        # Final Position Sizing math
        # We start with Kelly/Base Stake
        final_stake = proposed_stake * multiplier
        
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
            
            # Görev 1 Fix: Use PositionSizer to calculate fraction, which respects BayesianKelly and Autonomy logic
            fraction = self._position_sizer.calculate_stake_fraction(confidence)
            
            # Let it scale down to "dust" sizes if confidence is terribly low
            final_stake = final_stake * fraction
            
            # Phase 3.5.3: Risk Budget scaling — shrink if budget running low
            final_stake = self.risk_budget.scale_position(final_stake)

            # Autonomy max_stake cap (scales with real portfolio)
            portfolio_val = self.risk_budget.portfolio_value
            autonomy_cap = self.autonomy_manager.get_max_stake(portfolio_value=portfolio_val)
            if autonomy_cap is not None:
                final_stake = min(final_stake, autonomy_cap)

            # Phase 22: Funding rate check — extreme funding = reduce position
            try:
                funding = self.dp.funding_rate(pair)
                if funding and isinstance(funding, dict):
                    fr = funding.get('fundingRate', 0)
                    if fr and abs(fr) > 0.0005:  # >0.05% funding = extreme
                        final_stake *= 0.5  # Halve position on extreme funding
                        logger.info(f"[FundingRate] {pair} extreme funding {fr:.4%}, halving stake")
            except Exception:
                pass

            # Consume budget for this trade
            atr_volatility = last_candle.get('atr', 0.02) / current_rate if current_rate > 0 else 0.02
            self.risk_budget.consume_budget(final_stake, atr_volatility, confidence)

        # Trade-First: ALWAYS trade at least min_stake. Confidence modulates SIZE, never PERMISSION.
        if final_stake < min_stake:
            final_stake = min_stake

        return min(final_stake, max_stake)

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: str,
                            side: str, **kwargs) -> bool:
        """Mark the forgone P&L entry as ACTUALLY executed. Store AI metadata in Trade.custom_data."""
        logger.warning(
            f"[TRADE-ATTEMPT] confirm_trade_entry CALLED: {pair} side={side} "
            f"rate={rate:.6f} stake=${amount*rate:.2f}"
        )
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
                # Snapshot market state at entry for exit comparison
                dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                if dataframe is not None and len(dataframe) > 0:
                    last = dataframe.iloc[-1]
                    trade.set_custom_data("entry_fng", int(last.get('%-fng_index', 50)))
                    trade.set_custom_data("entry_rsi", round(float(last.get('rsi', 50)), 1))
                    trade.set_custom_data("entry_sentiment_24h", round(float(last.get('%-sentiment_24h', 0)), 3))
            except Exception as e:
                logger.debug(f"[custom_data] Failed to store: {e}")

        # Update the existing forgone entry (logged in populate_entry_trend) to was_executed=True
        fid = self._forgone_ids.pop(pair, None)
        if fid:
            self.forgone_engine.mark_executed(fid)

        # Phase 22: Notify via strategy message
        try:
            self.dp.send_msg(
                f"AI Entry: {pair} {signal_type} conf={confidence:.0%} stake=${amount*rate:.2f}"
            )
        except Exception:
            pass

        logger.info(f"[Trade Entry] {pair} {signal_type} conf={confidence:.2f} stake=${amount*rate:.2f} — {reasoning}")
        return True

    def confirm_trade_exit(self, pair: str, trade: 'Trade', order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        """Resolve forgone trades and update Bayesian Kelly with trade outcome."""
        # Forgone P&L resolution
        fid = self._forgone_ids.pop(pair, None)
        if fid:
            self.forgone_engine.resolve_forgone_trade(fid, exit_price=rate)
        
        logger.info(f"[Trade Exit] {pair} reason={exit_reason}")

        # Phase 3.5.2: Bayesian Kelly update — learn from this trade
        try:
            pnl_pct = trade.calc_profit_ratio(rate) if hasattr(trade, 'calc_profit_ratio') else 0.0
            won = pnl_pct > 0
            self._bayesian_kelly.update(won=won, pnl_pct=pnl_pct)
            logger.info(f"[BayesianKelly] Updated: {'WIN' if won else 'LOSS'} pnl={pnl_pct:.4f} → win_p={self._bayesian_kelly.win_probability():.3f} kelly_f={self._bayesian_kelly.kelly_fraction():.4f}")
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
                # Nuclear option: only if account drawdown >25% (very loose)
                "method": "MaxDrawdown",
                "lookback_period_candles": 72,  # 3 days window
                "trade_limit": 20,
                "stop_duration_candles": 4,  # Brief pause, not long lock
                "max_allowed_drawdown": 0.25,  # 25% (was 15%) — testnet, let it breathe
            },
        ]

    def informative_pairs(self):
        """Multi-timeframe + cross-pair data (Phase 22 #4)."""
        stake = self.config.get('stake_currency', 'USDT')
        return [
            (f"BTC/{stake}", "1h"),
            (f"BTC/{stake}", "4h"),
            (f"ETH/{stake}", "4h"),
        ]

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str,
                 side: str, **kwargs) -> float:
        """Dynamic leverage by AI confidence (Phase 22 #14)."""
        ai = self.ai_signal_cache.get(pair, {})
        confidence = ai.get('confidence', 0.0)

        if confidence >= 0.85:
            lev = min(self.leverage_max.value, max_leverage)
        elif confidence >= 0.70:
            lev = min(self.leverage_max.value * 0.7, max_leverage)
        elif confidence >= 0.50:
            lev = min(self.leverage_max.value * 0.5, max_leverage)
        else:
            lev = 1.0

        return max(1.0, round(lev, 1))

    def custom_exit(self, pair: str, trade: 'Trade', current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs) -> str | bool | None:
        """AI-driven exit logic (Phase 22 #13)."""
        if self.dp.runmode.value not in ('dry_run', 'live'):
            return None

        hours_held = (current_time - trade.open_date_utc).total_seconds() / 3600

        # 1. STALE TRADE
        if hours_held > self.stale_trade_hours.value and abs(current_profit) < 0.005:
            return f"stale_{hours_held:.0f}h_flat"

        # 2. SIGNAL REVERSAL
        cached = self.ai_signal_cache.get(pair, {})
        signal = cached.get('signal', 'NEUTRAL')
        confidence = cached.get('confidence', 0.0)

        if not trade.is_short and signal == 'BEARISH' and confidence >= 0.75:
            return f"ai_flip_bearish_{confidence:.0%}"
        if trade.is_short and signal == 'BULLISH' and confidence >= 0.75:
            return f"ai_flip_bullish_{confidence:.0%}"

        # 3. CONFIDENCE DEGRADATION
        entry_conf = trade.get_custom_data("ai_confidence", 0.5)
        if isinstance(entry_conf, (int, float)) and entry_conf > 0.7 and confidence < 0.3:
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

        # 5. FIRST-HOUR CRASH
        if hours_held <= 1.0 and current_profit <= -0.07:
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

        if confidence >= 0.80:
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
        """Orderbook-aware entry pricing (Phase 22 #11)."""
        try:
            ob = self.dp.orderbook(pair, 5)
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
            ob = self.dp.orderbook(pair, 5)
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
        """DCA + partial exit based on AI confidence changes (Phase 22 #16)."""
        if self.dp.runmode.value not in ('dry_run', 'live'):
            return None
        if trade.nr_of_successful_entries >= 4:
            return None

        cached = self.ai_signal_cache.get(trade.pair, {})
        confidence = cached.get('confidence', 0.0)
        signal = cached.get('signal', 'NEUTRAL')
        entry_conf = trade.get_custom_data("ai_confidence", 0.5)
        if not isinstance(entry_conf, (int, float)):
            entry_conf = 0.5

        hours_held = (current_time - trade.open_date_utc).total_seconds() / 3600
        if hours_held < 1.0:
            return None

        # PYRAMID: Confidence up + profitable
        if confidence > 0.80 and current_profit > 0.01 and confidence > entry_conf + 0.1:
            add_stake = max_stake * 0.3
            if min_stake and add_stake >= min_stake:
                logger.info(f"[DCA] {trade.pair} PYRAMID: conf {confidence:.0%}")
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
            ob = self.dp.orderbook(pair, 3)
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
