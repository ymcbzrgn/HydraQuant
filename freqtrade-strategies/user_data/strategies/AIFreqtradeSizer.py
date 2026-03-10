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
    FreqAI-powered strategy focusing on the "Sizing not Blocking" motto.
    Uses LightGBM under the hood and injects real-time SQLite sentiment
    metrics into the feature set.
    """
    
    INTERFACE_VERSION = 3

    # Enable FreqAI
    process_only_new_candles = True
    use_sell_signal = True
    sell_profit_only = False
    ignore_roi_if_buy_signal = False
    startup_candle_count = 200

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

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self.db_path = os.path.join(self.config['user_data_dir'], "db", "ai_data.sqlite")
        self.rag_script_path = os.path.join(self.config['user_data_dir'], "scripts", "rag_graph.py")
        self.ai_signal_cache = {} # Memory cache: { "BTC/USDT": {"signal": "BULLISH", "confidence": 0.8, "timestamp": datetime} }
        self.cache_ttl_hours = 4 # The LLM decision is valid for 4 hours unless trend sharply breaks
        
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
        
        logger.info("AIFreqtradeSizer initialized with MADAM-RAG, Forgone PNL, Risk Budget & Telegram.")

    def _get_sqlite_connection(self):
        try:
            conn = sqlite3.connect(self.db_path)
            return conn
        except sqlite3.Error as e:
            logger.error(f"Error connecting to AI SQLite DB: {e}")
            return None

    def _get_ai_signal(self, pair: str, current_time: datetime) -> dict:
        """
        The Bridge (Phase 5.1): Asks the LangGraph Trader Agent for a decision.
        Uses in-memory cache to prevent spamming the Gemini API on every candle.
        """
        # 1. Check Memory Cache
        cached = self.ai_signal_cache.get(pair)
        if cached:
            time_diff = (current_time - cached['timestamp']).total_seconds() / 3600
            if time_diff < self.cache_ttl_hours:
                return cached
                
        # 2. Cache Miss -> Trigger LangGraph (Subprocess blocks for ~5-10 seconds per pair occasionally)
        logger.info(f"AI Signal Cache Miss for {pair}. Asking LangGraph Trader Agent...")
        signal_data = {"signal": "NEUTRAL", "confidence": 0.0, "timestamp": current_time}
        
        try:
            import subprocess
            import json
            import sys
            
            # Call rag_graph.py as a separate process to avoid complex asyncio loop collisions
            result = subprocess.run(
                [sys.executable, self.rag_script_path, f"--pair={pair}"],
                capture_output=True,
                text=True,
                check=True
            )
            
            # Parse the JSON injected at the end of stdout
            output = result.stdout
            if "--- JSON OUTPUT ---" in output:
                json_str = output.split("--- JSON OUTPUT ---")[1].strip()
                parsed = json.loads(json_str)
                signal_data["signal"] = parsed.get("signal", "NEUTRAL")
                signal_data["confidence"] = parsed.get("confidence", 0.0)
                logger.info(f"LangGraph Agent decided: {signal_data['signal']} ({signal_data['confidence']}) for {pair}")
            else:
                logger.warning(f"Failed to find JSON output from Agent for {pair}")
                
        except Exception as e:
            logger.error(f"Error running LangGraph Trader Agent for {pair}: {e}")
            
        # 3. Save to Cache
        self.ai_signal_cache[pair] = signal_data
        return signal_data

    def feature_engineering_expand_all(self, dataframe: pd.DataFrame, period: int,
                                       metadata: dict, **kwargs) -> pd.DataFrame:
        """
        Populate features for the AI model. Includes technicals AND our custom Sentiment DB metrics.
        """
        # --- 1. Technical Features ---
        dataframe['%-rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['%-mfi'] = ta.MFI(dataframe, timeperiod=14)
        dataframe['%-adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['%-sma'] = ta.SMA(dataframe, timeperiod=20)
        dataframe['%-ema'] = ta.EMA(dataframe, timeperiod=20)
        
        # Bollinger Bands
        bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe['%-bb_lowerband'] = bollinger['lowerband']
        dataframe['%-bb_middleband'] = bollinger['middleband']
        dataframe['%-bb_upperband'] = bollinger['upperband']

        macd = ta.MACD(dataframe)
        dataframe['%-macd'] = macd['macd']
        dataframe['%-macdsignal'] = macd['macdsignal']
        dataframe['%-macdhist'] = macd['macdhist']

        # --- 2. Sentiment Features (DB Integration) ---
        conn = self._get_sqlite_connection()
        if conn:
            pair = metadata['pair']
            base_coin = pair.split('/')[0] # e.g. BTC from BTC/USDT
            
            # Fetch Fear and Greed Index
            try:
                fng_df = pd.read_sql_query("SELECT value as fng_value FROM fear_and_greed ORDER BY timestamp DESC LIMIT 1", conn)
                current_fng = fng_df['fng_value'].iloc[0] if not fng_df.empty else 50
                dataframe['%-fng_index'] = current_fng
            except Exception as e:
                logger.error(f"F&G fetch failed: {e}")
                dataframe['%-fng_index'] = 50

            # Fetch Coin Rolling Sentiment
            try:
                sent_df = pd.read_sql_query("SELECT sentiment_1h, sentiment_4h, sentiment_24h FROM coin_sentiment_rolling WHERE coin = ? ORDER BY timestamp DESC LIMIT 1", conn, params=(base_coin,))
                if not sent_df.empty:
                    dataframe['%-sentiment_1h'] = sent_df['sentiment_1h'].iloc[0]
                    dataframe['%-sentiment_4h'] = sent_df['sentiment_4h'].iloc[0]
                    dataframe['%-sentiment_24h'] = sent_df['sentiment_24h'].iloc[0]
                else:
                    dataframe['%-sentiment_1h'] = 0.0
                    dataframe['%-sentiment_4h'] = 0.0
                    dataframe['%-sentiment_24h'] = 0.0
            except Exception as e:
                logger.error(f"Sentiment fetch failed: {e}")
                dataframe['%-sentiment_1h'] = 0.0
                dataframe['%-sentiment_4h'] = 0.0
                dataframe['%-sentiment_24h'] = 0.0

            conn.close()
        else:
            # Fallback if DB is locked/missing
            dataframe['%-fng_index'] = 50
            dataframe['%-sentiment_1h'] = 0.0
            dataframe['%-sentiment_4h'] = 0.0
            dataframe['%-sentiment_24h'] = 0.0

        return dataframe

    def feature_engineering_expand_basic(self, dataframe: pd.DataFrame, metadata: dict, **kwargs) -> pd.DataFrame:
        """Add non-shiftable features for FreqAI"""
        dataframe['%-day_of_week'] = dataframe['date'].dt.dayofweek
        dataframe['%-hour_of_day'] = dataframe['date'].dt.hour
        return dataframe

    def feature_engineering_standard(self, dataframe: pd.DataFrame, metadata: dict, **kwargs) -> pd.DataFrame:
        """Standardizing features (already handled largely by FreqAI)"""
        return dataframe

    def set_freqai_targets(self, dataframe: pd.DataFrame, metadata: dict, **kwargs) -> pd.DataFrame:
        """
        Define what the model should predict.
        Here we predict the max high and min low in the next 10 candles.
        """
        dataframe['&s-up_or_down'] = np.where(dataframe['close'].shift(-10) > dataframe['close'], "up", "down")
        return dataframe

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe = self.freqai.start(dataframe, metadata, self)
        
        # Basic indicators for stoploss/sizing logic outside AI
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        return dataframe

    def populate_entry_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        enter_long = np.zeros(len(df), dtype=int)
        pair = metadata['pair']
        
        # Check if FreqAI predictions are available
        if "do_predict" in df.columns:
            # Sizing not blocking: If Technicals (AI) predicts "up", we ask the RAG Brain.
            tech_up = (df['do_predict'] == 1) & (df['&s-up_or_down'] == "up")
            
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
                
                ai_decision = self._get_ai_signal(pair, last_time)
                
                is_bullish = ai_decision['signal'] == 'BULLISH'
                confidence = ai_decision.get('confidence', 0.0)
                signal_type = ai_decision.get('signal', 'NEUTRAL')
                
                # Forgone P&L: Log EVERY AI signal — executed or not
                if signal_type != 'NEUTRAL':
                    was_executed = bool(tech_up.iloc[-1]) and is_bullish
                    fid = self.forgone_engine.log_forgone_signal(
                        pair=pair,
                        signal_type="BULL" if signal_type == "BULLISH" else "BEAR",
                        confidence=confidence,
                        entry_price=float(current_rate),
                        was_executed=was_executed
                    )
                    if fid and not was_executed:
                        # Store for later resolution
                        self._forgone_ids[pair] = fid
                
                enter_long = tech_up & is_bullish
            else:
                # Backtesting: Ignore RAG graph delay, just follow tech
                enter_long = tech_up
            
        df['enter_long'] = enter_long
        return df

    def populate_exit_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        exit_long = np.zeros(len(df), dtype=int)
        
        if "do_predict" in df.columns:
            # Exit if AI changes its mind to down
            exit_long = (df['do_predict'] == 1) & (df['&s-up_or_down'] == "down")
            
        df['exit_long'] = exit_long
        return df

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Dynamic ATR-based stoploss. Sizing manages risk, so we allow wide breathing room
        but cut if trend drastically reverses (e.g. 3x ATR).
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        
        if 'atr' in last_candle:
            atr = last_candle['atr']
            # Allow 3 ATRs of breathing room relative to current rate
            new_stop = (current_rate - (3 * atr)) / current_rate
            result = new_stop - 1
            # Never exceed hard stop of -0.20
            return max(result, self.stoploss)
            
        return self.stoploss

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float, max_stake: float,
                            leverage: float, entry_tag: str, side: str,
                            **kwargs) -> float:
        """
        CORE PRINCIPLE: TRADE-FIRST AUTONOMY (Sizing not blocking).
        Instead of blocking a trade, we scale the size based on FreqAI confidence/market regime.
        """
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
            
            # Consume budget for this trade
            atr_volatility = last_candle.get('atr', 0.02) / current_rate if current_rate > 0 else 0.02
            self.risk_budget.consume_budget(final_stake, atr_volatility, confidence)
            
        if final_stake < min_stake:
            # To honor exchange API limits, if dust is too small, Freqtrade rejects it internally.
            pass

        return min(final_stake, max_stake)

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                            time_in_force: str, current_time: datetime, entry_tag: str,
                            side: str, **kwargs) -> bool:
        """Log executed trade to Forgone P&L for comparison tracking."""
        ai_decision = self.ai_signal_cache.get(pair, {})
        confidence = ai_decision.get('confidence', 0.5)
        signal_type = "BULL" if side == "long" else "BEAR"
        reasoning = ai_decision.get('reasoning', "Technical entry with AI confirmation")
        
        self.forgone_engine.log_forgone_signal(
            pair=pair,
            signal_type=signal_type,
            confidence=confidence,
            entry_price=rate,
            was_executed=True
        )
        if self._telegram:
            self._telegram.send_trade_signal(
                pair=pair, 
                signal="long" if side == "long" else "short", 
                confidence=confidence, 
                reasoning_summary=reasoning
            )
        return True

    def confirm_trade_exit(self, pair: str, trade: 'Trade', order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:
        """Resolve forgone trades and update Bayesian Kelly with trade outcome."""
        # Forgone P&L resolution
        fid = self._forgone_ids.pop(pair, None)
        if fid:
            self.forgone_engine.resolve_forgone_trade(fid, exit_price=rate)
        
        # Telegram notification
        if self._telegram:
            try:
                pnl_pct = trade.calc_profit_ratio(rate) * 100 if hasattr(trade, 'calc_profit_ratio') else 0.0
                pnl_abs = trade.calc_profit(rate) if hasattr(trade, 'calc_profit') else 0.0
                sign = "+" if pnl_pct > 0 else ""
                self._telegram.send_alert(f"Trade Exited: {pair}\nReason: {exit_reason}\nPNL: {sign}${pnl_abs:.2f} ({sign}{pnl_pct:.2f}%)", level="INFO")
            except Exception:
                pass

        # Phase 3.5.2: Bayesian Kelly update — learn from this trade
        try:
            pnl_pct = trade.calc_profit_ratio(rate) if hasattr(trade, 'calc_profit_ratio') else 0.0
            won = pnl_pct > 0
            # Eagerly initialized in __init__
            self._bayesian_kelly.update(won=won, pnl_pct=pnl_pct)
            logger.info(f"[BayesianKelly] Updated: {'WIN' if won else 'LOSS'} pnl={pnl_pct:.4f} → win_p={self._bayesian_kelly.win_probability():.3f} kelly_f={self._bayesian_kelly.kelly_fraction():.4f}")
        except Exception as e:
            logger.warning(f"[BayesianKelly] Update failed: {e}")
        
        return True
