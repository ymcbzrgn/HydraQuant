# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
import logging
from datetime import datetime
from typing import Dict, Any, Callable, List
import pandas as pd
import numpy as np
import talib.abstract as ta

from freqtrade.strategy import IStrategy

class BaselineTechnical(IStrategy):
    """
    Baseline Technical Strategy for Backtest Comparison.
    Uses the exact same technical indicators as AIFreqtradeSizer 
    but with NO AI, NO RAG, NO Autonomy scaling, and NO sentiment.
    """
    
    INTERFACE_VERSION = 3
    
    # Needs to match AIFreqtradeSizer
    timeframe = '1h'
    
    can_short: bool = False

    minimal_roi = {
        "0": 0.15,
        "120": 0.05,
        "240": 0.02,
        "480": 0.0
    }
    
    stoploss = -0.10
    trailing_stop = False
    
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    
    startup_candle_count: int = 20

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Populate the exact same technicals used by AIFreqtradeSizer's feature expansion.
        """
        # RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        
        # MFI
        dataframe['mfi'] = ta.MFI(dataframe, timeperiod=14)
        
        # ADX
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        
        # SMA / EMA
        dataframe['sma'] = ta.SMA(dataframe, timeperiod=20)
        dataframe['ema'] = ta.EMA(dataframe, timeperiod=20)
        
        # Bollinger Bands
        bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe['bb_lowerband'] = bollinger['lowerband']
        dataframe['bb_middleband'] = bollinger['middleband']
        dataframe['bb_upperband'] = bollinger['upperband']
        
        # MACD
        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        
        # ATR for dynamic stoploss
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Basic technical conditions that mimic the "up" target of the FreqAI model.
        Since we don't have the AI predicting '&s-up_or_down', we use standard TA.
        """
        dataframe.loc[
            (
                (dataframe['rsi'] < 30) & # Oversold
                (dataframe['close'] > dataframe['bb_lowerband']) & # Bouncing from lower band
                (dataframe['macd'] > dataframe['macdsignal']) & # MACD crossing up
                (dataframe['volume'] > 0)
            ),
            'enter_long'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Basic exit conditions.
        """
        dataframe.loc[
            (
                (dataframe['rsi'] > 70) & # Overbought
                (dataframe['close'] < dataframe['bb_upperband']) & # Rejecting upper band
                (dataframe['macd'] < dataframe['macdsignal']) & # MACD crossing down
                (dataframe['volume'] > 0)
            ),
            'exit_long'] = 1
            
        return dataframe

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float, max_stake: float,
                            leverage: float, entry_tag: str, side: str,
                            **kwargs) -> float:
        """
        Static 2% position sizing without ANY Autonomy or Kelly fractions.
        """
        return proposed_stake
        
    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Dynamic ATR-based stoploss, same as AIFreqtradeSizer minus the sizing awareness.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        
        if 'atr' in last_candle:
            atr = last_candle['atr']
            new_stop = (current_rate - (3 * atr)) / current_rate
            result = new_stop - 1
            return max(result, self.stoploss)
            
        return self.stoploss
