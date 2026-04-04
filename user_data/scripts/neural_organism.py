"""
Phase 24: Neural Organism — Self-learning, brain-inspired parameter system.

546 hardcoded parameters become 546 neurons in a living organism.
Each neuron learns from trade outcomes via Thompson Sampling (Beta distribution).
8 brain subsystems coordinate: Hormones, Amygdala, Hippocampus, Synapses,
PrefrontalCortex, BasalGanglia, Proprioception, ImmuneMemory.

Usage (consumer files):
    from neural_organism import get_organism
    org = get_organism()
    value = org.get_param("evidence.q3.fng_extreme_low", regime="trending_bear")

    # Or via helper:
    from neural_organism import _p
    if fng < _p("evidence.q3.fng_extreme_low", 10):
        ...
"""

import os
import sys
import json
import math
import time
import random
import threading
import sqlite3
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple

sys.path.append(os.path.dirname(__file__))
from ai_config import AI_DB_PATH

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════

REGIMES = [
    "trending_bull", "trending_bear", "ranging",
    "high_volatility", "transitional", "_global",
]

# Metabolic decay rates per regime (hourly alpha/beta *= factor)
DECAY_BY_REGIME = {
    "high_volatility": 0.990,
    "transitional":    0.993,
    "trending_bear":   0.995,
    "ranging":         0.997,
    "trending_bull":   0.998,
    "_global":         0.996,
}


# ═══════════════════════════════════════════════════════════════════════════
# PARAM_REGISTRY — Every learnable parameter definition
# Format: param_id → {"organ", "default", "min", "max", "regime_defaults"?}
# ═══════════════════════════════════════════════════════════════════════════

PARAM_REGISTRY: Dict[str, dict] = {
    # ─── ORGAN: evidence_weights (6 params, SUM=1.0) — evidence_engine.py:85-92 ───
    "evidence.weights.q1_trend":    {"organ": "evidence_weights", "default": 0.22, "min": 0.05, "max": 0.40,
                                     "regime_defaults": {"ranging": 0.10, "high_volatility": 0.15}},
    "evidence.weights.q2_momentum": {"organ": "evidence_weights", "default": 0.20, "min": 0.05, "max": 0.35,
                                     "regime_defaults": {"ranging": 0.15, "high_volatility": 0.13}},
    "evidence.weights.q3_crowd":    {"organ": "evidence_weights", "default": 0.22, "min": 0.05, "max": 0.40,
                                     "regime_defaults": {"ranging": 0.28, "high_volatility": 0.22}},
    "evidence.weights.q4_evidence": {"organ": "evidence_weights", "default": 0.15, "min": 0.02, "max": 0.35,
                                     "regime_defaults": {"ranging": 0.28, "high_volatility": 0.18}},
    "evidence.weights.q5_macro":    {"organ": "evidence_weights", "default": 0.10, "min": 0.02, "max": 0.25,
                                     "regime_defaults": {"ranging": 0.08, "high_volatility": 0.08}},
    "evidence.weights.q6_risk":     {"organ": "evidence_weights", "default": 0.11, "min": 0.02, "max": 0.25,
                                     "regime_defaults": {"ranging": 0.11, "high_volatility": 0.24}},

    # ─── ORGAN: q1_trend_scoring (7 params) — evidence_engine.py:558-574 ───
    "evidence.q1.adx_strong_threshold": {"organ": "q1_trend", "default": 25, "min": 18, "max": 35},
    "evidence.q1.full_bullish_strong":  {"organ": "q1_trend", "default": 0.85, "min": 0.70, "max": 0.95},
    "evidence.q1.full_bullish":         {"organ": "q1_trend", "default": 0.70, "min": 0.55, "max": 0.85},
    "evidence.q1.full_bearish_strong":  {"organ": "q1_trend", "default": 0.15, "min": 0.05, "max": 0.30},
    "evidence.q1.full_bearish":         {"organ": "q1_trend", "default": 0.30, "min": 0.15, "max": 0.45},
    "evidence.q1.above_200":            {"organ": "q1_trend", "default": 0.62, "min": 0.52, "max": 0.72},
    "evidence.q1.below_200":            {"organ": "q1_trend", "default": 0.38, "min": 0.28, "max": 0.48},

    # ─── ORGAN: q2_momentum_scoring (8 params) — evidence_engine.py:586-621 ───
    "evidence.q2.rsi_momentum_zone":  {"organ": "q2_momentum", "default": 0.75, "min": 0.60, "max": 0.90},
    "evidence.q2.rsi_overbought":     {"organ": "q2_momentum", "default": 0.30, "min": 0.15, "max": 0.45},
    "evidence.q2.rsi_weak":           {"organ": "q2_momentum", "default": 0.42, "min": 0.30, "max": 0.55},
    "evidence.q2.rsi_oversold":       {"organ": "q2_momentum", "default": 0.55, "min": 0.40, "max": 0.70},
    "evidence.q2.macd_bullish":       {"organ": "q2_momentum", "default": 0.65, "min": 0.55, "max": 0.80},
    "evidence.q2.macd_bearish":       {"organ": "q2_momentum", "default": 0.35, "min": 0.20, "max": 0.45},
    "evidence.q2.htf_aligned":        {"organ": "q2_momentum", "default": 0.70, "min": 0.55, "max": 0.85},
    "evidence.q2.blend_rsi_w":        {"organ": "q2_momentum", "default": 0.50, "min": 0.30, "max": 0.70},

    # ─── ORGAN: q3_crowd (14 params — $400 KAYBIN ANA NEDENİ) — evidence_engine.py:631-671 ───
    "evidence.q3.fng_extreme_low":     {"organ": "q3_crowd", "default": 10,    "min": 5,     "max": 25},
    "evidence.q3.fng_fear":            {"organ": "q3_crowd", "default": 20,    "min": 10,    "max": 35},
    "evidence.q3.fng_mild_fear":       {"organ": "q3_crowd", "default": 30,    "min": 20,    "max": 45},
    "evidence.q3.fng_extreme_high":    {"organ": "q3_crowd", "default": 85,    "min": 70,    "max": 95},
    "evidence.q3.fng_greed":           {"organ": "q3_crowd", "default": 75,    "min": 60,    "max": 90},
    "evidence.q3.fng_mild_greed":      {"organ": "q3_crowd", "default": 65,    "min": 50,    "max": 80},
    "evidence.q3.fng_adj_extreme":     {"organ": "q3_crowd", "default": 0.22,  "min": 0.10,  "max": 0.35},
    "evidence.q3.fng_adj_fear":        {"organ": "q3_crowd", "default": 0.15,  "min": 0.05,  "max": 0.25},
    "evidence.q3.fng_adj_mild":        {"organ": "q3_crowd", "default": 0.06,  "min": 0.02,  "max": 0.15},
    "evidence.q3.funding_extreme":     {"organ": "q3_crowd", "default": 0.001, "min": 0.0003, "max": 0.003},
    "evidence.q3.funding_high":        {"organ": "q3_crowd", "default": 0.0005, "min": 0.0002, "max": 0.002},
    "evidence.q3.funding_moderate":    {"organ": "q3_crowd", "default": 0.0003, "min": 0.00005, "max": 0.0008},
    "evidence.q3.funding_adj_extreme": {"organ": "q3_crowd", "default": 0.18,  "min": 0.08,  "max": 0.30},
    "evidence.q3.funding_adj_high":    {"organ": "q3_crowd", "default": 0.12,  "min": 0.05,  "max": 0.20},
    "evidence.q3.funding_adj_mod":     {"organ": "q3_crowd", "default": 0.06,  "min": 0.02,  "max": 0.12},
    "evidence.q3.ls_crowded_long":     {"organ": "q3_crowd", "default": 1.5,   "min": 1.2,   "max": 2.5},
    "evidence.q3.ls_crowded_short":    {"organ": "q3_crowd", "default": 0.7,   "min": 0.3,   "max": 0.9},
    "evidence.q3.ls_adj":              {"organ": "q3_crowd", "default": 0.08,  "min": 0.03,  "max": 0.15},

    # ─── ORGAN: q4_evidence_scoring (12 params) — evidence_engine.py:679-724 ───
    "evidence.q4.knn_strong_bull":    {"organ": "q4_evidence", "default": 0.65, "min": 0.55, "max": 0.80},
    "evidence.q4.knn_mild_bull":      {"organ": "q4_evidence", "default": 0.55, "min": 0.50, "max": 0.65},
    "evidence.q4.knn_strong_bear":    {"organ": "q4_evidence", "default": 0.35, "min": 0.20, "max": 0.45},
    "evidence.q4.knn_mild_bear":      {"organ": "q4_evidence", "default": 0.45, "min": 0.35, "max": 0.50},
    "evidence.q4.knn_adj_strong":     {"organ": "q4_evidence", "default": 0.15, "min": 0.05, "max": 0.25},
    "evidence.q4.knn_adj_mild":       {"organ": "q4_evidence", "default": 0.07, "min": 0.02, "max": 0.12},
    "evidence.q4.knn_dist_bonus":     {"organ": "q4_evidence", "default": 0.05, "min": 0.01, "max": 0.10},
    "evidence.q4.bt_wr_good":         {"organ": "q4_evidence", "default": 0.60, "min": 0.50, "max": 0.75},
    "evidence.q4.bt_wr_bad":          {"organ": "q4_evidence", "default": 0.40, "min": 0.30, "max": 0.50},
    "evidence.q4.bt_adj":             {"organ": "q4_evidence", "default": 0.12, "min": 0.05, "max": 0.20},
    "evidence.q4.pf_good":            {"organ": "q4_evidence", "default": 2.0,  "min": 1.2,  "max": 3.0},
    "evidence.q4.pf_adj":             {"organ": "q4_evidence", "default": 0.08, "min": 0.03, "max": 0.15},

    # ─── ORGAN: q5_macro_scoring (8 params) — evidence_engine.py:738-764 ───
    "evidence.q5.dxy_threshold":  {"organ": "q5_macro", "default": 0.3,  "min": 0.1,  "max": 0.8},
    "evidence.q5.dxy_adj":        {"organ": "q5_macro", "default": 0.10, "min": 0.03, "max": 0.20},
    "evidence.q5.vix_high":       {"organ": "q5_macro", "default": 30,   "min": 20,   "max": 40},
    "evidence.q5.vix_low":        {"organ": "q5_macro", "default": 15,   "min": 10,   "max": 22},
    "evidence.q5.vix_adj_high":   {"organ": "q5_macro", "default": 0.08, "min": 0.03, "max": 0.15},
    "evidence.q5.vix_adj_low":    {"organ": "q5_macro", "default": 0.05, "min": 0.02, "max": 0.10},
    "evidence.q5.btcdom_high":    {"organ": "q5_macro", "default": 58,   "min": 50,   "max": 70},
    "evidence.q5.btcdom_low":     {"organ": "q5_macro", "default": 45,   "min": 35,   "max": 55},

    # ─── ORGAN: q6_risk_scoring (6 params) — evidence_engine.py:777-798 ───
    "evidence.q6.atr_very_high":  {"organ": "q6_risk", "default": 4.0,  "min": 2.5,  "max": 6.0},
    "evidence.q6.atr_high":       {"organ": "q6_risk", "default": 3.0,  "min": 2.0,  "max": 5.0},
    "evidence.q6.atr_low":        {"organ": "q6_risk", "default": 1.5,  "min": 0.8,  "max": 2.5},
    "evidence.q6.atr_adj_high":   {"organ": "q6_risk", "default": 0.15, "min": 0.05, "max": 0.25},
    "evidence.q6.knn_worst_thr":  {"organ": "q6_risk", "default": -5.0, "min": -10.0, "max": -2.0},
    "evidence.q6.vol_adj":        {"organ": "q6_risk", "default": 0.05, "min": 0.02, "max": 0.10},

    # ─── ORGAN: synthesis (12 params) — evidence_engine.py:922-1013 ───
    "evidence.synthesis.bullish_threshold":  {"organ": "synthesis", "default": 0.53, "min": 0.51, "max": 0.60},
    "evidence.synthesis.bearish_threshold":  {"organ": "synthesis", "default": 0.47, "min": 0.40, "max": 0.49},
    "evidence.synthesis.k_base":             {"organ": "synthesis", "default": 7.0,  "min": 4.0,  "max": 10.0},
    "evidence.synthesis.k_range":            {"organ": "synthesis", "default": 5.0,  "min": 2.0,  "max": 8.0},
    "evidence.synthesis.alignment_scale":    {"organ": "synthesis", "default": 5.0,  "min": 2.0,  "max": 8.0},
    "evidence.synthesis.uncertainty_floor":  {"organ": "synthesis", "default": 0.75, "min": 0.50, "max": 0.90},
    "evidence.synthesis.cap_5":              {"organ": "synthesis", "default": 0.90, "min": 0.75, "max": 0.95},
    "evidence.synthesis.cap_3":              {"organ": "synthesis", "default": 0.80, "min": 0.65, "max": 0.90},
    "evidence.synthesis.cap_1":              {"organ": "synthesis", "default": 0.70, "min": 0.55, "max": 0.85},
    "evidence.synthesis.cap_0":              {"organ": "synthesis", "default": 0.55, "min": 0.40, "max": 0.70},
    "evidence.synthesis.contradiction_per":  {"organ": "synthesis", "default": 0.05, "min": 0.02, "max": 0.10},
    "evidence.synthesis.contradiction_max":  {"organ": "synthesis", "default": 0.15, "min": 0.08, "max": 0.25},

    # ─── ORGAN: contradiction (6 params) — evidence_engine.py:811-839 ───
    "evidence.contradiction.bullish_thr":  {"organ": "contradiction", "default": 0.65, "min": 0.55, "max": 0.80},
    "evidence.contradiction.bearish_thr":  {"organ": "contradiction", "default": 0.35, "min": 0.20, "max": 0.45},
    "evidence.contradiction.groupthink_hi": {"organ": "contradiction", "default": 0.60, "min": 0.55, "max": 0.75},
    "evidence.contradiction.groupthink_lo": {"organ": "contradiction", "default": 0.40, "min": 0.25, "max": 0.45},
    "evidence.contradiction.db_only_pen":  {"organ": "contradiction", "default": 0.50, "min": 0.30, "max": 0.70},
    "evidence.contradiction.db_only_cap":  {"organ": "contradiction", "default": 0.35, "min": 0.25, "max": 0.50},

    # ─── ORGAN: evidence_first (3 params) — rag_graph.py:1721 ───
    "rag.evidence_first_threshold": {"organ": "evidence_first", "default": 0.40, "min": 0.25, "max": 0.65},
    "rag.bg_madam_timeout":         {"organ": "evidence_first", "default": 120,  "min": 30,   "max": 300},
    "rag.fg_madam_timeout":         {"organ": "evidence_first", "default": 30,   "min": 10,   "max": 60},

    # ─── ORGAN: coordinator_weights (5 params) — rag_graph.py:1394-1398 ───
    "rag.coord.data_quality":       {"organ": "coordinator", "default": 0.20, "min": 0.10, "max": 0.35},
    "rag.coord.signal_strength":    {"organ": "coordinator", "default": 0.30, "min": 0.15, "max": 0.45},
    "rag.coord.regime_confidence":  {"organ": "coordinator", "default": 0.20, "min": 0.10, "max": 0.35},
    "rag.coord.analyst_agreement":  {"organ": "coordinator", "default": 0.15, "min": 0.05, "max": 0.30},
    "rag.coord.backtest_alignment": {"organ": "coordinator", "default": 0.15, "min": 0.05, "max": 0.30},

    # ─── ORGAN: sizing (6 params) — position_sizer.py:121-175 ───
    "sizing.max_risk":              {"organ": "sizing", "default": 0.05, "min": 0.01, "max": 0.10},
    "sizing.confidence_exponent":   {"organ": "sizing", "default": 1.5,  "min": 1.0,  "max": 3.0},
    "sizing.kelly_cap":             {"organ": "sizing", "default": 0.25, "min": 0.10, "max": 0.50},
    "sizing.min_fraction_mult":     {"organ": "sizing", "default": 0.01, "min": 0.001, "max": 0.05},
    "sizing.max_fraction_mult":     {"organ": "sizing", "default": 1.5,  "min": 1.0,  "max": 2.0},
    "sizing.equal_risk_pct":        {"organ": "sizing", "default": 0.005, "min": 0.001, "max": 0.02},

    # ─── ORGAN: risk (8 params) — risk_budget.py:32-169 ───
    "risk.daily_var_pct":           {"organ": "risk", "default": 0.50, "min": 0.10, "max": 0.80},
    "risk.budget_cap_fraction":     {"organ": "risk", "default": 0.25, "min": 0.10, "max": 0.50},
    "risk.dust_fraction":           {"organ": "risk", "default": 0.01, "min": 0.005, "max": 0.05},
    "risk.dust_min_usd":            {"organ": "risk", "default": 1.0,  "min": 0.5,  "max": 5.0},
    "risk.weekly_mult_max":         {"organ": "risk", "default": 2.0,  "min": 1.5,  "max": 3.0},
    "risk.weekly_mult_min":         {"organ": "risk", "default": 0.5,  "min": 0.2,  "max": 0.8},
    "risk.weekly_win_mult":         {"organ": "risk", "default": 1.1,  "min": 1.0,  "max": 1.3},
    "risk.weekly_loss_mult":        {"organ": "risk", "default": 0.8,  "min": 0.5,  "max": 0.95},

    # ─── ORGAN: strategy_stoploss (8 params) — AIFreqtradeSizer.py:862-917 ───
    "strategy.stoploss_floor":        {"organ": "strategy_stoploss", "default": -0.15, "min": -0.30, "max": -0.05},
    "strategy.chandelier_high_conf":  {"organ": "strategy_stoploss", "default": 0.80, "min": 0.65, "max": 0.90},
    "strategy.chandelier_med_conf":   {"organ": "strategy_stoploss", "default": 0.60, "min": 0.45, "max": 0.75},
    "strategy.chandelier_atr_high":   {"organ": "strategy_stoploss", "default": 3.0,  "min": 2.0,  "max": 4.0},
    "strategy.chandelier_atr_med":    {"organ": "strategy_stoploss", "default": 2.5,  "min": 1.5,  "max": 3.5},
    "strategy.chandelier_atr_low":    {"organ": "strategy_stoploss", "default": 2.0,  "min": 1.0,  "max": 3.0},
    "strategy.breakeven_long":        {"organ": "strategy_stoploss", "default": 0.998, "min": 0.990, "max": 1.000},
    "strategy.breakeven_short":       {"organ": "strategy_stoploss", "default": 1.002, "min": 1.000, "max": 1.010},

    # ─── ORGAN: strategy_leverage (8 params) — AIFreqtradeSizer.py:1396-1438 ───
    "strategy.leverage_max":          {"organ": "strategy_leverage", "default": 3.0, "min": 1.0, "max": 5.0},
    "strategy.leverage_ranging_max":  {"organ": "strategy_leverage", "default": 2.0, "min": 1.0, "max": 3.0},
    "strategy.leverage_bear_max":     {"organ": "strategy_leverage", "default": 1.5, "min": 1.0, "max": 2.5},
    "strategy.leverage_conf_high":    {"organ": "strategy_leverage", "default": 0.75, "min": 0.60, "max": 0.90},
    "strategy.leverage_conf_med":     {"organ": "strategy_leverage", "default": 0.60, "min": 0.45, "max": 0.75},
    "strategy.leverage_conf_low":     {"organ": "strategy_leverage", "default": 0.45, "min": 0.30, "max": 0.60},
    "strategy.leverage_mult_high":    {"organ": "strategy_leverage", "default": 1.0, "min": 0.7, "max": 1.0},
    "strategy.leverage_mult_low":     {"organ": "strategy_leverage", "default": 0.5, "min": 0.3, "max": 0.8},

    # ─── ORGAN: strategy_exit (8 params) — AIFreqtradeSizer.py:1449-1498 ───
    "strategy.stale_trade_hours":     {"organ": "strategy_exit", "default": 8,    "min": 4,    "max": 24},
    "strategy.stale_flat_pct":        {"organ": "strategy_exit", "default": 0.005, "min": 0.001, "max": 0.02},
    "strategy.flip_exit_conf":        {"organ": "strategy_exit", "default": 0.55, "min": 0.40, "max": 0.70},
    "strategy.conf_degrade_exit":     {"organ": "strategy_exit", "default": 0.30, "min": 0.15, "max": 0.50},
    "strategy.first_hour_atr_mult":   {"organ": "strategy_exit", "default": 2.5,  "min": 1.5,  "max": 4.0},
    "strategy.first_hour_max":        {"organ": "strategy_exit", "default": -0.15, "min": -0.25, "max": -0.05},
    "strategy.first_hour_min":        {"organ": "strategy_exit", "default": -0.03, "min": -0.07, "max": -0.01},
    "strategy.max_equity_loss":       {"organ": "strategy_exit", "default": 0.15, "min": 0.05, "max": 0.25},

    # ─── ORGAN: strategy_roi (5 params) — AIFreqtradeSizer.py:44-49 ───
    "strategy.roi_0":    {"organ": "strategy_roi", "default": 0.15,  "min": 0.08,  "max": 0.30},
    "strategy.roi_60":   {"organ": "strategy_roi", "default": 0.05,  "min": 0.02,  "max": 0.10},
    "strategy.roi_120":  {"organ": "strategy_roi", "default": 0.03,  "min": 0.01,  "max": 0.08},
    "strategy.roi_360":  {"organ": "strategy_roi", "default": 0.015, "min": 0.005, "max": 0.05},
    "strategy.roi_720":  {"organ": "strategy_roi", "default": 0.005, "min": 0.001, "max": 0.02},

    # ─── ORGAN: strategy_protection (6 params) — AIFreqtradeSizer.py:1357-1376 ───
    "strategy.cooldown_candles":      {"organ": "strategy_protection", "default": 1,   "min": 1,   "max": 5},
    "strategy.stoploss_guard_lookback": {"organ": "strategy_protection", "default": 48, "min": 12,  "max": 96},
    "strategy.stoploss_guard_limit":  {"organ": "strategy_protection", "default": 6,   "min": 2,   "max": 12},
    "strategy.maxdd_lookback":        {"organ": "strategy_protection", "default": 168, "min": 48,  "max": 336},
    "strategy.maxdd_limit":           {"organ": "strategy_protection", "default": 20,  "min": 5,   "max": 50},
    "strategy.maxdd_threshold":       {"organ": "strategy_protection", "default": 0.50, "min": 0.20, "max": 0.80},

    # ─── ORGAN: strategy_funding (2 params) — AIFreqtradeSizer.py:1066-1067 ───
    "strategy.extreme_funding":       {"organ": "strategy_funding", "default": 0.0005, "min": 0.0002, "max": 0.002},
    "strategy.funding_cap_mult":      {"organ": "strategy_funding", "default": 0.5,  "min": 0.2,  "max": 0.8},

    # ─── ORGAN: opportunity_weights (5 params, SUM=1.0) — opportunity_scanner.py:46-52 ───
    "opp.weight.momentum":    {"organ": "opp_weights", "default": 0.25, "min": 0.10, "max": 0.40},
    "opp.weight.reversion":   {"organ": "opp_weights", "default": 0.25, "min": 0.10, "max": 0.40},
    "opp.weight.funding":     {"organ": "opp_weights", "default": 0.20, "min": 0.05, "max": 0.35},
    "opp.weight.regime_shift": {"organ": "opp_weights", "default": 0.15, "min": 0.05, "max": 0.30},
    "opp.weight.volume":      {"organ": "opp_weights", "default": 0.15, "min": 0.05, "max": 0.30},

    # ─── ORGAN: cross_pair (6 params) — cross_pair_intel.py:86-278 ───
    "cross_pair.strong_bias":   {"organ": "cross_pair", "default": 0.70, "min": 0.55, "max": 0.85},
    "cross_pair.mild_bias":     {"organ": "cross_pair", "default": 0.55, "min": 0.50, "max": 0.70},
    "cross_pair.btc_lead_conf": {"organ": "cross_pair", "default": 0.40, "min": 0.25, "max": 0.60},
    "cross_pair.funding_extreme": {"organ": "cross_pair", "default": 0.0005, "min": 0.0002, "max": 0.002},
    "cross_pair.bearish_penalty": {"organ": "cross_pair", "default": 0.05, "min": 0.02, "max": 0.10},
    "cross_pair.bullish_boost":   {"organ": "cross_pair", "default": 0.03, "min": 0.01, "max": 0.08},

    # ─── ORGAN: autonomy (6 key params) — autonomy_manager.py:34-49 ───
    "autonomy.kelly_l0": {"organ": "autonomy", "default": 0.03, "min": 0.01, "max": 0.10},
    "autonomy.kelly_l1": {"organ": "autonomy", "default": 0.07, "min": 0.03, "max": 0.15},
    "autonomy.kelly_l2": {"organ": "autonomy", "default": 0.15, "min": 0.05, "max": 0.30},
    "autonomy.kelly_l3": {"organ": "autonomy", "default": 0.30, "min": 0.10, "max": 0.50},
    "autonomy.kelly_l4": {"organ": "autonomy", "default": 0.50, "min": 0.20, "max": 0.75},
    "autonomy.kelly_l5": {"organ": "autonomy", "default": 0.75, "min": 0.30, "max": 1.00},

    # ─── ORGAN: regime (5 params) — regime_classifier.py:52-126 ───
    "regime.adx_trending":    {"organ": "regime", "default": 25, "min": 18, "max": 35},
    "regime.adx_ranging":     {"organ": "regime", "default": 20, "min": 12, "max": 28},
    "regime.atr_high_vol":    {"organ": "regime", "default": 2.0, "min": 1.3, "max": 3.0},
    "regime.mod_ranging":     {"organ": "regime", "default": 0.80, "min": 0.60, "max": 0.95},
    "regime.mod_high_vol":    {"organ": "regime", "default": 0.75, "min": 0.50, "max": 0.90},

    # ─── ORGAN: calibrator (3 params) — confidence_calibrator.py:42-170 ───
    "calibrator.min_trades":     {"organ": "calibrator", "default": 20,   "min": 10,   "max": 50},
    "calibrator.brier_threshold": {"organ": "calibrator", "default": 0.25, "min": 0.20, "max": 0.35},
    "calibrator.platt_lr":       {"organ": "calibrator", "default": 0.01, "min": 0.001, "max": 0.05},

    # ─── ORGAN: retriever (5 params) — hybrid_retriever.py:234-570 ───
    "retriever.rrf_k":           {"organ": "retriever", "default": 60,   "min": 20,   "max": 120},
    "retriever.temporal_half_life": {"organ": "retriever", "default": 7.0, "min": 1.0,  "max": 30.0},
    "retriever.temporal_alpha":  {"organ": "retriever", "default": 0.7,  "min": 0.3,  "max": 0.95},
    "retriever.rerank_alpha":    {"organ": "retriever", "default": 0.5,  "min": 0.2,  "max": 0.8},
    "retriever.unknown_date_decay": {"organ": "retriever", "default": 0.5, "min": 0.2, "max": 0.8},

    # ─── ORGAN: agent_pool (5 params) — agent_pool.py:242-573 ───
    "agent.perf_wr_weight":      {"organ": "agent_pool", "default": 0.60, "min": 0.30, "max": 0.80},
    "agent.perf_exp_weight":     {"organ": "agent_pool", "default": 0.40, "min": 0.20, "max": 0.70},
    "agent.perf_exp_normalizer": {"organ": "agent_pool", "default": 50,   "min": 20,   "max": 100},
    "agent.vote_weight_base":    {"organ": "agent_pool", "default": 0.8,  "min": 0.5,  "max": 1.0},
    "agent.vote_weight_scale":   {"organ": "agent_pool", "default": 0.4,  "min": 0.1,  "max": 0.7},

    # ─── ORGAN: sentiment (3 params) — sentiment_analyzer.py:87-135 ───
    "sentiment.llm_temperature": {"organ": "sentiment", "default": 0.1,  "min": 0.0,  "max": 0.5},
    "sentiment.llm_timeout":     {"organ": "sentiment", "default": 30,   "min": 10,   "max": 60},
    "sentiment.batch_size":      {"organ": "sentiment", "default": 20,   "min": 5,    "max": 50},

    # ─── ORGAN: cryptopanic (2 params) — cryptopanic_fetcher.py:24-102 ───
    "cryptopanic.fetch_limit":   {"organ": "cryptopanic", "default": 20,  "min": 5,    "max": 50},
    "cryptopanic.important_weight": {"organ": "cryptopanic", "default": 0.2, "min": 0.05, "max": 0.5},

    # ─── ORGAN: opp_momentum_scoring (~12 params) — opportunity_scanner.py:220-270 ───
    "opp.momentum.ema_full_bull":     {"organ": "opp_momentum", "default": 40, "min": 20, "max": 60},
    "opp.momentum.ema_partial_1":     {"organ": "opp_momentum", "default": 30, "min": 15, "max": 50},
    "opp.momentum.ema_partial_2":     {"organ": "opp_momentum", "default": 20, "min": 8,  "max": 35},
    "opp.momentum.ema_above_200":     {"organ": "opp_momentum", "default": 10, "min": 5,  "max": 25},
    "opp.momentum.ema_full_bear":     {"organ": "opp_momentum", "default": 35, "min": 15, "max": 50},
    "opp.momentum.adx_strong":        {"organ": "opp_momentum", "default": 35, "min": 25, "max": 50},
    "opp.momentum.adx_score_strong":  {"organ": "opp_momentum", "default": 30, "min": 15, "max": 45},
    "opp.momentum.adx_score_med":     {"organ": "opp_momentum", "default": 20, "min": 10, "max": 35},
    "opp.momentum.adx_score_low":     {"organ": "opp_momentum", "default": 10, "min": 3,  "max": 18},
    "opp.momentum.vol_extreme":       {"organ": "opp_momentum", "default": 2.0, "min": 1.5, "max": 4.0},
    "opp.momentum.vol_score_hi":      {"organ": "opp_momentum", "default": 30, "min": 15, "max": 45},
    "opp.momentum.vol_score_med":     {"organ": "opp_momentum", "default": 20, "min": 10, "max": 35},

    # ─── ORGAN: opp_reversion_scoring (~8 params) — opportunity_scanner.py:272-315 ───
    "opp.reversion.rsi_extreme":      {"organ": "opp_reversion", "default": 35, "min": 20, "max": 50},
    "opp.reversion.rsi_moderate":     {"organ": "opp_reversion", "default": 25, "min": 10, "max": 40},
    "opp.reversion.bb_touch":         {"organ": "opp_reversion", "default": 30, "min": 15, "max": 45},
    "opp.reversion.bb_near":          {"organ": "opp_reversion", "default": 20, "min": 10, "max": 35},
    "opp.reversion.fng_extreme":      {"organ": "opp_reversion", "default": 35, "min": 20, "max": 50},
    "opp.reversion.fng_moderate":     {"organ": "opp_reversion", "default": 20, "min": 10, "max": 35},
    "opp.reversion.fng_mild":         {"organ": "opp_reversion", "default": 10, "min": 3,  "max": 18},
    "opp.reversion.fng_extreme_thr":  {"organ": "opp_reversion", "default": 15, "min": 5,  "max": 25},
    "opp.reversion.rsi_mild":         {"organ": "opp_reversion", "default": 15, "min": 5,  "max": 30},

    # ─── ORGAN: opp_regime_shift (~6 params) — opportunity_scanner.py:349-389 ───
    "opp.regime.sweet_spot":       {"organ": "opp_regime", "default": 40, "min": 20, "max": 60},
    "opp.regime.macd_cross":       {"organ": "opp_regime", "default": 20, "min": 10, "max": 35},
    "opp.regime.ema_close":        {"organ": "opp_regime", "default": 25, "min": 10, "max": 40},
    "opp.regime.ema_near":         {"organ": "opp_regime", "default": 15, "min": 5,  "max": 25},
    "opp.regime.vol_rising":       {"organ": "opp_regime", "default": 15, "min": 5,  "max": 25},
    "opp.regime.pre_transition":   {"organ": "opp_regime", "default": 10, "min": 3,  "max": 20},

    # ─── ORGAN: opp_volume_anomaly (~6 params) — opportunity_scanner.py:391-418 ───
    "opp.volume.extreme_score":    {"organ": "opp_volume", "default": 50, "min": 25, "max": 70},
    "opp.volume.high_score":       {"organ": "opp_volume", "default": 35, "min": 15, "max": 50},
    "opp.volume.moderate_score":   {"organ": "opp_volume", "default": 20, "min": 10, "max": 35},
    "opp.volume.stealth_extreme":  {"organ": "opp_volume", "default": 50, "min": 25, "max": 70},
    "opp.volume.stealth_high":     {"organ": "opp_volume", "default": 30, "min": 15, "max": 45},
    "opp.volume.stealth_moderate": {"organ": "opp_volume", "default": 20, "min": 10, "max": 35},

    # ─── ORGAN: opp_partial (kalan 4 param) ───
    "opp.momentum.ema_partial_bear": {"organ": "opp_momentum", "default": 25, "min": 10, "max": 40},
    "opp.momentum.vol_score_low":  {"organ": "opp_momentum", "default": 10, "min": 3,  "max": 22},

    # ─── ORGAN: opp_funding_scoring (~6 params) — opportunity_scanner.py:317-347 ───
    "opp.funding.fr_very_extreme":    {"organ": "opp_funding", "default": 60, "min": 30, "max": 80},
    "opp.funding.fr_extreme":         {"organ": "opp_funding", "default": 40, "min": 20, "max": 60},
    "opp.funding.fr_notable":         {"organ": "opp_funding", "default": 20, "min": 10, "max": 35},
    "opp.funding.ls_very_unbalanced": {"organ": "opp_funding", "default": 40, "min": 20, "max": 60},
    "opp.funding.ls_unbalanced":      {"organ": "opp_funding", "default": 25, "min": 10, "max": 40},
    "opp.funding.ls_skewed":          {"organ": "opp_funding", "default": 10, "min": 3,  "max": 18},

    # ─── ORGAN: strategy_trailing (~6 params) — AIFreqtradeSizer.py:912-918 ───
    "strategy.trailing_pnl_high":     {"organ": "strategy_trailing", "default": 0.15, "min": 0.08, "max": 0.25},
    "strategy.trailing_pnl_med":      {"organ": "strategy_trailing", "default": 0.08, "min": 0.04, "max": 0.15},
    "strategy.trailing_pnl_low":      {"organ": "strategy_trailing", "default": 0.04, "min": 0.02, "max": 0.10},
    "strategy.trailing_atr_high":     {"organ": "strategy_trailing", "default": 1.0, "min": 0.3, "max": 1.8},
    "strategy.trailing_atr_med":      {"organ": "strategy_trailing", "default": 1.5, "min": 0.8, "max": 2.5},
    "strategy.trailing_atr_low":      {"organ": "strategy_trailing", "default": 2.0, "min": 1.0, "max": 3.0},

    # ─── ORGAN: strategy_dca (~8 params) — AIFreqtradeSizer.py:1620-1685 ───
    "strategy.dca_max_entries":       {"organ": "strategy_dca", "default": 4, "min": 1, "max": 8},
    "strategy.dca_wait_hours":        {"organ": "strategy_dca", "default": 0.5, "min": 0.1, "max": 4.0},
    "strategy.dca_lock1_pnl":         {"organ": "strategy_dca", "default": 0.06, "min": 0.03, "max": 0.15},
    "strategy.dca_lock2_pnl":         {"organ": "strategy_dca", "default": 0.12, "min": 0.06, "max": 0.25},
    "strategy.dca_lock_pct":          {"organ": "strategy_dca", "default": 0.25, "min": 0.10, "max": 0.50},
    "strategy.pyramid_conf":          {"organ": "strategy_dca", "default": 0.80, "min": 0.60, "max": 0.95},
    "strategy.pyramid_fraction":      {"organ": "strategy_dca", "default": 0.30, "min": 0.10, "max": 0.50},
    "strategy.reduce_fraction":       {"organ": "strategy_dca", "default": 0.30, "min": 0.10, "max": 0.50},

    # ─── ORGAN: legacy_fallback (~12 params) — rag_graph.py:442-578 ───
    "rag.legacy.rsi_weight":          {"organ": "legacy_fallback", "default": 0.15, "min": 0.05, "max": 0.25},
    "rag.legacy.macd_weight":         {"organ": "legacy_fallback", "default": 0.15, "min": 0.05, "max": 0.25},
    "rag.legacy.ema_cross_weight":    {"organ": "legacy_fallback", "default": 0.12, "min": 0.05, "max": 0.20},
    "rag.legacy.ema_align_weight":    {"organ": "legacy_fallback", "default": 0.10, "min": 0.03, "max": 0.18},
    "rag.legacy.bb_weight":           {"organ": "legacy_fallback", "default": 0.08, "min": 0.02, "max": 0.15},
    "rag.legacy.adx_ranging_mult":    {"organ": "legacy_fallback", "default": 0.70, "min": 0.40, "max": 0.90},
    "rag.legacy.vol_confirm_mult":    {"organ": "legacy_fallback", "default": 1.10, "min": 1.0,  "max": 1.3},
    "rag.legacy.htf_rsi_weight":      {"organ": "legacy_fallback", "default": 0.08, "min": 0.02, "max": 0.15},
    "rag.legacy.daily_weight":        {"organ": "legacy_fallback", "default": 0.08, "min": 0.02, "max": 0.15},
    "rag.legacy.sr_weight":           {"organ": "legacy_fallback", "default": 0.05, "min": 0.01, "max": 0.10},
    "rag.legacy.signal_dir_thr":      {"organ": "legacy_fallback", "default": 0.10, "min": 0.03, "max": 0.18},
    "rag.legacy.confidence_cap":      {"organ": "legacy_fallback", "default": 0.35, "min": 0.20, "max": 0.50},

    # ─── ORGAN: llm_penalties (~10 params) — llm_router.py:72-78 ───
    "llm.penalty.rate_limit_base":    {"organ": "llm_penalties", "default": 30,  "min": 10,  "max": 60},
    "llm.penalty.rate_limit_max":     {"organ": "llm_penalties", "default": 300, "min": 60,  "max": 600},
    "llm.penalty.timeout_base":       {"organ": "llm_penalties", "default": 15,  "min": 5,   "max": 30},
    "llm.penalty.overloaded_base":    {"organ": "llm_penalties", "default": 45,  "min": 15,  "max": 90},
    "llm.penalty.empty_base":         {"organ": "llm_penalties", "default": 30,  "min": 10,  "max": 60},

    # ─── ORGAN: llm_circuit (~4 params) — llm_router.py:181-182 ───
    "llm.circuit.threshold":          {"organ": "llm_circuit", "default": 10,  "min": 5,   "max": 25},
    "llm.circuit.window":             {"organ": "llm_circuit", "default": 60,  "min": 20,  "max": 180},
    "llm.circuit.min_open":           {"organ": "llm_circuit", "default": 30,  "min": 10,  "max": 60},
    "llm.circuit.close_after":        {"organ": "llm_circuit", "default": 3,   "min": 1,   "max": 5},

    # ─── ORGAN: btc_lead (~3 params) — cross_pair_intel.py:144-150 ───
    "cross_pair.btc_disagree":        {"organ": "cross_pair", "default": 0.30, "min": 0.15, "max": 0.50},
    "cross_pair.btc_agree":           {"organ": "cross_pair", "default": 0.50, "min": 0.30, "max": 0.70},
    "cross_pair.btc_adj":             {"organ": "cross_pair", "default": 0.05, "min": 0.02, "max": 0.10},

    # ─── ORGAN: funding_heatmap (~2 params) — cross_pair_intel.py:223-225 ───
    "cross_pair.crowded_long_pct":    {"organ": "cross_pair", "default": 70, "min": 55, "max": 85},
    "cross_pair.crowded_short_pct":   {"organ": "cross_pair", "default": 30, "min": 15, "max": 45},

    # ─── ORGAN: scheduler_thresholds (~3 params) — scheduler.py:688-720 ───
    "scheduler.fng_extreme_low":      {"organ": "scheduler", "default": 15, "min": 5,  "max": 25},
    "scheduler.fng_extreme_high":     {"organ": "scheduler", "default": 85, "min": 70, "max": 95},
    "scheduler.extreme_funding":      {"organ": "scheduler", "default": 0.001, "min": 0.0003, "max": 0.003},

    # ─── ORGAN: agent_debate (~4 params) — agent_pool.py:341-415 ───
    "agent.r1_temperature":           {"organ": "agent_debate", "default": 0.4, "min": 0.1, "max": 0.7},
    "agent.r2_temperature":           {"organ": "agent_debate", "default": 0.3, "min": 0.1, "max": 0.5},
    "agent.r3_temperature":           {"organ": "agent_debate", "default": 0.2, "min": 0.05, "max": 0.5},
    "agent.confidence_mod_range":     {"organ": "agent_debate", "default": 0.10, "min": 0.03, "max": 0.25},

    # ─── ORGAN: rag_tier (~4 params) — rag_graph.py:1600-1630 ───
    "rag.tier.agent_min_conf":        {"organ": "rag_tier", "default": 0.20, "min": 0.05, "max": 0.35},
    "rag.tier.ee_min_conf":           {"organ": "rag_tier", "default": 0.20, "min": 0.05, "max": 0.35},
    "rag.tier.max_coord_conf":        {"organ": "rag_tier", "default": 0.85, "min": 0.70, "max": 0.95},
    "rag.tier.neutral_upper":         {"organ": "rag_tier", "default": 0.55, "min": 0.50, "max": 0.65},

    # ─── ORGAN: rag_health (~2 params) — rag_graph.py:1875 ───
    "rag.health.chroma_penalty":      {"organ": "rag_health", "default": 0.85, "min": 0.60, "max": 0.95},
    "rag.health.chroma_floor":        {"organ": "rag_health", "default": 0.40, "min": 0.25, "max": 0.60},

    # ─── ORGAN: voting_fallback (~2 params) — rag_graph.py:618 ───
    "rag.voting.scaling":             {"organ": "voting_fallback", "default": 0.40, "min": 0.20, "max": 0.60},
    "rag.voting.cap":                 {"organ": "voting_fallback", "default": 0.30, "min": 0.15, "max": 0.50},

    # ─── ORGAN: Phase 25 subsystem params (~10 params) ───
    "cerebellum.hour_conf_min":       {"organ": "cerebellum", "default": 0.6, "min": 0.3, "max": 0.8},
    "cerebellum.hour_conf_max":       {"organ": "cerebellum", "default": 1.4, "min": 1.1, "max": 1.8},
    "predictive.min_episodes":        {"organ": "predictive", "default": 3,   "min": 1,   "max": 10},
    "predictive.error_lr_boost":      {"organ": "predictive", "default": 2.0, "min": 1.0, "max": 3.0},
    "sleep.replay_count":             {"organ": "sleep", "default": 50,  "min": 10,  "max": 100},
    "sleep.prune_decay":              {"organ": "sleep", "default": 0.95, "min": 0.85, "max": 0.99},
    "evolution.population_size":      {"organ": "evolution", "default": 5,   "min": 3,   "max": 15},
    "evolution.blend_rate":           {"organ": "evolution", "default": 0.10, "min": 0.05, "max": 0.30},
    "mirror.crowd_wrong_init":        {"organ": "mirror", "default": 0.50, "min": 0.20, "max": 0.80},
    "immunity.severity_rate":         {"organ": "immunity", "default": 0.10, "min": 0.03, "max": 0.25},

    # ─── ORGAN: immune / pair_ban (5 params) — NEW ───
    "immune.base_ban_minutes":   {"organ": "immune", "default": 60,   "min": 15,   "max": 240},
    "immune.loss_multiplier":    {"organ": "immune", "default": 2.0,  "min": 0.5,  "max": 5.0},
    "immune.consec_multiplier":  {"organ": "immune", "default": 1.5,  "min": 1.0,  "max": 3.0},
    "immune.max_hours":          {"organ": "immune", "default": 24,   "min": 4,    "max": 72},
    "immune.min_loss_to_ban":    {"organ": "immune", "default": -2.0, "min": -10.0, "max": -0.5},
}

# Organs that must sum to a target value
ORGAN_CONSTRAINTS = {
    "evidence_weights": 1.0,
    "opp_weights": 1.0,
}

# Seed causal synapses between neurons
SEED_SYNAPSES = [
    # (source, target, weight, type)
    # Chain 1: F&G → crowd weight → synthesis cap → sizing → leverage
    ("evidence.q3.fng_adj_extreme", "evidence.weights.q3_crowd", 0.6, "excitatory"),
    ("evidence.weights.q3_crowd", "evidence.synthesis.cap_5", 0.4, "excitatory"),
    ("evidence.synthesis.cap_5", "sizing.max_risk", 0.3, "excitatory"),
    ("sizing.max_risk", "strategy.leverage_max", 0.5, "inhibitory"),
    # Chain 2: ATR → stoploss → leverage (MATH: ATR×leverage ≤ equity_loss)
    ("evidence.q6.atr_adj_high", "strategy.chandelier_atr_high", 0.7, "excitatory"),
    ("strategy.chandelier_atr_high", "strategy.leverage_max", 0.8, "inhibitory"),
    # Chain 3: Regime → evidence weights
    ("regime.adx_trending", "evidence.weights.q1_trend", 0.5, "excitatory"),
    ("regime.adx_ranging", "evidence.weights.q3_crowd", 0.5, "excitatory"),
    ("regime.atr_high_vol", "evidence.weights.q6_risk", 0.6, "excitatory"),
    # Chain 4: Funding consistency (scheduler + cross_pair use SAME source)
    ("evidence.q3.funding_extreme", "cross_pair.funding_extreme", 0.9, "excitatory"),
    # Chain 5: Calibrator → confidence caps
    ("calibrator.brier_threshold", "evidence.synthesis.cap_5", 0.4, "inhibitory"),
    # Chain 6: Pair ban → ban duration
    ("immune.min_loss_to_ban", "immune.base_ban_minutes", 0.6, "excitatory"),
]

# Amygdala graduated fear tiers
FEAR_TIERS = [
    # (loss_threshold, fear_level, learning_mult, sizing_mult, tier_name)
    (-2.0,  0.2, 1.0, 1.00, "normal"),
    (-5.0,  0.5, 2.0, 0.75, "stress"),
    (-10.0, 0.8, 3.0, 0.50, "fear"),
    (-999,  1.0, 0.5, 0.10, "panic"),
]


# ═══════════════════════════════════════════════════════════════════════════
# PARAM NEURON — Single cell
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ParamNeuron:
    """A single learnable parameter with Beta distribution posterior + BCM metaplasticity."""
    param_id: str
    organ: str
    regime: str
    current_val: float
    default_val: float
    min_bound: float
    max_bound: float
    alpha: float = 2.0
    beta_param: float = 2.0
    prior_strength: float = 5.0
    frozen: bool = False
    update_count: int = 0
    # Phase 25: BCM metaplasticity — dynamic learning rate
    activity_ema: float = 0.0       # Running avg of reward magnitudes (BCM)
    theta_m: float = 0.0            # Sliding modification threshold (BCM)
    last_update_time: Optional[datetime] = None  # For STDP temporal credit

    def sample(self) -> float:
        """Thompson Sampling draw scaled to [min_bound, max_bound]."""
        raw = random.betavariate(max(self.alpha, 0.01), max(self.beta_param, 0.01))
        return self.min_bound + raw * (self.max_bound - self.min_bound)

    def mean(self) -> float:
        """Expected value (exploit mode)."""
        raw = self.alpha / (self.alpha + self.beta_param)
        return self.min_bound + raw * (self.max_bound - self.min_bound)

    def nudge(self, reward: float, magnitude: float = 1.0):
        """
        BCM-enhanced Bayesian update.
        - prior_strength: static habit resistance (BasalGanglia increases this)
        - theta_m: dynamic BCM threshold (high recent activity → harder to change)
        - Dead zone: |reward| < theta_m → too weak to cause change
        """
        if self.frozen:
            return
        # BCM: update activity trace
        self.activity_ema = 0.95 * self.activity_ema + 0.05 * (magnitude ** 2)
        self.theta_m = self.activity_ema

        # Effective magnitude: reduced by both habit resistance AND BCM threshold
        effective_mag = magnitude / (1.0 + self.prior_strength * 0.1 + self.theta_m * 2.0)

        # BCM sliding threshold: dead zone prevents noise from causing changes
        if reward > self.theta_m:       # LTP: reward exceeds threshold → potentiate
            self.alpha += effective_mag
        elif reward < -self.theta_m:    # LTD: negative exceeds threshold → depress
            self.beta_param += effective_mag
        # else: dead zone — signal too weak to modify this neuron

        self.last_update_time = datetime.now(tz=timezone.utc)
        self.current_val = self.mean()
        self.current_val = max(self.min_bound, min(self.max_bound, self.current_val))
        self.update_count += 1

    def belief_width(self) -> float:
        """How wide the Beta distribution is (0=certain, 1=uncertain)."""
        total = self.alpha + self.beta_param
        if total < 0.1:
            return 1.0
        variance = (self.alpha * self.beta_param) / (total * total * (total + 1))
        return min(1.0, variance ** 0.5 * 4.0)


# ═══════════════════════════════════════════════════════════════════════════
# HORMONES — Global broadcast modulators
# ═══════════════════════════════════════════════════════════════════════════

class Hormones:
    """4 hormones + Allostasis (anticipatory adjustment based on trends)."""

    def __init__(self):
        self.cortisol = 1.0
        self.dopamine = 1.0
        self.serotonin = 1.0
        self.adrenaline = 1.0
        self._stress = 0.0
        self._health = 0.5
        self._info_q = 0.5
        # Phase 25: Allostasis — trend tracking for anticipation
        self._fng_history: List[int] = []
        self._drawdown_history: List[float] = []

    def compute(self, fng: Optional[int] = None, drawdown_pct: float = 0.0,
                consec_wins: int = 0, consec_losses: int = 0,
                active_sources: int = 4, balance_vs_peak: float = 1.0) -> dict:
        """Recompute all hormone levels from raw inputs + allostatic anticipation."""
        # Market Stress: 0 (calm) → 1 (panic)
        stress = 0.0
        if fng is not None and fng < 20:
            stress += 0.3 * (1.0 - fng / 20.0)
        if drawdown_pct > 5:
            stress += min(0.3, drawdown_pct * 0.03)
        if consec_losses >= 3:
            stress += min(0.2, consec_losses * 0.07)
        stress = min(1.0, max(0.0, stress))

        health = max(0.0, min(1.0, balance_vs_peak))
        info_q = max(0.1, min(1.0, active_sources / 7.0))

        self._stress = stress
        self._health = health
        self._info_q = info_q

        self.cortisol = max(0.5, 1.0 - stress * 0.4)
        self.dopamine = min(1.10, 0.9 + health * 0.15)
        self.serotonin = max(0.6, 0.5 + info_q * 0.5)
        self.adrenaline = 0.0 if stress > 0.85 else 1.0

        # Phase 25: ALLOSTASIS — anticipate based on trends
        anticipation = self._compute_allostasis(fng, drawdown_pct)
        self.cortisol = max(0.5, min(1.0, self.cortisol + anticipation.get("cortisol_adj", 0)))

        return self.as_dict()

    def _compute_allostasis(self, fng: Optional[int], drawdown_pct: float) -> dict:
        """Allostasis: detect trends in F&G and drawdown, pre-adjust hormones."""
        self._fng_history.append(fng if fng is not None else 50)
        self._drawdown_history.append(drawdown_pct)
        self._fng_history = self._fng_history[-6:]
        self._drawdown_history = self._drawdown_history[-6:]

        adj = {"cortisol_adj": 0.0}

        # F&G declining trend → pre-tighten cortisol (anticipate fear)
        if len(self._fng_history) >= 3:
            recent = self._fng_history[-3:]
            if all(recent[i] < recent[i - 1] for i in range(1, len(recent))):
                decline_rate = (recent[0] - recent[-1]) / max(recent[0], 1)
                adj["cortisol_adj"] = -min(0.10, decline_rate * 0.5)
            elif all(recent[i] > recent[i - 1] for i in range(1, len(recent))):
                adj["cortisol_adj"] = 0.03  # Improving → slight relief

        # Drawdown worsening → anticipate more stress
        if len(self._drawdown_history) >= 3:
            recent = self._drawdown_history[-3:]
            if all(recent[i] > recent[i - 1] for i in range(1, len(recent))):
                adj["cortisol_adj"] -= 0.05

        return adj

    def as_dict(self) -> dict:
        return {
            "cortisol": round(self.cortisol, 3),
            "dopamine": round(self.dopamine, 3),
            "serotonin": round(self.serotonin, 3),
            "adrenaline": round(self.adrenaline, 3),
            "_stress": round(self._stress, 3),
            "_health": round(self._health, 3),
            "_info_q": round(self._info_q, 3),
        }


# ═══════════════════════════════════════════════════════════════════════════
# AMYGDALA — Graduated fear response
# ═══════════════════════════════════════════════════════════════════════════

class Amygdala:
    """Non-linear fear response. Fear decays with 24h half-life."""

    def __init__(self):
        self.fear_level = 0.0
        self.peak_fear = 0.0
        self.peak_time: Optional[datetime] = None
        self.tier = "normal"

    def process_loss(self, loss_pct: float) -> dict:
        """Compute fear response from trade loss."""
        for threshold, fear, learn_mult, size_mult, label in FEAR_TIERS:
            if loss_pct >= threshold:
                if fear > self.fear_level:
                    self.fear_level = fear
                    self.peak_fear = fear
                    self.peak_time = datetime.now(tz=timezone.utc)
                    self.tier = label
                return {
                    "fear_level": fear, "learning_mult": learn_mult,
                    "sizing_mult": size_mult, "tier": label,
                }
        return {"fear_level": 0.0, "learning_mult": 1.0, "sizing_mult": 1.0, "tier": "normal"}

    def get_current_fear(self) -> float:
        """Fear decays with 24h half-life."""
        if not self.peak_time or self.peak_fear <= 0:
            return 0.0
        hours = (datetime.now(tz=timezone.utc) - self.peak_time).total_seconds() / 3600.0
        decayed = self.peak_fear * (0.5 ** (hours / 24.0))
        self.fear_level = decayed
        if decayed < 0.05:
            self.tier = "normal"
        return decayed

    def as_dict(self) -> dict:
        return {"fear_level": round(self.get_current_fear(), 3), "tier": self.tier}


# ═══════════════════════════════════════════════════════════════════════════
# HIPPOCAMPUS — Pattern memory (situation fingerprint → outcome)
# ═══════════════════════════════════════════════════════════════════════════

class Hippocampus:
    """Stores situation fingerprints and recalls similar past episodes."""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    @staticmethod
    def get_fingerprint(fng: Optional[int], regime: str, adx: float = 20,
                        funding_rate: float = 0, consec_losses: int = 0,
                        stress: float = 0) -> str:
        """7-dimension bucketed fingerprint → deterministic JSON."""
        fng_bucket = "unknown"
        if fng is not None:
            if fng < 15:
                fng_bucket = "extreme_fear"
            elif fng < 30:
                fng_bucket = "fear"
            elif fng < 70:
                fng_bucket = "neutral"
            elif fng < 85:
                fng_bucket = "greed"
            else:
                fng_bucket = "extreme_greed"
        adx_bucket = "trending" if adx > 25 else "ranging" if adx < 20 else "transitional"
        funding_bucket = ("crowded_long" if funding_rate > 0.0005 else
                          "crowded_short" if funding_rate < -0.0005 else "neutral")
        stress_bucket = "high" if stress > 0.6 else "medium" if stress > 0.3 else "low"
        return json.dumps({
            "fng": fng_bucket, "regime": regime, "adx": adx_bucket,
            "funding": funding_bucket, "streak": min(consec_losses, 5),
            "stress": stress_bucket,
        }, sort_keys=True)

    def store_episode(self, pair: str, fingerprint: str, outcome_pnl: float, regime: str):
        try:
            with self._get_conn() as conn:
                conn.execute(
                    "INSERT INTO hippocampus_episodes (pair, fingerprint, outcome_pnl, regime, timestamp) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (pair, fingerprint, outcome_pnl, regime,
                     datetime.now(tz=timezone.utc).isoformat()))
                conn.commit()
        except Exception as e:
            logger.debug(f"[Hippocampus] Store failed: {e}")

    def recall(self, fingerprint: str, k: int = 5) -> list:
        """Recall k most recent episodes with same fingerprint."""
        try:
            with self._get_conn() as conn:
                rows = conn.execute(
                    "SELECT outcome_pnl, pair, timestamp FROM hippocampus_episodes "
                    "WHERE fingerprint = ? ORDER BY timestamp DESC LIMIT ?",
                    (fingerprint, k)).fetchall()
                return [dict(r) for r in rows]
        except Exception:
            return []


# ═══════════════════════════════════════════════════════════════════════════
# SYNAPSE NETWORK — Inter-neuron causal connections
# ═══════════════════════════════════════════════════════════════════════════

class SynapseNetwork:
    """Propagates changes between causally connected neurons (1-hop, dampened)."""

    def __init__(self):
        self._edges: Dict[str, List[Tuple[str, float, str]]] = {}  # source → [(target, weight, type)]
        for src, tgt, weight, stype in SEED_SYNAPSES:
            self._edges.setdefault(src, []).append((tgt, weight, stype))

    def propagate(self, source: str, delta: float,
                  neurons: Dict[Tuple[str, str], 'ParamNeuron'], regime: str):
        """Propagate delta from source to 1-hop connected neurons."""
        edges = self._edges.get(source, [])
        for tgt, weight, stype in edges:
            sign = 1.0 if stype == "excitatory" else -1.0
            pull = delta * weight * sign * 0.3  # 0.3 dampening factor
            target_neuron = neurons.get((tgt, regime)) or neurons.get((tgt, "_global"))
            if target_neuron and not target_neuron.frozen:
                target_neuron.nudge(pull, magnitude=abs(pull))
                logger.debug(f"[Synapse] {source} → {tgt}: pull={pull:.4f} ({stype})")


# ═══════════════════════════════════════════════════════════════════════════
# PREFRONTAL CORTEX — Hard rules that NEVER learn (veto power)
# ═══════════════════════════════════════════════════════════════════════════

class PrefrontalCortex:
    """Executive override layer. Non-learnable hard safety constraints."""

    ESSENTIAL_ORGANS = {"sizing", "risk", "strategy_stoploss"}

    def evaluate(self, neurons: Dict[Tuple[str, str], ParamNeuron],
                 hormones: Hormones, hippocampus_recalls: list,
                 amygdala: Amygdala) -> List[str]:
        """Apply hard rules. Returns list of override descriptions."""
        overrides = []

        # Rule 1: Leverage NEVER > 5x
        for key, n in neurons.items():
            if "leverage_max" in key[0] and n.current_val > 5.0:
                n.current_val = 5.0
                overrides.append(f"leverage hard-capped at 5x: {key[0]}")

        # Rule 2: Adrenaline freeze — non-essential organs snap to default
        if hormones.adrenaline == 0.0:
            for key, n in neurons.items():
                if n.organ not in self.ESSENTIAL_ORGANS and not n.frozen:
                    n.frozen = True
                    n.current_val = n.default_val
            if not any("adrenaline" in o for o in overrides):
                overrides.append("ADRENALINE FREEZE: non-essential params locked to defaults")

        # Rule 3: Hippocampus warning — 3+ similar episodes lost >3%
        if hippocampus_recalls:
            bad_episodes = [ep for ep in hippocampus_recalls if ep.get("outcome_pnl", 0) < -3.0]
            if len(bad_episodes) >= 3:
                for key, n in neurons.items():
                    if n.organ == "sizing":
                        n.current_val = max(n.min_bound, n.current_val * 0.5)
                overrides.append(f"HIPPOCAMPUS WARNING: {len(bad_episodes)}/5 similar episodes lost >3% → sizing halved")

        # Rule 4: Low info quality → cap confidence
        if hormones._info_q < 0.4:
            for key, n in neurons.items():
                if key[0].startswith("evidence.synthesis.cap"):
                    n.current_val = min(n.current_val, 0.65)
            overrides.append(f"LOW INFO QUALITY ({hormones._info_q:.2f}): confidence caps reduced to 0.65")

        # Rule 5: ATR × leverage ≤ 8% max equity loss (mathematical constraint)
        # This is checked at runtime in strategy, but we enforce bounds here too
        for key, n in neurons.items():
            if key[0] == "strategy.max_equity_loss":
                n.current_val = min(n.current_val, n.max_bound)

        return overrides


# ═══════════════════════════════════════════════════════════════════════════
# BASAL GANGLIA — Habit formation (consolidation)
# ═══════════════════════════════════════════════════════════════════════════

class BasalGanglia:
    """Consolidates stable parameters into habits (resist change)."""

    CONSOLIDATION_MIN_UPDATES = 50
    VARIANCE_THRESHOLD_FRACTION = 0.05  # 5% of range
    STRENGTH_BOOST = 10.0

    def check_consolidation(self, neuron: ParamNeuron):
        """Consolidate if enough updates + low variance (stable value)."""
        if neuron.update_count < self.CONSOLIDATION_MIN_UPDATES:
            return
        param_range = neuron.max_bound - neuron.min_bound
        if param_range <= 0:
            return
        # Check if belief is narrow enough → stable parameter
        if neuron.belief_width() < self.VARIANCE_THRESHOLD_FRACTION:
            neuron.prior_strength += self.STRENGTH_BOOST
            logger.info(f"[BasalGanglia] HABIT formed: {neuron.param_id} (regime={neuron.regime}) "
                        f"strength → {neuron.prior_strength:.0f}")


# ═══════════════════════════════════════════════════════════════════════════
# PROPRIOCEPTION — Self-awareness (organism lifecycle phase)
# ═══════════════════════════════════════════════════════════════════════════

class Proprioception:
    """Organism knows its own state. No fixed thresholds — statistical."""

    def assess(self, neurons: Dict[Tuple[str, str], ParamNeuron],
               consec_wins: int = 0, consec_losses: int = 0) -> dict:
        """Assess organism phase from belief statistics."""
        if not neurons:
            return {"phase": "learning", "lr_mod": 1.5, "safety_mod": 1.5}

        widths = [n.belief_width() for n in neurons.values()]
        avg_width = sum(widths) / len(widths) if widths else 0.5

        # Detect overconfidence: narrow beliefs + winning streak
        if avg_width < 0.15 and consec_wins >= 5:
            return {"phase": "overconfident", "lr_mod": 0.5, "safety_mod": 1.3}
        # Still learning: wide beliefs
        if avg_width > 0.3:
            return {"phase": "learning", "lr_mod": 1.5, "safety_mod": 1.5}
        # Mature: narrow beliefs, stable
        if avg_width < 0.15:
            return {"phase": "mature", "lr_mod": 0.7, "safety_mod": 1.0}
        # Maturing: in between
        return {"phase": "maturing", "lr_mod": 1.0, "safety_mod": 1.2}


# ═══════════════════════════════════════════════════════════════════════════
# IMMUNE MEMORY — Dynamic pair banning
# ═══════════════════════════════════════════════════════════════════════════

class ImmuneMemory:
    """Pair ban organ: temporarily avoids coins that caused losses."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._consec_cache: Dict[str, int] = {}

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def record_loss(self, pair: str, loss_pct: float, regime: str):
        """Record a losing trade for pair ban calculation."""
        self._consec_cache[pair] = self._consec_cache.get(pair, 0) + 1
        try:
            with self._get_conn() as conn:
                conn.execute(
                    "INSERT INTO immune_memory (pair, loss_pct, consecutive_losses, regime, timestamp) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (pair, loss_pct, self._consec_cache[pair], regime,
                     datetime.now(tz=timezone.utc).isoformat()))
                conn.commit()
        except Exception as e:
            logger.debug(f"[ImmuneMemory] Record failed: {e}")

    def record_win(self, pair: str):
        """Reset consecutive loss counter on win."""
        self._consec_cache[pair] = 0

    def compute_ban(self, pair: str, loss_pct: float,
                    get_param_fn) -> int:
        """Compute ban duration in minutes using adaptive params."""
        base = get_param_fn("immune.base_ban_minutes", 60)
        loss_mult = get_param_fn("immune.loss_multiplier", 2.0)
        consec_mult = get_param_fn("immune.consec_multiplier", 1.5)
        max_hours = get_param_fn("immune.max_hours", 24)
        min_loss = get_param_fn("immune.min_loss_to_ban", -2.0)

        if loss_pct > min_loss:
            return 0  # Loss not bad enough to ban

        consec = self._consec_cache.get(pair, 1)
        ban = base * (1.0 + abs(loss_pct) * loss_mult)
        ban *= consec_mult ** max(0, consec - 1)
        return int(min(ban, max_hours * 60))


# ═══════════════════════════════════════════════════════════════════════════
# CREDIT ASSIGNER — Maps trade outcomes to neuron contributions
# ═══════════════════════════════════════════════════════════════════════════

class CreditAssigner:
    """Assigns credit/blame to neurons using STDP temporal weighting + organ-based credit."""

    SIGNAL_ORGANS = {"evidence_weights", "q1_trend", "q2_momentum", "q3_crowd",
                     "q4_evidence", "q5_macro", "q6_risk", "synthesis", "contradiction",
                     "evidence_first", "coordinator"}
    SIZING_ORGANS = {"sizing", "risk", "autonomy", "opp_weights"}
    DEFENSE_ORGANS = {"strategy_stoploss", "strategy_leverage", "strategy_exit",
                      "strategy_protection", "immune"}

    @staticmethod
    def _stdp_temporal_weight(neuron: ParamNeuron, trade_time: datetime) -> float:
        """STDP: neurons updated closer to trade entry get MORE credit/blame."""
        if not neuron.last_update_time:
            return 0.5
        hours_before = (trade_time - neuron.last_update_time).total_seconds() / 3600
        if hours_before < 0 or hours_before > 6:
            return 0.3  # Outside STDP window
        return 0.3 + 0.7 * math.exp(-hours_before / 2.0)  # Exponential decay

    def assign(self, trade_pnl: float, confidence: float, stake_pct: float,
               duration_hours: float, regime: str,
               neurons: Dict[Tuple[str, str], ParamNeuron],
               trade_time: Optional[datetime] = None) -> List[Tuple[str, float]]:
        """Returns list of (param_id, credit) with STDP temporal weighting."""
        credits = []
        reward = 1.0 if trade_pnl > 0 else -1.0
        pnl_mag = min(abs(trade_pnl), 20.0) / 20.0
        now = trade_time or datetime.now(tz=timezone.utc)

        for (pid, r), neuron in neurons.items():
            if r != regime and r != "_global":
                continue

            # Base credit by organ type
            if neuron.organ in self.SIGNAL_ORGANS:
                base_credit = pnl_mag * max(confidence, 0.1) * 0.5
            elif neuron.organ in self.SIZING_ORGANS:
                base_credit = pnl_mag * max(stake_pct, 0.01) * 0.3
            elif neuron.organ in self.DEFENSE_ORGANS:
                dur_norm = min(duration_hours / 24.0, 1.0)
                base_credit = pnl_mag * dur_norm * 0.4
            else:
                base_credit = pnl_mag * 0.1

            # Phase 25: STDP temporal weighting — recent updates get more credit
            temporal_w = self._stdp_temporal_weight(neuron, now)
            credit = reward * base_credit * temporal_w

            credits.append((pid, credit))

        return credits


# ═══════════════════════════════════════════════════════════════════════════
# Phase 25: CEREBELLUM — Time-of-day performance model
# ═══════════════════════════════════════════════════════════════════════════

class Cerebellum:
    """24-slot hour-of-day model. Learns which trading hours are profitable."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.slots = [{"wins": 1, "losses": 1, "avg_pnl": 0.0, "n": 0} for _ in range(24)]

    def record_outcome(self, hour: int, won: bool, pnl_pct: float = 0.0):
        hour = hour % 24
        s = self.slots[hour]
        if won:
            s["wins"] += 1
        else:
            s["losses"] += 1
        s["n"] += 1
        s["avg_pnl"] = (s["avg_pnl"] * (s["n"] - 1) + pnl_pct) / s["n"] if s["n"] > 0 else pnl_pct

    def get_hour_multiplier(self, hour: int) -> float:
        """Returns 0.6-1.4 multiplier based on historical hourly performance."""
        hour = hour % 24
        s = self.slots[hour]
        total = s["wins"] + s["losses"]
        if total < 3:
            return 1.0  # Not enough data
        win_rate = s["wins"] / total
        return max(0.6, min(1.4, 0.6 + 0.8 * win_rate))

    def get_best_hours(self, top_n: int = 6) -> List[int]:
        hours_by_wr = []
        for h in range(24):
            s = self.slots[h]
            total = s["wins"] + s["losses"]
            wr = s["wins"] / total if total > 2 else 0.5
            hours_by_wr.append((h, wr))
        hours_by_wr.sort(key=lambda x: x[1], reverse=True)
        return [h for h, _ in hours_by_wr[:top_n]]


# ═══════════════════════════════════════════════════════════════════════════
# Phase 25: PREDICTIVE MODEL — Free Energy Principle (Friston)
# ═══════════════════════════════════════════════════════════════════════════

class PredictiveModel:
    """
    Predicts optimal parameter state based on current market fingerprint.
    Prediction error drives learning rate: big error → learn faster.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._last_prediction: Dict[str, float] = {}
        self._last_error: float = 0.5

    def predict_expected_pnl(self, fingerprint: str) -> float:
        """Given fingerprint, predict expected PnL from historical episodes."""
        try:
            conn = sqlite3.connect(self.db_path, timeout=10)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT outcome_pnl FROM hippocampus_episodes "
                "WHERE fingerprint = ? ORDER BY timestamp DESC LIMIT 10",
                (fingerprint,)).fetchall()
            conn.close()
            if len(rows) < 2:
                return 0.0
            avg_pnl = sum(r["outcome_pnl"] for r in rows) / len(rows)
            self._last_prediction = {"expected_pnl": avg_pnl, "n_episodes": len(rows)}
            return avg_pnl
        except Exception:
            return 0.0

    def compute_prediction_error(self, actual_pnl: float) -> float:
        """Free Energy = prediction error. Large error → need aggressive learning."""
        expected = self._last_prediction.get("expected_pnl", 0)
        if expected == 0 and actual_pnl == 0:
            return 0.0
        error = abs(actual_pnl - expected) / max(abs(expected), abs(actual_pnl), 1.0)
        self._last_error = min(1.0, error)
        return self._last_error

    def get_lr_boost(self) -> float:
        """Convert prediction error to learning rate boost. High error → 2x learning."""
        return 1.0 + self._last_error  # Range: 1.0 (perfect prediction) to 2.0 (max error)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 25: INTEROCEPTION — 8-sensor deep self-monitoring
# ═══════════════════════════════════════════════════════════════════════════

class Interoception:
    """The organism's deep internal state awareness — 8 sensors."""

    def __init__(self):
        self.sensors = {
            "param_drift_velocity": 0.0,    # How fast params are changing
            "belief_width_avg": 0.5,         # Average uncertainty across neurons
            "prediction_error_avg": 0.5,     # Average prediction error (Free Energy)
            "hormone_stability": 1.0,        # How stable hormones are (low variance = stable)
            "trade_frequency": 0.0,          # Trades per hour
            "win_rate_30d": 0.5,             # Rolling 30-day win rate
            "data_completeness": 0.5,        # How many data sources are active
            "consec_same_direction": 0,       # How many consecutive same-direction trades
        }

    def update(self, neurons: Dict, hormones: 'Hormones', trade_count: int,
               consec_wins: int, consec_losses: int, prediction_error: float):
        """Update all 8 sensors from current organism state."""
        # Belief width: average uncertainty
        widths = [n.belief_width() for n in neurons.values()]
        self.sensors["belief_width_avg"] = sum(widths) / len(widths) if widths else 0.5

        # Param drift: average theta_m (BCM activity) across neurons
        thetas = [n.theta_m for n in neurons.values()]
        self.sensors["param_drift_velocity"] = sum(thetas) / len(thetas) if thetas else 0.0

        # Prediction error
        self.sensors["prediction_error_avg"] = prediction_error

        # Hormone stability: inverse of recent hormone changes
        self.sensors["hormone_stability"] = max(0.0, 1.0 - hormones._stress * 0.5)

        # Win rate proxy
        total = consec_wins + consec_losses
        self.sensors["win_rate_30d"] = consec_wins / max(total, 1) if total > 0 else 0.5

        # Data completeness = serotonin proxy
        self.sensors["data_completeness"] = hormones._info_q

        # Consecutive same direction
        if consec_wins > 0:
            self.sensors["consec_same_direction"] = consec_wins
        elif consec_losses > 0:
            self.sensors["consec_same_direction"] = -consec_losses

    def get_organism_health(self) -> float:
        """Composite health score 0.0 (dying) to 1.0 (thriving)."""
        health = 0.0
        health += min(self.sensors["win_rate_30d"], 1.0) * 0.3
        health += self.sensors["hormone_stability"] * 0.2
        health += self.sensors["data_completeness"] * 0.2
        health += max(0, 1.0 - self.sensors["prediction_error_avg"]) * 0.2
        health += max(0, 1.0 - abs(self.sensors["param_drift_velocity"]) * 5) * 0.1
        return max(0.0, min(1.0, health))


# ═══════════════════════════════════════════════════════════════════════════
# Phase 25: MIRROR NEURONS — Modeling other traders via market microstructure
# ═══════════════════════════════════════════════════════════════════════════

class MirrorNeurons:
    """Models crowd behavior from funding rate, L/S ratio, and open interest."""

    def __init__(self):
        self.crowd_direction = "NEUTRAL"
        self.crowd_intensity = 0.0  # 0-1
        self.crowd_is_wrong_rate = 0.5  # Historical: how often the crowd is wrong

    def analyze_crowd(self, funding_rate: float = 0, ls_ratio: float = 1.0,
                      oi_change_pct: float = 0) -> dict:
        """Infer crowd positioning from market microstructure."""
        signals = []

        # Funding rate: positive = crowd is long, negative = crowd is short
        if funding_rate > 0.0005:
            signals.append(("LONG", min(funding_rate / 0.002, 1.0)))
        elif funding_rate < -0.0005:
            signals.append(("SHORT", min(abs(funding_rate) / 0.002, 1.0)))

        # L/S ratio: >1.2 = crowd is long, <0.8 = crowd is short
        if ls_ratio > 1.2:
            signals.append(("LONG", min((ls_ratio - 1.0) / 1.0, 1.0)))
        elif ls_ratio < 0.8:
            signals.append(("SHORT", min((1.0 - ls_ratio) / 0.5, 1.0)))

        # OI increasing with positive funding = more longs piling in
        if oi_change_pct > 5 and funding_rate > 0:
            signals.append(("LONG", 0.3))
        elif oi_change_pct > 5 and funding_rate < 0:
            signals.append(("SHORT", 0.3))

        if not signals:
            self.crowd_direction = "NEUTRAL"
            self.crowd_intensity = 0.0
            return {"direction": "NEUTRAL", "intensity": 0.0, "contrarian_signal": 0.0}

        # Aggregate
        long_score = sum(s for d, s in signals if d == "LONG")
        short_score = sum(s for d, s in signals if d == "SHORT")

        if long_score > short_score:
            self.crowd_direction = "LONG"
            self.crowd_intensity = min(1.0, long_score)
        else:
            self.crowd_direction = "SHORT"
            self.crowd_intensity = min(1.0, short_score)

        # Contrarian signal: high intensity = crowd is likely wrong
        contrarian = self.crowd_intensity * self.crowd_is_wrong_rate
        return {
            "direction": self.crowd_direction,
            "intensity": round(self.crowd_intensity, 3),
            "contrarian_signal": round(contrarian, 3),
        }

    def record_outcome(self, crowd_was_right: bool):
        """Update crowd accuracy rate."""
        lr = 0.05
        if crowd_was_right:
            self.crowd_is_wrong_rate = max(0.1, self.crowd_is_wrong_rate - lr)
        else:
            self.crowd_is_wrong_rate = min(0.9, self.crowd_is_wrong_rate + lr)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 25: ADAPTIVE IMMUNITY — B-cell/T-cell threat memory
# ═══════════════════════════════════════════════════════════════════════════

class AdaptiveImmunity:
    """
    Long-term threat memory beyond temporary pair bans.
    B-cells: remember specific threat PATTERNS (not just pairs).
    T-cells: rapid response when recognized threat reappears.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._bcells: Dict[str, dict] = {}  # fingerprint → {severity, encounters, antibody}

    def encounter_threat(self, fingerprint: str, loss_pct: float):
        """B-cell: remember this threat pattern."""
        if fingerprint in self._bcells:
            cell = self._bcells[fingerprint]
            cell["encounters"] += 1
            cell["severity"] = min(1.0, cell["severity"] + abs(loss_pct) * 0.05)
            cell["antibody"] = min(2.0, 1.0 + cell["encounters"] * 0.1)
        else:
            self._bcells[fingerprint] = {
                "encounters": 1,
                "severity": min(1.0, abs(loss_pct) * 0.1),
                "antibody": 1.0,
                "first_seen": datetime.now(tz=timezone.utc).isoformat(),
            }
        try:
            conn = sqlite3.connect(self.db_path, timeout=10)
            cell = self._bcells[fingerprint]
            conn.execute(
                "INSERT OR REPLACE INTO immune_bcells "
                "(threat_fingerprint, severity, encounter_count, antibody_strength, last_encounter) "
                "VALUES (?, ?, ?, ?, ?)",
                (fingerprint, cell["severity"], cell["encounters"], cell["antibody"],
                 datetime.now(tz=timezone.utc).isoformat()))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"[AdaptiveImmunity] Persist failed: {e}")

    def check_threat(self, fingerprint: str) -> dict:
        """T-cell: is this a known threat? How strong is the antibody?"""
        cell = self._bcells.get(fingerprint)
        if not cell:
            return {"known_threat": False, "antibody": 0.0, "sizing_reduction": 1.0}
        # Antibody strength → sizing reduction
        sizing_mult = max(0.2, 1.0 / cell["antibody"])
        return {
            "known_threat": True,
            "antibody": cell["antibody"],
            "encounters": cell["encounters"],
            "severity": cell["severity"],
            "sizing_reduction": sizing_mult,
        }

    def load_from_db(self):
        """Load B-cells from SQLite on startup."""
        try:
            conn = sqlite3.connect(self.db_path, timeout=10)
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM immune_bcells").fetchall()
            conn.close()
            for r in rows:
                self._bcells[r["threat_fingerprint"]] = {
                    "encounters": r["encounter_count"],
                    "severity": r["severity"],
                    "antibody": r["antibody_strength"],
                }
            if self._bcells:
                logger.info(f"[AdaptiveImmunity] Loaded {len(self._bcells)} B-cells from DB")
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════════
# Phase 25: DEFAULT MODE NETWORK — Idle background processing
# ═══════════════════════════════════════════════════════════════════════════

class DefaultModeNetwork:
    """Brain's idle processing: counterfactual analysis + synapse discovery."""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def run_idle_cycle(self, neurons: Dict, hippocampus: Hippocampus) -> dict:
        """Run when no trades are happening. Discovers patterns and optimizations."""
        results = {"counterfactuals": [], "discoveries": []}

        # 1. Counterfactual: what if the worst trades had different params?
        try:
            conn = sqlite3.connect(self.db_path, timeout=10)
            conn.row_factory = sqlite3.Row
            worst = conn.execute(
                "SELECT * FROM hippocampus_episodes "
                "ORDER BY outcome_pnl ASC LIMIT 5").fetchall()
            conn.close()

            for ep in worst:
                pnl = ep["outcome_pnl"]
                fp = ep["fingerprint"]
                # Find similar episodes with POSITIVE outcomes
                good = hippocampus.recall(fp, k=5)
                good_ones = [g for g in good if g.get("outcome_pnl", 0) > 0]
                if good_ones:
                    results["counterfactuals"].append({
                        "bad_pnl": pnl,
                        "fingerprint": fp,
                        "good_similar_count": len(good_ones),
                        "avg_good_pnl": sum(g["outcome_pnl"] for g in good_ones) / len(good_ones),
                    })
        except Exception:
            pass

        # 2. Synapse discovery: find correlated neuron pairs
        # (neurons that frequently change in the same direction)
        try:
            neuron_list = list(neurons.values())
            if len(neuron_list) > 10:
                sample = random.sample(neuron_list, min(20, len(neuron_list)))
                for i, n1 in enumerate(sample):
                    for n2 in sample[i + 1:]:
                        if n1.organ != n2.organ and n1.theta_m > 0.01 and n2.theta_m > 0.01:
                            # Both neurons are active — potential synapse
                            results["discoveries"].append({
                                "source": n1.param_id,
                                "target": n2.param_id,
                                "co_activity": round(min(n1.theta_m, n2.theta_m), 4),
                            })
        except Exception:
            pass

        logger.info(f"[DMN] Idle cycle: {len(results['counterfactuals'])} counterfactuals, "
                    f"{len(results['discoveries'])} synapse candidates")
        return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase 25: SLEEP CONSOLIDATION — Weekly memory processing
# ═══════════════════════════════════════════════════════════════════════════

class SleepConsolidation:
    """Weekly consolidation: experience replay + synapse pruning + habit review."""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def run_consolidation(self, neurons: Dict, synapses: SynapseNetwork,
                          ganglia: BasalGanglia, hippocampus: Hippocampus) -> dict:
        """Full sleep cycle. Called weekly by scheduler."""
        t0 = time.time()
        results = {"replayed": 0, "pruned": 0, "broken": 0}

        # 1. EXPERIENCE REPLAY: review recent episodes
        try:
            conn = sqlite3.connect(self.db_path, timeout=10)
            conn.row_factory = sqlite3.Row
            recent = conn.execute(
                "SELECT * FROM hippocampus_episodes "
                "ORDER BY timestamp DESC LIMIT 50").fetchall()
            conn.close()
            results["replayed"] = len(recent)

            # Identify consistently profitable fingerprints → strengthen related neurons
            fp_pnls: Dict[str, List[float]] = {}
            for ep in recent:
                fp = ep["fingerprint"]
                fp_pnls.setdefault(fp, []).append(ep["outcome_pnl"])

            for fp, pnls in fp_pnls.items():
                if len(pnls) >= 3:
                    avg = sum(pnls) / len(pnls)
                    if avg > 2.0:
                        logger.info(f"[Sleep] Profitable pattern found: {fp[:50]}... avg_pnl={avg:.1f}%")
        except Exception as e:
            logger.debug(f"[Sleep] Replay failed: {e}")

        # 2. SYNAPSE PRUNING: weaken synapses that haven't fired
        for source, edges in synapses._edges.items():
            for i, (tgt, weight, stype) in enumerate(edges):
                # Decay weight slightly each week
                new_weight = max(0.05, weight * 0.95)
                edges[i] = (tgt, new_weight, stype)
                if new_weight < 0.1:
                    results["pruned"] += 1

        # 3. HABIT REVIEW: break habits that are underperforming
        for (pid, regime), neuron in neurons.items():
            if neuron.prior_strength > 15.0:  # This is a habit
                # Check: has this neuron's organ been losing recently?
                # If so, break the habit (reduce prior_strength)
                if neuron.theta_m < 0.001 and neuron.update_count > 100:
                    # Stale habit — hasn't been updated in a while
                    neuron.prior_strength = max(5.0, neuron.prior_strength - 5.0)
                    results["broken"] += 1

        duration = time.time() - t0
        logger.info(f"[Sleep] Consolidation done in {duration:.1f}s: "
                    f"{results['replayed']} replayed, {results['pruned']} pruned, "
                    f"{results['broken']} habits broken")
        return results


# ═══════════════════════════════════════════════════════════════════════════
# Phase 25: NEUROEVOLUTION — Population-based parameter evolution
# ═══════════════════════════════════════════════════════════════════════════

class NeuroEvolution:
    """Maintains a population of parameter snapshots. Weekly tournament selection."""

    def __init__(self, db_path: str, population_size: int = 5):
        self.db_path = db_path
        self.population_size = population_size

    def snapshot_current(self, neurons: Dict, fitness: float):
        """Save current organism state as a genome in the population."""
        params = {f"{pid}:{r}": n.current_val for (pid, r), n in neurons.items()
                  if r == "_global"}  # Only global regime for simplicity
        try:
            conn = sqlite3.connect(self.db_path, timeout=10)
            conn.execute(
                "INSERT INTO evolution_population "
                "(params_json, fitness, generation, created_at, is_active) "
                "VALUES (?, ?, (SELECT COALESCE(MAX(generation),0)+1 FROM evolution_population), ?, 0)",
                (json.dumps(params), fitness, datetime.now(tz=timezone.utc).isoformat()))
            # Keep only top N genomes
            conn.execute(
                "DELETE FROM evolution_population WHERE genome_id NOT IN "
                "(SELECT genome_id FROM evolution_population ORDER BY fitness DESC LIMIT ?)",
                (self.population_size,))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.debug(f"[NeuroEvolution] Snapshot failed: {e}")

    def get_best_genome(self) -> Optional[Dict[str, float]]:
        """Get the highest-fitness genome from the population."""
        try:
            conn = sqlite3.connect(self.db_path, timeout=10)
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT params_json, fitness FROM evolution_population "
                "ORDER BY fitness DESC LIMIT 1").fetchone()
            conn.close()
            if row:
                return {"params": json.loads(row["params_json"]), "fitness": row["fitness"]}
        except Exception:
            pass
        return None

    def run_tournament(self, neurons: Dict, recent_pnl: float):
        """Weekly: save current state, compare with population, optionally blend."""
        self.snapshot_current(neurons, recent_pnl)
        best = self.get_best_genome()
        if not best or recent_pnl >= best["fitness"]:
            logger.info(f"[NeuroEvolution] Current organism IS the best (fitness={recent_pnl:.2f})")
            return

        # Current is worse than historical best — blend 10% toward best
        best_params = best["params"]
        blended = 0
        for (pid, r), neuron in neurons.items():
            key = f"{pid}:{r}"
            if key in best_params and r == "_global":
                target = best_params[key]
                neuron.current_val = neuron.current_val * 0.9 + target * 0.1
                neuron.current_val = max(neuron.min_bound, min(neuron.max_bound, neuron.current_val))
                blended += 1

        logger.info(f"[NeuroEvolution] Blended {blended} params 10% toward best genome "
                    f"(best_fitness={best['fitness']:.2f}, current={recent_pnl:.2f})")


# ═══════════════════════════════════════════════════════════════════════════
# NEURAL ORGANISM — The Central Coordinator (Singleton)
# ═══════════════════════════════════════════════════════════════════════════

class NeuralOrganism:
    """
    The living organism. Coordinates all 14 subsystems.
    ALL consumer files call get_param() to read adaptive parameter values.
    """

    def __init__(self, db_path: str = AI_DB_PATH):
        self.db_path = db_path
        self._neurons: Dict[Tuple[str, str], ParamNeuron] = {}
        # Phase 24: Original 8 subsystems
        self.hormones = Hormones()
        self.amygdala = Amygdala()
        self.hippocampus = Hippocampus(db_path)
        self.synapses = SynapseNetwork()
        self.prefrontal = PrefrontalCortex()
        self.ganglia = BasalGanglia()
        self.proprioception = Proprioception()
        self.immune = ImmuneMemory(db_path)
        self.credit_assigner = CreditAssigner()
        # Phase 25: 6 new subsystems (total 14)
        self.cerebellum = Cerebellum(db_path)
        self.predictive = PredictiveModel(db_path)
        self.interoception = Interoception()
        self.mirror = MirrorNeurons()
        self.immunity = AdaptiveImmunity(db_path)
        self.dmn = DefaultModeNetwork(db_path)
        self.sleep = SleepConsolidation(db_path)
        self.evolution = NeuroEvolution(db_path)
        # State
        self._last_persist = time.time()
        self._last_decay = time.time()
        self._trade_count = 0
        self._consec_wins = 0
        self._consec_losses = 0
        self._cumulative_pnl = 0.0
        self._ensure_tables()
        self._load_or_seed()
        self.immunity.load_from_db()
        self._load_cerebellum()
        logger.info(f"[NeuralOrganism] Initialized: {len(self._neurons)} neurons, "
                    f"{len(PARAM_REGISTRY)} params × {len(REGIMES)} regimes, 14 subsystems")

    # ─── SQLite Setup ───

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _ensure_tables(self):
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS neuron_state (
                    param_id TEXT NOT NULL, organ TEXT NOT NULL,
                    regime TEXT NOT NULL DEFAULT '_global',
                    current_val REAL NOT NULL, default_val REAL NOT NULL,
                    min_bound REAL NOT NULL, max_bound REAL NOT NULL,
                    alpha REAL DEFAULT 2.0, beta_param REAL DEFAULT 2.0,
                    prior_strength REAL DEFAULT 5.0, frozen INTEGER DEFAULT 0,
                    update_count INTEGER DEFAULT 0,
                    activity_ema REAL DEFAULT 0.0, theta_m REAL DEFAULT 0.0,
                    last_updated TEXT,
                    PRIMARY KEY (param_id, regime)
                );
                CREATE TABLE IF NOT EXISTS neuron_synapses (
                    source TEXT NOT NULL, target TEXT NOT NULL,
                    weight REAL DEFAULT 0.5, synapse_type TEXT DEFAULT 'excitatory',
                    fire_count INTEGER DEFAULT 0,
                    PRIMARY KEY (source, target)
                );
                CREATE TABLE IF NOT EXISTS hormone_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    cortisol REAL DEFAULT 1.0, dopamine REAL DEFAULT 1.0,
                    serotonin REAL DEFAULT 1.0, adrenaline REAL DEFAULT 1.0,
                    market_stress REAL DEFAULT 0.0, portfolio_health REAL DEFAULT 0.5,
                    info_quality REAL DEFAULT 0.5, updated_at TEXT
                );
                CREATE TABLE IF NOT EXISTS hippocampus_episodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pair TEXT, fingerprint TEXT NOT NULL,
                    outcome_pnl REAL NOT NULL, regime TEXT, timestamp TEXT
                );
                CREATE TABLE IF NOT EXISTS amygdala_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    fear_level REAL DEFAULT 0.0, peak_fear REAL DEFAULT 0.0,
                    peak_time TEXT, tier TEXT DEFAULT 'normal', updated_at TEXT
                );
                CREATE TABLE IF NOT EXISTS immune_memory (
                    pair TEXT NOT NULL, loss_pct REAL NOT NULL,
                    ban_until TEXT, consecutive_losses INTEGER DEFAULT 1,
                    regime TEXT, timestamp TEXT
                );
                CREATE TABLE IF NOT EXISTS organism_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_pair TEXT, trade_pnl REAL, hormones TEXT,
                    fear_tier TEXT, overrides TEXT, phase TEXT, timestamp TEXT
                );
                -- Phase 25: New tables
                CREATE TABLE IF NOT EXISTS cerebellum_hours (
                    hour INTEGER PRIMARY KEY, wins INTEGER DEFAULT 1,
                    losses INTEGER DEFAULT 1, avg_pnl REAL DEFAULT 0.0, updated_at TEXT
                );
                CREATE TABLE IF NOT EXISTS interoception_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    param_drift REAL DEFAULT 0.0, belief_width REAL DEFAULT 0.5,
                    pred_error REAL DEFAULT 0.5, hormone_stability REAL DEFAULT 1.0,
                    trade_freq REAL DEFAULT 0.0, win_rate REAL DEFAULT 0.5,
                    data_completeness REAL DEFAULT 0.5, consec_dir INTEGER DEFAULT 0,
                    updated_at TEXT
                );
                CREATE TABLE IF NOT EXISTS immune_bcells (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    threat_fingerprint TEXT NOT NULL UNIQUE,
                    severity REAL DEFAULT 0.0, encounter_count INTEGER DEFAULT 1,
                    last_encounter TEXT, antibody_strength REAL DEFAULT 1.0
                );
                CREATE TABLE IF NOT EXISTS evolution_population (
                    genome_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    params_json TEXT NOT NULL, fitness REAL DEFAULT 0.0,
                    novelty_score REAL DEFAULT 0.0, generation INTEGER DEFAULT 0,
                    created_at TEXT, is_active INTEGER DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS sleep_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_date TEXT, episodes_replayed INTEGER,
                    synapses_pruned INTEGER, habits_broken INTEGER,
                    counterfactuals TEXT, duration_sec REAL, timestamp TEXT
                );
                CREATE TABLE IF NOT EXISTS dmn_discoveries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    discovery_type TEXT, description TEXT,
                    param_ids TEXT, potential_improvement REAL, timestamp TEXT
                );
            """)
            # Seed hormones row if missing
            if conn.execute("SELECT COUNT(*) FROM hormone_state").fetchone()[0] == 0:
                conn.execute("INSERT INTO hormone_state (id) VALUES (1)")
            if conn.execute("SELECT COUNT(*) FROM amygdala_state").fetchone()[0] == 0:
                conn.execute("INSERT INTO amygdala_state (id) VALUES (1)")
            conn.commit()

    # ─── Cold Start / Load ───

    def _load_or_seed(self):
        """Load neurons from SQLite, or seed from PARAM_REGISTRY on first run."""
        with self._get_conn() as conn:
            rows = conn.execute("SELECT * FROM neuron_state").fetchall()

        if rows:
            for row in rows:
                # Parse last_update_time if present
                lut = None
                lut_str = row["last_updated"]
                if lut_str:
                    try:
                        lut = datetime.fromisoformat(lut_str.replace("Z", "+00:00"))
                        if lut.tzinfo is None:
                            lut = lut.replace(tzinfo=timezone.utc)
                    except (ValueError, TypeError):
                        pass
                n = ParamNeuron(
                    param_id=row["param_id"], organ=row["organ"], regime=row["regime"],
                    current_val=row["current_val"], default_val=row["default_val"],
                    min_bound=row["min_bound"], max_bound=row["max_bound"],
                    alpha=row["alpha"], beta_param=row["beta_param"],
                    prior_strength=row["prior_strength"], frozen=bool(row["frozen"]),
                    update_count=row["update_count"],
                    activity_ema=row["activity_ema"] if "activity_ema" in row.keys() else 0.0,
                    theta_m=row["theta_m"] if "theta_m" in row.keys() else 0.0,
                    last_update_time=lut,
                )
                self._neurons[(n.param_id, n.regime)] = n
            logger.info(f"[NeuralOrganism] Loaded {len(self._neurons)} neurons from SQLite (BCM+STDP fields)")
            # Migrate: add any new params from PARAM_REGISTRY not yet in DB
            self._migrate_new_params()
        else:
            self._cold_start()

    def _cold_start(self):
        """First run: seed all neurons from PARAM_REGISTRY."""
        neurons_to_insert = []
        for param_id, spec in PARAM_REGISTRY.items():
            regime_defaults = spec.get("regime_defaults", {})
            for regime in REGIMES:
                default = regime_defaults.get(regime, spec["default"])
                # Informative prior: position in [min, max] range
                param_range = spec["max"] - spec["min"]
                if param_range > 0:
                    position = (default - spec["min"]) / param_range
                    position = max(0.01, min(0.99, position))
                else:
                    position = 0.5
                prior = 5.0
                alpha = prior * position
                beta = prior * (1.0 - position)

                neuron = ParamNeuron(
                    param_id=param_id, organ=spec["organ"], regime=regime,
                    current_val=default, default_val=spec["default"],
                    min_bound=spec["min"], max_bound=spec["max"],
                    alpha=max(alpha, 0.1), beta_param=max(beta, 0.1),
                )
                self._neurons[(param_id, regime)] = neuron
                neurons_to_insert.append(neuron)

        self._persist_batch(neurons_to_insert)
        logger.info(f"[NeuralOrganism:ColdStart] Seeded {len(neurons_to_insert)} neurons "
                    f"({len(PARAM_REGISTRY)} params × {len(REGIMES)} regimes)")

    def _migrate_new_params(self):
        """Add new params from PARAM_REGISTRY that aren't in DB yet."""
        existing = set(self._neurons.keys())
        new_neurons = []
        for param_id, spec in PARAM_REGISTRY.items():
            regime_defaults = spec.get("regime_defaults", {})
            for regime in REGIMES:
                if (param_id, regime) not in existing:
                    default = regime_defaults.get(regime, spec["default"])
                    neuron = ParamNeuron(
                        param_id=param_id, organ=spec["organ"], regime=regime,
                        current_val=default, default_val=spec["default"],
                        min_bound=spec["min"], max_bound=spec["max"],
                    )
                    self._neurons[(param_id, regime)] = neuron
                    new_neurons.append(neuron)
        if new_neurons:
            self._persist_batch(new_neurons)
            logger.info(f"[NeuralOrganism:Migrate] Added {len(new_neurons)} new neurons")

    # ─── PUBLIC API: get_param / get_organ ───

    def get_param(self, param_id: str, regime: str = "_global") -> float:
        """
        Primary API. Returns FULLY MODULATED parameter value.
        Modulation chain: Hormones → Cerebellum → MirrorNeurons → AdaptiveImmunity → Interoception
        O(1) in-memory lookup. No SQLite hit.
        """
        neuron = self._neurons.get((param_id, regime))
        if neuron is None and regime != "_global":
            neuron = self._neurons.get((param_id, "_global"))
        if neuron is None:
            spec = PARAM_REGISTRY.get(param_id)
            if spec:
                return spec["default"]
            return None  # Unknown param → _p() will use its fallback

        if neuron.frozen:
            return neuron.current_val

        value = neuron.current_val
        h = self.hormones

        # ═══ Layer 1: HORMONE modulation (global broadcast) ═══
        if neuron.organ in ("sizing", "strategy_leverage", "risk", "immune",
                            "autonomy", "opp_weights", "strategy_trailing",
                            "strategy_dca", "strategy_stoploss"):
            value *= h.cortisol * h.dopamine
        elif neuron.organ in ("evidence_weights", "synthesis", "coordinator",
                              "q1_trend", "q2_momentum", "q3_crowd",
                              "q4_evidence", "q5_macro", "q6_risk",
                              "contradiction", "evidence_first"):
            value *= h.serotonin

        # Adrenaline freeze: non-essential → snap to default
        if h.adrenaline == 0.0 and neuron.organ not in ("sizing", "risk", "strategy_stoploss"):
            return neuron.default_val

        # ═══ Layer 2: CEREBELLUM time-of-day modulation (sizing organs only) ═══
        if neuron.organ in ("sizing", "strategy_leverage", "opp_weights"):
            hour_mult = self.cerebellum.get_hour_multiplier(datetime.now(tz=timezone.utc).hour)
            value *= hour_mult

        # ═══ Layer 3: MIRROR NEURONS crowd contrarian (confidence/sizing organs) ═══
        if neuron.organ in ("sizing", "strategy_leverage") and self.mirror.crowd_intensity > 0.5:
            # High crowd intensity → slight sizing reduction (contrarian caution)
            contrarian_adj = 1.0 - (self.mirror.crowd_intensity * self.mirror.crowd_is_wrong_rate * 0.15)
            value *= max(0.7, contrarian_adj)

        # ═══ Layer 4: INTEROCEPTION organism health (all sizing organs) ═══
        if neuron.organ in ("sizing", "strategy_leverage", "risk"):
            org_health = self.interoception.get_organism_health()
            if org_health < 0.4:
                # Unhealthy organism → reduce sizing further
                value *= max(0.6, 0.5 + org_health)

        return max(neuron.min_bound, min(neuron.max_bound, value))

    def get_organ(self, organ: str, regime: str = "_global") -> Dict[str, float]:
        """Get all params in an organ as a dict. Keys = last segment of param_id."""
        result = {}
        for (pid, r), neuron in self._neurons.items():
            if neuron.organ == organ and r == regime:
                key = pid.split(".")[-1]
                result[key] = self.get_param(pid, regime)
        if not result and regime != "_global":
            return self.get_organ(organ, "_global")

        # Apply sum constraint if needed
        target = ORGAN_CONSTRAINTS.get(organ)
        if target and result:
            total = sum(result.values())
            if total > 0:
                scale = target / total
                result = {k: v * scale for k, v in result.items()}
        return result

    # ─── UPDATE CYCLE — Runs when a trade closes ───

    def update_cycle(self, pair: str, pnl_pct: float, regime: str = "transitional",
                     confidence: float = 0.5, exit_reason: str = "",
                     duration_hours: float = 1.0, stake_amount: float = 0,
                     fng: Optional[int] = None, adx: float = 20,
                     funding_rate: float = 0, active_sources: int = 4,
                     balance_vs_peak: float = 1.0, ls_ratio: float = 1.0):
        """
        Full 16-step neural update cycle. Called from confirm_trade_exit.
        Phase 25: expanded from 10 to 16 steps with all brain subsystems.
        """
        self._trade_count += 1
        self._cumulative_pnl += pnl_pct
        won = pnl_pct > 0
        if won:
            self._consec_wins += 1
            self._consec_losses = 0
        else:
            self._consec_losses += 1
            self._consec_wins = 0

        portfolio_value = max(balance_vs_peak, 1.0)
        stake_pct = stake_amount / portfolio_value if portfolio_value > 0 else 0.01
        trade_hour = datetime.now(tz=timezone.utc).hour

        # ═══ 1. SENSE — Proprioception + Interoception (deep self-awareness) ═══
        self_state = self.proprioception.assess(
            self._neurons, self._consec_wins, self._consec_losses)

        # ═══ 2. PREDICT — Free Energy Principle (predict before observing) ═══
        stress = self.hormones._stress
        fingerprint = Hippocampus.get_fingerprint(fng, regime, adx, funding_rate,
                                                   self._consec_losses, stress)
        predicted_pnl = self.predictive.predict_expected_pnl(fingerprint)

        # ═══ 3. MIRROR — Model other traders via market microstructure ═══
        crowd = self.mirror.analyze_crowd(funding_rate, ls_ratio)

        # ═══ 4. HORMONES + ALLOSTASIS — global broadcast with anticipation ═══
        self.hormones.compute(
            fng=fng, drawdown_pct=0, consec_wins=self._consec_wins,
            consec_losses=self._consec_losses, active_sources=active_sources,
            balance_vs_peak=balance_vs_peak)

        # ═══ 5. FEAR — Amygdala graduated response ═══
        fear_response = (self.amygdala.process_loss(pnl_pct) if not won
                         else {"fear_level": 0, "learning_mult": 1.0, "sizing_mult": 1.0, "tier": "normal"})

        # ═══ 6. MEMORY — Hippocampus store + recall ═══
        self.hippocampus.store_episode(pair, fingerprint, pnl_pct, regime)
        recalls = self.hippocampus.recall(fingerprint, k=5)

        # ═══ 7. PREDICTION ERROR — Free Energy (predicted vs actual) ═══
        pred_error = self.predictive.compute_prediction_error(pnl_pct)
        pred_lr_boost = self.predictive.get_lr_boost()

        # ═══ 8. INTEROCEPTION — 8-sensor deep self-monitoring ═══
        self.interoception.update(
            self._neurons, self.hormones, self._trade_count,
            self._consec_wins, self._consec_losses, pred_error)

        # ═══ 9. LEARN — BCM nudge + STDP temporal credit ═══
        lr_mod = (self_state["lr_mod"] *
                  fear_response.get("learning_mult", 1.0) *
                  pred_lr_boost)  # Phase 25: prediction error boosts learning
        now = datetime.now(tz=timezone.utc)
        credits = self.credit_assigner.assign(
            pnl_pct, confidence, stake_pct, duration_hours, regime,
            self._neurons, trade_time=now)

        changed_neurons = []
        for param_id, credit in credits:
            neuron = self._neurons.get((param_id, regime)) or self._neurons.get((param_id, "_global"))
            if neuron:
                neuron.nudge(credit, magnitude=abs(credit) * lr_mod)
                changed_neurons.append(param_id)

        # ═══ 10. PROPAGATE — Synapse firing (1-hop) ═══
        for param_id in changed_neurons[:5]:
            delta = next((c for pid, c in credits if pid == param_id), 0)
            self.synapses.propagate(param_id, delta, self._neurons, regime)

        # ═══ 11. TIMING — Cerebellum hour-of-day performance ═══
        self.cerebellum.record_outcome(trade_hour, won, pnl_pct)

        # ═══ 12. HABITUATE — BasalGanglia consolidation ═══
        for key, neuron in self._neurons.items():
            if key[1] == regime:
                self.ganglia.check_consolidation(neuron)

        # ═══ 13. IMMUNE — Pair ban + Adaptive Immunity B-cell ═══
        ban_minutes = 0
        if not won:
            self.immune.record_loss(pair, pnl_pct, regime)
            ban_minutes = self.immune.compute_ban(pair, pnl_pct, self.get_param)
            self.immunity.encounter_threat(fingerprint, pnl_pct)
        else:
            self.immune.record_win(pair)
        # Mirror neuron feedback: was the crowd right?
        crowd_right = (crowd["direction"] == "LONG" and won) or (crowd["direction"] == "SHORT" and not won)
        self.mirror.record_outcome(crowd_right)

        # ═══ 14. VETO — PrefrontalCortex hard rules + AdaptiveImmunity ═══
        threat_check = self.immunity.check_threat(fingerprint)
        overrides = self.prefrontal.evaluate(
            self._neurons, self.hormones, recalls, self.amygdala)
        # AdaptiveImmunity: known threat → reduce sizing neurons
        if threat_check.get("known_threat"):
            sizing_reduction = threat_check["sizing_reduction"]
            for (pid, r), n in self._neurons.items():
                if n.organ in ("sizing", "strategy_leverage") and r in (regime, "_global"):
                    n.current_val = max(n.min_bound, n.current_val * sizing_reduction)
            overrides.append(f"ADAPTIVE IMMUNITY: known threat (antibody={threat_check['antibody']:.1f}, "
                           f"encounters={threat_check['encounters']}) → sizing ×{sizing_reduction:.2f}")

        # ═══ 15. REBALANCE — Organ sum constraints ═══
        for organ, target in ORGAN_CONSTRAINTS.items():
            self._rebalance_organ(organ, regime, target)

        # ═══ 16. PERSIST + EVOLUTION LOG ═══
        self._maybe_persist()
        self._write_audit(pair, pnl_pct, self.hormones.as_dict(), fear_response.get("tier", "normal"),
                          overrides, self_state["phase"])
        # NeuroEvolution: log fitness for weekly tournament
        if self._trade_count % 50 == 0:
            self.evolution.snapshot_current(self._neurons, self._cumulative_pnl)

        org_health = self.interoception.get_organism_health()
        logger.info(f"[NeuralOrganism] 16-step update: {pair} pnl={pnl_pct:+.2f}% "
                    f"phase={self_state['phase']} fear={fear_response.get('tier')} "
                    f"cortisol={self.hormones.cortisol:.2f} pred_error={pred_error:.2f} "
                    f"crowd={crowd['direction']}({crowd['intensity']:.1f}) "
                    f"hour={trade_hour} health={org_health:.2f} "
                    f"neurons={len(changed_neurons)} overrides={len(overrides)} "
                    f"ban={ban_minutes}min threat={'YES' if threat_check.get('known_threat') else 'no'}")

        return {"ban_minutes": ban_minutes, "overrides": overrides,
                "fear_tier": fear_response.get("tier"), "phase": self_state["phase"],
                "prediction_error": pred_error, "organism_health": org_health,
                "crowd": crowd, "threat": threat_check}

    def _rebalance_organ(self, organ: str, regime: str, target: float):
        """Normalize organ weights to sum to target."""
        organ_neurons = [(k, n) for (k, n) in self._neurons.items()
                         if n.organ == organ and k[1] == regime]
        if not organ_neurons:
            return
        total = sum(n.current_val for _, n in organ_neurons)
        if total > 0:
            scale = target / total
            for _, n in organ_neurons:
                n.current_val = max(n.min_bound, min(n.max_bound, n.current_val * scale))

    # ─── DECAY — Hourly metabolic decay ───

    def decay_all(self, regime: str = "_global"):
        """Decay alpha/beta toward 1.0. Called hourly by scheduler."""
        factor = DECAY_BY_REGIME.get(regime, 0.996)
        count = 0
        for (pid, r), neuron in self._neurons.items():
            if r == regime or regime == "_global":
                neuron.alpha = max(1.0, neuron.alpha * factor)
                neuron.beta_param = max(1.0, neuron.beta_param * factor)
                count += 1
        self._last_decay = time.time()
        logger.debug(f"[NeuralOrganism:Decay] {count} neurons decayed (factor={factor})")

    # ─── PERSISTENCE ───

    def _maybe_persist(self):
        """Persist every 10 trades or 5 minutes."""
        elapsed = time.time() - self._last_persist
        if self._trade_count % 10 == 0 or elapsed > 300:
            self._persist_batch(list(self._neurons.values()))
            self._persist_hormones()
            self._persist_amygdala()
            self._persist_cerebellum()
            self._last_persist = time.time()

    def _persist_batch(self, neurons: List[ParamNeuron]):
        """Batch upsert neurons to SQLite (includes BCM + STDP fields)."""
        try:
            with self._get_conn() as conn:
                for n in neurons:
                    conn.execute(
                        "INSERT OR REPLACE INTO neuron_state "
                        "(param_id, organ, regime, current_val, default_val, min_bound, max_bound, "
                        "alpha, beta_param, prior_strength, frozen, update_count, "
                        "activity_ema, theta_m, last_updated) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (n.param_id, n.organ, n.regime, n.current_val, n.default_val,
                         n.min_bound, n.max_bound, n.alpha, n.beta_param,
                         n.prior_strength, int(n.frozen), n.update_count,
                         n.activity_ema, n.theta_m,
                         n.last_update_time.isoformat() if n.last_update_time else None))
                conn.commit()
        except Exception as e:
            logger.error(f"[NeuralOrganism:Persist] Failed: {e}")

    def _persist_cerebellum(self):
        """Persist cerebellum 24-slot model to SQLite."""
        try:
            with self._get_conn() as conn:
                for hour in range(24):
                    s = self.cerebellum.slots[hour]
                    conn.execute(
                        "INSERT OR REPLACE INTO cerebellum_hours (hour, wins, losses, avg_pnl, updated_at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (hour, s["wins"], s["losses"], s["avg_pnl"],
                         datetime.now(tz=timezone.utc).isoformat()))
                conn.commit()
        except Exception as e:
            logger.debug(f"[NeuralOrganism:Cerebellum] Persist failed: {e}")

    def _load_cerebellum(self):
        """Load cerebellum from SQLite on startup."""
        try:
            with self._get_conn() as conn:
                rows = conn.execute("SELECT * FROM cerebellum_hours").fetchall()
                for r in rows:
                    h = r["hour"]
                    if 0 <= h < 24:
                        self.cerebellum.slots[h] = {
                            "wins": r["wins"], "losses": r["losses"],
                            "avg_pnl": r["avg_pnl"], "n": r["wins"] + r["losses"] - 2
                        }
                if rows:
                    logger.info(f"[NeuralOrganism] Loaded cerebellum {len(rows)} hours from SQLite")
        except Exception:
            pass

    def _persist_hormones(self):
        try:
            with self._get_conn() as conn:
                h = self.hormones
                conn.execute(
                    "UPDATE hormone_state SET cortisol=?, dopamine=?, serotonin=?, adrenaline=?, "
                    "market_stress=?, portfolio_health=?, info_quality=?, updated_at=? WHERE id=1",
                    (h.cortisol, h.dopamine, h.serotonin, h.adrenaline,
                     h._stress, h._health, h._info_q,
                     datetime.now(tz=timezone.utc).isoformat()))
                conn.commit()
        except Exception as e:
            logger.debug(f"[NeuralOrganism:Hormones] Persist failed: {e}")

    def _persist_amygdala(self):
        try:
            with self._get_conn() as conn:
                a = self.amygdala
                conn.execute(
                    "UPDATE amygdala_state SET fear_level=?, peak_fear=?, peak_time=?, "
                    "tier=?, updated_at=? WHERE id=1",
                    (a.fear_level, a.peak_fear,
                     a.peak_time.isoformat() if a.peak_time else None,
                     a.tier, datetime.now(tz=timezone.utc).isoformat()))
                conn.commit()
        except Exception as e:
            logger.debug(f"[NeuralOrganism:Amygdala] Persist failed: {e}")

    def _write_audit(self, pair, pnl, hormones_dict, fear_tier, overrides, phase):
        try:
            with self._get_conn() as conn:
                conn.execute(
                    "INSERT INTO organism_audit (trade_pair, trade_pnl, hormones, fear_tier, "
                    "overrides, phase, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (pair, pnl, json.dumps(hormones_dict), fear_tier,
                     json.dumps(overrides), phase,
                     datetime.now(tz=timezone.utc).isoformat()))
                conn.commit()
        except Exception as e:
            logger.debug(f"[NeuralOrganism:Audit] Write failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# SINGLETON + HELPER
# ═══════════════════════════════════════════════════════════════════════════

_organism_instance: Optional[NeuralOrganism] = None
_organism_lock = threading.Lock()


def get_organism() -> NeuralOrganism:
    """Thread-safe singleton. All consumer files use this."""
    global _organism_instance
    if _organism_instance is None:
        with _organism_lock:
            if _organism_instance is None:
                _organism_instance = NeuralOrganism()
    return _organism_instance


def _p(param_id: str, fallback: float = 0.5, regime: str = "_global") -> float:
    """
    Helper for consumer files. Graceful degradation: if organism unavailable, returns fallback.

    Usage:
        from neural_organism import _p
        if fng < _p("evidence.q3.fng_extreme_low", 10):
            score += _p("evidence.q3.fng_adj_extreme", 0.22)
    """
    try:
        val = get_organism().get_param(param_id, regime)
        return val if val is not None else fallback
    except Exception:
        return fallback


# ═══════════════════════════════════════════════════════════════════════════
# SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    print("=" * 70)
    print("Neural Organism Smoke Test")
    print("=" * 70)

    org = get_organism()
    print(f"\nNeurons: {len(org._neurons)}")
    print(f"Params in registry: {len(PARAM_REGISTRY)}")
    print(f"Regimes: {len(REGIMES)}")

    # Test get_param
    for pid in ["evidence.weights.q1_trend", "evidence.q3.fng_extreme_low",
                "rag.evidence_first_threshold", "sizing.max_risk", "immune.base_ban_minutes"]:
        val = org.get_param(pid)
        spec = PARAM_REGISTRY[pid]
        print(f"  {pid}: {val:.4f} (default={spec['default']}, range=[{spec['min']}, {spec['max']}])")

    # Test get_organ
    weights = org.get_organ("evidence_weights")
    print(f"\nEvidence weights organ: {weights}")
    print(f"  Sum: {sum(weights.values()):.4f} (should be ~1.0)")

    # Test hormone computation
    h = org.hormones.compute(fng=9, consec_losses=3, balance_vs_peak=0.8)
    print(f"\nHormones (F&G=9, 3 losses): {h}")

    # Test update cycle
    result = org.update_cycle(
        pair="ETH/USDT:USDT", pnl_pct=-5.2, regime="trending_bear",
        confidence=0.63, duration_hours=4.0, stake_amount=100,
        fng=9, adx=18, funding_rate=0.0008, balance_vs_peak=0.9)
    print(f"\nUpdate cycle result: {result}")

    # Test pair ban
    ban = org.immune.compute_ban("ETH/USDT:USDT", -5.2, org.get_param)
    print(f"Pair ban: {ban} minutes")

    # Test amygdala fear decay
    fear = org.amygdala.get_current_fear()
    print(f"Current fear: {fear:.3f} (tier={org.amygdala.tier})")

    # Verify no dead code: all PARAM_REGISTRY entries are accessible
    missing = []
    for pid in PARAM_REGISTRY:
        val = org.get_param(pid)
        if val is None:
            missing.append(pid)
    print(f"\nParam accessibility: {len(PARAM_REGISTRY) - len(missing)}/{len(PARAM_REGISTRY)} OK")
    if missing:
        print(f"  MISSING: {missing}")

    print("\n" + "=" * 70)
    print("Smoke test PASSED" if not missing else "Smoke test FAILED")
    print("=" * 70)
