import math
import os
import sys
import sqlite3
import logging
from datetime import datetime, timezone

sys.path.append(os.path.dirname(__file__))

from autonomy_manager import AutonomyManager

logger = logging.getLogger(__name__)

from ai_config import AI_DB_PATH as DB_PATH

# Phase 24: Neural Organism — adaptive parameters
try:
    from neural_organism import _p
except ImportError:
    def _p(param_id, fallback=0.5, regime="_global"):
        return fallback


class BayesianKelly:
    """
    Self-learning Kelly fraction using Beta distribution.
    Tracks win/loss history and computes optimal bet sizing.
    
    f* = (b*p - q) / b
    where p = win_probability (from Beta posterior), q = 1-p, b = avg win/loss ratio
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.alpha = 1.0  # Prior success count (Beta prior)
        self.beta_param = 1.0   # Prior failure count (Beta prior)
        self.avg_win_loss_ratio = 1.5  # Default W/L ratio
        self._ensure_table()
        self._load_from_db()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _ensure_table(self):
        with self._get_conn() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS bayesian_kelly (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    alpha REAL DEFAULT 1.0,
                    beta_param REAL DEFAULT 1.0,
                    avg_win REAL DEFAULT 0.0,
                    avg_loss REAL DEFAULT 0.0,
                    total_trades INTEGER DEFAULT 0,
                    updated_at TEXT
                )
            ''')
            row = conn.execute("SELECT COUNT(*) FROM bayesian_kelly").fetchone()
            if row[0] == 0:
                conn.execute(
                    "INSERT INTO bayesian_kelly (id, alpha, beta_param, updated_at) VALUES (1, 1.0, 1.0, ?)",
                    (datetime.now(tz=timezone.utc).isoformat(),)
                )
            conn.commit()

    def _load_from_db(self):
        with self._get_conn() as conn:
            row = conn.execute("SELECT alpha, beta_param, avg_win, avg_loss, total_trades FROM bayesian_kelly WHERE id = 1").fetchone()
            if row:
                self.alpha = float(row['alpha'])
                self.beta_param = float(row['beta_param'])
                avg_win = float(row['avg_win']) if row['avg_win'] else 0.0
                avg_loss = float(row['avg_loss']) if row['avg_loss'] else 0.0
                if avg_loss > 0:
                    self.avg_win_loss_ratio = avg_win / avg_loss

    def update(self, won: bool, pnl_pct: float = 0.0):
        """Update Beta distribution after a trade completes."""
        if won:
            self.alpha += 1
        else:
            self.beta_param += 1

        with self._get_conn() as conn:
            # Update running averages
            if won and pnl_pct > 0:
                conn.execute(
                    "UPDATE bayesian_kelly SET alpha = ?, avg_win = (avg_win * total_trades + ?) / (total_trades + 1), total_trades = total_trades + 1, updated_at = ? WHERE id = 1",
                    (self.alpha, abs(pnl_pct), datetime.now(tz=timezone.utc).isoformat())
                )
            elif not won and pnl_pct < 0:
                conn.execute(
                    "UPDATE bayesian_kelly SET beta_param = ?, avg_loss = (avg_loss * total_trades + ?) / (total_trades + 1), total_trades = total_trades + 1, updated_at = ? WHERE id = 1",
                    (self.beta_param, abs(pnl_pct), datetime.now(tz=timezone.utc).isoformat())
                )
            else:
                conn.execute(
                    "UPDATE bayesian_kelly SET alpha = ?, beta_param = ?, total_trades = total_trades + 1, updated_at = ? WHERE id = 1",
                    (self.alpha, self.beta_param, datetime.now(tz=timezone.utc).isoformat())
                )
            conn.commit()

    def win_probability(self) -> float:
        """Beta distribution mean: p = alpha / (alpha + beta)"""
        return self.alpha / (self.alpha + self.beta_param)

    def kelly_fraction(self) -> float:
        """
        Optimal Kelly fraction: f* = (b*p - q) / b
        Capped at 25% for safety.
        """
        p = self.win_probability()
        q = 1.0 - p
        b = max(self.avg_win_loss_ratio, 0.01)  # Prevent division by zero
        f = (b * p - q) / b
        return max(0.0, min(f, _p("sizing.kelly_cap", 0.25)))  # Phase 24: adaptive Kelly cap

class PositionSizer:
    """
    Phase 3.5.2: Position Sizing Engine
    Translates AI confidence metrics into actionable Freqtrade stake amounts.
    Uses a fractional Kelly-inspired Beta distribution approach to heavily 
    penalize low confidence and reward high conviction.
    """
    def __init__(self,
                 max_portfolio_risk_per_trade: float = 0.05,
                 confidence_exponent: float = 1.5):
        """
        Args:
            max_portfolio_risk_per_trade: Maximum fraction of the portfolio allowed on a single maximum-conviction trade (e.g. 5%)
            confidence_exponent: The power to raise confidence to. (confidence^1.5) reduces the size of 0.5 confidence trades compared to 0.9.
        """
        # Phase 24: Read from Neural Organism (adaptive), fallback to constructor args
        self.max_risk = _p("sizing.max_risk", max_portfolio_risk_per_trade)
        self.exponent = _p("sizing.confidence_exponent", confidence_exponent)
        
        # Phase 3.5.4: Autonomy level controls Kelly fraction cap
        self.autonomy = AutonomyManager()
        
        # Phase 3.5.2: Bayesian Kelly — self-learning from trade outcomes
        self.bayesian_kelly = BayesianKelly()
        
    def _effective_max_risk(self) -> float:
        """Max risk adjusted by autonomy level AND Bayesian Kelly.
        Trade-First: confidence modulates SIZE, never PERMISSION.
        Every level trades — kelly_autonomy is always > 0.
        """
        kelly_autonomy = self.autonomy.get_kelly_fraction()
        kelly_bayesian = self.bayesian_kelly.kelly_fraction()
        # Use the lesser of autonomy cap and Bayesian optimal fraction
        # But never below 0.5% — always trade, even if tiny
        effective = min(self.max_risk, kelly_autonomy, max(kelly_bayesian, 0.005))
        return effective
        
    def calculate_stake_fraction(self, confidence: float, current_regime_modifier: float = 1.0) -> float:
        """
        Calculates the fraction of total available capital to stake on this trade.
        
        Formula: max_risk * (confidence ^ exponent) * regime_modifier
        
        Args:
            confidence: LLM's confidence score (0.0 to 1.0)
            current_regime_modifier: Multiplier based on market regime (e.g. 1.2 for bull, 0.5 for bear)
            
        Returns:
            float: A percentage multiplier for Freqtrade's custom_stake_amount (e.g. 0.02 is 2% of capital)
        """
        # 2. Apply the exponential trust curve (e.g. 0.5^2 = 0.25 penalty factor)
        trust_curve_multiplier = math.pow(max(confidence, 0.01), self.exponent)

        # 3. Calculate raw fraction — capped by autonomy level's Kelly
        effective_risk = self._effective_max_risk()
        fraction = effective_risk * trust_curve_multiplier * current_regime_modifier

        # 4. Trade-First floor (Phase 24: adaptive)
        min_fraction = effective_risk * _p("sizing.min_fraction_mult", 0.01)
        fraction = max(fraction, min_fraction)

        # 5. Cap at Absolute Maximum Risk (Phase 24: adaptive)
        final_fraction = min(fraction, effective_risk * _p("sizing.max_fraction_mult", 1.5))

        return round(final_fraction, 4)

    def print_sizing_table(self):
        """Utility to demonstrate the confidence scaling curve."""
        logger.info(f"Position Sizing Curve (Max Risk: {self.max_risk*100}%, Exponent: {self.exponent})")
        logger.info("Confidence | Stake %")
        logger.info("--------------------")
        for i in range(1, 11):
            conf = i / 10.0
            stake = self.calculate_stake_fraction(conf)
            logger.info(f"   {conf:.1f}     |  {stake*100:.2f}%")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sizer = PositionSizer()
    sizer.print_sizing_table()
