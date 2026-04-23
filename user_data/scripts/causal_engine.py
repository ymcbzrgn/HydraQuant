"""
causal_engine.py — Phase 26 Sprint 2, Task 6A

Temporal Causal Graph Discovery via Tigramite PCMCI+.

Discovers causal relationships from time series data:
  - "funding_rate → pnl" (with time lag 2h)
  - "fear_greed → confidence → sizing_multiplier"
  - "cortisol → defensive_sizing" (hormone → behavior)

Pearl Causal Hierarchy Level 1 (Observation):
  PCMCI+ discovers temporal causal structure from data.
  Output goes to Grafeo graph DB + SQLite causal_discoveries.

Data Sources:
  - ai_decisions (confidence, outcome_pnl, regime)
  - evidence_audit_log (sub_scores: trend, momentum, crowd, risk, macro)
  - derivatives_data (funding_rate, open_interest, long_short_ratio)
  - hormone_state (cortisol, dopamine, serotonin, adrenaline)
  - hippocampus_episodes (pattern outcomes per regime)
  - fear_and_greed (fng_value)

Usage:
    python causal_engine.py --discover               # Run full causal discovery
    python causal_engine.py --discover --regime bull  # Regime-specific
    python causal_engine.py --status                  # Show discovered edges
    python causal_engine.py --validate                # Validate existing synapse seeds

Output:
    - Grafeo causal edges (add_causal_edge)
    - SQLite causal_discoveries table
    - JSON report at user_data/models/causal_graph_latest.json
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger("causal_engine")

from ai_config import AI_DB_PATH
from db import get_db_connection, get_connection, execute_with_retry, init_db

# ===========================================================================
# Constants
# ===========================================================================

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
CAUSAL_REPORT_PATH = os.path.join(MODEL_DIR, "causal_graph_latest.json")

# Minimum observations needed for reliable causal discovery
MIN_OBSERVATIONS = 30  # Tur-2 (M3): PCMCI+ literature minimum (Runge 2020); 20 was below that bound

# PCMCI+ significance threshold
ALPHA_LEVEL = 0.05

# Maximum time lag to test (in periods — 1 period = 1h for hourly data)
MAX_LAG = 12  # 12 hours

# Variables used in causal analysis (column names in the assembled DataFrame)
CAUSAL_VARIABLES = [
    "confidence",
    "outcome_pnl",
    "funding_rate",
    "long_short_ratio",
    "open_interest_norm",
    "fng_value",
    "sub_trend",
    "sub_momentum",
    "sub_crowd",
    "sub_risk",
    "sub_macro",
    "cortisol",
    "dopamine",
    "serotonin",
    "adrenaline",
    "market_stress",
]

# Seed synapse names from neural_organism — we validate these
SEED_SYNAPSES = {
    ("funding_rate", "outcome_pnl"): "Known: funding rate affects trade outcomes",
    ("fng_value", "confidence"): "Known: fear/greed influences AI confidence",
    ("cortisol", "outcome_pnl"): "Hypothesis: stress hormone affects performance",
    ("market_stress", "cortisol"): "Known: market stress drives cortisol",
    ("sub_momentum", "confidence"): "Known: momentum score feeds confidence",
    ("sub_trend", "outcome_pnl"): "Hypothesis: trend alignment predicts outcomes",
    ("long_short_ratio", "funding_rate"): "Known: crowded positioning affects funding",
    ("dopamine", "confidence"): "Hypothesis: recent wins boost confidence",
    ("sub_risk", "outcome_pnl"): "Hypothesis: risk assessment predicts outcomes",
    ("sub_crowd", "confidence"): "Known: crowd consensus feeds confidence",
    ("fng_value", "sub_crowd"): "Known: fear/greed correlated with crowd score",
    ("open_interest_norm", "outcome_pnl"): "Hypothesis: OI changes predict outcomes",
}


class CausalEngine:
    """Discovers temporal causal relationships using Tigramite PCMCI+."""

    def __init__(self, db_path: str = AI_DB_PATH):
        self.db_path = db_path
        self._graph_store = None
        init_db()

    def _get_graph_store(self):
        """Lazy-load graph store (only connects when needed)."""
        if self._graph_store is None:
            try:
                from graph_store import get_graph_store
                self._graph_store = get_graph_store()
            except Exception as e:
                logger.warning(f"[CausalEngine] Grafeo not available: {e}")
        return self._graph_store

    # ===================================================================
    # Data Assembly
    # ===================================================================

    def _assemble_time_series(self, regime: str = None,
                              lookback_days: int = 30) -> Optional[pd.DataFrame]:
        """Assemble aligned time series from multiple data sources.

        Returns a DataFrame with one row per hour, columns = CAUSAL_VARIABLES.
        Missing values are forward-filled then zero-filled.
        """
        conn = get_db_connection(self.db_path)
        try:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()

            # --- ai_decisions: confidence + outcome_pnl ---
            if regime:
                decisions = conn.execute("""
                    SELECT d.timestamp, d.confidence, d.outcome_pnl
                    FROM ai_decisions d
                    WHERE d.timestamp >= ? AND d.regime = ?
                    ORDER BY d.timestamp ASC
                """, (cutoff, regime)).fetchall()
            else:
                decisions = conn.execute("""
                    SELECT d.timestamp, d.confidence, d.outcome_pnl
                    FROM ai_decisions d
                    WHERE d.timestamp >= ?
                    ORDER BY d.timestamp ASC
                """, (cutoff,)).fetchall()

            if len(decisions) < MIN_OBSERVATIONS:
                logger.warning(f"[CausalEngine] Insufficient ai_decisions: "
                             f"{len(decisions)} < {MIN_OBSERVATIONS}")
                return None

            # --- evidence_audit_log: sub_scores ---
            evidence = conn.execute("""
                SELECT e.timestamp, e.sub_scores_json, e.confidence
                FROM evidence_audit_log e
                WHERE e.timestamp >= ?
                ORDER BY e.timestamp ASC
            """, (cutoff,)).fetchall()

            # --- derivatives_data: funding_rate, OI, L/S ratio ---
            derivs = conn.execute("""
                SELECT d.timestamp, d.funding_rate, d.long_short_ratio,
                       d.open_interest_usd
                FROM derivatives_data d
                WHERE d.timestamp >= ?
                ORDER BY d.timestamp ASC
            """, (cutoff,)).fetchall()

            # --- fear_and_greed ---
            fng = conn.execute("""
                SELECT timestamp, value as fng_value
                FROM fear_and_greed
                WHERE timestamp >= ?
                ORDER BY timestamp ASC
            """, (cutoff,)).fetchall()

            # --- hormone_state (single row, use hormone_history if available) ---
            hormones = []
            try:
                hormones = conn.execute("""
                    SELECT timestamp, cortisol, dopamine, serotonin,
                           adrenaline, market_stress
                    FROM hormone_history
                    WHERE timestamp >= ?
                    ORDER BY timestamp ASC
                """, (cutoff,)).fetchall()
            except Exception:
                # hormone_history might not exist on server yet
                pass

            if not hormones:
                # Fallback: use current hormone_state as constant
                try:
                    h = conn.execute("""
                        SELECT cortisol, dopamine, serotonin, adrenaline, market_stress
                        FROM hormone_state LIMIT 1
                    """).fetchone()
                    if h:
                        hormones = [{"timestamp": cutoff,
                                    "cortisol": h["cortisol"], "dopamine": h["dopamine"],
                                    "serotonin": h["serotonin"], "adrenaline": h["adrenaline"],
                                    "market_stress": h["market_stress"]}]
                except Exception:
                    pass

        finally:
            conn.close()

        # --- Build DataFrames ---
        records = []

        # Decisions
        for row in decisions:
            records.append({
                "timestamp": row["timestamp"],
                "confidence": row["confidence"] or 0.5,
                "outcome_pnl": row["outcome_pnl"] or 0.0,
            })

        df_decisions = pd.DataFrame(records)
        if df_decisions.empty:
            return None

        df_decisions["timestamp"] = pd.to_datetime(df_decisions["timestamp"], utc=True)
        df_decisions = df_decisions.set_index("timestamp").sort_index()

        # Evidence sub-scores
        ev_records = []
        for row in evidence:
            sub = {}
            if row["sub_scores_json"]:
                try:
                    sub = json.loads(row["sub_scores_json"])
                except Exception:
                    pass
            ev_records.append({
                "timestamp": row["timestamp"],
                "sub_trend": sub.get("trend", sub.get("technical", 0.5)),
                "sub_momentum": sub.get("momentum", sub.get("mom", 0.5)),
                "sub_crowd": sub.get("crowd", sub.get("sentiment", 0.5)),
                "sub_risk": sub.get("risk", 0.5),
                "sub_macro": sub.get("macro", 0.5),
            })

        if ev_records:
            df_evidence = pd.DataFrame(ev_records)
            df_evidence["timestamp"] = pd.to_datetime(df_evidence["timestamp"], utc=True)
            df_evidence = df_evidence.set_index("timestamp").sort_index()
        else:
            df_evidence = pd.DataFrame()

        # Derivatives
        deriv_records = []
        for row in derivs:
            deriv_records.append({
                "timestamp": row["timestamp"],
                "funding_rate": row["funding_rate"] or 0.0,
                "long_short_ratio": row["long_short_ratio"] or 1.0,
                "open_interest_norm": 0.0,  # Will normalize below
            })

        if deriv_records:
            df_derivs = pd.DataFrame(deriv_records)
            df_derivs["timestamp"] = pd.to_datetime(df_derivs["timestamp"], utc=True)
            df_derivs = df_derivs.set_index("timestamp").sort_index()
            # Normalize OI as percentage change
            if len(df_derivs) > 1:
                oi_values = [r["open_interest_usd"] or 0 for r in derivs]
                oi_series = pd.Series(oi_values, index=df_derivs.index)
                df_derivs["open_interest_norm"] = oi_series.pct_change().fillna(0).clip(-1, 1)
        else:
            df_derivs = pd.DataFrame()

        # Fear & Greed
        fng_records = []
        for row in fng:
            fng_records.append({
                "timestamp": row["timestamp"],
                "fng_value": (row["fng_value"] or 50) / 100.0,  # Normalize to 0-1
            })

        if fng_records:
            df_fng = pd.DataFrame(fng_records)
            df_fng["timestamp"] = pd.to_datetime(df_fng["timestamp"], utc=True)
            df_fng = df_fng.set_index("timestamp").sort_index()
        else:
            df_fng = pd.DataFrame()

        # Hormones
        h_records = []
        for row in hormones:
            ts = row["timestamp"] if isinstance(row, dict) else row[0]
            if isinstance(row, dict):
                h_records.append({
                    "timestamp": ts,
                    "cortisol": row.get("cortisol", 0.5),
                    "dopamine": row.get("dopamine", 0.5),
                    "serotonin": row.get("serotonin", 0.5),
                    "adrenaline": row.get("adrenaline", 0.5),
                    "market_stress": row.get("market_stress", 0.5),
                })
            else:
                h_records.append({
                    "timestamp": ts,
                    "cortisol": row["cortisol"] or 0.5,
                    "dopamine": row["dopamine"] or 0.5,
                    "serotonin": row["serotonin"] or 0.5,
                    "adrenaline": row["adrenaline"] or 0.5,
                    "market_stress": row["market_stress"] or 0.5,
                })

        if h_records:
            df_hormones = pd.DataFrame(h_records)
            df_hormones["timestamp"] = pd.to_datetime(df_hormones["timestamp"], utc=True)
            df_hormones = df_hormones.set_index("timestamp").sort_index()
        else:
            df_hormones = pd.DataFrame()

        # --- Merge all on hourly index ---
        # Resample decisions to hourly (take last value per hour)
        df_merged = df_decisions.resample("1h").last()

        for df_src in [df_evidence, df_derivs, df_fng, df_hormones]:
            if not df_src.empty:
                df_resampled = df_src.resample("1h").last()
                df_merged = df_merged.join(df_resampled, how="left")

        # Forward fill, then fill remaining with defaults
        df_merged = df_merged.ffill()

        # Ensure all CAUSAL_VARIABLES exist
        for var in CAUSAL_VARIABLES:
            if var not in df_merged.columns:
                df_merged[var] = 0.5 if var not in ("outcome_pnl", "funding_rate",
                                                     "open_interest_norm") else 0.0

        # Drop rows where outcome_pnl is NaN (no trade data)
        df_merged = df_merged.dropna(subset=["outcome_pnl"])

        # Fill any remaining NaN
        df_merged = df_merged.fillna(0.0)

        # Keep only causal variables
        df_merged = df_merged[[v for v in CAUSAL_VARIABLES if v in df_merged.columns]]

        logger.info(f"[CausalEngine] Assembled {len(df_merged)} observations, "
                    f"{len(df_merged.columns)} variables"
                    f"{f' (regime={regime})' if regime else ''}")

        if len(df_merged) < MIN_OBSERVATIONS:
            logger.warning(f"[CausalEngine] Insufficient merged data: "
                         f"{len(df_merged)} < {MIN_OBSERVATIONS}")
            return None

        return df_merged

    # ===================================================================
    # PCMCI+ Causal Discovery
    # ===================================================================

    def discover(self, regime: str = None, lookback_days: int = 30,
                 alpha: float = ALPHA_LEVEL, max_lag: int = MAX_LAG) -> Dict[str, Any]:
        """Run PCMCI+ causal discovery.

        Returns dict with discovered edges and validation results.
        """
        try:
            from tigramite import data_processing as pp
            from tigramite.pcmci import PCMCI
            from tigramite.independence_tests.parcorr import ParCorr
        except ImportError:
            logger.error("[CausalEngine] tigramite not installed. "
                        "Run: pip install tigramite")
            return {"error": "tigramite not installed", "edges": []}

        # Assemble data
        df = self._assemble_time_series(regime=regime, lookback_days=lookback_days)
        if df is None:
            logger.warning(
                f"[CausalEngine:{regime}] _assemble_time_series returned None — "
                f"insufficient trade data or regime labels. Check the "
                f"ai_decisions.regime distribution; if it's all one label, "
                f"regime-specific series cannot be built and PCMCI+ is skipped."
            )
            return {"error": "insufficient data", "edges": [], "regime": regime}

        var_names = list(df.columns)
        data_array = df.values

        logger.info(f"[CausalEngine] Running PCMCI+ on {len(var_names)} variables, "
                    f"{len(data_array)} observations, max_lag={max_lag}")

        # Standardize data
        data_array = (data_array - data_array.mean(axis=0)) / (data_array.std(axis=0) + 1e-10)

        # Build Tigramite dataframe
        dataframe = pp.DataFrame(
            data=data_array,
            var_names=var_names,
            datatime=np.arange(len(data_array)),
        )

        # Partial correlation test (linear, robust, fast)
        parcorr = ParCorr(significance='analytic')

        # Run PCMCI+
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=0)

        results = pcmci.run_pcmciplus(
            tau_min=0,
            tau_max=max_lag,
            pc_alpha=alpha,
        )

        # Extract significant edges
        graph = results['graph']  # (N, N, tau_max+1) array of edge types
        val_matrix = results['val_matrix']  # (N, N, tau_max+1) test statistic values
        p_matrix = results['p_matrix']  # (N, N, tau_max+1) p-values

        edges = []
        n_vars = len(var_names)

        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    continue
                for tau in range(max_lag + 1):
                    edge_type = graph[i, j, tau]
                    if edge_type != '' and edge_type != '':
                        # Significant causal link found
                        p_val = float(p_matrix[i, j, tau])
                        strength = float(abs(val_matrix[i, j, tau]))

                        # Determine direction
                        if edge_type == '-->':
                            source = var_names[i]
                            target = var_names[j]
                        elif edge_type == '<--':
                            source = var_names[j]
                            target = var_names[i]
                        elif edge_type == 'o-o' or edge_type == 'x-x':
                            # Undirected — skip
                            continue
                        else:
                            # Directed edge i → j at lag tau
                            source = var_names[i]
                            target = var_names[j]

                        edge = {
                            "source": source,
                            "target": target,
                            "strength": round(strength, 4),
                            "time_lag": tau,
                            "p_value": round(p_val, 6),
                            "edge_type": str(edge_type),
                            "regime": regime or "_global",
                            "method": "PCMCI+",
                            "n_observations": len(data_array),
                        }

                        # Avoid duplicates
                        edge_key = (source, target, tau)
                        if not any(e["source"] == source and e["target"] == target
                                   and e["time_lag"] == tau for e in edges):
                            edges.append(edge)

        logger.info(f"[CausalEngine] Discovered {len(edges)} significant causal edges")

        # Validate against seed synapses
        validation = self._validate_seeds(edges)

        # Build result
        result = {
            "n_observations": len(data_array),
            "n_variables": n_vars,
            "variable_names": var_names,
            "max_lag": max_lag,
            "alpha": alpha,
            "regime": regime or "_global",
            "n_edges": len(edges),
            "edges": edges,
            "seed_validation": validation,
            "discovered_at": datetime.utcnow().isoformat(),
        }

        return result

    def _validate_seeds(self, discovered_edges: List[Dict]) -> Dict[str, Any]:
        """Validate seed synapses against discovered causal edges."""
        validation = {"confirmed": [], "refuted": [], "not_tested": []}

        discovered_set = {(e["source"], e["target"]) for e in discovered_edges}

        for (src, tgt), description in SEED_SYNAPSES.items():
            entry = {
                "source": src, "target": tgt,
                "description": description,
            }

            if (src, tgt) in discovered_set:
                # Find the edge
                edge = next(e for e in discovered_edges
                           if e["source"] == src and e["target"] == tgt)
                entry["status"] = "CONFIRMED"
                entry["strength"] = edge["strength"]
                entry["time_lag"] = edge["time_lag"]
                entry["p_value"] = edge["p_value"]
                validation["confirmed"].append(entry)
            else:
                # Check if the variables were even in the data
                var_names_in_data = set()
                for e in discovered_edges:
                    var_names_in_data.add(e["source"])
                    var_names_in_data.add(e["target"])

                if src in var_names_in_data and tgt in var_names_in_data:
                    entry["status"] = "REFUTED"
                    entry["reason"] = "Variables present but no significant causal link found"
                    validation["refuted"].append(entry)
                else:
                    entry["status"] = "NOT_TESTED"
                    entry["reason"] = f"Missing variables: {src if src not in var_names_in_data else tgt}"
                    validation["not_tested"].append(entry)

        logger.info(f"[CausalEngine] Seed validation: "
                    f"{len(validation['confirmed'])} confirmed, "
                    f"{len(validation['refuted'])} refuted, "
                    f"{len(validation['not_tested'])} not tested")

        return validation

    # ===================================================================
    # Persistence (SQLite + Grafeo)
    # ===================================================================

    def persist_discoveries(self, result: Dict[str, Any]) -> int:
        """Write discovered edges to SQLite causal_discoveries + Grafeo graph.

        Returns count of edges persisted.
        """
        if "error" in result or not result.get("edges"):
            return 0

        edges = result["edges"]
        regime = result.get("regime", "_global")
        n_obs = result.get("n_observations", 0)
        persisted = 0

        # 1. Deactivate old discoveries for this regime
        try:
            execute_with_retry(
                "UPDATE causal_discoveries SET is_active = 0 WHERE regime = ?",
                (regime,), max_retries=5
            )
        except Exception as e:
            logger.warning(f"[CausalEngine] Failed to deactivate old discoveries: {e}")

        # 2. Insert new discoveries
        for edge in edges:
            try:
                execute_with_retry(
                    """INSERT OR REPLACE INTO causal_discoveries
                       (source_var, target_var, causal_strength, time_lag,
                        p_value, method, n_observations, regime, is_active)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)""",
                    (edge["source"], edge["target"], edge["strength"],
                     edge["time_lag"], edge["p_value"], edge["method"],
                     n_obs, regime),
                    max_retries=5
                )
                persisted += 1
            except Exception as e:
                logger.warning(f"[CausalEngine] Failed to persist edge "
                             f"{edge['source']}→{edge['target']}: {e}")

        # 3. Write to Grafeo graph
        gs = self._get_graph_store()
        if gs:
            grafeo_count = 0
            for edge in edges:
                try:
                    gs.add_causal_edge(
                        source_var=edge["source"],
                        target_var=edge["target"],
                        strength=edge["strength"],
                        time_lag=edge["time_lag"],
                        p_value=edge["p_value"],
                        regime=regime,
                    )
                    grafeo_count += 1
                except Exception as e:
                    logger.debug(f"[CausalEngine] Grafeo edge failed: {e}")

            logger.info(f"[CausalEngine] Persisted {grafeo_count} edges to Grafeo")

        # 4. Save JSON report
        try:
            os.makedirs(MODEL_DIR, exist_ok=True)
            with open(CAUSAL_REPORT_PATH, "w") as f:
                json.dump(result, f, indent=2, default=str)
            logger.info(f"[CausalEngine] Report saved: {CAUSAL_REPORT_PATH}")
        except Exception as e:
            logger.warning(f"[CausalEngine] Failed to save report: {e}")

        logger.info(f"[CausalEngine] Persisted {persisted} edges to SQLite")
        return persisted

    # ===================================================================
    # Query API (for other modules)
    # ===================================================================

    def get_active_edges(self, regime: str = None) -> List[Dict]:
        """Get all active causal discoveries for a regime."""
        conn = get_db_connection(self.db_path)
        try:
            if regime:
                rows = conn.execute("""
                    SELECT * FROM causal_discoveries
                    WHERE is_active = 1 AND regime = ?
                    ORDER BY causal_strength DESC
                """, (regime,)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT * FROM causal_discoveries
                    WHERE is_active = 1
                    ORDER BY causal_strength DESC
                """).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_causal_parents(self, target_var: str, regime: str = None) -> List[Dict]:
        """Get variables that causally influence a target variable."""
        conn = get_db_connection(self.db_path)
        try:
            if regime:
                rows = conn.execute("""
                    SELECT source_var, causal_strength, time_lag, p_value
                    FROM causal_discoveries
                    WHERE target_var = ? AND is_active = 1 AND regime = ?
                    ORDER BY causal_strength DESC
                """, (target_var, regime)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT source_var, causal_strength, time_lag, p_value, regime
                    FROM causal_discoveries
                    WHERE target_var = ? AND is_active = 1
                    ORDER BY causal_strength DESC
                """, (target_var,)).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_causal_children(self, source_var: str) -> List[Dict]:
        """Get variables causally influenced by a source variable."""
        conn = get_db_connection(self.db_path)
        try:
            rows = conn.execute("""
                SELECT target_var, causal_strength, time_lag, p_value, regime
                FROM causal_discoveries
                WHERE source_var = ? AND is_active = 1
                ORDER BY causal_strength DESC
            """, (source_var,)).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_causal_path(self, source: str, target: str, max_depth: int = 3) -> List[List[Dict]]:
        """Find causal paths from source to target (BFS)."""
        conn = get_db_connection(self.db_path)
        try:
            all_edges = conn.execute("""
                SELECT source_var, target_var, causal_strength, time_lag
                FROM causal_discoveries WHERE is_active = 1
            """).fetchall()
        finally:
            conn.close()

        # Build adjacency list
        adj = {}
        for e in all_edges:
            src = e["source_var"]
            if src not in adj:
                adj[src] = []
            adj[src].append({
                "target": e["target_var"],
                "strength": e["causal_strength"],
                "lag": e["time_lag"],
            })

        # BFS for paths
        paths = []
        queue = [([source], [])]

        while queue:
            current_path, edge_path = queue.pop(0)
            current_node = current_path[-1]

            if current_node == target and len(current_path) > 1:
                paths.append(edge_path)
                continue

            if len(current_path) > max_depth:
                continue

            for neighbor in adj.get(current_node, []):
                if neighbor["target"] not in current_path:  # Avoid cycles
                    new_path = current_path + [neighbor["target"]]
                    new_edges = edge_path + [{
                        "from": current_node,
                        "to": neighbor["target"],
                        "strength": neighbor["strength"],
                        "lag": neighbor["lag"],
                    }]
                    queue.append((new_path, new_edges))

        return paths

    # ===================================================================
    # Status / Reporting
    # ===================================================================

    def print_status(self):
        """Print human-readable status of causal discoveries."""
        edges = self.get_active_edges()

        print("\n" + "=" * 65)
        print("  CAUSAL ENGINE STATUS")
        print("=" * 65)

        if not edges:
            print("  No causal discoveries yet. Run --discover first.")
            print("=" * 65 + "\n")
            return

        # Group by regime
        by_regime = {}
        for e in edges:
            r = e.get("regime", "_global")
            if r not in by_regime:
                by_regime[r] = []
            by_regime[r].append(e)

        for regime, redges in by_regime.items():
            print(f"\n  Regime: {regime} ({len(redges)} edges)")
            print("  " + "-" * 60)
            print(f"  {'Source':<22} → {'Target':<22} | Str    | Lag | p-val")
            print("  " + "-" * 60)
            for e in redges[:20]:  # Top 20
                print(f"  {e['source_var']:<22} → {e['target_var']:<22} | "
                      f"{e['causal_strength']:5.3f}  | {e['time_lag']:>2}h | "
                      f"{e['p_value']:.4f}")

        # Seed validation from last report
        if os.path.exists(CAUSAL_REPORT_PATH):
            try:
                with open(CAUSAL_REPORT_PATH) as f:
                    report = json.load(f)
                val = report.get("seed_validation", {})
                if val:
                    print(f"\n  Seed Synapse Validation:")
                    print(f"    Confirmed: {len(val.get('confirmed', []))}")
                    print(f"    Refuted:   {len(val.get('refuted', []))}")
                    print(f"    Untested:  {len(val.get('not_tested', []))}")
            except Exception:
                pass

        print("=" * 65 + "\n")


# ===========================================================================
# Full Discovery Pipeline
# ===========================================================================

def run_causal_discovery(regime: str = None, lookback_days: int = 30,
                         persist: bool = True) -> Dict[str, Any]:
    """Run full causal discovery pipeline.

    Called by scheduler weekly and available as CLI.
    """
    engine = CausalEngine()

    # Run discovery
    result = engine.discover(regime=regime, lookback_days=lookback_days)

    if "error" in result:
        logger.error(f"[CausalEngine] Discovery failed: {result['error']}")
        return result

    # Persist to DB + Grafeo
    if persist and result.get("edges"):
        persisted = engine.persist_discoveries(result)
        result["persisted_count"] = persisted

    return result


def run_multi_regime_discovery(lookback_days: int = 30) -> Dict[str, Any]:
    """Run causal discovery per regime + global.

    Discovers regime-specific causal structures:
    - Bull: different causal graph than bear
    - Global: stable relationships across regimes
    """
    results = {}

    # Global first
    logger.info("[CausalEngine] Running global discovery...")
    results["_global"] = run_causal_discovery(regime=None, lookback_days=lookback_days)

    # Per-regime
    for regime in ["trending_bull", "trending_bear", "ranging", "high_volatility"]:
        logger.info(f"[CausalEngine] Running {regime} discovery...")
        result = run_causal_discovery(regime=regime, lookback_days=lookback_days)
        if "error" not in result:
            results[regime] = result
        else:
            logger.info(f"[CausalEngine] {regime}: {result.get('error', 'skipped')}")

    # Summary
    total_edges = sum(r.get("n_edges", 0) for r in results.values() if isinstance(r, dict))
    logger.info(f"[CausalEngine] Multi-regime discovery complete: "
                f"{total_edges} total edges across {len(results)} regimes")

    return results


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Causal Engine — Phase 26 Sprint 2 (6A)")
    parser.add_argument("--discover", action="store_true", help="Run causal discovery")
    parser.add_argument("--multi-regime", action="store_true", help="Run per-regime discovery")
    parser.add_argument("--regime", type=str, default=None, help="Specific regime to analyze")
    parser.add_argument("--lookback-days", type=int, default=30, help="Days of data to analyze")
    parser.add_argument("--status", action="store_true", help="Show discovery status")
    parser.add_argument("--validate", action="store_true", help="Validate seed synapses")
    parser.add_argument("--parents", type=str, help="Show causal parents of a variable")
    parser.add_argument("--path", nargs=2, metavar=("SOURCE", "TARGET"),
                       help="Find causal path from source to target")

    args = parser.parse_args()

    if args.status:
        engine = CausalEngine()
        engine.print_status()
        return

    if args.parents:
        engine = CausalEngine()
        parents = engine.get_causal_parents(args.parents)
        print(f"\nCausal parents of '{args.parents}':")
        for p in parents:
            print(f"  {p['source_var']} → {args.parents} "
                  f"(strength={p['causal_strength']:.3f}, lag={p['time_lag']}h)")
        return

    if args.path:
        engine = CausalEngine()
        paths = engine.get_causal_path(args.path[0], args.path[1])
        print(f"\nCausal paths: {args.path[0]} → {args.path[1]}")
        if not paths:
            print("  No paths found.")
        for i, path in enumerate(paths):
            chain = " → ".join(
                f"{e['from']}({e['strength']:.2f}, lag={e['lag']}h)" for e in path
            ) + f" → {args.path[1]}"
            print(f"  Path {i+1}: {chain}")
        return

    if args.discover:
        result = run_causal_discovery(
            regime=args.regime,
            lookback_days=args.lookback_days,
        )
        if "error" not in result:
            print(f"\nDiscovered {result['n_edges']} causal edges "
                  f"from {result['n_observations']} observations")
            print(f"Persisted: {result.get('persisted_count', 0)}")
        else:
            print(f"Error: {result['error']}")
        return

    if args.multi_regime:
        results = run_multi_regime_discovery(lookback_days=args.lookback_days)
        for regime, result in results.items():
            if isinstance(result, dict):
                print(f"  {regime}: {result.get('n_edges', 0)} edges")
        return

    if args.validate:
        engine = CausalEngine()
        result = engine.discover(lookback_days=args.lookback_days)
        if "error" not in result:
            val = result.get("seed_validation", {})
            print("\nSeed Synapse Validation:")
            for e in val.get("confirmed", []):
                print(f"  CONFIRMED: {e['source']} → {e['target']} "
                      f"(strength={e.get('strength', '?')})")
            for e in val.get("refuted", []):
                print(f"  REFUTED:   {e['source']} → {e['target']} — {e.get('reason', '')}")
            for e in val.get("not_tested", []):
                print(f"  UNTESTED:  {e['source']} → {e['target']} — {e.get('reason', '')}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
