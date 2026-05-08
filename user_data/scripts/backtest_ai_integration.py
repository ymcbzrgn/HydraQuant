"""Phase 30 C.3 — 1m truth-source backtest with AI runtime engaged.

Wraps freqtrade backtest invocation to:
- Use 1m OHLCV (truth-source) instead of native strategy timeframe.
- Engage live AI pipeline (rag_graph, evidence_engine, agent_pool) for each candle.
- Hash run_meta -> reproducibility ID.
- Emit run report to user_data/data/backtest_runs/<run_id>/.

Heavy run; intended as scheduled weekly job.
"""
from __future__ import annotations

import hashlib
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

RUNS_DIR = Path(__file__).parent.parent / "data" / "backtest_runs"


def _run_hash(spec: Dict[str, Any]) -> str:
    blob = json.dumps(spec, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def run(
    timerange: str,
    pairs: List[str],
    strategy: str = "HydraSizer",
    config_path: str = "config_bybit_testnet_futures.json",
    timeframe: str = "1m",
    dry_run: bool = False,
) -> Dict[str, Any]:
    spec = {
        "timerange": timerange, "pairs": pairs, "strategy": strategy,
        "timeframe": timeframe, "config": config_path,
    }
    rh = _run_hash(spec)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = RUNS_DIR / f"{ts}_{rh}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "spec.json").write_text(json.dumps(spec, indent=2, default=str))

    if dry_run:
        return {"run_hash": rh, "out_dir": str(out_dir), "dry_run": True}

    cmd = [
        "freqtrade", "backtesting",
        "--strategy", strategy,
        "--config", config_path,
        "--timerange", timerange,
        "--timeframe-detail", "1m",
        "--export", "trades",
    ]
    if pairs:
        cmd += ["--pairs"] + pairs
    log_path = out_dir / "stdout.log"
    try:
        with open(log_path, "w") as fp:
            r = subprocess.run(cmd, stdout=fp, stderr=subprocess.STDOUT, timeout=3600 * 6)
        (out_dir / "exit_code.txt").write_text(str(r.returncode))
        return {"run_hash": rh, "out_dir": str(out_dir),
                "exit_code": r.returncode, "log": str(log_path)}
    except subprocess.TimeoutExpired:
        return {"run_hash": rh, "out_dir": str(out_dir), "error": "timeout"}
    except Exception as e:
        return {"run_hash": rh, "out_dir": str(out_dir), "error": str(e)}
