"""Phase 30 C.4 — Trade Replay HTML site builder.

Per closed trade: HTML page with chart canvas (Plotly JSON), entry/exit annotations,
agent_pool debate transcript, evidence breakdown, regime, calibrator state.

Output: user_data/web/trade_replay/<trade_id>.html + index.html.
"""
from __future__ import annotations

import html
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

WEB_DIR = Path(__file__).parent.parent / "web" / "trade_replay"

INDEX_TEMPLATE = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>HydraQuant Trade Replay</title>
<style>body{font-family:monospace;background:#0a0a0a;color:#0f0;padding:20px}
table{border-collapse:collapse}td,th{padding:6px 12px;border:1px solid #0a4}
a{color:#4af}.win{color:#0f0}.loss{color:#f44}</style></head><body>
<h1>HydraQuant Trade Replay Index</h1>
<p>Built: {built_at}</p>
<table><tr><th>ID</th><th>Pair</th><th>Side</th><th>Open</th><th>Close</th><th>PnL%</th><th>Reason</th></tr>
{rows}
</table></body></html>"""

TRADE_TEMPLATE = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>Trade {tid} {pair}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>body{{font-family:monospace;background:#0a0a0a;color:#0f0;padding:20px}}
.win{{color:#0f0}}.loss{{color:#f44}}pre{{background:#111;padding:10px;border:1px solid #0a4}}
</style></head><body>
<h1>Trade #{tid} — {pair} {side}</h1>
<p>Open: {open_date} @ {open_rate} | Close: {close_date} @ {close_rate} | PnL: <span class="{pnl_class}">{pnl_pct:+.2%}</span> | Exit: {exit_reason}</p>
<div id="chart" style="width:100%;height:480px"></div>
<h2>Agent Debate</h2>
<pre>{debate}</pre>
<h2>Evidence</h2>
<pre>{evidence}</pre>
<script>Plotly.newPlot('chart', {plotly_json}, {{paper_bgcolor:'#111',plot_bgcolor:'#000',font:{{color:'#0f0'}}}});</script>
</body></html>"""


def build_for_trade(trade_id: int) -> Optional[Path]:
    try:
        import sqlite3

        with sqlite3.connect("/root/freqtrade/user_data/tradesv3.sqlite") as tdb:
            row = tdb.execute(
                """SELECT id, pair, is_short, open_date, close_date, open_rate, close_rate,
                          close_profit, exit_reason FROM trades WHERE id=?""",
                (trade_id,),
            ).fetchone()
        if not row:
            return None
        tid, pair, is_short, od, cd, op, cp, profit, exit_r = row
    except Exception:
        return None

    debate = "(no debate snapshot)"
    evidence = "(no evidence snapshot)"
    try:
        from db import AI_DB_PATH, get_db_connection

        with get_db_connection(AI_DB_PATH) as conn:
            cur = conn.execute(
                """SELECT reasoning_summary, agent_votes_json, regime FROM ai_decisions
                   WHERE pair=? AND DATETIME(timestamp) BETWEEN DATETIME(?, '-1 hour') AND DATETIME(?, '+1 hour')
                   ORDER BY id DESC LIMIT 1""",
                (pair, od, od),
            )
            r = cur.fetchone()
            if r:
                evidence = (r[0] or "")[:2000]
                debate = (r[1] or "")[:4000]
    except Exception:
        pass

    plotly_data = {
        "data": [{
            "x": [od, cd],
            "y": [op, cp],
            "type": "scatter",
            "mode": "lines+markers",
            "name": pair,
        }],
        "layout": {"title": f"{pair} entry/exit"},
    }

    pnl_class = "win" if (profit or 0) > 0 else "loss"
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    html_text = TRADE_TEMPLATE.format(
        tid=tid, pair=html.escape(pair), side="short" if is_short else "long",
        open_date=od, close_date=cd, open_rate=op, close_rate=cp,
        pnl_pct=float(profit or 0), pnl_class=pnl_class,
        exit_reason=html.escape(exit_r or ""),
        debate=html.escape(debate), evidence=html.escape(evidence),
        plotly_json=json.dumps(plotly_data, default=str),
    )
    out = WEB_DIR / f"{tid}.html"
    out.write_text(html_text)
    return out


def build_index(limit: int = 200) -> Optional[Path]:
    try:
        import sqlite3
        with sqlite3.connect("/root/freqtrade/user_data/tradesv3.sqlite") as tdb:
            rows = tdb.execute(
                f"""SELECT id, pair, is_short, open_date, close_date, close_profit, exit_reason
                    FROM trades WHERE close_date IS NOT NULL ORDER BY id DESC LIMIT {int(limit)}"""
            ).fetchall()
    except Exception:
        return None
    rows_html = []
    for tid, pair, is_short, od, cd, profit, exit_r in rows:
        cls = "win" if (profit or 0) > 0 else "loss"
        rows_html.append(
            f'<tr><td><a href="{tid}.html">{tid}</a></td>'
            f'<td>{html.escape(pair)}</td>'
            f'<td>{"S" if is_short else "L"}</td>'
            f'<td>{od}</td><td>{cd}</td>'
            f'<td class="{cls}">{(profit or 0) * 100:+.2f}%</td>'
            f'<td>{html.escape(exit_r or "")}</td></tr>'
        )
    WEB_DIR.mkdir(parents=True, exist_ok=True)
    out = WEB_DIR / "index.html"
    out.write_text(INDEX_TEMPLATE.format(
        built_at=datetime.now(timezone.utc).isoformat(),
        rows="\n".join(rows_html),
    ))
    return out
