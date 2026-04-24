#!/usr/bin/env python3
"""
HydraQuant Deploy Health Check Dashboard.

Usage (hydra sunucusunda):
    /root/freqtrade/.venv/bin/python /root/freqtrade/user_data/scripts/deploy_health_check.py

Usage (kontrol host'tan):
    ssh hydra '/root/freqtrade/.venv/bin/python /root/freqtrade/user_data/scripts/deploy_health_check.py'

Post-Mega+EK+Tur3 deploy (2026-04-24) verify için. Renkli terminal, metrik-başı PASS/WARN/FAIL.
"""
from __future__ import annotations
import os
import sys
import subprocess
import sqlite3
import time
from pathlib import Path
from typing import Tuple, List, Dict, Any

# Portable ai_config integration — falls back to hardcoded prod paths if script
# is run outside the HydraQuant user_data/scripts layout.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from ai_config import AI_DB_PATH as _AI_CONFIG_DB_PATH
except ImportError:
    _AI_CONFIG_DB_PATH = None

AI_DB = _AI_CONFIG_DB_PATH or "/root/freqtrade/user_data/db/ai_data.sqlite"
TRADES_DB = "/root/freqtrade/user_data/tradesv3.sqlite"
SERVICES = [
    "freqtrade",
    "freqtrade-scheduler",
    "freqtrade-rag",
    "freqtrade-models",
    "freqtrade-ai-api",
]

RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
BOLD = "\033[1m"
RESET = "\033[0m"
PASS = f"{GREEN}✅ PASS{RESET}"
WARN = f"{YELLOW}⚠️  WARN{RESET}"
FAIL = f"{RED}❌ FAIL{RESET}"


def section(title: str) -> None:
    print(f"\n{BOLD}{BLUE}━━━ {title} ━━━{RESET}")


def check(label: str, verdict: str, detail: str = "") -> None:
    print(f"  {verdict}  {label}" + (f"  {detail}" if detail else ""))


def run(cmd: List[str]) -> Tuple[int, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return r.returncode, (r.stdout or r.stderr or "").strip()
    except Exception as e:
        return 1, str(e)


def q(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list:
    try:
        return conn.execute(sql, params).fetchall()
    except Exception as e:
        return [("ERR", str(e))]


def count_results() -> Dict[str, int]:
    return {"pass": 0, "warn": 0, "fail": 0}


TOTALS = count_results()


def mark(verdict: str) -> str:
    if verdict == PASS:
        TOTALS["pass"] += 1
    elif verdict == WARN:
        TOTALS["warn"] += 1
    elif verdict == FAIL:
        TOTALS["fail"] += 1
    return verdict


# ── 1. Services ───────────────────────────────────────────────────
def check_services() -> None:
    section("1. Systemd Services")
    for s in SERVICES:
        rc, out = run(["systemctl", "is-active", s])
        if out == "active":
            rc2, pid = run(["systemctl", "show", s, "--property=MainPID", "--value"])
            rc3, mem = run(["systemctl", "show", s, "--property=MemoryCurrent", "--value"])
            try:
                mb = int(mem) // (1024 * 1024) if mem.isdigit() else 0
            except Exception:
                mb = 0
            detail = f"PID={pid} mem={mb}MB"
            verdict = PASS
            if s == "freqtrade" and mb > 2000:
                verdict = WARN
                detail += " (>2GB, pre-OOM zone)"
            elif s == "freqtrade-scheduler" and mb > 2500:
                verdict = WARN
                detail += " (>2.5GB)"
            check(s, mark(verdict), detail)
        else:
            check(s, mark(FAIL), f"status={out}")


# ── 2. OOM-kill + Restart ─────────────────────────────────────────
def check_restarts() -> None:
    section("2. OOM-Kill + Restart History (24h)")
    rc, out = run([
        "journalctl", "-u", "freqtrade", "--since", "24 hours ago",
        "--no-pager", "-o", "cat",
    ])
    oom = out.count("oom-kill")
    restart = out.count("Scheduled restart job")
    detail = f"oom_kill={oom} scheduled_restart={restart}"
    if oom == 0 and restart <= 1:
        check("freqtrade OOM stability", mark(PASS), detail)
    elif oom <= 2 or restart <= 5:
        check("freqtrade OOM stability", mark(WARN), detail)
    else:
        check("freqtrade OOM stability", mark(FAIL), detail)


# ── 3. Signal Source Distribution ─────────────────────────────────
def check_signal_dist() -> None:
    section("3. Signal Source Distribution (last 6h)")
    try:
        conn = sqlite3.connect(AI_DB)
        rows = q(conn, """
            SELECT signal_source, COUNT(*) FROM signal_health
            WHERE datetime(timestamp) >= datetime('now', '-6 hours')
            GROUP BY signal_source ORDER BY 2 DESC
        """)
        if not rows:
            check("signal_health 6h activity", mark(WARN), "no rows")
            return
        total = sum(r[1] for r in rows)
        dist = {r[0]: r[1] for r in rows}
        ensemble_cnt = dist.get("ENSEMBLE", 0) + dist.get("AGENT_POOL", 0) + dist.get("COORDINATOR", 0)
        ee_cnt = dist.get("EVIDENCE_ENGINE", 0)
        ensemble_pct = 100 * ensemble_cnt / max(total, 1)
        detail = f"total={total} | " + " | ".join(f"{k}={v}" for k, v in dist.items())
        check("Distribution", mark(PASS), detail)
        # Ensemble ratio target ≥ 25% (Tier 3.5 threshold 0.70 ile)
        if ensemble_pct >= 25:
            check(f"ENSEMBLE/COORD/POOL ratio {ensemble_pct:.1f}%", mark(PASS), "target ≥ 25%")
        elif ensemble_pct >= 10:
            check(f"ENSEMBLE/COORD/POOL ratio {ensemble_pct:.1f}%", mark(WARN), "target ≥ 25%")
        else:
            check(f"ENSEMBLE/COORD/POOL ratio {ensemble_pct:.1f}%", mark(FAIL), "MADAM still bypassed")
        # EE dominance check
        ee_pct = 100 * ee_cnt / max(total, 1)
        if ee_pct < 70:
            check(f"EvidenceFirst EE bypass {ee_pct:.1f}%", mark(PASS), "Tier 3.5 threshold working")
        else:
            check(f"EvidenceFirst EE bypass {ee_pct:.1f}%", mark(WARN), "EE still dominant — check threshold")
        conn.close()
    except Exception as e:
        check("signal_dist check", mark(FAIL), str(e))


# ── 4. LLM Calls — pair + task_name propagation ───────────────────
def check_llm_calls() -> None:
    section("4. LLM Calls: pair + task_name Propagation (last 1h)")
    try:
        conn = sqlite3.connect(AI_DB)
        rows = q(conn, """
            SELECT COUNT(*) total,
                   SUM(CASE WHEN agent_name != '' AND agent_name != 'default' THEN 1 ELSE 0 END) with_task,
                   SUM(CASE WHEN trading_pair != '' THEN 1 ELSE 0 END) with_pair,
                   SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) errors
            FROM llm_calls
            WHERE datetime(timestamp) >= datetime('now', '-1 hour')
        """)
        if not rows or rows[0][0] == 0:
            check("LLM activity 1h", mark(WARN), "no calls — bot idle?")
            return
        total, with_task, with_pair, errors = rows[0]
        task_pct = 100 * with_task / max(total, 1)
        pair_pct = 100 * with_pair / max(total, 1)
        err_pct = 100 * errors / max(total, 1)
        check(f"Total {total} calls", mark(PASS), "")
        # task_name propagation ≥ 15% (background default kalan %70+)
        if task_pct >= 15:
            check(f"task_name filled {task_pct:.1f}%", mark(PASS), f"{with_task}/{total} specific tag")
        elif task_pct >= 5:
            check(f"task_name filled {task_pct:.1f}%", mark(WARN), "most calls default")
        else:
            check(f"task_name filled {task_pct:.1f}%", mark(FAIL), "C2 fix not propagating")
        # pair propagation ≥ 15%
        if pair_pct >= 15:
            check(f"trading_pair filled {pair_pct:.1f}%", mark(PASS), f"{with_pair}/{total}")
        elif pair_pct >= 5:
            check(f"trading_pair filled {pair_pct:.1f}%", mark(WARN), "")
        else:
            check(f"trading_pair filled {pair_pct:.1f}%", mark(FAIL), "C2 retroactive dead")
        # Error rate
        if err_pct < 10:
            check(f"Error rate {err_pct:.1f}%", mark(PASS), f"{errors} errors")
        elif err_pct < 25:
            check(f"Error rate {err_pct:.1f}%", mark(WARN), "")
        else:
            check(f"Error rate {err_pct:.1f}%", mark(FAIL), "provider issues")
        # Task distribution top 5
        rows2 = q(conn, """
            SELECT agent_name, COUNT(*) FROM llm_calls
            WHERE datetime(timestamp) >= datetime('now', '-1 hour')
              AND agent_name != '' AND agent_name != 'default'
            GROUP BY agent_name ORDER BY 2 DESC LIMIT 8
        """)
        if rows2:
            dist_str = ", ".join(f"{r[0]}={r[1]}" for r in rows2)
            check(f"Task dist: {dist_str}", mark(PASS), "")
        conn.close()
    except Exception as e:
        check("llm_calls check", mark(FAIL), str(e))


# ── 5. Kelly State ─────────────────────────────────────────────────
def check_kelly() -> None:
    section("5. Bayesian Kelly (Real + Shadow)")
    try:
        conn = sqlite3.connect(AI_DB)
        real = q(conn, "SELECT COUNT(*), SUM(n_trades), AVG(alpha), AVG(beta_param), AVG(alpha*1.0/(alpha+beta_param)) FROM bayesian_kelly_per_pair")
        shadow = q(conn, "SELECT COUNT(*), SUM(n_trades), AVG(alpha), AVG(beta_param) FROM bayesian_kelly_shadow_per_pair")
        if real:
            cnt, n_trades, a, b, p_win = real[0]
            detail = f"pairs={cnt} total_trades={n_trades or 0} α̅={a:.2f} β̅={b:.2f} p̅_win={(p_win or 0):.3f}"
            if (n_trades or 0) == 0:
                check("Real Kelly (post-reset, learning)", mark(PASS), detail + " — fresh prior")
            elif (p_win or 0) > 0.30:
                check("Real Kelly learning", mark(PASS), detail + " — healthy")
            elif (p_win or 0) > 0.15:
                check("Real Kelly learning", mark(WARN), detail + " — low confidence")
            else:
                check("Real Kelly learning", mark(FAIL), detail + " — re-corrupting?")
        if shadow:
            s_cnt, s_n, s_a, s_b = shadow[0]
            sdetail = f"pairs={s_cnt} total_updates={s_n or 0} α̅={(s_a or 0):.2f} β̅={(s_b or 0):.2f}"
            if (s_n or 0) > 0:
                check("Shadow Kelly feeding", mark(PASS), sdetail)
            else:
                check("Shadow Kelly feeding", mark(WARN), sdetail + " — no shadow updates yet")
        conn.close()
    except Exception as e:
        check("kelly check", mark(FAIL), str(e))


# ── 6. LinUCB State ────────────────────────────────────────────────
def check_linucb() -> None:
    section("6. LinUCB Bandit Learning")
    try:
        conn = sqlite3.connect(AI_DB)
        rows = q(conn, "SELECT COUNT(*), SUM(n_updates), MAX(n_updates) FROM linucb_state")
        if rows and rows[0][0]:
            cnt, total, max_u = rows[0]
            detail = f"slots={cnt} total_updates={total or 0} max_per_slot={max_u or 0}"
            if (total or 0) > 500:
                check("LinUCB learning phase", mark(PASS), detail + " — post cold-start")
            elif (total or 0) > 50:
                check("LinUCB learning phase", mark(PASS), detail + " — warm-up")
            else:
                check("LinUCB learning phase", mark(WARN), detail + " — early cold-start")
        else:
            check("LinUCB persistence", mark(WARN), "table empty (save not fired yet)")
        conn.close()
    except Exception as e:
        check("linucb check", mark(FAIL), str(e))


# ── 7. Trade Performance ──────────────────────────────────────────
def check_trades() -> None:
    section("7. Trade Performance (since deploy 2026-04-23 23:57)")
    try:
        conn = sqlite3.connect(TRADES_DB)
        # Post-deploy trades
        rows = q(conn, """
            SELECT COUNT(*) total,
                   SUM(CASE WHEN is_open=1 THEN 1 ELSE 0 END) open,
                   SUM(CASE WHEN is_open=0 AND close_profit_abs > 0 THEN 1 ELSE 0 END) wins,
                   SUM(CASE WHEN is_open=0 AND close_profit_abs <= 0 THEN 1 ELSE 0 END) losses,
                   ROUND(SUM(CASE WHEN is_open=0 THEN close_profit_abs END), 2) pnl,
                   ROUND(AVG(CASE WHEN close_profit_abs > 0 THEN close_profit_abs END), 2) avg_win,
                   ROUND(AVG(CASE WHEN close_profit_abs <= 0 AND is_open=0 THEN -close_profit_abs END), 2) avg_loss
            FROM trades WHERE open_date >= '2026-04-23 20:57:00'
        """)
        if rows:
            t, o, w, l, pnl, aw, al = rows[0]
            closed = (w or 0) + (l or 0)
            wr = 100 * (w or 0) / max(closed, 1) if closed > 0 else 0
            payoff = (aw / al) if (aw and al and al > 0) else 0
            detail = f"total={t} open={o} wins={w or 0} losses={l or 0} PnL=${pnl or 0} WR={wr:.0f}% payoff={payoff:.2f}"
            if t == 0:
                check("Post-deploy trades", mark(WARN), detail + " — bot idle (market quiet?)")
            elif closed == 0:
                check("Post-deploy trades", mark(PASS), detail + " — all still open")
            elif payoff >= 1.2 and wr >= 50:
                check("Post-deploy trade performance", mark(PASS), detail + " — HEALTHY")
            elif payoff >= 1.0 and wr >= 45:
                check("Post-deploy trade performance", mark(PASS), detail)
            elif payoff >= 0.8:
                check("Post-deploy trade performance", mark(WARN), detail)
            else:
                check("Post-deploy trade performance", mark(FAIL), detail + " — asymmetry persisting")
        conn.close()
    except Exception as e:
        check("trade perf check", mark(FAIL), str(e))


# ── 8. argument_quality learning ──────────────────────────────────
def check_argument_quality() -> None:
    section("8. argument_quality Learning (was_correct fix)")
    try:
        conn = sqlite3.connect(AI_DB)
        rows = q(conn, """
            SELECT COUNT(*) total_patterns,
                   SUM(CASE WHEN times_correct > 0 THEN 1 ELSE 0 END) with_correct,
                   SUM(times_used) total_usage,
                   SUM(times_correct) total_correct,
                   ROUND(AVG(quality_score), 3) avg_q
            FROM argument_quality
        """)
        if rows:
            tp, wc, tu, tc, aq = rows[0]
            detail = f"patterns={tp} with_correct={wc} usage={tu or 0} correct={tc or 0} avg_q={aq or 0:.3f}"
            if (tc or 0) > 0 and (wc or 0) > 0:
                check("argument_quality learning", mark(PASS), detail + " — was_correct fix working")
            elif (tu or 0) > 10:
                check("argument_quality learning", mark(WARN), detail + " — no correct yet")
            else:
                check("argument_quality learning", mark(WARN), detail + " — needs closed trades")
        conn.close()
    except Exception as e:
        check("argument_quality", mark(FAIL), str(e))


# ── 9. Causal Discoveries + Dream ─────────────────────────────────
def check_engines() -> None:
    section("9. Causal + Dream + Hypothesis Engines")
    try:
        conn = sqlite3.connect(AI_DB)
        for tbl, time_col in [
            ("causal_discoveries", "discovered_at"),
            ("dream_scenarios", "timestamp"),
            ("hypothesis_history", "created_at"),
        ]:
            rows = q(conn, f"SELECT COUNT(*), IFNULL(MAX({time_col}),'none') FROM {tbl}")
            if rows:
                cnt, last = rows[0]
                if cnt > 0:
                    check(f"{tbl}", mark(PASS), f"rows={cnt} last={last}")
                else:
                    check(f"{tbl}", mark(WARN), f"empty — cron not fired yet (weekly)")
        conn.close()
    except Exception as e:
        check("engines check", mark(FAIL), str(e))


# ── 10. Scheduler Jobs + RSS ──────────────────────────────────────
def check_scheduler() -> None:
    section("10. Scheduler Health (RSS hang + job activity)")
    rc, out = run([
        "journalctl", "-u", "freqtrade-scheduler", "--since", "1 hour ago", "--no-pager",
    ])
    skipped = out.count("skipped: maximum number of running instances")
    if skipped == 0:
        check("APScheduler job misfire", mark(PASS), "no skips in 1h")
    elif skipped < 5:
        check("APScheduler job misfire", mark(WARN), f"{skipped} skips")
    else:
        check("APScheduler job misfire", mark(FAIL), f"{skipped} skips — hung job?")
    # RSS fetch activity
    rss_fetches = out.count("Fetching RSS feed from")
    if rss_fetches >= 20:
        check(f"RSS fetch activity", mark(PASS), f"{rss_fetches} fetches/1h")
    elif rss_fetches > 0:
        check(f"RSS fetch activity", mark(WARN), f"{rss_fetches} fetches — slow?")
    else:
        check(f"RSS fetch activity", mark(FAIL), "RSS possibly hung")


# ── 11. Scheduler Memory Trend ────────────────────────────────────
def check_memory() -> None:
    section("11. Memory Trend (trend toward/away from 2GB)")
    rc, status = run(["systemctl", "show", "freqtrade", "--property=MemoryCurrent,MemoryPeak", "--value"])
    if rc == 0 and status:
        lines = status.splitlines()
        if len(lines) >= 2:
            try:
                current = int(lines[0]) // (1024 * 1024)
                peak = int(lines[1]) // (1024 * 1024) if lines[1].isdigit() else 0
                detail = f"current={current}MB peak={peak}MB"
                if current < 1500 and peak < 2000:
                    check("freqtrade memory", mark(PASS), detail)
                elif current < 1800:
                    check("freqtrade memory", mark(WARN), detail)
                else:
                    check("freqtrade memory", mark(FAIL), detail + " — OOM risk")
            except Exception:
                check("memory", mark(WARN), "parse failed")


# ── 12. Recent Tracebacks ─────────────────────────────────────────
def check_tracebacks() -> None:
    section("12. Recent Python Tracebacks (6h)")
    for s in ["freqtrade", "freqtrade-scheduler", "freqtrade-rag"]:
        rc, out = run([
            "journalctl", "-u", s, "--since", "6 hours ago", "--no-pager",
        ])
        # Filter out known benign patterns
        benign = ["OrderNotFound", "un_watch_ohlcv", "Invalid HTTP request"]
        tb_count = 0
        for line in out.splitlines():
            if "Traceback" in line and not any(b in line for b in benign):
                tb_count += 1
        if tb_count == 0:
            check(f"{s} non-benign TB", mark(PASS), "0 new-code errors")
        elif tb_count < 5:
            check(f"{s} non-benign TB", mark(WARN), f"{tb_count} tracebacks")
        else:
            check(f"{s} non-benign TB", mark(FAIL), f"{tb_count} tracebacks — inspect!")


# ── MAIN ──────────────────────────────────────────────────────────
def main() -> int:
    t0 = time.time()
    print(f"{BOLD}{BLUE}╔═══════════════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}{BLUE}║  HydraQuant Deploy Health Check — {time.strftime('%Y-%m-%d %H:%M:%S')}  ║{RESET}")
    print(f"{BOLD}{BLUE}║  Post: Mega+EK+RevizeTur2 (4739c3855) + RevizeTur3 (53f6e3df2)  ║{RESET}")
    print(f"{BOLD}{BLUE}╚═══════════════════════════════════════════════════════╝{RESET}")

    check_services()
    check_restarts()
    check_signal_dist()
    check_llm_calls()
    check_kelly()
    check_linucb()
    check_trades()
    check_argument_quality()
    check_engines()
    check_scheduler()
    check_memory()
    check_tracebacks()

    # Summary
    section("FINAL VERDICT")
    total_checks = TOTALS["pass"] + TOTALS["warn"] + TOTALS["fail"]
    elapsed = time.time() - t0
    print(f"  {GREEN}Pass{RESET}:  {TOTALS['pass']}")
    print(f"  {YELLOW}Warn{RESET}:  {TOTALS['warn']}")
    print(f"  {RED}Fail{RESET}:  {TOTALS['fail']}")
    print(f"  Total: {total_checks} checks in {elapsed:.1f}s")
    print()

    if TOTALS["fail"] == 0 and TOTALS["warn"] <= 3:
        print(f"{BOLD}{GREEN}🚀 OVERALL: DEPLOY HEALTHY{RESET}")
        return 0
    elif TOTALS["fail"] <= 2:
        print(f"{BOLD}{YELLOW}⚠️  OVERALL: NEEDS ATTENTION — inspect warns above{RESET}")
        return 1
    else:
        print(f"{BOLD}{RED}💀 OVERALL: DEPLOY UNHEALTHY — rollback candidate{RESET}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
