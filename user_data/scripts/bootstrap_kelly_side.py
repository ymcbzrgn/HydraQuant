#!/usr/bin/env python3
"""
bootstrap_kelly_side.py — Sprint 2026-05-05 (B-CONNECT C1 BOOTSTRAP)

Replay all closed trades from tradesv3.sqlite into the side-aware
bayesian_kelly_per_pair Beta posteriors (long vs short separated).

Without this, the freshly-migrated side='long' / side='short' rows are
empty and the bot has to learn the long/short asymmetry from scratch
(would take 30+ days of new trades). Bootstrap warms the posteriors with
the existing trade history so day-1 sizing already reflects the
historical edge per direction (BTC LONG losing vs BTC SHORT winning).

Usage:
    PYTHONPATH=$(pwd)/user_data/scripts \\
        .venv/bin/python3 user_data/scripts/bootstrap_kelly_side.py [--dry-run]
"""
import argparse
import os
import sqlite3
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

DEFAULT_TRADES_DB = "/root/freqtrade/user_data/tradesv3.sqlite"


def fetch_trades(db_path: str):
    if not os.path.exists(db_path):
        raise SystemExit(f"[Bootstrap] trades DB not found: {db_path}")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT pair, is_short, close_profit, close_profit_abs,
               open_date, close_date
        FROM trades
        WHERE close_date IS NOT NULL AND close_profit IS NOT NULL
        ORDER BY close_date ASC
    """).fetchall()
    conn.close()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", default=False,
                        help="Print summary; do not write to Kelly table")
    parser.add_argument("--db-path", default=DEFAULT_TRADES_DB)
    args = parser.parse_args()

    print(f"[Bootstrap] Reading trades from: {args.db_path}")
    trades = fetch_trades(args.db_path)
    print(f"[Bootstrap] Found {len(trades)} closed trades")

    if args.dry_run:
        print("[Bootstrap] DRY-RUN — no DB writes\n")
        c = Counter()
        for t in trades:
            side = "short" if t["is_short"] else "long"
            won = (t["close_profit"] or 0) > 0
            c[(t["pair"], side, "win" if won else "loss")] += 1
        pairs = sorted({p for (p, _s, _k) in c.keys()})
        print(f"  {'Pair':<25} {'LONG W/L (n)':>20} {'SHORT W/L (n)':>20}")
        print("  " + "-" * 70)
        for p in pairs:
            lw = c.get((p, "long", "win"), 0)
            ll = c.get((p, "long", "loss"), 0)
            sw = c.get((p, "short", "win"), 0)
            sl = c.get((p, "short", "loss"), 0)
            print(f"  {p:<25} {lw:>4}/{ll:<4} (n={lw+ll:>3})    "
                  f"{sw:>4}/{sl:<4} (n={sw+sl:>3})")
        return

    # Live mode — replay into Kelly posteriors
    from position_sizer import get_real_kelly
    kelly = get_real_kelly()
    print(f"[Bootstrap] Replaying into: {kelly.table_name}\n")

    counts = defaultdict(int)
    skipped = 0
    for i, t in enumerate(trades, 1):
        try:
            pair = t["pair"]
            if not pair:
                skipped += 1
                continue
            side = "short" if t["is_short"] else "long"
            won = (t["close_profit"] or 0) > 0
            pnl_pct = float(t["close_profit"] or 0)
            kelly.update(
                won=won, pnl_pct=pnl_pct,
                pair=pair, regime="_global", side=side,
            )
            counts[f"{side}_{'win' if won else 'loss'}"] += 1
            if i % 200 == 0:
                print(f"  [Bootstrap] progress: {i}/{len(trades)} replayed")
        except Exception as exc:
            print(f"  [WARN] trade #{i} skipped: {exc}")
            skipped += 1

    print(f"\n[Bootstrap] Complete:")
    print(f"  LONG  wins={counts['long_win']:>4}  losses={counts['long_loss']:>4}")
    print(f"  SHORT wins={counts['short_win']:>4} losses={counts['short_loss']:>4}")
    print(f"  TOTAL replayed = {sum(counts.values())}, skipped = {skipped}")
    print()
    print("[Bootstrap] Verifying posteriors written ...")

    # Sanity-check: read back top 5 pairs by sample count
    db_path = kelly.db_path
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(f"""
        SELECT pair, side, alpha, beta_param, n_trades,
               ROUND(alpha / (alpha + beta_param), 3) AS p_win
        FROM {kelly.table_name}
        WHERE side IN ('long', 'short')
        ORDER BY n_trades DESC
        LIMIT 10
    """).fetchall()
    conn.close()

    print(f"  {'Pair':<25} {'Side':>6} {'n':>4} {'α':>8} {'β':>8} {'p_win':>6}")
    print("  " + "-" * 65)
    for r in rows:
        print(f"  {r['pair']:<25} {r['side']:>6} {r['n_trades']:>4} "
              f"{r['alpha']:>8.2f} {r['beta_param']:>8.2f} {r['p_win']:>6.3f}")


if __name__ == "__main__":
    main()
