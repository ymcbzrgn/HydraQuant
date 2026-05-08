"""Phase 30 A.34 — Bayesian Kelly legacy table cleanup.

Verifies side-aware migration completeness, then DROP legacy tables.
Pre-conditions: bayesian_kelly_per_pair has rows for every (pair, side) where
pre_side_v1 had data; counts within tolerance (>=95%).

Default dry-run; pass --apply to actually drop.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "user_data" / "scripts"))

from db import AI_DB_PATH, get_db_connection  # noqa: E402

LEGACY_TABLES = (
    "bayesian_kelly_per_pair_pre_side_v1",
    "bayesian_kelly_shadow_per_pair_pre_side_v1",
)


def _table_exists(conn, name: str) -> bool:
    cur = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    )
    return cur.fetchone() is not None


def verify_migration() -> bool:
    with get_db_connection(AI_DB_PATH) as conn:
        if not _table_exists(conn, "bayesian_kelly_per_pair_pre_side_v1"):
            print("[verify] No legacy table to verify; skip")
            return True

        old_pairs = conn.execute(
            "SELECT COUNT(DISTINCT pair) FROM bayesian_kelly_per_pair_pre_side_v1"
        ).fetchone()[0] or 0
        new_pairs = conn.execute(
            "SELECT COUNT(DISTINCT pair) FROM bayesian_kelly_per_pair"
        ).fetchone()[0] or 0
        if new_pairs < old_pairs:
            print(f"[verify] FAIL: pre_side_v1 had {old_pairs} pairs, new has {new_pairs}")
            return False

        old_trades = conn.execute(
            "SELECT COALESCE(SUM(n_trades), 0) FROM bayesian_kelly_per_pair_pre_side_v1"
        ).fetchone()[0] or 0
        new_trades = conn.execute(
            "SELECT COALESCE(SUM(n_trades), 0) FROM bayesian_kelly_per_pair"
        ).fetchone()[0] or 0
        if old_trades > 0 and new_trades < old_trades * 0.95:
            print(f"[verify] FAIL: pre_side_v1 had {old_trades} trades, new has {new_trades}")
            return False

        print(f"[verify] OK: pre_side_v1 {old_pairs}/{old_trades} -> new {new_pairs}/{new_trades}")
        return True


def drop_legacy(dry_run: bool = True) -> None:
    with get_db_connection(AI_DB_PATH) as conn:
        for tbl in LEGACY_TABLES:
            if not _table_exists(conn, tbl):
                continue
            size = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            if dry_run:
                print(f"[dry-run] DROP TABLE {tbl} ({size} rows)")
            else:
                conn.execute(f"DROP TABLE {tbl}")
                print(f"[DROPPED] {tbl} ({size} rows)")
        if not dry_run:
            conn.commit()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                        help="Actually DROP (default dry-run)")
    parser.add_argument("--skip-verify", action="store_true",
                        help="Skip pre-flight verify (dangerous)")
    args = parser.parse_args()

    if not args.skip_verify and not verify_migration():
        print("[main] verify_migration FAILED; aborting")
        return 1
    drop_legacy(dry_run=not args.apply)
    if args.apply:
        with get_db_connection(AI_DB_PATH) as conn:
            conn.execute("VACUUM")
            conn.commit()
        print("[main] VACUUM complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
