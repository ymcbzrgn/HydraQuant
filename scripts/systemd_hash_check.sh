#!/usr/bin/env bash
# Phase 30 C.13 — systemd ExecStartPre hash-match deploy gate.
# Runs at every freqtrade.service start. Persists mismatch counts to DB.
# Always exits 0 (does not block startup); just logs scp deploy state.

set +e
cd /root/freqtrade || exit 0

COMMIT_HASH=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
TS=$(date -Iseconds)
mismatches=0
for f in $(git ls-tree -r HEAD --name-only user_data/scripts/ user_data/strategies/ 2>/dev/null | grep '\.py$'); do
    [ -f "$f" ] || continue
    head_hash=$(git show "HEAD:$f" 2>/dev/null | sha256sum 2>/dev/null | awk '{print $1}')
    file_hash=$(sha256sum "$f" 2>/dev/null | awk '{print $1}')
    if [ -n "$head_hash" ] && [ -n "$file_hash" ] && [ "$head_hash" != "$file_hash" ]; then
        mismatches=$((mismatches + 1))
    fi
done

DB="/root/freqtrade/user_data/db/ai_data.sqlite"
if [ -f "$DB" ]; then
    sqlite3 "$DB" "INSERT INTO deploy_hash_history (commit_hash, mismatches, ts) VALUES ('$COMMIT_HASH', $mismatches, '$TS');" 2>/dev/null
fi

if [ "$mismatches" -gt 0 ]; then
    echo "[hash_check] $mismatches files differ from HEAD commit (scp pattern detected)" >&2
fi
exit 0
