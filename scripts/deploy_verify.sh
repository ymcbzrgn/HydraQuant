#!/usr/bin/env bash
# Phase 30 A.30 — Deploy hash verifier (scp pattern auditor).
# Compares local HEAD-commit hashes vs hydra server filesystem hashes.
# Exit 0 if all match, 1 if any mismatch.
set -uo pipefail

REMOTE_HOST="${1:-hydra}"
REMOTE_PATH="${2:-/root/freqtrade}"
LOCAL_LIST=$(mktemp)
REMOTE_LIST=$(mktemp)
trap 'rm -f "$LOCAL_LIST" "$REMOTE_LIST"' EXIT

# Local: HEAD commit hashes (not working tree, since scp may diverge)
git ls-tree -r HEAD --name-only user_data/scripts/ user_data/strategies/ \
    | grep '\.py$' \
    | while read -r f; do
        if [ -f "$f" ]; then
            hash=$(sha256sum "$f" | awk '{print $1}')
            echo "$hash  $f"
        fi
      done | sort > "$LOCAL_LIST"

# Remote: filesystem hashes
ssh "$REMOTE_HOST" "cd $REMOTE_PATH && find user_data/scripts user_data/strategies -name '*.py' -type f -exec sha256sum {} \;" \
    | sort > "$REMOTE_LIST"

mismatches=$(diff "$LOCAL_LIST" "$REMOTE_LIST" | grep -c "^[<>]" || true)
if [ "$mismatches" -eq 0 ]; then
    n_files=$(wc -l < "$LOCAL_LIST")
    echo "[deploy_verify] OK — $n_files files match"
    exit 0
else
    echo "[deploy_verify] MISMATCH — $mismatches diff lines"
    diff "$LOCAL_LIST" "$REMOTE_LIST" | head -50
    exit 1
fi
