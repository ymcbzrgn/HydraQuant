#!/usr/bin/env bash
# Phase 30 A.1 — API key rotation reminder + helper.
# Does NOT auto-rotate. Reminds operator to rotate at exchange UI,
# then update .env, render config, deploy.
set -uo pipefail

cat <<EOF
=== HydraQuant API Key Rotation Checklist ===

For each compromised key, follow these steps:

1. BYBIT TESTNET (if exposed):
   - Visit https://testnet.bybit.com/app/user/api-management
   - REVOKE the leaked key.
   - Generate new key pair (read+trade perms only, NO withdraw).
   - Copy key+secret.

2. BINANCE TESTNET (if exposed):
   - Visit https://testnet.binancefuture.com/en/futures/BTCUSDT
   - Account -> API Management.
   - Delete leaked key, create new (Futures perms only).

3. UPDATE .env locally:
   - BYBIT_TESTNET_KEY="new-key"
   - BYBIT_TESTNET_SECRET="new-secret"
   - (and / or Binance equivalents)

4. RENDER configs:
   python scripts/render_config.py --all

5. SCP to server:
   scp .env hydra:/root/freqtrade/.env
   scp config_bybit_testnet_futures.json hydra:/root/freqtrade/

6. RESTART services:
   ssh hydra "systemctl restart freqtrade freqtrade-scheduler freqtrade-rag freqtrade-models freqtrade-ai-api"

7. VERIFY:
   ssh hydra "journalctl -u freqtrade -n 20 --no-pager | grep -i 'key\\|auth\\|login'"

8. PURGE git history (if leaked key was committed):
   git filter-repo --path config_*.json --invert-paths
   git push --force origin main  # ONLY if private repo + coordinated

EOF

if [ "${1:-}" = "--check-leaks" ]; then
    echo
    echo "=== Searching git history for leaked patterns... ==="
    if git rev-list --all --objects 2>/dev/null \
        | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' 2>/dev/null \
        | grep -E "AIza[A-Za-z0-9_\-]{30,}|sk-or-v1-[a-f0-9]{60,}|gsk_[A-Za-z0-9]{50,}" -c
    then
        echo "WARNING: leak patterns detected. Run git filter-repo immediately."
    else
        echo "No leak patterns in git history."
    fi
fi
