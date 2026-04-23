"""Isolated subprocess entrypoint for the world-model + dream cycle.

Background: when `_world_model_and_dream` ran inside the long-lived
scheduler process, PyTorch's GPU/heap allocators held on to ~600-900 MB
between runs and compounded toward OOM-kill territory. Moving the whole
cycle into a fresh subprocess means the heap dies with the interpreter
and the scheduler RSS stays flat.

Contract (from scheduler._world_model_and_dream):

    python -u dream_runner.py '{"config": {}, "db_path": "...",
                                "pair_list": ["BTC/USDT", ...]}'

Exit codes: 0 success, 2 runtime error. Stdout is piped back into the
scheduler logger with the `[DreamRunner]` prefix.
"""
from __future__ import annotations

import json
import logging
import os
import resource
import sys


def _dream_runner(config: dict, db_path: str, pair_list: list) -> None:
    # 1.5 GB virtual-address cap is a safety net so a runaway tensor op
    # kills this subprocess instead of swapping out the trading bot.
    # Tur-2 (M11): log success/failure so we can see in journalctl whether
    # the limit was actually applied. Platforms without AS (macOS dev
    # laptops) silently skipped before.
    try:
        resource.setrlimit(resource.RLIMIT_AS, (1_610_612_736, -1))
        print("[DreamRunner] RLIMIT_AS set to 1.5GB", flush=True)
    except Exception as _rlim_err:
        print(
            f"[DreamRunner] RLIMIT_AS setrlimit failed: {_rlim_err} — running unbounded",
            flush=True,
        )

    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)

    # Revize Tur-2 (H2): lock AI_DB_PATH for every downstream import BEFORE
    # we load ai_config (which captures the env var at import time). If the
    # scheduler did not supply one, fall back to ai_config's own resolution
    # so the subprocess does not silently write to the wrong SQLite file.
    if db_path:
        os.environ["AI_DB_PATH"] = db_path
    import ai_config  # noqa: F401 — side effect: freezes AI_DB_PATH
    if not db_path:
        os.environ.setdefault("AI_DB_PATH", ai_config.AI_DB_PATH)

    logging.basicConfig(
        level=logging.INFO,
        format="[DreamRunner] %(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )
    log = logging.getLogger(__name__)

    try:
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    try:
        from world_model import get_world_model, WM_MODEL_PATH
        from dream_engine import get_dream_engine

        wm = get_world_model()
        try:
            wm.load(WM_MODEL_PATH)
        except Exception as load_err:
            log.info("world_model load skipped (%s) — starting cold", load_err)

        train_result = wm.train_from_buffer(n_epochs=30, batch_size=64)
        if "error" in train_result:
            log.info("train=%s", json.dumps(train_result))
        else:
            log.info("train=%s", json.dumps({k: float(v) for k, v in train_result.items()}))
            try:
                wm.save(WM_MODEL_PATH)
            except Exception as save_err:
                log.warning("world_model save failed: %s", save_err)

        de = get_dream_engine()
        dream_result = de.dream_session(n_dreams=100, horizon=5)
        log.info("dream=%s", json.dumps(dream_result, default=str))
        sys.exit(0)
    except Exception as e:
        log.exception("dream_runner failed: %s", e)
        sys.exit(2)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: dream_runner.py <json-payload>", file=sys.stderr)
        sys.exit(64)
    payload = json.loads(sys.argv[1])
    _dream_runner(
        payload.get("config", {}),
        payload.get("db_path", ""),
        payload.get("pair_list", []) or [],
    )
