"""Phase 30 A.35 — Systemd restart event capture + Telegram alert.

Polls 6 freqtrade-related services every 5 minutes; detects NRestarts delta;
classifies cause (oom/segfault/timeout/exit_code/unknown) from journalctl;
writes service_restart_events row; emits CRITICAL via severity_router.
"""
from __future__ import annotations

import logging
import subprocess
from typing import Dict, List

logger = logging.getLogger(__name__)

SERVICES: List[str] = [
    "freqtrade.service",
    "freqtrade-scheduler.service",
    "freqtrade-rag.service",
    "freqtrade-models.service",
    "freqtrade-ai-api.service",
    "freqtrade-tr-dry.service",
]


def _systemctl_show(svc: str) -> Dict[str, str]:
    try:
        r = subprocess.run(
            [
                "systemctl", "show", svc,
                "--property=NRestarts,ActiveEnterTimestamp,ExecMainStartTimestamp,Result",
            ],
            capture_output=True, text=True, timeout=10,
        )
    except Exception as e:
        logger.error(f"[RestartMonitor] systemctl show failed for {svc}: {e}")
        return {}
    out: Dict[str, str] = {}
    for line in r.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _classify_cause(svc: str, props: Dict[str, str]) -> str:
    try:
        r = subprocess.run(
            ["journalctl", "-u", svc, "-n", "10", "--no-pager"],
            capture_output=True, text=True, timeout=10,
        )
    except Exception:
        return "unknown"
    text = r.stdout.lower()
    if "out of memory" in text or "oom-killer" in text or "killed process" in text:
        return "oom"
    if "segmentation fault" in text or "core dumped" in text:
        return "segfault"
    if "timeout" in text or "timed out" in text:
        return "timeout"
    if props.get("Result") == "exit-code":
        return "exit_code"
    if props.get("Result") == "signal":
        return "signal"
    return "unknown"


def check_all() -> Dict[str, dict]:
    """Iterate services; record new restart events; emit alerts.

    Returns dict {svc: {n_restarts, delta, cause}}.
    """
    from db import AI_DB_PATH, get_db_connection

    out: Dict[str, dict] = {}
    with get_db_connection(AI_DB_PATH) as conn:
        for svc in SERVICES:
            props = _systemctl_show(svc)
            if not props:
                continue
            try:
                n = int(props.get("NRestarts", "0") or 0)
            except ValueError:
                n = 0
            ts = props.get("ActiveEnterTimestamp", "")

            row = conn.execute(
                """SELECT n_restarts FROM service_restart_events
                   WHERE service = ? ORDER BY id DESC LIMIT 1""",
                (svc,),
            ).fetchone()
            prev_n = int(row[0]) if row else 0
            if n > prev_n:
                delta = n - prev_n
                cause = _classify_cause(svc, props)
                conn.execute(
                    """INSERT INTO service_restart_events
                       (service, n_restarts, last_restart_ts, delta_since_last, suspected_cause)
                       VALUES (?, ?, ?, ?, ?)""",
                    (svc, n, ts, delta, cause),
                )
                msg = f"[RestartMonitor] {svc} restart x{delta} (total={n}), cause={cause}"
                logger.warning(msg)
                try:
                    from severity_router import emit

                    emit(
                        kind="systemd.restart",
                        severity="critical",
                        message=msg,
                        payload={"service": svc, "n_restarts": n, "delta": delta, "cause": cause},
                    )
                except Exception:
                    pass
                out[svc] = {"n_restarts": n, "delta": delta, "cause": cause}
        conn.commit()
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import json

    print(json.dumps(check_all(), indent=2))
