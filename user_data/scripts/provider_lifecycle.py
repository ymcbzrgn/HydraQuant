"""Phase 30 A.24 — Provider lifecycle cleanup.

LLM router and any HTTP-pooled service (RAG, model_server, exchange API)
must close their connections + cancel pending tasks at shutdown to avoid
file-descriptor leaks during systemd restart cascades.

Registry pattern: services register a `shutdown_callback`; shutdown_all() runs
them on atexit + SIGTERM/SIGINT.
"""
from __future__ import annotations

import atexit
import logging
import signal
import threading
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)

_CALLBACKS: List[Tuple[str, Callable[[], None]]] = []
_LOCK = threading.RLock()
_SHUTDOWN_DONE = False


def register(name: str, cb: Callable[[], None]) -> None:
    with _LOCK:
        _CALLBACKS.append((name, cb))
    logger.debug(f"[ProviderLifecycle] registered {name}")


def shutdown_all(reason: str = "atexit") -> None:
    global _SHUTDOWN_DONE
    with _LOCK:
        if _SHUTDOWN_DONE:
            return
        _SHUTDOWN_DONE = True
        cbs = list(_CALLBACKS)
    logger.info(f"[ProviderLifecycle] shutdown_all reason={reason} n={len(cbs)}")
    for name, cb in cbs:
        try:
            cb()
            logger.debug(f"[ProviderLifecycle] {name} closed")
        except Exception as e:
            logger.error(f"[ProviderLifecycle] {name} close failed: {e}")


def _signal_handler(signum, frame):
    logger.warning(f"[ProviderLifecycle] signal {signum} -> shutdown_all")
    shutdown_all(reason=f"signal_{signum}")


_INSTALLED = False


def install(handle_signals: bool = True) -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    atexit.register(shutdown_all)
    if handle_signals:
        try:
            signal.signal(signal.SIGTERM, _signal_handler)
            signal.signal(signal.SIGINT, _signal_handler)
        except (ValueError, AttributeError):
            pass
    _INSTALLED = True
    logger.info("[ProviderLifecycle] handlers installed")


def reset_for_tests() -> None:
    global _SHUTDOWN_DONE, _INSTALLED
    with _LOCK:
        _CALLBACKS.clear()
        _SHUTDOWN_DONE = False
        _INSTALLED = False
