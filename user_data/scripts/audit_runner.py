"""Phase 30 D.8 — Audit-as-code full integration.

Reads YAML test cases under audits/, executes them, reports pass/fail.
Hookable from CI (.github/workflows/audit.yml) or manual cron weekly.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

AUDITS_DIR = Path(__file__).parent.parent.parent / "audits"


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore

        return yaml.safe_load(path.read_text())
    except Exception:
        return _minimal_yaml(path.read_text())


def _minimal_yaml(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    cur_key = None
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" in line and not line.startswith(" "):
            k, v = line.split(":", 1)
            v = v.strip()
            cur_key = k.strip()
            out[cur_key] = v if v else {}
        elif cur_key and line.startswith("  "):
            inner = line.strip()
            if isinstance(out[cur_key], dict) and ":" in inner:
                ik, iv = inner.split(":", 1)
                out[cur_key][ik.strip()] = iv.strip()
    return out


def run_one(spec_path: Path) -> Dict[str, Any]:
    spec = _load_yaml(spec_path)
    name = spec.get("name", spec_path.stem)
    kind = spec.get("kind", "sql_check")
    try:
        if kind == "sql_check":
            from db import AI_DB_PATH, get_db_connection

            with get_db_connection(AI_DB_PATH) as conn:
                row = conn.execute(spec["sql"]).fetchone()
                v = (row[0] if row else 0)
                expected = int(spec.get("expected", 0))
                op = spec.get("op", "eq")
                ok = (op == "eq" and v == expected) or (op == "lte" and v <= expected) or \
                     (op == "gte" and v >= expected)
                return {"name": name, "kind": kind, "value": v, "expected": expected, "ok": bool(ok)}
        elif kind == "module_import":
            __import__(spec["module"])
            return {"name": name, "kind": kind, "ok": True}
    except Exception as e:
        return {"name": name, "kind": kind, "ok": False, "error": str(e)}
    return {"name": name, "kind": kind, "ok": False, "error": "unknown_kind"}


def run_all(audits_dir: Path = AUDITS_DIR) -> List[Dict[str, Any]]:
    if not audits_dir.is_dir():
        return []
    results: List[Dict[str, Any]] = []
    for f in sorted(audits_dir.glob("*.yaml")):
        results.append(run_one(f))
    return results


if __name__ == "__main__":
    out = run_all()
    print(json.dumps(out, indent=2))
    sys.exit(0 if all(r.get("ok") for r in out) else 1)
