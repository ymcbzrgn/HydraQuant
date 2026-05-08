#!/usr/bin/env python3
"""Phase 30 A.1 — Config template renderer.

Reads `config_*.template.json` (placeholder ${ENV_VAR}), substitutes from .env,
writes `config_*.json` (gitignored). Run before every deploy.

Usage:
    python scripts/render_config.py [--all] [config_x.template.json]
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Iterable

PLACEHOLDER_RE = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip().strip('"').strip("'")
        if k.strip() and k.strip() not in os.environ:
            os.environ[k.strip()] = v


def render(template_text: str) -> str:
    missing = []

    def repl(m: re.Match) -> str:
        key = m.group(1)
        v = os.environ.get(key)
        if v is None:
            missing.append(key)
            return m.group(0)
        return v

    rendered = PLACEHOLDER_RE.sub(repl, template_text)
    if missing:
        raise SystemExit(
            f"[render_config] Missing env vars: {sorted(set(missing))}. "
            f"Populate .env or export them first."
        )
    return rendered


def render_one(template_path: Path) -> Path:
    target = template_path.with_name(template_path.name.replace(".template.json", ".json"))
    text = template_path.read_text()
    target.write_text(render(text))
    print(f"[render_config] {template_path} -> {target}")
    return target


def find_templates(root: Path) -> Iterable[Path]:
    return sorted(root.glob("config_*.template.json"))


def main() -> int:
    project_root = Path(__file__).resolve().parent.parent
    load_dotenv(project_root / ".env")

    args = sys.argv[1:]
    if not args or args[0] == "--all":
        templates = list(find_templates(project_root))
        if not templates:
            print("[render_config] No config_*.template.json files found")
            return 0
        for t in templates:
            render_one(t)
    else:
        for a in args:
            render_one(Path(a).resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
