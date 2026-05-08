#!/usr/bin/env python3
"""Phase 30 A.30 — Deploy verifier with structured JSON output."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def local_file_hashes(files: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for f in files:
        p = Path(f)
        if p.is_file():
            out[f] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def remote_file_hashes(host: str, path: str, files: List[str]) -> Dict[str, str]:
    if not files:
        return {}
    cmd = " && ".join([f"sha256sum {path}/{f} 2>/dev/null" for f in files])
    r = subprocess.run(
        ["ssh", host, cmd], capture_output=True, text=True, timeout=120
    )
    out: Dict[str, str] = {}
    for line in r.stdout.splitlines():
        if "  " in line:
            h, fpath = line.split("  ", 1)
            key = fpath.replace(path + "/", "").strip()
            out[key] = h.strip()
    return out


def main() -> int:
    host = sys.argv[1] if len(sys.argv) > 1 else "hydra"
    remote_path = "/root/freqtrade"

    proc = subprocess.run(
        [
            "git", "ls-tree", "-r", "HEAD", "--name-only",
            "user_data/scripts/", "user_data/strategies/",
        ],
        capture_output=True, text=True,
    )
    files = [f for f in proc.stdout.splitlines() if f.endswith(".py")]

    local = local_file_hashes(files)
    remote = remote_file_hashes(host, remote_path, files)

    mismatches = []
    for f, h in local.items():
        if remote.get(f) != h:
            mismatches.append(
                {
                    "file": f,
                    "local": h,
                    "remote": remote.get(f, "MISSING"),
                }
            )

    report = {
        "host": host,
        "remote_path": remote_path,
        "total_files": len(files),
        "matched": len(files) - len(mismatches),
        "mismatched": len(mismatches),
        "details": mismatches[:20],
    }
    print(json.dumps(report, indent=2))
    return 0 if not mismatches else 1


if __name__ == "__main__":
    sys.exit(main())
