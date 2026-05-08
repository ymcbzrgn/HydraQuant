"""Phase 30 C.11 — Workflow DAG YAML loader + runner.

Loads workflow_dag.yaml, builds dependency graph, executes nodes in topological
order with failure policies (abort / retry_N / log_only).
"""
from __future__ import annotations

import importlib
import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DAG_PATH = Path(__file__).parent.parent / "data" / "workflow_dag.yaml"


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore

        return yaml.safe_load(path.read_text())
    except Exception:
        return _minimal_yaml_parse(path.read_text())


def _minimal_yaml_parse(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"nodes": [], "failure_policy": {}}
    cur: Optional[Dict[str, Any]] = None
    section = None
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if line.startswith("nodes:"):
            section = "nodes"; continue
        if line.startswith("failure_policy:"):
            section = "failure_policy"; continue
        if section == "nodes":
            if line.lstrip().startswith("- id:"):
                if cur:
                    out["nodes"].append(cur)
                cur = {"id": line.split(":", 1)[1].strip()}
            elif cur and ":" in line:
                k, v = line.strip().split(":", 1)
                k = k.strip()
                v = v.strip()
                if v.startswith("[") and v.endswith("]"):
                    v = [x.strip() for x in v[1:-1].split(",") if x.strip()]
                cur[k] = v
        elif section == "failure_policy" and ":" in line and not line.startswith(" "):
            continue
        elif section == "failure_policy" and ":" in line:
            k, v = line.strip().split(":", 1)
            out["failure_policy"][k.strip()] = v.strip()
    if cur:
        out["nodes"].append(cur)
    return out


def _topo_sort(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_id = {n["id"]: n for n in nodes}
    indeg = defaultdict(int)
    children = defaultdict(list)
    for n in nodes:
        deps = n.get("depends_on", []) or []
        for d in deps:
            indeg[n["id"]] += 1
            children[d].append(n["id"])
    queue = deque([n["id"] for n in nodes if indeg[n["id"]] == 0])
    out: List[str] = []
    while queue:
        cur = queue.popleft()
        out.append(cur)
        for c in children[cur]:
            indeg[c] -= 1
            if indeg[c] == 0:
                queue.append(c)
    return [by_id[i] for i in out]


@dataclass
class NodeResult:
    node_id: str
    ok: bool
    output: Any = None
    error: str = ""


def _resolve_handler(module_name: str, handler: str):
    mod = importlib.import_module(module_name)
    if "." in handler:
        cls_name, method = handler.split(".", 1)
        cls = getattr(mod, cls_name)
        return getattr(cls(), method)
    return getattr(mod, handler)


def run(dag_path: Path = DAG_PATH, context: Optional[Dict[str, Any]] = None) -> List[NodeResult]:
    spec = _load_yaml(dag_path)
    ordered = _topo_sort(spec.get("nodes", []))
    policy = spec.get("failure_policy", {}) or {}
    results: List[NodeResult] = []
    ctx = dict(context or {})
    for node in ordered:
        nid = node["id"]
        module = node.get("module", "")
        handler_name = node.get("handler", "")
        try:
            fn = _resolve_handler(module, handler_name)
            output = fn(**ctx) if callable(fn) else None
            results.append(NodeResult(nid, True, output=output))
            ctx[nid] = output
        except Exception as e:
            policy_action = policy.get(nid, "log_only")
            results.append(NodeResult(nid, False, error=str(e)))
            if policy_action == "abort":
                logger.error(f"[DAG] node {nid} failed; abort policy")
                break
            elif policy_action.startswith("retry_"):
                tries = int(policy_action.split("_", 1)[1])
                retry_ok = False
                for _ in range(tries):
                    try:
                        output = fn(**ctx)
                        results[-1] = NodeResult(nid, True, output=output)
                        ctx[nid] = output
                        retry_ok = True
                        break
                    except Exception:
                        continue
                if not retry_ok:
                    logger.error(f"[DAG] node {nid} retries exhausted; abort")
                    break
    return results
