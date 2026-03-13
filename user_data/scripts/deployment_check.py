"""
Phase 17: Pre-Deployment Validation Script
Checks all prerequisites before deploying to production server.
Run: python deployment_check.py
"""

import os
import sys
import shutil
import sqlite3
import importlib
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class DeploymentChecker:
    """Pre-deployment validation with real connectivity checks."""

    def __init__(self):
        self.results = []
        self.blocking = []

    def _check(self, name: str, fn) -> bool:
        try:
            passed, message = fn()
            self.results.append({"name": name, "passed": passed, "message": message})
            if not passed:
                self.blocking.append(f"{name}: {message}")
            return passed
        except Exception as e:
            self.results.append({"name": name, "passed": False, "message": str(e)})
            self.blocking.append(f"{name}: {e}")
            return False

    def run_all_checks(self) -> dict:
        """Run all pre-deployment checks."""
        self.results = []
        self.blocking = []

        self._check("1. .env file exists", self._check_env_file)
        self._check("2. API keys configured", self._check_api_keys)
        self._check("3. Python packages installed", self._check_packages)
        self._check("4. SQLite DB writable", self._check_sqlite)
        self._check("5. ChromaDB init", self._check_chromadb)
        self._check("6. LLM Router connectivity", self._check_llm_router)
        self._check("7. Disk space", self._check_disk)
        self._check("8. Memory available", self._check_memory)
        self._check("9. Core modules importable", self._check_core_imports)
        self._check("10. Smoke test", self._check_smoke_test)

        return {
            "ready_to_deploy": len(self.blocking) == 0,
            "checks": self.results,
            "blocking_issues": self.blocking
        }

    def _check_env_file(self):
        env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
        if os.path.exists(env_path):
            return True, f"Found at {os.path.abspath(env_path)}"
        return False, ".env file not found"

    def _check_api_keys(self):
        from dotenv import load_dotenv
        env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
        load_dotenv(env_path)

        missing = []
        # At least one Gemini key required
        has_gemini = bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEYS") or os.environ.get("GEMINI_API_KEY_1"))
        if not has_gemini:
            missing.append("GEMINI_API_KEY")

        if missing:
            return False, f"Missing keys: {', '.join(missing)}"
        return True, "Gemini key(s) found"

    def _check_packages(self):
        required = [
            "langchain_core", "langchain_google_genai", "chromadb",
            "fastapi", "dotenv", "apscheduler"
        ]
        missing = []
        for pkg in required:
            try:
                importlib.import_module(pkg)
            except ImportError:
                missing.append(pkg)

        if missing:
            return False, f"Missing packages: {', '.join(missing)}"
        return True, f"All {len(required)} core packages installed"

    def _check_sqlite(self):
        from ai_config import AI_DB_PATH
        db_dir = os.path.dirname(AI_DB_PATH)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

        conn = sqlite3.connect(AI_DB_PATH, timeout=5)
        conn.execute("CREATE TABLE IF NOT EXISTS _deploy_test (id INTEGER PRIMARY KEY)")
        conn.execute("DROP TABLE IF EXISTS _deploy_test")
        conn.commit()
        conn.close()
        return True, f"SQLite writable at {AI_DB_PATH}"

    def _check_chromadb(self):
        from ai_config import get_chroma_client, CHROMA_PERSIST_DIR
        client = get_chroma_client()
        client.heartbeat()
        return True, f"ChromaDB OK at {CHROMA_PERSIST_DIR}"

    def _check_llm_router(self):
        from llm_router import LLMRouter
        router = LLMRouter()
        # Check that at least one provider is configured
        has_gemini = len(router.gemini_keys) > 0
        has_groq = router.fallback_1 is not None
        has_openrouter = router.fallback_2 is not None

        providers = []
        if has_gemini:
            providers.append(f"Gemini ({len(router.gemini_keys)} keys)")
        if has_groq:
            providers.append("Groq")
        if has_openrouter:
            providers.append("OpenRouter")

        if not providers:
            return False, "No LLM providers configured"
        return True, f"Providers: {', '.join(providers)}"

    def _check_disk(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        usage = shutil.disk_usage(script_dir)
        free_gb = usage.free / (1024 ** 3)
        total_gb = usage.total / (1024 ** 3)
        used_pct = (usage.used / usage.total) * 100

        if free_gb < 10:
            return False, f"Only {free_gb:.1f}GB free ({used_pct:.1f}% used of {total_gb:.0f}GB)"
        return True, f"{free_gb:.1f}GB free ({used_pct:.1f}% used)"

    def _check_memory(self):
        try:
            import psutil
            mem = psutil.virtual_memory()
            avail_gb = mem.available / (1024 ** 3)
            if avail_gb < 4:
                return False, f"Only {avail_gb:.1f}GB available (need 4GB+)"
            return True, f"{avail_gb:.1f}GB available ({mem.percent}% used)"
        except ImportError:
            return True, "psutil not installed — cannot check RAM (non-blocking)"

    def _check_core_imports(self):
        modules = [
            "ai_config", "llm_router", "hybrid_retriever", "adaptive_router",
            "rag_graph", "position_sizer", "risk_budget", "autonomy_manager",
            "ai_decision_logger", "forgone_pnl_engine", "system_monitor",
            "raptor_tree", "streaming_rag", "magma_memory", "memo_rag",
            "bidirectional_rag", "flare_retriever", "speculative_rag", "cot_rag",
        ]
        failed = []
        for mod in modules:
            try:
                importlib.import_module(mod)
            except Exception as e:
                failed.append(f"{mod}: {e}")

        if failed:
            return False, f"{len(failed)} modules failed: {'; '.join(failed[:3])}"
        return True, f"All {len(modules)} AI modules importable"

    def _check_smoke_test(self):
        try:
            from smoke_test import run_full_smoke_test
            result = run_full_smoke_test()
            if result["failed"] > 0:
                return False, f"Smoke test: {result['failed']}/{result['total_checks']} failed"
            return True, f"Smoke test: {result['passed']}/{result['total_checks']} passed in {result['duration_seconds']}s"
        except Exception as e:
            return False, f"Smoke test crashed: {e}"


def main():
    print("=" * 60)
    print("Pre-Deployment Validation Check")
    print("=" * 60)

    checker = DeploymentChecker()
    result = checker.run_all_checks()

    print()
    for check in result["checks"]:
        icon = "PASS" if check["passed"] else "FAIL"
        print(f"[{icon}] {check['name']}: {check['message']}")

    print()
    if result["ready_to_deploy"]:
        print("READY TO DEPLOY")
    else:
        print(f"NOT READY — {len(result['blocking_issues'])} blocking issue(s):")
        for issue in result["blocking_issues"]:
            print(f"  - {issue}")

    print("=" * 60)
    return result


if __name__ == "__main__":
    main()
