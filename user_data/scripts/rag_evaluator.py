"""
Phase 6.2: RAGAS-Inspired RAG Quality Evaluation
Measures RAG pipeline quality with 3 key metrics:
  1. Faithfulness: Generated answer is supported by retrieved context
  2. Context Precision: Retrieved docs are relevant to the query
  3. Answer Relevancy: Generated answer addresses the query

Targets: Faithfulness > 0.90, Context Precision > 0.85, Answer Relevancy > 0.90
"""

import os
import re
import sys
import sqlite3
import logging
import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

sys.path.append(os.path.dirname(__file__))

from llm_router import LLMRouter
from langchain_core.messages import SystemMessage, HumanMessage
from json_utils import extract_json_strict

logger = logging.getLogger(__name__)

from ai_config import AI_DB_PATH as DB_PATH

# ── Evaluation Prompts ──────────────────────────────────────────────

FAITHFULNESS_PROMPT = """Evaluate whether this AI answer is FAITHFULLY supported by the retrieved context.

Context Documents:
{context}

AI Answer:
{answer}

SCORING (be STRICT — hallucinated claims must be penalized heavily):
- 1.0: Every factual claim in the answer traces to a specific context passage
- 0.8: Almost all claims supported, 1 minor unsourced but plausible claim
- 0.5: Mixed — some claims supported, others clearly hallucinated or fabricated
- 0.3: Mostly hallucinated — answer invents facts not in context
- 0.0: Answer contradicts or completely ignores context

RED FLAGS (auto-reduce score below 0.4):
- Fabricated numbers (indicator values, prices, percentages not in context)
- Invented news events not mentioned in any document
- Claims that directly CONTRADICT the context documents

Output ONLY a JSON: {{"score": 0.XX, "reason": "brief explanation citing specific supported/unsupported claims"}}"""

CONTEXT_PRECISION_PROMPT = """Evaluate whether the retrieved documents are relevant to the query.

Query: {query}

Retrieved Documents:
{context}

SCORING:
- 1.0: ALL documents directly address the query with specific, actionable information
- 0.8: Most documents relevant, 1-2 tangential
- 0.5: About half relevant, half off-topic or outdated
- 0.3: Mostly irrelevant, only vague topical connection
- 0.0: None relate to the query

CONSIDERATIONS:
- Freshness matters: old documents for "current" queries reduce score
- Specificity matters: generic crypto articles for a specific pair query reduce score
- Redundancy: 5 documents saying the same thing = low precision (1 useful, 4 redundant)

Output ONLY a JSON: {{"score": 0.XX, "reason": "brief explanation noting relevant vs irrelevant doc count"}}"""

ANSWER_RELEVANCY_PROMPT = """Evaluate whether this AI answer addresses the original query.

Query: {query}

AI Answer:
{answer}

SCORING:
- 1.0: Directly and completely answers what was asked, with actionable detail
- 0.8: Answers the query well but misses one minor aspect
- 0.5: Partially addresses the query but includes significant tangents or misses key aspects
- 0.3: Vaguely related to the topic but doesn't answer the specific question
- 0.0: Completely irrelevant, generic, or a non-answer ("I cannot determine...")

PENALTY: Generic hedging ("it depends on many factors") without specific analysis → cap at 0.4

Output ONLY a JSON: {{"score": 0.XX, "reason": "brief explanation of what was addressed vs missed"}}"""


# ── Test Queries ────────────────────────────────────────────────────

DEFAULT_TEST_QUERIES = [
    "What is the current Bitcoin price trend?",
    "How will the Federal Reserve interest rate decision affect crypto?",
    "What is the Fear and Greed Index for crypto today?",
    "BTC/ETH correlation analysis with macro factors",
    "Latest Ethereum DeFi protocol developments",
    "Bitcoin on-chain metrics whale accumulation",
    "Crypto market sentiment analysis from recent news",
    "What are the key support and resistance levels for BTC?",
]


class RAGQualityEvaluator:
    """
    Measures RAG pipeline quality using LLM-as-judge approach
    inspired by RAGAS framework.
    """

    def __init__(self, router: Optional[LLMRouter] = None, db_path: str = DB_PATH):
        self.router = router or LLMRouter(temperature=0.1, request_timeout=30)
        self.db_path = db_path
        self._ensure_table()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _ensure_table(self):
        with self._get_conn() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS rag_quality_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    query TEXT NOT NULL,
                    faithfulness REAL,
                    context_precision REAL,
                    answer_relevancy REAL,
                    avg_score REAL,
                    details TEXT
                )
            ''')
            conn.commit()

    def _llm_judge(self, prompt: str) -> Dict[str, Any]:
        """Use LLM to score a specific metric."""
        messages = [
            SystemMessage(content="You are a strict RAG quality evaluator. Your ENTIRE response must be a single valid JSON object with 'score' and 'reason' keys. No text before or after."),
            HumanMessage(content=prompt)
        ]
        try:
            response = self.router.invoke(messages, priority="low")
            text = str(response.content).strip()
            if not text:
                logger.warning("[RAGAS] Empty LLM response. Returning default score.")
                return {"score": 0.5, "reason": "Empty LLM response"}
            result = extract_json_strict(text, required_keys=["score"])
            if result is None:
                logger.warning(f"[RAGAS] JSON extraction failed. Raw: {text[:200]}")
                return {"score": 0.5, "reason": "JSON extraction failed"}
            return result
        except Exception as e:
            logger.warning(f"[RAGAS] LLM judge error: {e}")
            return {"score": 0.5, "reason": f"Error: {str(e)[:100]}"}

    def evaluate_single(
        self,
        query: str,
        context_docs: List[str],
        answer: str
    ) -> Dict[str, float]:
        """
        Evaluate a single query-context-answer triple.

        Returns:
            Dict with faithfulness, context_precision, answer_relevancy, avg_score
        """
        context_text = "\n---\n".join(context_docs[:5])  # Max 5 docs for eval

        # 1. Faithfulness
        faith_result = self._llm_judge(
            FAITHFULNESS_PROMPT.format(context=context_text, answer=answer)
        )
        faithfulness = float(faith_result.get("score", 0.5))

        # 2. Context Precision
        cp_result = self._llm_judge(
            CONTEXT_PRECISION_PROMPT.format(query=query, context=context_text)
        )
        context_precision = float(cp_result.get("score", 0.5))

        # 3. Answer Relevancy
        ar_result = self._llm_judge(
            ANSWER_RELEVANCY_PROMPT.format(query=query, answer=answer)
        )
        answer_relevancy = float(ar_result.get("score", 0.5))

        avg_score = (faithfulness + context_precision + answer_relevancy) / 3.0

        return {
            "faithfulness": round(faithfulness, 3),
            "context_precision": round(context_precision, 3),
            "answer_relevancy": round(answer_relevancy, 3),
            "avg_score": round(avg_score, 3),
            "details": {
                "faith_reason": faith_result.get("reason", ""),
                "cp_reason": cp_result.get("reason", ""),
                "ar_reason": ar_result.get("reason", ""),
            }
        }

    def evaluate_pipeline(
        self,
        test_queries: Optional[List[str]] = None,
        retriever=None,
        rag_graph_fn=None
    ) -> Dict[str, Any]:
        """
        Run full evaluation over test queries.

        Args:
            test_queries: List of queries to test (default: DEFAULT_TEST_QUERIES)
            retriever: HybridRetriever instance for retrieval
            rag_graph_fn: Function that takes (query, retriever) → answer text

        Returns:
            Aggregate metrics dict
        """
        queries = test_queries or DEFAULT_TEST_QUERIES
        all_metrics = []

        for query in queries:
            try:
                # Retrieve docs
                if retriever:
                    results = retriever.search(query, top_k=5)
                    context_docs = [r.get("text", "") for r in results]
                else:
                    context_docs = ["[No retriever configured]"]

                # Generate answer
                if rag_graph_fn:
                    answer = rag_graph_fn(query, retriever)
                else:
                    answer = f"[Mock answer for: {query}]"

                # Evaluate
                metrics = self.evaluate_single(query, context_docs, answer)
                metrics["query"] = query
                all_metrics.append(metrics)

                # Log to DB
                self.log_metrics(query, metrics)

                logger.info(
                    f"[RAGAS] Query: '{query[:40]}...' → "
                    f"F={metrics['faithfulness']:.2f} "
                    f"CP={metrics['context_precision']:.2f} "
                    f"AR={metrics['answer_relevancy']:.2f} "
                    f"AVG={metrics['avg_score']:.2f}"
                )

            except Exception as e:
                logger.error(f"[RAGAS] Eval failed for '{query[:40]}': {e}")

        # Aggregate
        if all_metrics:
            avg_faith = sum(m["faithfulness"] for m in all_metrics) / len(all_metrics)
            avg_cp = sum(m["context_precision"] for m in all_metrics) / len(all_metrics)
            avg_ar = sum(m["answer_relevancy"] for m in all_metrics) / len(all_metrics)
            avg_total = (avg_faith + avg_cp + avg_ar) / 3.0
        else:
            avg_faith = avg_cp = avg_ar = avg_total = 0.0

        return {
            "total_queries": len(queries),
            "evaluated": len(all_metrics),
            "avg_faithfulness": round(avg_faith, 3),
            "avg_context_precision": round(avg_cp, 3),
            "avg_answer_relevancy": round(avg_ar, 3),
            "avg_total": round(avg_total, 3),
            "targets_met": {
                "faithfulness": avg_faith >= 0.90,
                "context_precision": avg_cp >= 0.85,
                "answer_relevancy": avg_ar >= 0.90,
            },
            "per_query": all_metrics
        }

    def log_metrics(self, query: str, metrics: Dict):
        """Persist evaluation metrics to SQLite."""
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO rag_quality_metrics 
                   (timestamp, query, faithfulness, context_precision, answer_relevancy, avg_score, details)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now(tz=timezone.utc).isoformat(),
                    query,
                    metrics.get("faithfulness", 0),
                    metrics.get("context_precision", 0),
                    metrics.get("answer_relevancy", 0),
                    metrics.get("avg_score", 0),
                    json.dumps(metrics.get("details", {}))
                )
            )
            conn.commit()

    def weekly_report(self) -> str:
        """Generate a weekly RAG quality summary."""
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT AVG(faithfulness) as f, AVG(context_precision) as cp, 
                          AVG(answer_relevancy) as ar, AVG(avg_score) as avg,
                          COUNT(*) as cnt
                   FROM rag_quality_metrics 
                   WHERE timestamp > datetime('now', '-7 days')"""
            ).fetchone()

        if not rows or rows['cnt'] == 0:
            return "No RAG evaluations in the last 7 days."

        report = f"""Weekly RAG Quality Report
Evaluations: {rows['cnt']}
Faithfulness:       {rows['f']:.3f} {'OK' if rows['f'] >= 0.90 else 'LOW'} (target: >=0.90)
Context Precision:  {rows['cp']:.3f} {'OK' if rows['cp'] >= 0.85 else 'LOW'} (target: >=0.85)
Answer Relevancy:   {rows['ar']:.3f} {'OK' if rows['ar'] >= 0.90 else 'LOW'} (target: >=0.90)
Overall Average:    {rows['avg']:.3f}"""
        return report

    def get_weekly_quality_report(self) -> dict:
        """Structured weekly quality report for scheduler feedback loop."""
        try:
            conn = sqlite3.connect(self.db_path, timeout=30)
            conn.row_factory = sqlite3.Row

            this_week = conn.execute("""
                SELECT AVG(faithfulness) as f, AVG(context_precision) as cp,
                       AVG(answer_relevancy) as ar, COUNT(*) as n
                FROM rag_quality_metrics
                WHERE timestamp > datetime('now', '-7 days')
            """).fetchone()

            last_week = conn.execute("""
                SELECT AVG(faithfulness) as f
                FROM rag_quality_metrics
                WHERE timestamp BETWEEN datetime('now', '-14 days') AND datetime('now', '-7 days')
            """).fetchone()

            conn.close()

            if not this_week or this_week["n"] == 0:
                return None

            trend = "stable"
            if last_week and last_week["f"]:
                diff = (this_week["f"] or 0) - (last_week["f"] or 0)
                trend = "improving" if diff > 0.05 else ("declining" if diff < -0.05 else "stable")

            return {
                "avg_faithfulness": this_week["f"] or 0,
                "avg_context_precision": this_week["cp"] or 0,
                "avg_answer_relevancy": this_week["ar"] or 0,
                "sample_count": this_week["n"],
                "trend": trend,
            }
        except Exception:
            return None
