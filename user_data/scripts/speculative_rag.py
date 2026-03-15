import logging
import re
from typing import List, Dict, Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from ai_config import AI_DB_PATH
from llm_router import LLMRouter

logger = logging.getLogger(__name__)

class SpeculativeRAG:
    """
    Phase 16 - Speculative RAG (Draft-Verify Pattern)
    Generates multiple drafts using subsets of context, and uses a verification
    step to mathematically determine the best-supported draft.
    """
    def __init__(self, llm_router: Optional[LLMRouter] = None, retriever=None):
        self.router = llm_router or LLMRouter(temperature=0.3)
        self.retriever = retriever

    def draft_and_verify(self, query: str, num_drafts: int = 3) -> dict:
        """
        Produce multiple distinct scenario drafts from separate subsets of data, 
        and verify the most accurate one across the global data pool.
        """
        logger.info(f"[SpeculativeRAG] Initiating generation for query: {query[:50]}...")
        
        evidence_used = 0
        all_drafts = []
        all_evidence = []
        
        # 1. Retrieve total evidence pool
        if self.retriever:
            try:
                results = self.retriever.search(query=query, top_k=num_drafts * 5)
                docs = [r['text'] if isinstance(r, dict) and 'text' in r else str(r) for r in results]
                evidence_used = len(docs)
                all_evidence = docs
            except Exception as e:
                logger.error(f"[SpeculativeRAG] Retrieval failed: {e}")
                docs = []
        else:
            docs = []
            
        # 2. Divide docs into subsets
        if not all_evidence:
            logger.warning("[SpeculativeRAG] No evidence found. Returning generic fallback.")
            fallback = self.router.invoke([HumanMessage(content=query)])
            return {
                "best_draft": str(fallback.content).strip(),
                "best_draft_index": 0,
                "verification_reasoning": "No evidence retrieved. Fallback basic response.",
                "all_drafts": [],
                "evidence_used": 0
            }

        chunk_size = max(1, len(all_evidence) // num_drafts)
        subsets = [all_evidence[i:i + chunk_size] for i in range(0, len(all_evidence), chunk_size)][:num_drafts]
        
        # 3. Generate individual drafts
        for i, subset in enumerate(subsets):
            draft = self._generate_draft(query, subset, index=i)
            all_drafts.append(draft)
            
        # 4. Verify the best draft
        if len(all_drafts) == 0:
            return {"best_draft": "Failed to generate any drafts.", "best_draft_index": -1, "verification_reasoning": "", "all_drafts": [], "evidence_used": evidence_used}
            
        if len(all_drafts) == 1:
            return {"best_draft": all_drafts[0], "best_draft_index": 0, "verification_reasoning": "Only one draft generated.", "all_drafts": all_drafts, "evidence_used": evidence_used}
            
        verification_result = self._verify_best_draft(query, all_drafts, all_evidence)
        
        return {
            "best_draft": verification_result["best_draft"],
            "best_draft_index": verification_result["best_draft_index"],
            "verification_reasoning": verification_result["reasoning"],
            "all_drafts": all_drafts,
            "evidence_used": evidence_used
        }

    def _generate_draft(self, query: str, evidence_subset: List[str], index: int) -> str:
        """Generates a single draft scenario from a subset of evidence."""
        context = "\\n".join(evidence_subset)
        prompt = (
            f"You are drafting scenario #{index+1} for a crypto trading analysis.\\n"
            f"Based STRICTLY on the evidence subset below, answer the query.\\n\\n"
            f"Evidence Subset:\\n{context}\\n\\n"
            f"Query: {query}\\n\\n"
            f"RULES:\\n"
            f"1. Use ONLY information present in the evidence above. Do NOT hallucinate or add external knowledge.\\n"
            f"2. Cite specific data points from the evidence (numbers, dates, events).\\n"
            f"3. If the evidence is insufficient, say 'Evidence insufficient for strong conclusion.'\\n"
            f"4. Provide a 2-3 sentence draft scenario."
        )
        try:
            response = self.router.invoke([
                SystemMessage(content="You generate evidence-grounded analytical drafts. ONLY cite information present in the provided evidence. NEVER fabricate data."),
                HumanMessage(content=prompt)
            ])
            draft = str(response.content).strip()
            return draft
        except Exception as e:
            logger.error(f"[SpeculativeRAG] Draft {index} generation failed: {e}")
            return f"Draft {index+1} failed to generate."

    def _verify_best_draft(self, query: str, drafts: List[str], all_evidence: List[str]) -> dict:
        """Selects the draft most heavily supported by the total combined evidence pool."""
        global_context = "\\n".join(all_evidence)
        drafts_formatted = "\\n\\n".join([f"--- Draft {i} ---\\n{d}" for i, d in enumerate(drafts)])
        
        prompt = (
            f"You are the Verification Judge. Evaluate {len(drafts)} drafts against the master evidence pool.\\n\\n"
            f"Query: {query}\\n\\n"
            f"Master Evidence Pool:\\n{global_context}\\n\\n"
            f"Generated Drafts:\\n{drafts_formatted}\\n\\n"
            f"EVALUATION CRITERIA (in order of importance):\\n"
            f"1. FACTUAL ALIGNMENT: Which draft's claims are best supported by the master evidence? Penalize fabricated data.\\n"
            f"2. COMPLETENESS: Which draft addresses the query most fully?\\n"
            f"3. SPECIFICITY: Which draft cites the most specific data points from the evidence?\\n\\n"
            f"OUTPUT FORMAT:\\n"
            f"Start EXACTLY with 'BEST_DRAFT_INDEX: [number]' (e.g., BEST_DRAFT_INDEX: 0).\\n"
            f"Follow with a 1-sentence explanation citing which specific evidence supports that draft.\\n"
        )
        try:
            response = self.router.invoke([
                SystemMessage(content="You are a strict verification judge. Evaluate ONLY factual alignment with evidence. The draft with the most evidence-backed claims wins."),
                HumanMessage(content=prompt)
            ])
            verification_text = str(response.content).strip()

            # Parse index with multi-tier extraction
            best_idx = 0
            reasoning = verification_text

            # Tier 1: Exact format "BEST_DRAFT_INDEX: N"
            if "BEST_DRAFT_INDEX:" in verification_text:
                parts = verification_text.split("BEST_DRAFT_INDEX:", 1)
                after_idx = parts[1].strip()
                idx_str = "".join([c for c in after_idx.split()[0] if c.isdigit()])
                if idx_str:
                    idx = int(idx_str)
                    if 0 <= idx < len(drafts):
                        best_idx = idx
                reasoning = parts[1].replace(idx_str, "", 1).strip()
            else:
                # Tier 2: Find "Draft N" or "draft N" pattern in text
                draft_match = re.search(r'[Dd]raft\s*#?\s*(\d+)', verification_text)
                if draft_match:
                    idx = int(draft_match.group(1))
                    if 0 <= idx < len(drafts):
                        best_idx = idx
                        logger.info(f"[SpeculativeRAG] Extracted draft index via 'Draft N' pattern: {best_idx}")
                else:
                    # Tier 3: Find any standalone digit that could be a draft index
                    digit_match = re.search(r'\b(\d)\b', verification_text)
                    if digit_match:
                        idx = int(digit_match.group(1))
                        if 0 <= idx < len(drafts):
                            best_idx = idx
                            logger.info(f"[SpeculativeRAG] Extracted draft index via digit pattern: {best_idx}")
                
            return {
                "best_draft": drafts[best_idx],
                "best_draft_index": best_idx,
                "reasoning": reasoning
            }
            
        except Exception as e:
            logger.error(f"[SpeculativeRAG] Verification failed: {e}")
            return {
                "best_draft": drafts[0],
                "best_draft_index": 0,
                "reasoning": "Verification failed. Defaulting to Draft 0."
            }
