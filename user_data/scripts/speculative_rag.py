import logging
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
            f"You are a scenario analyst drafting possibility #{index+1}.\\n"
            f"Based strictly on the following subset of evidence, answer the query.\\n\\n"
            f"Evidence Subset:\\n{context}\\n\\n"
            f"Query: {query}\\n\\n"
            f"Provide a 2-3 sentence draft scenario based ONLY on the evidence above."
        )
        try:
            response = self.router.invoke([
                SystemMessage(content="You generate specific analytical drafts."),
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
            f"You are the Verification Overlord. You are presented with a query, a master pool of evidence, and {len(drafts)} distinct drafts.\\n"
            f"Query: {query}\\n\\n"
            f"Master Evidence Pool:\\n{global_context}\\n\\n"
            f"Generated Drafts:\\n{drafts_formatted}\\n\\n"
            f"TASK:\\n"
            f"1. Evaluate which Draft is most factually aligned with the combined Master Evidence Pool.\\n"
            f"2. Return your response starting EXACTLY with 'BEST_DRAFT_INDEX: [number]' (e.g. BEST_DRAFT_INDEX: 0).\\n"
            f"3. Follow it with a 1-sentence verification reasoning why it is best.\\n"
        )
        try:
            response = self.router.invoke([
                SystemMessage(content="You are a strict verifier AI."),
                HumanMessage(content=prompt)
            ])
            verification_text = str(response.content).strip()
            
            # Parse index
            best_idx = 0
            reasoning = verification_text
            
            if "BEST_DRAFT_INDEX:" in verification_text:
                parts = verification_text.split("BEST_DRAFT_INDEX:", 1)
                after_idx = parts[1].strip()
                idx_str = "".join([c for c in after_idx.split()[0] if c.isdigit()])
                if idx_str:
                    idx = int(idx_str)
                    if 0 <= idx < len(drafts):
                        best_idx = idx
                reasoning = parts[1].replace(idx_str, "", 1).strip()
                
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
