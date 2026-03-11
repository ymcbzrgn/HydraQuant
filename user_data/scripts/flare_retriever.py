import logging
import re
from typing import List, Dict, Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from ai_config import AI_DB_PATH
from llm_router import LLMRouter

logger = logging.getLogger(__name__)

class FLARERetriever:
    """
    Phase 16 - FLARE (Forward-Looking Active Retrieval)
    Detects uncertainty during generation and triggers active retrieval when needed.
    """
    def __init__(self, llm_router: Optional[LLMRouter] = None, retriever=None):
        self.router = llm_router or LLMRouter()
        self.retriever = retriever
        
    def generate_with_active_retrieval(self, query: str, context: str = "") -> dict:
        """
        FLARE pattern: Identify uncertain sentences during generation and perform additional retrieval.
        """
        logger.info(f"[FLARE] Starting generation for query: {query[:50]}...")
        
        # 1. Initial generation
        prompt = f"Context: {context}\n\nQuery: {query}\n\nGenerate an analytical response:"
        messages = [
            SystemMessage(content="You are a financial analyst. Generate a detailed response."),
            HumanMessage(content=prompt)
        ]
        
        try:
            initial_response = self.router.invoke(messages)
            initial_text = str(initial_response.content).strip()
        except Exception as e:
            logger.error(f"[FLARE] Initial generation failed: {e}")
            return {"analysis": "Error generating analysis.", "retrievals_triggered": 0, "low_confidence_sentences": [], "sources_used": []}
            
        # 2. Split into sentences
        sentences = self._split_into_sentences(initial_text)
        
        final_sentences = []
        retrievals_triggered = 0
        low_confidence_sentences = []
        sources_used = []
        
        # 3. Assess confidence per sentence
        for sentence in sentences:
            conf = self._assess_confidence(sentence)
            
            if conf < 0.5:
                logger.info(f"[FLARE] Low confidence ({conf}) detected for sentence: {sentence[:30]}...")
                low_confidence_sentences.append(sentence)
                retrievals_triggered += 1
                
                # Retrieve extra context
                extra_context_docs = self._retrieve_for_sentence(sentence)
                extra_context_text = "\n".join(extra_context_docs)
                sources_used.extend(extra_context_docs)
                
                # Regenerate sentence with extra context
                regen_prompt = (
                    f"Original query: {query}\n"
                    f"Additional Context:\n{extra_context_text}\n\n"
                    f"Please rewrite and improve the following sentence using the additional context to make it accurate and confident. "
                    f"Only return the rewritten sentence.\n\nSentence to rewrite: {sentence}"
                )
                try:
                    regen_response = self.router.invoke([HumanMessage(content=regen_prompt)])
                    regenerated_sentence = str(regen_response.content).strip()
                    final_sentences.append(regenerated_sentence)
                except Exception as e:
                    logger.error(f"[FLARE] Failed to regenerate sentence: {e}")
                    final_sentences.append(sentence)
            else:
                final_sentences.append(sentence)
                
        final_analysis = " ".join(final_sentences)
        
        return {
            "analysis": final_analysis,
            "retrievals_triggered": retrievals_triggered,
            "low_confidence_sentences": low_confidence_sentences,
            "sources_used": sources_used
        }

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using simple regex."""
        # Split on standard punctuation followed by whitespace, keeping the punctuation
        parts = re.split(r'(?<=[.!?])\s+', text)
        return [p.strip() for p in parts if p.strip()]

    def _assess_confidence(self, sentence: str) -> float:
        """Bir cümlenin güven skorunu LLM ile değerlendir (0-1)."""
        prompt = (
            "You are evaluating the confidence of a financial claim. "
            "Rate how confident/factual the following sentence is from 0.0 to 1.0. "
            "0.0 means highly uncertain or speculative, 1.0 means highly confident or factual. "
            "Return ONLY a float score.\n\n"
            f"Sentence: {sentence}"
        )
        
        try:
            response = self.router.invoke([HumanMessage(content=prompt)])
            score_text = str(response.content).strip()
            # Extract float
            match = re.search(r'\d+\.\d+|\d+', score_text)
            if match:
                return float(match.group())
            return 1.0
        except Exception as e:
            logger.error(f"[FLARE] Confidence assessment failed: {e}")
            return 1.0

    def _retrieve_for_sentence(self, sentence: str) -> List[str]:
        """Retrieve additional context for a low-confidence sentence."""
        if not self.retriever:
            logger.warning("[FLARE] No retriever provided. Cannot perform active retrieval.")
            return []
            
        try:
            # Assuming HybridRetriever.search exists and returns a list of dicts with 'text'
            results = self.retriever.search(query=sentence, top_k=2)
            if results and isinstance(results[0], dict) and 'text' in results[0]:
                return [r['text'] for r in results]
            elif results and isinstance(results[0], str):
                 return results
            return []
        except Exception as e:
            logger.error(f"[FLARE] Retrieval failed for sentence: {e}")
            return []
