import torch
import logging
from typing import List, Dict
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

class ColBERTReranker:
    """ColBERTv2 Late Interaction Reranker. Evaluates fine-grained token-level match scores."""
    def __init__(self, model_name="jinaai/jina-colbert-v2"):
        logger.info(f"Loading ColBERT locally: {model_name} (CPU mode)")
        # trust_remote_code=True is required for jina-colbert
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval()

    def _get_embeddings(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Embedding outputs for query and document tokens
            return outputs.last_hidden_state[0]

    def _max_sim_score(self, query_embs: torch.Tensor, doc_embs: torch.Tensor) -> float:
        """
        Compute MaxSim between query tokens and document tokens.
        """
        # Normalize to allow dot product as cosine similarity
        q_norm = torch.nn.functional.normalize(query_embs, p=2, dim=1)
        d_norm = torch.nn.functional.normalize(doc_embs, p=2, dim=1)
        
        # Sim matrix shape: (query_len, doc_len)
        sim_matrix = torch.matmul(q_norm, d_norm.T)
        
        # Max over doc tokens: shape (query_len,)
        max_sims = torch.max(sim_matrix, dim=1).values
        
        # Sum of max similarities across query tokens
        return float(torch.sum(max_sims))

    def rerank(self, query: str, documents: List[Dict], top_k: int = 5) -> List[Dict]:
        """Token-level late interaction scoring."""
        if not documents:
            return []
            
        try:
            query_embs = self._get_embeddings(query)
        except Exception as e:
            logger.error(f"Failed to encode query in ColBERT: {e}")
            return documents[:top_k]
            
        scored_docs = []
        for doc in documents:
            text = doc.get("content", doc.get("text", ""))
            if not text:
                scored_docs.append(doc)
                continue
                
            try:
                doc_embs = self._get_embeddings(text)
                score = self._max_sim_score(query_embs, doc_embs)
                
                scored_doc = doc.copy()
                scored_doc["colbert_score"] = score
                scored_docs.append(scored_doc)
            except Exception as e:
                logger.error(f"Failed to score doc in ColBERT: {e}")
                scored_docs.append(doc)
                
        # Filter and normalize scores
        scored_docs.sort(key=lambda x: x.get("colbert_score", 0.0), reverse=True)
        top_docs = scored_docs[:top_k]
        
        # Normalize colbert scores between 0 and 1 for ensemble purposes
        if top_docs:
            max_score = max(doc.get("colbert_score", 0.0) for doc in top_docs)
            min_score = min(doc.get("colbert_score", 0.0) for doc in top_docs)
            range_score = max_score - min_score if max_score > min_score else 1.0
            
            for doc in top_docs:
                if "colbert_score" in doc:
                    doc["colbert_normalized"] = (doc["colbert_score"] - min_score) / range_score
                else:
                    doc["colbert_normalized"] = 0.0
                    
        return top_docs
