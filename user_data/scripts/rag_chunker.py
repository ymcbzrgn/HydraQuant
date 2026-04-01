import uuid
import logging
from typing import List, Dict

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    # Fallback to simple chunking if langchain isn't available
    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size=512, chunk_overlap=76, separators=None):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
            
        def split_text(self, text: str) -> List[str]:
            # Extremely naive fallback
            chunks = []
            start = 0
            while start < len(text):
                chunks.append(text[start:start+self.chunk_size])
                start += (self.chunk_size - self.chunk_overlap)
            return chunks

logger = logging.getLogger(__name__)

class ContentChunker:
    """
    Advanced Chunking Pipeline for the Hybrid RAG engine.
    Supports Recursive, Parent-Child, and Contextual chunking strategies.
    """
    
    @staticmethod
    def chunk_recursive(text: str, chunk_size: int = 512, chunk_overlap: int = 76) -> List[str]:
        """
        Standard chunking strategy for general documents and short news.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        return splitter.split_text(text)

    @staticmethod
    def chunk_parent_child(text: str, parent_size: int = 512, child_size: int = 128) -> List[Dict[str, str]]:
        """
        Parent-Child chunking for precision retrieval but broad LLM context.
        Splits text into large parents, then splits each parent into multiple smaller children.
        """
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_size,
            chunk_overlap=int(parent_size * 0.15)
        )
        
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_size,
            chunk_overlap=int(child_size * 0.15)
        )
        
        parents = parent_splitter.split_text(text)
        result = []
        
        for parent_text in parents:
            parent_id = str(uuid.uuid4())
            children = child_splitter.split_text(parent_text)
            
            for child_text in children:
                # We save both the child and its reference to the parent.
                # During retrieval, ChromaDB will match the `child_text`, 
                # but we will feed the `parent_text` to the LLM.
                result.append({
                    "child_text": child_text,
                    "parent_text": parent_text,
                    "parent_id": parent_id
                })
                
        return result

    @staticmethod
    def construct_contextual_prompt(chunk: str, document_summary) -> str:
        """
        Contextual Chunking (Anthropic-style, enhanced).
        Accepts either a string summary or a full article dict for richer context.
        Anthropic reports 49% retrieval failure reduction with contextual prefixes.
        """
        if isinstance(document_summary, dict):
            # Rich context from article metadata
            article = document_summary
            source = article.get('source', 'Unknown')
            date = str(article.get('published_at', ''))[:10]
            sentiment = article.get('sentiment_score', 0.5)
            if sentiment is None:
                sentiment = 0.5
            sent_label = 'Positive' if sentiment > 0.6 else 'Negative' if sentiment < 0.4 else 'Neutral'
            title = str(article.get('title', ''))[:120]
            return f"[{source} | {date} | {sent_label}] {title}\n\n{chunk}"
        else:
            # Legacy string summary (backward compatible)
            return f"Document context: {document_summary}\n\nExcerpt: {chunk}"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample_text = (
        "Bitcoin represents a major shift in the financial landscape. " * 10 + 
        "It was created by Satoshi Nakamoto to be a decentralized currency. " * 10
    )
    
    logger.info("Testing Recursive Chunking...")
    recursive_chunks = ContentChunker.chunk_recursive(sample_text, chunk_size=200, chunk_overlap=20)
    logger.info(f"Created {len(recursive_chunks)} recursive chunks.")
    
    logger.info("Testing Parent-Child Chunking...")
    pc_chunks = ContentChunker.chunk_parent_child(sample_text, parent_size=300, child_size=100)
    logger.info(f"Created {len(pc_chunks)} parent-child pairs.")
    for i, pair in enumerate(pc_chunks[:2]):
        logger.info(f"  Child {i+1} length: {len(pair['child_text'])}")
        logger.info(f"  Parent {i+1} length: {len(pair['parent_text'])}")
