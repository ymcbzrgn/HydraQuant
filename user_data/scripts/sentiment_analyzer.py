import sqlite3
import logging
import os
import sys
import re
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(__file__))
from db import get_db_connection

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
ONNX_DIR = os.path.join(MODELS_DIR, "onnx")
PT_DIR = os.path.join(MODELS_DIR, "pytorch")

# Graceful import: don't crash if transformers/torch not installed
_HAS_TRANSFORMERS = False
_USE_ONNX = False
try:
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer, pipeline as hf_pipeline
    _HAS_TRANSFORMERS = True
    _USE_ONNX = True
except ImportError:
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline as hf_pipeline
        import torch
        _HAS_TRANSFORMERS = True
    except ImportError:
        logger.warning("[Sentiment] transformers/torch not available. Will use LLM API fallback.")


def load_sentiment_pipeline(model_name):
    """Load a local sentiment model. Returns None if not available."""
    if not _HAS_TRANSFORMERS:
        return None

    onnx_path = os.path.join(ONNX_DIR, model_name)
    pt_path = os.path.join(PT_DIR, model_name)

    try:
        if _USE_ONNX and os.path.exists(onnx_path):
            model_path = onnx_path
            logger.info(f"Loading ONNX model for {model_name} from {model_path}")
            model = ORTModelForSequenceClassification.from_pretrained(model_path)
        elif os.path.exists(pt_path):
            model_path = pt_path
            logger.info(f"Loading PyTorch model for {model_name} from {model_path}")
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
        else:
            logger.warning(f"Model {model_name} not found locally at {onnx_path} or {pt_path}")
            return None

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        return hf_pipeline("text-classification", model=model, tokenizer=tokenizer, truncation=True, max_length=512)
    except Exception as e:
        logger.error(f"[Sentiment] Failed to load {model_name}: {e}")
        return None


def clean_text(text: str) -> str:
    """Emoji ve unicode karakterleri temizle"""
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _llm_sentiment_batch(articles: list) -> list:
    """
    Fallback: Use LLM Router to score sentiment when local models are unavailable.
    Processes articles in batches for cost efficiency.
    """
    try:
        from llm_router import LLMRouter
        from langchain_core.messages import SystemMessage, HumanMessage
        import json

        from json_utils import extract_json_array
        router = LLMRouter(temperature=0.0, request_timeout=30)

        SYSTEM = """IDENTITY: You are a crypto market sentiment classifier with expertise in financial NLP.

TASK: Score each news headline/summary from -1.0 (very bearish) to +1.0 (very bullish). 0.0 = neutral.

CALIBRATION GUIDE:
-1.0: Catastrophic (exchange hack, regulatory ban, protocol exploit)
-0.7: Strongly bearish (major sell-off, negative regulation, large hack)
-0.4: Moderately bearish (disappointing earnings, minor FUD, whale selling)
-0.1: Slightly bearish (mild concern, uncertainty)
 0.0: Neutral (factual report, no directional implication)
+0.1: Slightly bullish (mild positive, routine development)
+0.4: Moderately bullish (partnership, adoption news, positive regulation)
+0.7: Strongly bullish (ETF approval, major institutional adoption)
+1.0: Euphoric (paradigm shift, unprecedented positive catalyst)

RULES:
1. Score EACH article independently. Do not let one article's sentiment influence another.
2. Consider CRYPTO-SPECIFIC context: "SEC delays ETF" is bearish for crypto even if neutral in traditional finance.
3. Account for SARCASM and CLICKBAIT: "Bitcoin is DEAD (again)" is likely neutral/ironic, not truly bearish.
4. When uncertain, bias toward 0.0 (neutral) rather than guessing extreme scores.
5. Output ONLY a JSON array of numbers, one per article, same order. Example: [-0.7, 0.3, 0.0, -0.1]
No markdown, no backticks, ONLY raw JSON array."""

        results = []
        batch_size = 20
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i + batch_size]
            texts = []
            for a in batch:
                text = a['summary'] if a['summary'] and len(str(a['summary'])) > 20 else a['title']
                texts.append(clean_text(str(text or "")))

            numbered = "\n".join(f"{j+1}. {t[:300]}" for j, t in enumerate(texts))
            messages = [
                SystemMessage(content=SYSTEM),
                HumanMessage(content=f"Score these {len(texts)} articles:\n{numbered}")
            ]

            response = router.invoke(messages)
            content = str(response.content).strip()

            scores = extract_json_array(content)
            if isinstance(scores, list):
                for j, a in enumerate(batch):
                    score = float(scores[j]) if j < len(scores) else 0.0
                    score = max(-1.0, min(1.0, score))
                    results.append((score, a['id']))
            else:
                logger.warning(f"[Sentiment LLM] Unexpected response format: {content[:100]}")

        return results
    except Exception as e:
        logger.error(f"[Sentiment LLM Fallback] Failed: {e}")
        return []


def analyze_unscored_news():
    """Fetches unscored news from the DB, analyzes sentiment, and saves back."""
    conn = get_db_connection()
    c = conn.cursor()

    c.execute("SELECT id, summary, title, source FROM market_news WHERE sentiment_score IS NULL LIMIT 200")
    articles = c.fetchall()

    if not articles:
        logger.info("No new articles to score.")
        conn.close()
        return

    logger.info(f"Found {len(articles)} unscored articles. Processing...")

    # Strategy: LLM Primary → Local Model Fallback
    # LLMs are smarter (nuance, sarcasm, coin-specific), batch-efficient, and already free via 7 providers.
    # Local models (CryptoBERT/FinBERT) are insurance for when ALL APIs are down.

    updates = []

    # 1. Try LLM API first (smarter, coin-aware, batch-efficient)
    logger.info("[Sentiment] Trying LLM API (primary)...")
    articles_list = [dict(a) for a in articles]
    updates = _llm_sentiment_batch(articles_list)

    if updates:
        logger.info(f"[Sentiment] LLM scored {len(updates)}/{len(articles)} articles successfully.")
    else:
        # 2. LLM failed → fall back to local models
        logger.warning("[Sentiment] LLM API failed. Falling back to local models...")
        cryptobert = load_sentiment_pipeline("cryptobert")
        finbert = load_sentiment_pipeline("finbert")

        if cryptobert is None and finbert is None:
            logger.error("[Sentiment] Both LLM API and local models unavailable. No scoring possible.")
        else:
            for article in articles:
                text_to_analyze = article['summary'] if article['summary'] and len(article['summary']) > 20 else article['title']
                if not text_to_analyze:
                    continue

                text_to_analyze = clean_text(text_to_analyze)

                try:
                    source = article['source'].lower()
                    if ('yahoo' in source or 'alpha' in source) and finbert:
                        result = finbert(text_to_analyze)[0]
                        label_mapping = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
                    elif cryptobert:
                        result = cryptobert(text_to_analyze)[0]
                        label_mapping = {"Bullish": 1.0, "Neutral": 0.0, "Bearish": -1.0}
                    else:
                        continue

                    label = result['label']
                    confidence = result['score']

                    base_score = 0.0
                    for key, val in label_mapping.items():
                        if key.lower() in label.lower():
                            base_score = val
                            break

                    final_score = base_score * confidence
                    updates.append((final_score, article['id']))

                except Exception as e:
                    logger.error(f"Error scoring article {article['id']}: {e}")

    if updates:
        c.executemany("UPDATE market_news SET sentiment_score = ? WHERE id = ?", updates)
        conn.commit()
        logger.info(f"Successfully scored {len(updates)} articles.")

    conn.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    analyze_unscored_news()
