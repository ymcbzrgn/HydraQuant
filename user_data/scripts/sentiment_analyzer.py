import sqlite3
import logging
import os
import sys
import pandas as pd
import re
from datetime import datetime, timedelta

# Auto-detect ONNX vs PyTorch
try:
    from optimum.onnxruntime import ORTModelForSequenceClassification
    USE_ONNX = True
except ImportError:
    from transformers import AutoModelForSequenceClassification
    USE_ONNX = False

from transformers import AutoTokenizer, pipeline
import torch

sys.path.append(os.path.dirname(__file__))
from db import get_db_connection

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
ONNX_DIR = os.path.join(MODELS_DIR, "onnx")
PT_DIR = os.path.join(MODELS_DIR, "pytorch")

# Load models from local storage
def load_sentiment_pipeline(model_name):
    # Determine the correct local path (ONNX or PyTorch)
    onnx_path = os.path.join(ONNX_DIR, model_name)
    pt_path = os.path.join(PT_DIR, model_name)
    
    if USE_ONNX and os.path.exists(onnx_path):
        model_path = onnx_path
        logger.info(f"Loading ONNX model for {model_name} from {model_path}")
        model = ORTModelForSequenceClassification.from_pretrained(model_path)
    elif os.path.exists(pt_path):
        model_path = pt_path
        logger.info(f"Loading PyTorch model for {model_name} from {model_path}")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    else:
        logger.error(f"Model {model_name} not found locally! Run download_models.py first.")
        sys.exit(1)
        
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return pipeline("text-classification", model=model, tokenizer=tokenizer, truncation=True, max_length=512)

# Phase 7: Clean Text for Model Input
def clean_text(text: str) -> str:
    """Emoji ve unicode karakterleri temizle"""
    # Remove Non-ASCII
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def analyze_unscored_news():
    """Fetches unscored news from the DB, analyzes sentiment, and saves back."""
    logger.info("Loading Sentiment Models (CryptoBERT and FinBERT)...")
    
    # We use CryptoBERT for general crypto news
    cryptobert = load_sentiment_pipeline("cryptobert")
    # We use FinBERT for broader macroeconomic news / F&G context if needed
    finbert = load_sentiment_pipeline("finbert")
    
    conn = get_db_connection()
    c = conn.cursor()
    
    # Get articles that haven't been scored yet (sentiment_score is NULL)
    c.execute("SELECT id, summary, title, source FROM market_news WHERE sentiment_score IS NULL LIMIT 200")
    articles = c.fetchall()
    
    if not articles:
        logger.info("No new articles to score.")
        return
        
    logger.info(f"Found {len(articles)} unscored articles. Processing...")
    
    updates = []
    for article in articles:
        # Prefer summary, fallback to title
        text_to_analyze = article['summary'] if article['summary'] and len(article['summary']) > 20 else article['title']
        if not text_to_analyze:
            continue
            
        # Phase 7: Clean text before model inference
        text_to_analyze = clean_text(text_to_analyze)
            
        try:
            # Choose model based on source
            source = article['source'].lower()
            if 'yahoo' in source or 'alpha' in source:
                result = finbert(text_to_analyze)[0]
                label_mapping = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
            else:
                result = cryptobert(text_to_analyze)[0]
                # CryptoBERT labels: Bearish, Neutral, Bullish
                label_mapping = {"Bullish": 1.0, "Neutral": 0.0, "Bearish": -1.0}
                
            label = result['label']
            confidence = result['score']
            
            # Map robustly in case model outputs differ slightly
            base_score = 0.0
            for key, val in label_mapping.items():
                if key.lower() in label.lower():
                    base_score = val
                    break
                    
            # Final score is direction * confidence
            final_score = base_score * confidence
            
            updates.append((final_score, article['id']))
            logger.debug(f"Scored {article['id']}: {label} ({confidence:.2f}) -> {final_score:.2f}")
            
        except Exception as e:
            logger.error(f"Error scoring article {article['id']}: {e}")
            
    # Batch update the database
    if updates:
        c.executemany("UPDATE market_news SET sentiment_score = ? WHERE id = ?", updates)
        conn.commit()
        logger.info(f"Successfully scored {len(updates)} articles.")
        
    conn.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    analyze_unscored_news()
