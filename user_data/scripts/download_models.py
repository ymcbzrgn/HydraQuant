import os
import sys
import logging
import platform

logger = logging.getLogger(__name__)

# Base path for models to make it easy for user to clear
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
ONNX_DIR = os.path.join(MODELS_DIR, "onnx")

# Models to download
MODELS = {
    "cryptobert": "ElKulako/cryptobert",
    "finbert": "ProsusAI/finbert"
}

def download_models():
    """Downloads models and converts to ONNX if environment supports it."""
    os.makedirs(ONNX_DIR, exist_ok=True)
    
    # Simple heuristic: try ONNX if we have the optimum library
    use_onnx = False
    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer
        use_onnx = True
        logger.info("Optimum ONNX Runtime found. Downloading & converting models to ONNX...")
    except ImportError:
        logger.warning("Optimum not found. Falling back to standard PyTorch format.")
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

    for name, repo_id in MODELS.items():
        base_path = os.path.join(ONNX_DIR, name) if use_onnx else os.path.join(MODELS_DIR, "pytorch", name)
        
        if os.path.exists(base_path) and os.listdir(base_path):
            logger.info(f"Model {name} already exists at {base_path}. Skipping.")
            continue
            
        os.makedirs(base_path, exist_ok=True)
        logger.info(f"Downloading {repo_id}...")
        
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        
        if use_onnx:
            # Automatic ONNX export
            model = ORTModelForSequenceClassification.from_pretrained(repo_id, export=True)
            model.save_pretrained(base_path)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(repo_id)
            model.save_pretrained(base_path)
            
        tokenizer.save_pretrained(base_path)
        logger.info(f"Successfully saved {name} to {base_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info(f"System architecture: {platform.machine()} ({platform.system()})")
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    # Write a .gitignore inside the models folder so it's excluded from source control automatically
    with open(os.path.join(MODELS_DIR, ".gitignore"), "w") as f:
        f.write("*\n!.gitignore\n")
        
    download_models()
