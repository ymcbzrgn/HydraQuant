import subprocess
import time
import logging
import sys
import os
import threading

logger = logging.getLogger(__name__)

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

def run_script(script_name):
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    logger.info(f"Executing {script_name}...")
    try:
        # Assuming uv is in the path or we use current python executable from virtualenv
        subprocess.run([sys.executable, script_path], check=True)
        logger.info(f"Completed {script_name}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running {script_name}: {e}")

def run_periodic_tasks():
    logger.info("Starting Data Pipeline cron manager...")
    runs = 0
    while True:
        # Run RSS feeds every 10 mins
        if runs % 10 == 0:
            run_script("rss_fetcher.py")
            
        # Run Fear&Greed and CryptoCompare every 15 mins
        if runs % 15 == 0:
            run_script("fng_fetcher.py")
            run_script("cryptocompare_fetcher.py")
            
        # Run Sentiment Analysis and Aggregation every 15 mins (staggered by 5 mins after fetchers)
        if (runs - 5) % 15 == 0:
            run_script("sentiment_analyzer.py")
            run_script("coin_sentiment_aggregator.py")
            
        time.sleep(60)
        runs += 1

def start_sse_stream():
    """Run the streaming script indefinitely with auto-restart on failure."""
    while True:
        logger.info("Starting SSE Stream processor...")
        script_path = os.path.join(SCRIPTS_DIR, "crypto_cv_stream.py")
        try:
            subprocess.run([sys.executable, script_path], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Stream interrupted. Restarting in 10s... {e}")
            time.sleep(10)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Start the SSE thread (Real-Time)
    sse_thread = threading.Thread(target=start_sse_stream, daemon=True)
    sse_thread.start()
    
    # Run the polling thread (Batch Fetchers)
    run_periodic_tasks()
