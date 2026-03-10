import subprocess
import sys
import json

rag_script_path = "user_data/scripts/rag_graph.py"
pair = "BTC/USDT"

print(f"Testing subprocess bridge to {rag_script_path} for {pair}...")

try:
    result = subprocess.run(
        [sys.executable, rag_script_path, f"--pair={pair}"],
        capture_output=True,
        text=True,
        check=True
    )
    
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    
    if "--- JSON OUTPUT ---" in result.stdout:
        json_str = result.stdout.split("--- JSON OUTPUT ---")[1].strip()
        parsed = json.loads(json_str)
        print("SUCCESS! Parsed JSON:", parsed)
    else:
        print("FAILED to find JSON output marker.")

except subprocess.CalledProcessError as e:
    print(f"Subprocess failed with exit code {e.returncode}")
    print("STDOUT:", e.stdout)
    print("STDERR:", e.stderr)
except Exception as e:
    print("Exception:", e)
