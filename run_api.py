
import uvicorn
import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.getcwd())

if __name__ == "__main__":
    print("🚀 Starting AI Data Analyzer API...")
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)
