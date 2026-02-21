
import sys
import os
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, os.getcwd())

from orchestrator.master import MasterOrchestrator

def main():
    print("🚀 Starting Preview Analysis...")
    
    # 1. Setup paths
    sample_csv = Path("tests/sample_data.csv")
    if not sample_csv.exists():
        # Create dummy data if missing
        import pandas as pd
        import numpy as np
        df = pd.DataFrame({
            "id": range(1, 101),
            "category": np.random.choice(["A", "B", "C"], 100),
            "value": np.random.normal(50, 15, 100),
            "date": pd.date_range("2023-01-01", periods=100)
        })
        sample_csv.parent.mkdir(exist_ok=True)
        df.to_csv(sample_csv, index=False)
        print(f"Created dummy sample data at {sample_csv}")

    output_dir = Path("preview_output")
    
    # 2. Run Pipeline
    orchestrator = MasterOrchestrator()
    result = orchestrator.run(sample_csv, output_dir)
    
    if result.status == "completed":
        print(f"✅ Analysis Complete!")
        print(f"Report: {result.markdown_report_path}")
        print(f"Dashboard: {result.dashboard_html_path}")
    else:
        print(f"❌ Analysis Failed: {result.errors}")

if __name__ == "__main__":
    main()
