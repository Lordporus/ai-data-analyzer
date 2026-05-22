
import logging
import pandas as pd
from agents.nl_query import NLQueryAgent

# Configure logging to see errors
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agents.nl_query")
logger.setLevel(logging.DEBUG)

def test_agent():
    print("Initializing Agent...")
    agent = NLQueryAgent()
    
    df = pd.DataFrame({
        "Date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
        "Sales": [100, 150, 120],
        "Region": ["North", "South", "North"]
    })
    
    query = "Show me the trend of Sales over time"
    print(f"Running Query: '{query}'")
    
    result = agent.run({"query": query, "df": df})
    
    print("\n--- Result ---")
    print(f"Explanation: {result.explanation}")
    print(f"Chart Config: {result.chart_config}")
    print(f"Error: {result.error}")

if __name__ == "__main__":
    test_agent()
