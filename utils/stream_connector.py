import requests
import time
import pandas as pd
from typing import Callable, Optional

def poll_once(url: str) -> pd.DataFrame:
    """
    Polls a streaming API endpoint once and returns the result as a DataFrame.
    """
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    data = response.json()
    
    # If the response is a single dictionary, wrap it in a list
    if isinstance(data, dict):
        data = [data]
        
    df = pd.DataFrame(data)
    return df

def poll_api_stream(url: str, interval_seconds: int = 30, callback: Optional[Callable[[pd.DataFrame], None]] = None):
    """
    Periodically polls an API endpoint returning JSON data,
    converts it to a pandas DataFrame, and executes a callback.
    """
    while True:
        try:
            df = poll_once(url)
            if callback:
                callback(df)
        except Exception as e:
            print(f"Error polling API stream: {e}")
        time.sleep(interval_seconds)
