"""
Tests for ForecastEngine 3.0 (Adaptive Forecasting)
"""

import pytest
import pandas as pd
import numpy as np
from agents.forecast import ForecastAgent, ForecastResult
from agents.repair import RepairResult
from agents.insight import InsightResult

@pytest.fixture
def seasonal_data():
    # 60 days, weekly cycle (lag=7)
    # len(y) = 60, dominant_lag = 7. 60 >= 2*7 (14) - SUCCESS
    dates = pd.date_range(start="2023-01-01", periods=60, freq="D")
    x = np.arange(60, dtype=float)
    y = 50 + 0.5 * x + 10 * np.sin(2 * np.pi * x / 7) + np.random.normal(0, 1, 60)
    
    df = pd.DataFrame({"date": dates, "val": y})
    types = {"date": "datetime", "val": "numeric"}
    return RepairResult(dataframe=df, detected_types=types)

@pytest.fixture
def insufficient_seasonal_data():
    # 10 days, weekly cycle (lag=7)
    # len(y) = 10, dominant_lag = 7. 10 < 2*7 (14) - FALLBACK TO LINEAR
    dates = pd.date_range(start="2023-01-01", periods=10, freq="D")
    x = np.arange(10, dtype=float)
    y = 50 + 0.5 * x + 10 * np.sin(2 * np.pi * x / 7)
    
    df = pd.DataFrame({"date": dates, "val": y})
    types = {"date": "datetime", "val": "numeric"}
    return RepairResult(dataframe=df, detected_types=types)

def test_holt_winters_selection(seasonal_data):
    agent = ForecastAgent()
    insight_mock = InsightResult(dataframe=seasonal_data.dataframe)
    
    result = agent.run({
        "repair": seasonal_data,
        "insight": insight_mock
    })
    
    assert result.is_time_series is True
    f = result.get_forecast("val")
    assert f["forecast_model_type"] == "Holt-Winters"
    assert f["seasonality_detected"] is True
    assert f["dominant_lag"] == 7
    assert len(f["values_forecast"]) == 10

def test_linear_fallback_on_insufficiency(insufficient_seasonal_data):
    agent = ForecastAgent()
    insight_mock = InsightResult(dataframe=insufficient_seasonal_data.dataframe)
    
    result = agent.run({
        "repair": insufficient_seasonal_data,
        "insight": insight_mock
    })
    
    f = result.get_forecast("val")
    # Even if seasonal, it should fallback because N < 2*Lag
    assert f["forecast_model_type"] == "Linear"

def test_confidence_boost(seasonal_data):
    # This data is very clean sine wave + trend, should get boost
    agent = ForecastAgent()
    insight_mock = InsightResult(dataframe=seasonal_data.dataframe)
    
    result = agent.run({
        "repair": seasonal_data,
        "insight": insight_mock
    })
    
    f = result.get_forecast("val")
    # Clean sine waves usually result in HIGH confidence with Holt-Winters boost
    assert f["confidence_level"] in ["MEDIUM", "HIGH"]
