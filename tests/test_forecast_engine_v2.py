"""
Tests for ForecastEngine 2.0 (Phase 27)
"""

import pytest
import pandas as pd
import numpy as np
from agents.forecast import ForecastAgent, ForecastResult
from agents.repair import RepairResult
from agents.insight import InsightResult

@pytest.fixture
def complex_ts_data():
    # Create 60 days of data for better seasonality/smoothing testing
    dates = pd.date_range(start="2023-01-01", periods=60, freq="D")
    
    # 1. Linear trend
    x = np.arange(60, dtype=float)
    y = 2 * x + 10.0
    
    # 2. Seasonality (7-day cycle)
    y += 5.0 * np.sin(2.0 * np.pi * x / 7.0)
    
    # 3. Noise (Volatility)
    y += np.random.normal(0, 2, 60)
    
    df = pd.DataFrame({
        "date": dates,
        "revenue": y
    })
    
    types = {
        "date": "datetime",
        "revenue": "numeric"
    }
    
    return RepairResult(dataframe=df, detected_types=types)

@pytest.fixture
def small_ts_data():
    dates = pd.date_range(start="2023-01-01", periods=5, freq="D")
    df = pd.DataFrame({"date": dates, "val": [1, 2, 3, 4, 5]})
    types = {"date": "datetime", "val": "numeric"}
    return RepairResult(dataframe=df, detected_types=types)

def test_v2_core_logic(complex_ts_data):
    agent = ForecastAgent()
    insight_mock = InsightResult(dataframe=complex_ts_data.dataframe)
    
    result = agent.run({
        "repair": complex_ts_data,
        "insight": insight_mock
    })
    
    assert result.is_time_series is True
    assert "revenue" in result.forecasts
    
    f = result.get_forecast("revenue")
    
    # Verify smoothing
    assert "values_smoothed" in f
    assert len(f["values_smoothed"]) == 60
    
    # Verify metadata
    assert "slope" in f
    assert "r2" in f
    assert "confidence_level" in f
    assert "volatility_index" in f
    assert "seasonality_detected" in f
    
    # Verify interpretation object
    interp = f["interpretation"]
    assert interp["confidence"] in ["LOW", "MEDIUM", "HIGH"]
    assert "trend_direction" in interp
    assert "business_summary" in interp

def test_v2_small_data_suppression(small_ts_data):
    agent = ForecastAgent()
    insight_mock = InsightResult(dataframe=small_ts_data.dataframe)
    
    result = agent.run({
        "repair": small_ts_data,
        "insight": insight_mock
    })
    
    # Should suppress individual forecasts for N < 8
    assert result.is_time_series is True
    assert "val" not in result.forecasts

def test_v2_seasonality_detection(complex_ts_data):
    # This data has a clear 7-day sine wave
    agent = ForecastAgent()
    f_data = agent._generate_forecast_v2(
        complex_ts_data.dataframe, 
        "date", "revenue", "D"
    )
    
    # It might not always detect exactly 7 if noise is high, 
    # but with 60 points and sine wave it should detect SOMETHING
    assert f_data["seasonality_detected"] is True
    assert f_data["dominant_lag"] > 0
