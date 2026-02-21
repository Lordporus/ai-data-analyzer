"""
Tests for ForecastAgent (Phase 14)
"""

import pytest
import pandas as pd
import numpy as np
from agents.forecast import ForecastAgent, ForecastResult
from agents.repair import RepairResult
from agents.insight import InsightResult

@pytest.fixture
def time_series_data():
    # Create 30 days of data
    dates = pd.date_range(start="2023-01-01", periods=30, freq="D")
    
    # Linear trend: y = 2x + 10
    x = np.arange(30)
    sales = 2 * x + 10
    
    # Add some noise
    noise = np.random.normal(0, 1, 30)
    sales = sales + noise
    
    df = pd.DataFrame({
        "date": dates,
        "sales": sales,
        "category": ["A"] * 30
    })
    
    types = {
        "date": "datetime",
        "sales": "numeric",
        "category": "categorical"
    }
    
    return RepairResult(dataframe=df, detected_types=types)

@pytest.fixture
def non_time_series_data():
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "val": [10, 20, 30]
    })
    types = {"id": "numeric", "val": "numeric"}
    return RepairResult(dataframe=df, detected_types=types)

def test_time_series_detection(time_series_data):
    agent = ForecastAgent()
    date_col, period = agent._detect_time_series(
        time_series_data.dataframe, 
        time_series_data.detected_types
    )
    assert date_col == "date"
    assert period == "D"

def test_no_time_series(non_time_series_data):
    agent = ForecastAgent()
    date_col, period = agent._detect_time_series(
        non_time_series_data.dataframe, 
        non_time_series_data.detected_types
    )
    assert date_col == ""
    assert period == ""

def test_forecast_generation(time_series_data):
    agent = ForecastAgent()
    insight_mock = InsightResult(dataframe=time_series_data.dataframe)
    
    result = agent.run({
        "repair": time_series_data,
        "insight": insight_mock
    })
    
    assert result.is_time_series is True
    assert result.primary_date_col == "date"
    assert "sales" in result.forecasts
    
    f_data = result.get_forecast("sales")
    assert len(f_data["values_forecast"]) == 10 # Default periods
    assert f_data["values_forecast"][0] > f_data["values_hist"][-1] # Upward trend

def test_scenario_simulation(time_series_data):
    agent = ForecastAgent()
    df = time_series_data.dataframe
    
    # Initial sum
    initial_sales = df["sales"].sum()
    
    # Simulate +10%
    new_sales = agent.simulate_scenario(df, "sales", 1.10)
    
    assert new_sales > initial_sales
    assert np.isclose(new_sales, initial_sales * 1.10)
