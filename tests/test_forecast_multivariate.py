"""
Tests for VAR Multivariate Forecasting (Phase 1, Upgrade 4)
"""

import pytest
import pandas as pd
import numpy as np
from agents.forecast import ForecastAgent
from agents.report_validation import ReportValidationEngine

@pytest.fixture
def multivariate_ts_data():
    # Create 40 days of data for VAR (requires at least 30)
    dates = pd.date_range(start="2023-01-01", periods=40, freq="D")
    
    # Generate two mutually dependent series
    # y1 = 2 * t + noise
    # y2 = y1 * 0.5 + noise
    x = np.arange(40, dtype=float)
    y1 = 10 + 1.5 * x + np.random.normal(0, 1, 40)
    y2 = y1 * 0.5 + 5 + np.random.normal(0, 0.5, 40)
    
    df = pd.DataFrame({
        "date": dates,
        "metric_a": y1,
        "metric_b": y2,
        "row_id": np.arange(40)  # structural column to test exclusion
    })
    
    return df

def test_var_validation_rules(multivariate_ts_data):
    engine = ReportValidationEngine()
    
    # 1. Valid columns, sufficient data
    assert engine.validate_var_inputs(multivariate_ts_data, ["metric_a", "metric_b"]) is True
    
    # 2. Too few columns (VAR requires at least 2)
    assert engine.validate_var_inputs(multivariate_ts_data, ["metric_a"]) is False
    
    # 3. Too few rows (requires at least 30)
    short_df = multivariate_ts_data.head(20)
    assert engine.validate_var_inputs(short_df, ["metric_a", "metric_b"]) is False
    
    # 4. Structural columns should be rejected
    assert engine.validate_var_inputs(multivariate_ts_data, ["metric_a", "row_id"]) is False
    
    # 5. Non-existent column
    assert engine.validate_var_inputs(multivariate_ts_data, ["metric_a", "invalid_col"]) is False

def test_forecast_multivariate_execution(multivariate_ts_data):
    agent = ForecastAgent()
    
    # Run multivariate VAR forecast
    results = agent.forecast_multivariate(multivariate_ts_data, ["metric_a", "metric_b"], periods=10)
    
    assert results is not None
    assert "metric_a" in results
    assert "metric_b" in results
    
    for col in ["metric_a", "metric_b"]:
        col_res = results[col]
        assert "forecast" in col_res
        assert "lower" in col_res
        assert "upper" in col_res
        assert "history" in col_res
        assert "r2" in col_res
        assert "k_ar" in col_res
        
        # Verify forecast length
        assert len(col_res["forecast"]) == 10
        assert len(col_res["lower"]) == 10
        assert len(col_res["upper"]) == 10
        assert len(col_res["history"]) == 40
        
        # Check lower <= forecast <= upper
        for f, l, u in zip(col_res["forecast"], col_res["lower"], col_res["upper"]):
            assert l <= f <= u

def test_forecast_multivariate_insufficient_data(multivariate_ts_data):
    agent = ForecastAgent()
    short_df = multivariate_ts_data.head(15)
    
    results = agent.forecast_multivariate(short_df, ["metric_a", "metric_b"], periods=10)
    assert results is None
