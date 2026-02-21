"""
Tests for InsightAgent (AI-Reasoned Upgrade)
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from agents.insight import InsightAgent, TrendInfo
from agents.repair import RepairResult

@pytest.fixture
def sample_data():
    # Add noise to prevent perfect linear correlation (which gets filtered as "derived")
    # Revenue ~ 2 * Cost but with noise
    costs = np.array([50, 55, 60, 65, 70, 75, 80, 85, 90, 250])
    noise = np.random.normal(0, 2, 10) 
    revenue = (costs * 2) + noise
    
    df = pd.DataFrame({
        "revenue": revenue,
        "cost": costs,
        "tax": costs * 0.1, # Derived (perfect)
        "marketing": [5, 4, 6, 5, 7, 6, 8, 7, 9, 10], # Weak correlation
        "category": ["A"] * 5 + ["B"] * 5,
        "date": pd.date_range(start="2023-01-01", periods=10)
    })
    types = {
        "revenue": "numeric", "cost": "numeric", "tax": "numeric", 
        "marketing": "numeric", "category": "categorical", "date": "datetime"
    }
    return RepairResult(dataframe=df, detected_types=types)

def test_kpi_calculation(sample_data):
    agent = InsightAgent()
    agent.llm.provider = "mock" # Enable LLM path
    agent.llm.generate_json = MagicMock(return_value={})
    
    result = agent._execute(sample_data)
    
    # Check KPI existence
    kpi_names = [k.name for k in result.kpi_list]
    assert "Total Rows" in kpi_names
    assert "revenue — Mean" in kpi_names
    
    # Check value
    assert result.kpi_list[0].value == 10

def test_trend_detection(sample_data):
    agent = InsightAgent()
    agent.llm.provider = "mock"
    agent.llm.generate_json = MagicMock(return_value={})
    result = agent._execute(sample_data)
    
    # Cost is strictly increasing
    cost_trend = next(t for t in result.trend_summary if t.column == "cost")
    assert cost_trend.direction == "increasing"
    assert cost_trend.slope > 0

def test_correlation_filtering(sample_data):
    agent = InsightAgent()
    agent.llm.provider = "mock"
    agent.llm.generate_json = MagicMock(return_value={})
    
    # Force correlation calculation
    df = sample_data.dataframe
    num_cols = ["revenue", "cost", "tax", "marketing"]
    corr_matrix = df[num_cols].corr()
    
    filtered = agent._filter_correlations(df, num_cols, corr_matrix)
    
    # "revenue vs cost" should be prevalent (strong correlation but not perfect due to noise)
    # "cost vs tax" is perfect (derived), should be filtered.
    
    assert len(filtered) > 0
    # Check that revenue vs cost is there
    assert any("revenue" in s and "cost" in s for s in filtered)
    
    # Check that cost vs tax is NOT there (perfect correlation)
    tax_cost_present = any("cost" in s and "tax" in s for s in filtered)
    assert not tax_cost_present

def test_ai_strategy_integration(sample_data):
    agent = InsightAgent()
    agent.llm.provider = "mock" # IMPORTANT: Override default 'none'
    
    # Mock successful LLM response
    mock_strategy = {
        "executive_summary": "Revenue is growing but shows volatility.",
        "risks": [{"category": "Operational", "detail": "Outlier detected in revenue."}],
        "opportunities": ["Expand to new markets."],
        "actions": ["Audit last transaction."],
        "relationships": ["Revenue tracks Cost closely."]
    }
    agent.llm.generate_json = MagicMock(return_value=mock_strategy)
    
    result = agent._execute(sample_data)
    
    assert result.executive_summary == "Revenue is growing but shows volatility."
    assert len(result.top_risks) == 1
    assert result.top_risks[0]["category"] == "Operational"
    assert "Audit last transaction." in result.business_recommendations
