"""
Tests for NLQueryAgent.
"""

import pytest
import pandas as pd
from unittest.mock import MagicMock
from agents.nl_query import NLQueryAgent, NLQueryResult

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Region": ["North", "South", "East", "West"],
        "Sales": [100, 150, 120, 130],
        "Date": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"])
    })

def test_nl_query_prompt_structure(sample_df):
    agent = NLQueryAgent()
    schema = agent._get_schema_summary(sample_df)
    
    assert "Region (object)" in schema
    assert "Sales (int64)" in schema
    
    prompt = agent._build_user_prompt("Show me sales by region", schema)
    assert "Show me sales by region" in prompt
    assert "Region (object)" in prompt

def test_nl_query_mock_execution(sample_df):
    agent = NLQueryAgent()
    
    # Mock LLM response
    mock_response = {
        "explanation": "Sales by Region breakdown.",
        "chart_config": {
            "type": "bar",
            "x": "Region",
            "y": "Sales",
            "agg": "sum"
        }
    }
    
    # Patch the generate_json method
    agent.llm.generate_json = MagicMock(return_value=mock_response)
    
    result = agent.run({"query": "Show me sales by region", "df": sample_df})
    
    assert result.explanation == "Sales by Region breakdown."
    assert result.chart_config["type"] == "bar"
    assert result.chart_config["x"] == "Region"

def test_nl_query_invalid_column(sample_df):
    agent = NLQueryAgent()
    
    # Mock LLM response with non-existent column
    mock_response = {
        "explanation": "Here is the chart.",
        "chart_config": {
            "type": "bar",
            "x": "InvalidCol",
            "y": "Sales"
        }
    }
    
    agent.llm.generate_json = MagicMock(return_value=mock_response)
    
    result = agent.run({"query": "Show me invalid val", "df": sample_df})
    
    # Chart config should be dropped due to validation failure
    assert result.chart_config is None
    assert result.explanation == "Here is the chart."
