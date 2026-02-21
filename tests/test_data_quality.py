"""
Test: DataQualityAgent — Verify score computation, color logic,
metric detection, and edge cases.
"""

import sys
from pathlib import Path

import pytest
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.data_quality import DataQualityAgent, score_color, risk_level


class TestScoreColor:
    def test_green(self):
        assert score_color(95) == "#22c55e"
        assert score_color(80) == "#22c55e"

    def test_yellow(self):
        assert score_color(79) == "#eab308"
        assert score_color(60) == "#eab308"

    def test_red(self):
        assert score_color(59) == "#ef4444"
        assert score_color(0) == "#ef4444"


class TestRiskLevel:
    def test_low(self):
        assert risk_level(85) == "Low"

    def test_medium(self):
        assert risk_level(70) == "Medium"

    def test_high(self):
        assert risk_level(40) == "High"


class TestDataQualityAgent:
    def setup_method(self):
        self.agent = DataQualityAgent()

    def test_perfect_data(self):
        df = pd.DataFrame({
            "id": range(100),
            "value": np.random.randn(100) * 10 + 50,
            "category": np.random.choice(["A", "B", "C"], 100),
        })
        result = self.agent.run({
            "dataframe": df,
            "detected_types": {"id": "numeric", "value": "numeric", "category": "categorical"},
        })
        # Clean data should score high
        assert result.quality_score >= 80
        assert result.risk_level == "Low"
        assert result.missing_percent == 0.0
        assert result.duplicate_percent == 0.0

    def test_messy_data(self):
        df = pd.DataFrame({
            "a": [1, np.nan, np.nan, np.nan, 5, 6, 7, np.nan, 9, 10],
            "b": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # constant
            "c": ["x", "y", "x", "y", "x", "y", "x", "y", "x", "y"],
        })
        result = self.agent.run({
            "dataframe": df,
            "detected_types": {"a": "numeric", "b": "numeric", "c": "categorical"},
        })
        # Has missing data + a constant column
        assert result.missing_percent > 0
        assert "b" in result.constant_columns
        assert result.quality_score < 100

    def test_duplicates_detected(self):
        df = pd.DataFrame({"x": [1, 2, 3, 1, 2, 3], "y": [4, 5, 6, 4, 5, 6]})
        result = self.agent.run({
            "dataframe": df,
            "detected_types": {"x": "numeric", "y": "numeric"},
        })
        assert result.duplicate_percent > 0

    def test_null_heavy_columns(self):
        n = 100
        df = pd.DataFrame({
            "good": range(n),
            "bad": [np.nan] * 40 + list(range(60)),
        })
        result = self.agent.run({
            "dataframe": df,
            "detected_types": {"good": "numeric", "bad": "numeric"},
        })
        assert "bad" in result.null_heavy_columns

    def test_output_structure(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        result = self.agent.run({
            "dataframe": df,
            "detected_types": {"a": "numeric", "b": "categorical"},
        })
        assert isinstance(result.quality_score, float)
        assert isinstance(result.summary_text, str)
        assert isinstance(result.problem_columns, list)
        assert result.total_rows == 3
        assert result.total_cols == 2

    def test_agent_log(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        self.agent.run({"dataframe": df, "detected_types": {"a": "numeric"}})
        assert self.agent.log.status == "success"

    def test_sample_csv(self):
        csv_path = Path(__file__).parent / "sample_data.csv"
        if csv_path.exists():
            from agents.ingestion import IngestionAgent
            ing = IngestionAgent()
            ing_result = ing.run(csv_path)
            result = self.agent.run({
                "dataframe": ing_result.dataframe,
                "detected_types": ing_result.detected_types,
            })
            assert 0 <= result.quality_score <= 100
            assert result.risk_level in ("Low", "Medium", "High")
            assert len(result.summary_text) > 0
