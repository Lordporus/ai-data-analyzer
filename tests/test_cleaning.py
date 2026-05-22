"""
Test: CleaningAgent — Verify deduplication, type coercion,
null handling, and outlier detection.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.ingestion import IngestionAgent
from agents.cleaning import CleaningAgent

SAMPLE_CSV = Path(__file__).parent / "sample_data.csv"


class TestCleaningAgent:
    def setup_method(self):
        ingestion = IngestionAgent()
        self.ingestion_result = ingestion.run(SAMPLE_CSV)
        self.agent = CleaningAgent()

    def test_cleans_data(self):
        result = self.agent.run(self.ingestion_result)
        # Cleaning should produce a DataFrame with rows <= original
        # (may remove exact duplicates if any exist)
        assert len(result.dataframe) <= self.ingestion_result.row_count

    def test_no_null_numerics(self):
        result = self.agent.run(self.ingestion_result)
        num_cols = [c for c, t in result.detected_types.items() if t == "numeric"]
        for col in num_cols:
            if col in result.dataframe.columns:
                assert result.dataframe[col].isnull().sum() == 0, \
                    f"Column '{col}' still has nulls after cleaning"

    def test_cleaning_log_populated(self):
        result = self.agent.run(self.ingestion_result)
        assert len(result.cleaning_log.steps) > 0

    def test_outlier_report(self):
        result = self.agent.run(self.ingestion_result)
        # Row 60 has salary=300000 which is an outlier
        assert "salary" in result.outlier_report

    def test_agent_log_success(self):
        self.agent.run(self.ingestion_result)
        assert self.agent.log.status == "success"
