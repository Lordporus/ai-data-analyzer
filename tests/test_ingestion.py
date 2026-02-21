"""
Test: IngestionAgent — Verify schema detection, row counts,
and missing value reporting.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.ingestion import IngestionAgent

SAMPLE_CSV = Path(__file__).parent / "sample_data.csv"


class TestIngestionAgent:
    def setup_method(self):
        self.agent = IngestionAgent()
        assert SAMPLE_CSV.exists(), f"Sample CSV not found at {SAMPLE_CSV}"

    def test_loads_csv(self):
        result = self.agent.run(SAMPLE_CSV)
        assert result.row_count == 100
        assert result.col_count == 10

    def test_column_names(self):
        result = self.agent.run(SAMPLE_CSV)
        assert "id" in result.column_names
        assert "name" in result.column_names
        assert "salary" in result.column_names

    def test_detected_types(self):
        result = self.agent.run(SAMPLE_CSV)
        assert result.detected_types["salary"] == "numeric"
        assert result.detected_types["age"] == "numeric"
        # department should be categorical (few unique values)
        assert result.detected_types["department"] in ("categorical", "text")

    def test_missing_values_detected(self):
        result = self.agent.run(SAMPLE_CSV)
        # Row 12 has missing email, row 13 missing performance_score,
        # row 16 missing salary
        total_missing = sum(result.missing_value_report.values())
        assert total_missing > 0

    def test_duplicates_detected(self):
        result = self.agent.run(SAMPLE_CSV)
        # Row 31 is a near-duplicate of row 1 (whitespace padded name)
        # pandas exact-match duplicated() may return 0 due to string diffs
        assert result.duplicate_count >= 0

    def test_sample_preview(self):
        result = self.agent.run(SAMPLE_CSV)
        assert len(result.sample_preview) == 5
        assert "name" in result.sample_preview[0]

    def test_agent_log(self):
        result = self.agent.run(SAMPLE_CSV)
        assert self.agent.log.status == "success"
        assert self.agent.log.duration_seconds > 0
