"""
Test: Full pipeline — Run MasterOrchestrator on sample CSV
and verify all output files are generated.
"""

import sys
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from orchestrator.master import MasterOrchestrator

SAMPLE_CSV = Path(__file__).parent / "sample_data.csv"
TEST_OUTPUT = Path(__file__).parent / "_test_output"


class TestPipeline:
    def setup_method(self):
        if TEST_OUTPUT.exists():
            shutil.rmtree(TEST_OUTPUT)

    def teardown_method(self):
        if TEST_OUTPUT.exists():
            shutil.rmtree(TEST_OUTPUT)

    def test_full_pipeline(self):
        orch = MasterOrchestrator()
        result = orch.run(SAMPLE_CSV, TEST_OUTPUT)

        # Pipeline completed
        assert result.status == "completed"
        assert result.total_duration_seconds > 0
        assert result.job_id != ""

        # All output files exist
        assert Path(result.cleaned_csv_path).exists(), "Cleaned CSV not found"
        assert Path(result.dashboard_html_path).exists(), "Dashboard HTML not found"
        assert Path(result.pdf_report_path).exists(), "PDF report not found"
        assert Path(result.markdown_report_path).exists(), "Markdown report not found"

    def test_pipeline_produces_insights(self):
        orch = MasterOrchestrator()
        result = orch.run(SAMPLE_CSV, TEST_OUTPUT)

        assert result.insight is not None
        assert len(result.insight.kpi_list) > 0
        assert len(result.insight.business_recommendations) > 0

    def test_pipeline_produces_logs(self):
        orch = MasterOrchestrator()
        result = orch.run(SAMPLE_CSV, TEST_OUTPUT)

        assert len(result.agent_logs) == 9
        agents = [log["agent"] for log in result.agent_logs]
        assert "IngestionAgent" in agents
        assert "ReportAgent" in agents
        for log in result.agent_logs:
            assert log["status"] == "success"

    def test_pipeline_quality_scores(self):
        orch = MasterOrchestrator()
        result = orch.run(SAMPLE_CSV, TEST_OUTPUT)

        assert result.quality_before is not None
        assert result.quality_after is not None
        assert 0 <= result.quality_before.quality_score <= 100
        assert 0 <= result.quality_after.quality_score <= 100
        assert result.quality_after.quality_score >= result.quality_before.quality_score

    def test_cleaned_csv_is_valid(self):
        orch = MasterOrchestrator()
        result = orch.run(SAMPLE_CSV, TEST_OUTPUT)

        import pandas as pd
        original = pd.read_csv(SAMPLE_CSV)
        cleaned = pd.read_csv(result.cleaned_csv_path)
        # Cleaned should have rows <= original (dedup may reduce)
        assert len(cleaned) <= len(original)
        # Cleaning log should have steps applied
        assert len(result.cleaning.cleaning_log.steps) > 0
