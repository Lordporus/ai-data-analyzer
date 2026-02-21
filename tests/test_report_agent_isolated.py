
import sys
import shutil
from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.report import ReportAgent
from agents.ingestion import IngestionResult
from agents.cleaning import CleaningResult
from agents.repair import RepairResult
from agents.insight import InsightResult, KPI, TrendInfo
from agents.forecast import ForecastResult
from agents.data_quality import DataQualityResult

TEST_OUTPUT = Path(__file__).parent / "_test_report_agent"

class TestReportAgentIsolated:
    def setup_method(self):
        if TEST_OUTPUT.exists():
            shutil.rmtree(TEST_OUTPUT)

    def teardown_method(self):
        # pass # Keep for inspection if needed
        if TEST_OUTPUT.exists():
             shutil.rmtree(TEST_OUTPUT)

    def test_report_generation(self):
        # Mock Data
        df = pd.DataFrame({
            "Revenue": [100, 110, 120, 130, 140],
            "Cost": [50, 55, 60, 65, 70],
            "ID": [1, 2, 3, 4, 5]
        })
        
        ingestion = IngestionResult(
            file_size_bytes=1000,
            row_count=5,
            col_count=3,
            detected_types={"Revenue": "int64", "Cost": "int64", "ID": "int64"},
            dataframe=df
        )
        
        cleaning = CleaningResult(
            dataframe=df,
            cleaning_log=None 
        )
        
        repair = RepairResult(
            dataframe=df,
            detected_types={"Revenue": "int64", "Cost": "int64", "ID": "int64"}
        )

        dq_result = DataQualityResult(
            quality_score=90.0,
            missing_percent=0.0,
            duplicate_percent=0.0
        )
        
        insight = InsightResult(
            kpi_list=[KPI("Revenue", 100, "total"), KPI("Total Cost", 50, "total")],
            trend_summary=[TrendInfo("Revenue", 10.0, "increasing", 0.01, 0.9)],
            correlation_matrix=pd.DataFrame({"Revenue": [1.0, 0.9], "Cost": [0.9, 1.0]}, index=["Revenue", "Cost"]),
            business_recommendations=["Increase revenue", "Decrease cost"],
            executive_summary="Executive Summary text.",
            top_risks=["High risk"],
            top_opportunities=["High growth"],
            dataframe=df,
            detected_types={"Revenue": "int64", "Cost": "int64", "ID": "int64"}
        )
        
        forecast = ForecastResult(
            is_time_series=True,
            forecasts={"Revenue": {"values_hist": [100, 110], "values_forecast": [120, 130]}}
        )
        
        agent = ReportAgent()
        input_data = {
            "ingestion": ingestion,
            "cleaning": cleaning,
            "repair": repair,
            "insight": insight,
            "forecast": forecast,
            "quality_before": dq_result,
            "quality_after": dq_result,
            "output_dir": TEST_OUTPUT
        }
        
        result = agent._execute(input_data)
        
        assert Path(result.pdf_path).exists()
        assert Path(result.markdown_path).exists()
