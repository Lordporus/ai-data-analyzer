"""
Test Excel and Google Sheets Ingestion capabilities.
"""
import sys
import tempfile
from pathlib import Path
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.ingestion import IngestionAgent
from utils.gsheets import load_google_sheet

SAMPLE_CSV = Path(__file__).parent / "sample_data.csv"

def test_excel_xlsx_ingestion():
    agent = IngestionAgent()
    assert SAMPLE_CSV.exists()
    
    # Read sample data and save to temp excel
    df = pd.read_csv(SAMPLE_CSV)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        excel_path = Path(tmpdir) / "sample.xlsx"
        df.to_excel(excel_path, index=False, engine="openpyxl")
        
        result = agent.run(excel_path)
        assert result.row_count == 100
        assert result.col_count == 10
        assert "salary" in result.column_names
        assert result.detected_types["salary"] == "numeric"

@patch("utils.gsheets.pd.read_csv")
def test_gsheets_keyless_public_url(mock_read_csv):
    # Mock return value of pandas read_csv for gsheets export
    mock_df = pd.DataFrame({"id": [1, 2], "name": ["A", "B"]})
    mock_read_csv.return_value = mock_df
    
    url = "https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit?usp=sharing"
    df = load_google_sheet(url)
    
    # Assert public export url was constructed and requested
    mock_read_csv.assert_called_once_with(
        "https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/export?format=csv"
    )
    assert not df.empty
    assert len(df) == 2
    assert "name" in df.columns

@patch("utils.gsheets.pd.read_csv")
def test_gsheets_url_invalid(mock_read_csv):
    mock_read_csv.side_effect = Exception("HTTP 404")
    url = "https://docs.google.com/spreadsheets/d/invalid_id/edit"
    
    with pytest.raises(ValueError) as excinfo:
        load_google_sheet(url)
    
    assert "Failed to load Google Sheet" in str(excinfo.value)

def test_ingest_chunked_directly():
    agent = IngestionAgent()
    assert SAMPLE_CSV.exists()
    
    # Run _ingest_chunked directly on the sample CSV with a small chunk size of 10
    df = agent._ingest_chunked(SAMPLE_CSV, chunk_size=10)
    
    assert len(df) == 100
    assert "salary" in df.columns

@patch("agents.ingestion.Path.stat")
def test_chunked_trigger_on_large_file(mock_stat):
    # Mock file size to be 60MB
    mock_stat.return_value.st_size = 60 * 1024 * 1024
    
    agent = IngestionAgent()
    with patch.object(agent, "_ingest_chunked", return_value=pd.read_csv(SAMPLE_CSV)) as mock_ingest:
        agent.run(SAMPLE_CSV)
        mock_ingest.assert_called_once()

from orchestrator.master import MasterOrchestrator

@patch("orchestrator.master.IngestionAgent.run")
def test_master_orchestrator_downsampling(mock_ingestion_run):
    # Create an ingestion result with 1.2M rows
    large_df = pd.DataFrame({"col": range(1200000)})
    mock_result = MagicMock()
    mock_result.row_count = 1200000
    mock_result.dataframe = large_df
    mock_ingestion_run.return_value = mock_result
    
    orch = MasterOrchestrator()
    # Mock other downstream agents to avoid running full pipeline
    with patch("orchestrator.master.DataQualityAgent.run"), \
         patch("orchestrator.master.CleaningAgent.run"), \
         patch("orchestrator.master.RepairReasoningAgent.run"), \
         patch("orchestrator.master.InsightAgent.run"), \
         patch("orchestrator.master.ForecastAgent.run"), \
         patch("orchestrator.master.DashboardAgent.run"), \
         patch("orchestrator.master.ReportAgent.run"):
         
         # Mock saving CSV
         with patch("pandas.DataFrame.to_csv"):
             res = orch.run("dummy.csv", "dummy_out")
             # It should downsample the dataframe to MAX_ROWS_FULL (1,000,000)
             assert len(res.ingestion.dataframe) == 1000000
