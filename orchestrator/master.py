"""
MasterOrchestrator — Runs the full analysis pipeline in sequence,
passing structured outputs between agents and collecting logs/timing.

Usage:
    from orchestrator import MasterOrchestrator
    result = MasterOrchestrator().run(csv_path, output_dir)
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

import pandas as pd

from agents.ingestion import IngestionAgent, IngestionResult
from agents.cleaning import CleaningAgent, CleaningResult
from agents.repair import RepairReasoningAgent, RepairResult
from agents.data_quality import DataQualityAgent, DataQualityResult
from agents.insight import InsightAgent, InsightResult
from agents.forecast import ForecastAgent, ForecastResult
from agents.dashboard import DashboardAgent, DashboardResult
from agents.report import ReportAgent, ReportResult
from agents.base import AgentLog

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Aggregated output of the entire pipeline."""
    # Updated with Data Quality fields
    job_id: str = ""
    status: str = "pending"
    total_duration_seconds: float = 0.0

    # Stage outputs
    ingestion: Optional[IngestionResult] = None
    cleaning: Optional[CleaningResult] = None
    repair: Optional[RepairResult] = None
    quality_before: Optional[DataQualityResult] = None
    quality_after: Optional[DataQualityResult] = None
    insight: Optional[InsightResult] = None
    forecast: Optional[ForecastResult] = None
    dashboard: Optional[DashboardResult] = None
    report: Optional[ReportResult] = None

    # Paths to deliverables
    cleaned_csv_path: str = ""
    dashboard_html_path: str = ""
    pdf_report_path: str = ""
    markdown_report_path: str = ""

    # Logs
    agent_logs: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def summary_dict(self) -> Dict[str, Any]:
        """Serialisable summary for API responses."""
        return {
            "job_id": self.job_id,
            "status": self.status,
            "duration_seconds": self.total_duration_seconds,
            "row_count": self.ingestion.row_count if self.ingestion else 0,
            "col_count": self.ingestion.col_count if self.ingestion else 0,
            "quality_score_before": self.quality_before.quality_score if self.quality_before else None,
            "quality_score_after": self.quality_after.quality_score if self.quality_after else None,
            "charts": self.dashboard.chart_count if self.dashboard else 0,
            "kpis": self.dashboard.kpi_count if self.dashboard else 0,
            "cleaned_csv": self.cleaned_csv_path,
            "dashboard_html": self.dashboard_html_path,
            "pdf_report": self.pdf_report_path,
            "markdown_report": self.markdown_report_path,
            "recommendations": (
                self.insight.business_recommendations if self.insight else []
            ),
            "audit_logs": {
                "insights": self.insight.audit_log if self.insight else {},
                "forecasts": self.forecast.audit_log if self.forecast else {},
            },
            "errors": self.errors,
        }


class MasterOrchestrator:
    """Execute the full Ingestion → Cleaning → Repair → Insight →
    Dashboard → Report pipeline and return all artefacts."""

    def run(self, csv_path: str | Path | pd.DataFrame, output_dir: str | Path, branding: Optional[Dict[str, Any]] = None, progress_callback: Optional[Callable[[str, float], None]] = None, dataset_name: Optional[str] = None) -> PipelineResult:
        if not isinstance(csv_path, pd.DataFrame):
            csv_path = Path(csv_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        job_id = uuid.uuid4().hex[:12]
        result = PipelineResult(job_id=job_id, status="running")
        start = time.perf_counter()

        try:
            # ── 1) Ingestion ─────────────────────────────────────────
            if progress_callback:
                progress_callback("Ingesting data...", 0.05)
            ingestion_agent = IngestionAgent()
            ingestion_result: IngestionResult = ingestion_agent.run(csv_path)
            
            # Handle large datasets by downsampling for stable downstream processing
            from config.settings import MAX_ROWS_FULL
            if ingestion_result.row_count > MAX_ROWS_FULL:
                logger.info(
                    "Ingested dataset has %d rows, exceeding MAX_ROWS_FULL (%d). "
                    "Downsampling to avoid downstream memory/performance issues.",
                    ingestion_result.row_count, MAX_ROWS_FULL
                )
                ingestion_result.dataframe = ingestion_result.dataframe.sample(
                    n=MAX_ROWS_FULL, random_state=42
                ).reset_index(drop=True)
                
            result.ingestion = ingestion_result
            result.agent_logs.append(self._log_entry(ingestion_agent))

            # ── 2) Data Quality BEFORE cleaning ──────────────────────
            if progress_callback:
                progress_callback("Assessing data quality...", 0.15)
            dq_before_agent = DataQualityAgent()
            quality_before: DataQualityResult = dq_before_agent.run({
                "dataframe": ingestion_result.dataframe,
                "detected_types": ingestion_result.detected_types,
            })
            result.quality_before = quality_before
            result.agent_logs.append(self._log_entry(dq_before_agent))

            # ── 3) Cleaning ──────────────────────────────────────────
            if progress_callback:
                progress_callback("Cleaning & deduplicating...", 0.30)
            cleaning_agent = CleaningAgent()
            cleaning_result: CleaningResult = cleaning_agent.run(ingestion_result)
            result.cleaning = cleaning_result
            result.agent_logs.append(self._log_entry(cleaning_agent))

            # ── 4) Repair ────────────────────────────────────────────
            if progress_callback:
                progress_callback("Applying intelligent repairs...", 0.45)
            repair_agent = RepairReasoningAgent()
            repair_result: RepairResult = repair_agent.run(cleaning_result)
            result.repair = repair_result
            result.agent_logs.append(self._log_entry(repair_agent))

            # ── 5) Data Quality AFTER cleaning ───────────────────────
            if progress_callback:
                progress_callback("Re-assessing quality...", 0.55)
            dq_after_agent = DataQualityAgent()
            quality_after: DataQualityResult = dq_after_agent.run({
                "dataframe": repair_result.dataframe,
                "detected_types": cleaning_result.detected_types,
            })
            result.quality_after = quality_after
            result.agent_logs.append(self._log_entry(dq_after_agent))

            # ── 6) Insight ───────────────────────────────────────────
            if progress_callback:
                progress_callback("Generating insights...", 0.65)
            insight_agent = InsightAgent()
            insight_result: InsightResult = insight_agent.run(repair_result)
            result.insight = insight_result
            result.agent_logs.append(self._log_entry(insight_agent))

            # ── 6b) Forecast ─────────────────────────────────────────
            if progress_callback:
                progress_callback("Running forecast model...", 0.75)
            forecast_agent = ForecastAgent()
            forecast_result: ForecastResult = forecast_agent.run({
                "repair": repair_result,
                "insight": insight_result
            })
            result.forecast = forecast_result
            result.agent_logs.append(self._log_entry(forecast_agent))

            # ── 7) Save cleaned CSV ──────────────────────────────────
            cleaned_path = output_dir / "cleaned_data.csv"
            repair_result.dataframe.to_csv(cleaned_path, index=False)
            result.cleaned_csv_path = str(cleaned_path)

            # ── 8) Dashboard ─────────────────────────────────────────
            if progress_callback:
                progress_callback("Building dashboard...", 0.85)
            dashboard_agent = DashboardAgent()
            dash_result: DashboardResult = dashboard_agent.run({
                "insight": insight_result,
                "forecast": forecast_result,
                "quality_before": quality_before,
                "quality_after": quality_after,
                "output_dir": output_dir,
            })
            result.dashboard = dash_result
            result.dashboard_html_path = dash_result.html_path
            result.agent_logs.append(self._log_entry(dashboard_agent))

            # ── 9) Report ────────────────────────────────────────────
            if progress_callback:
                progress_callback("Creating PDF/Markdown reports...", 0.95)
            report_agent = ReportAgent()
            report_result: ReportResult = report_agent.run({
                "ingestion": ingestion_result,
                "cleaning": cleaning_result,
                "repair": repair_result,
                "insight": insight_result,
                "forecast": forecast_result,
                "quality_before": quality_before,
                "quality_after": quality_after,
                "output_dir": output_dir,
                "branding": branding,
                "dataset_name": dataset_name,
            })
            result.report = report_result
            result.pdf_report_path = report_result.pdf_path
            result.markdown_report_path = report_result.markdown_path
            result.agent_logs.append(self._log_entry(report_agent))

            result.status = "completed"

        except Exception as exc:
            result.status = "failed"
            result.errors.append(str(exc))
            logger.exception("Pipeline failed at job %s", job_id)

        result.total_duration_seconds = round(time.perf_counter() - start, 3)
        logger.info(
            "Pipeline %s %s in %.2fs",
            job_id, result.status, result.total_duration_seconds,
        )
        return result

    @staticmethod
    def _log_entry(agent) -> Dict[str, Any]:
        log: AgentLog = agent.log
        return {
            "agent": log.agent_name,
            "status": log.status,
            "duration": log.duration_seconds,
            "messages": log.messages,
            "errors": log.errors,
        }
