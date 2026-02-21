"""
Upload route — Accepts a CSV file, runs the full analysis pipeline,
and returns structured results with download links.
"""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from config.settings import MAX_FILE_SIZE_BYTES, OUTPUT_DIR, UPLOAD_DIR
from orchestrator.master import MasterOrchestrator

router = APIRouter()


@router.post("/upload")
async def upload_and_analyze(file: UploadFile = File(...)):
    """Upload a CSV file and trigger the full analysis pipeline.

    Returns a JSON summary with download links for all generated
    artefacts (cleaned CSV, dashboard HTML, PDF report).
    """
    # ── Validate ─────────────────────────────────────────────────────
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Only CSV files are accepted.")

    # ── Save upload ──────────────────────────────────────────────────
    job_id = uuid.uuid4().hex[:12]
    upload_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    job_output_dir = OUTPUT_DIR / job_id

    try:
        content = await file.read()
        if len(content) > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                413,
                f"File too large. Maximum size is "
                f"{MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB.",
            )
        upload_path.write_bytes(content)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Failed to save file: {exc}")

    # ── Run pipeline ─────────────────────────────────────────────────
    try:
        orchestrator = MasterOrchestrator()
        pipeline = orchestrator.run(upload_path, job_output_dir)
    except Exception as exc:
        raise HTTPException(500, f"Pipeline failed: {exc}")

    # ── Response ─────────────────────────────────────────────────────
    base = f"/outputs/{job_id}"
    return {
        "job_id": pipeline.job_id,
        "status": pipeline.status,
        "duration_seconds": pipeline.total_duration_seconds,
        "row_count": pipeline.ingestion.row_count if pipeline.ingestion else 0,
        "col_count": pipeline.ingestion.col_count if pipeline.ingestion else 0,
        "downloads": {
            "cleaned_csv": f"{base}/cleaned_data.csv",
            "dashboard_html": f"{base}/dashboard.html",
            "pdf_report": f"{base}/report.pdf",
            "markdown_report": f"{base}/report.md",
        },
        "recommendations": (
            pipeline.insight.business_recommendations if pipeline.insight else []
        ),
        "agent_logs": pipeline.agent_logs,
        "errors": pipeline.errors,
    }
