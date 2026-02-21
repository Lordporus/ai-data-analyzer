"""
Enterprise API route — JSON-based analysis endpoint with API key
authentication for programmatic / SaaS integration.
"""

from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Header, HTTPException, UploadFile

from config.settings import API_KEY_ENTERPRISE, MAX_FILE_SIZE_BYTES, OUTPUT_DIR
from orchestrator.master import MasterOrchestrator

router = APIRouter()


@router.post("/analyze")
async def enterprise_analyze(
    file: UploadFile = File(...),
    x_api_key: str = Header(..., alias="X-API-Key"),
):
    """Enterprise analysis endpoint.

    Requires a valid API key in the X-API-Key header.
    Returns full pipeline results as JSON.
    """
    # ── Auth ─────────────────────────────────────────────────────────
    if x_api_key != API_KEY_ENTERPRISE:
        raise HTTPException(403, "Invalid API key.")

    # ── Validate ─────────────────────────────────────────────────────
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Only CSV files are accepted.")

    content = await file.read()
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(413, "File exceeds maximum size limit.")

    # ── Save & process ───────────────────────────────────────────────
    job_id = uuid.uuid4().hex[:12]
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    upload_path = job_dir / file.filename
    upload_path.write_bytes(content)

    orchestrator = MasterOrchestrator()
    result = orchestrator.run(upload_path, job_dir)

    # ── Return structured JSON ───────────────────────────────────────
    summary = result.summary_dict()
    base = f"/outputs/{job_id}"
    summary["downloads"] = {
        "cleaned_csv": f"{base}/cleaned_data.csv",
        "dashboard_html": f"{base}/dashboard.html",
        "pdf_report": f"{base}/report.pdf",
        "markdown_report": f"{base}/report.md",
    }

    # Include KPIs in response
    if result.insight:
        summary["kpis"] = [
            {"name": k.name, "value": k.value, "description": k.description}
            for k in result.insight.kpi_list[:20]
        ]

    return summary
