"""
Upload route — Accepts a CSV file, runs the full analysis pipeline,
and returns structured results with download links.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from config.settings import MAX_FILE_SIZE_BYTES, OUTPUT_DIR, UPLOAD_DIR
from utils.task_queue import run_analysis_task

router = APIRouter()


@router.post("/upload")
async def upload_and_analyze(file: UploadFile = File(...)):
    """Upload a CSV file and queue the analysis pipeline.

    Returns a JSON object with job_id and status.
    """
    # ── Validate ─────────────────────────────────────────────────────
    filename_lower = file.filename.lower() if file.filename else ""
    if not file.filename or not (filename_lower.endswith(".csv") or filename_lower.endswith(".xlsx") or filename_lower.endswith(".xls")):
        raise HTTPException(400, "Only CSV and Excel (.xlsx, .xls) files are accepted.")

    # ── Save upload ──────────────────────────────────────────────────
    job_id = uuid.uuid4().hex[:12]
    # Strip directory components from the filename to prevent path traversal
    safe_name = Path(file.filename).name
    upload_path = UPLOAD_DIR / f"{job_id}_{safe_name}"
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

    # ── Queue pipeline task ──────────────────────────────────────────
    if run_analysis_task is None:
        raise HTTPException(
            503,
            "Task queue (Celery) is not available — Redis is unreachable or not configured. "
            "Use the Streamlit frontend for in-process analysis."
        )
    try:
        task = run_analysis_task.delay(str(upload_path), str(job_output_dir))
    except Exception as exc:
        raise HTTPException(500, f"Failed to queue task: {exc}")

    # ── Response ─────────────────────────────────────────────────────
    return {
        "job_id": task.id,
        "status": "queued",
        "message": "Analysis queued successfully. Poll status to fetch results.",
        "poll_url": f"/api/status/{task.id}"
    }
