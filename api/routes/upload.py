"""
Upload route — Accepts a CSV file, runs the full analysis pipeline,
and returns structured results with download links.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Cookie, File, Header, HTTPException, UploadFile

from config.settings import MAX_FILE_SIZE_BYTES, OUTPUT_DIR, UPLOAD_DIR
from utils.auth import load_session_token
from utils.monetization import FREE_FILE_SIZE_BYTES, can_run_analysis, is_pro_org
from utils.task_queue import run_analysis_task
from utils.workspace import get_organizations

router = APIRouter()


def _resolve_user_org(user_id: str) -> dict | None:
    """Resolve an authenticated user's organization from their analysis history.

    Returns the first organization the user has recorded runs for, falling
    back to None (which means free-tier limits apply).
    """
    from utils.workspace import get_org_analysis_history
    orgs = get_organizations()
    if not orgs:
        return None
    # Check each org for runs belonging to this user
    for org in orgs:
        history = get_org_analysis_history(org.get("id", ""))
        for run in history:
            if run.get("user_id") == user_id:
                return org
    # No matching org found — return the first org (default workspace)
    # but do NOT trust its plan; caller will treat as free.
    return None


@router.post("/upload")
async def upload_and_analyze(
    file: UploadFile = File(...),
    session_token: str | None = Cookie(default=None),
    x_org_id: str = Header(default="default"),
):
    """Upload a CSV file and queue the analysis pipeline.

    Returns a JSON object with job_id and status.
    """
    # ── Authenticate & Resolve Plan Server-Side ──────────────────────
    # The organization used for monetization decisions MUST come from
    # authenticated server-side identity only. The client-provided
    # x_org_id header is NOT used for plan/limit decisions.
    user = None
    org = None

    if session_token:
        user = load_session_token(session_token)

    if user:
        org = _resolve_user_org(user.get("id", ""))

    # Plan determination: only trust server-resolved org
    is_pro = is_pro_org(org) if org else False
    effective_limit = MAX_FILE_SIZE_BYTES if is_pro else min(MAX_FILE_SIZE_BYTES, FREE_FILE_SIZE_BYTES)

    # ── Quota enforcement (authenticated free-tier users only) ───────
    # Anonymous users are not quota-checked: can_run_analysis(None)
    # counts against org_id="default", which would incorrectly deny
    # all anonymous traffic once any org hits the limit.
    if user and not is_pro:
        allowed, quota_msg, _, _ = can_run_analysis(org)
        if not allowed:
            raise HTTPException(429, quota_msg)

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
        if len(content) > effective_limit:
            raise HTTPException(
                413,
                f"File too large. Maximum size is "
                f"{effective_limit // (1024 * 1024)} MB for this plan.",
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
