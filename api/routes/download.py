"""
Download route — Serve generated files by job ID.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Cookie, Header, Query, Depends
from fastapi.responses import FileResponse

from config.settings import APP_BASE_URL, OUTPUT_DIR, API_KEY_ENTERPRISE
from utils.auth import load_session_token
from utils.monetization import is_beta_full_access
from utils.workspace import check_job_ownership, get_job_org_plan
from utils.share_reports import get_shared_report
from utils.rate_limit import rate_limiter

router = APIRouter()

FILE_MAP = {
    "csv": ("cleaned_data.csv", "text/csv"),
    "pdf": ("report.pdf", "application/pdf"),
    "html": ("dashboard.html", "text/html"),
    "md": ("report.md", "text/markdown"),
    "xlsx": ("analysis_report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
}
FILENAME_MAP = {v[0]: v[1] for k, v in FILE_MAP.items()}

@router.get("/download/{job_id}/{filename}", dependencies=[Depends(rate_limiter(limit=120, window=60))])
async def download_file(
    job_id: str, 
    filename: str,
    session_token: str | None = Cookie(default=None),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    share_token: str | None = Query(default=None)
):
    """Download a specific output file by job_id securely."""
    # ── Auth ─────────────────────────────────────────────────────────
    is_authorized = False
    is_pro = False

    # 1. Enterprise API Key Auth — always Pro (paid B2B access)
    if API_KEY_ENTERPRISE and x_api_key == API_KEY_ENTERPRISE:
        is_authorized = True
        is_pro = True

    # 2. Shared Report Token Auth — plan comes from the share record
    elif share_token:
        _share_plan = "free"
        try:
            record, result = get_shared_report(share_token)
            if getattr(result, "job_id", "") != job_id:
                raise HTTPException(403, "Share token is not valid for this job.")
            is_authorized = True
            _share_plan = record.get("plan", "free")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(403, f"Invalid share token: {e}")
        is_pro = str(_share_plan).lower() == "pro"

    # 3. User Session Token Auth — plan resolved from the owning org
    elif session_token:
        user = load_session_token(session_token)
        if not user:
            raise HTTPException(401, "Invalid or expired session.")

        # Verify ownership
        if not check_job_ownership(job_id, user.get("id")):
            raise HTTPException(403, "You do not have permission to access this file.")
        is_authorized = True
        is_pro = get_job_org_plan(job_id) == "pro"

    if not is_authorized:
        raise HTTPException(401, "Authentication required.")

    if is_beta_full_access():
        is_pro = True

    # ── Plan gate — all file exports require Pro ──────────────────────
    if not is_pro:
        upgrade_url = APP_BASE_URL.rstrip("/")
        raise HTTPException(
            403,
            f"Export downloads require a Pro plan. Upgrade at {upgrade_url} to access your analysis outputs.",
        )

    # Validate filename
    if filename in FILE_MAP:
        actual_filename, media_type = FILE_MAP[filename]
    elif filename in FILENAME_MAP:
        actual_filename = filename
        media_type = FILENAME_MAP[filename]
    else:
        raise HTTPException(
            400,
            f"Invalid file '{filename}'. Valid types are: {', '.join(FILE_MAP.keys())} or their respective filenames.",
        )

    file_path = OUTPUT_DIR / job_id / actual_filename

    if not file_path.exists():
        raise HTTPException(404, f"File not found for job '{job_id}'.")

    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=actual_filename,
    )
