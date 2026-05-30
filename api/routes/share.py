from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from utils.share_reports import get_shared_report, render_shared_report_html

router = APIRouter()


@router.get("/report/{share_token}", response_class=HTMLResponse)
async def view_shared_report(share_token: str):
    """Render a public, read-only shared report."""
    try:
        record, result = get_shared_report(share_token)
        return HTMLResponse(render_shared_report_html(record, result))
    except ValueError as exc:
        raise HTTPException(status_code=410, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not load shared report: {exc}")
