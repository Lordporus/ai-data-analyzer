from __future__ import annotations

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import HTMLResponse

from utils.share_reports import get_shared_report, render_shared_report_html
from utils.rate_limit import rate_limiter

router = APIRouter()


@router.get("/report/{share_token}", response_class=HTMLResponse, dependencies=[Depends(rate_limiter(limit=120, window=60))])
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
