"""
Download route — Serve generated files by job ID.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from config.settings import OUTPUT_DIR

router = APIRouter()

FILE_MAP = {
    "csv": ("cleaned_data.csv", "text/csv"),
    "pdf": ("report.pdf", "application/pdf"),
    "html": ("dashboard.html", "text/html"),
    "md": ("report.md", "text/markdown"),
}


@router.get("/download/{job_id}/{file_type}")
async def download_file(job_id: str, file_type: str):
    """Download a specific output file by job_id and type.

    file_type must be one of: csv, pdf, html, md
    """
    if file_type not in FILE_MAP:
        raise HTTPException(
            400,
            f"Invalid file_type '{file_type}'. Use: {', '.join(FILE_MAP.keys())}",
        )

    filename, media_type = FILE_MAP[file_type]
    file_path = OUTPUT_DIR / job_id / filename

    if not file_path.exists():
        raise HTTPException(404, f"File not found for job '{job_id}'.")

    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=filename,
    )
