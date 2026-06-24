import os
import logging
import shutil
import boto3
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# R2 Environment Credentials
R2_ENDPOINT  = os.getenv("R2_ENDPOINT", "")
R2_ACCESS_KEY = os.getenv("R2_ACCESS_KEY", "")
R2_SECRET_KEY = os.getenv("R2_SECRET_KEY", "")
R2_BUCKET    = os.getenv("R2_BUCKET", "ai-analyzer-outputs")
R2_PUBLIC_URL = os.getenv("R2_PUBLIC_URL", "")

# Local Fallback Destination (persisted via Docker volume ./outputs:/app/outputs)
LOCAL_STORAGE_DIR = Path(__file__).resolve().parent.parent / "outputs" / "persistent_storage"
LOCAL_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# ── Startup health-check ─────────────────────────────────────────────────────
_r2_configured = all([R2_ENDPOINT, R2_ACCESS_KEY, R2_SECRET_KEY, R2_PUBLIC_URL])
if _r2_configured:
    logger.info(
        "✅ R2 Storage configured — uploads will be persisted to Cloudflare R2 "
        f"bucket '{R2_BUCKET}' at {R2_PUBLIC_URL}"
    )
else:
    logger.warning(
        "⚠️  R2 Storage NOT configured (R2_ENDPOINT / R2_ACCESS_KEY / R2_SECRET_KEY / "
        "R2_PUBLIC_URL not set). Falling back to local volume storage at "
        f"{LOCAL_STORAGE_DIR}. Files will survive container restarts via the "
        "Docker volume mount, but will be lost if the EC2 volume is unmounted. "
        "Set R2 credentials in .env to enable persistent cloud storage."
    )


def upload_to_r2(file_path: str, key: str) -> str:
    """
    Uploads a file to Cloudflare R2 bucket if credentials are set,
    otherwise copies the file to the local persistent outputs folder.
    Returns the public URL or local download path.
    """
    if _r2_configured:
        try:
            s3 = boto3.client(
                "s3",
                endpoint_url=R2_ENDPOINT,
                aws_access_key_id=R2_ACCESS_KEY,
                aws_secret_access_key=R2_SECRET_KEY,
                region_name="auto",
            )
            s3.upload_file(file_path, R2_BUCKET, key)
            public_url = f"{R2_PUBLIC_URL.rstrip('/')}/{key}"
            logger.info(f"R2 upload successful: {public_url}")
            return public_url
        except Exception as e:
            logger.warning(f"R2 upload failed: {e}. Falling back to local storage.")

    # Local fallback — persisted via Docker volume
    dest_path = LOCAL_STORAGE_DIR / key
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(file_path, dest_path)

    # Return path relative to the /outputs static mount for web downloads
    return f"/outputs/persistent_storage/{key}"


def upload_deliverables_to_r2(result, job_id: str) -> None:
    """
    Upload all deliverable output files (PDF, CSV, HTML, Excel, Markdown)
    from a PipelineResult to R2, then update the result's path attributes
    to point to the public R2 URLs.

    This ensures that after container restarts or cross-instance sharing,
    the download buttons still resolve to valid files.
    """
    _deliverable_fields = {
        "cleaned_csv_path":     "cleaned_data.csv",
        "pdf_report_path":      "report.pdf",
        "dashboard_html_path":  "dashboard.html",
        "excel_report_path":    "analysis_report.xlsx",
        "markdown_report_path": "report.md",
    }

    for field, default_name in _deliverable_fields.items():
        local_path = getattr(result, field, "")
        if not local_path or not Path(local_path).exists():
            continue

        # Build a unique R2 key under the job_id namespace
        ext = Path(local_path).suffix
        r2_key = f"{job_id}/{default_name}"

        try:
            public_url = upload_to_r2(local_path, r2_key)
            setattr(result, field, public_url)
            logger.info(f"Deliverable uploaded: {field} → {public_url}")
        except Exception as exc:
            logger.warning(f"Deliverable upload failed for {field}: {exc}")
            # Leave the original local path in place as fallback


def fetch_file_bytes(path: str) -> Optional[bytes]:
    """
    Given a path that is either a local file path or a remote URL,
    return the file contents as bytes.

    Used by the Streamlit download buttons to support both local files
    and R2-hosted URLs transparently.
    """
    import requests as _requests

    if path.startswith("http://") or path.startswith("https://"):
        try:
            resp = _requests.get(path, timeout=30)
            resp.raise_for_status()
            return resp.content
        except Exception as exc:
            logger.warning(f"Failed to fetch remote file {path}: {exc}")
            return None

    # Local path
    local = Path(path)
    if local.exists():
        return local.read_bytes()

    # Try the persistent storage fallback
    filename = path.split("/")[-1]
    fallback = LOCAL_STORAGE_DIR / filename
    if fallback.exists():
        return fallback.read_bytes()

    return None
