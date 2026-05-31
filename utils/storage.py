import os
import logging
import shutil
import boto3
from pathlib import Path
from typing import Optional

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
