import os
import shutil
import boto3
from pathlib import Path
from typing import Optional

# R2 Environment Credentials
R2_ENDPOINT = os.getenv("R2_ENDPOINT", "")
R2_ACCESS_KEY = os.getenv("R2_ACCESS_KEY", "")
R2_SECRET_KEY = os.getenv("R2_SECRET_KEY", "")
R2_BUCKET = os.getenv("R2_BUCKET", "ai-analyzer-outputs")
R2_PUBLIC_URL = os.getenv("R2_PUBLIC_URL", "")

# Local Fallback Destination
LOCAL_STORAGE_DIR = Path(__file__).resolve().parent.parent / "outputs" / "persistent_storage"
LOCAL_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

def upload_to_r2(file_path: str, key: str) -> str:
    """
    Uploads a file to Cloudflare R2 bucket if credentials are set,
    otherwise copies the file to the local persistent outputs folder.
    Returns the public URL / local download path.
    """
    if R2_ENDPOINT and R2_ACCESS_KEY and R2_SECRET_KEY and R2_PUBLIC_URL:
        try:
            s3 = boto3.client(
                's3',
                endpoint_url=R2_ENDPOINT,
                aws_access_key_id=R2_ACCESS_KEY,
                aws_secret_access_key=R2_SECRET_KEY
            )
            s3.upload_file(file_path, R2_BUCKET, key)
            return f"{R2_PUBLIC_URL}/{key}"
        except Exception as e:
            # Fall back gracefully to local storage if AWS/R2 client fails
            print(f"R2 Upload failed: {e}. Falling back to local storage.")
            
    # Local fallback system
    dest_path = LOCAL_STORAGE_DIR / key
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(file_path, dest_path)
    
    # Return path relative to outputs mount point for web downloads
    return f"/outputs/persistent_storage/{key}"
