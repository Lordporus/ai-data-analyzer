"""
Centralized configuration for the AI Data Analyzer system.
All settings are read from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Server ───────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
STREAMLIT_PORT = int(os.getenv("STREAMLIT_PORT", "8501"))

# ── Limits ───────────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

# ── LLM (optional) ──────────────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "none").lower()
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_ENDPOINT = os.getenv(
    "LLM_ENDPOINT", "https://api.openai.com/v1/chat/completions"
)

def is_llm_enabled() -> bool:
    """Check if LLM is configured and enabled."""
    return LLM_PROVIDER != "none" and bool(LLM_API_KEY)

# ── Branding ─────────────────────────────────────────────────────────────
BRAND_NAME = os.getenv("BRAND_NAME", "AI Data Analyzer")
BRAND_COLOR = os.getenv("BRAND_COLOR", "#6C63FF")
BRAND_LOGO_URL = os.getenv("BRAND_LOGO_URL", "")

# ── Auth ─────────────────────────────────────────────────────────────────
JWT_SECRET = os.getenv("JWT_SECRET", "")
API_KEY_ENTERPRISE = os.getenv("API_KEY_ENTERPRISE", "")

# ── Usage tracking ───────────────────────────────────────────────────────
ENABLE_USAGE_TRACKING = os.getenv("ENABLE_USAGE_TRACKING", "false").lower() == "true"
