"""
FastAPI application — main entry point.

Run with:  uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.routes.upload import router as upload_router
from api.routes.download import router as download_router
from api.routes.enterprise import router as enterprise_router
from api.routes.status import router as status_router
from api.routes.share import router as share_router
from api.routes.razorpay_billing import router as razorpay_billing_router
from config.settings import BRAND_NAME, OUTPUT_DIR, is_llm_enabled

# ── Logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("API")
if is_llm_enabled():
    logger.info("✨ LLM-enhanced mode active")
else:
    logger.info("⚙️ Deterministic mode active (No LLM configured)")

# ── App ──────────────────────────────────────────────────────────────
app = FastAPI(
    title=f"{BRAND_NAME} API",
    description="Upload CSV files for automated AI-powered analysis, cleaning, "
                "insight generation, and report creation.",
    version="1.0.0",
)

# ── CORS ─────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files (output downloads) ─────────────────────────────────
# ── Static files (output downloads) ─────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# Removed public /outputs mount to secure generated files

# ── Routes ───────────────────────────────────────────────────────────
app.include_router(upload_router, prefix="/api", tags=["Upload & Analyze"])
app.include_router(status_router, prefix="/api", tags=["Job Status"])
app.include_router(download_router, prefix="/api", tags=["Downloads"])
app.include_router(enterprise_router, prefix="/api/v1", tags=["Enterprise API"])
app.include_router(share_router, tags=["Shared Reports"])
app.include_router(razorpay_billing_router, prefix="/api", tags=["Razorpay Billing"])


@app.get("/", tags=["Health"])
async def health_check():
    """Health-check endpoint."""
    return {"status": "healthy", "service": BRAND_NAME}


@app.get("/api/mock-stream", tags=["Mock Stream"])
async def mock_stream():
    """Generates random mock time-series sales data for real-time analysis testing."""
    import random
    from datetime import datetime, timedelta
    now = datetime.now()
    records = []
    # Generate 50 points of historical data
    for i in range(50):
        timestamp = (now - timedelta(minutes=(50 - i) * 30)).strftime("%Y-%m-%d %H:%M:%S")
        records.append({
            "Date": timestamp,
            "Sales": round(100 + i * 2.5 + random.uniform(-15, 15), 2),
            "Quantity": random.randint(1, 10),
            "Store_ID": random.choice([1, 2, 3]),
            "Category": random.choice(["Electronics", "Apparel", "Home"])
        })
    return records
