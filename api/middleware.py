"""
Middleware components — CORS, request logging, usage tracking.

Most middleware is configured directly in main.py via FastAPI's
add_middleware. This module provides any custom middleware.
"""

from __future__ import annotations

import logging
import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from config.settings import ENABLE_USAGE_TRACKING

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log every request with method, path, status, and duration."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start
        logger.info(
            "%s %s → %d (%.3fs)",
            request.method,
            request.url.path,
            response.status_code,
            duration,
        )
        return response


class UsageTrackingMiddleware(BaseHTTPMiddleware):
    """Stub for usage tracking / billing integration.

    When ENABLE_USAGE_TRACKING is True, this logs request metadata
    that could be forwarded to a billing system (Stripe, etc.).
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        if ENABLE_USAGE_TRACKING and request.url.path.startswith("/api"):
            # In production, push to a billing queue
            logger.info(
                "USAGE | %s %s | status=%d | user=%s",
                request.method,
                request.url.path,
                response.status_code,
                request.headers.get("X-User-Id", "anonymous"),
            )

        return response
