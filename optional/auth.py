"""
Authentication stubs — Multi-user JWT auth and role-based access.

This is a stub module for SaaS-ready architecture.
Integrate with a real user database and auth provider in production.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Optional

from config.settings import JWT_SECRET


@dataclass
class User:
    user_id: str
    email: str
    role: str = "viewer"  # "admin" | "editor" | "viewer"
    org_id: str = ""


def create_token(user: User, expires_in: int = 3600) -> str:
    """Create a simple JWT-like token (stub).

    In production, use a proper JWT library (PyJWT) with RS256.
    """
    payload = {
        "sub": user.user_id,
        "email": user.email,
        "role": user.role,
        "org": user.org_id,
        "exp": int(time.time()) + expires_in,
    }
    payload_str = json.dumps(payload, sort_keys=True)
    signature = hmac.new(
        JWT_SECRET.encode(), payload_str.encode(), hashlib.sha256
    ).hexdigest()
    return f"{payload_str}|{signature}"


def verify_token(token: str) -> Optional[User]:
    """Verify a token and return the User if valid.

    Returns None if token is invalid or expired.
    """
    try:
        payload_str, signature = token.rsplit("|", 1)
        expected_sig = hmac.new(
            JWT_SECRET.encode(), payload_str.encode(), hashlib.sha256
        ).hexdigest()

        if not hmac.compare_digest(signature, expected_sig):
            return None

        payload = json.loads(payload_str)

        if payload.get("exp", 0) < time.time():
            return None

        return User(
            user_id=payload["sub"],
            email=payload["email"],
            role=payload.get("role", "viewer"),
            org_id=payload.get("org", ""),
        )
    except Exception:
        return None


def require_role(user: Optional[User], required: str) -> bool:
    """Check if a user has the required role.

    Role hierarchy: admin > editor > viewer
    """
    if user is None:
        return False

    hierarchy = {"admin": 3, "editor": 2, "viewer": 1}
    user_level = hierarchy.get(user.role, 0)
    required_level = hierarchy.get(required, 0)
    return user_level >= required_level
