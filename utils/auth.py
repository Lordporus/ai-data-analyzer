import os
import json
import uuid
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict
from supabase import create_client, Client

SUPABASE_URL      = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")

IS_PRODUCTION = os.getenv("RENDER") == "true" or os.getenv("ENVIRONMENT") == "production"

# Fallback local directory for JSON database
_BASE_DIR = Path(__file__).resolve().parent.parent
LOCAL_DB_DIR = _BASE_DIR / "outputs" / "local_db"
LOCAL_DB_DIR.mkdir(parents=True, exist_ok=True)
USERS_FILE = LOCAL_DB_DIR / "users.json"

# ── Persistent Session Token Store ───────────────────────────────────────────
SESSION_STORE_PATH = _BASE_DIR / "auth" / "session_store.json"
SESSION_EXPIRY_DAYS = 7


class AuthError(Exception):
    """Auth failure with a stable code for UI and tests."""

    def __init__(self, message: str, code: str = "auth_error"):
        super().__init__(message)
        self.code = code


def _is_supabase_auth_configured() -> bool:
    return bool(os.getenv("SUPABASE_URL", SUPABASE_URL) and os.getenv("SUPABASE_ANON_KEY", SUPABASE_ANON_KEY))


def _is_supabase_service_configured() -> bool:
    return bool(os.getenv("SUPABASE_URL", SUPABASE_URL) and os.getenv("SUPABASE_SERVICE_KEY", SUPABASE_SERVICE_KEY))


def _is_network_error(exc: Exception) -> bool:
    err_msg = str(exc).lower()
    return any(x in err_msg for x in [
        "getaddrinfo failed",
        "connectionerror",
        "connection refused",
        "timeout",
        "socket",
        "failed to establish a new connection",
        "temporarily unavailable",
        "connection aborted",
    ]) or "socket" in type(exc).__name__.lower()


def _is_invalid_credentials_error(exc: Exception) -> bool:
    err_msg = str(exc).lower()
    return any(x in err_msg for x in [
        "invalid login credentials",
        "invalid email or password",
        "invalid credentials",
    ])


def _get_auth_client() -> Optional[Client]:
    """Anon-key client — used only for user sign-up / sign-in via Supabase Auth."""
    url = os.getenv("SUPABASE_URL", SUPABASE_URL)
    anon_key = os.getenv("SUPABASE_ANON_KEY", SUPABASE_ANON_KEY)
    if url and anon_key:
        try:
            return create_client(url, anon_key)
        except Exception:
            return None
    return None


def _get_service_client() -> Optional[Client]:
    """
    Service-role client — bypasses RLS, used for all server-side database
    operations (session_store, organizations, analysis_runs).
    """
    url = os.getenv("SUPABASE_URL", SUPABASE_URL)
    service_key = os.getenv("SUPABASE_SERVICE_KEY", SUPABASE_SERVICE_KEY)
    if url and service_key:
        try:
            return create_client(url, service_key)
        except Exception:
            return None
    return None


# Backward-compatible alias used by frontend/app.py for auth operations
def _get_supabase_client() -> Optional[Client]:
    return _get_service_client()


def get_supabase_client() -> Optional[Client]:
    """Public auth client for session restore in the Streamlit frontend."""
    return _get_auth_client()


def get_service_supabase_client() -> Optional[Client]:
    """Public service-role client for server-side DB reads (e.g. profiles table).
    Use this instead of get_supabase_client() when querying tables protected by
    RLS, because the anon client has no user JWT and will be silently blocked."""
    return _get_service_client()


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def signup_user(email: str, password: str, full_name: str = "") -> Dict:
    """
    Signs up a new user using Supabase auth if credentials are set,
    otherwise falls back to local JSON database.

    IMPORTANT: When Supabase "Confirm email" is enabled, this intentionally
    does NOT return session tokens. The caller must show a verification
    prompt instead of logging the user in immediately.
    The returned dict includes ``pending_verification=True`` so the caller
    can branch correctly.
    """
    client = _get_auth_client()
    if _is_supabase_auth_configured():
        if not client:
            raise AuthError(
                "Supabase auth is configured, but the auth client could not be created.",
                "supabase_config",
            )
        try:
            res = client.auth.sign_up({"email": email, "password": password})
            if res.user:
                # Upsert profile with full_name
                if full_name:
                    try:
                        # Use service client to bypass RLS for inserting if anon cannot
                        service_client = _get_service_client()
                        if service_client:
                            service_client.table("profiles").upsert({
                                "id": res.user.id,
                                "full_name": full_name,
                                "plan": "free"
                            }).execute()
                    except Exception as e:
                        print(f"Failed to save profile on signup: {e}")
                        
                # Detect whether Supabase requires email confirmation.
                # When confirmation is required, res.session is None and
                # email_confirmed_at is None on the freshly created user.
                needs_verification = (
                    res.session is None
                    or res.user.email_confirmed_at is None
                )
                return {
                    "id": res.user.id,
                    "email": res.user.email,
                    "type": "supabase",
                    # Do NOT expose tokens when verification is pending —
                    # the user must confirm their email before a session
                    # should be created.
                    "access_token": None if needs_verification else (
                        res.session.access_token if res.session else None
                    ),
                    "refresh_token": None if needs_verification else (
                        res.session.refresh_token if res.session else None
                    ),
                    "pending_verification": needs_verification,
                }
        except Exception as e:
            if isinstance(e, AuthError):
                raise
            if _is_network_error(e):
                raise AuthError(
                    "Could not reach Supabase during signup. Please try again in a moment.",
                    "supabase_connection",
                )
            raise AuthError(f"Supabase signup failed: {str(e)}", "supabase_signup")
        raise AuthError("Signup failed. Supabase did not return a user.", "supabase_signup")

    if IS_PRODUCTION:
        raise AuthError(
            "Production auth requires Supabase credentials (SUPABASE_URL). "
            "Local JSON fallback is disabled in production.",
            "supabase_required",
        )

    # Fallback local system
    users = {}
    if USERS_FILE.exists():
        try:
            users = json.loads(USERS_FILE.read_text())
        except Exception:
            users = {}

    if email in users:
        raise AuthError("User already exists.", "user_exists")

    user_id = str(uuid.uuid4())
    users[email] = {"id": user_id, "password": _hash_password(password)}

    USERS_FILE.write_text(json.dumps(users, indent=4))
    return {"id": user_id, "email": email, "type": "local"}


def login_user(email: str, password: str) -> Dict:
    """
    Logs in a user using Supabase auth if credentials are set,
    otherwise validates against local JSON database.

    Raises an explicit error when the user's email has not been confirmed
    (``email_confirmed_at`` is None) so that the frontend can surface a
    clear verification prompt instead of granting access.
    """
    client = _get_auth_client()
    if _is_supabase_auth_configured():
        if not client:
            raise AuthError(
                "Supabase auth is configured, but the auth client could not be created.",
                "supabase_config",
            )
        try:
            res = client.auth.sign_in_with_password({"email": email, "password": password})
            if res.user:
                # Guard: enforce email verification before granting a session.
                # Supabase populates email_confirmed_at only after the user
                # clicks the confirmation link in their inbox.
                if res.user.email_confirmed_at is None:
                    raise AuthError(
                        "Please verify your email first. "
                        "Check your inbox for a confirmation link from us.",
                        "email_unverified",
                    )
                return {
                    "id": res.user.id,
                    "email": res.user.email,
                    "type": "supabase",
                    "access_token": res.session.access_token if res.session else None,
                    "refresh_token": res.session.refresh_token if res.session else None,
                    "pending_verification": False,
                }
            raise AuthError("Invalid email or password.", "invalid_credentials")
        except Exception as e:
            if isinstance(e, AuthError):
                raise
            if _is_network_error(e):
                raise AuthError(
                    "Could not reach Supabase during login. Please try again in a moment.",
                    "supabase_connection",
                )
            if _is_invalid_credentials_error(e):
                raise AuthError("Invalid email or password.", "invalid_credentials")
            raise AuthError(f"Supabase login failed: {str(e)}", "supabase_login")

    if IS_PRODUCTION:
        raise AuthError(
            "Production auth requires Supabase credentials (SUPABASE_URL). "
            "Local JSON fallback is disabled in production.",
            "supabase_required",
        )

    # Fallback local system
    if not USERS_FILE.exists():
        raise AuthError("User not found. Please sign up first.", "user_not_found")

    users = json.loads(USERS_FILE.read_text())

    if email not in users or users[email]["password"] != _hash_password(password):
        raise AuthError("Invalid email or password.", "invalid_credentials")

    return {"id": users[email]["id"], "email": email, "type": "local"}


# ── Persistent Session Token API ──────────────────────────────────────────────

def save_session_token(user: Dict) -> str:
    token = str(uuid.uuid4())
    now = datetime.utcnow()
    expiry = (now + timedelta(days=SESSION_EXPIRY_DAYS)).isoformat()

    client = _get_service_client()
    if _is_supabase_service_configured():
        if not client:
            raise AuthError(
                "Supabase service role is configured, but the service client could not be created.",
                "supabase_service_config",
            )
        try:
            res = client.table("session_store").insert({
                "token": token,
                "user_data": user,
                "expiry": expiry
            }).execute()
            if not getattr(res, "data", None):
                raise AuthError("Supabase did not persist the session token.", "session_persist_failed")
        except Exception as e:
            if isinstance(e, AuthError):
                raise
            raise AuthError(f"Failed to save session to Supabase: {e}", "session_persist_failed")
        return token

    if _is_supabase_auth_configured() and IS_PRODUCTION:
        raise AuthError(
            "Production Supabase auth requires SUPABASE_SERVICE_KEY for persistent sessions.",
            "supabase_service_required",
        )

    if IS_PRODUCTION:
        raise AuthError(
            "Production auth requires Supabase credentials (SUPABASE_URL). "
            "Local file storage is disabled.",
            "supabase_required",
        )

    SESSION_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    store: Dict = {}
    if SESSION_STORE_PATH.exists():
        try:
            store = json.loads(SESSION_STORE_PATH.read_text())
        except Exception:
            store = {}

    store = {t: v for t, v in store.items() if _parse_dt(v.get("expiry")) > now}
    store[token] = {"user": user, "expiry": expiry}

    SESSION_STORE_PATH.write_text(json.dumps(store, indent=2))
    return token


def load_session_token(token: str) -> Optional[Dict]:
    if not token:
        return None

    client = _get_service_client()
    if _is_supabase_service_configured() and client:
        try:
            res = client.table("session_store").select("*").eq("token", token).execute()
            if res.data and len(res.data) > 0:
                entry = res.data[0]
                if _parse_dt(entry.get("expiry")) > datetime.utcnow():
                    return entry.get("user_data")
        except Exception as e:
            print(f"Failed to load session from Supabase: {e}")
        return None

    if IS_PRODUCTION:
        return None

    if not SESSION_STORE_PATH.exists():
        return None
    try:
        store = json.loads(SESSION_STORE_PATH.read_text())
        entry = store.get(token)
        if not entry:
            return None
        if _parse_dt(entry.get("expiry")) <= datetime.utcnow():
            return None
        return entry["user"]
    except Exception:
        return None


def delete_session_token(token: str) -> None:
    if not token:
        return

    client = _get_service_client()
    if _is_supabase_service_configured() and client:
        try:
            client.table("session_store").delete().eq("token", token).execute()
        except Exception:
            pass
        return

    if IS_PRODUCTION:
        return

    if not SESSION_STORE_PATH.exists():
        return
    try:
        store = json.loads(SESSION_STORE_PATH.read_text())
        store.pop(token, None)
        SESSION_STORE_PATH.write_text(json.dumps(store, indent=2))
    except Exception:
        pass


def _parse_dt(value: Optional[str]) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return datetime(1970, 1, 1)
