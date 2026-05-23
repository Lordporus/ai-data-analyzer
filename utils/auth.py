import os
import json
import uuid
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict
from supabase import create_client, Client

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")
IS_PRODUCTION = os.getenv("RENDER") == "true" or os.getenv("ENVIRONMENT") == "production"

# Fallback local directory for JSON database
_BASE_DIR = Path(__file__).resolve().parent.parent
LOCAL_DB_DIR = _BASE_DIR / "outputs" / "local_db"
LOCAL_DB_DIR.mkdir(parents=True, exist_ok=True)
USERS_FILE = LOCAL_DB_DIR / "users.json"

# ── Persistent Session Token Store ───────────────────────────────────────────
SESSION_STORE_PATH = _BASE_DIR / "auth" / "session_store.json"
SESSION_EXPIRY_DAYS = 7


def _get_supabase_client() -> Optional[Client]:
    if SUPABASE_URL and SUPABASE_ANON_KEY:
        try:
            return create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
        except Exception:
            return None
    return None

def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def signup_user(email: str, password: str) -> Dict:
    """
    Signs up a new user using Supabase auth if credentials are set,
    otherwise falls back to local JSON database.
    """
    client = _get_supabase_client()
    if client:
        try:
            res = client.auth.sign_up({"email": email, "password": password})
            if res.user:
                return {"id": res.user.id, "email": res.user.email, "type": "supabase"}
        except Exception as e:
            raise Exception(f"Supabase Signup Error: {str(e)}")
        raise Exception("Signup failed.")

    if IS_PRODUCTION or not SUPABASE_URL and IS_PRODUCTION:
        raise Exception("Production auth requires Supabase credentials (SUPABASE_URL). Local JSON fallback is disabled in production.")

    # Fallback local system
    users = {}
    if USERS_FILE.exists():
        try:
            users = json.loads(USERS_FILE.read_text())
        except Exception:
            users = {}

    if email in users:
        raise Exception("User already exists.")

    user_id = str(uuid.uuid4())
    users[email] = {"id": user_id, "password": _hash_password(password)}

    USERS_FILE.write_text(json.dumps(users, indent=4))
    return {"id": user_id, "email": email, "type": "local"}


def login_user(email: str, password: str) -> Dict:
    """
    Logs in a user using Supabase auth if credentials are set,
    otherwise validates against local JSON database.
    """
    client = _get_supabase_client()
    if client:
        try:
            res = client.auth.sign_in_with_password({"email": email, "password": password})
            if res.user:
                return {"id": res.user.id, "email": res.user.email, "type": "supabase"}
        except Exception as e:
            raise Exception(f"Supabase Login Error: {str(e)}")
        raise Exception("Invalid email or password.")

    if IS_PRODUCTION or not SUPABASE_URL and IS_PRODUCTION:
        raise Exception("Production auth requires Supabase credentials (SUPABASE_URL). Local JSON fallback is disabled in production.")

    # Fallback local system
    if not USERS_FILE.exists():
        raise Exception("User not found. Please sign up first.")

    users = json.loads(USERS_FILE.read_text())

    if email not in users or users[email]["password"] != _hash_password(password):
        raise Exception("Invalid email or password.")

    return {"id": users[email]["id"], "email": email, "type": "local"}


# ── Persistent Session Token API ──────────────────────────────────────────────

def save_session_token(user: Dict) -> str:
    token = str(uuid.uuid4())
    now = datetime.utcnow()
    expiry = (now + timedelta(days=SESSION_EXPIRY_DAYS)).isoformat()

    client = _get_supabase_client()
    if client:
        try:
            client.table("session_store").insert({
                "token": token,
                "user_data": user,
                "expiry": expiry
            }).execute()
        except Exception as e:
            print(f"Failed to save session to Supabase: {e}")
        return token

    if IS_PRODUCTION:
        raise Exception("Production auth requires Supabase credentials (SUPABASE_URL). Local file storage is disabled.")

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
        
    client = _get_supabase_client()
    if client:
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
        
    client = _get_supabase_client()
    if client:
        try:
            client.table("session_store").delete().eq("token", token).execute()
        except Exception as e:
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
