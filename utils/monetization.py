from __future__ import annotations

import calendar
import hashlib
import hmac
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from config.settings import (
    APP_BASE_URL,
    RAZORPAY_KEY_ID,
    RAZORPAY_KEY_SECRET,
    RAZORPAY_PORTAL_URL,
    RAZORPAY_PRO_PLAN_ID,
    RAZORPAY_WEBHOOK_SECRET,
)
from utils.auth import _get_service_client
from utils.workspace import _load_local_data, _save_local_data

FREE_ANALYSIS_LIMIT = 5
FREE_FILE_SIZE_BYTES = 10 * 1024 * 1024
PRO_FILE_SIZE_BYTES = 50 * 1024 * 1024
FREE_NL_QUERY_DAILY_LIMIT = 3


def normalize_plan(plan: str | None) -> str:
    return "pro" if str(plan or "free").lower() == "pro" else "free"


def org_plan(org: dict | None) -> str:
    return normalize_plan((org or {}).get("plan"))


def is_pro_org(org: dict | None) -> bool:
    return org_plan(org) == "pro"


def get_month_window(now: datetime | None = None) -> tuple[str, str]:
    now = now or datetime.now(timezone.utc)
    start = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
    last_day = calendar.monthrange(now.year, now.month)[1]
    end = datetime(now.year, now.month, last_day, 23, 59, 59, tzinfo=timezone.utc)
    return start.isoformat(), end.isoformat()


def count_monthly_analyses(org_id: str, user_id: str = "") -> int:
    start, end = get_month_window()
    client = _get_service_client()
    if client:
        try:
            query = (
                client.table("analysis_runs")
                .select("id", count="exact")
                .eq("org_id", org_id)
                .gte("created_at", start)
                .lte("created_at", end)
            )
            if user_id:
                query = query.eq("user_id", user_id)
            res = query.execute()
            return int(getattr(res, "count", None) or len(res.data or []))
        except Exception as exc:
            print(f"Supabase analysis count error: {exc}. Using fallback.")

    data = _load_local_data()
    count = 0
    for run in data.get("analysis_runs", []):
        if run.get("org_id") != org_id:
            continue
        if user_id and run.get("user_id") != user_id:
            continue
        created = run.get("created_at", "")
        if start <= _coerce_utc_iso(created) <= end:
            count += 1
    return count


def can_run_analysis(org: dict | None, user_id: str = "") -> tuple[bool, str, int, int | None]:
    plan = org_plan(org)
    used = count_monthly_analyses((org or {}).get("id", "default"), user_id=user_id)
    if plan == "pro":
        return True, "Pro plan: unlimited analyses.", used, None
    if used >= FREE_ANALYSIS_LIMIT:
        return (
            False,
            f"You've reached your limit of {FREE_ANALYSIS_LIMIT} free analyses this month. Upgrade to Pro for unlimited access.",
            used,
            FREE_ANALYSIS_LIMIT,
        )
    return True, f"{FREE_ANALYSIS_LIMIT - used} free analyses remaining this month.", used, FREE_ANALYSIS_LIMIT


def can_upload_file(org: dict | None, size_bytes: int) -> tuple[bool, str]:
    limit = PRO_FILE_SIZE_BYTES if is_pro_org(org) else FREE_FILE_SIZE_BYTES
    if size_bytes <= limit:
        return True, ""
    limit_mb = limit // (1024 * 1024)
    return False, f"File exceeds the {limit_mb}MB {'Pro' if is_pro_org(org) else 'free'} limit. Upgrade to Pro to upload up to 50MB."


def update_org_plan(
    org_id: str,
    plan: str,
    razorpay_customer_id: str = "",
    razorpay_subscription_id: str = "",
) -> dict:
    plan = normalize_plan(plan)
    payload = {"plan": plan}
    if razorpay_customer_id:
        payload["razorpay_customer_id"] = razorpay_customer_id
    if razorpay_subscription_id:
        payload["razorpay_subscription_id"] = razorpay_subscription_id

    client = _get_service_client()
    if client:
        try:
            res = client.table("organizations").update(payload).eq("id", org_id).execute()
            if res.data:
                return res.data[0]
        except Exception as exc:
            print(f"Supabase org plan update error: {exc}. Using fallback.")

    data = _load_local_data()
    org = data.setdefault("organizations", {}).setdefault(org_id, {"id": org_id, "name": "Default Team Workspace"})
    org.update(payload)
    _save_local_data(data)
    return org


def create_razorpay_subscription(org_id: str, org_name: str, user_email: str) -> dict:
    if not (RAZORPAY_KEY_ID and RAZORPAY_KEY_SECRET and RAZORPAY_PRO_PLAN_ID):
        raise RuntimeError(
            "Razorpay checkout is not configured. Set RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET, and RAZORPAY_PRO_PLAN_ID."
        )

    payload = {
        "plan_id": RAZORPAY_PRO_PLAN_ID,
        "total_count": 120,
        "quantity": 1,
        "customer_notify": 1,
        "notes": {
            "org_id": org_id,
            "org_name": org_name,
            "user_email": user_email,
        },
        "notify_info": {"notify_email": user_email} if user_email else {},
    }
    res = requests.post(
        "https://api.razorpay.com/v1/subscriptions",
        auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET),
        json=payload,
        timeout=20,
    )
    if res.status_code >= 400:
        raise RuntimeError(f"Razorpay subscription creation failed: {res.text}")
    data = res.json()
    checkout_url = data.get("short_url")
    if not checkout_url:
        raise RuntimeError("Razorpay did not return a hosted subscription checkout URL.")
    return {
        "subscription_id": data.get("id", ""),
        "checkout_url": checkout_url,
        "status": data.get("status", ""),
    }


def verify_razorpay_signature(body: bytes, signature: str) -> bool:
    if not (RAZORPAY_WEBHOOK_SECRET and signature):
        return False
    expected = hmac.new(RAZORPAY_WEBHOOK_SECRET.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


def apply_razorpay_webhook(payload: dict[str, Any]) -> dict:
    event = payload.get("event", "")
    subscription = _extract_subscription_entity(payload)
    if not subscription:
        return {"updated": False, "reason": "No subscription entity in webhook."}

    notes = subscription.get("notes") or {}
    org_id = notes.get("org_id") or payload.get("org_id")
    if not org_id:
        return {"updated": False, "reason": "Missing org_id in Razorpay subscription notes."}

    subscription_id = subscription.get("id", "")
    customer_id = subscription.get("customer_id", "")
    status = str(subscription.get("status", "")).lower()

    if event in {"subscription.charged", "subscription.activated"} or status in {"active", "authenticated"}:
        org = update_org_plan(org_id, "pro", customer_id, subscription_id)
        return {"updated": True, "plan": "pro", "org": org}

    if event in {"subscription.cancelled", "subscription.completed"} or status in {"cancelled", "completed", "expired"}:
        org = update_org_plan(org_id, "free", customer_id, subscription_id)
        return {"updated": True, "plan": "free", "org": org}

    if event == "subscription.updated":
        plan = "pro" if status in {"active", "authenticated"} else "free"
        org = update_org_plan(org_id, plan, customer_id, subscription_id)
        return {"updated": True, "plan": plan, "org": org}

    return {"updated": False, "reason": f"Unhandled event/status: {event}/{status}"}


def billing_portal_url() -> str:
    return RAZORPAY_PORTAL_URL or "https://dashboard.razorpay.com/"


def upgrade_success_url(org_id: str) -> str:
    return f"{APP_BASE_URL.rstrip('/')}?upgrade=success&org_id={org_id}"


def _extract_subscription_entity(payload: dict[str, Any]) -> dict | None:
    payment = payload.get("payload") or {}
    subscription_wrapper = payment.get("subscription") or {}
    return subscription_wrapper.get("entity")


def _coerce_utc_iso(value: str) -> str:
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        return ""
