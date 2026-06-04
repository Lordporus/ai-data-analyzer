from __future__ import annotations

import html
import json
import pickle
import secrets
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path regardless of who imports this module
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import APP_BASE_URL, OUTPUT_DIR

LOCAL_DB_DIR = Path(__file__).resolve().parent.parent / "outputs" / "local_db"
LOCAL_DB_DIR.mkdir(parents=True, exist_ok=True)
SHARED_REPORTS_FILE = LOCAL_DB_DIR / "shared_reports.json"
SHARED_RESULTS_DIR = Path(__file__).resolve().parent.parent / "outputs" / "shared_reports"
SHARED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PLAN_EXPIRY_DAYS = {
    "free": 7,
    "pro": 30,
    "team": 90,
    "enterprise": 365,
}


def create_share_link(result: Any, owner_user_id: str, dataset_name: str, plan: str = "free") -> dict:
    """Persist a read-only report snapshot and return share metadata."""
    token = secrets.token_urlsafe(24)
    share_id = str(uuid.uuid4())
    now = datetime.utcnow()
    expires_at = now + timedelta(days=PLAN_EXPIRY_DAYS.get(plan, 7))

    snapshot_path = SHARED_RESULTS_DIR / f"{token}.pkl"
    with snapshot_path.open("wb") as f:
        pickle.dump(result, f)

    record = {
        "id": share_id,
        "share_token": token,
        "owner_user_id": owner_user_id,
        "dataset_name": dataset_name,
        "result_path": str(snapshot_path),
        "created_at": now.isoformat(),
        "expires_at": expires_at.isoformat(),
        "is_revoked": False,
        "view_count": 0,
        "plan": plan,
    }

    data = _load_records()
    data.append(record)
    _save_records(data)
    record["url"] = f"{APP_BASE_URL.rstrip('/')}/report/{token}"
    return record


def revoke_share_link(token: str, owner_user_id: str | None = None) -> bool:
    data = _load_records()
    changed = False
    for record in data:
        if record.get("share_token") == token and (owner_user_id is None or record.get("owner_user_id") == owner_user_id):
            record["is_revoked"] = True
            changed = True
    if changed:
        _save_records(data)
    return changed


def get_shared_report(token: str) -> tuple[dict, Any]:
    data = _load_records()
    for record in data:
        if record.get("share_token") != token:
            continue
        if record.get("is_revoked"):
            raise ValueError("This report link has been revoked.")
        if _parse_dt(record.get("expires_at")) <= datetime.utcnow():
            raise ValueError("This report link has expired.")

        result_path = Path(record.get("result_path", ""))
        if not result_path.exists():
            raise FileNotFoundError("The shared report snapshot is missing.")

        record["view_count"] = int(record.get("view_count", 0)) + 1
        _save_records(data)
        with result_path.open("rb") as f:
            return record, pickle.load(f)

    raise FileNotFoundError("Shared report not found.")


def render_shared_report_html(record: dict, result: Any) -> str:
    """Render a compact read-only HTML report for public share links."""
    dataset = html.escape(record.get("dataset_name") or "Shared report")
    created_at = html.escape(record.get("created_at", "")[:19].replace("T", " "))
    expires_at = html.escape(record.get("expires_at", "")[:19].replace("T", " "))

    summary = result.summary_dict() if hasattr(result, "summary_dict") else {}
    kpis = getattr(getattr(result, "insight", None), "kpi_list", []) or []
    recs = getattr(getattr(result, "insight", None), "business_recommendations", []) or []
    exec_summary = getattr(getattr(result, "insight", None), "executive_summary", "")

    kpi_html = "".join(
        f"<div class='card'><span>{html.escape(str(getattr(k, 'name', 'Metric')))}</span>"
        f"<strong>{html.escape(str(getattr(k, 'value', '')))}</strong></div>"
        for k in kpis[:8]
    )
    rec_html = "".join(f"<li>{html.escape(str(rec))}</li>" for rec in recs[:6])

    dashboard_html = ""
    dashboard_path = Path(getattr(result, "dashboard_html_path", "") or "")
    if dashboard_path.exists():
        try:
            dashboard_html = dashboard_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            dashboard_html = ""

    downloads = []
    for label, path_attr, mime in [
        ("PDF Report", "pdf_report_path", "application/pdf"),
        ("Excel Report", "excel_report_path", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        ("Cleaned CSV", "cleaned_csv_path", "text/csv"),
    ]:
        url = _output_url(getattr(result, path_attr, ""))
        if url:
            downloads.append(f"<a class='button' href='{html.escape(url)}?share_token={token}' type='{mime}'>{html.escape(label)}</a>")

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{dataset} - Shared Report</title>
  <style>
    body {{ margin:0; background:#0d1117; color:#e6edf3; font-family:Inter,Segoe UI,Arial,sans-serif; }}
    main {{ max-width:1120px; margin:0 auto; padding:32px 20px 56px; }}
    .eyebrow {{ color:#8b949e; font-size:13px; text-transform:uppercase; letter-spacing:.08em; }}
    h1 {{ margin:8px 0 8px; font-size:36px; }}
    .meta {{ color:#8b949e; margin-bottom:24px; }}
    .notice {{ border:1px solid #30363d; background:#161b22; padding:14px 16px; border-radius:8px; margin:18px 0; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; margin:20px 0; }}
    .card {{ border:1px solid #30363d; background:#161b22; border-radius:8px; padding:14px; }}
    .card span {{ display:block; color:#8b949e; font-size:12px; margin-bottom:6px; }}
    .card strong {{ font-size:20px; }}
    .button {{ display:inline-block; margin:6px 8px 6px 0; padding:10px 14px; background:#6C63FF; color:white; text-decoration:none; border-radius:6px; }}
    .dashboard {{ margin-top:24px; background:white; color:#111; border-radius:8px; overflow:hidden; }}
    li {{ margin-bottom:8px; }}
  </style>
</head>
<body>
  <main>
    <div class="eyebrow">Read-only shared report</div>
    <h1>{dataset}</h1>
    <div class="meta">Created {created_at} · Expires {expires_at}</div>
    <div class="notice">{html.escape(str(exec_summary or 'Analysis completed successfully.'))}</div>
    <section>
      <h2>Key Metrics</h2>
      <div class="grid">{kpi_html or '<p>No KPI summary available.</p>'}</div>
    </section>
    <section>
      <h2>Recommendations</h2>
      <ul>{rec_html or '<li>No recommendations available.</li>'}</ul>
    </section>
    <section>
      <h2>Downloads</h2>
      {''.join(downloads) or '<p>No downloadable artifacts are available for this report.</p>'}
    </section>
    <section>
      <h2>Dashboard Snapshot</h2>
      <div class="dashboard">{dashboard_html or '<div style="padding:20px">Dashboard HTML was not available in this snapshot.</div>'}</div>
    </section>
  </main>
</body>
</html>"""


def _output_url(path_value: str) -> str:
    if not path_value:
        return ""
    if path_value.startswith("http://") or path_value.startswith("https://") or path_value.startswith("/api/download/"):
        return path_value
    path = Path(path_value)
    try:
        rel = path.resolve().relative_to(OUTPUT_DIR.resolve())
        return f"/api/download/{rel.as_posix()}"
    except Exception:
        return ""


def _load_records() -> list[dict]:
    if not SHARED_REPORTS_FILE.exists():
        return []
    try:
        return json.loads(SHARED_REPORTS_FILE.read_text())
    except Exception:
        return []


def _save_records(data: list[dict]) -> None:
    SHARED_REPORTS_FILE.write_text(json.dumps(data, indent=2))


def _parse_dt(value: str | None) -> datetime:
    try:
        return datetime.fromisoformat(value or "")
    except Exception:
        return datetime(1970, 1, 1)
