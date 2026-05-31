import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional
from utils.auth import _get_service_client

# Local Fallback Destination
LOCAL_DB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs", "local_db"
)
os.makedirs(LOCAL_DB_DIR, exist_ok=True)
WORKSPACE_FILE = os.path.join(LOCAL_DB_DIR, "workspace.json")


def _load_local_data() -> Dict:
    if not os.path.exists(WORKSPACE_FILE):
        return {"organizations": {}, "analysis_runs": []}
    try:
        with open(WORKSPACE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {"organizations": {}, "analysis_runs": []}


def _save_local_data(data: Dict):
    with open(WORKSPACE_FILE, "w") as f:
        json.dump(data, f, indent=4)


def create_organization(name: str) -> Dict:
    """
    Creates a new team organization. Uses Supabase (service role) if available,
    otherwise falls back to local JSON database.
    """
    client = _get_service_client()
    org_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat()

    if client:
        try:
            res = client.table("organizations").insert({
                "id": org_id,
                "name": name,
                "created_at": created_at
            }).execute()
            if res.data:
                return res.data[0]
        except Exception as e:
            print(f"Supabase Org Create Error: {e}. Using fallback.")

    # Fallback Local Storage
    data = _load_local_data()
    org = {"id": org_id, "name": name, "created_at": created_at}
    data["organizations"][org_id] = org
    _save_local_data(data)
    return org


def get_organizations() -> List[Dict]:
    """
    Fetches all available team organizations via service role client.
    """
    client = _get_service_client()
    if client:
        try:
            res = client.table("organizations").select("*").execute()
            if res.data:
                return res.data
        except Exception as e:
            print(f"Supabase Org Fetch Error: {e}. Using fallback.")

    # Fallback Local Storage
    data = _load_local_data()
    orgs = list(data["organizations"].values())
    if not orgs:
        default_org = create_organization("Default Team Workspace")
        return [default_org]
    return orgs


def add_analysis_run(
    org_id: str,
    user_id: str,
    dataset_name: str,
    status: str,
    output_path: str,
) -> Dict:
    """
    Records an analysis job run metadata via service role client.
    """
    run_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat()
    client = _get_service_client()

    run_entry = {
        "id": run_id,
        "org_id": org_id,
        "user_id": user_id,
        "dataset_name": dataset_name,
        "created_at": created_at,
        "status": status,
        "output_path": output_path,
    }

    if client:
        try:
            res = client.table("analysis_runs").insert(run_entry).execute()
            if res.data:
                return res.data[0]
        except Exception as e:
            print(f"Supabase Run Insert Error: {e}. Using fallback.")

    # Fallback Local Storage
    data = _load_local_data()
    data["analysis_runs"].append(run_entry)
    _save_local_data(data)
    return run_entry


def get_org_analysis_history(org_id: str) -> List[Dict]:
    """
    Gets the shared analysis run history for a team organization.
    """
    client = _get_service_client()
    if client:
        try:
            res = (
                client.table("analysis_runs")
                .select("*")
                .eq("org_id", org_id)
                .order("created_at", desc=True)
                .execute()
            )
            if res.data:
                return res.data
        except Exception as e:
            print(f"Supabase Run Fetch Error: {e}. Using fallback.")

    # Fallback Local Storage
    data = _load_local_data()
    runs = [run for run in data["analysis_runs"] if run["org_id"] == org_id]
    runs.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return runs
