import os
import shutil
import pytest
import uuid
import json
from unittest.mock import patch, MagicMock

from utils.auth import signup_user, login_user, USERS_FILE, _hash_password
from utils.workspace import (
    create_organization,
    get_organizations,
    add_analysis_run,
    get_org_analysis_history,
    WORKSPACE_FILE
)
from utils.storage import upload_to_r2, LOCAL_STORAGE_DIR

@pytest.fixture(autouse=True)
def clean_local_db():
    """
    Cleans up any local JSON DB files or storage files before and after tests.
    """
    for file_path in [USERS_FILE, WORKSPACE_FILE]:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass
                
    if LOCAL_STORAGE_DIR.exists():
        for item in LOCAL_STORAGE_DIR.iterdir():
            if item.is_file():
                try:
                    item.unlink()
                except OSError:
                    pass
                    
    yield
    
    for file_path in [USERS_FILE, WORKSPACE_FILE]:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass
                
    if LOCAL_STORAGE_DIR.exists():
        for item in LOCAL_STORAGE_DIR.iterdir():
            if item.is_file():
                try:
                    item.unlink()
                except OSError:
                    pass

def test_auth_fallback(monkeypatch):
    """
    Tests local JSON authentication signup and login.
    """
    monkeypatch.setattr("utils.auth._get_supabase_client", lambda: None)
    monkeypatch.setattr("utils.auth.IS_PRODUCTION", False)
    
    email = f"test_{uuid.uuid4().hex[:6]}@domain.com"
    password = "securePassword123"
    
    # 1. Sign up user
    res_signup = signup_user(email, password)
    assert res_signup["email"] == email
    assert "id" in res_signup
    assert res_signup["type"] == "local"
    
    # Asserting user entry exists in file
    assert os.path.exists(USERS_FILE)
    with open(USERS_FILE, "r") as f:
        users = json.load(f)
    assert email in users
    assert users[email]["password"] == _hash_password(password)
    
    # 2. Login user
    res_login = login_user(email, password)
    assert res_login["email"] == email
    assert res_login["id"] == res_signup["id"]
    
    # 3. Failures
    with pytest.raises(Exception, match="User already exists"):
        signup_user(email, password)
        
    with pytest.raises(Exception, match="Invalid email or password"):
        login_user(email, "wrongPassword")

def test_workspace_fallback():
    """
    Tests local JSON workspace creation, org listing, and runs history.
    """
    # 1. List organizations (should auto-create default org if empty)
    orgs = get_organizations()
    assert len(orgs) == 1
    assert orgs[0]["name"] == "Default Team Workspace"
    default_org_id = orgs[0]["id"]
    
    # 2. Create organization
    custom_name = "Acme Data Corp"
    new_org = create_organization(custom_name)
    assert new_org["name"] == custom_name
    assert "id" in new_org
    
    # 3. List again - should have default + new org
    all_orgs = get_organizations()
    assert len(all_orgs) == 2
    org_ids = [org["id"] for org in all_orgs]
    assert default_org_id in org_ids
    assert new_org["id"] in org_ids
    
    # 4. Add analysis run metadata
    user_id = str(uuid.uuid4())
    dataset = "sales_q1.csv"
    run_status = "completed"
    out_path = f"/outputs/persistent_storage/{uuid.uuid4().hex}.pkl"
    
    run_entry = add_analysis_run(
        org_id=new_org["id"],
        user_id=user_id,
        dataset_name=dataset,
        status=run_status,
        output_path=out_path
    )
    assert run_entry["dataset_name"] == dataset
    assert run_entry["org_id"] == new_org["id"]
    
    # 5. Fetch org run history
    history = get_org_analysis_history(new_org["id"])
    assert len(history) == 1
    assert history[0]["id"] == run_entry["id"]
    assert history[0]["dataset_name"] == dataset

def test_storage_local_fallback(tmp_path):
    """
    Tests local storage fallback copying file into output directory.
    """
    # Create temporary dummy pipeline result file
    dummy_file = tmp_path / "test_result.pkl"
    dummy_file.write_text("dummy_content_binary_simulation")
    
    storage_key = f"test_run_{uuid.uuid4().hex[:8]}.pkl"
    
    # Upload to R2 (without credentials, falls back to local storage)
    returned_url = upload_to_r2(str(dummy_file), storage_key)
    
    # Assert return relative URL
    assert returned_url == f"/outputs/persistent_storage/{storage_key}"
    
    # Assert physical file exists in local storage
    destination_file = LOCAL_STORAGE_DIR / storage_key
    assert destination_file.exists()
    assert destination_file.read_text() == "dummy_content_binary_simulation"
