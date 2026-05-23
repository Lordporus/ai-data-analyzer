"""
AI Data Analyzer — Streamlit Frontend

A premium, interactive UI for uploading CSV files and viewing
analysis results. Supports white-label branding via sidebar config.

Run with:  streamlit run frontend/app.py --server.port 8501
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import requests
import pickle

from config.settings import BRAND_NAME, BRAND_COLOR, OUTPUT_DIR, UPLOAD_DIR, is_llm_enabled, SCHEDULER_ENABLED
from orchestrator.master import MasterOrchestrator, PipelineResult
from agents.data_quality import score_color, risk_level
from utils.auth import login_user, signup_user, save_session_token, load_session_token, delete_session_token, get_supabase_client
from utils.workspace import create_organization, get_organizations, add_analysis_run, get_org_analysis_history
from utils.storage import upload_to_r2, LOCAL_STORAGE_DIR

def load_analysis_result_from_storage(output_path: str):
    """
    Downloads or reads the pickled pipeline result.
    If it is a URL (Supabase/R2), downloads via HTTP.
    If it is a local path (outputs/persistent_storage/key), loads locally.
    """
    if output_path.startswith("http://") or output_path.startswith("https://"):
        res = requests.get(output_path)
        res.raise_for_status()
        return pickle.loads(res.content)
    else:
        # Local storage fallback
        filename = output_path.split("/")[-1]
        local_path = LOCAL_STORAGE_DIR / filename
        if local_path.exists():
            with open(local_path, "rb") as f:
                return pickle.load(f)
        else:
            raise FileNotFoundError(f"Local persistent analysis file not found: {local_path}")

def save_to_history(dataset_name: str, result_summary):
    """
    Saves the completed analysis run into the session state history (keeps last 5).
    """
    import uuid
    if "history" not in st.session_state:
        st.session_state["history"] = []
    
    # Calculate stats
    insights_count = 0
    top_insight = "N/A"
    if result_summary and result_summary.insight:
        recs = result_summary.insight.business_recommendations or []
        insights_count = len(recs)
        if recs:
            top_insight = recs[0]
            
    run = {
        "id": str(uuid.uuid4()),
        "dataset": dataset_name,
        "timestamp": datetime.now().strftime("%d %b %Y, %I:%M %p"),
        "insights_count": insights_count,
        "top_insight": top_insight,
        "result": result_summary
    }
    
    # Prepend and limit to 5
    st.session_state["history"] = [run] + [r for r in st.session_state["history"] if r["id"] != run["id"]][:4]

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title=f"{BRAND_NAME}",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── UI-2: Landing Page (before login) ────────────────────────────────
def show_landing_page():
    """Marketing landing page shown before the login gate."""
    st.markdown("""
    <style>
        .landing-hero {
            text-align: center;
            padding: 48px 0 32px 0;
        }
        .landing-hero h2 {
            font-size: 2.4rem;
            font-weight: 700;
            background: linear-gradient(135deg, #6C63FF, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 12px;
        }
        .landing-hero p {
            font-size: 1.15rem;
            color: #8b949e;
            max-width: 560px;
            margin: 0 auto 32px auto;
        }
        .landing-proof {
            text-align: center;
            color: #8b949e;
            font-size: 0.95rem;
            margin: 16px 0 24px 0;
        }
        @media (prefers-reduced-motion: reduce) {
            .landing-hero * { animation: none !important; }
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="landing-hero">
        <h2>🤖 AI Data Analyzer</h2>
        <p>Upload any CSV or Excel file. Get a full AI-powered business analysis in 15 seconds.</p>
    </div>
    """, unsafe_allow_html=True)

    # 3 feature cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📊 **Auto Insights**\nTrends, anomalies, correlations detected automatically")
    with col2:
        st.info("📈 **Forecasting**\nPredict next 10 periods with confidence intervals")
    with col3:
        st.info("📄 **PDF Report**\nDownload a branded executive report instantly")

    # Social proof
    st.markdown("---")
    st.markdown(
        "<div class='landing-proof'>✨ <strong>Used by 500+ analysts, consultants, and business owners</strong></div>",
        unsafe_allow_html=True,
    )

    # CTA
    if st.button("🚀 Try Free — No Credit Card", type="primary", use_container_width=True, key="landing_cta_btn"):
        st.session_state["show_auth"] = True
        st.rerun()


# ── UI-2: Load Sample Dataset ─────────────────────────────────────────
def load_sample_dataset():
    """Load the built-in Walmart sample CSV into session state for demo."""
    sample_path = ROOT / "data" / "sample_walmart.csv"
    if sample_path.exists():
        st.session_state["sample_dataset_path"] = str(sample_path)
        st.session_state["sample_dataset_name"] = "sample_walmart.csv"
        st.rerun()
    else:
        st.error("Sample dataset not found. Please contact support.")

# ── Persistent Session Restore (BUG 2 fix) ──────────────────────────────
# Runs on every page load BEFORE the login gate. If the URL carries a valid
# session token (written to st.query_params on login/signup), we restore the
# user into session_state so they are never asked to log in again within the
# 7-day window — even after a browser refresh or Streamlit WebSocket timeout.
if "user" not in st.session_state:
    _saved_token = st.query_params.get("session", "")
    if _saved_token:
        _restored_user = None
        _supabase_client = get_supabase_client()
        if _supabase_client and ":::" in _saved_token:
            try:
                _parts = _saved_token.split(":::")
                _access_token = _parts[0]
                _refresh_token = _parts[1] if len(_parts) > 1 else None
                
                # Try getting the user using the current access token
                _user_res = _supabase_client.auth.get_user(_access_token)
                if _user_res and _user_res.user:
                    _restored_user = {
                        "id": _user_res.user.id,
                        "email": _user_res.user.email,
                        "type": "supabase",
                        "access_token": _access_token,
                        "refresh_token": _refresh_token
                    }
                elif _refresh_token:
                    # Access token might have expired, try refreshing session
                    _refresh_res = _supabase_client.auth.refresh_session(_refresh_token)
                    if _refresh_res and _refresh_res.user and _refresh_res.session:
                        _new_access_token = _refresh_res.session.access_token
                        _new_refresh_token = _refresh_res.session.refresh_token
                        _new_token_str = f"{_new_access_token}:::{_new_refresh_token}"
                        st.query_params["session"] = _new_token_str
                        _saved_token = _new_token_str
                        _restored_user = {
                            "id": _refresh_res.user.id,
                            "email": _refresh_res.user.email,
                            "type": "supabase",
                            "access_token": _new_access_token,
                            "refresh_token": _new_refresh_token
                        }
            except Exception:
                pass
        
        # Fallback to local session token restore if not Supabase or Supabase check failed
        if not _restored_user:
            _restored_user = load_session_token(_saved_token)
            
        if _restored_user:
            st.session_state["user"] = _restored_user
            st.session_state["_session_token"] = _saved_token

# ── UI-2: Landing Page Gate (before login) ───────────────────────────
if "user" not in st.session_state and not st.session_state.get("show_auth", False):
    show_landing_page()
    st.stop()

# ── Team Workspace Login Gate ─────────────────────────────────────────
if "user" not in st.session_state:
    st.markdown("""
    <div style='text-align: center; margin-top: 40px;'>
        <h2 style='font-size: 2.2rem; font-weight: 700; color: #a78bfa; margin-bottom: 8px;'>🔒 Team Workspace Auth</h2>
        <p style='color: #8b949e;'>Sign in or create a team account to access shared analysis history.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        auth_mode = st.tabs(["🔑 Sign In", "📝 Sign Up"])
        
        with auth_mode[0]:
            login_email = st.text_input("Email Address", key="login_email_input")
            login_pass = st.text_input("Password", type="password", key="login_pass_input")
            if st.button("🔑 Sign In Now", type="primary", use_container_width=True):
                _login_ok = False
                try:
                    user = login_user(login_email, login_pass)
                    st.session_state["user"] = user
                    
                    if user.get("type") == "supabase" and user.get("access_token") and user.get("refresh_token"):
                        _token = f"{user['access_token']}:::{user['refresh_token']}"
                        st.session_state["_session_token"] = _token
                        st.query_params["session"] = _token
                    else:
                        # Save persistent session token (BUG 2 fix)
                        _token = save_session_token(user)
                        st.session_state["_session_token"] = _token
                        st.query_params["session"] = _token
                        
                    st.success("🎉 Welcome back! Logging you in...")
                    time.sleep(1)
                    _login_ok = True
                except Exception as e:
                    st.error(f"❌ {str(e)}")
                if _login_ok:
                    st.rerun()
                    
        with auth_mode[1]:
            signup_email = st.text_input("Email Address", key="signup_email_input")
            signup_pass = st.text_input("Password", type="password", key="signup_pass_input")
            signup_pass_conf = st.text_input("Confirm Password", type="password", key="signup_pass_conf")
            if st.button("📝 Sign Up Now", type="primary", use_container_width=True):
                if signup_pass != signup_pass_conf:
                    st.error("❌ Passwords do not match.")
                else:
                    _signup_ok = False
                    try:
                        user = signup_user(signup_email, signup_pass)
                        st.session_state["user"] = user
                        
                        if user.get("type") == "supabase" and user.get("access_token") and user.get("refresh_token"):
                            _token = f"{user['access_token']}:::{user['refresh_token']}"
                            st.session_state["_session_token"] = _token
                            st.query_params["session"] = _token
                        else:
                            # Save persistent session token (BUG 2 fix)
                            _token = save_session_token(user)
                            st.session_state["_session_token"] = _token
                            st.query_params["session"] = _token
                            
                        st.success("🎉 Account created successfully! Logging you in...")
                        time.sleep(1)
                        _signup_ok = True
                    except Exception as e:
                        st.error(f"❌ {str(e)}")
                    if _signup_ok:
                        st.rerun()
    st.stop()

# Ensure an active organization is loaded in session state
if "active_org" not in st.session_state:
    orgs = get_organizations()
    if orgs:
        st.session_state["active_org"] = orgs[0]

# ── Sidebar — Branding & Team Workspaces ─────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="padding:12px 0 8px 0;">'
        '<strong style="font-size:1.1rem; color:#e6edf3;">AI Data Analyzer</strong><br>'
        '<span style="font-size:0.8rem; color:#8b949e;">Enterprise Data Intelligence Platform</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    
    # ── User Account Info & Logout ────────────────────────────────────
    user_info = st.session_state.get("user", {})
    st.markdown(f"👤 **{user_info.get('email', 'User')}**")
    if st.button("🚪 Log Out", type="secondary", use_container_width=True):
        # Revoke persistent session token (BUG 2 fix)
        _logout_token = st.session_state.get("_session_token", "") or st.query_params.get("session", "")
        if _logout_token:
            _supabase_client = get_supabase_client()
            if _supabase_client and ":::" in _logout_token:
                try:
                    _supabase_client.auth.sign_out()
                except Exception:
                    pass
            else:
                delete_session_token(_logout_token)
        if "session" in st.query_params:
            del st.query_params["session"]
        st.session_state.clear()
        st.success("Logged out successfully!")
        time.sleep(0.5)
        st.rerun()
        
    st.markdown("---")
    
    # Collapsible groups for Sidebar settings
    with st.expander("🏢 Team Workspace", expanded=False):
        # ── Workspace Switcher & Creator ─────────────────────────────────
        orgs = get_organizations()
        org_names = [org["name"] for org in orgs]
        current_idx = 0
        if "active_org" in st.session_state:
            for idx, org in enumerate(orgs):
                if org["id"] == st.session_state["active_org"]["id"]:
                    current_idx = idx
                    break
                    
        selected_org_name = st.selectbox(
            "Switch Workspace",
            options=org_names,
            index=current_idx,
            label_visibility="collapsed"
        )
        
        for org in orgs:
            if org["name"] == selected_org_name:
                if "active_org" not in st.session_state or st.session_state["active_org"]["id"] != org["id"]:
                    st.session_state["active_org"] = org
                    st.rerun()
                    
        st.markdown("**Create Workspace**")
        new_org_name = st.text_input("Workspace Name", key="new_org_name_input", placeholder="New team name...")
        if st.button("Create", use_container_width=True):
            if new_org_name.strip():
                new_org = create_organization(new_org_name.strip())
                st.session_state["active_org"] = new_org
                st.success(f"Workspace '{new_org_name}' created!")
                time.sleep(0.5)
                st.rerun()
            else:
                st.error("Name cannot be empty.")
    
    with st.expander("🎨 Report Branding", expanded=False):
        company_name = st.text_input("Company Name", value="AI Data Analyzer")
        brand_color = st.color_picker("Brand Color", value="#6C63FF")
        logo_file = st.file_uploader("Upload Logo (PNG)", type=["png"])
        analyst_name = st.text_input("Analyst Name (optional)")
        brand_name = company_name

    with st.expander("📋 Recent Analyses", expanded=True):
        recent_history = st.session_state.get("history", [])
        if not recent_history:
            st.info("No analyses run in this session yet.")
        else:
            for run in recent_history:
                with st.container():
                    st.markdown(
                        f"""
                        <div style='background-color: #161b22; border: 1px solid #30363d; padding: 10px; border-radius: 6px; margin-bottom: 8px;'>
                            <strong style='color: #a78bfa;'>📊 {run['dataset']}</strong><br/>
                            <span style='font-size: 0.8rem; color: #8b949e;'>🕐 {run['timestamp']}</span><br/>
                            <span style='font-size: 0.8rem; color: #8b949e;'>💡 {run['insights_count']} insights found</span>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    if st.button("🔄 Load", key=f"session_run_btn_{run['id']}", use_container_width=True):
                        st.session_state["analysis_result"] = run["result"]
                        st.session_state["analysis_complete"] = True
                        st.success("Loaded from session history!")
                        time.sleep(0.5)
                        st.rerun()

    with st.expander("📜 Shared History", expanded=False):
        if "active_org" in st.session_state:
            active_org = st.session_state["active_org"]
            history = get_org_analysis_history(active_org["id"])
            if not history:
                st.info("No shared analysis runs in this workspace yet.")
            else:
                for run in history:
                    date_str = run.get("created_at", "")
                    try:
                        dt = datetime.fromisoformat(date_str)
                        formatted_date = dt.strftime("%b %d, %H:%M")
                    except Exception:
                        formatted_date = date_str[:16]
                    
                    run_label = f"📊 {run.get('dataset_name', 'Dataset')} ({formatted_date})"
                    if st.button(run_label, key=f"run_btn_{run['id']}", use_container_width=True):
                        with st.spinner("Loading analysis..."):
                            _load_ok = False
                            try:
                                loaded_result = load_analysis_result_from_storage(run["output_path"])
                                st.session_state["analysis_result"] = loaded_result
                                st.session_state["analysis_complete"] = True
                                save_to_history(run.get("dataset_name", "Dataset"), loaded_result)
                                st.success("Loaded!")
                                time.sleep(0.5)
                                _load_ok = True
                            except Exception as e:
                                st.error(f"Failed to load run: {e}")
                        if _load_ok:
                            st.rerun()

    with st.expander("ℹ️ About & System", expanded=False):
        # LLM Status Indicator
        if is_llm_enabled():
            st.success("AI Narrative Mode", icon="✅")
            st.caption("AI-enhanced executive explanations.")
        else:
            st.info("Deterministic Mode", icon="ℹ️")
            st.caption("AI narrative mode is off.")
            
        st.markdown(
            "Upload a CSV file and get:\n"
            "- Automated cleaning\n"
            "- Intelligent repair\n"
            "- Statistical insights\n"
            "- Interactive dashboard\n"
            "- PDF report"
        )
        st.caption("\u00a9 2026 AI Data Analyzer")

# ── Custom CSS ───────────────────────────────────────────────────────
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    @media (max-width: 768px) {{
        .css-1d391kg {{ width: 100% !important; }}
        .stButton button {{ width: 100% !important; }}
    }}

    :root {{
        --brand: {brand_color};
        --spacing: 8px;
    }}

    .stApp {{
        font-family: 'Inter', sans-serif;
    }}

    .main-header {{
        text-align: center;
        padding: calc(var(--spacing) * 3) 0;
        margin-bottom: calc(var(--spacing) * 3);
    }}
    .main-header h1 {{
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, {brand_color}, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }}
    .main-header p {{
        color: #8b949e;
        font-size: 1.1rem;
        margin-top: 8px;
    }}

    .kpi-card {{
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: transform 0.2s;
    }}
    .kpi-card:hover {{
        transform: translateY(-2px);
    }}
    .kpi-card .label {{
        font-size: 0.75rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    .kpi-card .value {{
        font-size: 1.8rem;
        font-weight: 700;
        color: {brand_color};
        margin-top: 4px;
    }}

    .step-indicator {{
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 16px;
        background: #161b22;
        border-radius: 8px;
        margin: 4px 0;
        border-left: 3px solid {brand_color};
    }}

    @media (prefers-reduced-motion: reduce) {{
        .kpi-card {{ transition: none; }}
        .health-card {{ transition: none; }}
    }}

    .health-card {{
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
        transition: transform 0.2s;
        border-top: 3px solid var(--brand);
    }}
    .health-card:hover {{
        transform: translateY(-2px);
    }}
    .health-card .h-label {{
        font-size: 0.7rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 0;
    }}
    .health-card .h-value {{
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 4px;
    }}
    .health-section {{
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
        border: 1px solid #30363d;
        border-radius: 16px;
        padding: calc(var(--spacing) * 3);
        margin-bottom: calc(var(--spacing) * 3);
    }}
</style>
""", unsafe_allow_html=True)


# ── Header ───────────────────────────────────────────────────────────
st.markdown(f"""
<section class="main-header">
    <h1>🔬 {brand_name}</h1>
    <p>AI-Powered Data Analysis • Clean • Analyze • Report</p>
</section>
""", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────────
tab_upload, tab_compare, tab_stream = st.tabs([
    "📤 Single File Analysis", 
    "🔀 Multi-File Comparison", 
    "📡 Live Stream Analysis"
])

# ═══════════════════════════════════════════════════════════════════════
# TAB 1 — Single File Analysis
# ═══════════════════════════════════════════════════════════════════════
with tab_upload:
    st.markdown("### 📥 Ingest Dataset")

    # Initialize Session State
    if "analysis_complete" not in st.session_state:
        st.session_state["analysis_complete"] = False
    if "analysis_result" not in st.session_state:
        st.session_state["analysis_result"] = None
    if "db_dataframe" not in st.session_state:
        st.session_state["db_dataframe"] = None
    if "ingestion_method" not in st.session_state:
        st.session_state["ingestion_method"] = "file"

    input_ready = False
    temp_path = None
    temp_dir = OUTPUT_DIR / "_temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    # ── UI-2: Handle sample dataset loaded from empty state button ───────
    if st.session_state.get("sample_dataset_path"):
        _sample_path = Path(st.session_state["sample_dataset_path"])
        if _sample_path.exists():
            temp_path = temp_dir / st.session_state.get("sample_dataset_name", "sample_walmart.csv")
            import shutil as _shutil
            _shutil.copy2(_sample_path, temp_path)
            st.session_state["ingestion_method"] = "file"
            input_ready = True
            st.session_state.pop("sample_dataset_path", None)
            st.session_state.pop("sample_dataset_name", None)
            st.success("✅ Sample Walmart dataset loaded! Click **Run Full Analysis** below to start.")

    # ── UI-2: Empty State (no file uploaded yet, no analysis running) ─────
    if (
        not input_ready
        and not st.session_state.get("analysis_complete")
        and not st.session_state.get("analysis_result")
    ):
        st.markdown("### 👋 Welcome! What would you like to analyze today?")

        empty_col1, empty_col2 = st.columns(2)
        with empty_col1:
            st.markdown("**What you'll get:**")
            st.markdown("✅ Revenue trends & forecasts")
            st.markdown("✅ Anomaly detection")
            st.markdown("✅ Correlation analysis")
            st.markdown("✅ Downloadable PDF report")
        with empty_col2:
            if st.button(
                "📂 Try with Sample Data (Walmart Sales)",
                use_container_width=True,
                key="empty_state_sample_btn",
            ):
                load_sample_dataset()

        st.markdown("---")

    # Sub-tabs for Ingestion Source
    source_tab_file, source_tab_db = st.tabs(["📄 Upload File / Google Sheet", "🎛️ Connect Database"])

    with source_tab_file:
        uploaded_file = st.file_uploader(
            "Upload your CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            help="Maximum file size: 50 MB",
        )
        
        st.markdown("<div style='text-align: center; margin: 10px 0; color: #8b949e;'>— OR —</div>", unsafe_allow_html=True)
        
        gsheets_url = st.text_input(
            "Paste Google Sheets URL",
            placeholder="https://docs.google.com/spreadsheets/d/...",
            help="Make sure the sheet is public or service account has access."
        )

        if uploaded_file is not None:
            temp_path = temp_dir / uploaded_file.name
            temp_path.write_bytes(uploaded_file.getvalue())
            st.success(f"✅ Uploaded: **{uploaded_file.name}** ({len(uploaded_file.getvalue()) / 1024:.1f} KB)")
            st.session_state["ingestion_method"] = "file"
            input_ready = True
            
        elif gsheets_url:
            try:
                with st.spinner("Fetching Google Sheet data..."):
                    from utils.gsheets import load_google_sheet
                    df = load_google_sheet(gsheets_url)
                    if df.empty:
                        st.error("❌ Google Sheet is empty or could not be parsed.")
                    else:
                        temp_path = temp_dir / "google_sheet_download.csv"
                        df.to_csv(temp_path, index=False)
                        st.success("✅ Connected to Google Sheet successfully!")
                        st.session_state["ingestion_method"] = "file"
                        input_ready = True
            except Exception as e:
                st.error(f"❌ Google Sheets Ingestion Error: {str(e)}")

    with source_tab_db:
        st.markdown("#### 🔌 Connect to a Live Database")
        db_type = st.selectbox("Database Type", ["PostgreSQL", "Google BigQuery", "Snowflake"])
        
        if db_type == "PostgreSQL":
            col1, col2 = st.columns(2)
            with col1:
                pg_host = st.text_input("Host", value="localhost", key="pg_host")
                pg_port = st.text_input("Port", value="5432", key="pg_port")
                pg_db = st.text_input("Database Name", value="postgres", key="pg_db")
            with col2:
                pg_user = st.text_input("Username", value="postgres", key="pg_user")
                pg_pass = st.text_input("Password", type="password", key="pg_pass")
            pg_query = st.text_area("SQL Query", value="SELECT * FROM my_table LIMIT 1000", key="pg_query")
            
            if st.button("🔌 Connect & Query PostgreSQL", key="pg_btn"):
                with st.spinner("Executing query..."):
                    try:
                        from utils.db_connector import connect_postgres
                        db_df = connect_postgres(
                            host=pg_host,
                            port=int(pg_port),
                            db=pg_db,
                            user=pg_user,
                            password=pg_pass,
                            query=pg_query
                        )
                        if db_df.empty:
                            st.warning("⚠️ Query returned 0 rows.")
                        else:
                            st.session_state["db_dataframe"] = db_df
                            st.session_state["ingestion_method"] = "db"
                            st.success(f"✅ Loaded {len(db_df)} rows successfully!")
                    except Exception as e:
                        st.error(f"❌ PostgreSQL Connection Error: {str(e)}")
                        
        elif db_type == "Google BigQuery":
            bq_project = st.text_input("BigQuery Project ID", key="bq_proj")
            bq_creds_path = st.text_input("Service Account JSON Path (Optional)", help="Leave blank if local/env authentication is configured.", key="bq_creds")
            bq_query = st.text_area("SQL Query", value="SELECT * FROM `project.dataset.table` LIMIT 1000", key="bq_query")
            
            if st.button("🔌 Connect & Query BigQuery", key="bq_btn"):
                with st.spinner("Executing BigQuery job..."):
                    try:
                        from utils.db_connector import connect_bigquery
                        db_df = connect_bigquery(
                            project_id=bq_project,
                            query=bq_query,
                            credentials_json_path=bq_creds_path if bq_creds_path else None
                        )
                        if db_df.empty:
                            st.warning("⚠️ Query returned 0 rows.")
                        else:
                            st.session_state["db_dataframe"] = db_df
                            st.session_state["ingestion_method"] = "db"
                            st.success(f"✅ Loaded {len(db_df)} rows successfully!")
                    except Exception as e:
                        st.error(f"❌ BigQuery Connection Error: {str(e)}")
                        
        elif db_type == "Snowflake":
            col1, col2 = st.columns(2)
            with col1:
                sf_account = st.text_input("Account Identifier", placeholder="xy12345.west-us-2", key="sf_acc")
                sf_user = st.text_input("Username", key="sf_user")
                sf_pass = st.text_input("Password", type="password", key="sf_pass")
            with col2:
                sf_db = st.text_input("Database Name", key="sf_db")
                sf_schema = st.text_input("Schema Name", key="sf_schema")
                sf_wh = st.text_input("Warehouse Name", key="sf_wh")
            sf_query = st.text_area("SQL Query", value="SELECT * FROM my_table LIMIT 1000", key="sf_query")
            
            if st.button("🔌 Connect & Query Snowflake", key="sf_btn"):
                with st.spinner("Executing Snowflake query..."):
                    try:
                        from utils.db_connector import connect_snowflake
                        db_df = connect_snowflake(
                            account=sf_account,
                            user=sf_user,
                            password=sf_pass,
                            database=sf_db,
                            schema=sf_schema,
                            warehouse=sf_wh,
                            query=sf_query
                        )
                        if db_df.empty:
                            st.warning("⚠️ Query returned 0 rows.")
                        else:
                            st.session_state["db_dataframe"] = db_df
                            st.session_state["ingestion_method"] = "db"
                            st.success(f"✅ Loaded {len(db_df)} rows successfully!")
                    except Exception as e:
                        st.error(f"❌ Snowflake Connection Error: {str(e)}")

    # Handle dataset setup based on Ingestion Method
    if st.session_state.get("ingestion_method") == "db" and st.session_state.get("db_dataframe") is not None:
        temp_path = temp_dir / "database_query_result.csv"
        st.session_state["db_dataframe"].to_csv(temp_path, index=False)
        input_ready = True

    if input_ready and temp_path is not None:
        # Show preview
        with st.expander("🔍 Raw Data Preview", expanded=False):
            ext = temp_path.suffix.lower()
            if ext in (".xlsx", ".xls"):
                preview_df = pd.read_excel(temp_path, nrows=10, engine="openpyxl")
            else:
                preview_df = pd.read_csv(temp_path, nrows=10)
            st.dataframe(preview_df, use_container_width=True)

        # ── EXECUTION BLOCK ──────────────────────────────────────────────
        st.caption("⏱️ Estimated time: 10-15 seconds for datasets under 10MB")
        run_clicked = st.button("🚀 Run Full Analysis", type="primary", use_container_width=True)
        
        if run_clicked:
            # Clear previous state
            st.session_state["analysis_result"] = None
            st.session_state["analysis_complete"] = False
            
            job_output = OUTPUT_DIR / f"streamlit_{int(time.time())}"

            with st.status("🔬 Running AI Analysis...", expanded=True) as status:
                # Stage 1 — progress bar init; _sync_progress will write the label
                progress_bar = st.progress(0, text="Starting pipeline...")
                progress_bar.progress(0.05, text="📥 Ingesting data...")

                # Run the actual pipeline
                orchestrator = MasterOrchestrator()

                # Save brand logo PNG if uploaded
                logo_path = None
                if logo_file is not None:
                    import uuid
                    logo_suffix = Path(logo_file.name).suffix
                    temp_logo_path = UPLOAD_DIR / f"logo_{uuid.uuid4().hex[:8]}{logo_suffix}"
                    temp_logo_path.write_bytes(logo_file.getvalue())
                    logo_path = str(temp_logo_path)

                branding_config = {
                    "company_name": company_name,
                    "primary_color": brand_color,
                    "logo_path": logo_path,
                    "footer_text": f"Confidential — Generated by {company_name}",
                    "analyst_name": analyst_name
                }

                # ── Auto-detect Redis and choose Celery vs synchronous execution ──
                import pickle
                import uuid as _uuid
                from utils.task_queue import is_redis_available, run_analysis_sync

                _use_celery = False
                try:
                    if is_redis_available():
                        from utils.task_queue import run_analysis_task
                        if run_analysis_task is not None:
                            _use_celery = True
                except Exception:
                    _use_celery = False

                # Map Celery PROGRESS stage names → display labels and progress %
                _stage_map = {
                    "ingestion":    ("📥 Ingesting data...",              0.15),
                    "quality":      ("🩺 Assessing data quality...",       0.30),
                    "cleaning":     ("🧹 Cleaning & deduplicating...",     0.45),
                    "repair":       ("🔧 Applying intelligent repairs...", 0.55),
                    "re-quality":   ("🩺 Re-assessing quality...",         0.60),
                    "insights":     ("🔍 Detecting insights...",           0.70),
                    "forecasting":  ("📈 Generating forecasts...",         0.80),
                    "report":       ("📄 Building PDF report...",          0.90),
                }
                _last_stage = [""]  # mutable container so nested functions can mutate without nonlocal

                try:
                    if _use_celery:
                        # ── Celery path (Redis is running) ────────────────────────
                        task = run_analysis_task.delay(
                            str(temp_path),
                            str(job_output),
                            branding=branding_config,
                            dataset_name=uploaded_file.name if uploaded_file else temp_path.name,
                        )

                        # Poll task progress and update st.status live
                        while True:
                            task_state = task.state
                            if task_state == "PROGRESS":
                                meta = task.info or {}
                                pct = meta.get("pct", 0.05)
                                stage_key = meta.get("stage", "").lower()
                                if stage_key != _last_stage[0]:
                                    _last_stage[0] = stage_key
                                    label, _ = _stage_map.get(stage_key, (f"⏳ {meta.get('stage', 'Processing...')}", pct))
                                    st.write(label)
                                progress_bar.progress(pct, text=f"⏳ {meta.get('stage', 'Processing...')}")
                            elif task_state == "SUCCESS":
                                result_dict = task.result or {}
                                if result_dict.get("status") == "failed":
                                    errors = result_dict.get("errors", ["Pipeline failed."])
                                    raise Exception(" | ".join(errors))

                                pickle_path = result_dict.get("pickle_path")
                                if pickle_path and Path(pickle_path).exists():
                                    with open(pickle_path, "rb") as f:
                                        result = pickle.load(f)
                                else:
                                    raise Exception("Pipeline completed but results are missing from disk.")
                                break
                            elif task_state == "FAILURE":
                                raise Exception(str(task.result) if task.result else "Celery execution failed.")

                            time.sleep(0.3)

                    else:
                        # ── Synchronous fallback (no Redis / no Celery worker) ────
                        st.info("⚡ Running in direct mode (Redis not detected — using in-process execution)", icon="ℹ️")

                        def _sync_progress(stage: str, pct: float):
                            stage_key = stage.lower().split()[0] if stage else ""
                            if stage_key != _last_stage[0]:
                                _last_stage[0] = stage_key
                                label, _ = _stage_map.get(stage_key, (f"⏳ {stage}", pct))
                                st.write(label)
                            progress_bar.progress(pct, text=f"⏳ {stage}")

                        # Write stage 2 manually since sync runs ingestion first
                        st.write("🧹 Cleaning & validating...")
                        result_dict = run_analysis_sync(
                            str(temp_path),
                            str(job_output),
                            branding=branding_config,
                            progress_callback=_sync_progress,
                            dataset_name=uploaded_file.name if uploaded_file else temp_path.name,
                        )

                        if result_dict.get("status") == "failed":
                            errors = result_dict.get("errors", ["Pipeline failed."])
                            raise Exception(" | ".join(errors))

                        pickle_path = result_dict.get("pickle_path")
                        if pickle_path and Path(pickle_path).exists():
                            with open(pickle_path, "rb") as f:
                                result = pickle.load(f)
                        else:
                            raise Exception("Pipeline completed but results are missing from disk.")

                    # ── Write final stages ────────────────────────────────────
                    st.write("🔍 Detecting insights...")
                    st.write("📈 Generating forecasts...")
                    st.write("📄 Building PDF report...")
                    progress_bar.progress(1.0, text="✅ Complete!")

                    # ── Persist analysis run in workspace (both paths) ────────────
                    run_id = result.job_id or str(_uuid.uuid4())
                    storage_key = f"{run_id}_pipeline_result.pkl"
                    try:
                        from utils.storage import upload_to_r2
                        from utils.workspace import add_analysis_run
                        public_url = upload_to_r2(pickle_path, storage_key)
                        active_org = st.session_state.get("active_org", {"id": "default"})
                        user_info_ws = st.session_state.get("user", {"id": "anonymous"})
                        dataset_name = uploaded_file.name if uploaded_file else "database_query_result.csv"
                        add_analysis_run(
                            org_id=active_org["id"],
                            user_id=user_info_ws["id"],
                            dataset_name=dataset_name,
                            status="completed",
                            output_path=public_url,
                        )
                    except Exception as persist_err:
                        print(f"Workspace persist error (non-fatal): {persist_err}")

                    # STORE RESULT IN SESSION STATE
                    st.session_state["analysis_result"] = result
                    st.session_state["analysis_complete"] = True
                    save_to_history(dataset_name, result)
                    status.update(label="✅ Analysis Complete!", state="complete")

                except Exception as exc:
                    st.error(f"❌ Analysis Failed: {exc}")
                    st.session_state["analysis_complete"] = False
                    status.update(label="❌ Analysis Failed", state="error")

            time.sleep(0.5)
            if st.session_state.get("analysis_complete"):
                st.balloons()
            st.rerun()  # Force rerun to render from state

    # ── RENDERING BLOCK (FROM STATE) ─────────────────────────────────────
    if st.session_state.get("analysis_complete") and st.session_state.get("analysis_result"):
        result = st.session_state["analysis_result"]

        if result.status == "completed":
            from frontend.dashboard_ui import render_interactive_dashboard
            
            # Use new modular dashboard
            render_interactive_dashboard(result)

            st.markdown("---")

            # ── Downloads (PERSISTENT BUTTONS) ───────────────────
            st.markdown("### 📥 Download Results")
            dl_cols = st.columns(4)

            # Check files exist on disk (pipeline saves them)
            # Using paths from stored result object
            
            if result.cleaned_csv_path and Path(result.cleaned_csv_path).exists():
                with dl_cols[0]:
                    st.download_button(
                        "📊 Cleaned CSV",
                        data=Path(result.cleaned_csv_path).read_bytes(),
                        file_name="cleaned_data.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

            if result.pdf_report_path and Path(result.pdf_report_path).exists():
                with dl_cols[1]:
                    st.download_button(
                        "📄 PDF Report",
                        data=Path(result.pdf_report_path).read_bytes(),
                        file_name="report.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )

            if result.dashboard_html_path and Path(result.dashboard_html_path).exists():
                with dl_cols[2]:
                    st.download_button(
                        "📊 Dashboard HTML",
                        data=Path(result.dashboard_html_path).read_bytes(),
                        file_name="dashboard.html",
                        mime="text/html",
                        use_container_width=True,
                    )

            if result.markdown_report_path and Path(result.markdown_report_path).exists():
                with dl_cols[3]:
                    st.download_button(
                        "📝 Markdown Report",
                        data=Path(result.markdown_report_path).read_bytes(),
                        file_name="report.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )

            # Duration info
            st.caption(
                f"⏱ Total pipeline duration: {result.total_duration_seconds:.2f}s | "
                f"Job ID: {result.job_id}"
            )

            # ── Scheduled Reports UI ───────────────────────────────────────
            if SCHEDULER_ENABLED:
                st.markdown("---")
                st.subheader("📅 Schedule Recurring Email Reports")
                st.markdown("Set up a regular automated run of this analysis and have the PDF report emailed to you automatically.")
                
                sched_col1, sched_col2 = st.columns(2)
                with sched_col1:
                    schedule_email = st.text_input("Recipient Email Address", placeholder="analyst@company.com")
                with sched_col2:
                    schedule_day = st.selectbox("Frequency (Send every)", ["Monday", "Wednesday", "Friday", "Daily"])
                
                if st.button("⏰ Set Recurring Schedule", type="secondary"):
                    if not schedule_email:
                        st.error("Please enter a valid email address.")
                    else:
                        import uuid
                        import shutil
                        from utils.scheduler import add_scheduled_job
                        
                        # Ensure the dataset file is persistently stored for scheduling runs
                        schedules_dir = UPLOAD_DIR / "schedules"
                        schedules_dir.mkdir(parents=True, exist_ok=True)
                        persistent_path = schedules_dir / f"sched_{uuid.uuid4().hex[:8]}_{temp_path.name}"
                        shutil.copy2(temp_path, persistent_path)
                        
                        # Map day selection to APScheduler day_of_week
                        day_mapping = {
                            "Monday": "mon",
                            "Wednesday": "wed",
                            "Friday": "fri",
                            "Daily": "*"
                        }
                        cron_day = day_mapping.get(schedule_day, "*")
                        
                        # Add to APScheduler
                        job_id = f"sched_{uuid.uuid4().hex[:12]}"
                        success = add_scheduled_job(
                            job_id=job_id,
                            day_of_week_val=cron_day,
                            email=schedule_email,
                            dataset_path=str(persistent_path)
                        )
                        
                        if success:
                            st.success(f"🎉 Scheduled successfully! You'll receive this report every **{schedule_day}** at 9:00 AM at **{schedule_email}**.")
                        else:
                            st.error("Failed to schedule the report. Please check if the scheduler is enabled in your environment.")


        else:
            st.error(f"❌ Pipeline failed: {', '.join(result.errors)}")


# ═══════════════════════════════════════════════════════════════════════
# TAB 2 — Multi-File Comparison
# ═══════════════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown("### 🔀 Multi-File Comparison")
    st.markdown("Upload multiple CSV or Excel files to compare their statistics side-by-side.")

    compare_files = st.file_uploader(
        "Upload CSV/Excel files for comparison",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
        key="compare_upload",
    )

    if compare_files and len(compare_files) >= 2:
        if st.button("🔄 Compare Files", type="primary", use_container_width=True):
            comparison_data = []
            progress = st.progress(0, text="Analyzing files...")

            for idx, cfile in enumerate(compare_files):
                ext = cfile.name.lower()
                if ext.endswith(".xlsx") or ext.endswith(".xls"):
                    df = pd.read_excel(cfile, engine="openpyxl")
                else:
                    df = pd.read_csv(cfile)
                num_cols = df.select_dtypes(include="number").columns.tolist()
                stats_row = {
                    "File": cfile.name,
                    "Rows": len(df),
                    "Columns": len(df.columns),
                    "Missing Values": int(df.isnull().sum().sum()),
                    "Duplicates": int(df.duplicated().sum()),
                }
                for col in num_cols[:5]:
                    stats_row[f"{col} (mean)"] = round(df[col].mean(), 2)
                    stats_row[f"{col} (std)"] = round(df[col].std(), 2)

                comparison_data.append(stats_row)
                progress.progress((idx + 1) / len(compare_files))

            progress.empty()

            comp_df = pd.DataFrame(comparison_data)
            st.markdown("#### 📊 Comparison Table")
            st.dataframe(comp_df, use_container_width=True)

            # Overlay chart for shared numeric columns
            import plotly.graph_objects as go

            shared_metric_cols = [
                c for c in comp_df.columns
                if c not in ("File", "Rows", "Columns", "Missing Values", "Duplicates")
            ]
            if shared_metric_cols:
                st.markdown("#### 📈 Metric Comparison")
                fig = go.Figure()
                for _, row in comp_df.iterrows():
                    fig.add_trace(go.Bar(
                        name=str(row["File"]),
                        x=shared_metric_cols,
                        y=[row[c] for c in shared_metric_cols],
                    ))
                fig.update_layout(
                    barmode="group",
                    template="plotly_dark",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

    elif compare_files and len(compare_files) < 2:
        st.info("Please upload at least 2 files for comparison.")


# ═══════════════════════════════════════════════════════════════════════
# TAB 3 — Real-time Streaming Data Pipeline
# ═══════════════════════════════════════════════════════════════════════
with tab_stream:
    st.markdown("### 📡 Real-time Streaming Data Pipeline")
    st.markdown("Connect to a live streaming API endpoint to automatically poll, clean, and analyze data in real-time.")

    stream_url = st.text_input(
        "Data Stream API Endpoint URL",
        value="http://localhost:8000/api/mock-stream",
        placeholder="https://api.yourcompany.com/realtime-sales",
        key="stream_url_input"
    )

    col_interval, col_toggle = st.columns([1, 1])
    with col_interval:
        poll_interval = st.slider("Polling Interval (seconds)", min_value=5, max_value=120, value=30, step=5)
    with col_toggle:
        auto_refresh = st.toggle("Enable Live Auto-Refresh", value=False, key="stream_auto_refresh")

    poll_now = st.button("🔄 Poll & Analyze Now", type="primary", use_container_width=True)

    if "stream_result" not in st.session_state:
        st.session_state["stream_result"] = None
    if "stream_last_polled" not in st.session_state:
        st.session_state["stream_last_polled"] = 0.0

    current_time = time.time()
    should_poll = poll_now

    if auto_refresh:
        time_elapsed = current_time - st.session_state["stream_last_polled"]
        if time_elapsed >= poll_interval:
            should_poll = True

    if should_poll:
        with st.spinner("Fetching data from stream..."):
            try:
                from utils.stream_connector import poll_once
                df = poll_once(stream_url)
                if df.empty:
                    st.error("❌ Data stream returned an empty response.")
                else:
                    st.success(f"✅ Polled successfully! Fetched {len(df)} rows.")
                    st.session_state["stream_last_polled"] = current_time
                    
                    temp_stream_path = temp_dir / "live_stream_data.csv"
                    df.to_csv(temp_stream_path, index=False)
                    
                    with st.spinner("Running streaming analytics pipeline..."):
                        orchestrator = MasterOrchestrator()
                        
                        logo_path = None
                        if logo_file is not None:
                            import uuid
                            logo_suffix = Path(logo_file.name).suffix
                            temp_logo_path = UPLOAD_DIR / f"logo_{uuid.uuid4().hex[:8]}{logo_suffix}"
                            temp_logo_path.write_bytes(logo_file.getvalue())
                            logo_path = str(temp_logo_path)
                            
                        branding_config = {
                            "company_name": company_name,
                            "primary_color": brand_color,
                            "logo_path": logo_path,
                            "footer_text": f"Confidential — Generated by {company_name}",
                            "analyst_name": analyst_name
                        }
                        
                        result = orchestrator.run(
                            file_path=str(temp_stream_path),
                            output_dir=str(OUTPUT_DIR / "live_stream_output"),
                            branding=branding_config
                        )
                        st.session_state["stream_result"] = result
                        
                        # Save streaming run results to pickle & persist in workspace
                        import uuid
                        import pickle
                        from utils.storage import upload_to_r2
                        from utils.workspace import add_analysis_run
                        
                        stream_job_id = result.job_id or f"stream_{int(time.time())}"
                        stream_output_dir = OUTPUT_DIR / "live_stream_output"
                        stream_output_dir.mkdir(parents=True, exist_ok=True)
                        stream_pickle_path = stream_output_dir / f"{stream_job_id}_pipeline_result.pkl"
                        with open(stream_pickle_path, "wb") as f:
                            pickle.dump(result, f)
                            
                        storage_key = f"{stream_job_id}_pipeline_result.pkl"
                        try:
                            public_url = upload_to_r2(str(stream_pickle_path), storage_key)
                            active_org = st.session_state.get("active_org", {"id": "default"})
                            user_info = st.session_state.get("user", {"id": "anonymous"})
                            
                            add_analysis_run(
                                org_id=active_org["id"],
                                user_id=user_info["id"],
                                dataset_name="live_stream_data.csv",
                                status="completed",
                                output_path=public_url
                            )
                        except Exception as persist_err:
                            print(f"Streaming persist error: {persist_err}")
            except Exception as e:
                st.error(f"❌ Error polling stream: {str(e)}")

    if st.session_state.get("stream_result") is not None:
        stream_res = st.session_state["stream_result"]
        if stream_res.status == "completed":
            st.info(f"Last Polled: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.session_state['stream_last_polled']))}")
            from frontend.dashboard_ui import render_interactive_dashboard
            render_interactive_dashboard(stream_res)
        else:
            st.error(f"❌ Pipeline failed: {', '.join(stream_res.errors)}")

    if auto_refresh:
        time.sleep(1)
        st.rerun()

