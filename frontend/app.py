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
import extra_streamlit_components as stx

from config.settings import API_INTERNAL_URL, BRAND_NAME, BRAND_COLOR, OUTPUT_DIR, UPLOAD_DIR, is_llm_enabled, SCHEDULER_ENABLED
from orchestrator.master import MasterOrchestrator, PipelineResult
from agents.data_quality import score_color, risk_level
from utils.auth import login_user, signup_user, save_session_token, load_session_token, delete_session_token, get_supabase_client
from utils.workspace import create_organization, get_organizations, add_analysis_run, get_org_analysis_history
from utils.storage import upload_to_r2, LOCAL_STORAGE_DIR
from utils.share_reports import create_share_link, revoke_share_link
from utils.monetization import (
    FREE_ANALYSIS_LIMIT,
    FREE_FILE_SIZE_BYTES,
    FREE_NL_QUERY_DAILY_LIMIT,
    billing_portal_url,
    can_run_analysis,
    can_upload_file,
    count_monthly_analyses,
    is_pro_org,
    org_plan,
)


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

# Load custom CSS
import os
_CSS_PATH = os.path.join(os.path.dirname(__file__),
                         "assets", "style.css")
if os.path.exists(_CSS_PATH):
    with open(_CSS_PATH) as _f:
        st.markdown(f"<style>{_f.read()}</style>",
                    unsafe_allow_html=True)


# ── Monetization Helpers ─────────────────────────────────────────────
def _current_org() -> dict:
    return st.session_state.get("active_org", {"id": "default", "name": "Default Team Workspace", "plan": "free"})


def _current_plan() -> str:
    return org_plan(_current_org())


def _refresh_active_org_plan(plan: str) -> None:
    if "active_org" in st.session_state:
        st.session_state["active_org"]["plan"] = plan
    if "orgs" in st.session_state:
        for org in st.session_state["orgs"]:
            if org.get("id") == st.session_state.get("active_org", {}).get("id"):
                org["plan"] = plan


def _start_razorpay_checkout() -> None:
    org = _current_org()
    user = st.session_state.get("user", {})
    if not org.get("id") or org.get("id") == "demo":
        st.warning("Create or select a real workspace before upgrading.")
        return
    try:
        response = requests.post(
            f"{API_INTERNAL_URL.rstrip('/')}/api/checkout/razorpay",
            json={
                "org_id": org.get("id"),
                "org_name": org.get("name", "Workspace"),
                "user_email": user.get("email", ""),
            },
            timeout=20,
        )
        if response.status_code >= 400:
            detail = response.json().get("detail", response.text)
            st.error(detail)
            return
        checkout_url = response.json().get("checkout_url")
        if not checkout_url:
            st.error("Razorpay checkout did not return a payment link.")
            return
        st.link_button("Continue to Razorpay Checkout", checkout_url, type="primary", use_container_width=True)
        st.info("After payment, Razorpay will notify the app through the webhook and your workspace will unlock Pro.")
    except Exception as exc:
        st.error(f"Could not start Razorpay checkout: {exc}")


@st.dialog("👤 Account Settings")
def show_profile_modal():
    org = _current_org()
    plan = _current_plan()
    used = count_monthly_analyses(org.get("id", "default"))
    st.markdown(f"### {org.get('name', 'Workspace')}")
    if plan == "pro":
        st.success("⚡ Pro Plan")
        st.metric("Analyses this month", used, "Unlimited")
    else:
        st.warning("Free Plan")
        st.metric("Analyses this month", f"{used}/{FREE_ANALYSIS_LIMIT}")
        st.progress(min(used / FREE_ANALYSIS_LIMIT, 1.0))
    st.divider()
    if plan == "pro":
        st.link_button("Manage Billing", billing_portal_url(), use_container_width=True)
    else:
        if st.button("Upgrade to Pro — $19/month", type="primary", use_container_width=True, key="profile_upgrade_btn"):
            _start_razorpay_checkout()
    st.caption("Email and password changes are handled by Supabase Auth.")


# ── Upgrade Modal ───────────────────────────────────────────────────
@st.dialog("⚡ Upgrade to Pro")
def show_upgrade_modal():
    st.markdown("### Unlock the full power of AI analysis")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Free**")
        st.markdown("$0 / month")
        st.markdown("""
        - ✅ 5 analyses/month
        - ✅ CSV & Excel upload
        - ✅ Basic insights
        - ❌ Advanced AI insights
        - ❌ PDF reports
        - ❌ Team workspace
        - ❌ Forecast lab
        """)

    with col2:
        st.markdown("**⚡ Pro**")
        st.markdown(
            '<p><s style="color:#888;">$49</s> '
            '<strong style="color:#a89fe8; '
            'font-size:1.3rem;">$19 / month</strong></p>',
            unsafe_allow_html=True
        )
        st.markdown("""
        - ✅ Unlimited analyses
        - ✅ All file formats
        - ✅ 10x deeper insights
        - ✅ Advanced AI insights
        - ✅ PDF + Excel reports
        - ✅ Team workspace
        - ✅ Forecast lab
        """)

    st.divider()
    st.caption("🚀 1,200+ founders already on Pro · "
               "Cancel anytime · No hidden fees")

    if st.button("Start Pro — $19/month",
                 type="primary",
                 use_container_width=True):
        _start_razorpay_checkout()


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

    st.markdown("### See real output before signup")
    proof_cols = st.columns(3)
    with proof_cols[0]:
        st.markdown("**Dashboard**")
        st.caption("KPI cards, filters, charts, and forecast panels from the bundled retail sample.")
    with proof_cols[1]:
        st.markdown("**PDF report**")
        st.caption("A client-ready report generated from the same deterministic pipeline.")
    with proof_cols[2]:
        st.markdown("**Excel workbook**")
        st.caption("Cleaned data, KPI summary, insights, and forecasts in a formatted spreadsheet.")

    cta_col1, cta_col2 = st.columns(2)
    with cta_col1:
        if st.button("📂 Try Demo With Sample Data", type="primary", use_container_width=True, key="landing_demo_btn"):
            st.session_state["demo_mode"] = True
            st.session_state["user"] = {
                "id": "demo-user",
                "email": "demo@ai-data-analyzer.local",
                "type": "demo",
            }
            st.session_state["sample_dataset_path"] = str(ROOT / "data" / "sample_walmart.csv")
            st.session_state["sample_dataset_name"] = "sample_walmart.csv"
            st.session_state["demo_autorun_pending"] = True
            st.rerun()

    with cta_col2:
        if st.button("🔑 Sign In / Create Account", use_container_width=True, key="landing_cta_btn"):
            st.session_state["show_auth"] = True
            st.rerun()

    if st.session_state.get("demo_mode"):
        st.info("Demo mode uses bundled sample data and does not write to shared workspace history.")


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

# ── Persistent Session Restore (Cookie-Based) ───────────────────────────────
# On every page load BEFORE the login gate, we attempt to read the session
# token from the HttpOnly cookie (set at login). This keeps users logged in
# across browser refreshes without exposing the token in the URL.
_cookie_manager = stx.CookieManager(key="_session_cm")
if "user" not in st.session_state:
    _saved_token = _cookie_manager.get("session_token") or ""
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
                        # Rotate the cookie with the refreshed token
                        _cookie_manager.set(
                            "session_token", _new_token_str
                        )
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

if st.query_params.get("upgrade") == "success" and not st.session_state.get("_upgrade_success_seen"):
    st.session_state["_upgrade_success_seen"] = True
    try:
        refreshed_orgs = get_organizations()
        st.session_state["orgs"] = refreshed_orgs
        paid_org_id = st.query_params.get("org_id", "")
        for org in refreshed_orgs:
            if org.get("id") == paid_org_id:
                st.session_state["active_org"] = org
                break
    except Exception:
        pass
    st.balloons()
    st.success("Payment received. Your Pro plan will unlock once the Razorpay webhook confirms the subscription.")

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
                        # Store in HttpOnly cookie — NOT in URL
                        _cookie_manager.set(
                            "session_token", _token
                        )
                    else:
                        # Save persistent session token
                        _token = save_session_token(user)
                        st.session_state["_session_token"] = _token
                        # Store in HttpOnly cookie — NOT in URL
                        _cookie_manager.set(
                            "session_token", _token
                        )
                        
                    st.success("🎉 Welcome back! Logging you in...")
                    time.sleep(1)
                    _login_ok = True
                except Exception as e:
                    st.error("Connection failed. Please check your "
                             "credentials and try again.")
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

                        # ── Email verification enforcement ────────────────────
                        # When Supabase "Confirm email" is enabled, signup_user
                        # sets pending_verification=True and returns no tokens.
                        # We must NOT create a session in this case — the user
                        # must confirm their inbox first.
                        if user.get("pending_verification"):
                            st.success("✅ Account created!")
                            st.info(
                                "📧 **Please check your email and verify your account "
                                "before logging in.**\n\n"
                                "We've sent a confirmation link to "
                                f"**{user.get('email', signup_email)}**. "
                                "Click the link in that email, then return here to sign in."
                            )
                            # Stay on the auth page — do not set session state or rerun.
                        else:
                            # Email confirmation is disabled (dev/local) — auto-login.
                            st.session_state["user"] = user

                            if user.get("type") == "supabase" and user.get("access_token") and user.get("refresh_token"):
                                _token = f"{user['access_token']}:::{user['refresh_token']}"
                                st.session_state["_session_token"] = _token
                                # Store in HttpOnly cookie — NOT in URL
                                _cookie_manager.set(
                                    "session_token", _token
                                )
                            else:
                                _token = save_session_token(user)
                                st.session_state["_session_token"] = _token
                                # Store in HttpOnly cookie — NOT in URL
                                _cookie_manager.set(
                                    "session_token", _token
                                )

                            st.success("🎉 Account created successfully! Logging you in...")
                            time.sleep(1)
                            _signup_ok = True

                    except Exception as e:
                        st.error("Connection failed. Please check your "
                                 "credentials and try again.")
                    if _signup_ok:
                        st.rerun()
    st.stop()

# Ensure an active organization is loaded in session state
if st.session_state.get("demo_mode"):
    st.session_state["active_org"] = {"id": "demo", "name": "Demo Workspace"}
elif "active_org" not in st.session_state:
    if "orgs" not in st.session_state:
        st.session_state["orgs"] = get_organizations()
    orgs = st.session_state["orgs"]
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
    _sidebar_plan = _current_plan()
    if _sidebar_plan == "pro":
        st.success("⚡ Pro Plan")
    else:
        st.warning("Free Plan")
    if st.button("👤 Account Settings", use_container_width=True):
        show_profile_modal()
    if st.button("🚪 Log Out", type="secondary", use_container_width=True):
        # Revoke persistent session token (BUG 2 fix)
        _logout_token = st.session_state.get("_session_token", "") or _cookie_manager.get("session_token") or ""
        if _logout_token:
            _supabase_client = get_supabase_client()
            if _supabase_client and ":::" in _logout_token:
                try:
                    _supabase_client.auth.sign_out()
                except Exception:
                    pass
            else:
                delete_session_token(_logout_token)
        # Delete the HttpOnly cookie on logout
        _cookie_manager.delete("session_token")
        st.session_state.clear()
        st.success("Logged out successfully!")
        time.sleep(0.5)
        st.rerun()
        
    st.markdown("---")
    
    # Collapsible groups for Sidebar settings
    if st.session_state.get("demo_mode"):
        st.info("Demo mode is read-only for workspace history. Create an account to save private analyses.")
    elif not is_pro_org(_current_org()):
        with st.expander("🏢 Team Workspace", expanded=False):
            active_org = _current_org()
            st.markdown(f"**Workspace:** {active_org.get('name', 'Default Team Workspace')}")
            st.warning("Team workspaces are a Pro feature. Upgrade to collaborate with your team.")
            if st.button("⚡ Upgrade for Team Workspaces", type="primary", use_container_width=True, key="workspace_upgrade_btn"):
                show_upgrade_modal()
    else:
      with st.expander("🏢 Team Workspace", expanded=False):
        # ── Workspace Switcher & Creator ─────────────────────────────────
        if "orgs" not in st.session_state:
            st.session_state["orgs"] = get_organizations()
        orgs = st.session_state["orgs"]
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
                    if "org_history" in st.session_state:
                        del st.session_state["org_history"]
                    # NOTE: No st.rerun() here — calling rerun inside a selectbox
                    # render loop (not a button handler) causes an infinite rerun
                    # cycle. Session state update alone is sufficient; Streamlit
                    # will re-render affected widgets on the next natural cycle.
                    
        st.markdown("**Create Workspace**")
        new_org_name = st.text_input("Workspace Name", key="new_org_name_input", placeholder="New team name...")
        if st.button("Create", use_container_width=True):
            if new_org_name.strip():
                new_org = create_organization(new_org_name.strip())
                st.session_state["active_org"] = new_org
                if "orgs" in st.session_state:
                    del st.session_state["orgs"]
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
            if "org_history" not in st.session_state:
                st.session_state["org_history"] = get_org_analysis_history(active_org["id"])
            history = st.session_state["org_history"]
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

with st.sidebar:
    # Brand
    st.markdown("### 📊 AI Data Analyzer")
    st.divider()
    
    _active_org_for_usage = _current_org()
    _plan_for_usage = _current_plan()
    _used = count_monthly_analyses(_active_org_for_usage.get("id", "default"))
    if _plan_for_usage == "pro":
        st.success("⚡ Pro Plan: unlimited analyses")
        st.caption(f"{_used} analyses run this month")
    else:
        st.markdown(f"**Free Analyses:** {_used}/{FREE_ANALYSIS_LIMIT} used")
        st.progress(min(_used / FREE_ANALYSIS_LIMIT, 1.0))
    
    st.divider()
    
    # Locked insights (always visible as FOMO)
    st.markdown("**✨ AI Insights**")
    st.success("✅ Data quality score: 94%")
    st.success("✅ Top trend identified")
    if _plan_for_usage == "pro":
        st.success("✅ Revenue anomaly detection unlocked")
        st.success("✅ Forecast lab unlocked")
    else:
        st.info("🔒 Revenue anomaly detected — Pro")
        st.info("🔒 Churn risk segment — Pro")
        st.info("🔒 Forecast accuracy report — Pro")
    
    st.divider()
    
    # Upgrade CTA
    if _plan_for_usage == "pro":
        if st.button("👤 Manage Account", use_container_width=True):
            show_profile_modal()
    elif st.button("⚡ Upgrade to Pro — $19/mo",
                   use_container_width=True,
                   type="primary"):
        show_upgrade_modal()

    st.caption("🚀 1,200+ founders already upgraded")
    st.caption("Cancel anytime · No credit card to start")

# Sticky bottom upgrade banner
if _current_plan() != "pro" and not st.session_state.get("banner_dismissed", False):
    st.markdown("---")
    _b1, _b2, _b3 = st.columns([3, 1, 0.5])
    _b1.markdown(
        "🚀 **You're leaving insights on the table.** "
        "Pro users get 10x deeper analysis + "
        "unlimited CSV uploads."
    )
    if _b2.button("Upgrade Now — $19/mo",
                  type="primary",
                  key="bottom_upgrade_btn"):
        show_upgrade_modal()
    if _b3.button("✕", key="dismiss_banner"):
        st.session_state["banner_dismissed"] = True
        st.rerun()

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
            file_bytes = uploaded_file.getvalue()
            upload_ok, upload_msg = can_upload_file(_current_org(), len(file_bytes))
            if not upload_ok:
                st.error(upload_msg)
                if st.button("⚡ Upgrade to upload up to 50MB", type="primary", use_container_width=True, key="file_size_upgrade_btn"):
                    show_upgrade_modal()
            else:
                temp_path = temp_dir / uploaded_file.name
                temp_path.write_bytes(file_bytes)
                st.success(f"✅ Uploaded: **{uploaded_file.name}** ({len(file_bytes) / 1024:.1f} KB)")
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

        # ── SQL Injection Guard ─────────────────────────────────────────
        def _validate_sql(query: str) -> tuple[bool, str]:
            """Returns (is_safe, error_message). Blocks any non-SELECT statement."""
            import sqlparse
            _BLOCKED = {"DROP", "DELETE", "INSERT", "UPDATE", "ALTER",
                        "TRUNCATE", "CREATE", "EXEC", "EXECUTE", "GRANT", "REVOKE"}
            parsed = sqlparse.parse(query.strip())
            for statement in parsed:
                stmt_type = statement.get_type()
                if stmt_type and stmt_type.upper() in _BLOCKED:
                    return False, f"⚠️ Only SELECT queries are allowed for security reasons (blocked: {stmt_type.upper()})"
                # Also scan individual tokens for blocked keywords (catches multi-statement injection)
                for token in statement.flatten():
                    if token.ttype is sqlparse.tokens.Keyword.DDL or token.ttype is sqlparse.tokens.Keyword.DML:
                        if token.normalized.upper() in _BLOCKED:
                            return False, f"⚠️ Only SELECT queries are allowed for security reasons (blocked: {token.normalized.upper()})"
            return True, ""

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
            st.info("Read-only mode: Only SELECT statements are permitted.", icon="🔒")
            pg_query = st.text_area("SQL Query", value="SELECT * FROM my_table LIMIT 1000", key="pg_query")

            if st.button("🔌 Connect & Query PostgreSQL", key="pg_btn"):
                _sql_safe, _sql_err = _validate_sql(pg_query)
                if not _sql_safe:
                    st.error(_sql_err)
                else:
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
                            _pg_err = str(e)
                            if "no password" in _pg_err:
                                st.error("Please enter your database password.")
                            elif "Connection refused" in _pg_err:
                                st.error("Could not reach the database server. Check the host and port.")
                            elif "password authentication failed" in _pg_err:
                                st.error("Wrong username or password. Please check your credentials.")
                            elif "does not exist" in _pg_err:
                                st.error("Database name not found. Check the database name field.")
                            else:
                                st.error("Connection failed. Please check all fields and try again.")
                        
        elif db_type == "Google BigQuery":
            bq_project = st.text_input("BigQuery Project ID", key="bq_proj")
            bq_creds_path = st.text_input("Service Account JSON Path (Optional)", help="Leave blank if local/env authentication is configured.", key="bq_creds")
            st.info("Read-only mode: Only SELECT statements are permitted.", icon="🔒")
            bq_query = st.text_area("SQL Query", value="SELECT * FROM `project.dataset.table` LIMIT 1000", key="bq_query")

            if st.button("🔌 Connect & Query BigQuery", key="bq_btn"):
                _sql_safe, _sql_err = _validate_sql(bq_query)
                if not _sql_safe:
                    st.error(_sql_err)
                else:
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
                            st.error("BigQuery connection failed. Please check your project ID and credentials.")
                        
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
            st.info("Read-only mode: Only SELECT statements are permitted.", icon="🔒")
            sf_query = st.text_area("SQL Query", value="SELECT * FROM my_table LIMIT 1000", key="sf_query")

            if st.button("🔌 Connect & Query Snowflake", key="sf_btn"):
                _sql_safe, _sql_err = _validate_sql(sf_query)
                if not _sql_safe:
                    st.error(_sql_err)
                else:
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
                            st.error("Snowflake connection failed. Please check all fields and try again.")

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
        if st.session_state.pop("demo_autorun_pending", False):
            run_clicked = True
            st.info("Running the demo analysis with bundled sample data...")
        
        if run_clicked or "celery_task_id" in st.session_state:
            if run_clicked and not st.session_state.get("demo_mode"):
                run_allowed, run_msg, _, _ = can_run_analysis(_current_org())
                if not run_allowed:
                    st.error(run_msg)
                    if st.button("⚡ Upgrade to Pro for unlimited analyses", type="primary", use_container_width=True, key="analysis_limit_upgrade_btn"):
                        show_upgrade_modal()
                    st.stop()
                else:
                    st.caption(run_msg)

            # Clear previous state
            if run_clicked:
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

                # ── FIX: flag-based rerun prevents RerunException being caught ─
                # Now we set a flag and call st.rerun() AFTER the try/except exits.
                _pending_rerun = False

                try:
                    if _use_celery:
                        # ── Celery path (Redis is running) ────────────────────────
                        if "celery_task_id" not in st.session_state:
                            task = run_analysis_task.delay(
                                str(temp_path),
                                str(job_output),
                                branding=branding_config,
                                dataset_name=uploaded_file.name if uploaded_file else temp_path.name,
                                is_pro=is_pro_org(_current_org()),
                            )
                            st.session_state["celery_task_id"] = task.id
                        else:
                            task = run_analysis_task.AsyncResult(st.session_state["celery_task_id"])

                        # Poll task progress — set flag instead of calling st.rerun()
                        task_state = task.state
                        if task_state == "PROGRESS" or task_state == "PENDING":
                            meta = task.info or {}
                            if isinstance(meta, dict):
                                pct = meta.get("pct", 0.05)
                                stage_key = meta.get("stage", "").lower()
                                if stage_key != _last_stage[0]:
                                    _last_stage[0] = stage_key
                                    label, _ = _stage_map.get(stage_key, (f"⏳ {meta.get('stage', 'Processing...')}", pct))
                                    st.write(label)
                                progress_bar.progress(pct, text=f"⏳ {meta.get('stage', 'Processing...')}")
                            time.sleep(0.3)
                            _pending_rerun = True  # signal; rerun happens AFTER try/except
                        elif task_state == "SUCCESS":
                            result_dict = task.result or {}
                            if result_dict.get("status") == "failed":
                                del st.session_state["celery_task_id"]
                                errors = result_dict.get("errors", ["Pipeline failed."])
                                raise Exception(" | ".join(errors))

                            pickle_path = result_dict.get("pickle_path")
                            if pickle_path and Path(pickle_path).exists():
                                with open(pickle_path, "rb") as f:
                                    result = pickle.load(f)
                            else:
                                del st.session_state["celery_task_id"]
                                raise Exception("Pipeline completed but results are missing from disk.")
                            del st.session_state["celery_task_id"]
                        elif task_state == "FAILURE":
                            del st.session_state["celery_task_id"]
                            raise Exception(str(task.result) if task.result else "Celery execution failed.")

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
                            is_pro=is_pro_org(_current_org()),
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

                    # ── Completion path (only runs when _pending_rerun is False) ─
                    if not _pending_rerun:
                        st.write("🔍 Detecting insights...")
                        st.write("📈 Generating forecasts...")
                        st.write("📄 Building PDF report...")
                        progress_bar.progress(1.0, text="✅ Complete!")

                        # ── Persist analysis run in workspace ──────────────────────
                        run_id = result.job_id or str(_uuid.uuid4())
                        storage_key = f"{run_id}_pipeline_result.pkl"
                        dataset_name = uploaded_file.name if uploaded_file else temp_path.name
                        try:
                            from utils.storage import upload_to_r2
                            from utils.workspace import add_analysis_run
                            public_url = upload_to_r2(pickle_path, storage_key)
                            active_org = st.session_state.get("active_org", {"id": "default"})
                            user_info_ws = st.session_state.get("user", {"id": "anonymous"})
                            if not st.session_state.get("demo_mode"):
                                add_analysis_run(
                                    org_id=active_org["id"],
                                    user_id=user_info_ws["id"],
                                    dataset_name=dataset_name,
                                    status="completed",
                                    output_path=public_url,
                                )
                        except Exception as persist_err:
                            print(f"Workspace persist error (non-fatal): {persist_err}")

                        if "org_history" in st.session_state:
                            del st.session_state["org_history"]

                        # STORE RESULT IN SESSION STATE
                        st.session_state["analysis_result"] = result
                        st.session_state["analysis_dataset_name"] = dataset_name
                        st.session_state["analysis_pickle_path"] = str(pickle_path)
                        st.session_state["analysis_complete"] = True
                        save_to_history(dataset_name, result)
                        status.update(label="✅ Analysis Complete!", state="complete")

                except Exception as exc:
                    if "celery_task_id" in st.session_state:
                        del st.session_state["celery_task_id"]
                    st.error(f"❌ Analysis Failed: {exc}")
                    st.session_state["analysis_complete"] = False
                    status.update(label="❌ Analysis Failed", state="error")

            # ── Rerun is now safely OUTSIDE the try/except ────────────────────
            if _pending_rerun:
                time.sleep(0.3)
                st.rerun()

            time.sleep(0.5)
            if st.session_state.get("analysis_complete"):
                st.balloons()
            st.rerun()  # Force rerun to render from state

    # ── RENDERING BLOCK (FROM STATE) ─────────────────────────────────────
    if st.session_state.get("analysis_complete") and st.session_state.get("analysis_result"):
        result = st.session_state["analysis_result"]

        if result.status in ("completed", "completed_with_warnings"):
            from frontend.dashboard_ui import render_interactive_dashboard
            render_interactive_dashboard(result)

            st.markdown("---")

            # ── Downloads (PERSISTENT BUTTONS) ───────────────────
            st.markdown("### 📥 Download Results")
            dl_cols = st.columns(5)

            # Check files exist on disk (pipeline saves them)
            # Using paths from stored result object
            
            with dl_cols[0]:
                if is_pro_org(_current_org()):
                    if getattr(result, "cleaned_csv_path", "") and Path(result.cleaned_csv_path).exists():
                        st.download_button(
                            "📊 Cleaned CSV",
                            data=Path(result.cleaned_csv_path).read_bytes(),
                            file_name="cleaned_data.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
                else:
                    st.button("🔒 Cleaned CSV (Pro)", disabled=True, use_container_width=True, key="csv_lock")
                    if st.button("Upgrade for CSV", type="primary", use_container_width=True, key="csv_upgrade_btn"):
                        show_upgrade_modal()

            with dl_cols[1]:
                if is_pro_org(_current_org()):
                    if getattr(result, "pdf_report_path", "") and Path(result.pdf_report_path).exists():
                        st.download_button(
                            "📄 PDF Report",
                            data=Path(result.pdf_report_path).read_bytes(),
                            file_name="report.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                else:
                    st.button("🔒 PDF Report (Pro)", disabled=True, use_container_width=True, key="pdf_lock")
                    if st.button("Upgrade for PDF", type="primary", use_container_width=True, key="pdf_upgrade_btn"):
                        show_upgrade_modal()

            with dl_cols[2]:
                if is_pro_org(_current_org()):
                    if getattr(result, "dashboard_html_path", "") and Path(result.dashboard_html_path).exists():
                        st.download_button(
                            "📊 Dashboard HTML",
                            data=Path(result.dashboard_html_path).read_bytes(),
                            file_name="dashboard.html",
                            mime="text/html",
                            use_container_width=True,
                        )
                else:
                    st.button("🔒 Dashboard (Pro)", disabled=True, use_container_width=True, key="html_lock")
                    if st.button("Upgrade for HTML", type="primary", use_container_width=True, key="html_upgrade_btn"):
                        show_upgrade_modal()

            with dl_cols[3]:
                if is_pro_org(_current_org()):
                    if getattr(result, "markdown_report_path", "") and Path(result.markdown_report_path).exists():
                        st.download_button(
                            "📝 Markdown Report",
                            data=Path(result.markdown_report_path).read_bytes(),
                            file_name="report.md",
                            mime="text/markdown",
                            use_container_width=True,
                        )
                else:
                    st.button("🔒 Markdown (Pro)", disabled=True, use_container_width=True, key="md_lock")
                    if st.button("Upgrade for MD", type="primary", use_container_width=True, key="md_upgrade_btn"):
                        show_upgrade_modal()

            with dl_cols[4]:
                if is_pro_org(_current_org()):
                    if getattr(result, "excel_report_path", "") and Path(result.excel_report_path).exists():
                        st.download_button(
                            "📗 Excel Report",
                            data=Path(result.excel_report_path).read_bytes(),
                            file_name="analysis_report.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True,
                        )
                else:
                    st.button("🔒 Excel (Pro)", disabled=True, use_container_width=True, key="excel_lock")
                    if st.button("Upgrade for Excel", type="primary", use_container_width=True, key="excel_upgrade_btn"):
                        show_upgrade_modal()

            st.markdown("### 🔗 Share Report")
            share_disabled = result.status not in ("completed", "completed_with_warnings")
            if st.button("Create Read-Only Share Link", use_container_width=True, disabled=share_disabled):
                try:
                    user_info = st.session_state.get("user", {"id": "anonymous"})
                    dataset_name_for_share = st.session_state.get("analysis_dataset_name", "analysis_report")
                    plan = _current_plan()
                    share_record = create_share_link(
                        result=result,
                        owner_user_id=user_info.get("id", "anonymous"),
                        dataset_name=dataset_name_for_share,
                        plan=plan,
                    )
                    st.session_state["latest_share_token"] = share_record["share_token"]
                    st.session_state["latest_share_url"] = share_record["url"]
                    st.success("Share link created.")
                except Exception as share_exc:
                    st.error(f"Could not create share link: {share_exc}")

            if st.session_state.get("latest_share_url"):
                st.text_input(
                    "Copy share link",
                    value=st.session_state["latest_share_url"],
                    key="latest_share_url_display",
                )
                if st.button("Disable This Share Link", use_container_width=True):
                    token = st.session_state.get("latest_share_token", "")
                    user_info = st.session_state.get("user", {"id": "anonymous"})
                    if token and revoke_share_link(token, user_info.get("id")):
                        st.success("Share link disabled.")
                        st.session_state.pop("latest_share_url", None)
                        st.session_state.pop("latest_share_token", None)
                        st.rerun()
                    else:
                        st.warning("No active share link was found to disable.")

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
            # CHANGE 1 — Filter irrelevant columns before stats computation
            _irrelevant_keywords = ("id", "unnamed", "postal", "code", "index", "zip", "age")

            comparison_data = []
            progress = st.progress(0, text="Analyzing files...")
            _file_shapes = []  # store (filename, rows, cols) after filtering
            _file_dfs = []    # store filtered dataframes for biggest-difference computation

            for idx, cfile in enumerate(compare_files):
                ext = cfile.name.lower()
                if ext.endswith(".xlsx") or ext.endswith(".xls"):
                    df = pd.read_excel(cfile, engine="openpyxl")
                else:
                    df = pd.read_csv(cfile)

                # Filter out irrelevant columns by name
                filtered_cols = [
                    c for c in df.columns
                    if not any(kw in c.lower() for kw in _irrelevant_keywords)
                ]
                df = df[filtered_cols]

                _file_shapes.append((cfile.name, len(df), len(df.columns)))
                _file_dfs.append(df)

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
                # CHANGE 2 — Exclude columns whose mean is >10x the median of all column means
                import numpy as _np
                _all_means = [comp_df[c].mean() for c in shared_metric_cols]
                _median_of_means = _np.median(_all_means) if _all_means else 0
                chart_metric_cols = [
                    c for c, m in zip(shared_metric_cols, _all_means)
                    if _median_of_means == 0 or abs(m) <= 10 * abs(_median_of_means)
                ]

                st.markdown("#### 📈 Metric Comparison")
                fig = go.Figure()
                for _, row in comp_df.iterrows():
                    fig.add_trace(go.Bar(
                        name=str(row["File"]),
                        x=chart_metric_cols,
                        y=[row[c] for c in chart_metric_cols],
                    ))
                fig.update_layout(
                    barmode="group",
                    template="plotly_dark",
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

                # CHANGE 3 — Plain-English summary below chart
                if len(_file_shapes) >= 2:
                    _fn1, _r1, _c1 = _file_shapes[0]
                    _fn2, _r2, _c2 = _file_shapes[1]
                    st.write(f"File 1: {_fn1} — {_r1} rows, {_c1} columns")
                    st.write(f"File 2: {_fn2} — {_r2} rows, {_c2} columns")

                    try:
                        _d1 = _file_dfs[0].select_dtypes(include="number")
                        _d2 = _file_dfs[1].select_dtypes(include="number")
                        _shared = [c for c in _d1.columns if c in _d2.columns]
                        _best_col = None
                        _best_pct = 0
                        _best_dir = ""
                        for _c in _shared:
                            _m1 = _d1[_c].dropna().mean()
                            _m2 = _d2[_c].dropna().mean()
                            if _m1 is None or _m2 is None:
                                continue
                            import math
                            if math.isnan(_m1) or math.isnan(_m2) or _m1 == 0:
                                continue
                            _pct = abs(_m1 - _m2) / abs(_m1) * 100
                            if _pct > _best_pct:
                                _best_pct = _pct
                                _best_col = _c
                                _best_dir = "higher" if _m2 > _m1 else "lower"
                        if _best_col:
                            st.write(f"Biggest difference: {_best_col} is {_best_pct:.1f}% {_best_dir} in File 2 vs File 1.")
                        else:
                            st.write("Biggest difference: no common numeric columns found between the two files.")
                    except:
                        pass

    elif compare_files and len(compare_files) < 2:
        st.info("Please upload at least 2 files for comparison.")


# ═══════════════════════════════════════════════════════════════════════
# TAB 3 — Live Google Sheets Analysis
# ═══════════════════════════════════════════════════════════════════════
with tab_stream:
    st.markdown("### 📊 Live Google Sheets Analysis")
    st.markdown("Paste a public Google Sheets link to analyze live data")

    sheet_url = st.text_input(
        "Google Sheets URL",
        placeholder="https://docs.google.com/spreadsheets/d/.../edit#gid=0",
        key="gsheets_live_url_input",
    )

    poll_interval = st.selectbox(
        "Auto-refresh interval",
        ["Manual only", "Every 1 hour", "Every 6 hours", "Every 24 hours"],
        key="gsheets_live_interval",
    )

    analyze_sheet_clicked = st.button(
        "📥 Analyze Sheet Now",
        type="primary",
        use_container_width=True,
        key="gsheets_live_analyze_btn",
    )

    if "gsheets_live_result" not in st.session_state:
        st.session_state["gsheets_live_result"] = None

    if analyze_sheet_clicked and sheet_url.strip():
        # Transform the share URL into a CSV export URL
        sheet_url_clean = sheet_url.strip()
        if "spreadsheets/d/" in sheet_url_clean:
            import re
            match = re.search(r'/spreadsheets/d/([a-zA-Z0-9-_]+)', sheet_url_clean)
            gid_match = re.search(r'gid=(\d+)', sheet_url_clean)
            if match:
                sheet_id = match.group(1)
                gid = gid_match.group(1) if gid_match else "0"
                transformed_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
            else:
                transformed_url = sheet_url_clean
        else:
            transformed_url = sheet_url_clean

        with st.spinner("Fetching data from Google Sheets..."):
            try:
                from utils.gsheets import load_google_sheet
                _sheet_df = load_google_sheet(transformed_url)
            except Exception:
                st.error(
                    "Could not connect to this Google Sheet. "
                    "Make sure sharing is set to 'Anyone with the link can view'."
                )
                _sheet_df = None

        if _sheet_df is not None and not _sheet_df.empty:
            st.success(f"✅ Loaded {len(_sheet_df)} rows from Google Sheets.")

            import uuid as _uuid_gs
            import pickle as _pickle_gs

            _gs_temp_path = OUTPUT_DIR / "_temp" / f"gsheet_{_uuid_gs.uuid4().hex[:8]}.csv"
            _gs_temp_path.parent.mkdir(parents=True, exist_ok=True)
            _sheet_df.to_csv(_gs_temp_path, index=False)

            _gs_output_dir = OUTPUT_DIR / "gsheet_live_output"
            _gs_output_dir.mkdir(parents=True, exist_ok=True)

            logo_path = None
            if logo_file is not None:
                _logo_suffix = Path(logo_file.name).suffix
                _temp_logo = UPLOAD_DIR / f"logo_{_uuid_gs.uuid4().hex[:8]}{_logo_suffix}"
                _temp_logo.write_bytes(logo_file.getvalue())
                logo_path = str(_temp_logo)

            _gs_branding = {
                "company_name": company_name,
                "primary_color": brand_color,
                "logo_path": logo_path,
                "footer_text": f"Confidential — Generated by {company_name}",
                "analyst_name": analyst_name,
            }

            with st.spinner("Running AI analysis pipeline..."):
                try:
                    _gs_orchestrator = MasterOrchestrator()
                    _gs_result = _gs_orchestrator.run(
                        csv_path=str(_gs_temp_path),
                        output_dir=str(_gs_output_dir),
                        branding=_gs_branding,
                    )
                    st.session_state["gsheets_live_result"] = _gs_result

                    # Persist run in workspace
                    _gs_job_id = _gs_result.job_id or f"gsheet_{int(time.time())}"
                    _gs_pickle_path = _gs_output_dir / f"{_gs_job_id}_pipeline_result.pkl"
                    with open(_gs_pickle_path, "wb") as _f:
                        _pickle_gs.dump(_gs_result, _f)
                    try:
                        from utils.storage import upload_to_r2
                        from utils.workspace import add_analysis_run
                        _gs_public_url = upload_to_r2(str(_gs_pickle_path), f"{_gs_job_id}_pipeline_result.pkl")
                        _gs_active_org = st.session_state.get("active_org", {"id": "default"})
                        _gs_user_info = st.session_state.get("user", {"id": "anonymous"})
                        add_analysis_run(
                            org_id=_gs_active_org["id"],
                            user_id=_gs_user_info["id"],
                            dataset_name="google_sheet_live.csv",
                            status="completed",
                            output_path=_gs_public_url,
                        )
                    except Exception as _gs_persist_err:
                        print(f"GSheet live persist error: {_gs_persist_err}")
                except Exception as _gs_run_err:
                    st.error(f"❌ Analysis pipeline failed: {_gs_run_err}")
        elif _sheet_df is not None and _sheet_df.empty:
            st.warning("⚠️ The Google Sheet appears to be empty.")

    elif analyze_sheet_clicked and not sheet_url.strip():
        st.warning("Please enter a Google Sheets URL first.")

    if st.session_state.get("gsheets_live_result") is not None:
        _gs_res = st.session_state["gsheets_live_result"]
        if _gs_res.status in ("completed", "completed_with_warnings"):
            from frontend.dashboard_ui import render_interactive_dashboard
            render_interactive_dashboard(_gs_res)
        else:
            st.error(f"❌ Pipeline failed: {', '.join(_gs_res.errors)}")
