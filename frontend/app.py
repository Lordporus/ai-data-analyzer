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

from config.settings import BRAND_NAME, BRAND_COLOR, OUTPUT_DIR, is_llm_enabled
from orchestrator.master import MasterOrchestrator, PipelineResult
from agents.data_quality import score_color, risk_level

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title=f"{BRAND_NAME}",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {{
        --brand: {BRAND_COLOR};
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
        background: linear-gradient(135deg, {BRAND_COLOR}, #a78bfa);
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
        color: {BRAND_COLOR};
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
        border-left: 3px solid {BRAND_COLOR};
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


# ── Sidebar — Branding ───────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div style="padding:12px 0 8px 0;">'
        '<strong style="font-size:1.1rem; color:#e6edf3;">AI Data Analyzer</strong><br>'
        '<span style="font-size:0.8rem; color:#8b949e;">Enterprise Data Intelligence Platform</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    brand_name = "AI Data Analyzer"
    brand_color = st.color_picker("Brand Color", value=BRAND_COLOR)
    
    # LLM Status Indicator
    if is_llm_enabled():
        st.success("AI Narrative Mode (Optional)", icon="✅")
        st.caption("Enables AI-enhanced executive explanations. System works without this.")
    else:
        st.info("Deterministic Mode", icon="ℹ️")
        st.caption("AI Narrative Mode is off. All outputs are fully deterministic.")
    st.markdown("---")
    st.markdown("## About")
    st.markdown(
        "Upload a CSV file and get:\n"
        "- Automated cleaning\n"
        "- Intelligent repair\n"
        "- Statistical insights\n"
        "- Interactive dashboard\n"
        "- PDF report"
    )
    st.markdown("---")
    st.caption("\u00a9 2026 AI Data Analyzer \u00b7 Portfolio Edition")




# ── Header ───────────────────────────────────────────────────────────
st.markdown(f"""
<section class="main-header">
    <h1>🔬 {brand_name}</h1>
    <p>AI-Powered Data Analysis • Clean • Analyze • Report</p>
</section>
""", unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────────────────
tab_upload, tab_compare = st.tabs(["📤 Single File Analysis", "🔀 Multi-File Comparison"])

# ═══════════════════════════════════════════════════════════════════════
# TAB 1 — Single File Analysis
# ═══════════════════════════════════════════════════════════════════════
with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload your CSV file",
        type=["csv"],
        help="Maximum file size: 50 MB",
    )

    # Initialize Session State
    if "analysis_complete" not in st.session_state:
        st.session_state["analysis_complete"] = False
    if "analysis_result" not in st.session_state:
        st.session_state["analysis_result"] = None

    if uploaded_file is not None:
        # Save temp file
        temp_dir = OUTPUT_DIR / "_temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / uploaded_file.name
        temp_path.write_bytes(uploaded_file.getvalue())

        st.success(f"✅ Uploaded: **{uploaded_file.name}** ({len(uploaded_file.getvalue()) / 1024:.1f} KB)")

        # Show preview
        with st.expander("🔍 Raw Data Preview", expanded=False):
            preview_df = pd.read_csv(temp_path, nrows=10)
            st.dataframe(preview_df, use_container_width=True)

        # ── EXECUTION BLOCK ──────────────────────────────────────────────
        run_clicked = st.button("🚀 Run Full Analysis", type="primary", use_container_width=True)
        
        if run_clicked:
            # Clear previous state
            st.session_state["analysis_result"] = None
            st.session_state["analysis_complete"] = False
            
            job_output = OUTPUT_DIR / f"streamlit_{int(time.time())}"

            # Progress container
            progress_bar = st.progress(0, text="Initializing pipeline...")
            
            steps = [
                (0.05, "📥 Ingesting data..."),
                (0.15, "🩺 Assessing data quality..."),
                (0.30, "🧹 Cleaning & deduplicating..."),
                (0.45, "🔧 Applying intelligent repairs..."),
                (0.55, "🩺 Re-assessing quality..."),
                (0.65, "📈 Generating insights..."),
                (0.80, "📊 Building dashboard..."),
                (0.90, "📄 Creating report..."),
            ]

            for pct, msg in steps[:1]:
                progress_bar.progress(pct, text=msg)

            # Run the actual pipeline
            orchestrator = MasterOrchestrator()

            # Simulate progress updates
            for pct, msg in steps[1:]:
                progress_bar.progress(pct, text=msg)

            result: PipelineResult = orchestrator.run(temp_path, job_output)
            
            # STORE RESULT IN SESSION STATE
            st.session_state["analysis_result"] = result
            st.session_state["analysis_complete"] = True

            progress_bar.progress(1.0, text="✅ Analysis complete!")
            time.sleep(0.5)
            progress_bar.empty()
            st.balloons()
            st.rerun() # Force rerun to render from state

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

        else:
            st.error(f"❌ Pipeline failed: {', '.join(result.errors)}")


# ═══════════════════════════════════════════════════════════════════════
# TAB 2 — Multi-File Comparison
# ═══════════════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown("### 🔀 Multi-File Comparison")
    st.markdown("Upload multiple CSV files to compare their statistics side-by-side.")

    compare_files = st.file_uploader(
        "Upload CSV files for comparison",
        type=["csv"],
        accept_multiple_files=True,
        key="compare_upload",
    )

    if compare_files and len(compare_files) >= 2:
        if st.button("🔄 Compare Files", type="primary", use_container_width=True):
            comparison_data = []
            progress = st.progress(0, text="Analyzing files...")

            for idx, cfile in enumerate(compare_files):
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
