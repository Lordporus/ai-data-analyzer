"""
Dashboard UI Component
Handles the interactive dashboard logic: filters, smart charts, and drill-downs.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List, Optional
from pathlib import Path
import sys

# Ensure project root is on sys.path regardless of import order
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import BRAND_COLOR, BRAND_NAME
from orchestrator.master import PipelineResult
from agents.data_quality import score_color

# ── Styling Constants ────────────────────────────────────────────────
PLOT_BG = "#161b22"
PAPER_BG = "#0d1117"
TEXT_COLOR = "#e6edf3"
GRID_COLOR = "#30363d"


def render_interactive_dashboard(result: PipelineResult):
    """
    Main entry point for rendering the interactive dashboard.
    """
    original_df = result.repair.dataframe if result.repair else pd.DataFrame()
    if original_df.empty:
        st.warning("No data available for dashboard.")
        return

    # 1. Sidebar Filters
    filtered_df = _render_filters(original_df)
    
    # Store in session state as requested
    st.session_state["filtered_df"] = filtered_df
    
    row_count = len(filtered_df)
    if row_count == 0:
        st.warning("No records match the selected filters.")
        return

    st.markdown("---")
    
    # 2. Strategic Analysis (AI)
    # Use getattr to handle stale session state objects (backward compatibility)
    exec_summary = getattr(result.insight, "executive_summary", None)
    if result.insight and exec_summary:
        st.markdown("### 🧠 AI Strategic Analysis")
        _render_strategic_analysis(result.insight)
        st.markdown("---")
    
    # 3. Insight Context & KPIs
    _render_kpis(filtered_df, original_df)

    # 4. Forecast & Scenario Lab (New Phase 14)
    # Check if forecast data exists (it might not if data isn't time-series)
    forecast_result = getattr(result, "forecast", None)
    if forecast_result and forecast_result.is_time_series:
        st.markdown("### 🔮 Forecast & Scenario Lab")
        _render_forecast_lab(forecast_result, filtered_df)
        st.markdown("---")
    
    # 5. Ask Your Data (New Phase 15)
    st.markdown("### 💬 Ask Your Data")
    _render_nl_query_section(filtered_df)
    st.markdown("---")

    # 6. Smart Charts (Main View)
    st.markdown("### 📈 Interactive Analysis 2.0")
    _render_smart_charts(filtered_df)

    # 6. Drill-Down
    st.markdown("### 🔬 Drill-Down Analytics")
    _render_drill_down(filtered_df)
    
    # 5. Data Preview (Filtered)
    with st.expander(f"📋 View Filtered Data ({row_count} rows)", expanded=False):
        st.dataframe(filtered_df, use_container_width=True)


def _render_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Renders filters in the sidebar/expandable section."""
    filtered_df = df.copy()
    
    st.sidebar.header("🔍 Filters")
    
    # Identify column types
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    date_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

    # A. Quick Search
    search_term = st.sidebar.text_input("Text Search (All Columns)", placeholder="Type to search...")
    if search_term:
        mask = filtered_df.astype(str).apply(
            lambda x: x.str.contains(search_term, case=False, na=False)
        ).any(axis=1)
        filtered_df = filtered_df[mask]

    # B. Categorical Filters
    if cat_cols:
        st.sidebar.subheader("Categories")
        # Heuristic: Only show filters for cols with < 50 unique values
        valid_cat_cols = [c for c in cat_cols if df[c].nunique() < 50]
        
        for col in valid_cat_cols[:5]: # Limit to top 5 categorical columns to avoid clutter
            options = sorted(df[col].dropna().unique().tolist())
            selected = st.sidebar.multiselect(f"{col}", options, default=options)
            if selected:
                filtered_df = filtered_df[filtered_df[col].isin(selected)]
    
    # C. Numeric Range Filters
    if num_cols:
        st.sidebar.subheader("Numeric Ranges")
        # Only show for top 3 numeric cols to save space
        for col in num_cols[:3]:
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            if min_val < max_val:
                step = (max_val - min_val) / 100
                vals = st.sidebar.slider(
                    f"{col}", min_val, max_val, (min_val, max_val), step=step,
                    format="%.2f"
                )
                filtered_df = filtered_df[
                    (filtered_df[col] >= vals[0]) & (filtered_df[col] <= vals[1])
                ]

    return filtered_df


def _render_kpis(curr_df: pd.DataFrame, orig_df: pd.DataFrame):
    """Renders dynamic KPIs with deltas."""
    
    count_curr = len(curr_df)
    count_orig = len(orig_df)
    pct_total = (count_curr / count_orig) * 100 if count_orig > 0 else 0
    
    # Identify primary numeric metric (e.g., Revenue, Salary, Amount)
    num_cols = curr_df.select_dtypes(include="number").columns.tolist()
    # Prioritize columns with 'price', 'amount', 'salary', 'revenue', 'cost'
    priority_cols = [c for c in num_cols if any(k in c.lower() for k in ['price', 'amt', 'amount', 'salary', 'rev', 'cost', 'sales'])]
    primary_col = priority_cols[0] if priority_cols else (num_cols[0] if num_cols else None)
    
    cols = st.columns(4)
    
    # 1. Record Count
    cols[0].metric(
        "Records", 
        f"{count_curr:,}", 
        delta=f"{pct_total:.1f}% of total",
        delta_color="off"
    )
    
    if primary_col:
        curr_sum = curr_df[primary_col].sum()
        orig_sum = orig_df[primary_col].sum()
        delta_sum = ((curr_sum - orig_sum) / orig_sum) * 100 if orig_sum > 0 else 0
        
        curr_avg = curr_df[primary_col].mean()
        orig_avg = orig_df[primary_col].mean()
        delta_avg = ((curr_avg - orig_avg) / orig_avg) * 100 if orig_avg > 0 else 0
        
        cols[1].metric(
            f"Total {primary_col}", 
            _human_format(curr_sum), 
            delta=f"{delta_sum:+.1f}%"
        )
        cols[2].metric(
            f"Avg {primary_col}", 
            _human_format(curr_avg), 
            delta=f"{delta_avg:+.1f}%"
        )
        
    # 4. Filter Context
    with cols[3]:
        st.markdown(f"""
        <div style="background:{PLOT_BG}; padding:10px; border-radius:8px; border:1px solid {GRID_COLOR}; font-size:0.8rem; height: 80px; display:flex; align-items:center;">
            <span>ℹ️ <b>Insight Context</b><br>
            Filtering has retained <b>{count_curr}</b> rows. 
            Top impact driver: <b>{primary_col or 'N/A'}</b>.</span>
        </div>
        """, unsafe_allow_html=True)


def _render_smart_charts(df: pd.DataFrame):
    """Interactive Analysis 2.0 — Product-grade analytics interface."""

    # ── CSS Animation ─────────────────────────────────────────────
    st.markdown("""
    <style>
    .ia2-fade { animation: ia2FadeIn 0.4s ease-in; }
    @keyframes ia2FadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
    .ia2-card { background:#161b22; border:1px solid #30363d; border-radius:8px; padding:14px 16px; text-align:center; }
    .ia2-card-label { font-size:0.72rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:4px; }
    .ia2-card-value { font-size:1.25rem; font-weight:700; color:#e6edf3; }
    .ia2-card-sub { font-size:0.7rem; color:#8b949e; margin-top:2px; }
    .ia2-strip { display:flex; gap:8px; margin-bottom:12px; }
    .ia2-strip-item { flex:1; background:#161b22; border:1px solid #30363d; border-radius:6px; padding:10px 12px; }
    .ia2-strip-label { font-size:0.65rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.4px; }
    .ia2-strip-val { font-size:0.95rem; font-weight:600; color:#e6edf3; margin-top:2px; }
    .ia2-interp { background:#161b22; border-left:3px solid #58a6ff; padding:12px 16px; border-radius:4px; margin-top:12px; font-size:0.85rem; color:#c9d1d9; }
    .ia2-divider { border:0; border-top:1px solid #21262d; margin:16px 0; }
    </style>
    """, unsafe_allow_html=True)

    # ── Column Detection ──────────────────────────────────────────
    all_cols = df.columns.tolist()
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    date_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

    # ── Smart Defaults ────────────────────────────────────────────
    def _best_x():
        if date_cols: return date_cols[0]
        if cat_cols: return cat_cols[0]
        return all_cols[0] if all_cols else None

    def _best_y():
        if not num_cols: return None
        # Prefer highest-variance numeric column
        variances = {c: df[c].var() for c in num_cols if df[c].var() == df[c].var()}  # filter NaN
        if variances:
            return max(variances, key=variances.get)
        return num_cols[0]

    default_x = _best_x()
    default_y = _best_y()

    if not default_x or not default_y:
        st.info("Insufficient columns for interactive analysis.")
        return

    # ── Controls ──────────────────────────────────────────────────
    ctrl_col, chart_col = st.columns([1, 2.5])

    with ctrl_col:
        st.markdown("#### 🎛️ Visualization Controls")
        st.markdown('<hr class="ia2-divider">', unsafe_allow_html=True)

        x_idx = all_cols.index(default_x) if default_x in all_cols else 0
        y_idx = num_cols.index(default_y) if default_y in num_cols else 0

        x_axis = st.selectbox("X-Axis (Dimension)", options=all_cols, index=x_idx, key="ia2_x")
        y_axis = st.selectbox("Y-Axis (Metric)", options=num_cols, index=y_idx, key="ia2_y")
        chart_type = st.radio("Chart Type", ["Auto", "Bar", "Line", "Scatter", "Box", "Histogram"], horizontal=True, key="ia2_ct")

        st.markdown('<hr class="ia2-divider">', unsafe_allow_html=True)
        st.markdown("##### Overlays")
        show_trendline = st.checkbox("Show Trendline", value=False, key="ia2_trend")
        show_rolling = st.checkbox("Rolling Average (5)", value=False, key="ia2_roll")
        show_outliers = st.checkbox("Highlight Outliers", value=False, key="ia2_out")
        show_all_cats = st.checkbox("Show All Categories", value=False, key="ia2_allcat")

    with chart_col:
        if x_axis not in df.columns or y_axis not in df.columns:
            st.warning("Selected columns not found in dataset.")
            return

        # ── Insight Summary Strip ─────────────────────────────────
        y_data = df[y_axis].dropna()
        y_mean = float(y_data.mean()) if len(y_data) > 0 else 0
        y_std = float(y_data.std()) if len(y_data) > 1 else 0
        cv = (y_std / abs(y_mean)) * 100 if abs(y_mean) > 1e-9 else 0

        # Trend direction (simple: compare first half mean vs second half mean)
        half = len(y_data) // 2
        if half > 0:
            first_half = float(y_data.iloc[:half].mean())
            second_half = float(y_data.iloc[half:].mean())
            if second_half > first_half * 1.02:
                trend_dir, trend_icon = "Upward", "📈"
            elif second_half < first_half * 0.98:
                trend_dir, trend_icon = "Downward", "📉"
            else:
                trend_dir, trend_icon = "Stable", "➡️"
        else:
            trend_dir, trend_icon = "Stable", "➡️"

        vol_tier = "Low" if cv < 10 else ("Moderate" if cv < 25 else "High")
        vol_color = "#3fb950" if vol_tier == "Low" else ("#d29922" if vol_tier == "Moderate" else "#f85149")
        conf_tier = "High" if len(y_data) > 100 else ("Moderate" if len(y_data) > 30 else "Low")

        st.markdown(f"""
        <div class="ia2-fade">
        <div class="ia2-strip">
            <div class="ia2-strip-item">
                <div class="ia2-strip-label">Trend Direction</div>
                <div class="ia2-strip-val">{trend_icon} {trend_dir}</div>
            </div>
            <div class="ia2-strip-item">
                <div class="ia2-strip-label">Volatility</div>
                <div class="ia2-strip-val" style="color:{vol_color}">{vol_tier} ({cv:.1f}%)</div>
            </div>
            <div class="ia2-strip-item">
                <div class="ia2-strip-label">Confidence</div>
                <div class="ia2-strip-val">{conf_tier}</div>
            </div>
            <div class="ia2-strip-item">
                <div class="ia2-strip-label">Impact Driver</div>
                <div class="ia2-strip-val">{y_axis}</div>
            </div>
        </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Mini KPI Row ──────────────────────────────────────────
        y_median = float(y_data.median()) if len(y_data) > 0 else 0
        q1, q3 = float(y_data.quantile(0.25)), float(y_data.quantile(0.75))
        iqr = q3 - q1
        outlier_count = int(((y_data < q1 - 1.5 * iqr) | (y_data > q3 + 1.5 * iqr)).sum())

        st.markdown(f"""
        <div class="ia2-fade">
        <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:8px; margin-bottom:12px;">
            <div class="ia2-card">
                <div class="ia2-card-label">Mean</div>
                <div class="ia2-card-value">{_human_format(y_mean)}</div>
            </div>
            <div class="ia2-card">
                <div class="ia2-card-label">Median</div>
                <div class="ia2-card-value">{_human_format(y_median)}</div>
            </div>
            <div class="ia2-card">
                <div class="ia2-card-label">Std Dev</div>
                <div class="ia2-card-value">{_human_format(y_std)}</div>
            </div>
            <div class="ia2-card">
                <div class="ia2-card-label">Outliers</div>
                <div class="ia2-card-value">{outlier_count}</div>
                <div class="ia2-card-sub">IQR Method</div>
            </div>
        </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Chart Construction ────────────────────────────────────
        is_x_cat = df[x_axis].dtype == 'object' or df[x_axis].dtype.name == 'category'
        is_x_num = pd.api.types.is_numeric_dtype(df[x_axis])
        is_x_date = pd.api.types.is_datetime64_any_dtype(df[x_axis])

        final_type = chart_type
        if chart_type == "Auto":
            if is_x_date: final_type = "Line"
            elif is_x_cat: final_type = "Bar"
            elif is_x_num: final_type = "Scatter"

        cat_limit = None if show_all_cats else 25
        fig = go.Figure()

        if final_type == "Bar":
            if x_axis == y_axis:
                counts = df[x_axis].value_counts()
                if cat_limit: counts = counts.head(cat_limit)
                labels = [str(l)[:20] + "…" if len(str(l)) > 20 else str(l) for l in counts.index]
                fig = go.Figure(go.Bar(x=labels, y=counts.values, marker_color=BRAND_COLOR,
                                      customdata=counts.index, hovertemplate="%{customdata}: %{y}<extra></extra>"))
                fig.update_layout(xaxis_title=x_axis, yaxis_title="Count")
            else:
                agg_df = df.groupby(x_axis, as_index=False)[y_axis].sum().sort_values(y_axis, ascending=False)
                if cat_limit: agg_df = agg_df.head(cat_limit)
                labels = [str(l)[:20] + "…" if len(str(l)) > 20 else str(l) for l in agg_df[x_axis]]
                fig = go.Figure(go.Bar(x=labels, y=agg_df[y_axis].values, marker_color=BRAND_COLOR,
                                      customdata=agg_df[x_axis].values, hovertemplate="%{customdata}: %{y:,.2f}<extra></extra>"))
                fig.update_layout(xaxis_title=x_axis, yaxis_title=y_axis)

        elif final_type == "Line":
            agg_df = df.groupby(x_axis)[y_axis].sum().reset_index().sort_values(x_axis)
            fig = px.line(agg_df, x=x_axis, y=y_axis, markers=True, color_discrete_sequence=[BRAND_COLOR])
            # Rolling Average Overlay
            if show_rolling and len(agg_df) >= 5:
                rolling_vals = agg_df[y_axis].rolling(window=5, min_periods=1).mean()
                fig.add_trace(go.Scatter(x=agg_df[x_axis], y=rolling_vals, mode='lines',
                                         name='Rolling Avg (5)', line=dict(color='#d29922', dash='dash', width=2)))

        elif final_type == "Scatter":
            fig = px.scatter(df, x=x_axis, y=y_axis, opacity=0.7, color_discrete_sequence=[BRAND_COLOR])

        elif final_type == "Box":
            fig = px.box(df, x=x_axis, y=y_axis, color_discrete_sequence=[BRAND_COLOR])

        elif final_type == "Histogram":
            fig = px.histogram(df, x=x_axis, y=y_axis, color_discrete_sequence=[BRAND_COLOR], nbins=30)

        # ── Trendline Overlay ─────────────────────────────────────
        if show_trendline and is_x_num and final_type in ["Scatter", "Line"]:
            try:
                x_vals = df[x_axis].dropna().values.astype(float)
                y_vals = df[y_axis].dropna().values.astype(float)
                min_len = min(len(x_vals), len(y_vals))
                x_vals, y_vals = x_vals[:min_len], y_vals[:min_len]
                if min_len > 2:
                    z = np.polyfit(x_vals, y_vals, 1)
                    p = np.poly1d(z)
                    x_sorted = np.sort(x_vals)
                    fig.add_trace(go.Scatter(x=x_sorted, y=p(x_sorted), mode='lines',
                                             name='Trendline', line=dict(color='#f85149', dash='dot', width=2)))
            except Exception:
                pass

        # ── Outlier Highlight ─────────────────────────────────────
        if show_outliers and final_type in ["Scatter", "Bar", "Line"]:
            outlier_mask = (y_data < q1 - 1.5 * iqr) | (y_data > q3 + 1.5 * iqr)
            outlier_df = df.loc[outlier_mask.index[outlier_mask]]
            if not outlier_df.empty and x_axis in outlier_df.columns and y_axis in outlier_df.columns:
                fig.add_trace(go.Scatter(x=outlier_df[x_axis], y=outlier_df[y_axis], mode='markers',
                                         name='Outliers', marker=dict(color='#f85149', size=10, symbol='x')))

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=PLOT_BG,
            plot_bgcolor=PAPER_BG,
            margin=dict(l=40, r=20, t=40, b=40),
            height=460,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True, key="ia2_chart")

        # ── Interpretation Footer ─────────────────────────────────
        if cv > 25:
            interp = f"{y_axis} exhibits elevated dispersion (CV: {cv:.1f}%), suggesting sensitivity to underlying operational drivers. Further segmentation is recommended."
        elif trend_dir == "Upward":
            interp = f"{y_axis} shows a sustained upward trajectory across the observed period, indicating progressive performance expansion."
        elif trend_dir == "Downward":
            interp = f"{y_axis} reflects downward drift, signaling potential structural pressure requiring monitoring."
        else:
            interp = f"{y_axis} remains structurally stable with negligible directional shift, indicating consistent operational control."

        st.markdown(f'<div class="ia2-interp"><b>Interpretation:</b> {interp}</div>', unsafe_allow_html=True)

        # ── Export Utilities ───────────────────────────────────────
        exp_cols = st.columns([1, 1, 1, 3])
        with exp_cols[0]:
            csv_buffer = io.StringIO()
            df[[x_axis, y_axis]].to_csv(csv_buffer, index=False)
            st.download_button("📥 CSV", csv_buffer.getvalue(), file_name="analysis_export.csv", mime="text/csv", key="ia2_csv")
        with exp_cols[1]:
            st.download_button("📋 Insight", interp, file_name="insight_summary.txt", mime="text/plain", key="ia2_insight")



def _render_drill_down(df: pd.DataFrame):
    """Drill-down view with breadcrumb navigation."""

    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if not cat_cols:
        st.info("No categorical columns available for breakdown.")
        return

    # ── Session State for Drill Path ──────────────────────────────
    if "drill_path" not in st.session_state:
        st.session_state["drill_path"] = []

    drill_path = st.session_state["drill_path"]

    # ── Dimension Selection ───────────────────────────────────────
    col1, col2 = st.columns([1, 2])
    with col1:
        dim = st.selectbox("Breakdown Dimension", cat_cols, key="dd_dim")

    # ── Breadcrumb ────────────────────────────────────────────────
    if drill_path:
        crumbs = " → ".join([f"<b>{d['dim']}</b>: {d['val']}" for d in drill_path])
        bc_html = (
            f'<div style="background:#161b22; border:1px solid #30363d; border-radius:6px; '
            f'padding:8px 14px; margin-bottom:12px; font-size:0.82rem; color:#c9d1d9;">'
            f'🧭 {crumbs}</div>'
        )
        st.markdown(bc_html, unsafe_allow_html=True)
        if st.button("↩ Reset Drill-Down", key="dd_reset"):
            st.session_state["drill_path"] = []
            st.rerun()

    # ── Apply existing drill filters ──────────────────────────────
    drill_df = df.copy()
    for step in drill_path:
        if step["dim"] in drill_df.columns:
            drill_df = drill_df[drill_df[step["dim"]] == step["val"]]

    # ── Top categories ────────────────────────────────────────────
    if dim not in drill_df.columns:
        return

    top_cats = drill_df[dim].value_counts().head(10).index.tolist()
    if not top_cats:
        st.info("No categories to display.")
        return

    with col2:
        selected_cat = st.radio("Select Category", top_cats, horizontal=True, key="dd_cat")

    if selected_cat:
        # Check if user can drill deeper
        subset = drill_df[drill_df[dim] == selected_cat]
        remaining_dims = [c for c in cat_cols if c != dim and subset[c].nunique() > 1]

        # ── Breadcrumb drill-in button ────────────────────────────
        if remaining_dims:
            if st.button(f"🔽 Drill into \"{selected_cat}\"", key="dd_drill_in"):
                st.session_state["drill_path"].append({"dim": dim, "val": selected_cat})
                st.rerun()

        st.markdown(f"#### 🔍 Report for: **{selected_cat}**")

        # Metrics for this category
        num_cols_dd = subset.select_dtypes(include="number").columns.tolist()
        if num_cols_dd:
            m_cols = st.columns(min(len(num_cols_dd), 4))
            for i, c in enumerate(num_cols_dd[:4]):
                val = subset[c].sum()
                avg = subset[c].mean()
                m_cols[i].metric(c, f"{_human_format(val)}", f"Avg: {_human_format(avg)}")

        st.dataframe(subset.head(50), use_container_width=True)



def _human_format(num):
    """Format numbers human readable (e.g. 1.2K, 1M)."""
    if num is None: return "0"
    num = float('{:.3g}'.format(num))
    suffixes = ['', 'K', 'M', 'B', 'T', 'Qa', 'Qi']
    magnitude = 0
    while abs(num) >= 1000 and magnitude < len(suffixes) - 1:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), suffixes[magnitude])


def _render_strategic_analysis(insight):
    """Render Executive Summary, Risks, and Opportunities."""
    
    # Executive Summary
    st.info(f"**Executive Summary:** {insight.executive_summary}")
    
    # Audit Trail / Calculation Panel
    if getattr(insight, "audit_log", None):
        with st.expander("ⓘ How was this calculated? (Audit Trail & Lineage)"):
            st.json(insight.audit_log)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if insight.top_risks:
            st.markdown("#### ⚠️ Operational Risks")
            for risk in insight.top_risks:
                # Handle both dict (new) and string (legacy)
                if isinstance(risk, dict):
                    cat = risk.get("category", "General")
                    desc = risk.get("detail", "")
                    st.markdown(
                        f"""<div style="padding:10px; border-left:4px solid #ef4444; background:#1f1313; margin-bottom:8px; border-radius:4px;">
                        <span style="color:#ef4444; font-weight:bold; font-size:0.8em; text-transform:uppercase;">{cat} RISK</span><br>
                        {desc}
                        </div>""", unsafe_allow_html=True
                    )
                else:
                    st.warning(str(risk))

    with col2:
        if insight.top_opportunities:
            st.markdown("#### 🚀 Growth Opportunities")
            for opp in insight.top_opportunities:
                st.markdown(
                    f"""<div style="padding:10px; border-left:4px solid #22c55e; background:#131f16; margin-bottom:8px; border-radius:4px;">
                    <span style="color:#22c55e; font-weight:bold; font-size:0.8em; text-transform:uppercase;">OPPORTUNITY</span><br>
                    {opp}
                    </div>""", unsafe_allow_html=True
                )


    if insight.key_relationships:
        with st.expander("🔗 Key Relationships (Non-Obvious)", expanded=False):
            for rel in insight.key_relationships:
                 st.markdown(f"- {rel}")


def _render_forecast_lab(forecast, df: pd.DataFrame):
    """
    Renders the Forecast & Scenario Lab with interactive sliders.
    Uses 'forecast' object (ForecastResult) and current 'df' (for context).
    """
    # ── Stability Guard: Datetime Presence ───────────────────────────────
    date_cols = df.select_dtypes(include=["datetime64", "datetime", "datetimetz"])
    
    if date_cols.empty:
        st.markdown("#### Forecast Unavailable")
        st.info(
            "**No time-series column detected in this dataset.**\n\n"
            "To enable forecasting, include a Date or Time column with at least 8 time-based records."
        )
        return

    if not forecast or not forecast.available_metrics:
        st.markdown("#### Forecast Unavailable")
        st.warning(
            "**Dataset insufficient for reliable forecasting.**\n\n"
            "Analysis requires at least 8 time-based records or higher data confidence thresholds."
        )
        return
        
    # ── Advanced Multivariate VAR Setup ──────────────────────────────────
    from agents.report_validation import ReportValidationEngine
    val_engine = ReportValidationEngine()
    num_cols = df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in num_cols if not val_engine._is_structural_column(c) and df[c].nunique() > 5]
    can_use_var = len(numeric_cols) >= 2 and len(df) >= 30
    
    var_mode = False
    var_results = None
    selected_var_cols = []
    
    if can_use_var:
        st.markdown("### 🎛️ Advanced Forecasting Controls")
        var_mode = st.toggle(
            "⚡ Advanced Mode: Joint Multivariate Forecasting (VAR)",
            value=False,
            help="Forecast multiple columns together using Vector Autoregression (VAR), capturing cross-column feedback loops and correlations."
        )
        if var_mode:
            selected_var_cols = st.multiselect(
                "Select Variables for Joint VAR Model",
                options=numeric_cols,
                default=numeric_cols[:2],
                help="Select 2 or more numeric columns to model together."
            )
            if len(selected_var_cols) < 2:
                st.warning("⚠️ Please select at least 2 columns to run a multivariate VAR model.")
                return
                
            with st.spinner("Fitting VAR model & projecting joint matrices..."):
                from agents.forecast import ForecastAgent
                agent = ForecastAgent()
                var_results = agent.forecast_multivariate(df, selected_var_cols)
                
            if not var_results:
                st.error("❌ Failed to fit joint VAR model. Check for multicollinearity (perfectly correlated columns) or missing data.")
                return
                
            c1, c2, c3 = st.columns([1, 2, 1])
            with c1:
                metric = st.selectbox(
                    "Select Metric to Focus Plot",
                    options=selected_var_cols,
                    index=0
                )
            with c2:
                scenario_pct = st.slider(
                    "Scenario Simulation: Impact Driver %",
                    min_value=-20, max_value=20, value=0, step=1,
                    format="%d%%",
                    help="Simulate an increase or decrease in the underlying driver (e.g. price, volume)."
                )
                
            # Construct VAR f_data
            date_col = forecast.primary_date_col
            period = forecast.period_type
            
            ts_df = df[[date_col] + selected_var_cols].dropna().sort_values(by=date_col)
            ts_df = ts_df.set_index(date_col)
            if not isinstance(ts_df.index, pd.DatetimeIndex):
                ts_df.index = pd.to_datetime(ts_df.index, errors="coerce")
            ts_df = ts_df.dropna(subset=[ts_df.index.name]) if (ts_df.index.name and ts_df.index.name in ts_df.columns) else ts_df.dropna()
            
            _vol_keywords = {"sales", "revenue", "count", "qty", "quantity", "total", "volume", "orders"}
            _is_volume = any(kw in metric.lower() for kw in _vol_keywords)
            agg_fn = "sum" if _is_volume else "mean"
            ts_resampled = ts_df.resample(period).agg({c: agg_fn for c in selected_var_cols})
            ts_resampled = ts_resampled.interpolate(method='linear')
            
            hist_dates = ts_resampled.index.strftime('%Y-%m-%d').tolist()
            hist_values = ts_resampled[metric].values.tolist()
            
            periods = 10
            last_date = ts_resampled.index[-1]
            freq_offset = pd.tseries.frequencies.to_offset(period)
            future_dates = [last_date + (i * freq_offset) for i in range(1, periods + 1)]
            future_dates_str = [d.strftime('%Y-%m-%d') for d in future_dates]
            
            f_data = {
                "dates_hist": hist_dates,
                "values_hist": hist_values,
                "dates_forecast": future_dates_str,
                "values_forecast": var_results[metric]["forecast"],
                "lower_bound": var_results[metric]["lower"],
                "upper_bound": var_results[metric]["upper"],
                "r2": var_results[metric]["r2"],
                "forecast_model_type": "VAR (Vector Autoregression)",
                "interpretation": {
                    "model_type": "VAR",
                    "confidence": "MEDIUM",
                    "volatility_comment": "Joint multivariate lag-based correlation model.",
                    "seasonality_comment": f"VAR model fit with lag order k={var_results[metric]['k_ar']}.",
                    "business_summary": f"Joint multivariate forecast for '{metric}' taking feedback and cross-column interactions with {', '.join([c for c in selected_var_cols if c != metric])} into account.",
                    "primary_risk": "Multi-variable parameter sensitivity may increase variance over long horizons.",
                    "primary_opportunity": "Captures dynamic feedback loops that single-column models ignore.",
                    "confidence_comment": f"Model evaluated with R² = {var_results[metric]['r2']:.4f}."
                }
            }
            
    if not var_mode:
        # 1. Config Row
        c1, c2, c3 = st.columns([1, 2, 1])
        
        with c1:
            metric = st.selectbox(
                "Select Metric to Forecast", 
                options=forecast.available_metrics,
                index=0
            )
            
        with c2:
            scenario_pct = st.slider(
                "Scenario Simulation: Impact Driver %", 
                min_value=-20, max_value=20, value=0, step=1,
                format="%d%%",
                help="Simulate an increase or decrease in the underlying driver (e.g. price, volume)."
            )
    
        # 2. Get Forecast Data
        f_data = forecast.get_forecast(metric)
        if not f_data:
            st.warning(f"No statistical model could be derived for the selected metric '{metric}'.")
            return

    # 3. Apply Scenario Logic (Simple scaling)
    # If scenario_pct is +10%, we scale future values by 1.10
    scale_factor = 1 + (scenario_pct / 100.0)
    
    hist_dates = f_data["dates_hist"]
    hist_values = f_data["values_hist"]
    
    future_dates = f_data["dates_forecast"]
    future_values = [v * scale_factor for v in f_data["values_forecast"]]
    lower_bound = [v * scale_factor for v in f_data["lower_bound"]]
    upper_bound = [v * scale_factor for v in f_data["upper_bound"]]
    
    # 4. Display Impact Summary
    with c3:
        # Compare base forecast sum vs simulated forecast sum
        base_sum = sum(f_data["values_forecast"])
        sim_sum = sum(future_values)
        delta = sim_sum - base_sum
        
        delta_color = "normal"
        if delta > 0: delta_color = "g" # green (using emoji/style workaround)
        elif delta < 0: delta_color = "r" # red
        
        st.metric(
            f"Projected {metric} (Next 10 periods)",
            _human_format(sim_sum),
            delta=_human_format(delta),
            help="Difference between base forecast and simulated scenario."
        )

    # 5. Render Plotly Chart
    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=hist_dates, y=hist_values,
        mode='lines', name='Historical',
        line=dict(color='#94a3b8', width=2)
    ))
    
    # Forecast (Solid if no scenario, else dashed base + colored sim)
    if scenario_pct == 0:
        fig.add_trace(go.Scatter(
            x=future_dates, y=future_values,
            mode='lines', name='Forecast',
            line=dict(color=BRAND_COLOR, width=3, dash='dash')
        ))
        
        # Confidence Interval (Shaded)
        fig.add_trace(go.Scatter(
            x=future_dates + future_dates[::-1],
            y=upper_bound + lower_bound[::-1],
            fill='toself',
            fillcolor='rgba(108, 99, 255, 0.2)', # Brand color with opacity
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False
        ))
        
    else:
        # Scenario Mode
        # Show Base Forecast as faint dashed
        fig.add_trace(go.Scatter(
            x=future_dates, y=f_data["values_forecast"],
            mode='lines', name='Base Baseline',
            line=dict(color='#475569', width=1, dash='dot')
        ))
        
        # Show Simulated Forecast as vibrant
        sim_color = "#22c55e" if scenario_pct > 0 else "#ef4444"
        fig.add_trace(go.Scatter(
            x=future_dates, y=future_values,
            mode='lines', name=f'Scenario ({scenario_pct:+}%)',
            line=dict(color=sim_color, width=3, dash='dash')
        ))

    fig.update_layout(
        title=f"Forecast: {metric} ({forecast.period_type}-level)",
        template="plotly_dark",
        paper_bgcolor=PLOT_BG, 
        plot_bgcolor=PAPER_BG,
        font=dict(family="Inter, sans-serif"),
        hovermode="x unified",
        margin=dict(l=40, r=20, t=40, b=40),
        height=350,
        legend=dict(orientation="h", y=1.1)
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # Forecast Audit Trail / Model Lineage
    if var_mode and var_results is not None:
        with st.expander(f"ⓘ How was '{metric}' joint VAR forecast calculated? (Model Lineage)"):
            st.json({
                "columns_used": selected_var_cols,
                "formula": "Vector Autoregression (VAR) Model (statsmodels.tsa.vector_ar.var_model.VAR)",
                "threshold": f"Akaike Information Criterion (AIC) optimal lag order k={var_results[metric]['k_ar']}",
                "method": "deterministic",
                "result": f"Forecasted {metric} taking cross-column correlation and feed-forward lags into account (R² = {f_data['r2']:.4f})."
            })
    elif getattr(forecast, "audit_log", None) and metric in forecast.audit_log:
        with st.expander(f"ⓘ How was '{metric}' forecast calculated? (Model Lineage)"):
            st.json(forecast.audit_log[metric])


def _render_nl_query_section(df: pd.DataFrame):
    """
    Renders the 'Ask Your Data' interface using NLQueryAgent.

    Results are stored in st.session_state["nl_query_result"] so they persist
    across Streamlit reruns (sidebar filter changes, scrolling, etc.).
    Without this, the result disappears on the very next script rerun because
    the button state resets to False and the inline render block is skipped.
    """
    # Dynamic import to avoid circular dependency
    from agents.nl_query import NLQueryAgent

    # Initialise persistent state slots
    if "nl_query_result" not in st.session_state:
        st.session_state["nl_query_result"] = None
    if "nl_query_last_query" not in st.session_state:
        st.session_state["nl_query_last_query"] = ""

    with st.expander("💬 Ask a question about this data", expanded=True):
        col1, col2 = st.columns([4, 1])
        with col1:
            query = st.text_input(
                "Query",
                placeholder="e.g. 'Show me the trend of Sales over time' or 'Which Region has the highest Profit?'",
                label_visibility="collapsed",
                key="nl_query_input",
            )
        with col2:
            analyze_btn = st.button("Analyze", type="primary", use_container_width=True, key="nl_analyze_btn")

        # Clear stored result when the user types a new query
        if query != st.session_state["nl_query_last_query"]:
            st.session_state["nl_query_result"] = None
            st.session_state["nl_query_last_query"] = query

        # Run the agent and store result in session_state
        if analyze_btn and query:
            with st.spinner("Analyzing your question..."):
                try:
                    agent = NLQueryAgent()
                    nl_result = agent.run({"query": query, "df": df})
                    st.session_state["nl_query_result"] = {
                        "error": nl_result.error,
                        "explanation": nl_result.explanation,
                        "chart_config": nl_result.chart_config,
                        "confidence_level": nl_result.confidence_level,
                    }
                except Exception as exc:
                    st.session_state["nl_query_result"] = {
                        "error": str(exc),
                        "explanation": "",
                        "chart_config": None,
                        "confidence_level": "error",
                    }

        # Always render from session_state so results survive reruns
        stored = st.session_state.get("nl_query_result")
        if stored:
            if stored.get("error"):
                st.error(stored["error"])
                if stored.get("explanation"):
                    st.info(stored["explanation"])
            else:
                confidence = stored.get("confidence_level", "")
                badge = "🤖 AI" if confidence == "LLM" else "📐 Deterministic"
                st.success(f"{badge} — {stored['explanation']}")

                if stored.get("chart_config"):
                    _render_dynamic_chart(df, stored["chart_config"])


def _render_dynamic_chart(df: pd.DataFrame, config: Dict[str, Any]):
    """
    Renders a Plotly chart based on the LLM-generated configuration.
    Config schema: {type, x, y, agg, title}
    """
    chart_type = config.get("type", "bar").lower()
    x = config.get("x")
    y = config.get("y")
    agg = config.get("agg")
    title = config.get("title", f"{chart_type.title()} Chart")
    
    if x not in df.columns:
        st.warning(f"Chart Error: Column '{x}' not found.")
        return

    # Aggregate if requested
    chart_df = df.copy()
    if agg in ["sum", "mean", "count"]:
        if isinstance(y, list):
            grp_cols = [x]
        else:
            grp_cols = [x]
        
        if agg == "sum":
            chart_df = chart_df.groupby(grp_cols, as_index=False)[y].sum()
        elif agg == "mean":
            chart_df = chart_df.groupby(grp_cols, as_index=False)[y].mean()
        elif agg == "count":
            chart_df = chart_df.groupby(grp_cols, as_index=False)[y].count()
            
    fig = go.Figure()
    
    # 1. Bar Chart
    if "bar" in chart_type:
        fig = px.bar(chart_df, x=x, y=y, title=title, color_discrete_sequence=[BRAND_COLOR])
        
    # 2. Line Chart
    elif "line" in chart_type:
        # lines usually imply sorting by X
        chart_df = chart_df.sort_values(x)
        fig = px.line(chart_df, x=x, y=y, title=title, markers=True, color_discrete_sequence=[BRAND_COLOR])
        
    # 3. Scatter Chart
    elif "scatter" in chart_type:
        fig = px.scatter(chart_df, x=x, y=y, title=title, color_discrete_sequence=[BRAND_COLOR])
        
    # 4. Pie Chart
    elif "pie" in chart_type:
        fig = px.pie(chart_df, names=x, values=y, title=title, color_discrete_sequence=px.colors.sequential.Bluyl)
        
    # 5. Histogram
    elif "hist" in chart_type:
        fig = px.histogram(chart_df, x=x, y=y, title=title, color_discrete_sequence=[BRAND_COLOR])
        
    # 6. Box Plot
    elif "box" in chart_type:
        fig = px.box(chart_df, x=x, y=y, title=title, color_discrete_sequence=[BRAND_COLOR])

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=PLOT_BG, 
        plot_bgcolor=PAPER_BG,
        font=dict(family="Inter, sans-serif"),
        height=400,
        margin=dict(l=40, r=20, t=40, b=40)
    )
    
    st.plotly_chart(fig, use_container_width=True)

