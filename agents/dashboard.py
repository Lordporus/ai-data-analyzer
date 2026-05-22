"""
DashboardAgent — Generate an interactive Plotly dashboard as a
self-contained HTML file.

Input:  InsightResult
Output: DashboardResult (path to HTML dashboard)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from agents.base import BaseAgent
from agents.insight import InsightResult
from agents.forecast import ForecastResult
from agents.data_quality import DataQualityResult, score_color
from config.settings import BRAND_COLOR, BRAND_NAME

logger = logging.getLogger(__name__)


@dataclass
class DashboardResult:
    html_path: str = ""
    chart_count: int = 0
    kpi_count: int = 0


class DashboardAgent(BaseAgent):
    """Build an interactive HTML dashboard from insight results."""

    name = "DashboardAgent"

    def _execute(self, input_data: dict) -> DashboardResult:
        insight: InsightResult = input_data["insight"]
        # forecast: ForecastResult | None = input_data.get("forecast") # Not used in HTML dash, but available
        quality_before: DataQualityResult | None = input_data.get("quality_before")
        quality_after: DataQualityResult | None = input_data.get("quality_after")
        output_dir: Path = Path(input_data["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        df = insight.dataframe
        types = insight.detected_types
        num_cols = [c for c, t in types.items() if t == "numeric" and c in df.columns]
        cat_cols = [c for c, t in types.items() if t == "categorical" and c in df.columns]

        figures: List[str] = []
        chart_count = 0

        # ── Data Health Overview (gauge + comparison) ─────────────────
        health_html = ""
        if quality_after:
            qa = quality_after
            health_html = self._build_health_section(quality_before, qa)
            chart_count += 1

        # ── KPI Cards ────────────────────────────────────────────────
        kpi_html = self._build_kpi_cards(insight.kpi_list[:12])

        # ── Histograms for numeric columns ───────────────────────────
        for col in num_cols[:6]:
            fig = go.Figure(go.Histogram(
                x=df[col], nbinsx=30,
                marker_color=BRAND_COLOR, opacity=0.85,
            ))
            fig.update_layout(
                title=f"Distribution: {col}",
                xaxis_title=col, yaxis_title="Frequency",
                template="plotly_dark",
                paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                font=dict(family="Inter, sans-serif"),
                margin=dict(l=40, r=20, t=50, b=40),
            )
            figures.append(fig.to_html(full_html=False, include_plotlyjs=False))
            chart_count += 1

        # ── Bar charts for categorical columns ───────────────────────
        for col in cat_cols[:4]:
            counts = df[col].value_counts().head(15)
            fig = go.Figure(go.Bar(
                x=counts.index.astype(str), y=counts.values,
                marker_color=BRAND_COLOR, opacity=0.85,
            ))
            fig.update_layout(
                title=f"Distribution: {col}",
                xaxis_title=col, yaxis_title="Count",
                template="plotly_dark",
                paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                font=dict(family="Inter, sans-serif"),
                margin=dict(l=40, r=20, t=50, b=40),
            )
            figures.append(fig.to_html(full_html=False, include_plotlyjs=False))
            chart_count += 1

        # ── Correlation heatmap ──────────────────────────────────────
        if insight.correlation_matrix is not None and len(num_cols) >= 2:
            corr = insight.correlation_matrix
            fig = go.Figure(go.Heatmap(
                z=corr.values, x=corr.columns.tolist(),
                y=corr.index.tolist(),
                colorscale="RdBu_r", zmid=0,
                text=corr.values.round(2), texttemplate="%{text}",
            ))
            fig.update_layout(
                title="Correlation Matrix",
                template="plotly_dark",
                paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                font=dict(family="Inter, sans-serif"),
                height=500,
                margin=dict(l=80, r=20, t=50, b=80),
            )
            figures.append(fig.to_html(full_html=False, include_plotlyjs=False))
            chart_count += 1

        # ── Scatter plot for top correlated pair ─────────────────────
        if insight.correlation_matrix is not None and len(num_cols) >= 2:
            corr = insight.correlation_matrix
            max_corr = 0
            pair = (num_cols[0], num_cols[1])
            for i, c1 in enumerate(num_cols):
                for c2 in num_cols[i + 1:]:
                    if abs(corr.loc[c1, c2]) > max_corr:
                        max_corr = abs(corr.loc[c1, c2])
                        pair = (c1, c2)

            fig = go.Figure(go.Scatter(
                x=df[pair[0]], y=df[pair[1]],
                mode="markers",
                marker=dict(color=BRAND_COLOR, opacity=0.6, size=6),
            ))
            fig.update_layout(
                title=f"Scatter: {pair[0]} vs {pair[1]} (r={corr.loc[pair[0], pair[1]]:.2f})",
                xaxis_title=pair[0], yaxis_title=pair[1],
                template="plotly_dark",
                paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                font=dict(family="Inter, sans-serif"),
                margin=dict(l=40, r=20, t=50, b=40),
            )
            figures.append(fig.to_html(full_html=False, include_plotlyjs=False))
            chart_count += 1

        # ── Box plots ────────────────────────────────────────────────
        if len(num_cols) >= 2:
            fig = go.Figure()
            for col in num_cols[:8]:
                fig.add_trace(go.Box(y=df[col], name=col, marker_color=BRAND_COLOR))
            fig.update_layout(
                title="Box Plots — Numeric Columns",
                template="plotly_dark",
                paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                font=dict(family="Inter, sans-serif"),
                margin=dict(l=40, r=20, t=50, b=40),
            )
            figures.append(fig.to_html(full_html=False, include_plotlyjs=False))
            chart_count += 1

        # ── Assemble full HTML ───────────────────────────────────────
        charts_html = "\n".join(
            f'<section class="chart-card">{fig_html}</section>'
            for fig_html in figures
        )

        recommendations_html = "\n".join(
            f"<li>{r}</li>" for r in insight.business_recommendations
        )

        full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{BRAND_NAME} — Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --brand: {BRAND_COLOR};
            --bg: #0d1117;
            --surface: #161b22;
            --border: #30363d;
            --text: #e6edf3;
            --text-muted: #8b949e;
            --spacing: 8px;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Inter', system-ui, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
            padding: calc(var(--spacing) * 4);
        }}
        header {{
            text-align: center;
            padding: calc(var(--spacing) * 4) 0;
            border-bottom: 1px solid var(--border);
            margin-bottom: calc(var(--spacing) * 4);
        }}
        header h1 {{
            font-size: 2rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--brand), #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        header p {{ color: var(--text-muted); margin-top: var(--spacing); }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
            gap: calc(var(--spacing) * 2);
            margin-bottom: calc(var(--spacing) * 4);
        }}
        .kpi-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: calc(var(--spacing) * 1.5);
            padding: calc(var(--spacing) * 2.5);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .kpi-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(108, 99, 255, 0.15);
        }}
        .kpi-card .label {{
            font-size: 0.8rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .kpi-card .value {{
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--brand);
            margin-top: 4px;
        }}
        .chart-card {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: calc(var(--spacing) * 1.5);
            padding: calc(var(--spacing) * 2);
            margin-bottom: calc(var(--spacing) * 3);
        }}
        .recommendations {{
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: calc(var(--spacing) * 1.5);
            padding: calc(var(--spacing) * 3);
            margin-top: calc(var(--spacing) * 4);
        }}
        .recommendations h2 {{
            font-size: 1.3rem;
            margin-bottom: calc(var(--spacing) * 2);
        }}
        .recommendations li {{
            margin-bottom: var(--spacing);
            padding-left: var(--spacing);
        }}
        @media (prefers-reduced-motion: reduce) {{
            .kpi-card {{ transition: none; }}
            .health-card {{ transition: none; }}
        }}
        .health-overview {{
            background: linear-gradient(135deg, var(--surface) 0%, #1a1f2e 100%);
            border: 1px solid var(--border);
            border-radius: calc(var(--spacing) * 2);
            padding: calc(var(--spacing) * 3);
            margin-bottom: calc(var(--spacing) * 4);
        }}
        .health-overview h2 {{
            font-size: 1.3rem;
            margin-bottom: calc(var(--spacing) * 2);
        }}
        .health-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: calc(var(--spacing) * 2);
            margin-top: calc(var(--spacing) * 2);
        }}
        .health-card {{
            background: var(--bg);
            border: 1px solid var(--border);
            border-radius: calc(var(--spacing) * 1.5);
            padding: calc(var(--spacing) * 2);
            text-align: center;
            transition: transform 0.2s;
        }}
        .health-card:hover {{
            transform: translateY(-2px);
        }}
        .health-card .h-label {{
            font-size: 0.7rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .health-card .h-value {{
            font-size: 1.4rem;
            font-weight: 700;
            margin-top: 4px;
        }}
    </style>
</head>
<body>
    <header>
        <h1>{BRAND_NAME}</h1>
        <p>Interactive Data Analysis Dashboard</p>
    </header>

    {health_html}

    <section class="kpi-grid">
        {kpi_html}
    </section>

    {charts_html}

    <section class="recommendations">
        <h2>💡 Business Recommendations</h2>
        <ul>{recommendations_html}</ul>
    </section>
</body>
</html>"""

        html_path = output_dir / "dashboard.html"
        html_path.write_text(full_html, encoding="utf-8")
        self._log(f"Dashboard saved → {html_path} ({chart_count} charts)")

        return DashboardResult(
            html_path=str(html_path),
            chart_count=chart_count,
            kpi_count=min(len(insight.kpi_list), 12),
        )

    # ── Helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _build_kpi_cards(kpis) -> str:
        cards = []
        for kpi in kpis:
            cards.append(
                f'<article class="kpi-card">'
                f'<p class="label">{kpi.name}</p>'
                f'<p class="value">{kpi.value}</p>'
                f"</article>"
            )
        return "\n".join(cards)

    @staticmethod
    def _build_health_section(
        qb: DataQualityResult | None,
        qa: DataQualityResult,
    ) -> str:
        """Build the Data Health Overview HTML block."""
        qa_color = qa.score_color

        # Gauge using Plotly
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=qa.quality_score,
            title={"text": "Quality Score", "font": {"size": 16, "color": "#c9d1d9"}},
            number={"font": {"size": 40, "color": qa_color}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#484f58"},
                "bar": {"color": qa_color},
                "bgcolor": "#21262d",
                "bordercolor": "#30363d",
                "steps": [
                    {"range": [0, 60], "color": "rgba(239,68,68,0.15)"},
                    {"range": [60, 80], "color": "rgba(234,179,8,0.15)"},
                    {"range": [80, 100], "color": "rgba(34,197,94,0.15)"},
                ],
            },
        ))
        gauge_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=250,
            margin=dict(l=30, r=30, t=40, b=10),
        )
        gauge_html = gauge_fig.to_html(full_html=False, include_plotlyjs=False)

        # Before/after bar
        comparison_html = ""
        if qb:
            qb_color = score_color(qb.quality_score)
            cmp_fig = go.Figure()
            cmp_fig.add_trace(go.Bar(
                name="Before", x=["Score", "Missing %", "Duplicate %"],
                y=[qb.quality_score, qb.missing_percent, qb.duplicate_percent],
                marker_color=qb_color,
            ))
            cmp_fig.add_trace(go.Bar(
                name="After", x=["Score", "Missing %", "Duplicate %"],
                y=[qa.quality_score, qa.missing_percent, qa.duplicate_percent],
                marker_color=qa_color,
            ))
            cmp_fig.update_layout(
                barmode="group", template="plotly_dark",
                height=250, paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=30, r=30, t=30, b=30),
                legend=dict(orientation="h", y=1.1),
            )
            comparison_html = (
                '<h3 style="margin-top:16px;">⚡ Cleaning Impact</h3>'
                + cmp_fig.to_html(full_html=False, include_plotlyjs=False)
            )

        # Mini KPI cards
        m_color = score_color(100 - qa.missing_percent * 2)
        d_color = score_color(100 - qa.duplicate_percent * 5)
        s_color = score_color(100 if qa.schema_issues == 0 else (70 if qa.schema_issues < 5 else 40))

        cards = f"""
        <div class="health-grid">
            <article class="health-card" style="border-top:3px solid {m_color}">
                <p class="h-label">Missing %</p>
                <p class="h-value" style="color:{m_color}">{qa.missing_percent}%</p>
            </article>
            <article class="health-card" style="border-top:3px solid {d_color}">
                <p class="h-label">Duplicate %</p>
                <p class="h-value" style="color:{d_color}">{qa.duplicate_percent}%</p>
            </article>
            <article class="health-card" style="border-top:3px solid {s_color}">
                <p class="h-label">Schema Issues</p>
                <p class="h-value" style="color:{s_color}">{qa.schema_issues}</p>
            </article>
            <article class="health-card" style="border-top:3px solid {qa_color}">
                <p class="h-label">Risk Level</p>
                <p class="h-value" style="color:{qa_color}">{qa.risk_level}</p>
            </article>
        </div>
        """

        return (
            '<section class="health-overview">'
            '<h2>🩺 Data Health Overview</h2>'
            f'{gauge_html}{cards}{comparison_html}'
            '</section>'
        )
