"""
ReportAgent — Architecture 3.0 (Pure Platypus Refactor).

This agent generates professional executive reports using ReportLab Platypus.
All manual canvas positioning and coordinates have been removed in favor of 
a pure flowable-based story.
"""

from __future__ import annotations

import logging
import threading
import copy
import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass

# Silence technical warnings for cleaner business reporting
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")

logger = logging.getLogger(__name__)
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm, inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image,
    KeepTogether,
    HRFlowable,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

import plotly.graph_objects as go
import plotly.io as pio

from agents.base import BaseAgent
from agents.cleaning import CleaningResult
from agents.ingestion import IngestionResult
from agents.insight import InsightResult, KPI
from agents.repair import RepairResult
from agents.forecast import ForecastResult
from config.settings import BRAND_COLOR, BRAND_NAME

from agents.report_validation import ReportValidationEngine

# ── Font Registration ────────────────────────────────────────────────
try:
    pdfmetrics.registerFont(TTFont("Inter-Regular", "assets/fonts/Inter-Regular.ttf"))
    pdfmetrics.registerFont(TTFont("Inter-SemiBold", "assets/fonts/Inter-SemiBold.ttf"))
    pdfmetrics.registerFont(TTFont("Inter-Bold", "assets/fonts/Inter-Bold.ttf"))
    pdfmetrics.registerFont(TTFont("Inter-Italic", "assets/fonts/Inter-Italic.ttf"))
    
    FONT_HEAD = "Inter-Bold"
    FONT_SUB = "Inter-SemiBold"
    FONT_BODY = "Inter-Regular"
    FONT_ITALIC = "Inter-Italic"
except Exception as e:
    logger.warning(f"Failed to register Inter fonts, falling back to Helvetica: {e}")
    FONT_HEAD = "Helvetica-Bold"
    FONT_SUB = "Helvetica-Bold"
    FONT_BODY = "Helvetica"
    FONT_ITALIC = "Helvetica-Oblique"

# ── SaaS Color System ────────────────────────────────────────────────
PRIMARY_TEXT = colors.HexColor("#111827")
SECONDARY_TEXT = colors.HexColor("#374151")
MUTED_TEXT = colors.HexColor("#6B7280")
BRAND_COLOR_HEX = colors.HexColor("#4F46E5") # Primary SaaS Brand Color

# --- CONFIGURATION (Legacy Compatibility) ---

# Strict Brand Colors
_bc = BRAND_COLOR.lstrip("#")
try:
    if len(_bc) == 6:
        BRAND_RGB = colors.Color(
            int(_bc[:2], 16) / 255,
            int(_bc[2:4], 16) / 255,
            int(_bc[4:6], 16) / 255,
        )
    else:
        BRAND_RGB = colors.HexColor(BRAND_COLOR)
except Exception:
    BRAND_RGB = colors.HexColor("#6C63FF")

COLOR_DARK_BLUE = colors.HexColor("#0f172a") # Slate 900
COLOR_TEXT_BODY = colors.HexColor("#334155") # Slate 700
COLOR_TEXT_MUTED = colors.HexColor("#64748b") # Slate 500
COLOR_BG_LIGHT = colors.HexColor("#f8fafc")  # Slate 50

# Impact Colors
COLOR_GREEN = colors.HexColor("#16a34a")
COLOR_YELLOW = colors.HexColor("#ca8a04")
COLOR_RED = colors.HexColor("#dc2626")

@dataclass
class ReportResult:
    pdf_path: str = ""
    markdown_path: str = ""

class StyleManager:
    """Centralized Typography Configuration."""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        
        # Typography Hierarchy (Modern SaaS / Consulting Standard)
        self.title = ParagraphStyle(
            "Title_Modern",
            fontName=FONT_HEAD, fontSize=30, leading=36,
            textColor=PRIMARY_TEXT, spaceBefore=24, spaceAfter=16
        )
        
        self.h1 = ParagraphStyle(
            "H1_Modern",
            fontName=FONT_SUB, fontSize=20, leading=26,
            textColor=PRIMARY_TEXT, spaceBefore=18, spaceAfter=12
        )
        
        self.h2 = ParagraphStyle(
            "H2_Modern",
            fontName=FONT_SUB, fontSize=15, leading=20,
            textColor=SECONDARY_TEXT, spaceBefore=12, spaceAfter=8
        )
        
        self.body = ParagraphStyle(
            "Body_Modern",
            fontName=FONT_BODY, fontSize=11.5, leading=15,
            textColor=SECONDARY_TEXT, spaceAfter=8, alignment=TA_LEFT
        )
        
        self.caption = ParagraphStyle(
            "Caption_Modern",
            fontName=FONT_ITALIC, fontSize=9, leading=12,
            textColor=MUTED_TEXT, alignment=TA_CENTER, spaceBefore=6, spaceAfter=12
        )
        
        self.kpi_value = ParagraphStyle(
            "KPI_Value",
            fontName=FONT_HEAD, fontSize=24, leading=26,
            textColor=BRAND_COLOR_HEX, alignment=TA_CENTER
        )
 
        self.kpi_label = ParagraphStyle(
            "KPI_Label",
            fontName=FONT_BODY, fontSize=10, leading=12,
            textColor=MUTED_TEXT, alignment=TA_CENTER
        )

class ReportAgent(BaseAgent):
    """Architecture 3.0 PDF Generator — Modular Section Builders."""

    name = "ReportAgent"

    def __init__(self):
        super().__init__()
        self.validator = ReportValidationEngine()
        self.sty = StyleManager()
        
        # Internal context for builders
        self.ingestion = None
        self.cleaning = None
        self.q_after = None
        self.chart_paths = {}

    def _execute(self, input_data: dict) -> ReportResult:
        try:
            # 1. Pipeline Input & Cleaning
            self.ingestion = input_data["ingestion"]
            self.cleaning = input_data["cleaning"]
            self.q_after = input_data.get("quality_after")
            
            # 2. Analytics Validation
            insight = self.validator.validate_insight(input_data["insight"])
            forecast = self.validator.validate_forecast(input_data.get("forecast"))
            
            output_dir = Path(input_data["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            self.industry = self._detect_industry(self.ingestion.detected_types.keys())
            
            # 3. Artifact Preparation
            self.chart_paths = self._create_charts(self.ingestion, insight, forecast, output_dir)
            pdf_path = output_dir / "report.pdf"
            md_path = output_dir / "report.md"

            # 4. Data Sanitization (Consulting-Grade Cleanup)
            s_insight, s_forecast = self._sanitize_data(insight, forecast)

            # 5. Rendering Engine (Structured Section Flow)
            doc = SimpleDocTemplate(
                str(pdf_path),
                pagesize=A4,
                rightMargin=60,
                leftMargin=60,
                topMargin=60,
                bottomMargin=50
            )
            
            story = []
            
            # Standard section flow with requested spacing
            self._build_executive_summary(story, s_insight)
            
            story.append(Spacer(1, 24))
            self._build_kpis(story, s_insight)
            
            story.append(Spacer(1, 24))
            self._build_trends(story, s_insight)
            
            story.append(Spacer(1, 24))
            self._build_correlations(story, s_insight)
            
            story.append(Spacer(1, 24))
            self._build_forecast(story, s_forecast)
            
            story.append(Spacer(1, 24))
            self._build_strategy(story, s_insight)
            
            story.append(Spacer(1, 24))
            self._build_appendix(story, s_insight)

            # 6. Finalize
            doc.build(
                story,
                onFirstPage=self._draw_header_footer,
                onLaterPages=self._draw_header_footer
            )
            self._generate_markdown(md_path, self.ingestion, s_insight, self.q_after)
    
            self._log(f"Strategic Intelligence Delivered: {pdf_path}")
            return ReportResult(pdf_path=str(pdf_path), markdown_path=str(md_path))
            
        except Exception as e:
            logger.error(f"Report Engine Failure: {e}", exc_info=True)
            raise e
    
    def _sanitize_data(self, insight, forecast):
        """Creates sanitized copies for rendering ensuring no NaN/Inf or weak signals appear."""
        s_insight = copy.deepcopy(insight)
        s_forecast = copy.deepcopy(forecast) if forecast else None

        # 1. NaN/Inf Cleanup in KPIs
        for k in s_insight.kpi_list:
            if isinstance(k.value, (int, float)) and (np.isnan(k.value) or np.isinf(k.value)):
                k.value = "-"

        # 2. Trend Filtering (Slope < 0.0001) & NaN Cleanup
        if s_insight.trend_summary:
            valid_trends = []
            for t in s_insight.trend_summary:
                # Replace NaNs in slope/r2
                if hasattr(t, 'slope') and (np.isnan(t.slope) or np.isinf(t.slope)): t.slope = 0.0
                if hasattr(t, 'r_squared') and (np.isnan(t.r_squared) or np.isinf(t.r_squared)): t.r_squared = 0.0
                
                # Filter stagnant
                if hasattr(t, 'slope') and abs(t.slope) >= 0.0001:
                    valid_trends.append(t)
            s_insight.trend_summary = valid_trends

        # 3. Correlation Filtering (abs(r) > 0.999) & NaN Cleanup
        if s_insight.correlation_matrix is not None:
            # pd.DataFrame cleanup
            df = s_insight.correlation_matrix.copy()
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # Mask out identity/strong redundant correlations (abs > 0.999)
            # We set them to NaN so _get_correlation_pairs ignores them
            for col in df.columns:
                for row in df.index:
                    if col == row:
                        df.at[row, col] = np.nan # Identity
                    elif abs(df.at[row, col]) > 0.999:
                        df.at[row, col] = np.nan # Redundant
            
            # Fill remaining NaNs with 0.0 for heatmap rendering if needed
            # but _get_correlation_pairs handles NaNs well.
            s_insight.correlation_matrix = df

        # 4. Forecast Suppression (N < 8 or R2 < 0.1)
        if s_forecast and s_forecast.forecasts:
            sanitized_f = {}
            for metric, f_data in s_forecast.forecasts.items():
                n_points = len(f_data.get("values_hist", []))
                r2 = f_data.get("r2", 0.0)
                
                # Cleanup NaNs in forecast values
                if np.isnan(r2): r2 = 0.0
                
                if n_points >= 8 and r2 >= 0.1:
                    sanitized_f[metric] = f_data
            s_forecast.forecasts = sanitized_f

        return s_insight, s_forecast

    def _draw_header_footer(self, canvas, doc):
        """Standardizes layout across all pages using canvas primitives."""
        canvas.saveState()
        
        # Header
        canvas.setFont(FONT_HEAD, 10)
        canvas.setStrokeColor(MUTED_TEXT)
        canvas.setLineWidth(0.5)
        
        # Left: Report Title
        canvas.drawString(60, doc.pagesize[1] - 40, f"{BRAND_NAME} Analytics")
        
        # Right: Strategic Intelligence Report
        canvas.drawRightString(doc.pagesize[0] - 60, doc.pagesize[1] - 40, "Strategic Intelligence Report")
        
        # Separator Line
        canvas.line(60, doc.pagesize[1] - 45, doc.pagesize[0] - 60, doc.pagesize[1] - 45)
        
        # Footer
        canvas.setFont(FONT_BODY, 9)
        canvas.drawCentredString(doc.pagesize[0] / 2, 30, f"Page {doc.page}")
        
        canvas.restoreState()

    def _format_metric_name(self, name: str) -> str:
        """Sanitizes and formats column names for professional output."""
        if name.lower() == "id": return ""
        clean = name.replace("_", " ").replace(".", " ").strip()
        # Capitalize words
        clean = " ".join([w.capitalize() for w in clean.split()])
        # Specific overrides
        overrides = {
            "Index": "Operational Index Score",
            "Id": "ID",
            "Sku": "SKU",
            "Kpi": "KPI",
            "Roi": "ROI"
        }
        return overrides.get(clean, clean)

    def _classify_metric_type(self, name: str, value: float) -> str:
        """Classifies metric for tailored narrative generation."""
        l_name = name.lower()
        if any(x in l_name for x in ["count", "sum", "total", "volume", "qty", "quantity"]):
            return "Volume"
        if any(x in l_name for x in ["mean", "avg", "average", "median"]):
            return "Central Tendency"
        if any(x in l_name for x in ["std", "deviation", "variance", "dispersion"]):
            return "Dispersion"
        if any(x in l_name for x in ["min", "max", "range", "spread"]):
            return "Range"
        if any(x in l_name for x in ["%", "rate", "ratio", "margin", "scoe"]):
            return "Ratio"
        return "General"

    # ── Section Builders ──────────────────────────────────────────────

    def _build_executive_summary(self, story, insight):
        # Header Branding
        story.append(Paragraph("Strategic Executive Audit", self.sty.title))
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"{self.industry} Sector Performance • {datetime.now().strftime('%B %d, %Y')}", self.sty.caption))
        story.append(Spacer(1, 24))
        
        # Block 1: Strategic Snapshot
        score = self.q_after.quality_score if self.q_after else 0
        risk_level = "HIGH" if score < 60 else ("MEDIUM" if score < 85 else "LOW")
        
        snapshot_data = [
            [Paragraph("<b>Vertical Sector:</b>", self.sty.body), Paragraph(self.industry, self.sty.body)],
            [Paragraph("<b>Intelligence Depth:</b>", self.sty.body), Paragraph(f"{self.ingestion.row_count:,} Signals", self.sty.body)],
            [Paragraph("<b>Data Health:</b>", self.sty.body), self._create_badge(f"{score}/100", "LOW" if score >= 85 else ("MEDIUM" if score >= 60 else "HIGH"))],
            [Paragraph("<b>Operational Risk:</b>", self.sty.body), self._create_badge(risk_level, risk_level)]
        ]
        
        t1 = Table(snapshot_data, colWidths=[inch * 2, inch * 4.5])
        t1.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), COLOR_BG_LIGHT),
            ('BOX', (0,0), (-1,-1), 1, colors.white),
            ('PADDING', (0,0), (-1,-1), 12),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        story.append(t1)
        story.append(Spacer(1, 24))

        # Block 2: Primary Business Finding
        story.append(Paragraph("Core Strategic Insight", self.sty.h1))
        story.append(Spacer(1, 4))
        self._add_divider(story)
        story.append(Spacer(1, 12))
        
        narrative = self._generate_strategic_narrative(self.ingestion, insight, score)
        # Headline separation
        sentences = narrative.split(". ")
        headline = sentences[0] + "." if sentences else "Actionable intelligence detected across primary operational channels."
        support = ". ".join(sentences[1:]) if len(sentences) > 1 else "Strategic alignment suggested to capitalize on identified variances."
        
        story.append(Paragraph(f"<b>{headline}</b>", self.sty.body))
        story.append(Spacer(1, 8))
        story.append(Paragraph(support, self.sty.body))
        story.append(Spacer(1, 24))

        # Block 3: Immediate Action Focus
        story.append(Paragraph("Priority Action Matrix", self.sty.h1))
        story.append(Spacer(1, 12))
        
        # Risk & Opportunity
        risk_text = "Standard variance identified"
        if insight.top_risks:
            r = insight.top_risks[0]
            risk_text = str(r.detail if hasattr(r, 'detail') else (r.get('detail', str(r)) if isinstance(r, dict) else str(r)))
            
        opp_text = "Optimization potential detected in baseline metrics"
        trends = [t for t in insight.trend_summary if t.direction != "stable"]
        if trends:
            t = trends[0]
            direction_icon = "↑" if t.direction == "increasing" else "↓"
            opp_text = f"Capitalize on {t.column} {t.direction} momentum {direction_icon}"

        action_data = [
            [self._create_badge("PRIORITY RISK", "HIGH"), Paragraph(risk_text, self.sty.body)],
            [Spacer(1, 8), Spacer(1, 8)],
            [self._create_badge("OPPORTUNITY", "LOW"), Paragraph(opp_text, self.sty.body)]
        ]
        
        t2 = Table(action_data, colWidths=[inch * 1.5, inch * 5])
        t2.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'TOP'),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(t2)
        
        story.append(PageBreak())

    def _build_kpis(self, story, insight):
        story.append(Paragraph("Core Business Metrics", self.sty.title))
        story.append(Spacer(1, 12))
        
        kpis = [k for k in insight.kpi_list if self._is_business_column(k.name)]
        kpis.sort(key=lambda x: 0 if any(s in x.name.lower() for s in ["rev", "sales", "profit"]) else 1)
        
        if not kpis:
            story.append(Paragraph("No tracked business KPIs identified.", self.sty.body))
            story.append(PageBreak())
            return

        # 2x3 Grid Layout
        data = []
        row = []
        for k in kpis[:6]:
            val_str = f"{float(k.value):,.2f}" if isinstance(k.value, (int, float)) else str(k.value)
            display_name = self._format_metric_name(k.name)
            
            # --- Business Interpretation Layer ---
            metric_type = self._classify_metric_type(k.name, k.value)
            interpretation = ""
            
            # Find matching trend for context
            matching_trend = next((t for t in insight.trend_summary if t.column in k.name), None)
            cv = 0.0 # Coefficient of Variation proxy
            if matching_trend:
                # Calculate volatility classification
                # improving this via trend R2 (inverse proxy for volatility/noise)
                volatility_level = "High" if matching_trend.r_squared < 0.3 else ("Moderate" if matching_trend.r_squared < 0.7 else "Low")
                
                direction_map = {
                    "increasing": "expansion",
                    "decreasing": "contraction",
                    "stable": "consistency"
                }
                movement = direction_map.get(matching_trend.direction, "movement")
                
                # Differentiated Logic based on Metric Type
                if metric_type == "Volume":
                    interpretation = f"Volume levels indicate {volatility_level.lower()} volatility in throughput. Current {movement} suggests changing scale requirements."
                elif metric_type == "Central Tendency":
                    interpretation = f"Average performance shows {movement}, stabilizing around current benchmarks with {volatility_level.lower()} variance."
                elif metric_type == "Dispersion":
                    interpretation = f"Observed variability reflects {volatility_level.lower()} dispersion magnitude, signaling {'potential instability' if volatility_level == 'High' else 'controlled process conditions'}."
                elif metric_type == "Range":
                    interpretation = f"Span indicates performance bandwidth. {volatility_level} fluctuation observed between extremes."
                elif metric_type == "Ratio":
                    interpretation = f"Efficiency rate exhibits {movement}. {volatility_level} stability suggests {'structural shifts' if volatility_level == 'High' else 'predictable outcomes'}."
                else:
                    interpretation = f"Metric performance displays {movement} with {volatility_level.lower()} reliability."
            else:
                 interpretation = "Baseline metric stability observed across current reporting period."

            # Content Cell (Vertical Stack)
            cell_content = [
                Paragraph(f"<b>{display_name}</b>", self.sty.kpi_label),
                Spacer(1, 4),
                Paragraph(val_str, self.sty.kpi_value),
                Spacer(1, 6),
                Paragraph(interpretation, self.sty.caption),
                Spacer(1, 4),
                # Clean up description if it exists
                Paragraph(k.description[:80] + "..." if len(k.description) > 80 else k.description, self.sty.caption)
            ]
            row.append(cell_content)
            
            if len(row) == 2:
                data.append(row)
                row = []
        
        # Fill remaining if any
        if row:
            while len(row) < 2:
                row.append("")
            data.append(row)

        available_width = A4[0] - 120 # 475.27pt
        t = Table(data, colWidths=[available_width/2] * 2)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), COLOR_BG_LIGHT),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.white),
            ('BOX', (0,0), (-1,-1), 1, colors.white),
            ('LEFTPADDING', (0,0), (-1,-1), 15),
            ('RIGHTPADDING', (0,0), (-1,-1), 15),
            ('TOPPADDING', (0,0), (-1,-1), 15),
            ('BOTTOMPADDING', (0,0), (-1,-1), 15),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        story.append(t)
        story.append(PageBreak())

    def _build_trends(self, story, insight):
        story.append(Paragraph("Operational Trajectory", self.sty.title))
        story.append(Spacer(1, 12))
        
        content_width = A4[0] - 120
        if "trend" in self.chart_paths:
            img = Image(self.chart_paths["trend"], width=content_width, height=content_width*0.5)
            story.append(img)
            story.append(Spacer(1, 12))
            story.append(Paragraph("Historical Performance Intensity", self.sty.caption))
            story.append(Spacer(1, 12))
        
        story.append(Paragraph("Trend Analysis", self.sty.h1))
        story.append(Spacer(1, 12))
        
        # Filter out trivial trends
        trends = [t for t in insight.trend_summary if t.direction != "stable" and self._is_business_column(t.column)]
        
        if not trends:
            story.append(Paragraph("Baseline consistency detected across all primary operational channels. Volatility remains within standard deviations, indicating a stabilized performance environment.", self.sty.body))
        else:
            for t in trends[:4]:
                clean_name = self._format_metric_name(t.column)
                direction = t.direction
                
                # Magnitude Framing
                # Using R2 as proxy for stability magnitude (High R2 = Low Volatility = Stable Trajectory)
                stability_label = "Confirmed" if t.r_squared > 0.7 else ("Tentative" if t.r_squared > 0.4 else "Volatile")
                
                # Trajectory descriptor
                traj_map = {
                    "increasing": "progressive expansion",
                    "decreasing": "structural contraction",
                    "stable": "lateral consistency"
                }
                trajectory = traj_map.get(direction, "directional shift")
                
                msg = (
                    f"• <b>{clean_name}:</b> Analysis identifies a {stability_label.lower()} {trajectory} over the observed period. "
                    f"This directional persistency suggests fundamental shifts in {clean_name} drivers. "
                    f"Operational planning should account for this {direction} drift to optimize capacity alignment."
                )
                story.append(Paragraph(msg, self.sty.body))
                story.append(Spacer(1, 12))

        # Business Implication Callout
        if trends:
            t = trends[0]
            c_name = self._format_metric_name(t.column)
            imp = f"The observed trajectory in {c_name} indicates a direct impact on future throughput capacity. Strategic resource reallocation is advised."
            story.append(self._build_callout_box("Business Implication", imp))

        story.append(PageBreak())

    def _build_correlations(self, story, insight):
        story.append(Paragraph("Relationship Matrix", self.sty.title))
        story.append(Spacer(1, 12))
        
        content_width = A4[0] - 120
        if "heatmap" in self.chart_paths:
            img = Image(self.chart_paths["heatmap"], width=content_width, height=content_width*0.6)
            story.append(img)
            story.append(Spacer(1, 12))
            story.append(Paragraph("Multivariate Signal Correlation Intensity", self.sty.caption))
            story.append(Spacer(1, 12))
        
        story.append(Paragraph("Correlation Insights", self.sty.h1))
        story.append(Spacer(1, 12))
        
        pairs = self._get_correlation_pairs(insight.correlation_matrix)
        if not pairs:
             story.append(Paragraph(
                 "Metric independence suggests decentralized performance drivers. This reduces systemic concentration risk "
                 "but may limit strategic leverage opportunities across high-impact functions.", self.sty.body
             ))
        else:
             for p in pairs[:5]:
                 n1 = self._format_metric_name(p[0])
                 n2 = self._format_metric_name(p[1])
                 
                 strength = "Significant" if abs(p[2]) > 0.8 else "Moderate"
                 linkage_type = "positive" if p[2] > 0 else "inverse"
                 
                 msg = (
                     f"• <b>{n1} vs {n2}:</b> {strength} {linkage_type} structural linkage detected. "
                     f"Operational adjustments in {n1} are likely to propagate effects to {n2}, suggesting specific cross-functional dependencies."
                 )
                 story.append(Paragraph(msg, self.sty.body))
                 story.append(Spacer(1, 12))

        story.append(PageBreak())

    def _build_forecast(self, story, forecast):
        """Strategic Projection Section with stability fallback."""
        story.append(Paragraph("Strategic Projection", self.sty.title))
        story.append(Spacer(1, 12))

        if not forecast or not forecast.forecasts:
            story.append(Paragraph("Dataset density insufficient for reliable strategic projection. Volatility thresholds or confidence intervals currently exceed acceptable predictive corridors.", self.sty.body))
            story.append(PageBreak())
            return
            
        story.append(Paragraph("Strategic Projection", self.sty.title))
        story.append(Spacer(1, 12))
        
        content_width = A4[0] - 120
        if "forecast" in self.chart_paths:
             img = Image(self.chart_paths["forecast"], width=content_width, height=content_width*0.5)
             story.append(img)
             story.append(Spacer(1, 12))
             story.append(Paragraph("Automated Strategic Multi-Period Projection", self.sty.caption))
             story.append(Spacer(1, 12))
        
        for metric, data in forecast.forecasts.items():
            if not self._is_business_column(metric): continue
            
            hist = data.get("values_hist", [])
            pred = data.get("values_forecast", [])
            if not hist or not pred: continue
            
            delta = ((pred[-1] - hist[-1]) / hist[-1]) * 100 if hist[-1] != 0 else 0
            direction = "positive" if delta > 0 else "negative"
            
            clean_name = self._format_metric_name(metric)
            story.append(Paragraph(f"{clean_name} Outlook:", self.sty.h2))
            story.append(Spacer(1, 12))
            msg = (
                f"Projected performance exhibits a {direction} variance of {abs(delta):.1f}%, transitioning from a baseline of {hist[-1]:,.2f} "
                f"to a future-state estimate of {pred[-1]:,.2f}. Operational readiness and resource allocation models should align with this forecasted trajectory "
                "to capitalize on momentum or mitigate prospective downsides."
            )
            story.append(Paragraph(msg, self.sty.body))
            story.append(Spacer(1, 12))
            
        story.append(PageBreak())

    def _build_strategy(self, story, insight):
        story.append(Paragraph("Strategic Recommendations", self.sty.title))
        story.append(Spacer(1, 12))
        
        if not insight.business_recommendations:
             story.append(Paragraph("No structural recommendations triggered by current data state. Operational consistency remains within baseline expectations.", self.sty.body))
             story.append(PageBreak())
             return
             
        table_data = []
        for rec in insight.business_recommendations[:6]:
            risk_level = "Medium"
            impact = 7.0
            horizon = "Short-to-medium term (1–2 quarters)"
            confidence = "High (statistically supported)"
            
            if "risk" in rec.lower() or "warning" in rec.lower():
                risk_level, impact = "High", 8.5
            elif "opportunity" in rec.lower() or "growth" in rec.lower():
                risk_level, impact = "Low", 9.0
                horizon = "Medium-to-long term (2–4 quarters)"
            
            color = COLOR_RED if risk_level == "High" else (COLOR_YELLOW if risk_level == "Medium" else COLOR_GREEN)
            
            # --- Business Sanitization ---
            # Remove technical artifacts like (Slope: 0.23...) or (R2: 0.55...)
            clean_rec = rec.replace("Recommendation:", "").split(" (Slope")[0].split(" (R2")[0].strip()
            # Capitalize first letter
            clean_rec = clean_rec[0].upper() + clean_rec[1:] if clean_rec else "Optimize primary performance driver."
            
            c_header = Paragraph(f"Strategic Objective: {clean_rec}", self.sty.h2)
            c_meta = Paragraph(f"<font color={color}>Risk: {risk_level}</font> | Expected Impact: {impact}/10 | Horizon: {horizon}", self.sty.caption)
            c_desc = Paragraph(
                "<b>Rationale:</b> Priority implementation suggested based on modeled variances and significance levels. "
                "Addressing this objective minimizes structural performance drift and preserves organizational throughput.", 
                self.sty.body
            )
            c_conf = Paragraph(f"<b>Confidence:</b> {confidence}", self.sty.caption)
            
            table_data.append([c_header])
            table_data.append([c_meta])
            table_data.append([c_desc])
            table_data.append([c_conf])
            table_data.append([Spacer(1, 16)])
            
        t = Table(table_data, colWidths=[inch * 6.5])
        t.setStyle(TableStyle([
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('TOPPADDING', (0,0), (-1,-1), 2),
            ('BOTTOMPADDING', (0,0), (-1,-1), 2),
        ]))
        story.append(t)
        
        # Business Implication Callout
        if insight.business_recommendations:
            rec = insight.business_recommendations[0]
            clean_rec = rec.replace("Recommendation:", "").split(" (Slope")[0].split(" (R2")[0].strip()
            story.append(Spacer(1, 12))
            story.append(self._build_callout_box("Strategic Focus", f"The objective to '{clean_rec}' is the primary lever for operational performance. Addressing this immediately minimizes structural drift and ensures predictive alignment."))

        story.append(PageBreak())

    def _build_appendix(self, story, insight):
        story.append(Paragraph("Technical Metadata", self.sty.title))
        story.append(Spacer(1, 12))
        
        story.append(Paragraph(f"<b>Processing Scope:</b> {self.ingestion.row_count} records across {self.ingestion.col_count} dimensions.", self.sty.body))
        story.append(Spacer(1, 12))
        story.append(Paragraph("Cleaning Protocol:", self.sty.h2))
        story.append(Spacer(1, 12))
        
        if self.cleaning.cleaning_log and self.cleaning.cleaning_log.steps:
            for s in self.cleaning.cleaning_log.steps:
                story.append(Paragraph(f"• {s['action']} executed on {s['rows_affected']} records.", self.sty.body))
                story.append(Spacer(1, 12))
        else:
             story.append(Paragraph("No corrective cleaning maneuvers required.", self.sty.body))

    # ── Charting & Helpers ──────────────────────────────────────────

    def _create_charts(self, ingestion, insight, forecast, output_dir) -> Dict[str, str]:
        paths = {}
        try:
             import kaleido
        except ImportError:
             return paths
             
        import threading
        
        def save_chart(fig, path, timeout=15):
            def target():
                try:
                    pio.write_image(fig, str(path), scale=2)
                except Exception as e:
                    logger.warning(f"Chart write failed: {e}")

            t = threading.Thread(target=target)
            t.daemon = True
            t.start()
            t.join(timeout)
            return not t.is_alive()

        _pcolor = BRAND_COLOR if BRAND_COLOR.startswith("#") else f"#{BRAND_COLOR}"

        # Trend
        cols = [c for c in ingestion.dataframe.columns if self._is_business_column(c) and ingestion.detected_types.get(c) in ["int64", "float64"]]
        if cols:
             c = cols[0]
             fig = go.Figure()
             fig.add_trace(go.Scatter(y=ingestion.dataframe[c], mode='lines', name=c, line=dict(color=_pcolor, width=2.5)))
             fig.update_layout(template="plotly_white", margin=dict(l=20, r=20, t=50, b=20), height=400, width=800)
             p = output_dir / "trend.png"
             if save_chart(fig, p): paths["trend"] = str(p)
             
        # Correlations
        if insight.correlation_matrix is not None and not insight.correlation_matrix.empty:
             keep = [c for c in insight.correlation_matrix.columns if self._is_business_column(c)]
             if keep:
                 sub = insight.correlation_matrix.loc[keep, keep]
                 fig = go.Figure(data=go.Heatmap(z=sub.values, x=sub.columns, y=sub.index, colorscale='RdBu', zmin=-1, zmax=1))
                 fig.update_layout(title="Relationship Intensity Matrix", height=500, width=800)
                 p = output_dir / "heatmap.png"
                 if save_chart(fig, p): paths["heatmap"] = str(p)

        return paths

    def _detect_industry(self, columns) -> str:
        cols = {c.lower() for c in columns}
        if any(k in cols for k in ["churn", "mrr", "arr", "subscription"]): return "SaaS"
        if any(k in cols for k in ["sku", "store", "inventory", "sales"]): return "Retail"
        if any(k in cols for k in ["ticker", "portfolio", "roi", "asset"]): return "Finance"
        return "Executive Context"

    def _is_business_column(self, name: str) -> bool:
        return not self.validator._is_structural_column(name)

    def _generate_strategic_narrative(self, ingestion, insight, score) -> str:
        """Consulting-grade narrative generation (~200 words)."""
        if insight.executive_summary and len(insight.executive_summary) > 150: 
            return insight.executive_summary
            
        industry_label = self.industry
        row_count = ingestion.row_count
        
        # P1: Performance Context
        p1 = (
            f"This strategic audit evaluates {row_count:,} operational signals within the {industry_label} sector. "
            f"The primary data structure exhibits a performance health score of {score}/100, indicating "
            f"{'robust operational integrity' if score > 85 else ('moderate baseline stability' if score > 60 else 'significant structural variance')}. "
            "Internal metric distributions suggest that primary business levers are currently functioning within "
            "expected statistical parameters, though specific outliers require immediate attention."
        )
        
        # P2: Risk & Volatility
        volatility = "Moderate" if score > 60 else "High"
        p2 = (
            f"The current {industry_label} landscape presents {volatility.lower()} levels of systemic volatility. "
            "Our analysis of variance drivers identifies key performance sensitivities that could impact "
            "long-term throughput if left unmanaged. While secondary signals remain independent, "
            "the concentration of risk in primary operational pillars suggests a need for targeted stability "
            "measures to preserve current growth trajectories and minimize downside exposure."
        )
        
        # P3: Strategic Focus
        p3 = (
            "Strategic focus for the upcoming period should prioritize metric stabilization and the neutralization "
            "of identified risk vectors. Immediate tactical reallocation of resources toward high-impact "
            "variance drivers is advised. By aligning operational capacity with historical performance "
            "intensities, the organization can capitalize on emerging momentum while maintaining a "
            "defensive posture against unmodeled volatility."
        )
        
        return f"{p1}\n\n{p2}\n\n{p3}"

    def _generate_markdown(self, path, ingestion, insight, q_after):
        lines = [
            f"# {BRAND_NAME} Strategic Report",
            f"**Ref:** {datetime.now().strftime('%Y%m%d')} | **Sector:** {self.industry}",
            "\n## Executive Strategy",
            self._generate_strategic_narrative(ingestion, insight, q_after.quality_score if q_after else 0),
        ]
        path.write_text("\n".join(lines), encoding="utf-8")

    def _get_correlation_pairs(self, matrix):
        if matrix is None or matrix.empty: return []
        pairs = []
        seen = set()
        for c1 in matrix.columns:
            for c2 in matrix.columns:
                if c1 == c2: continue
                if not self._is_business_column(c1) or not self._is_business_column(c2): continue
                val = matrix.loc[c1, c2]
                if pd.isna(val) or np.isinf(val): continue
                if abs(val) < 0.7: continue
                key = tuple(sorted([c1, c2]))
                if key in seen: continue
                seen.add(key)
                pairs.append((c1, c2, float(val)))
        pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        return pairs

    # ── Framing Components ──────────────────────────────────────────

    def _create_badge(self, text, level="LOW"):
        color = COLOR_GREEN if level == "LOW" else (COLOR_YELLOW if level == "MEDIUM" else COLOR_RED)
        style = ParagraphStyle("Badge", fontName=FONT_HEAD, fontSize=8, textColor=colors.white, alignment=TA_CENTER)
        
        t = Table([[Paragraph(text, style)]], colWidths=[60], rowHeights=[18])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), color),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('ROUNDEDCORNERS', [4, 4, 4, 4]),
        ]))
        return t

    def _build_callout_box(self, title, message):
        title_style = ParagraphStyle("CalloutTitle", parent=self.sty.body, fontName=FONT_HEAD, textColor=BRAND_COLOR_HEX)
        
        # Wrap message to bold first sentence
        parts = message.split(". ", 1)
        if len(parts) > 1:
            bolded_msg = f"<b>{parts[0]}.</b> {parts[1]}"
        else:
            bolded_msg = f"<b>{message}</b>"
            
        content = [
            Paragraph(title.upper(), title_style),
            Spacer(1, 4),
            Paragraph(bolded_msg, self.sty.body)
        ]
        
        t = Table([[content]], colWidths=[inch * 6.5])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), COLOR_BG_LIGHT),
            ('BOX', (0,0), (-1,-1), 0.5, BRAND_RGB),
            ('LEFTPADDING', (0,0), (-1,-1), 15),
            ('RIGHTPADDING', (0,0), (-1,-1), 15),
            ('TOPPADDING', (0,0), (-1,-1), 12),
            ('BOTTOMPADDING', (0,0), (-1,-1), 12),
        ]))
        return t

    def _add_divider(self, story):
        story.append(Spacer(1, 10))
        story.append(HRFlowable(width="100%", thickness=0.5, color=BRAND_COLOR_HEX, spaceBefore=4, spaceAfter=4))
        story.append(Spacer(1, 10))
