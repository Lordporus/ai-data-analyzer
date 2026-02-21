"""
InsightAgent — Generate KPIs, detect trends, compute correlations,
flag anomalies, and produce business recommendations.

Input:  RepairResult
Output: InsightResult
"""

from __future__ import annotations

import logging
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from agents.base import BaseAgent
from agents.repair import RepairResult
from utils.intelligence_engine import IntelligenceEngine

logger = logging.getLogger(__name__)


@dataclass
class KPI:
    name: str
    value: Any
    unit: str = ""
    description: str = ""


@dataclass
class TrendInfo:
    column: str
    slope: float
    direction: str  # "increasing" | "decreasing" | "stable"
    p_value: float = 0.0
    r_squared: float = 0.0
    context: str = "" # AI-generated context


@dataclass
class AnomalyFlag:
    column: str
    row_indices: List[int] = field(default_factory=list)
    count: int = 0
    method: str = "z-score"


@dataclass
class InsightResult:
    kpi_list: List[KPI] = field(default_factory=list)
    trend_summary: List[TrendInfo] = field(default_factory=list)
    correlation_matrix: Optional[pd.DataFrame] = None
    anomaly_flags: List[AnomalyFlag] = field(default_factory=list)
    
    # Legacy field - kept for backward compatibility but populated with results if available
    business_recommendations: List[str] = field(default_factory=list)
    
    # Strategic Narrative Fields (Managed by IntelligenceEngine)
    executive_summary: str = ""
    primary_risk: str = ""
    primary_opportunity: str = ""
    confidence_comment: str = ""
    
    # Additional metadata (Optional/Legacy support)
    top_risks: List[Dict[str, Any]] = field(default_factory=list)
    top_opportunities: List[str] = field(default_factory=list)
    key_relationships: List[str] = field(default_factory=list)     # Filtered, non-obvious correlations
    
    detected_types: Dict[str, str] = field(default_factory=dict)
    dataframe: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)


class InsightAgent(BaseAgent):
    """Analyze the cleaned dataset to produce actionable insights."""

    name = "InsightAgent"

    def __init__(self):
        super().__init__()
        self.intelligence_engine = IntelligenceEngine()

    def _execute(self, input_data: RepairResult) -> InsightResult:
        df = input_data.dataframe.copy()
        types = input_data.detected_types
        num_cols = [c for c, t in types.items() if t == "numeric" and c in df.columns]
        cat_cols = [c for c, t in types.items() if t == "categorical" and c in df.columns]
        date_cols = [c for c, t in types.items() if t == "datetime" and c in df.columns]

        kpis = self._compute_kpis(df, num_cols, cat_cols)
        trends = self._detect_trends(df, num_cols, date_cols)
        corr_matrix = self._compute_correlations(df, num_cols)
        anomalies = self._flag_anomalies(df, num_cols)
        
        # New Logic: Filter Connections
        filtered_corrs = self._filter_correlations(df, num_cols, corr_matrix)
        
        # ── Intelligence Engine Context Synthesis ─────────────────────
        primary_trend = "Stable"
        primary_conf = "Medium"
        max_vol = 0.0
        
        if trends:
            sig_trends = [t for t in trends if t.direction != "stable"]
            if sig_trends:
                top_t = sorted(sig_trends, key=lambda x: abs(x.slope), reverse=True)[0]
                primary_trend = "Upward" if top_t.direction == "increasing" else "Downward"
                primary_conf = "High" if top_t.r_squared > 0.7 else "Medium"
        
        for col in num_cols:
            mean = df[col].mean()
            if abs(mean) > 1e-9:
                cv = abs(df[col].std() / mean)
                max_vol = max(max_vol, cv)

        # Build context for IntelligenceEngine
        context = {
            "trend_direction": primary_trend,
            "confidence_level": primary_conf,
            "volatility_index": round(max_vol, 3),
            "seasonality_detected": False,
            "forecast_model_type": "Linear"
        }

        # Generate Strategic Narrative via Abstraction Layer
        narrative = self.intelligence_engine.generate_strategic_summary(context)
        
        # Fallback recommendations if AI fails or is disabled
        legacy_recs = self._generate_recommendations(
            df, num_cols, cat_cols, trends, anomalies, kpis
        )

        self._log(
            f"Generated {len(kpis)} KPIs. Intelligence Mode: {self.intelligence_engine.mode.upper()}"
        )

        return InsightResult(
            kpi_list=kpis,
            trend_summary=trends,
            correlation_matrix=corr_matrix,
            anomaly_flags=anomalies,
            business_recommendations=legacy_recs[:5],
            executive_summary=narrative["executive_summary"],
            primary_risk=narrative["primary_risk"],
            primary_opportunity=narrative["primary_opportunity"],
            confidence_comment=narrative["confidence_comment"],
            key_relationships=filtered_corrs,
            detected_types=types,
            dataframe=df,
        )

    # ── KPIs ─────────────────────────────────────────────────────────
    def _compute_kpis(
        self, df: pd.DataFrame, num_cols: List[str], cat_cols: List[str]
    ) -> List[KPI]:
        kpis: List[KPI] = [
            KPI("Total Rows", len(df), "rows", "Number of records in the dataset"),
            KPI("Total Columns", len(df.columns), "cols", "Number of features"),
        ]

        for col in num_cols[:10]:  # limit to first 10 for readability
            kpis.extend([
                KPI(f"{col} — Mean", round(float(df[col].mean()), 2), "",
                    f"Average value of {col}"),
                KPI(f"{col} — Median", round(float(df[col].median()), 2), "",
                    f"Median value of {col}"),
                KPI(f"{col} — Std Dev", round(float(df[col].std()), 2), "",
                    f"Standard deviation of {col}"),
                KPI(f"{col} — Min", round(float(df[col].min()), 2), "",
                    f"Minimum value of {col}"),
                KPI(f"{col} — Max", round(float(df[col].max()), 2), "",
                    f"Maximum value of {col}"),
            ])

        for col in cat_cols[:5]:
            top = df[col].value_counts().head(1)
            if len(top) > 0:
                kpis.append(KPI(
                    f"{col} — Most Common",
                    f"{top.index[0]} ({top.values[0]})",
                    "", f"Most frequent value in {col}",
                ))

        return kpis

    # ── Trends ───────────────────────────────────────────────────────
    def _detect_trends(
        self, df: pd.DataFrame, num_cols: List[str], date_cols: List[str]
    ) -> List[TrendInfo]:
        trends: List[TrendInfo] = []
        for col in num_cols:
            if len(df[col].dropna()) < 5:
                continue
            # Use row index as x-axis (proxy for temporal order)
            x = np.arange(len(df))
            y = df[col].values.astype(float)
            mask = ~np.isnan(y)
            if mask.sum() < 5:
                continue
            slope, intercept, r, p, se = stats.linregress(x[mask], y[mask])
            if p < 0.05:
                direction = "increasing" if slope > 0 else "decreasing"
            else:
                direction = "stable"
            trends.append(TrendInfo(
                column=col,
                slope=round(slope, 6),
                direction=direction,
                p_value=round(p, 4),
                r_squared=round(r**2, 4)
            ))
        return trends

    # ── Correlations ─────────────────────────────────────────────────
    def _compute_correlations(
        self, df: pd.DataFrame, num_cols: List[str]
    ) -> Optional[pd.DataFrame]:
        if len(num_cols) < 2:
            return None
        return df[num_cols].corr().round(3)
        
    def _filter_correlations(self, df: pd.DataFrame, num_cols: List[str], corr_matrix: pd.DataFrame) -> List[str]:
        """
        Identify meaningful correlations, excluding:
        1. Trivial self-correlations (1.0)
        2. Derived columns (e.g. Tax = Total * 0.05)
        """
        if corr_matrix is None or len(num_cols) < 2:
            return []
            
        meaningful = []
        seen = set()
        
        for i, c1 in enumerate(num_cols):
            for c2 in num_cols[i + 1:]:
                pair_key = tuple(sorted((c1, c2)))
                if pair_key in seen:
                    continue
                seen.add(pair_key)
                
                r = corr_matrix.loc[c1, c2]
                if abs(r) > 0.7:
                    # Check for deterministic relationship (derived column)
                    # Simple heuristic: if variance of ratio is near zero
                    ratio = df[c1] / df[c2].replace(0, np.nan)
                    if ratio.std() < 0.01:
                        # Likely derived (e.g. c1 = k * c2)
                        continue
                        
                    meaningful.append(f"{c1} vs {c2} (r={r:.2f})")
                    
        return meaningful[:10] # Top 10

    # ── Anomalies ────────────────────────────────────────────────────
    def _flag_anomalies(
        self, df: pd.DataFrame, num_cols: List[str]
    ) -> List[AnomalyFlag]:
        flags: List[AnomalyFlag] = []
        for col in num_cols:
            series = df[col].dropna()
            if len(series) < 10:
                continue
            z = np.abs(stats.zscore(series))
            outlier_idx = list(series.index[z > 3])
            if outlier_idx:
                flags.append(AnomalyFlag(
                    column=col,
                    row_indices=outlier_idx[:50],  # cap
                    count=len(outlier_idx),
                ))
        return flags



    # ── Legacy Recommendations (Fallback) ────────────────────────────
    def _generate_recommendations(
        self,
        df: pd.DataFrame,
        num_cols: List[str],
        cat_cols: List[str],
        trends: List[TrendInfo],
        anomalies: List[AnomalyFlag],
        kpis: List[KPI],
    ) -> List[str]:
        recs: List[str] = []

        # High-variance columns
        for col in num_cols:
            cv = df[col].std() / (df[col].mean() + 1e-9)
            if abs(cv) > 1.0:
                recs.append(
                    f"⚠️ '{col}' has very high variability (CV={cv:.2f}). "
                    "Investigate causes — may indicate data quality issues or "
                    "genuine business volatility."
                )

        # Trending columns
        for t in trends:
            if t.direction == "increasing":
                recs.append(
                    f"📈 '{t.column}' is trending upward (slope={t.slope:.4f}). "
                    "Monitor for growth opportunities or capacity planning."
                )
            elif t.direction == "decreasing":
                recs.append(
                    f"📉 '{t.column}' is trending downward (slope={t.slope:.4f}). "
                    "Investigate root causes — may signal declining performance."
                )

        # Anomalies
        for a in anomalies:
            if a.count > 0:
                recs.append(
                    f"🔍 '{a.column}' has {a.count} anomalous records (z-score > 3). "
                    "Review these data points for errors or significant events."
                )

        # Strong correlations
        if len(num_cols) >= 2:
            corr = df[num_cols].corr()
            for i, c1 in enumerate(num_cols):
                for c2 in num_cols[i + 1:]:
                    r = corr.loc[c1, c2]
                    if abs(r) > 0.8:
                        recs.append(
                            f"🔗 Strong correlation ({r:.2f}) between '{c1}' and "
                            f"'{c2}'. Consider whether one can predict the other."
                        )

        if not recs:
            recs.append("✅ Dataset looks healthy — no major issues detected.")

        return recs
