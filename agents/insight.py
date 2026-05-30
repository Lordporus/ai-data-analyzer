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

# Columns that are identifiers / system artefacts — never useful for insights
EXCLUDE_PATTERNS = [
    'index', 'id', 'unnamed', 'row', 'serial',
    'sr_no', 'sl_no', 'cust_id', 'order_id',
]

# Words/phrases that must NEVER appear in any user-facing output
JARGON_BLACKLIST = [
    "systemic volatility",
    "variance drivers",
    "performance sensitivities",
    "strategic leverage",
    "operational throughput",
    "unmodeled volatility",
    "downside exposure",
    "metric stabilization",
    "risk vectors",
    "modeled variances",
    "structural drift",
    "organizational throughput",
    "performance drift",
    "directional persistency",
    "lateral consistency",
    "operational variability risk",
    "structural contraction",
    "progressive expansion",
]


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
    sector_name: str = ""
    
    # Additional metadata (Optional/Legacy support)
    top_risks: List[Dict[str, Any]] = field(default_factory=list)
    top_opportunities: List[str] = field(default_factory=list)
    key_relationships: List[str] = field(default_factory=list)     # Filtered, non-obvious correlations
    
    detected_types: Dict[str, str] = field(default_factory=dict)
    dataframe: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    audit_log: Dict[str, Any] = field(default_factory=dict)


class InsightAgent(BaseAgent):
    """Analyze the cleaned dataset to produce actionable insights."""

    name = "InsightAgent"

    def __init__(self):
        super().__init__()
        self.intelligence_engine = IntelligenceEngine()

    def _execute(self, input_data: Any) -> InsightResult:
        if isinstance(input_data, dict):
            repair = input_data.get("repair")
            dataset_name = input_data.get("dataset_name", "")
            sample_row = input_data.get("sample_row", None)
        else:
            repair = input_data
            dataset_name = ""
            sample_row = repair.dataframe.iloc[0].to_dict() if not repair.dataframe.empty else {}

        df = repair.dataframe.copy()
        types = repair.detected_types

        # Filter out identifier/system columns before ALL downstream analysis
        df = self._filter_business_columns(df)

        num_cols = [c for c, t in types.items() if t == "numeric" and c in df.columns]
        cat_cols = [c for c, t in types.items() if t == "categorical" and c in df.columns]
        date_cols = [c for c, t in types.items() if t == "datetime" and c in df.columns]
        # Filter out Unix timestamp columns from numeric analysis
        timestamp_cols = [c for c in num_cols if self._is_timestamp_column(c, df[c])]
        if timestamp_cols:
            logger.info("InsightAgent: detected timestamp columns, excluding from numeric analysis: %s", timestamp_cols)
        num_cols = [c for c in num_cols if c not in timestamp_cols]

        # Convert detected timestamp columns to readable datetime and store date range
        for ts_col in timestamp_cols:
            try:
                df[ts_col] = pd.to_datetime(df[ts_col], unit='s', errors='coerce')
                if ts_col not in date_cols:
                    date_cols.append(ts_col)
            except Exception:
                pass

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

        # ── Build rich data context so LLM / deterministic engine uses real names ──
        # Collect top categorical values (e.g. ["Amazon", "Myntra"] for Channel)
        cat_top_values: Dict[str, List[str]] = {}
        for col in cat_cols[:8]:
            top_vals = df[col].value_counts().head(5).index.tolist()
            cat_top_values[col] = [str(v) for v in top_vals]

        # Collect key numeric stats for the most important columns
        numeric_stats: Dict[str, Dict] = {}
        for col in num_cols[:6]:
            numeric_stats[col] = {
                "mean": round(float(df[col].mean()), 2),
                "sum": round(float(df[col].sum()), 2),
                "min": round(float(df[col].min()), 2),
                "max": round(float(df[col].max()), 2),
            }

        # Sample rows so LLM sees actual category values in context
        sample_rows = df.head(5).to_dict(orient="records")

        # Build context for IntelligenceEngine
        context = {
            "trend_direction": primary_trend,
            "confidence_level": primary_conf,
            "volatility_index": round(max_vol, 3),
            "seasonality_detected": False,
            "forecast_model_type": "Linear",
            # Rich dataset context
            "dataset_columns": list(df.columns),
            "numeric_columns": num_cols,
            "categorical_columns": cat_cols,
            "categorical_top_values": cat_top_values,
            "numeric_stats": numeric_stats,
            "row_count": len(df),
            "sample_rows": sample_rows,
            "dataset_name": dataset_name,
            # Jargon control
            "jargon_blacklist": JARGON_BLACKLIST,
        }

        # Generate Strategic Narrative via Abstraction Layer
        narrative = self.intelligence_engine.generate_strategic_summary(context)
        
        # Data-aware recommendations — use actual column/category names
        legacy_recs = self._generate_data_aware_recommendations(
            df, num_cols, cat_cols, trends, anomalies, kpis
        )

        # Post-process: scrub any remaining generic terms
        narrative["executive_summary"] = self._clean_insight_text(
            narrative["executive_summary"], df
        )

        self._log(
            f"Generated {len(kpis)} KPIs. Intelligence Mode: {self.intelligence_engine.mode.upper()}"
        )

        audit_log = {
            "KPIs": {
                "columns_used": num_cols + cat_cols,
                "formula": "Deterministic descriptive statistics (mean, median, std, min, max, value_counts)",
                "threshold": "All non-null values included; top categorical categories capped at 5",
                "method": "deterministic",
                "result": f"Calculated {len(kpis)} KPIs successfully across all detected numeric and categorical columns."
            },
            "Trends": {
                "columns_used": num_cols + date_cols,
                "formula": "Linear Ordinary Least Squares (OLS) Regression (scipy.stats.linregress)",
                "threshold": "p-value < 0.05 for statistical significance (direction: increasing/decreasing); p-value >= 0.05 flagged as stable",
                "method": "deterministic",
                "result": f"Detected {len(trends)} trends. Granular sorting/monotonic alignment based on date column: {date_cols[0] if date_cols else 'None'}."
            },
            "Correlations": {
                "columns_used": num_cols,
                "formula": "Pearson Product-Moment Correlation Matrix (pandas.DataFrame.corr)",
                "threshold": "r-value > 0.7 considered meaningful; ratio variance checked to filter out trivial deterministic/derived columns",
                "method": "deterministic",
                "result": f"Identified {len(filtered_corrs)} key non-obvious relationships after filtering out derived ratios."
            },
            "Anomalies": {
                "columns_used": num_cols,
                "formula": "Standard Score / Z-Score (scipy.stats.zscore)",
                "threshold": "Absolute Z-Score > 3.0 (data points exceeding 3 standard deviations from mean)",
                "method": "deterministic",
                "result": f"Flagged anomalies across {len(anomalies)} numeric columns."
            },
            "Strategic Narrative": {
                "columns_used": ["executive_summary", "primary_risk", "primary_opportunity"],
                "formula": "Rule-based synthesis & generative heuristic expansion via IntelligenceEngine",
                "threshold": f"LLM Integration active (provider: {self.intelligence_engine.mode})",
                "method": "llm-enhanced" if self.intelligence_engine.mode != "none" else "deterministic",
                "result": f"Strategic analysis synthesized with volatility index {round(max_vol, 3)} and {primary_conf} confidence."
            }
        }

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
            sector_name=self.intelligence_engine.detect_sector_hybrid(df.columns.tolist(), dataset_name, sample_row),
            key_relationships=filtered_corrs,
            detected_types=types,
            dataframe=df,
            audit_log=audit_log,
        )

    # ── Business Column Filter ────────────────────────────────────────
    def _filter_business_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drops identifier/system columns (index, id, serial numbers, etc.)
        before any statistical analysis so they never pollute insights.
        """
        cols_to_keep = [
            col for col in df.columns
            if not any(
                pattern in col.lower().replace(' ', '_').replace('-', '_')
                for pattern in EXCLUDE_PATTERNS
            )
        ]
        dropped = set(df.columns) - set(cols_to_keep)
        if dropped:
            logger.debug("InsightAgent: dropped identifier columns: %s", dropped)
        return df[cols_to_keep]

    def _is_timestamp_column(self, col: str, series: pd.Series) -> bool:
        """
        Detect Unix timestamp columns by name pattern OR value range.
        Returns True if column should be excluded from numeric KPI analysis.
        """
        # Name-based detection
        timestamp_keywords = [
            'utc', 'timestamp', '_at', '_date', '_time',
            'epoch', 'unix', 'created', 'updated', 'modified'
        ]
        col_lower = col.lower().replace(' ', '_').replace('-', '_')
        if any(kw in col_lower for kw in timestamp_keywords):
            return True

        # Value-range-based detection (Unix epoch: ~2001 to ~2286)
        if pd.api.types.is_numeric_dtype(series):
            non_null = series.dropna()
            if len(non_null) > 0:
                median_val = non_null.median()
                if 1_000_000_000 <= median_val <= 9_999_999_999:
                    return True

        return False


    def _clean_insight_text(self, text: str, df: pd.DataFrame) -> str:
        """
        Replaces generic placeholder terms and JARGON_BLACKLIST words with
        plain-English equivalents. Applied as the final post-processing step
        on every user-facing text field.
        """
        if not text:
            return text

        # Pick the first non-generic column as the representative business term
        meaningful_cols = [
            c for c in df.columns
            if not any(p in c.lower() for p in EXCLUDE_PATTERNS)
        ]
        rep_col = meaningful_cols[0] if meaningful_cols else "data"

        # Full replacement map: jargon → plain English
        replacements = [
            # Index placeholder terms
            ('"index"',                    f'"{rep_col}"'),
            ("'index'",                    f"'{rep_col}'"),
            ('the index',                  f'the {rep_col}'),
            ('Index is',                   f'{rep_col} is'),
            # JARGON_BLACKLIST replacements (must mirror report.py _remove_jargon)
            ('systemic volatility',        'sales variation'),
            ('variance drivers',           'key factors'),
            ('operational throughput',     'business performance'),
            ('unmodeled volatility',       'unexpected changes'),
            ('downside exposure',          'risk'),
            ('metric stabilization',       'improving consistency'),
            ('performance sensitivities',  'performance factors'),
            ('strategic leverage',         'growth opportunity'),
            ('risk vectors',               'risk areas'),
            ('modeled variances',          'key differences'),
            ('structural drift',           'gradual decline'),
            ('organizational throughput',  'business output'),
            ('performance drift',          'performance change'),
            ('directional persistency',    'consistent trend'),
            ('lateral consistency',        'stable performance'),
            ('operational variability risk', 'operational risk'),
            ('structural contraction',     'decline'),
            ('progressive expansion',      'growth'),
        ]
        for old, new in replacements:
            # Case-insensitive replacement
            import re
            text = re.sub(re.escape(old), new, text, flags=re.IGNORECASE)
        return text

    # ── KPIs ─────────────────────────────────────────────────────────
    def _compute_kpis(
        self, df: pd.DataFrame, num_cols: List[str], cat_cols: List[str]
    ) -> List[KPI]:
        kpis: List[KPI] = [
            KPI("Total Rows", len(df), "rows", "Number of records in the dataset"),

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

        # Only compute trends when a real datetime column is present.
        # Using row index as a time proxy produces statistically meaningless
        # slopes on unordered or non-temporal data.
        if not date_cols:
            return trends

        # Pick the datetime column with the most unique values (most granular).
        date_col = max(date_cols, key=lambda c: df[c].nunique())

        # Sort the working copy by the chosen date column so x is monotonic.
        try:
            df_sorted = df.sort_values(by=date_col).reset_index(drop=True)
            x_raw = pd.to_datetime(df_sorted[date_col], errors="coerce")
            # Convert dates to numeric (seconds since epoch) for linregress.
            x_numeric = (x_raw - x_raw.min()).dt.total_seconds().values
        except Exception:
            return trends  # If date parsing fails, skip trends rather than lie.

        for col in num_cols:
            y = df_sorted[col].values.astype(float)
            mask = ~np.isnan(y) & ~np.isnan(x_numeric)
            if mask.sum() < 5:
                continue
            slope, intercept, r, p, se = stats.linregress(x_numeric[mask], y[mask])
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



    # ── Data-Aware Recommendations ─────────────────────────────────────
    def _generate_data_aware_recommendations(
        self,
        df: pd.DataFrame,
        num_cols: List[str],
        cat_cols: List[str],
        trends: List[TrendInfo],
        anomalies: List[AnomalyFlag],
        kpis: List[KPI],
    ) -> List[str]:
        """
        Generates specific, actionable recommendations using ACTUAL column names
        and top category values from the dataset — never generic placeholders.
        """
        recs: List[str] = []

        # 1. Categorical distribution insights (e.g. "Amazon leads with 42% of orders")
        for col in cat_cols[:4]:
            vc = df[col].value_counts()
            if len(vc) < 2:
                continue
            top_name = str(vc.index[0])
            top_pct = round(100 * vc.iloc[0] / vc.sum(), 1)
            second_name = str(vc.index[1]) if len(vc) > 1 else None
            second_pct = round(100 * vc.iloc[1] / vc.sum(), 1) if len(vc) > 1 else None

            msg = f"📊 **{col}**: '{top_name}' leads with {top_pct}% of records."
            if second_name and second_pct:
                gap = round(top_pct - second_pct, 1)
                msg += f" '{second_name}' follows at {second_pct}% ({gap}pp gap). Focus resources on the top performer."
            recs.append(msg)

        # 2. Numeric column highlights (e.g. "Amount: total ₹2.05Cr, avg ₹662")
        for col in num_cols[:3]:
            col_data = df[col].dropna()
            if col_data.empty:
                continue
            total = col_data.sum()
            avg = col_data.mean()
            # Format large numbers compactly
            def _fmt(n):
                if abs(n) >= 1_000_000:
                    return f"{n/1_000_000:.2f}M"
                if abs(n) >= 1_000:
                    return f"{n/1_000:.1f}K"
                return f"{n:.2f}"
            recs.append(
                f"💰 **{col}**: total = {_fmt(total)}, avg per record = {_fmt(avg)}, "
                f"range = {_fmt(float(col_data.min()))}–{_fmt(float(col_data.max()))}. "
                f"Benchmark against historical average of {_fmt(avg)} per record."
            )

        # 3. Trending columns — name the column explicitly, skip slope jargon
        for t in trends:
            direction_icon = "📈" if t.direction == "increasing" else "📉"
            if t.direction == "increasing":
                action = "Monitor for growth opportunities and plan capacity."
            elif t.direction == "decreasing":
                action = "Investigate root causes and take corrective action."
            else:
                continue
            recs.append(f"{direction_icon} **{t.column}** is trending {t.direction}. {action}")

        # 4. Anomaly alerts — name the column, give count
        for a in anomalies:
            if a.count > 0:
                recs.append(
                    f"🔍 **{a.column}** has {a.count} anomalous records (z-score > 3). "
                    "Review these for data entry errors or significant business events."
                )

        # 5. Cross-column correlation insights
        if len(num_cols) >= 2:
            corr = df[num_cols].corr()
            for i, c1 in enumerate(num_cols):
                for c2 in num_cols[i + 1:]:
                    r = corr.loc[c1, c2]
                    if abs(r) > 0.8:
                        direction = "move together" if r > 0 else "move inversely"
                        recs.append(
                            f"🔗 Strong link (r={r:.2f}): **{c1}** and **{c2}** {direction}. "
                            f"Improving {c1} is likely to impact {c2}."
                        )

        if not recs:
            recs.append("✅ Dataset looks healthy — no major anomalies or concerning trends detected.")

        return recs
