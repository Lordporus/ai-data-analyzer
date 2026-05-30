"""
DataQualityAgent — Compute a comprehensive 0–100 Data Quality Score
using weighted metrics: missing values, duplicates, type consistency,
column integrity, and variance health.

Designed to run twice (pre- and post-cleaning) to show improvement.

Input:  pandas DataFrame + detected_types dict
Output: DataQualityResult (score, percentages, problem columns, summary)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from agents.base import BaseAgent

logger = logging.getLogger(__name__)

# ── Scoring weights ──────────────────────────────────────────────────
WEIGHT_MISSING = 0.30
WEIGHT_DUPLICATES = 0.20
WEIGHT_TYPE_CONSISTENCY = 0.20
WEIGHT_COLUMN_INTEGRITY = 0.15
WEIGHT_VARIANCE_HEALTH = 0.15


def score_color(score: float) -> str:
    """Return a hex colour based on the quality score band."""
    if score >= 80:
        return "#22c55e"   # green
    elif score >= 60:
        return "#eab308"   # yellow
    return "#ef4444"       # red


def risk_level(score: float) -> str:
    """Human-readable risk label."""
    if score >= 80:
        return "Low"
    elif score >= 60:
        return "Medium"
    return "High"


@dataclass
class DataQualityResult:
    """Structured output of the DataQualityAgent."""
    quality_score: float = 0.0
    missing_percent: float = 0.0
    duplicate_percent: float = 0.0
    problem_columns: List[Dict[str, Any]] = field(default_factory=list)
    summary_text: str = ""

    # Breakdown for UI
    schema_issues: int = 0
    risk_level: str = "Unknown"
    score_color: str = "#8b949e"

    # Detailed metrics (for advanced consumers)
    total_rows: int = 0
    total_cols: int = 0
    null_heavy_columns: List[str] = field(default_factory=list)
    constant_columns: List[str] = field(default_factory=list)
    low_variance_columns: List[str] = field(default_factory=list)
    type_mismatch_count: int = 0

    # Sub-scores (0–100 each, before weighting)
    sub_missing: float = 100.0
    sub_duplicates: float = 100.0
    sub_type_consistency: float = 100.0
    sub_column_integrity: float = 100.0
    sub_variance_health: float = 100.0


class DataQualityAgent(BaseAgent):
    """Assess data quality and produce a weighted 0–100 score."""

    name = "DataQualityAgent"

    def _execute(self, input_data: dict) -> DataQualityResult:
        df: pd.DataFrame = input_data["dataframe"]
        types: Dict[str, str] = input_data.get("detected_types", {})

        total_rows, total_cols = df.shape

        # ── Detect dataset type for dynamic thresholds ──────────────────
        col_names_lower = [c.lower() for c in df.columns]
        
        is_financial = any(kw in c for c in col_names_lower 
                           for kw in ['price', 'revenue', 'profit', 'close', 
                                      'open', 'volume', 'return', 'yield'])
        is_timeseries = any(kw in c for c in col_names_lower 
                            for kw in ['timestamp', 'utc', '_at', '_date', 
                                       '_time', 'epoch', 'created', 'updated'])
        is_transactional = any(kw in c for c in col_names_lower 
                               for kw in ['order', 'transaction', 'invoice', 
                                          'payment', 'receipt'])
        is_social = any(kw in c for c in col_names_lower 
                        for kw in ['post', 'comment', 'subreddit', 'upvote', 
                                   'reddit', 'tweet', 'like', 'share'])

        # Dynamic thresholds based on dataset type
        if is_financial:
            dup_penalty_multiplier = 10     # duplicates are critical error
            null_heavy_threshold = 10       # very strict
            dup_weight, missing_weight = 0.30, 0.30
        elif is_timeseries:
            dup_penalty_multiplier = 8      # duplicates are serious
            null_heavy_threshold = 20       # stricter missing tolerance
            dup_weight, missing_weight = 0.25, 0.35
        elif is_transactional:
            dup_penalty_multiplier = 2      # duplicates more acceptable
            null_heavy_threshold = 50       # higher tolerance for missing
            dup_weight, missing_weight = 0.10, 0.25
        elif is_social:
            dup_penalty_multiplier = 3      # some duplicate posts ok
            null_heavy_threshold = 40       # social data often sparse
            dup_weight, missing_weight = 0.15, 0.25
        else:
            dup_penalty_multiplier = 5      # default (original behavior)
            null_heavy_threshold = 30       # default
            dup_weight, missing_weight = 0.20, 0.30

        remaining = 1.0 - dup_weight - missing_weight
        type_w    = round(remaining * (0.20 / 0.50), 4)
        integrity_w = round(remaining * (0.15 / 0.50), 4)
        variance_w  = round(remaining - type_w - integrity_w, 4)

        result = DataQualityResult(total_rows=total_rows, total_cols=total_cols)

        # ── 1) Missing values (30%) ──────────────────────────────────
        total_cells = total_rows * total_cols
        missing_cells = int(df.isnull().sum().sum())
        result.missing_percent = round(
            (missing_cells / total_cells * 100) if total_cells else 0, 2
        )
        # Score: 0% missing → 100, 100% missing → 0
        result.sub_missing = max(0.0, 100.0 - result.missing_percent)

        # ── 2) Duplicates (20%) ──────────────────────────────────────
        dup_count = int(df.duplicated().sum())
        result.duplicate_percent = round(
            (dup_count / total_rows * 100) if total_rows else 0, 2
        )
        result.sub_duplicates = max(0.0, 100.0 - result.duplicate_percent * dup_penalty_multiplier)
        # ×5 penalty: 10% duplicates → score 50, 20% → 0

        # ── 3) Type consistency (20%) ────────────────────────────────
        mismatch_count = 0
        for col, expected_type in types.items():
            if col not in df.columns:
                continue
            if expected_type == "numeric":
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    non_numeric = pd.to_numeric(non_null, errors="coerce").isnull().sum()
                    mismatch_count += int(non_numeric)
            elif expected_type == "datetime":
                non_null = df[col].dropna()
                if len(non_null) > 0 and non_null.dtype == object:
                    coerced = pd.to_datetime(non_null, errors="coerce")
                    mismatch_count += int(coerced.isnull().sum())

        result.type_mismatch_count = mismatch_count
        mismatch_pct = (mismatch_count / total_cells * 100) if total_cells else 0
        result.sub_type_consistency = max(0.0, 100.0 - mismatch_pct * 10)

        # ── 4) Column integrity (15%) ────────────────────────────────
        # Null-heavy cols (>30% missing)
        null_heavy: List[str] = []
        constant_cols: List[str] = []
        for col in df.columns:
            missing_pct = df[col].isnull().mean() * 100
            if missing_pct > null_heavy_threshold:
                null_heavy.append(col)
            if df[col].nunique(dropna=True) <= 1:
                constant_cols.append(col)

        result.null_heavy_columns = null_heavy
        result.constant_columns = constant_cols

        problem_col_count = len(null_heavy) + len(constant_cols)
        integrity_penalty = (problem_col_count / total_cols * 100) if total_cols else 0
        result.sub_column_integrity = max(0.0, 100.0 - integrity_penalty)

        # ── 5) Variance Health (15%) ─────────────────────────────────
        num_cols = [c for c, t in types.items() if t == "numeric" and c in df.columns]
        low_var: List[str] = []
        if num_cols:
            for col in num_cols:
                series = df[col].dropna()
                if len(series) > 1:
                    cv = series.std() / series.mean() if series.mean() != 0 else 0
                    if abs(cv) < 0.01:  # coefficient of variation < 1%
                        low_var.append(col)

        result.low_variance_columns = low_var
        low_var_pct = (len(low_var) / len(num_cols) * 100) if num_cols else 0
        result.sub_variance_health = max(0.0, 100.0 - low_var_pct)

        # ── Weighted final score ─────────────────────────────────────
        result.quality_score = round(
            result.sub_missing * missing_weight
            + result.sub_duplicates * dup_weight
            + result.sub_type_consistency * type_w
            + result.sub_column_integrity * integrity_w
            + result.sub_variance_health * variance_w,
            1,
        )

        # ── Derived fields ───────────────────────────────────────────
        result.schema_issues = mismatch_count + len(constant_cols)
        result.risk_level = risk_level(result.quality_score)
        result.score_color = score_color(result.quality_score)

        # ── Problem columns roster ───────────────────────────────────
        problems: List[Dict[str, Any]] = []
        for col in null_heavy:
            problems.append({"column": col, "issue": "null_heavy", "detail": f">{null_heavy_threshold}% missing"})
        for col in constant_cols:
            problems.append({"column": col, "issue": "constant", "detail": "single unique value"})
        for col in low_var:
            problems.append({"column": col, "issue": "low_variance", "detail": "CV < 1%"})
        result.problem_columns = problems

        # ── Summary text ─────────────────────────────────────────────
        if is_financial:
            detected_type = "Financial"
        elif is_timeseries:
            detected_type = "Time-Series"
        elif is_transactional:
            detected_type = "Transactional"
        elif is_social:
            detected_type = "Social Media"
        else:
            detected_type = "General"

        result.summary_text = (
            f"[{detected_type} Dataset] Data Quality Score: {result.quality_score}/100 ({result.risk_level} risk). "
            f"{result.missing_percent}% missing, {result.duplicate_percent}% duplicates, "
            f"{result.schema_issues} schema issues, "
            f"{len(null_heavy)} null-heavy column(s), "
            f"{len(constant_cols)} constant column(s), "
            f"{len(low_var)} low-variance column(s)."
        )

        self._log(result.summary_text)
        return result
