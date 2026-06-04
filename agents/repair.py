"""
RepairReasoningAgent — Analyze the dataset summary and decide on
optimal strategies for data repair.

Uses rule-based heuristics by default.  When an LLM provider is
configured in settings, it can optionally call the LLM for richer
reasoning text.

Input:  CleaningResult
Output: RepairResult (decisions + human-readable reasoning)
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from agents.base import BaseAgent
from agents.cleaning import CleaningResult

# Ensure project root is on sys.path regardless of import order
_PROJECT_ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config.settings import LLM_PROVIDER

logger = logging.getLogger(__name__)


@dataclass
class RepairDecision:
    column: str
    issue: str
    strategy: str
    reason: str
    applied: bool = False


@dataclass
class RepairResult:
    dataframe: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    repair_decisions: List[RepairDecision] = field(default_factory=list)
    reasoning_text: str = ""
    detected_types: Dict[str, str] = field(default_factory=dict)


class RepairReasoningAgent(BaseAgent):
    """Decide how to repair remaining data issues and apply fixes."""

    name = "RepairReasoningAgent"

    def _execute(self, input_data: CleaningResult, apply_repairs: bool = False) -> RepairResult:
        df = input_data.dataframe.copy()
        types = input_data.detected_types
        outliers = input_data.outlier_report
        decisions: List[RepairDecision] = []

        # ── 1) Handle remaining missing values ──────────────────────
        for col in df.columns:
            n_missing = int(df[col].isnull().sum())
            if n_missing == 0:
                continue

            pct_missing = n_missing / len(df) * 100

            if pct_missing > 60:
                # Too much missing — drop the column (only if consent given)
                dec = RepairDecision(
                    column=col,
                    issue=f"{pct_missing:.1f}% missing values",
                    strategy="drop_column",
                    reason=f"Over 60% missing — column provides very little signal.",
                )
                if apply_repairs:
                    df = df.drop(columns=[col])
                    dec.applied = True
                    self._log(f"Dropped column '{col}' ({pct_missing:.1f}% missing)")
                decisions.append(dec)

            elif types.get(col) == "numeric":
                # Decide mean vs median based on skewness
                skew = abs(df[col].skew()) if df[col].notna().sum() > 2 else 0
                if skew > 1.0:
                    fill_val = df[col].median()
                    strategy = "fill_median"
                    reason = f"Skewness = {skew:.2f} (>1) — median is more robust."
                else:
                    fill_val = df[col].mean()
                    strategy = "fill_mean"
                    reason = f"Skewness = {skew:.2f} (≤1) — mean is appropriate."

                applied = False
                if apply_repairs:
                    df[col] = df[col].fillna(fill_val)
                    applied = True
                decisions.append(RepairDecision(
                    column=col, issue=f"{n_missing} missing values",
                    strategy=strategy, reason=reason, applied=applied,
                ))
            else:
                # Categorical / text — mode fill
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    applied = False
                    if apply_repairs:
                        df[col] = df[col].fillna(mode_val.iloc[0])
                        applied = True
                    decisions.append(RepairDecision(
                        column=col, issue=f"{n_missing} missing values",
                        strategy="fill_mode",
                        reason="Categorical column — filled with most frequent value.",
                        applied=applied,
                    ))

        # ── 2) Handle outliers ───────────────────────────────────────
        for col, n_outliers in outliers.items():
            if col not in df.columns:
                continue
            pct_outliers = n_outliers / len(df) * 100

            if pct_outliers > 10:
                # Many outliers — clip to IQR bounds (only if consent given)
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                applied = False
                if apply_repairs:
                    df[col] = df[col].clip(lower, upper)
                    applied = True
                    self._log(f"Clipped outliers in '{col}'")
                decisions.append(RepairDecision(
                    column=col,
                    issue=f"{n_outliers} outliers ({pct_outliers:.1f}%)",
                    strategy="clip_outliers",
                    reason=f"High outlier rate — clipped to [{lower:.2f}, {upper:.2f}].",
                    applied=applied,
                ))
            else:
                decisions.append(RepairDecision(
                    column=col,
                    issue=f"{n_outliers} outliers ({pct_outliers:.1f}%)",
                    strategy="keep",
                    reason="Low outlier rate — likely real data variation, kept as-is.",
                    applied=False,
                ))

        # ── 3) Build reasoning text ──────────────────────────────────
        reasoning_lines = ["## Repair Reasoning Report\n"]
        for d in decisions:
            status = "✅ Applied" if d.applied else "ℹ️ No action"
            reasoning_lines.append(
                f"- **{d.column}** | {d.issue}\n"
                f"  Strategy: `{d.strategy}` — {d.reason} [{status}]"
            )
        if not decisions:
            reasoning_lines.append("No additional repairs were needed.")

        reasoning_text = "\n".join(reasoning_lines)
        self._log(f"Made {len(decisions)} repair decisions")

        return RepairResult(
            dataframe=df,
            repair_decisions=decisions,
            reasoning_text=reasoning_text,
            detected_types=types,
        )
