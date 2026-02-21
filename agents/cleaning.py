"""
CleaningAgent — Deduplicate, normalize, type-coerce, and handle
missing values & outliers in the ingested DataFrame.

Input:  IngestionResult
Output: CleaningResult (cleaned DataFrame + detailed log)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from agents.base import BaseAgent
from agents.ingestion import IngestionResult

logger = logging.getLogger(__name__)


@dataclass
class CleaningLog:
    """Human-readable record of every transformation applied."""
    steps: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, action: str, details: str, affected: int = 0) -> None:
        self.steps.append(
            {"action": action, "details": details, "rows_affected": affected}
        )

    def summary_text(self) -> str:
        lines = []
        for i, s in enumerate(self.steps, 1):
            lines.append(
                f"{i}. [{s['action']}] {s['details']} "
                f"(affected {s['rows_affected']} rows)"
            )
        return "\n".join(lines)


@dataclass
class CleaningResult:
    dataframe: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    cleaning_log: CleaningLog = field(default_factory=CleaningLog)
    detected_types: Dict[str, str] = field(default_factory=dict)
    outlier_report: Dict[str, int] = field(default_factory=dict)


class CleaningAgent(BaseAgent):
    """Apply a sequence of deterministic cleaning steps."""

    name = "CleaningAgent"

    def _execute(self, input_data: IngestionResult) -> CleaningResult:
        df = input_data.dataframe.copy()
        types = dict(input_data.detected_types)
        clog = CleaningLog()

        # 1 ── Normalize string columns (before dedup for better matching) ─
        str_cols = [c for c, t in types.items() if t in ("text", "categorical")]
        for col in str_cols:
            if col in df.columns:
                before = df[col].copy()
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.strip()
                    .str.replace(r"\s+", " ", regex=True)
                )
                changed = int((before != df[col]).sum())
                if changed:
                    clog.add("normalize_strings", f"Trimmed whitespace in '{col}'", changed)

        # 2 ── Remove exact duplicates (after normalization) ───────────
        n_dupes = int(df.duplicated().sum())
        if n_dupes > 0:
            df = df.drop_duplicates().reset_index(drop=True)
            clog.add("remove_duplicates", f"Removed {n_dupes} duplicate rows", n_dupes)
            self._log(f"Removed {n_dupes} duplicates")

        # 3 ── Convert date columns ───────────────────────────────────
        date_cols = [c for c, t in types.items() if t == "datetime"]
        for col in date_cols:
            if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
                try:
                    df[col] = pd.to_datetime(df[col])
                    clog.add("convert_dates", f"Parsed '{col}' to datetime", len(df))
                    self._log(f"Parsed '{col}' to datetime")
                except Exception:
                    clog.add("convert_dates", f"Could not parse '{col}' as datetime", 0)

        # 4 ── Handle missing numeric values (median fill) ────────────
        num_cols = [c for c, t in types.items() if t == "numeric"]
        for col in num_cols:
            if col in df.columns:
                n_missing = int(df[col].isnull().sum())
                if n_missing > 0:
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                    clog.add(
                        "fill_missing",
                        f"Filled {n_missing} nulls in '{col}' with median ({median_val:.2f})",
                        n_missing,
                    )

        # 5 ── Handle missing categorical values (mode fill) ──────────
        for col in str_cols:
            if col in df.columns:
                n_missing = int(df[col].isin(["nan", ""]).sum() + df[col].isnull().sum())
                if n_missing > 0:
                    mode_val = df[col].replace(["nan", ""], np.nan).mode()
                    if len(mode_val) > 0:
                        df[col] = df[col].replace(["nan", ""], mode_val.iloc[0])
                        df[col] = df[col].fillna(mode_val.iloc[0])
                        clog.add(
                            "fill_missing",
                            f"Filled {n_missing} blanks in '{col}' with mode ('{mode_val.iloc[0]}')",
                            n_missing,
                        )

        # 6 ── Detect outliers via IQR ────────────────────────────────
        outlier_report: Dict[str, int] = {}
        for col in num_cols:
            if col in df.columns:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                n_outliers = int(((df[col] < lower) | (df[col] > upper)).sum())
                if n_outliers > 0:
                    outlier_report[col] = n_outliers
                    clog.add(
                        "detect_outliers",
                        f"Found {n_outliers} outliers in '{col}' (IQR: {lower:.2f}–{upper:.2f})",
                        n_outliers,
                    )

        self._log(f"Cleaning complete — {len(clog.steps)} steps applied")
        return CleaningResult(
            dataframe=df,
            cleaning_log=clog,
            detected_types=types,
            outlier_report=outlier_report,
        )
