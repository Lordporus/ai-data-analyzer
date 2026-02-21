"""
IngestionAgent — Parse a raw CSV file, detect its schema, and produce
a structured summary ready for downstream agents.

Input:  pathlib.Path pointing to the uploaded CSV
Output: IngestionResult dataclass
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
import warnings
from typing import Any, Dict, List

import pandas as pd

# Silence technical warnings for cleaner business reporting
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")

from agents.base import BaseAgent

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    """Structured output of the IngestionAgent."""
    dataframe: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    column_names: List[str] = field(default_factory=list)
    detected_types: Dict[str, str] = field(default_factory=dict)
    missing_value_report: Dict[str, int] = field(default_factory=dict)
    missing_pct_report: Dict[str, float] = field(default_factory=dict)
    row_count: int = 0
    col_count: int = 0
    duplicate_count: int = 0
    sample_preview: List[Dict[str, Any]] = field(default_factory=list)
    file_size_bytes: int = 0


class IngestionAgent(BaseAgent):
    """Read a CSV, detect schema, and produce a structured summary."""

    name = "IngestionAgent"

    def _execute(self, input_data: Path) -> IngestionResult:
        file_path = Path(input_data)
        if not file_path.exists():
            raise FileNotFoundError(f"CSV not found: {file_path}")

        self._log(f"Reading {file_path.name} ({file_path.stat().st_size:,} bytes)")

        # Read CSV with smart type inference
        df = pd.read_csv(file_path, low_memory=False)
        self._log(f"Loaded {len(df):,} rows × {len(df.columns)} columns")

        # Detect column types
        detected_types: Dict[str, str] = {}
        for col in df.columns:
            dtype = str(df[col].dtype)
            if "int" in dtype or "float" in dtype:
                detected_types[col] = "numeric"
            elif "datetime" in dtype:
                detected_types[col] = "datetime"
            elif "bool" in dtype:
                detected_types[col] = "boolean"
            else:
                # Try to detect dates stored as strings
                sample = df[col].dropna().head(50)
                if len(sample) > 0:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            warnings.filterwarnings("ignore", category=UserWarning)
                            # Strict detection: >70% of sample must parse correctly as non-null dates
                            converted = pd.to_datetime(sample, errors="coerce")
                            if converted.notnull().sum() / len(sample) > 0.7:
                                detected_types[col] = "datetime"
                                continue
                    except (ValueError, TypeError):
                        pass
                nunique = df[col].nunique()
                if nunique <= min(20, len(df) * 0.05):
                    detected_types[col] = "categorical"
                else:
                    detected_types[col] = "text"

        # Missing values
        missing_counts = df.isnull().sum().to_dict()
        missing_pcts = (df.isnull().sum() / len(df) * 100).round(2).to_dict()

        result = IngestionResult(
            dataframe=df,
            column_names=list(df.columns),
            detected_types=detected_types,
            missing_value_report={k: int(v) for k, v in missing_counts.items()},
            missing_pct_report=missing_pcts,
            row_count=len(df),
            col_count=len(df.columns),
            duplicate_count=int(df.duplicated().sum()),
            sample_preview=df.head(5).to_dict(orient="records"),
            file_size_bytes=file_path.stat().st_size,
        )
        self._log(
            f"Schema detected — {sum(1 for v in detected_types.values() if v == 'numeric')} numeric, "
            f"{sum(1 for v in detected_types.values() if v == 'categorical')} categorical, "
            f"{sum(1 for v in detected_types.values() if v == 'datetime')} datetime columns"
        )
        return result
