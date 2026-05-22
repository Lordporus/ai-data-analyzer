"""
Multi-file comparison module — Compare statistics across multiple
uploaded CSV files for side-by-side analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import numpy as np


@dataclass
class ComparisonResult:
    file_names: List[str] = field(default_factory=list)
    comparison_table: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    shared_columns: List[str] = field(default_factory=list)
    metric_diffs: Dict[str, Any] = field(default_factory=dict)


def compare_datasets(file_paths: List[Path]) -> ComparisonResult:
    """Compare basic statistics across multiple CSV files.

    Args:
        file_paths: List of paths to CSV files.

    Returns:
        ComparisonResult with summary statistics per file.
    """
    records: List[Dict[str, Any]] = []
    all_columns: List[set] = []

    for fp in file_paths:
        df = pd.read_csv(fp)
        num_cols = df.select_dtypes(include="number").columns.tolist()

        row: Dict[str, Any] = {
            "file": fp.name,
            "rows": len(df),
            "columns": len(df.columns),
            "missing_total": int(df.isnull().sum().sum()),
            "missing_pct": round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 2),
            "duplicates": int(df.duplicated().sum()),
            "numeric_cols": len(num_cols),
        }

        # Per-column stats for shared comparison
        for col in num_cols[:10]:
            row[f"{col}_mean"] = round(float(df[col].mean()), 2)
            row[f"{col}_std"] = round(float(df[col].std()), 2)
            row[f"{col}_min"] = round(float(df[col].min()), 2)
            row[f"{col}_max"] = round(float(df[col].max()), 2)

        records.append(row)
        all_columns.append(set(df.columns))

    # Find shared columns across all files
    shared = set.intersection(*all_columns) if all_columns else set()

    comp_df = pd.DataFrame(records)

    return ComparisonResult(
        file_names=[fp.name for fp in file_paths],
        comparison_table=comp_df,
        shared_columns=sorted(shared),
    )
