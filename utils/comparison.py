from __future__ import annotations

from io import BytesIO
from typing import Any

import pandas as pd


COMPARISON_FILE_SIZE_LIMIT_BYTES = 10 * 1024 * 1024


class ComparisonValidationError(ValueError):
    """User-facing validation error for two-file CSV comparison."""


def load_comparison_csv(uploaded_file: Any, max_size_bytes: int = COMPARISON_FILE_SIZE_LIMIT_BYTES) -> pd.DataFrame:
    if uploaded_file is None:
        raise ComparisonValidationError("Upload a CSV file to compare.")

    size = getattr(uploaded_file, "size", None)
    if size is None and hasattr(uploaded_file, "getbuffer"):
        size = len(uploaded_file.getbuffer())
    if size is not None and size > max_size_bytes:
        raise ComparisonValidationError("Please upload a smaller CSV for comparison.")

    try:
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
    except pd.errors.EmptyDataError as exc:
        raise ComparisonValidationError("This CSV has no rows.") from exc
    except Exception as exc:
        raise ComparisonValidationError("Could not read this CSV. Please check formatting.") from exc

    if df.empty:
        raise ComparisonValidationError("This CSV has no rows.")
    return df


def summarize_file(df: pd.DataFrame) -> dict[str, Any]:
    empty_cols = [col for col in df.columns if df[col].isna().all()]
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_values": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "empty_columns": empty_cols,
    }


def _type_name(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    return "text"


def compare_two_dataframes(df_a: pd.DataFrame, df_b: pd.DataFrame) -> dict[str, Any]:
    cols_a = set(df_a.columns)
    cols_b = set(df_b.columns)
    shared_columns = sorted(cols_a & cols_b)
    only_a = sorted(cols_a - cols_b)
    only_b = sorted(cols_b - cols_a)

    type_mismatches = []
    for col in shared_columns:
        type_a = _type_name(df_a[col])
        type_b = _type_name(df_b[col])
        if type_a != type_b:
            type_mismatches.append({"column": col, "file_a_type": type_a, "file_b_type": type_b})

    numeric_rows = []
    for col in shared_columns:
        a_num = pd.to_numeric(df_a[col], errors="coerce")
        b_num = pd.to_numeric(df_b[col], errors="coerce")
        if a_num.notna().sum() == 0 or b_num.notna().sum() == 0:
            continue
        mean_a = float(a_num.mean())
        mean_b = float(b_num.mean())
        diff = mean_b - mean_a
        pct_diff = None if mean_a == 0 else (diff / abs(mean_a)) * 100
        higher = "Tie"
        if mean_b > mean_a:
            higher = "File B"
        elif mean_a > mean_b:
            higher = "File A"
        numeric_rows.append({
            "Metric": col,
            "File A Mean": round(mean_a, 2),
            "File B Mean": round(mean_b, 2),
            "Difference": round(diff, 2),
            "% Difference": None if pct_diff is None else round(pct_diff, 2),
            "Higher File": higher,
            "_rank": abs(pct_diff) if pct_diff is not None else abs(diff),
        })

    numeric_rows.sort(key=lambda row: row["_rank"], reverse=True)
    for row in numeric_rows:
        row.pop("_rank", None)

    summary_a = summarize_file(df_a)
    summary_b = summarize_file(df_b)
    row_diff = summary_b["rows"] - summary_a["rows"]

    return {
        "overview": {
            "file_a_rows": summary_a["rows"],
            "file_b_rows": summary_b["rows"],
            "row_difference": row_diff,
            "shared_columns_count": len(shared_columns),
            "only_a_count": len(only_a),
            "only_b_count": len(only_b),
        },
        "schema": {
            "shared_columns": shared_columns,
            "only_a": only_a,
            "only_b": only_b,
            "type_mismatches": type_mismatches,
        },
        "quality": {
            "file_a": summary_a,
            "file_b": summary_b,
        },
        "numeric_differences": numeric_rows,
        "plain_english_summary": build_plain_english_summary(row_diff, summary_a, summary_b, only_a, only_b, numeric_rows),
    }


def build_plain_english_summary(
    row_diff: int,
    summary_a: dict[str, Any],
    summary_b: dict[str, Any],
    only_a: list[str],
    only_b: list[str],
    numeric_rows: list[dict[str, Any]],
) -> str:
    parts = []
    if summary_a["rows"]:
        pct = (row_diff / summary_a["rows"]) * 100
        if row_diff > 0:
            parts.append(f"File B has {abs(pct):.1f}% more rows than File A.")
        elif row_diff < 0:
            parts.append(f"File B has {abs(pct):.1f}% fewer rows than File A.")
        else:
            parts.append("Both files have the same number of rows.")

    if numeric_rows:
        top = numeric_rows[0]
        pct_text = "an undefined percentage change" if top["% Difference"] is None else f"{abs(top['% Difference']):.1f}%"
        direction = "higher" if top["Difference"] > 0 else "lower" if top["Difference"] < 0 else "unchanged"
        parts.append(f"{top['Metric']} is {pct_text} {direction} in File B.")
    else:
        parts.append("No shared numeric columns were available for metric comparison.")

    if only_a:
        parts.append(f"File A has {len(only_a)} column(s) not present in File B.")
    if only_b:
        parts.append(f"File B has {len(only_b)} column(s) not present in File A.")

    if summary_a["missing_values"] or summary_b["missing_values"]:
        parts.append(
            f"Missing values: File A has {summary_a['missing_values']}, "
            f"File B has {summary_b['missing_values']}."
        )

    return " ".join(parts)


def dataframe_from_csv_text(text: str) -> pd.DataFrame:
    return pd.read_csv(BytesIO(text.encode("utf-8")))
