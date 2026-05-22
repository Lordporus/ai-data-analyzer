"""
Predictive forecasting module — Apply simple time-series forecasting
to date-indexed numeric columns.

Uses linear regression + exponential smoothing as lightweight
forecasting methods (no heavy ML dependencies required).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ForecastResult:
    column: str
    method: str
    forecast_values: List[float] = field(default_factory=list)
    confidence_lower: List[float] = field(default_factory=list)
    confidence_upper: List[float] = field(default_factory=list)
    r_squared: float = 0.0
    historical_values: List[float] = field(default_factory=list)


def forecast_column(
    series: pd.Series,
    periods: int = 10,
    method: str = "linear",
) -> ForecastResult:
    """Forecast future values for a numeric series.

    Args:
        series: Numeric series to forecast (index is ignored, order matters).
        periods: Number of future periods to forecast.
        method: 'linear' or 'exponential_smoothing'.

    Returns:
        ForecastResult with predictions and confidence bounds.
    """
    values = series.dropna().values.astype(float)
    n = len(values)

    if n < 5:
        return ForecastResult(
            column=series.name or "unknown",
            method=method,
            historical_values=list(values),
        )

    if method == "exponential_smoothing":
        return _exponential_smoothing(values, series.name, periods)
    else:
        return _linear_forecast(values, series.name, periods)


def _linear_forecast(
    values: np.ndarray, name: str, periods: int
) -> ForecastResult:
    """Simple linear regression forecast."""
    x = np.arange(len(values))
    slope, intercept, r, p, se = stats.linregress(x, values)
    r_sq = r ** 2

    future_x = np.arange(len(values), len(values) + periods)
    predictions = slope * future_x + intercept

    # Confidence interval (±2 * residual std)
    residuals = values - (slope * x + intercept)
    res_std = np.std(residuals)

    return ForecastResult(
        column=str(name),
        method="linear",
        forecast_values=[round(float(v), 2) for v in predictions],
        confidence_lower=[round(float(v - 2 * res_std), 2) for v in predictions],
        confidence_upper=[round(float(v + 2 * res_std), 2) for v in predictions],
        r_squared=round(float(r_sq), 4),
        historical_values=[round(float(v), 2) for v in values[-20:]],
    )


def _exponential_smoothing(
    values: np.ndarray, name: str, periods: int, alpha: float = 0.3
) -> ForecastResult:
    """Simple exponential smoothing forecast."""
    # Compute smoothed series
    smoothed = np.zeros(len(values))
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = alpha * values[i] + (1 - alpha) * smoothed[i - 1]

    # Forecast = last smoothed value repeated
    last_val = smoothed[-1]
    predictions = [round(float(last_val), 2)] * periods

    # Confidence based on recent residual std
    recent_residuals = values[-20:] - smoothed[-20:]
    res_std = float(np.std(recent_residuals))

    # Widening confidence intervals
    lower = [round(last_val - 2 * res_std * (1 + 0.1 * i), 2) for i in range(periods)]
    upper = [round(last_val + 2 * res_std * (1 + 0.1 * i), 2) for i in range(periods)]

    return ForecastResult(
        column=str(name),
        method="exponential_smoothing",
        forecast_values=predictions,
        confidence_lower=lower,
        confidence_upper=upper,
        r_squared=0.0,
        historical_values=[round(float(v), 2) for v in values[-20:]],
    )


def auto_forecast(
    df: pd.DataFrame,
    detected_types: Dict[str, str],
    periods: int = 10,
) -> List[ForecastResult]:
    """Automatically forecast all numeric columns in the DataFrame."""
    results = []
    num_cols = [c for c, t in detected_types.items() if t == "numeric" and c in df.columns]

    for col in num_cols[:5]:  # limit to 5 columns
        result = forecast_column(df[col], periods=periods, method="linear")
        results.append(result)

    return results
