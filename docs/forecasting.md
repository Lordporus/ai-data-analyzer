# Forecast Engine 3.0 -- Adaptive Model Selection

---

## Forecasting Objective

The Forecast Engine transforms historical time-series data into forward-looking projections with associated confidence grades. The engine is designed to maximize reliability by selecting the simplest model that adequately captures the data's structure. It does not default to complex models when simpler alternatives suffice.

**Location**: `agents/forecast.py`

---

## Model Selection Logic

The engine implements a two-tier model selection strategy. Model choice is determined entirely by data characteristics, not user configuration.

### Decision Tree

```
1. Detect date column and numeric metric columns
2. Determine resampling period (Daily / Weekly / Monthly / Yearly)
3. Run autocorrelation analysis across lags 1-12
4. Evaluate selection criteria:

   IF seasonality_detected
      AND dominant_lag >= 2
      AND data_length >= 2 * dominant_lag:
          --> Holt-Winters Exponential Smoothing (additive)
   ELSE:
          --> Linear Regression with confidence intervals
```

### Linear Regression

The default model. Fits a first-degree polynomial to the time-indexed metric values using `numpy.polyfit`. Produces:

- Point forecast for N future periods
- Slope and intercept
- R-squared goodness-of-fit
- P-value for slope significance
- Confidence intervals based on residual standard deviation

Linear regression is preferred when:
- No seasonal pattern is detected.
- Data length is insufficient for seasonal decomposition.
- Holt-Winters fitting fails or does not converge.

### Holt-Winters Exponential Smoothing

Used when seasonal structure is confirmed. Configured with:

- **Trend**: Additive
- **Seasonal**: Additive, with period set to the dominant autocorrelation lag
- **Initialization**: Estimated from data

Requires `statsmodels`. If the library is unavailable, the engine falls back to linear regression without error.

### Seasonality Detection

Autocorrelation is computed for lags 1 through 12 (or `data_length // 2`, whichever is smaller). A seasonal pattern is confirmed when:

1. Autocorrelation at any lag exceeds 0.5 in absolute value.
2. The dominant lag (highest absolute autocorrelation) is at least 2.
3. The data contains at least twice as many observations as the dominant lag.

These conditions prevent false-positive seasonality detection on short or noisy series.

---

## Confidence Grading

Each forecast receives a deterministic confidence grade based on statistical metrics computed during model fitting.

| Grade | R-squared | Additional Criteria |
|:------|:----------|:--------------------|
| High | > 0.7 | P-value < 0.05, low volatility index |
| Moderate | 0.4 -- 0.7 | Mixed statistical support |
| Low | < 0.4 | Weak fit or high residual variance |

### Volatility Index

Computed as the ratio of rolling standard deviation (mean) to the overall mean of the series. A high volatility index reduces effective confidence regardless of R-squared, as it indicates the series is inherently unpredictable.

```
volatility_index = rolling_std.mean() / (series_mean + epsilon)
```

### Residual Variance

After model fitting, residuals (observed minus predicted) are analyzed. High residual variance relative to the signal indicates the model is not capturing the underlying structure, which downgrades the confidence grade.

---

## Deterministic Fallback

The forecasting engine implements multiple fallback layers to ensure reliable output under all conditions.

### Minimum Data Thresholds

| Threshold | Value | Effect |
|:----------|:------|:-------|
| Minimum observations for any forecast | 8 | Below this, forecast is not generated |
| Minimum observations for Holt-Winters | 2 * seasonal_period | Below this, falls back to linear |
| Maximum NaN ratio in resampled series | 50% | Above this, switches from sum to mean aggregation |

### Automatic Fallback Chain

```
Holt-Winters (if criteria met)
    |
    | [failure / insufficient data]
    v
Linear Regression (default)
    |
    | [fewer than 8 points]
    v
Forecast suppressed (None returned)
```

### Why Deterministic Fallback Improves Reliability

Stochastic models can produce plausible-looking but fundamentally unreliable projections when applied to insufficient data. A linear model on 5 data points will produce a line, but that line has no statistical backing. Rather than presenting weak projections as valid forecasts, the engine suppresses output when confidence cannot be established. This prevents downstream consumers (the report layer, the dashboard) from displaying misleading projections that could influence business decisions.

The fallback chain is strict and non-configurable. This is intentional: relaxing thresholds at runtime would allow edge cases to produce unreliable output, undermining the system's credibility as a decision-support tool.

---

## Forecast Suppression Conditions

A forecast is excluded from the final output if any of the following conditions are met:

| Condition | Check Point |
|:----------|:------------|
| Fewer than 8 historical data points | ForecastAgent |
| Forecast array contains NaN values | ReportValidationEngine |
| Metric column matches a structural pattern (id, index, uuid) | ReportValidationEngine |
| No valid date column detected in the dataset | ForecastAgent (returns `is_time_series = False`) |
| All forecasts filtered by validation | ReportValidationEngine (returns `None`) |

When all forecasts are suppressed, the system logs the event at INFO level and proceeds with the remaining pipeline stages. The report and dashboard omit the forecast section gracefully rather than displaying empty or error states.
