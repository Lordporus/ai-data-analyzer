"""
ForecastAgent — Project future trends and simulate business scenarios.

Input:  InsightResult (for context), RepairResult (for data)
Output: ForecastResult

Key Capabilities:
1. Auto-detect time-series suitability.
2. Generate base forecast (linear/rolling).
3. Simulate "what-if" scenarios (e.g. Price +5%).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

from agents.base import BaseAgent
from agents.repair import RepairResult
from agents.insight import InsightResult
from utils.intelligence_engine import IntelligenceEngine

logger = logging.getLogger(__name__)

# Columns that should NEVER be forecasted — identifiers, codes, serials
FORECAST_EXCLUDE = [
    'id', 'cust_id', 'order_id', 'customer_id', 'index',
    'postal_code', 'zip', 'pincode', 'phone', 'mobile',
    'sr_no', 'serial', 'row_number', 'unnamed',
    'utc', 'timestamp', 'epoch', 'unix', 'created_at', 'updated_at',
    'modified_at', 'created', 'updated', 'modified'
]


@dataclass
class ForecastResult:
    is_time_series: bool = False
    primary_date_col: str = ""
    period_type: str = "D"  # D=Daily, W=Weekly, M=Monthly, Y=Yearly

    # Structure: {metric_name: {dates: [], values: [], lower: [], upper: [], r2: float, confidence: str, model_type: str, ...}}
    forecasts: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Simulation Defaults (to help UI)
    available_metrics: List[str] = field(default_factory=list)
    audit_log: Dict[str, Any] = field(default_factory=dict)

    # User-facing warnings surfaced from model fallbacks or data issues
    warnings: List[str] = field(default_factory=list)

    def get_forecast(self, metric: str) -> Optional[Dict[str, Any]]:
        return self.forecasts.get(metric)


class ForecastAgent(BaseAgent):
    """
    ForecastEngine 3.0 (Adaptive): Automatically selects between Linear 
    and Holt-Winters models based on seasonality and data sufficiency.
    """

    name = "ForecastAgent"

    def __init__(self):
        super().__init__()
        self.intelligence_engine = IntelligenceEngine()

    def _execute(self, input_data: Dict[str, Any]) -> ForecastResult:
        """
        Main execution point.
        Expects input_data = {"repair": RepairResult, "insight": InsightResult}
        """
        repair_result: RepairResult = input_data["repair"]
        
        try:
            df = repair_result.dataframe.copy()
            types = repair_result.detected_types
            
            # 1. Detect Time-Series Suitability
            date_col, period = self._detect_time_series(df, types)
            
            if not date_col:
                self._log("No suitable time-series structure detected.")
                return ForecastResult(is_time_series=False)
                
            self._log(f"Detected time-series on '{date_col}' (Freq: {period})")

            # 2. Identify Metrics to Forecast — business columns only
            num_cols = [c for c, t in types.items() if t == "numeric"]
            # Filter out timestamp columns — same logic as InsightAgent
            _ts_keywords = ['utc', 'timestamp', '_at', '_date', '_time',
                            'epoch', 'unix', 'created', 'updated', 'modified']
            num_cols = [
                c for c in num_cols
                if not any(kw in c.lower() for kw in _ts_keywords)
            ]
            forecastable = [
                c for c in num_cols
                if self._is_forecastable_column(c, df[c])
            ]
            # Cap at top 5 most relevant columns
            forecastable = forecastable[:5]

            # 3. Generate Base Forecasts
            forecasts = {}
            hw_warnings: List[str] = []
            for metric in forecastable:
                f_data = self._generate_forecast_v2(df, date_col, metric, period, hw_warnings)
                if f_data:
                    forecasts[metric] = f_data

            self._log(f"Generated ForecastEngine 3.0 results for {len(forecasts)} metrics.")

            audit_log = {}
            for metric, f_data in forecasts.items():
                model_type = f_data.get("forecast_model_type", "Linear")
                r2 = f_data.get("r2", 0.0)
                conf = f_data.get("confidence_level", "LOW")

                if model_type == "Holt-Winters":
                    formula = "Holt-Winters Triple Exponential Smoothing (statsmodels.tsa.holtwinters.ExponentialSmoothing)"
                    threshold = f"Seasonal cycle of {f_data.get('dominant_lag', 0)} periods detected; residual variance based confidence bounds"
                else:
                    formula = "Ordinary Least Squares (OLS) Linear Regression (scipy.stats.linregress)"
                    threshold = "Trend-line extrapolated over future index; standard deviation based rolling bounds"

                audit_log[metric] = {
                    "columns_used": [metric, date_col],
                    "formula": formula,
                    "threshold": threshold,
                    "method": "deterministic",
                    "result": f"Forecast generated successfully with model '{model_type}' (R² = {r2:.4f}, Confidence = {conf})."
                }

            result = ForecastResult(
                is_time_series=True,
                primary_date_col=date_col,
                period_type=period,
                forecasts=forecasts,
                available_metrics=list(forecasts.keys()),
                audit_log=audit_log,
                warnings=hw_warnings,
            )

            return result

        except Exception as e:
            self.log.errors.append(str(e))
            logger.error(f"ForecastAgent failed: {e}", exc_info=True)
            return ForecastResult(is_time_series=False)

    def _is_forecastable_column(self, col_name: str, series: pd.Series) -> bool:
        """
        Returns True only if a numeric column represents a real business metric
        worth forecasting.  Rejects identifier-like, high-cardinality, and
        near-zero-variance columns.
        """
        col_lower = col_name.lower().replace(' ', '_').replace('-', '_')

        # 1. Exclude columns whose name matches any exclusion pattern
        if any(excl in col_lower for excl in FORECAST_EXCLUDE):
            logger.debug("Forecast skip (exclude pattern): %s", col_name)
            return False

        # 2. Exclude sequential-integer pseudo-IDs.
        # Strategy: a column is an ID when it has many unique integer values whose
        # range is close to their count (i.e. they look like auto-increment keys).
        # Genuine business metrics (Amount, Revenue) have a wide value range
        # relative to their count, so they pass safely.
        nuniq = series.nunique()
        n = len(series)
        if nuniq > 0.9 * n:
            # Only reject if the values look sequential (range ≈ count)
            try:
                numeric_series = pd.to_numeric(series, errors='coerce').dropna()
                value_range = float(numeric_series.max() - numeric_series.min())
                if value_range > 0 and (value_range / nuniq) < 10:
                    # Values are tightly packed integers → looks like an ID sequence
                    logger.debug("Forecast skip (sequential-integer ID): %s", col_name)
                    return False
            except Exception:
                pass

        # 3. Exclude near-constant columns (std < 0.001)
        if series.std() < 0.001:
            logger.debug("Forecast skip (near-zero variance): %s", col_name)
            return False

        return True

    def _detect_time_series(self, df: pd.DataFrame, types: Dict[str, str]) -> Tuple[str, str]:
        date_cols = [c for c, t in types.items() if t == "datetime"]
        if not date_cols:
            return "", ""
            
        best_col = max(date_cols, key=lambda c: df[c].nunique())
        
        try:
            dt_series = pd.to_datetime(df[best_col]).sort_values()
            diff = dt_series.diff().dropna()
            if diff.empty:
                return "", ""
                
            median_diff = diff.median()
            
            if median_diff <= pd.Timedelta(days=1):
                return best_col, "D"
            elif median_diff <= pd.Timedelta(days=7):
                return best_col, "W"
            elif median_diff <= pd.Timedelta(days=31):
                return best_col, "M"
            else:
                return best_col, "Y"
        except:
            return best_col, "D"

    def _generate_forecast_v2(self, df: pd.DataFrame, date_col: str, metric: str, period: str, hw_warnings: List[str] = None, periods: int = None) -> Optional[Dict]:
        """
        ForecastEngine 3.0 Adaptive Implementation.
        Periods are determined dynamically by frequency if not explicitly passed.
        """
        # Dynamic period selection based on data frequency
        if periods is None:
            _PERIOD_MAP = {"D": 30, "W": 12, "M": 12, "Y": 5}
            periods = _PERIOD_MAP.get(period, 10)

        if hw_warnings is None:
            hw_warnings = []
        ts_df = df[[date_col, metric]].dropna().sort_values(by=date_col)
        
        # Validation Rule: N < 8 suppressed
        if len(ts_df) < 8:
            return None
            
        try:
            ts_df = ts_df.set_index(date_col)
            if not isinstance(ts_df.index, pd.DatetimeIndex):
                ts_df.index = pd.to_datetime(ts_df.index, errors="coerce")
            
            # Drop rows that failed to parse as dates (e.g. if column contains mixed text)
            ts_df = ts_df[ts_df.index.notnull()].dropna()
            
            if len(ts_df) < 8:
                return None
            
            # Infer whether this is a volume metric (sum) or a rate/average metric (mean)
            _vol_keywords = {"sales", "revenue", "count", "qty", "quantity", "total", "volume", "orders"}
            _is_volume = any(kw in metric.lower() for kw in _vol_keywords)
            agg_fn = "sum" if _is_volume else "mean"
            ts_resampled = ts_df.resample(period).agg(agg_fn)
            ts_resampled = ts_resampled.interpolate(method='linear')
            
            y_raw = ts_resampled[metric].values
            
            # ── Preprocessing ──────────────────────
            window = max(3, len(y_raw) // 10)
            y_smoothed = pd.Series(y_raw).rolling(window=window, min_periods=1, center=True).mean().values
            
            rolling_std = pd.Series(y_raw).rolling(window=window, min_periods=1).std()
            volatility_index = float(rolling_std.mean() / (y_raw.mean() + 1e-9))
            
            # ── Seasonality Detection ───────────────────────────────────────
            seasonality_detected = False
            dominant_lag = 0
            
            if len(y_raw) > 12:
                lags = range(1, min(13, len(y_raw) // 2))
                autocorr = [pd.Series(y_raw).autocorr(lag=l) for l in lags]
                max_corr = 0
                for i, corr in enumerate(autocorr):
                    if abs(corr) > 0.5 and abs(corr) > max_corr:
                        max_corr = abs(corr)
                        dominant_lag = i + 1
                        seasonality_detected = True

            # ── Adaptive Model Selection ──────────────────────────────────────
            model_type = "Linear"
            y_future = None
            ci = None
            r2 = 0.0
            p_value = 1.0
            slope = 0.0
            
            # Holt-Winters Switch: Requires seasonality AND 2x seasonal periods sufficiency
            if HAS_STATSMODELS and seasonality_detected and dominant_lag >= 2 and len(y_raw) >= 2 * dominant_lag:
                try:
                    model = ExponentialSmoothing(
                        y_raw,
                        trend="add",
                        seasonal="add",
                        seasonal_periods=dominant_lag
                    )
                    model_fit = model.fit()
                    y_future = model_fit.forecast(periods)
                    model_type = "Holt-Winters"
                    
                    # Compute residual variance for confidence bands
                    residuals = y_raw - model_fit.fittedvalues
                    res_std = np.std(residuals)
                    ci = 1.96 * res_std * np.sqrt(np.arange(1, periods + 1))
                    
                    # Estimate R2 for compatibility
                    ss_res = np.sum(residuals**2)
                    ss_tot = np.sum((y_raw - np.mean(y_raw))**2)
                    r2 = float(1 - (ss_res / (ss_tot + 1e-9)))
                    
                    # Determine trend direction from forecasted trend
                    hw_slope = y_future[-1] - y_future[0]
                    direction = "increasing" if hw_slope > 0.0001 else "decreasing" if hw_slope < -0.0001 else "stable"
                    
                except Exception as hw_err:
                    hw_warnings.append(
                        f"Holt-Winters model failed for '{metric}' (possible data issue: {hw_err}). "
                        "Falling back to linear regression."
                    )
                    logger.warning(f"Holt-Winters failed for {metric}, falling back: {hw_err}")
                    model_type = "Linear"

            # Linear Fallback / Default
            if model_type == "Linear":
                x = np.arange(len(y_raw))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y_smoothed)
                r2 = r_value**2
                x_future = np.arange(len(y_raw), len(y_raw) + periods)
                y_future = slope * x_future + intercept
                y_std = np.std(y_raw)
                ci = 1.96 * y_std * np.sqrt(1 + x_future/len(y_raw))
                # Define direction from slope so it is always set in the Linear path
                direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable"

            # ── Confidence Logic ───────────────────────────────────
            confidence_level = "LOW"
            if r2 > 0.6 and volatility_index < 0.3:
                confidence_level = "HIGH"
            elif r2 > 0.3:
                confidence_level = "MEDIUM"
            
            # HW Confidence Boost: If HW used and residual variance is low relative to mean
            if model_type == "Holt-Winters":
                res_std = np.std(y_raw - model_fit.fittedvalues)
                if res_std / (np.mean(y_raw) + 1e-9) < 0.15:
                    # Tier upgrade
                    if confidence_level == "MEDIUM": confidence_level = "HIGH"
                    elif confidence_level == "LOW": confidence_level = "MEDIUM"

            if r2 < 0.1:
                confidence_level = "LOW"

            # ── Strategic Interpretation (IntelligenceEngine) ────────────────
            context = {
                "trend_direction": direction,
                "confidence_level": confidence_level.capitalize(), # Normalize to Title Case
                "volatility_index": round(volatility_index, 3),
                "seasonality_detected": seasonality_detected,
                "forecast_model_type": model_type
            }
            
            narrative = self.intelligence_engine.generate_strategic_summary(context)
            
            interpretation = {
                "trend_direction": direction,
                "confidence": confidence_level,
                "model_type": model_type,
                "volatility_comment": "High variance detected." if volatility_index > 0.4 else "Stable pattern.",
                "seasonality_comment": f"Seasonal cycle ({dominant_lag} periods) detected." if seasonality_detected else "No significant seasonality.",
                "business_summary": narrative["executive_summary"],
                "primary_risk": narrative["primary_risk"],
                "primary_opportunity": narrative["primary_opportunity"],
                "confidence_comment": narrative["confidence_comment"]
            }

            # ── Assembly ─────────────────────────────────────────────
            last_date = ts_resampled.index[-1]
            freq_offset = pd.tseries.frequencies.to_offset(period)
            future_dates = [last_date + (i * freq_offset) for i in range(1, periods + 1)]
            
            return {
                "dates_hist": ts_resampled.index.strftime('%Y-%m-%d').tolist(),
                "values_hist": y_raw.tolist(),
                "values_smoothed": y_smoothed.tolist(),
                "dates_forecast": [d.strftime('%Y-%m-%d') for d in future_dates],
                "values_forecast": y_future.tolist(),
                "lower_bound": (y_future - ci).tolist(),
                "upper_bound": (y_future + ci).tolist(),
                "slope": float(slope),
                "r2": float(r2),
                "volatility_index": volatility_index,
                "seasonality_detected": seasonality_detected,
                "dominant_lag": dominant_lag,
                "confidence_level": confidence_level,
                "forecast_model_type": model_type,
                "interpretation": interpretation
            }

            
        except Exception as e:
            logger.warning(f"Adaptive Forecast failed for {metric}: {e}")
            return None

    def forecast_multivariate(self, df: pd.DataFrame, columns: List[str], periods: int = 10) -> Optional[Dict[str, Dict[str, Any]]]:
        """
        Generates joint multivariate forecasts using Vector Autoregression (VAR).
        """
        from statsmodels.tsa.vector_ar.var_model import VAR
        
        # Keep only the columns selected and drop rows with NaNs
        sub_df = df[columns].dropna()
        if len(sub_df) < 30:
            logger.warning("Multivariate VAR forecast requires at least 30 data points.")
            return None
            
        try:
            # Detect lag order automatically or cap at 5
            maxlags = min(5, len(sub_df) // (len(columns) + 1))
            if maxlags < 1:
                maxlags = 1
                
            model = VAR(sub_df)
            fitted = model.fit(maxlags=maxlags, ic='aic')
            
            # Forecast with interval
            lag_order = fitted.k_ar
            last_values = sub_df.values[-lag_order:]
            
            # forecast_interval returns (forecast, lower, upper)
            fc, lower, upper = fitted.forecast_interval(last_values, steps=periods)
            
            # Calculate R2 manually or use getattr
            rsquared_vals = getattr(fitted, "rsquared", None)
            results = {}
            for idx, col in enumerate(columns):
                # Calculate simple R-squared if not available
                r2_val = 0.0
                if rsquared_vals is not None:
                    if isinstance(rsquared_vals, dict):
                        r2_val = float(rsquared_vals.get(col, 0.0))
                    elif isinstance(rsquared_vals, (list, np.ndarray)):
                        r2_val = float(rsquared_vals[idx])
                results[col] = {
                    "forecast": fc[:, idx].tolist(),
                    "lower": lower[:, idx].tolist(),
                    "upper": upper[:, idx].tolist(),
                    "history": sub_df[col].values.tolist(),
                    "r2": r2_val,
                    "k_ar": lag_order
                }
            return results
        except Exception as e:
            logger.error(f"Multivariate VAR forecast failed: {e}", exc_info=True)
            return None

    def simulate_scenario(self, df: pd.DataFrame, metric: str, driver_factor: float) -> float:
        if metric not in df.columns:
            return 0.0
        return df[metric].sum() * driver_factor
