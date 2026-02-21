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


@dataclass
class ForecastResult:
    is_time_series: bool = False
    primary_date_col: str = ""
    period_type: str = "D"  # D=Daily, W=Weekly, M=Monthly, Y=Yearly
    
    # Structure: {metric_name: {dates: [], values: [], lower: [], upper: [], r2: float, confidence: str, model_type: str, ...}}
    forecasts: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Simulation Defaults (to help UI)
    available_metrics: List[str] = field(default_factory=list)
    
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

            # 2. Identify Metrics to Forecast
            num_cols = [c for c, t in types.items() if t == "numeric"]
            metrics = [c for c in num_cols if df[c].nunique() > 5]
            
            # 3. Generate Base Forecasts
            forecasts = {}
            for metric in metrics[:5]: # Limit to top 5
                f_data = self._generate_forecast_v2(df, date_col, metric, period)
                if f_data:
                    forecasts[metric] = f_data
            
            self._log(f"Generated ForecastEngine 3.0 results for {len(forecasts)} metrics.")

            result = ForecastResult(
                is_time_series=True,
                primary_date_col=date_col,
                period_type=period,
                forecasts=forecasts,
                available_metrics=list(forecasts.keys())
            )
            
            return result

        except Exception as e:
            self.log.errors.append(str(e))
            logger.error(f"ForecastAgent failed: {e}", exc_info=True)
            return ForecastResult(is_time_series=False)

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

    def _generate_forecast_v2(self, df: pd.DataFrame, date_col: str, metric: str, period: str, periods=10) -> Optional[Dict]:
        """
        ForecastEngine 3.0 Adaptive Implementation
        """
        ts_df = df[[date_col, metric]].dropna().sort_values(by=date_col)
        
        # Validation Rule: N < 8 suppressed
        if len(ts_df) < 8:
            return None
            
        try:
            ts_df = ts_df.set_index(date_col)
            if not isinstance(ts_df.index, pd.DatetimeIndex):
                ts_df.index = pd.to_datetime(ts_df.index, errors="coerce")
            
            # Drop rows that failed to parse as dates (e.g. if column contains mixed text)
            ts_df = ts_df.dropna(subset=[ts_df.index.name]) if ts_df.index.name else ts_df.dropna()
            
            if len(ts_df) < 8:
                return None
            
            ts_resampled = ts_df.resample(period).sum() 
            if ts_resampled[metric].isna().mean() > 0.5:
                 ts_resampled = ts_df.resample(period).mean()
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
                    
                except Exception as hw_err:
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
                "dates_forecast": [d.strftime('%Y-%m-%d') for d in future_dates],
                "values_forecast": y_future.tolist(),
                "lower_bound": (y_future - ci).tolist(),
                "upper_bound": (y_future + ci).tolist(),
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

    def simulate_scenario(self, df: pd.DataFrame, metric: str, driver_factor: float) -> float:
        if metric not in df.columns:
            return 0.0
        return df[metric].sum() * driver_factor
