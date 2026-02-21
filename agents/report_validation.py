"""
ReportValidationEngine — Quality Gate for Analytical Reporting.

This module acts as a strict filter between raw analysis (InsightAgent/ForecastAgent)
and the final presentation layer (ReportAgent). It ensures statistical rigor,
removes technical artifacts, and cleans data for executive consumption.
"""

import math
import re
import logging
from typing import List, Dict, Any, Optional, Set
import pandas as pd
import numpy as np
from dataclasses import replace

from agents.insight import InsightResult, TrendInfo, KPI
from agents.forecast import ForecastResult

logger = logging.getLogger(__name__)

class ReportValidationEngine:
    """
    Validates, sanitizes, and filters analytical results before reporting.
    """

    def __init__(self):
        # Configurable thresholds
        self.MIN_VARIANCE = 1e-9
        self.MIN_TREND_SLOPE = 0.0001
        self.MIN_TREND_R2 = 0.1  # Weak trend threshold
        self.DERIVED_CORR_THRESHOLD = 0.999
        self.MIN_FORECAST_POINTS = 8
        self.MIN_FORECAST_R2 = 0.3

        # Regex for technical/structural columns
        self.STRUCTURAL_PATTERNS = [
            r"^id$", r"^index$", r"^row_?id$", r"^guid$", r"^uuid$", 
            r"^key$", r"^_", r"unnamed", r"auto_increment"
        ]

    def validate_insight(self, insight: InsightResult) -> InsightResult:
        """
        Main entry point for sanitizing insight results.
        Returns a new, sanitized InsightResult object.
        """
        # 1. Sanitize KPIs
        valid_kpis = self._validate_kpis(insight.kpi_list)

        # 2. Sanitize Trends
        valid_trends = self._validate_trends(insight.trend_summary)

        # 3. Sanitize Correlations (Matrix)
        valid_corr = self._validate_correlations(insight.correlation_matrix)

        # 4. Filter Risks/Opps (remove empty/generic)
        valid_risks = [r for r in insight.top_risks if r]
        valid_opps = [o for o in insight.top_opportunities if o]

        # Return new object (shallow copy of others is fine)
        return replace(
            insight,
            kpi_list=valid_kpis,
            trend_summary=valid_trends,
            correlation_matrix=valid_corr,
            top_risks=valid_risks,
            top_opportunities=valid_opps
        )

    def validate_forecast(self, forecast: Optional[ForecastResult]) -> Optional[ForecastResult]:
        """
        Validates forecast results. Returns None if quality is insufficient.
        """
        if not forecast:
            return None

        valid_forecasts = {}
        for metric, data in forecast.forecasts.items():
            # Check 1: Is it a business column?
            if self._is_structural_column(metric):
                continue
            
            # Check 2: Sanity check values (no NaNs in forecast)
            if np.isnan(data.get("values_forecast", [])).any():
                logger.warning(f"Validation: Forecast for {metric} contains NaNs. Dropping.")
                continue

            # Check 3: Length check (already done in forecast agent, but double check)
            if len(data.get("values_hist", [])) < self.MIN_FORECAST_POINTS:
                continue

            valid_forecasts[metric] = data

        if not valid_forecasts:
            logger.info("Validation: No valid forecasts remained after filtering (Dataset may be too small or sparse).")
            return None
        
        return replace(forecast, forecasts=valid_forecasts)

    # ── KPI Validation ───────────────────────────────────────────────

    def _validate_kpis(self, kpis: List[KPI]) -> List[KPI]:
        sanitized = []
        seen_names = set()
        
        for k in kpis:
            # Filter 1: Structural Columns
            if self._is_structural_column(k.name):
                continue
            
            # Filter 2: Duplicates
            if k.name in seen_names:
                continue
            seen_names.add(k.name)

            # Filter 3: NaN/Inf values
            val = k.value
            if isinstance(val, (int, float)):
                if np.isnan(val) or np.isinf(val):
                    val = "N/A" # Or drop? Let's keep as N/A but strictly non-numeric
            
            # Update KPI
            k.value = val
            sanitized.append(k)
            
        return sanitized

    # ── Trend Validation ─────────────────────────────────────────────

    def _validate_trends(self, trends: List[TrendInfo]) -> List[TrendInfo]:
        valid = []
        for t in trends:
            # Filter 1: Structural
            if self._is_structural_column(t.column):
                continue
            
            # Filter 2: Significance
            # If slope is effectively zero, mark as stable or drop?
            # User wants: "No meaningful trend detected" if weak.
            # We will force direction to "stable" if weak, so it doesn't show up as a trend.
            
            if abs(t.slope) < self.MIN_TREND_SLOPE:
                t.direction = "stable"
            elif t.r_squared < self.MIN_TREND_R2:
                # Weak correlation, treat as noise/stable
                t.direction = "stable"
            
            # Filter 3: NaNs
            if np.isnan(t.slope) or np.isnan(t.p_value):
                continue
                
            valid.append(t)
        return valid

    # ── Correlation Validation ───────────────────────────────────────

    def _validate_correlations(self, matrix: pd.DataFrame) -> pd.DataFrame:
        if matrix is None or matrix.empty:
            return matrix

        # Drop structural columns from index/columns
        cols_to_drop = [c for c in matrix.columns if self._is_structural_column(c)]
        matrix = matrix.drop(index=cols_to_drop, columns=cols_to_drop, errors='ignore')

        # Detect and mask zero variance (should be NaN in corr matrix usually, but sometimes 0 or 1 artifact)
        # Verify: If we don't have original data here, we rely on corr matrix values.
        # Usually corr is between -1 and 1.
        
        # Detect Derived Columns (High Correlation Clique)
        # If A & B have corr > 0.999, we should probably hide one.
        # For the matrix, we can just mask them? 
        # Actually ReportAgent iterates pairs. We can leave it to ReportAgent logic IF report agent uses this Validator's output.
        # But `validate_insight` returns the matrix.
        
        # Let's replace NaNs with 0 (no correlation)
        matrix = matrix.fillna(0)
        
        return matrix

    # ── Helpers ──────────────────────────────────────────────────────

    def _is_structural_column(self, name: str) -> bool:
        """Check if column is an ID, index, or technical artifact."""
        name_lower = str(name).lower().strip()
        
        # 1. Exact list check
        if name_lower in ["id", "index", "row_id", "row_number", "guid", "uuid", "pk", "fk"]:
            return True
            
        # 2. Regex patterns
        for pattern in self.STRUCTURAL_PATTERNS:
            if re.search(pattern, name_lower):
                return True
                
        # 3. Suffix check
        if name_lower.endswith("_id") or name_lower.endswith("index"):
            return True
            
        return False

    def _sanitize_value(self, val: Any) -> Any:
        if isinstance(val, (float, np.floating)):
            if np.isnan(val) or np.isinf(val):
                return None
        return val
