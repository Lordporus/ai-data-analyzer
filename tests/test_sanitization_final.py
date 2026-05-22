
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.report import ReportAgent
from agents.insight import InsightResult, KPI, TrendInfo
from agents.forecast import ForecastResult

def test_sanitization():
    agent = ReportAgent()
    
    # 1. Dirty Data setup
    kpis = [
        KPI("Good", 100),
        KPI("NaN_Val", np.nan),
        KPI("Inf_Val", np.inf)
    ]
    
    trends = [
        TrendInfo("Stagnant", 0.00001, "stable"), # Should be removed
        TrendInfo("Growth", 0.5, "increasing"),   # Should stay
        TrendInfo("NaN_Trend", np.nan, "decreasing") # Slope replaced by 0, then removed
    ]
    
    corr_df = pd.DataFrame({
        "A": [1.0, 0.9999, 0.5],
        "B": [0.9999, 1.0, 0.4],
        "C": [0.5, 0.4, 1.0]
    }, index=["A", "B", "C"])
    
    insight = InsightResult(
        kpi_list=kpis,
        trend_summary=trends,
        correlation_matrix=corr_df
    )
    
    forecast = ForecastResult(
        forecasts={
            "FewPoints": {"values_hist": [1, 2, 3], "r2": 0.9}, # Removed (N=3 < 8)
            "LowR2": {"values_hist": [1]*10, "r2": 0.05}        # Removed (R2=0.05 < 0.1)
        }
    )
    
    # Execution
    s_insight, s_forecast = agent._sanitize_data(insight, forecast)
    
    # Verifications
    print("\n--- KPI Verification ---")
    for k in s_insight.kpi_list:
        print(f"{k.name}: {k.value} (Type: {type(k.value)})")
        if k.name in ["NaN_Val", "Inf_Val"]:
            assert k.value == "-"
        else:
            assert k.value == 100

    print("\n--- Trend Verification ---")
    trend_cols = [t.column for t in s_insight.trend_summary]
    print(f"Remaining trends: {trend_cols}")
    assert "Growth" in trend_cols
    assert "Stagnant" not in trend_cols
    assert "NaN_Trend" not in trend_cols

    print("\n--- Correlation Verification ---")
    pairs = agent._get_correlation_pairs(s_insight.correlation_matrix)
    print(f"Pairs found: {pairs}")
    for p in pairs:
        assert abs(p[2]) <= 0.999
        assert p[0] != p[1] # No identites

    print("\n--- Forecast Verification ---")
    print(f"Remaining forecasts: {list(s_forecast.forecasts.keys())}")
    assert "FewPoints" not in s_forecast.forecasts
    assert "LowR2" not in s_forecast.forecasts

    print("\nSanitization Logic: PASSED")

if __name__ == "__main__":
    test_sanitization()
