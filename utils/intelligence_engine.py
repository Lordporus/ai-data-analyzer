"""
IntelligenceEngine — Centralized Strategic Narrative Generation.
Implements an LLM-optional design with deterministic fallbacks.
"""

import logging
import os
from typing import Dict, Any, Optional
from utils.llm import LLMClient

logger = logging.getLogger(__name__)

class IntelligenceEngine:
    """
    Centralized engine for strategic narrative generation.
    Operates in 'deterministic' mode by default, upgrading to 'llm' mode if configured.
    """

    def __init__(self):
        # Reuse existing LLM wrapper (do NOT duplicate provider logic)
        self.llm_client = LLMClient()
        self.provider = self.llm_client.provider
        self.api_key = self.llm_client.api_key

        if not self.api_key or self.provider in ["none", ""]:
            self.mode = "deterministic"
            self.llm = None
        else:
            self.mode = "llm"
            self.llm = self.llm_client

    def generate_strategic_summary(self, context: dict) -> dict:
        """
        Primary interface for generating strategic narratives.
        Automatically falls back to deterministic logic on failure or if LLM is disabled.
        """
        if self.mode == "llm":
            try:
                result = self._generate_with_llm(context)
                if result and isinstance(result, dict) and "executive_summary" in result:
                    return result
                # If LLM returns None or invalid structure, use fallback
                return self._generate_deterministic(context)
            except Exception as e:
                logger.error(f"IntelligenceEngine LLM mode failed: {e}")
                return self._generate_deterministic(context)
        else:
            return self._generate_deterministic(context)

    def _generate_with_llm(self, context: dict) -> Optional[dict]:
        """Builds structured prompt and calls LLM for strategic analysis."""
        system_prompt = (
            "You are a Senior Data Strategist. Analyze the provided data context and return a JSON object "
            "containing strategic insights. Focus on business value, risk, and opportunity. "
            "You must return ONLY the JSON object."
            "\n\nExpected JSON Structure:\n"
            "{\n"
            "  'executive_summary': '...',\n"
            "  'primary_risk': '...',\n"
            "  'primary_opportunity': '...',\n"
            "  'confidence_comment': '...'\n"
            "}"
        )
        
        user_prompt = f"Analyze this business data context: {context}. Return the strategic summary JSON."
        
        return self.llm_client.generate_json(system_prompt, user_prompt)

    def _generate_deterministic(self, context: dict) -> dict:
        """
        Rule-based narrative generation. 
        Requires context fields: trend_direction, confidence_level, volatility_index, 
        seasonality_detected, forecast_model_type.
        """
        trend = context.get("trend_direction", "Stable")
        conf = context.get("confidence_level", "Medium")
        vol = context.get("volatility_index", 0.0)
        seasonal = context.get("seasonality_detected", False)
        model = context.get("forecast_model_type", "Linear")

        # 1. Executive Summary Logic
        if trend == "Upward" and conf == "High":
            exec_sum = "Performance shows stable upward growth supported by strong statistical confidence."
        elif trend == "Downward":
            exec_sum = "Observed downward trajectory signals potential performance deterioration."
        elif trend == "Upward":
            exec_sum = "Positive growth signals detected, though variance suggests monitoring."
        else:
            exec_sum = "Metric performance remains stable across the analyzed period."

        # 2. Confidence Comment Logic (Fully Deterministic)
        if conf == "Low":
            conf_comm = "Forecast reliability remains limited; strategic decisions should be conservative."
        elif conf == "High" and model == "Holt-Winters":
            conf_comm = "High-precision seasonal modeling indicates strong predictive reliability."
        else:
            conf_comm = "Statistical confidence is within acceptable parameters for standard planning."

        # 3. Risk Logic
        if vol > 0.3:
            risk = "High volatility introduces significant operational variability risk."
        elif trend == "Downward":
            risk = "Downward trajectory indicates risk of sustained performance decline."
        else:
            risk = "No immediate high-priority risks detected in current trend data."

        # 4. Opportunity Logic
        if trend == "Upward" and seasonal:
            opp = "Recurring seasonal peaks provide a high-leverage opportunity for growth."
        elif trend == "Upward":
            opp = "Sustained upward momentum suggests opportunity for strategic resource allocation."
        else:
            opp = "Focus on baseline stability and efficiency optimization."

        return {
            "executive_summary": exec_sum,
            "primary_risk": risk,
            "primary_opportunity": opp,
            "confidence_comment": conf_comm
        }
