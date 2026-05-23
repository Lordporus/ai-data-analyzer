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
        """Builds structured prompt with actual dataset context and calls LLM."""
        # Pull rich dataset context injected by InsightAgent
        col_names   = context.get("dataset_columns", [])
        cat_top     = context.get("categorical_top_values", {})
        num_stats   = context.get("numeric_stats", {})
        row_count   = context.get("row_count", "unknown")
        sample_rows = context.get("sample_rows", [])
        blacklist   = context.get("jargon_blacklist", [])

        # Build a compact data description so the LLM has concrete grounding
        cat_summary = "; ".join(
            f"{col}: top values = {', '.join(vals[:3])}"
            for col, vals in list(cat_top.items())[:5]
        ) or "(none)"

        num_summary = "; ".join(
            f"{col}: total={s.get('sum')}, avg={s.get('mean')}, range {s.get('min')}–{s.get('max')}"
            for col, s in list(num_stats.items())[:4]
        ) or "(none)"

        # Embed the blacklist directly in the prompt so the LLM knows what is forbidden
        blacklist_str = ", ".join(f'"{w}"' for w in blacklist) if blacklist else "(none)"

        system_prompt = (
            "You are a Business Analyst writing a report for a non-technical business owner "
            "(e.g. a shop owner, a sales manager). Your writing must be plain, direct English.\n\n"

            "MANDATORY STRUCTURE for executive_summary (exactly 3 sentences):\n"
            "  Sentence 1: What is this dataset? (mention row count and main business area)\n"
            "  Sentence 2: The single most important finding — include a specific number, percentage, or total.\n"
            "  Sentence 3: One concrete action the business should take THIS WEEK.\n\n"

            "RULES — you will be penalised for breaking any of these:\n"
            "  1. Always use ACTUAL column/category names from the data provided below.\n"
            "  2. Never say 'index', 'metric', 'column', 'data point', or any abstract placeholder.\n"
            "  3. Give specific numbers: percentages, totals, counts.\n"
            f" 4. FORBIDDEN WORDS — never use any of: {blacklist_str}\n"
            "  5. No consulting jargon. No passive voice. Write as if explaining to a friend.\n\n"

            "Return ONLY this JSON (no markdown, no explanation):\n"
            "{\n"
            '  "executive_summary": "<3 sentences as above>",\n'
            '  "primary_risk": "<one specific risk naming the column and a number>",\n'
            '  "primary_opportunity": "<one specific opportunity naming the category and a number>",\n'
            '  "confidence_comment": "<one sentence on how reliable this data is>"\n'
            "}"
        )

        user_prompt = (
            f"Dataset: {row_count} rows, columns: {col_names}\n"
            f"Categories: {cat_summary}\n"
            f"Numbers: {num_summary}\n"
            f"Sample rows: {sample_rows[:3]}\n"
            f"Overall trend: {context.get('trend_direction', 'Stable')}, "
            f"confidence: {context.get('confidence_level', 'Medium')}\n"
            "Write the strategic summary JSON now."
        )

        return self.llm_client.generate_json(system_prompt, user_prompt)

    def _generate_deterministic(self, context: dict) -> dict:
        """
        Rule-based narrative generation using actual column/category names
        when available, falling back to generic phrasing otherwise.
        """
        trend    = context.get("trend_direction", "Stable")
        conf     = context.get("confidence_level", "Medium")
        vol      = context.get("volatility_index", 0.0)
        seasonal = context.get("seasonality_detected", False)
        model    = context.get("forecast_model_type", "Linear")

        # Pull rich context injected by InsightAgent (may be absent for forecast calls)
        row_count  = context.get("row_count", None)
        cat_top    = context.get("categorical_top_values", {})
        num_stats  = context.get("numeric_stats", {})
        col_names  = context.get("dataset_columns", [])

        # Pick the most prominent categorical column for naming
        top_cat_col = next(iter(cat_top), None)
        top_cat_val = cat_top[top_cat_col][0] if top_cat_col and cat_top[top_cat_col] else None

        # Pick the primary numeric column (first in stats dict)
        top_num_col = next(iter(num_stats), None)
        top_num_sum = num_stats[top_num_col]["sum"] if top_num_col else None

        # 1. Executive Summary — strict 3-sentence structure: what/finding/action
        if row_count and top_cat_col and top_cat_val and top_num_col and top_num_sum:
            def _fmt(n):
                if abs(n) >= 1_000_000: return f"{n/1_000_000:.2f}M"
                if abs(n) >= 1_000:     return f"{n/1_000:.1f}K"
                return f"{n:.2f}"
            top_second   = cat_top[top_cat_col][1] if len(cat_top[top_cat_col]) > 1 else None
            top_num_avg  = num_stats[top_num_col].get("mean", 0)
            # Sentence 1: what is the data
            s1 = f"This dataset contains {row_count:,} records covering {len(col_names)} business metrics."
            # Sentence 2: most important finding with a number
            s2 = (
                f"'{top_cat_val}' is the leading {top_cat_col}"
                + (f", followed by '{top_second}'" if top_second else "") +
                f"; total {top_num_col} across all records is {_fmt(top_num_sum)} "
                f"(average {_fmt(top_num_avg)} per record)."
            )
            # Sentence 3: one concrete action
            if trend == "Upward":
                s3 = f"Increase stock and marketing budget for '{top_cat_val}' this week to capitalise on the upward trend."
            elif trend == "Downward":
                s3 = f"Review pricing and promotions for underperforming segments in '{top_cat_col}' this week."
            else:
                s3 = f"Focus this week on growing the '{top_cat_val}' segment, which currently drives the most revenue."
            exec_sum = f"{s1} {s2} {s3}"
        elif trend == "Upward" and conf == "High":
            exec_sum = "Sales are growing steadily and the data strongly supports this trend. Focus this week on scaling top-performing channels to meet rising demand."
        elif trend == "Downward":
            exec_sum = "Sales are declining across tracked metrics. Investigate the top sales channels this week and identify what is causing the drop."
        elif trend == "Upward":
            exec_sum = "Sales show positive growth, though with some variation between segments. Monitor the top channels closely and plan capacity for the next 30 days."
        else:
            exec_sum = "Sales performance is stable with no major swings detected. Use this week to review your lowest-performing categories and set improvement targets."

        # 2. Confidence Comment
        if conf == "Low":
            conf_comm = "Forecast reliability remains limited; strategic decisions should be conservative."
        elif conf == "High" and model == "Holt-Winters":
            conf_comm = "High-precision seasonal modeling indicates strong predictive reliability."
        else:
            conf_comm = "Statistical confidence is within acceptable parameters for standard planning."

        # 3. Risk — use column names when available, no jargon
        if top_num_col and vol > 0.3:
            risk = f"'{top_num_col}' shows high variation between records — check for data entry errors or unusual spikes."
        elif vol > 0.3:
            risk = "Values vary widely across records — look for unusual spikes or data entry errors."
        elif trend == "Downward" and top_cat_col:
            risk = f"'{top_cat_col}' sales are declining — act before the drop becomes harder to reverse."
        elif trend == "Downward":
            risk = "Sales are trending downward — investigate root causes before the end of this month."
        else:
            risk = "No major risks detected right now — keep monitoring weekly to catch changes early."

        # 4. Opportunity — use column names when available
        if top_cat_col and top_cat_val and trend == "Upward" and seasonal:
            opp = f"Recurring seasonal peaks in '{top_cat_val}' ({top_cat_col}) provide a high-leverage opportunity for growth campaigns."
        elif top_cat_col and top_cat_val:
            opp = f"'{top_cat_val}' is the leading {top_cat_col} — concentrate marketing and inventory resources here for maximum ROI."
        elif trend == "Upward":
            opp = "Sustained upward momentum suggests opportunity for strategic resource allocation."
        else:
            opp = "Focus on baseline stability and efficiency optimization across top-performing segments."

        return {
            "executive_summary": exec_sum,
            "primary_risk": risk,
            "primary_opportunity": opp,
            "confidence_comment": conf_comm
        }
