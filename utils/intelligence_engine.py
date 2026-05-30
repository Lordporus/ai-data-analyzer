"""
IntelligenceEngine — Centralized Strategic Narrative Generation.
Implements an LLM-optional design with deterministic fallbacks.
"""

import logging
import os
import re
from typing import Dict, Any, List, Optional
from utils.llm import LLMClient

logger = logging.getLogger(__name__)

# ── Jargon control (mirrors agents/insight.py — kept in sync manually) ────────
JARGON_REPLACEMENTS: List[tuple] = [
    ("systemic volatility",          "sales variation"),
    ("variance drivers",             "key factors"),
    ("performance sensitivities",    "performance factors"),
    ("strategic leverage",           "growth opportunity"),
    ("operational throughput",       "business performance"),
    ("unmodeled volatility",         "unexpected changes"),
    ("downside exposure",            "risk"),
    ("metric stabilization",         "improving consistency"),
    ("risk vectors",                 "risk areas"),
    ("modeled variances",            "key differences"),
    ("structural drift",             "gradual decline"),
    ("organizational throughput",    "business output"),
    ("performance drift",            "performance change"),
    ("directional persistency",      "consistent trend"),
    ("lateral consistency",          "stable performance"),
    ("operational variability risk", "operational risk"),
    ("structural contraction",       "decline"),
    ("progressive expansion",        "growth"),
    # Generic placeholder terms
    ('"index"',                      '"data"'),
    ("'index'",                      "'data'"),
    ("the index",                    "the data"),
    ("Index is",                     "Data shows"),
    ("Executive Context",            "Business Analytics"),
]

# ── Sector keyword map for auto-detection ─────────────────────────────────────
_SECTOR_KEYWORDS: List[tuple] = [
    # Format: (keywords_list, sector_label, match_type)
    # Name-level patterns (Tier 1 - Highest priority, conf = 0.95)
    (["reddit", "twitter", "comment", "tweet", "socialmedia", "instagram", "tiktok", "facebook"], "Social Media Analytics", "name"),
    (["tsla", "aapl", "stock", "market", "trading", "crypto", "bitcoin", "portfolio", "equity"], "Financial Markets", "name"),
    (["complaint", "feedback", "support", "ticket", "survey", "reviews"], "Consumer Services", "name"),
    (["sales", "ecom", "retail", "store", "commerce", "shopify", "transaction", "orders"], "Retail & Commerce", "name"),
    (["hospital", "patient", "clinical", "medical", "health", "covid", "disease"], "Healthcare", "name"),
    (["loan", "credit", "bank", "finance", "mortgage", "payment"], "Financial Services", "name"),
    (["employee", "attrition", "hr", "workforce", "salary", "hiring", "talent"], "HR & Workforce", "name"),
    (["shipment", "delivery", "logistics", "supplychain", "warehouse", "freight", "route"], "Logistics", "name"),

    # Column-level patterns (Tier 2 - Lower priority, conf = 0.85)
    (["subreddit", "upvote", "downvote", "flair", "karma", "post", "comment", "thread", "reply", "submission", "author"], "Social Media Analytics", "column"),
    (["ticker", "stock", "close", "open", "high", "low", "volume", "dividend", "yield"], "Financial Markets", "column"),
    (["complaint", "issue", "response", "grievance", "resolved", "sentiment"], "Consumer Services", "column"),
    (["revenue", "sales", "amount", "price", "discount", "quantity", "product", "category", "channel", "order", "shipment"], "Retail & Commerce", "column"),
    (["patient", "diagnosis", "hospital", "treatment", "medication", "doctor", "admission"], "Healthcare", "column"),
    (["transaction", "balance", "loan", "mortgage", "credit", "debit", "account", "interest"], "Financial Services", "column"),
    (["employee", "attrition", "salary", "department", "tenure", "performance", "job"], "HR & Workforce", "column"),
    (["shipment", "delivery", "carrier", "origin", "destination", "tracking", "warehouse", "shipping"], "Logistics", "column"),
]

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
            msg = f"IntelligenceEngine: Initialized LLM mode using provider '{self.provider}' and model '{self.llm_client.model}'"
            logger.info(msg)
            print(msg)

    # ── Jargon Filtering ─────────────────────────────────────────────
    def _remove_jargon(self, text: str) -> str:
        """Scrubs all JARGON_REPLACEMENTS from a text field (case-insensitive)."""
        if not text:
            return text
        for old, new in JARGON_REPLACEMENTS:
            text = re.sub(re.escape(old), new, text, flags=re.IGNORECASE)
        return text

    # ── Sector Auto-Detection ────────────────────────────────────────
    def _detect_sector(self, col_names: List[str], dataset_name: str = "") -> tuple[str, float]:
        """
        Infers the business sector from dataset name or column names.
        Returns a tuple of (sector_label, confidence_score).
        """
        # Stage 1a: Check dataset name first (Tier 1, conf = 0.95)
        if dataset_name:
            normalized_name = dataset_name.lower().replace(" ", "").replace("_", "").replace(".", "").replace("-", "")
            for keywords, label, match_type in _SECTOR_KEYWORDS:
                if match_type == "name":
                    if any(kw in normalized_name for kw in keywords):
                        return label, 0.95

        # Stage 1b: Check column names (Tier 2, conf = 0.85)
        lowered_cols = [c.lower().replace(" ", "").replace("_", "").replace(".", "").replace("-", "") for c in col_names]
        for keywords, label, match_type in _SECTOR_KEYWORDS:
            if match_type == "column":
                if any(kw in col for kw in keywords for col in lowered_cols):
                    return label, 0.85

        # Default: first non-trivial column name
        skip = {"index", "id", "unnamed", "row", "serial", "sr_no", "sl_no"}
        for col in col_names:
            col_clean = col.lower().replace(" ", "_").replace("-", "_")
            if col_clean not in skip:
                # Sanitize column name: replace dots/underscores with spaces, title case
                readable = col.replace(".", " ").replace("_", " ").title()
                return f"{readable} Analytics", 0.60
        return "General Analytics", 0.50

    def _detect_sector_llm(self, col_names: List[str], dataset_name: str = "", sample_row: Optional[dict] = None) -> tuple[str, float]:
        """
        Uses the LLM to classify the business sector of the dataset.
        Returns a tuple of (sector_label, confidence_score).
        """
        if not self.llm:
            return "General Analytics", 0.0

        system_prompt = (
            "You are an expert Data Classifier. Your job is to analyze dataset metadata and determine its business sector.\n"
            "Choose from one of these standard sectors if applicable:\n"
            "- Social Media Analytics\n"
            "- Financial Markets\n"
            "- Consumer Services\n"
            "- Retail & Commerce\n"
            "- Healthcare\n"
            "- Financial Services\n"
            "- HR & Workforce\n"
            "- Logistics\n"
            "If none of these fit, generate a short, clean, descriptive sector name (max 3 words, ending with 'Analytics' or 'Services').\n\n"
            "Respond ONLY with a JSON object matching this schema:\n"
            "{\n"
            '  "sector": "<sector_label>"\n'
            "}"
        )

        user_prompt = f"Dataset Name: {dataset_name or 'Unknown'}\n"
        user_prompt += f"Columns: {col_names}\n"
        if sample_row:
            user_prompt += f"Sample Row: {sample_row}\n"
        user_prompt += "Determine the best business sector for this dataset."

        try:
            res = self.llm.generate_json(system_prompt, user_prompt)
            if res and isinstance(res, dict) and "sector" in res:
                sector = res["sector"].strip()
                if sector:
                    logger.info(f"LLM sector detection success: {sector}")
                    return sector, 0.90
            logger.warning("LLM sector detection returned invalid or empty JSON.")
        except Exception as e:
            logger.warning(f"LLM sector detection failed: {e}", exc_info=True)

        return "General Analytics", 0.0

    def detect_sector_hybrid(self, col_names: List[str], dataset_name: str = "", sample_row: Optional[dict] = None) -> str:
        """
        Two-stage hybrid sector detection.
        Stage 1: Deterministic check with high priority on dataset name (conf >= 0.85).
        Stage 2: Fallback to LLM-based classification if confidence < 0.85.
        """
        logger.info(f"Sector detection initiated for dataset: '{dataset_name}'")
        
        # Stage 1: Deterministic
        sector, conf = self._detect_sector(col_names, dataset_name)
        logger.info(f"Stage 1 (Deterministic) detected sector: '{sector}' with confidence {conf:.2f}")

        # Stage 2: LLM Fallback
        if conf < 0.85 and self.mode == "llm":
            logger.info("Confidence < 0.85 and LLM is enabled. Escalating to Stage 2 (LLM classification)...")
            llm_sector, llm_conf = self._detect_sector_llm(col_names, dataset_name, sample_row)
            if llm_conf > 0.0:
                logger.info(f"Stage 2 (LLM) detected sector: '{llm_sector}' with confidence {llm_conf:.2f}")
                return self._remove_jargon(llm_sector)
            else:
                logger.warning("Stage 2 (LLM) failed. Falling back to Stage 1 result.")

        return self._remove_jargon(sector)

    def generate_strategic_summary(self, context: dict) -> dict:
        """
        Primary interface for generating strategic narratives.
        Automatically falls back to deterministic logic on failure or if LLM is disabled.
        """
        if self.mode == "llm":
            try:
                result = self._generate_with_llm(context)
                if result and isinstance(result, dict) and "executive_summary" in result:
                    if "sector_name" not in result:
                        dataset_name = context.get("dataset_name", "")
                        col_names = context.get("dataset_columns", [])
                        sample_rows = context.get("sample_rows", [])
                        sample_row = sample_rows[0] if sample_rows else None
                        result["sector_name"] = self.detect_sector_hybrid(col_names, dataset_name, sample_row)
                    return result
                # If LLM returns None or invalid structure, use fallback
                return self._generate_deterministic(context)
            except Exception as e:
                logger.error(f"IntelligenceEngine LLM mode failed: {e}")
                return self._generate_deterministic(context)
        else:
            return self._generate_deterministic(context)

    def _build_safe_prompt_context(self, context: dict) -> dict:
        """
        Compress context to stay under 1000 tokens for LLM input.
        Prioritizes most important signals, truncates the rest.
        """
        col_names   = context.get("dataset_columns", [])
        cat_top     = context.get("categorical_top_values", {})
        num_stats   = context.get("numeric_stats", {})
        sample_rows = context.get("sample_rows", [])

        # Cap columns to 15 most relevant
        safe_cols = col_names[:15]
        if len(col_names) > 15:
            safe_cols.append(f"... and {len(col_names) - 15} more columns")

        # Cap categories to top 3 columns, top 2 values each
        safe_cat = {}
        for col, vals in list(cat_top.items())[:3]:
            safe_cat[col] = vals[:2]

        # Cap numeric stats to top 3 columns only
        safe_num = dict(list(num_stats.items())[:3])

        # Cap sample rows to 1 row, max 5 keys per row
        safe_sample = []
        if sample_rows:
            row = sample_rows[0]
            keys = list(row.keys())[:5]
            safe_sample = [{k: row[k] for k in keys}]

        return {
            "safe_cols":   safe_cols,
            "safe_cat":    safe_cat,
            "safe_num":    safe_num,
            "safe_sample": safe_sample,
        }

    def _generate_with_llm(self, context: dict) -> Optional[dict]:
        """Builds structured prompt with actual dataset context and calls LLM."""
        # Compress context to safe token budget before building prompt
        safe = self._build_safe_prompt_context(context)
        col_names   = safe["safe_cols"]
        cat_top     = safe["safe_cat"]
        num_stats   = safe["safe_num"]
        sample_rows = safe["safe_sample"]
        row_count   = context.get("row_count", "unknown")
        blacklist   = context.get("jargon_blacklist", [])

        # Build compact data description (token-safe)
        cat_summary = "; ".join(
            f"{col}: top values = {', '.join(str(v) for v in vals)}"
            for col, vals in cat_top.items()
        ) or "(none)"

        num_summary = "; ".join(
            f"{col}: total={s.get('sum')}, avg={s.get('mean')}, "
            f"range {s.get('min')}–{s.get('max')}"
            for col, s in num_stats.items()
        ) or "(none)"

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
            f"Sample row: {sample_rows}\n"
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
                s3 = f"Review performance trends for underperforming segments in '{top_cat_col}' this week."
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

        # ── Sector name via auto-detection (never hardcoded) ──────────
        dataset_name = context.get("dataset_name", "")
        sample_rows = context.get("sample_rows", [])
        sample_row = sample_rows[0] if sample_rows else None
        sector = self.detect_sector_hybrid(col_names, dataset_name, sample_row)

        # ── Jargon filter on ALL text fields before returning ─────────
        return {
            "executive_summary":   self._remove_jargon(exec_sum),
            "primary_risk":        self._remove_jargon(risk),
            "primary_opportunity": self._remove_jargon(opp),
            "confidence_comment":  self._remove_jargon(conf_comm),
            "sector_name":         self._remove_jargon(sector),
        }
