"""
NLQueryAgent — Ask natural language questions about the dataset.

Input:  User Query (str), Context (DataFrame summary)
Output: NLQueryResult (Explanation + Chart Config)

Key Capabilities:
1. Interpret user intent (Trend, Ranking, Distribution, etc.).
2. Map natural language to specific columns.
3. Determine optimization chart type and aggregation.
4. Return "safe" valid configuration for frontend rendering.
"""

from __future__ import annotations

import logging
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd
from agents.base import BaseAgent
from utils.intelligence_engine import IntelligenceEngine

logger = logging.getLogger(__name__)


@dataclass
class NLQueryResult:
    explanation: str = ""
    chart_config: Optional[Dict[str, Any]] = None
    confidence_level: str = "Deterministic" # "LLM" or "Deterministic"
    error: Optional[str] = None


class NLQueryAgent(BaseAgent):
    """
    Interprets natural language queries to generate insights and charts.
    """

    name = "NLQueryAgent"

    def __init__(self):
        super().__init__()
        self.intelligence_engine = IntelligenceEngine()

    def _execute(self, input_data: Dict[str, Any]) -> NLQueryResult:
        """
        Main execution point.
        """
        query = input_data.get("query", "").lower()
        df = input_data.get("df")
        
        if not query or df is None or df.empty:
            return NLQueryResult(error="Invalid input: Query or Data missing.")

        # ── LLM Mode ────────────────────────────────────────────────
        is_ai_enabled = self.intelligence_engine.mode == "llm" or (
            self.intelligence_engine.api_key and self.intelligence_engine.provider != "none"
        )
        if is_ai_enabled:
            if not self.intelligence_engine.llm:
                self.intelligence_engine.llm = self.intelligence_engine.llm_client
            try:
                schema_summary = self._get_schema_summary(df)
                system_prompt = self._build_system_prompt(self._detect_dataset_type(df))
                user_prompt = self._build_user_prompt(query, schema_summary)

                # Use IntelligenceEngine's client if available
                response = self.intelligence_engine.llm.generate_json(system_prompt, user_prompt)

                if response:
                    chart_config = response.get("chart_config")
                    # Validate that referenced columns exist in the dataframe
                    if chart_config:
                        x_col = chart_config.get("x")
                        y_col = chart_config.get("y")
                        df_cols = set(df.columns.tolist())
                        x_valid = x_col is None or x_col in df_cols
                        y_valid = y_col is None or (isinstance(y_col, list) and all(c in df_cols for c in y_col)) or (isinstance(y_col, str) and y_col in df_cols)
                        if not x_valid or not y_valid:
                            chart_config = None
                    return NLQueryResult(
                        explanation=response.get("explanation", "Analysis complete."),
                        chart_config=chart_config,
                        confidence_level="LLM"
                    )
            except Exception as e:
                logger.warning(f"LLM interpretation failed: {e}")
                return NLQueryResult(
                    error=(
                        "AI features are currently unavailable. "
                        "Please use the manual chart builder instead."
                    ),
                    confidence_level="Deterministic"
                )

        # ── AI disabled entirely — surface clear error instead of guessing ──
        return NLQueryResult(
            error=(
                "AI features are currently unavailable. "
                "Please use the manual chart builder instead."
            ),
            confidence_level="Deterministic"
        )


    def _detect_dataset_type(self, df: pd.DataFrame) -> str:
        """Detect dataset domain from column names for context-aware prompting."""
        col_lower = [c.lower() for c in df.columns]
        if any(kw in c for c in col_lower for kw in ['price', 'close', 'open', 'volume', 'revenue', 'profit']):
            return "Financial"
        if any(kw in c for c in col_lower for kw in ['subreddit', 'upvote', 'reddit', 'tweet', 'post', 'comment']):
            return "Social Media"
        if any(kw in c for c in col_lower for kw in ['order', 'transaction', 'invoice', 'payment']):
            return "Transactional"
        if any(kw in c for c in col_lower for kw in ['patient', 'diagnosis', 'hospital', 'drug', 'dose']):
            return "Healthcare"
        if any(kw in c for c in col_lower for kw in ['timestamp', 'utc', '_at', '_date', '_time', 'epoch', 'created', 'updated']):
            return "Time-Series"
        return "General"

    def _get_schema_summary(self, df: pd.DataFrame) -> str:
        """Compact schema representation for the prompt."""
        summary = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            unique_count = df[col].nunique()
            example_vals = df[col].dropna().unique()[:3].tolist()
            summary.append(f"- {col} ({dtype}): {unique_count} unique. Ex: {example_vals}")
        return "\n".join(summary)

    def _build_system_prompt(self, dataset_type: str = "General") -> str:
        return f"""You are an expert Data Analyst Agent specialized in {dataset_type} data.
Your goal is to answer the user's question based STRICTLY on the provided dataset schema.

Output ONLY valid JSON in this exact format:
{{
    "explanation": "Concise executive summary answering the question.",
    "chart_config": {{
        "type": "bar|line|scatter|pie|box|histogram",
        "x": "column_name",
        "y": "column_name_or_list",
        "agg": "sum|mean|count|none",
        "title": "Chart Title"
    }}
}}

Rules:
1. "chart_config" is optional. Return null if no chart is needed.
2. If the user asks for a trend, use 'line' chart and ensure 'x' is a date/time column.
3. If the user asks for ranking/comparison, use 'bar' chart.
4. If the user asks for relationship, use 'scatter'.
5. Use "agg": "sum" or "mean" for numeric metrics grouped by categorical dimensions.
6. If the query is unrelated to the data, explain politely that you can only analyze this dataset.
7. NEVER invent column names — only use columns listed in the schema.
8. NEVER return markdown, code blocks, or any text outside the JSON object.

--- FEW-SHOT EXAMPLES ---

Example 1 — Trend query:
User: "Show me score trends over time"
Schema includes: created_utc (datetime), score (int)
Response:
{{"explanation": "The line chart shows how scores changed over time using the created_utc column as the time axis.", "chart_config": {{"type": "line", "x": "created_utc", "y": "score", "agg": "mean", "title": "Average Score Over Time"}}}}

Example 2 — Ranking query:
User: "Which subreddit has the most posts?"
Schema includes: subreddit (str), id (str)
Response:
{{"explanation": "The bar chart ranks subreddits by post count.", "chart_config": {{"type": "bar", "x": "subreddit", "y": "id", "agg": "count", "title": "Post Count by Subreddit"}}}}

Example 3 — Relationship query:
User: "Is there a relationship between score and number of comments?"
Schema includes: score (int), num_comments (int)
Response:
{{"explanation": "The scatter plot reveals the correlation between post score and comment count.", "chart_config": {{"type": "scatter", "x": "score", "y": "num_comments", "agg": "none", "title": "Score vs Comments"}}}}

Example 4 — No chart needed:
User: "What is the average score?"
Schema includes: score (int)
Response:
{{"explanation": "The average score across all posts is calculated from the score column.", "chart_config": null}}

Example 5 — Financial query:
User: "Show closing price trend"
Schema includes: date (datetime), close (float)
Response:
{{"explanation": "The line chart tracks closing price movement over time.", "chart_config": {{"type": "line", "x": "date", "y": "close", "agg": "mean", "title": "Closing Price Trend"}}}}
--- END EXAMPLES ---
"""

    def _build_user_prompt(self, query: str, schema: str) -> str:
        return f"""
Dataset Schema:
{schema}

User Question: "{query}"

Analyze the question and provide the JSON response.
"""
