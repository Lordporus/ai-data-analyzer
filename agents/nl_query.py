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
        if self.intelligence_engine.mode == "llm":
            try:
                schema_summary = self._get_schema_summary(df)
                system_prompt = self._build_system_prompt()
                user_prompt = self._build_user_prompt(query, schema_summary)
                
                # Use IntelligenceEngine's client if available
                response = self.intelligence_engine.llm.generate_json(system_prompt, user_prompt)
                
                if response:
                    return NLQueryResult(
                        explanation=response.get("explanation", "Analysis complete."),
                        chart_config=response.get("chart_config"),
                        confidence_level="LLM"
                    )
            except Exception as e:
                logger.warning(f"LLM interpretation failed, falling back: {e}")

        # ── Deterministic Fallback ──────────────────────────────────
        return self._execute_deterministic(query, df)

    def _execute_deterministic(self, query: str, df: pd.DataFrame) -> NLQueryResult:
        """Rule-based interpretation of user query."""
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
        
        # Heuristic: Pick columns with most variance/data
        best_num = num_cols[0] if num_cols else None
        if len(num_cols) > 1:
            best_num = df[num_cols].std().idxmax()
            
        best_cat = cat_cols[0] if cat_cols else None
        
        chart_type = "bar"
        x_col = best_cat or (date_cols[0] if date_cols else None)
        y_col = best_num
        agg = "mean"
        
        # Keyword Mapping
        if any(w in query for w in ["trend", "over time", "history", "series"]):
            chart_type = "line"
            x_col = date_cols[0] if date_cols else (x_col)
            agg = "sum"
        elif any(w in query for w in ["dist", "spread", "histogram", "range"]):
            chart_type = "histogram"
            x_col = best_num
            y_col = None
            agg = "none"
        elif any(w in query for w in ["corr", "relat", "vs", "against"]):
            chart_type = "scatter"
            x_col = num_cols[0] if num_cols else None
            y_col = num_cols[1] if len(num_cols) > 1 else num_cols[0]
            agg = "none"
        elif any(w in query for w in ["by", "per", "group", "breakdown"]):
            chart_type = "bar"
            # Try to find which categorical column is mentioned
            for c in cat_cols:
                if c.lower() in query:
                    x_col = c
                    break
            agg = "sum"

        # Safe Column Check
        if not x_col and num_cols: x_col = num_cols[0]
        
        chart_config = {
            "type": chart_type,
            "x": x_col,
            "y": y_col,
            "agg": agg,
            "title": f"Deterministic Analysis: {query.capitalize()}"
        }

        return NLQueryResult(
            explanation=f"Interpreted query '{query}' using rule-based logic (AI is disabled or unavailable).",
            chart_config=chart_config,
            confidence_level="Deterministic"
        )

    def _get_schema_summary(self, df: pd.DataFrame) -> str:
        """Compact schema representation for the prompt."""
        summary = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            unique_count = df[col].nunique()
            example_vals = df[col].dropna().unique()[:3].tolist()
            summary.append(f"- {col} ({dtype}): {unique_count} unique. Ex: {example_vals}")
        return "\n".join(summary)

    def _build_system_prompt(self) -> str:
        return """You are an expert Data Analyst Agent. 
Your goal is to answer the user's question based STRICTLY on the provided dataset schema.

Output JSON format:
{
    "explanation": "Concise executive summary answering the question.",
    "chart_config": {
        "type": "bar|line|scatter|pie|box|histogram",
        "x": "column_name",
        "y": "column_name_or_list",
        "agg": "sum|mean|count|none",
        "title": "Chart Title"
    }
}

Rules:
1. "chart_config" is optional. Return 'null' if no chart is needed.
2. If the user asks for a trend, use 'line' chart and ensure 'x' is a date/time column.
3. If the user asks for ranking/comparison, use 'bar' chart.
4. If the user asks for relationship, use 'scatter'.
5. Use "agg": "sum" or "mean" for numeric metrics grouped by categorical dimensions.
6. If the query is unrelated to the data, explain politely that you can only analyze this dataset.
"""

    def _build_user_prompt(self, query: str, schema: str) -> str:
        return f"""
Dataset Schema:
{schema}

User Question: "{query}"

Analyze the question and provide the JSON response.
"""
