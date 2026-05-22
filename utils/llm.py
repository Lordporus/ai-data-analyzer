"""
LLM Client Utility
Handles communication with Large Language Models (OpenAI, Gemini, Ollama)
for strategic reasoning and report generation.
"""

import json
import logging
import os
import requests
from typing import Dict, Any, Optional

from config.settings import (
    LLM_PROVIDER, LLM_API_KEY, LLM_MODEL, LLM_ENDPOINT
)

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Unified client for LLM interactions.
    Supports: 'openai', 'gemini' (via OpenAI compat or direct), 'ollama', 'none'.
    """

    def __init__(self):
        self.provider = LLM_PROVIDER.lower()
        self.api_key = LLM_API_KEY
        self.model = LLM_MODEL
        self.endpoint = LLM_ENDPOINT

    def generate_json(self, system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
        """
        Generates a JSON response from the LLM.
        Returns None if LLM is disabled or fails.
        """
        if self.provider == "none":
            return None

        try:
            if self.provider in ["openai", "gemini", "ollama"]:
                return self._call_openai_compatible(system_prompt, user_prompt)
            else:
                logger.warning(f"Unknown LLM provider: {self.provider}")
                return None
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return None

    def _call_openai_compatible(self, system_prompt: str, user_prompt: str) -> Optional[Dict[str, Any]]:
        """
        Generic handler for OpenAI-compatible APIs (including Ollama and Gemini via OpenAI adapter).
        Force JSON mode where possible.
        """
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.2, # Low temperature for analytical tasks
            "response_format": {"type": "json_object"}
        }

        # Ollama specific adjustment (doesn't always support response_format in all versions, but good to try)
        if self.provider == "ollama":
            # Ollama standard typically uses 'format': 'json' at top level
            del payload["response_format"]
            payload["format"] = "json"
            payload["stream"] = False

        try:
            response = requests.post(self.endpoint, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            
            content = ""
            if self.provider == "ollama":
                content = data.get("message", {}).get("content", "{}")
            else:
                content = data["choices"][0]["message"]["content"]
            
            # Clean Markdown fences if present
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            
            return json.loads(content.strip())

        except requests.exceptions.RequestException as e:
            logger.error(f"API Request Error ({self.endpoint}): {e}")
            if e.response:
                logger.error(f"Response: {e.response.text}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {e}")
            logger.debug(f"Raw content: {content}")
            return None
