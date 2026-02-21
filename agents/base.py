"""
BaseAgent — abstract contract that all agents must implement.

Every agent receives a typed input, performs its task, and returns
a typed output.  Logging, timing, and error wrapping are handled
by the base class so agents only need to implement `_execute`.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class AgentLog:
    """Structured log entry produced by every agent run."""
    agent_name: str = ""
    status: str = "pending"          # pending | running | success | error
    duration_seconds: float = 0.0
    messages: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Abstract base for every agent in the pipeline."""

    name: str = "BaseAgent"

    def __init__(self) -> None:
        self.log = AgentLog(agent_name=self.name)

    # ── public entry point ───────────────────────────────────────────
    def run(self, input_data: Any) -> Any:
        """Execute the agent with timing, logging, and error handling."""
        self.log = AgentLog(agent_name=self.name, status="running")
        start = time.perf_counter()
        try:
            self._log(f"{self.name} started")
            result = self._execute(input_data)
            self.log.status = "success"
            self._log(f"{self.name} completed successfully")
            return result
        except Exception as exc:
            self.log.status = "error"
            self.log.errors.append(str(exc))
            logger.exception("[%s] failed", self.name)
            raise
        finally:
            self.log.duration_seconds = round(time.perf_counter() - start, 3)

    # ── subclasses implement this ────────────────────────────────────
    @abstractmethod
    def _execute(self, input_data: Any) -> Any:
        """Core logic — must be overridden by each concrete agent."""
        ...

    # ── helpers ──────────────────────────────────────────────────────
    def _log(self, message: str) -> None:
        self.log.messages.append(message)
        logger.info("[%s] %s", self.name, message)
