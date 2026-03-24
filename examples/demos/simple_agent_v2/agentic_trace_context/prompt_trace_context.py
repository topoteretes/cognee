from __future__ import annotations

from cognee.infrastructure.engine import DataPoint
from pydantic import Field
from typing import Any


class AgentContextTrace(DataPoint):
    """Minimal context payload for agentic tracing as a Cognee DataPoint."""

    task_query: str = ""
    trace_metadata: dict[str, Any] = Field(default_factory=dict)
