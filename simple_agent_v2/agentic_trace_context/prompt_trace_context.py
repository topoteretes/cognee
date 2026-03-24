from __future__ import annotations

from cognee.infrastructure.engine import DataPoint


class AgentContextTrace(DataPoint):
    """Minimal context payload for agentic tracing as a Cognee DataPoint."""

    origin_function: str = ""
    task_query: str = ""
