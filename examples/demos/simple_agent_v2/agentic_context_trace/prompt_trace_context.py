from __future__ import annotations

from typing import Any

from cognee.infrastructure.engine import DataPoint
from pydantic import Field


class AgentContextTrace(DataPoint):
    """Minimal context payload for agentic tracing as a Cognee DataPoint."""

    origin_function: str = ""
    with_memory: bool = False
    save_traces: bool = False
    task_query: str = ""
    memory_context: str = ""
    method_params: dict[str, Any] = Field(default_factory=dict)
    method_return_value: Any = None
    # JSON string of method_params + method_return_value; indexed for embedding/search.
    text: str = ""
    metadata: dict = {"index_fields": ["text"]}

    async def get_memory_context(self, query_text: str) -> None:
        if not self.with_memory:
            return

        from cognee import SearchType, search
        from cognee.modules.retrieval.exceptions.exceptions import NoDataError

        memory_query = self.task_query or query_text
        try:
            memory_results = await search(
                query_text=memory_query,
                query_type=SearchType.GRAPH_COMPLETION,
                system_prompt=(
                    "Summarize Agentic traces and agent behaviors focusing on "
                    "method_return_values and how agent reacted to different inputs"
                ),
            )
            self.memory_context = str(memory_results)
        except NoDataError:
            self.memory_context = ""
