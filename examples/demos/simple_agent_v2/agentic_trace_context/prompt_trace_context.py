from __future__ import annotations

from cognee.infrastructure.engine import DataPoint


class AgentContextTrace(DataPoint):
    """Minimal context payload for agentic tracing as a Cognee DataPoint."""

    origin_function: str = ""
    with_memory: bool = False
    task_query: str = ""
    memory_context: str = ""

    async def get_memory_context(self, query_text: str) -> None:
        if not self.with_memory:
            return

        from cognee import SearchType, search

        memory_query = self.task_query or query_text
        memory_results = await search(
            query_text=memory_query,
            query_type=SearchType.GRAPH_COMPLETION,
        )
        self.memory_context = str(memory_results)
