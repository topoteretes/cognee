from __future__ import annotations

from typing import Any
from uuid import NAMESPACE_OID, uuid5

from cognee.infrastructure.engine import DataPoint
from cognee.modules.engine.models import NodeSet
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
    belongs_to_set: list[NodeSet] = NodeSet(
            id=uuid5(NAMESPACE_OID, "NodeSet:agentic_traces"),
            name="agentic_traces",
        )
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
                system_prompt='Answer the query, in a case of empty context return empty string.',
                top_k=5
            )
            self.memory_context = str(memory_results)
        except NoDataError:
            self.memory_context = ""
