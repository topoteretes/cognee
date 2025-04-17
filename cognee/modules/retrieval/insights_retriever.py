import asyncio
from typing import Any, Optional

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.exceptions.exceptions import NoDataError
from cognee.infrastructure.databases.vector.exceptions.exceptions import CollectionNotFoundError


class InsightsRetriever(BaseRetriever):
    """Retriever for handling graph connection-based insights."""

    def __init__(self, exploration_levels: int = 1, top_k: int = 5):
        """Initialize retriever with exploration levels and search parameters."""
        self.exploration_levels = exploration_levels
        self.top_k = top_k

    async def get_context(self, query: str) -> list:
        """Find the neighbours of a given node in the graph."""
        if query is None:
            return []

        node_id = query
        graph_engine = await get_graph_engine()
        exact_node = await graph_engine.extract_node(node_id)

        if exact_node is not None and "id" in exact_node:
            node_connections = await graph_engine.get_connections(str(exact_node["id"]))
        else:
            vector_engine = get_vector_engine()

            try:
                results = await asyncio.gather(
                    vector_engine.search("Entity_name", query_text=query, limit=self.top_k),
                    vector_engine.search("EntityType_name", query_text=query, limit=self.top_k),
                )
            except CollectionNotFoundError as error:
                raise NoDataError("No data found in the system, please add data first.") from error

            results = [*results[0], *results[1]]
            relevant_results = [result for result in results if result.score < 0.5][: self.top_k]

            if len(relevant_results) == 0:
                return []

            node_connections_results = await asyncio.gather(
                *[graph_engine.get_connections(result.id) for result in relevant_results]
            )

            node_connections = []
            for neighbours in node_connections_results:
                node_connections.extend(neighbours)

        unique_node_connections_map = {}
        unique_node_connections = []

        for node_connection in node_connections:
            if "id" not in node_connection[0] or "id" not in node_connection[2]:
                continue

            unique_id = f"{node_connection[0]['id']} {node_connection[1]['relationship_name']} {node_connection[2]['id']}"
            if unique_id not in unique_node_connections_map:
                unique_node_connections_map[unique_id] = True
                unique_node_connections.append(node_connection)

        return unique_node_connections

    async def get_completion(self, query: str, context: Optional[Any] = None) -> Any:
        """Returns the graph connections context."""
        if context is None:
            context = await self.get_context(query)
        return context
