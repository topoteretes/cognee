import asyncio
from typing import Any, Optional

from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.exceptions.exceptions import NoDataError
from cognee.infrastructure.databases.vector.exceptions.exceptions import CollectionNotFoundError

logger = get_logger("InsightsRetriever")


class InsightsRetriever(BaseRetriever):
    """
    Retriever for handling graph connection-based insights.

    Public methods include:
    - get_context
    - get_completion

    Instance variables include:
    - exploration_levels
    - top_k
    """

    def __init__(self, exploration_levels: int = 1, top_k: int = 5):
        """Initialize retriever with exploration levels and search parameters."""
        self.exploration_levels = exploration_levels
        self.top_k = top_k
        logger.info(
            f"Initialized InsightsRetriever with exploration_levels={self.exploration_levels}, top_k={self.top_k}"
        )

    async def get_context(self, query: str) -> list:
        """
        Find neighbours of a given node in the graph.

        If the provided query does not correspond to an existing node,
        search for similar entities and retrieve their connections.
        Reraises NoDataError if there is no data found in the system.

        Parameters:
        -----------

            - query (str): A string identifier for the node whose neighbours are to be
              retrieved.

        Returns:
        --------

            - list: A list of unique connections found for the queried node.
        """
        logger.info(
            f"Starting insights retrieval for query: '{query[:100] if query else 'None'}{'...' if query and len(query) > 100 else ''}'"
        )

        if query is None:
            logger.warning("Query is None, returning empty list")
            return []

        node_id = query
        logger.debug(f"Looking for exact node with id: {node_id}")

        graph_engine = await get_graph_engine()
        exact_node = await graph_engine.extract_node(node_id)

        if exact_node is not None and "id" in exact_node:
            logger.info(f"Found exact node with id: {exact_node['id']}")
            node_connections = await graph_engine.get_connections(str(exact_node["id"]))
            logger.info(f"Retrieved {len(node_connections)} direct connections for exact node")
        else:
            logger.info("Exact node not found, performing vector search for similar entities")
            vector_engine = get_vector_engine()

            try:
                logger.debug("Searching Entity_name and EntityType_name collections")
                results = await asyncio.gather(
                    vector_engine.search("Entity_name", query_text=query, limit=self.top_k),
                    vector_engine.search("EntityType_name", query_text=query, limit=self.top_k),
                )
                logger.info(
                    f"Vector search returned {len(results[0])} Entity_name results and {len(results[1])} EntityType_name results"
                )
            except CollectionNotFoundError as error:
                logger.error("Entity collections not found in vector database")
                raise NoDataError("No data found in the system, please add data first.") from error
            except Exception as e:
                logger.error(f"Unexpected error during vector search: {str(e)}")
                raise

            results = [*results[0], *results[1]]
            relevant_results = [result for result in results if result.score < 0.5][: self.top_k]
            logger.info(f"Filtered to {len(relevant_results)} relevant results (score < 0.5)")

            if len(relevant_results) == 0:
                logger.warning("No relevant results found, returning empty list")
                return []

            logger.debug(f"Getting connections for {len(relevant_results)} relevant nodes")
            node_connections_results = await asyncio.gather(
                *[graph_engine.get_connections(result.id) for result in relevant_results]
            )

            node_connections = []
            for i, neighbours in enumerate(node_connections_results):
                logger.debug(f"Node {i}: found {len(neighbours)} connections")
                node_connections.extend(neighbours)

            logger.info(f"Total connections found: {len(node_connections)}")

        unique_node_connections_map = {}
        unique_node_connections = []

        for node_connection in node_connections:
            if "id" not in node_connection[0] or "id" not in node_connection[2]:
                logger.debug("Skipping connection with missing node IDs")
                continue

            unique_id = f"{node_connection[0]['id']} {node_connection[1]['relationship_name']} {node_connection[2]['id']}"
            if unique_id not in unique_node_connections_map:
                unique_node_connections_map[unique_id] = True
                unique_node_connections.append(node_connection)

        logger.info(
            f"Returning {len(unique_node_connections)} unique connections after deduplication"
        )
        return unique_node_connections

    async def get_completion(self, query: str, context: Optional[Any] = None) -> Any:
        """
        Returns the graph connections context.

        If a context is not provided, it fetches the context using the query provided.

        Parameters:
        -----------

            - query (str): A string identifier used to fetch the context.
            - context (Optional[Any]): An optional context to use for the completion; if None,
              it fetches the context based on the query. (default None)

        Returns:
        --------

            - Any: The context used for the completion, which is either provided or fetched
              based on the query.
        """
        logger.info(
            f"Starting completion generation for query: '{query[:100]}{'...' if len(query) > 100 else ''}'"
        )

        if context is None:
            logger.debug("No context provided, retrieving context from graph")
            context = await self.get_context(query)
        else:
            logger.debug("Using provided context")

        logger.info(
            f"Returning context with {len(context) if isinstance(context, list) else 1} item(s)"
        )
        return context
