from typing import Any, Optional
from fastapi.encoders import jsonable_encoder

from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.utils.completion import generate_completion
from cognee.modules.retrieval.exceptions import SearchTypeNotSupported, CypherSearchError
from cognee.shared.logging_utils import get_logger

logger = get_logger("CypherSearchRetriever")


class CypherSearchRetriever(BaseRetriever):
    """
    Retriever for handling cypher-based search.

    Public methods include:
    - get_context: Retrieves relevant context using a cypher query.
    - get_completion: Returns the graph connections context.
    """

    # Runs a Cypher query and returns rows; the prompt paths it accepts are never sent
    # to an LLM, so an only_context preview must not render a prompt for it.
    supports_prompt_preview = False

    def __init__(
        self,
        user_prompt_path: str = "context_for_question.txt",
        system_prompt_path: str = "answer_simple_question.txt",
        session_id: Optional[str] = None,
    ):
        """Initialize retriever with optional custom prompt paths."""
        self.user_prompt_path = user_prompt_path
        self.system_prompt_path = system_prompt_path
        self.session_id = session_id

    async def get_retrieved_objects(self, query: str) -> Any:
        try:
            graph_engine = await get_graph_engine()

            # Cypher support is declared on the adapter class
            # (GraphDBInterface.supports_cypher_queries), so the check needs no
            # imports of optional backend packages absent from slim images.
            if not getattr(graph_engine, "supports_cypher_queries", True):
                raise SearchTypeNotSupported(
                    f"Cypher search is not supported with the "
                    f"{type(graph_engine).__name__} graph backend. Use a "
                    "Cypher-capable graph backend (Neo4j, Ladybug) for raw Cypher queries."
                )

            is_empty = await graph_engine.is_empty()

            if is_empty:
                logger.warning("Search attempt on an empty knowledge graph")
                return []

            result = await graph_engine.query(query)
        except SearchTypeNotSupported:
            raise
        except Exception as e:
            logger.error("Failed to execute cypher search retrieval: %s", str(e))
            raise CypherSearchError() from e
        return result

    async def get_context_from_objects(self, query: str, retrieved_objects: Any) -> Any:
        """
        Retrieves relevant context using a cypher query.

        If any error occurs during execution, logs the error and raises CypherSearchError.

        Parameters:
        -----------

            - query (str): The cypher query used to retrieve context.

        Returns:
        --------

            - Any: The result of the cypher query execution.
        """
        # TODO: Do we want to return a string response here?
        # return jsonable_encoder(retrieved_objects)
        return None

    async def get_completion_from_context(
        self, query: str, retrieved_objects: Any, context: Optional[Any] = None
    ) -> Any:
        """
        Returns the graph connections context.

        If no context is provided, it retrieves the context using the specified query.

        Parameters:
        -----------

            - query (str): The query to retrieve context.
            - context (Optional[Any]): Optional context to use, otherwise fetched using the
              query. (default None)
              defaults to 'default_session'. (default None)

        Returns:
        --------

            - Any: The context, either provided or retrieved.
        """
        # TODO: Do we want to generate a completion using LLM here?
        return None
