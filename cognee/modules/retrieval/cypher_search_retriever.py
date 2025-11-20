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

    def __init__(
        self,
        user_prompt_path: str = "context_for_question.txt",
        system_prompt_path: str = "answer_simple_question.txt",
    ):
        """Initialize retriever with optional custom prompt paths."""
        self.user_prompt_path = user_prompt_path
        self.system_prompt_path = system_prompt_path

    async def get_context(self, query: str) -> Any:
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
        try:
            graph_engine = await get_graph_engine()
            is_empty = await graph_engine.is_empty()

            if is_empty:
                logger.warning("Search attempt on an empty knowledge graph")
                return []

            result = jsonable_encoder(await graph_engine.query(query))
        except Exception as e:
            logger.error("Failed to execture cypher search retrieval: %s", str(e))
            raise CypherSearchError() from e
        return result

    async def get_completion(
        self, query: str, context: Optional[Any] = None, session_id: Optional[str] = None
    ) -> Any:
        """
        Returns the graph connections context.

        If no context is provided, it retrieves the context using the specified query.

        Parameters:
        -----------

            - query (str): The query to retrieve context.
            - context (Optional[Any]): Optional context to use, otherwise fetched using the
              query. (default None)
            - session_id (Optional[str]): Optional session identifier for caching. If None,
              defaults to 'default_session'. (default None)

        Returns:
        --------

            - Any: The context, either provided or retrieved.
        """
        if context is None:
            context = await self.get_context(query)
        return context
