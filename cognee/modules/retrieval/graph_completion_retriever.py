import asyncio
from typing import Any, Optional, Type, List, Union

from cognee.infrastructure.engine import DataPoint
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge
from cognee.modules.retrieval.utils.validate_queries import validate_retriever_input
from cognee.modules.graph.utils import resolve_edges_to_text
from cognee.modules.graph.utils.convert_node_to_data_point import get_all_subclasses
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.utils.brute_force_triplet_search import brute_force_triplet_search
from cognee.modules.retrieval.utils.completion import (
    generate_completion,
    generate_completion_batch,
)
from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.context_global_variables import session_user
from cognee.infrastructure.databases.cache.config import CacheConfig

logger = get_logger("GraphCompletionRetriever")


class GraphCompletionRetriever(BaseRetriever):
    """
    Retriever for handling graph-based completion searches.

    This class implements the retrieval pipeline by searching for graph triplets (get_retrieved_objects function),
    resolving those triplets into human-readable text context (get_context_from_objects function), and generating
    LLM completions using the retrieved graph data (get_completion_from_context function).
    """

    def __init__(
        self,
        user_prompt_path: str = "graph_context_for_question.txt",
        system_prompt_path: str = "answer_simple_question.txt",
        system_prompt: Optional[str] = None,
        top_k: Optional[int] = 5,
        node_type: Optional[Type] = None,
        node_name: Optional[List[str]] = None,
        wide_search_top_k: Optional[int] = 100,
        triplet_distance_penalty: Optional[float] = 3.5,
        session_id: Optional[str] = None,
        response_model: Type = str,
    ):
        """Initialize retriever with prompt paths and search parameters."""
        self.user_prompt_path = user_prompt_path
        self.system_prompt_path = system_prompt_path
        self.system_prompt = system_prompt
        self.top_k = top_k if top_k is not None else 5
        self.wide_search_top_k = wide_search_top_k
        self.node_type = node_type
        self.node_name = node_name
        self.triplet_distance_penalty = triplet_distance_penalty
        # session_id (Optional[str]): Identifier for managing conversation history.
        self.session_id = session_id
        # response_model (Type): The Pydantic model or type for the expected response.
        self.response_model = response_model

    def _use_session_cache(self) -> bool:
        """Check if session caching is enabled for the current user."""
        user = session_user.get()
        user_id = getattr(user, "id", None)
        return bool(user_id and CacheConfig().caching)

    @staticmethod
    def _get_vector_index_collections() -> List[str]:
        """Collect vector index collection names from all DataPoint subclasses."""
        collections = []
        for subclass in get_all_subclasses(DataPoint):
            metadata = subclass.model_fields.get("metadata")
            if metadata is None:
                continue
            default = getattr(metadata, "default", None)
            if isinstance(default, dict):
                for field_name in default.get("index_fields", []):
                    collections.append(f"{subclass.__name__}_{field_name}")
        return collections

    async def get_retrieved_objects(
        self, query: Optional[str] = None, query_batch: Optional[List[str]] = None
    ) -> Union[List[Edge], List[List[Edge]]]:
        """
        Performs a brute-force triplet search on the graph and updates access timestamps.

        Args:
            query (str): The search query to find relevant graph triplets.
            query_batch (str): The batch of search queries to find relevant graph triplets.

        Returns:
            List[Edge]: A list of retrieved Edge objects (triplets).
                       Returns an empty list if the graph is empty or no results are found.
        """

        validate_retriever_input(query, query_batch, self._use_session_cache())

        graph_engine = await get_graph_engine()
        is_empty = await graph_engine.is_empty()

        if is_empty:
            logger.warning("Search attempt on an empty knowledge graph")
            return []

        triplets = await self.get_triplets(query, query_batch)

        # Check if all triplets are empty, in case of batch queries
        if query_batch and all(len(batched_triplets) == 0 for batched_triplets in triplets):
            logger.warning("Empty context was provided to the completion")
            return []

        if len(triplets) == 0:
            logger.warning("Empty context was provided to the completion")
            return []

        return triplets

    async def resolve_edges_to_text(self, retrieved_edges: list) -> str:
        """
        Converts retrieved graph edges into a human-readable string format.

        Parameters:
        -----------

            - retrieved_edges (list): A list of edges retrieved from the graph.

        Returns:
        --------

            - str: A formatted string representation of the nodes and their connections.
        """
        return await resolve_edges_to_text(retrieved_edges)

    async def get_triplets(
        self,
        query: Optional[str] = None,
        query_batch: Optional[List[str]] = None,
    ) -> Union[List[Edge], List[List[Edge]]]:
        """
        Retrieves relevant graph triplets based on a query string.

        Parameters:
        -----------

            - query (str): The query string used to search for relevant triplets in the graph.

        Returns:
        --------

            - list: A list of found triplets that match the query.
        """
        collections = self._get_vector_index_collections()
        return await brute_force_triplet_search(
            query,
            query_batch,
            top_k=self.top_k,
            collections=collections or None,
            node_type=self.node_type,
            node_name=self.node_name,
            wide_search_top_k=self.wide_search_top_k,
            triplet_distance_penalty=self.triplet_distance_penalty,
        )

    async def get_context_from_objects(
        self,
        query: Optional[str] = None,
        query_batch: Optional[List[str]] = None,
        retrieved_objects=None,
    ) -> Union[str, List[str]]:
        """
        Transforms raw retrieved graph triplets into a textual context string.

        Args:
            query (str): The original search query.
            query_batch (List[str]): The batch of original search queries.
            retrieved_objects (List[Edge]): The raw triplets returned from the search.
                                            Output of the get_retrieved_objects method.

        Returns:
            str: A string representing the resolved graph context.
                 Returns an empty list (as string) if no triplets are provided.

        Note: To avoid duplicate retrievals, ensure that retrieved_objects
              are provided from get_retrieved_objects method call.
        """

        triplets = retrieved_objects

        if query_batch:
            # Check if all triplets are empty, in case of batch queries
            if not triplets or all(len(batched_triplets) == 0 for batched_triplets in triplets):
                logger.warning("Empty context was provided to the completion")
                return ["" for _ in query_batch]

            return await asyncio.gather(
                *[self.resolve_edges_to_text(batched_triplets) for batched_triplets in triplets]
            )

        if not triplets:
            logger.warning("Empty context was provided to the completion")
            return ""

        return await self.resolve_edges_to_text(triplets)

    def _completion_kwargs(self, context: str) -> dict:
        """Common kwargs for completion calls (no session)."""
        return {
            "context": context,
            "user_prompt_path": self.user_prompt_path,
            "system_prompt_path": self.system_prompt_path,
            "system_prompt": self.system_prompt,
            "response_model": self.response_model,
        }

    async def _generate_completion_without_session(
        self,
        query: Optional[str],
        query_batch: Optional[List[str]],
        context: str,
    ) -> List[Any]:
        """Generate completion(s) without session; returns list of completions."""
        kwargs = self._completion_kwargs(context)
        if query_batch:
            return await generate_completion_batch(query_batch=query_batch, **kwargs)
        completion = await generate_completion(query=query, **kwargs)
        return [completion]

    async def get_completion_from_context(
        self,
        query: Optional[str] = None,
        query_batch: Optional[List[str]] = None,
        retrieved_objects: Optional[List[Edge]] = None,
        context: str = None,
    ) -> List[Any]:
        """
        Generates an LLM response based on the query, context, and conversation history.
        Optionally saves the interaction and updates the session cache.

        Args:
            query (str): The user's question or prompt.
            query_batch (List[str]): The batch of user queries.
            retrieved_objects (Optional[List[Edge]]): Raw triplets used for interaction mapping.
                                                     Output of get_retrieved_objects method.
            context (str): The text-resolved graph context.
                           Output of the get_context_from_objects method.

        Returns:
            List[Any]: A list containing the generated response (completion).

        Note: To avoid duplicate retrievals, ensure that retrieved_objects and context
              are provided from previous method calls.
        """
        use_session = self._use_session_cache() and not query_batch
        if use_session:
            sm = get_session_manager()
            completion = await sm.generate_completion_with_session(
                session_id=self.session_id,
                query=query,
                context=context,
                user_prompt_path=self.user_prompt_path,
                system_prompt_path=self.system_prompt_path,
                system_prompt=self.system_prompt,
                response_model=self.response_model,
                summarize_context=False,
            )
            return [completion]
        return await self._generate_completion_without_session(query, query_batch, context)

    async def get_completion(
        self, query: Optional[str] = None, query_batch: Optional[List[str]] = None
    ) -> List[Any]:
        """
        Generates a final output or answer based on the query and retrieved context.

        Args:
            query (str): The original user query.
            query_batch (List[str]): The batch of user queries.

        Returns:
            List[Any]: A list containing the generated completions or response objects.
        """
        validate_retriever_input(query, query_batch)

        retrieved_objects = await self.get_retrieved_objects(query=query, query_batch=query_batch)
        context = await self.get_context_from_objects(
            query=query, query_batch=query_batch, retrieved_objects=retrieved_objects
        )
        completion = await self.get_completion_from_context(
            query=query,
            query_batch=query_batch,
            retrieved_objects=retrieved_objects,
            context=context,
        )

        return completion
