import asyncio
from typing import Any, List, Optional, Type

from cognee.infrastructure.databases.unified import get_unified_engine
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.prompts import read_query_prompt
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge
from cognee.modules.retrieval.exceptions.exceptions import QueryValidationError
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.retrieval.utils.completion import generate_completion
from cognee.modules.retrieval.utils.query_decomposition import (
    DecompositionMode,
    DecompositionRunState,
    QueryDecomposition,
    SubqueryRunState,
    build_subquery_answer_context,
    merge_deduplicated_edges,
    normalize_subqueries,
)
from cognee.modules.retrieval.utils.validate_queries import validate_retriever_input
from cognee.shared.logging_utils import get_logger

logger = get_logger("GraphCompletionDecompositionRetriever")


class GraphCompletionDecompositionRetriever(GraphCompletionRetriever):
    """
    Graph completion retriever that decomposes a single user query into a small
    set of focused subqueries before running the standard graph-completion flow.
    """

    def __init__(
        self,
        user_prompt_path: str = "graph_context_for_question.txt",
        system_prompt_path: str = "answer_simple_question.txt",
        system_prompt: Optional[str] = None,
        top_k: Optional[int] = 5,
        node_type: Optional[Type] = None,
        node_name: Optional[List[str]] = None,
        node_name_filter_operator: str = "OR",
        wide_search_top_k: Optional[int] = 100,
        triplet_distance_penalty: Optional[float] = 6.5,
        feedback_influence: float = 0.0,
        session_id: Optional[str] = None,
        response_model: Type = str,
        neighborhood_depth: Optional[int] = None,
        neighborhood_seed_top_k: Optional[int] = 10,
        decomposition_mode: DecompositionMode = DecompositionMode.ANSWER_PER_SUBQUERY,
    ):
        super().__init__(
            user_prompt_path=user_prompt_path,
            system_prompt_path=system_prompt_path,
            system_prompt=system_prompt,
            top_k=top_k,
            node_type=node_type,
            node_name=node_name,
            node_name_filter_operator=node_name_filter_operator,
            wide_search_top_k=wide_search_top_k,
            triplet_distance_penalty=triplet_distance_penalty,
            feedback_influence=feedback_influence,
            session_id=session_id,
            response_model=response_model,
            neighborhood_depth=neighborhood_depth,
            neighborhood_seed_top_k=neighborhood_seed_top_k,
        )
        self.decomposition_mode = DecompositionMode(decomposition_mode)
        self._decomposition_state: Optional[DecompositionRunState] = None

    def _validate_single_query_input(
        self,
        query: Optional[str],
        query_batch: Optional[List[str]],
    ) -> None:
        """Validate the retriever's single-query public contract."""

        if query_batch is not None:
            raise QueryValidationError(
                message=(
                    "GraphCompletionDecompositionRetriever accepts only a single query. "
                    "Decomposition batching is handled internally."
                )
            )
        validate_retriever_input(query, None, self._use_session_cache())

    async def _decompose_query(self, query: str) -> List[str]:
        """Decompose the original query into focused subqueries."""

        system_prompt = read_query_prompt("graph_completion_decomposition_system_prompt.txt")
        if not system_prompt:
            logger.warning("Decomposition prompt not found, falling back to original query.")
            return [query]

        try:
            decomposition = await LLMGateway.acreate_structured_output(
                text_input=query,
                system_prompt=system_prompt,
                response_model=QueryDecomposition,
            )
        except Exception as error:
            logger.warning(
                "Query decomposition failed, falling back to original query: %s",
                error,
                exc_info=False,
            )
            return [query]

        return normalize_subqueries(query, getattr(decomposition, "subqueries", None))

    async def _ensure_state(self, query: Optional[str]) -> DecompositionRunState:
        """Return cached run state or initialize it from the query."""

        if (
            self._decomposition_state is not None
            and query == self._decomposition_state.original_query
        ):
            return self._decomposition_state

        if query is None:
            raise QueryValidationError(message="A non-empty query is required.")

        await self.get_retrieved_objects(query=query)
        if self._decomposition_state is None:
            raise QueryValidationError(message="Failed to initialize decomposition state.")
        return self._decomposition_state

    async def _resolve_subquery_context_and_answer(
        self,
        subquery: str,
        edge_batch: List[Edge],
    ) -> tuple[str, str]:
        """Resolve context and answer for one subquery."""

        subquery_context = await super().get_context_from_objects(
            query=subquery,
            retrieved_objects=edge_batch,
        )
        subquery_context = subquery_context if isinstance(subquery_context, str) else ""

        subquery_answer = await generate_completion(
            query=subquery,
            context=subquery_context,
            user_prompt_path=self.user_prompt_path,
            system_prompt_path=self.system_prompt_path,
            system_prompt=self.system_prompt,
            response_model=str,
        )

        return subquery_context, subquery_answer

    async def get_retrieved_objects(
        self, query: Optional[str] = None, query_batch: Optional[List[str]] = None
    ) -> List[Edge]:
        """Retrieve and merge edges for the decomposed subqueries."""

        self._validate_single_query_input(query, query_batch)

        self._decomposition_state = None
        self._unified_engine = await get_unified_engine()
        is_empty = await self._unified_engine.graph.is_empty()

        if is_empty:
            logger.warning("Search attempt on an empty knowledge graph")
            self._decomposition_state = DecompositionRunState(
                original_query=query,
                subqueries=[SubqueryRunState(query=query)],
            )
            return []

        subqueries = await self._decompose_query(query)
        state = DecompositionRunState(
            original_query=query,
            subqueries=[SubqueryRunState(query=subquery) for subquery in subqueries],
        )
        self._decomposition_state = state

        edge_batches = await self.get_triplets_batch(subqueries)
        for subquery_state, edge_batch in zip(state.subqueries, edge_batches):
            subquery_state.edges = edge_batch

        state.merged_edges = merge_deduplicated_edges(
            [subquery_state.edges for subquery_state in state.subqueries]
        )

        if not state.merged_edges:
            logger.warning("Empty context was provided to the completion")

        return state.merged_edges

    async def get_context_from_objects(
        self,
        query: Optional[str] = None,
        query_batch: Optional[List[str]] = None,
        retrieved_objects=None,
    ) -> str:
        """Build the final context for the original query."""

        self._validate_single_query_input(query, query_batch)

        state = await self._ensure_state(query)
        if state.final_context is not None:
            return state.final_context

        retrieved_objects = state.merged_edges if retrieved_objects is None else retrieved_objects
        if not retrieved_objects:
            state.final_context = ""
            return state.final_context

        if self.decomposition_mode is DecompositionMode.COMBINED_TRIPLETS_CONTEXT:
            state.final_context = await super().get_context_from_objects(
                query=query,
                retrieved_objects=retrieved_objects,
            )
            return state.final_context

        subquery_results = await asyncio.gather(
            *[
                self._resolve_subquery_context_and_answer(
                    subquery_state.query, subquery_state.edges
                )
                for subquery_state in state.subqueries
            ]
        )
        for subquery_state, (context, answer) in zip(state.subqueries, subquery_results):
            subquery_state.context = context
            subquery_state.answer = answer

        state.final_context = build_subquery_answer_context(state)
        return state.final_context

    async def get_completion_from_context(
        self,
        query: Optional[str] = None,
        query_batch: Optional[List[str]] = None,
        retrieved_objects: Optional[List[Edge]] = None,
        context: str = None,
    ) -> List[Any]:
        """Generate the final completion for the original query."""

        self._validate_single_query_input(query, query_batch)

        state = await self._ensure_state(query)
        retrieved_objects = state.merged_edges if retrieved_objects is None else retrieved_objects
        if context is None:
            context = await self.get_context_from_objects(
                query=query,
                retrieved_objects=retrieved_objects,
            )

        return await super().get_completion_from_context(
            query=query,
            retrieved_objects=retrieved_objects,
            context=context,
        )
