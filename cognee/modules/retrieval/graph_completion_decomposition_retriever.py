import json
from dataclasses import dataclass, field
from typing import Any, List, Optional, Type

from pydantic import BaseModel

from cognee.exceptions import CogneeValidationError
from cognee.infrastructure.databases.unified import get_unified_engine
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.llm.prompts import read_query_prompt
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge
from cognee.modules.retrieval.exceptions.exceptions import QueryValidationError
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.retrieval.utils.completion import generate_completion
from cognee.modules.retrieval.utils.validate_queries import validate_retriever_input
from cognee.shared.logging_utils import get_logger

logger = get_logger("GraphCompletionDecompositionRetriever")

ANSWER_PER_SUBQUERY_MODE = "answer_per_subquery"
COMBINED_TRIPLETS_CONTEXT_MODE = "combined_triplets_context"
DECOMPOSITION_MODES = {
    ANSWER_PER_SUBQUERY_MODE,
    COMBINED_TRIPLETS_CONTEXT_MODE,
}
DECOMPOSITION_PROMPT_PATH = "graph_completion_decomposition_system_prompt.txt"
MAX_SUBQUERIES = 5


class QueryDecomposition(BaseModel):
    subqueries: List[str]


@dataclass
class DecompositionRunState:
    original_query: str
    subqueries: List[str] = field(default_factory=list)
    subquery_edge_batches: List[List[Edge]] = field(default_factory=list)
    subquery_contexts: List[str] = field(default_factory=list)
    subquery_answers: List[str] = field(default_factory=list)
    merged_edges: List[Edge] = field(default_factory=list)
    final_context: Optional[str] = None


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
        decomposition_mode: str = ANSWER_PER_SUBQUERY_MODE,
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
        self.decomposition_mode = self._normalize_decomposition_mode(decomposition_mode)
        self._decomposition_state: Optional[DecompositionRunState] = None

    @staticmethod
    def _normalize_decomposition_mode(mode: str) -> str:
        normalized_mode = (mode or "").strip().lower()
        if normalized_mode not in DECOMPOSITION_MODES:
            raise CogneeValidationError(
                message=(
                    f"Invalid decomposition_mode: {mode!r}. "
                    f"Must be one of {sorted(DECOMPOSITION_MODES)}."
                ),
                name="InvalidDecompositionMode",
            )
        return normalized_mode

    @staticmethod
    def _normalize_subqueries(original_query: str, subqueries: Optional[List[str]]) -> List[str]:
        normalized_queries: List[str] = []
        seen_queries: set[str] = set()

        for subquery in subqueries or []:
            if not isinstance(subquery, str):
                continue
            cleaned_query = " ".join(subquery.split())
            if not cleaned_query:
                continue
            query_key = cleaned_query.casefold()
            if query_key in seen_queries:
                continue
            normalized_queries.append(cleaned_query)
            seen_queries.add(query_key)
            if len(normalized_queries) >= MAX_SUBQUERIES:
                break

        if normalized_queries:
            return normalized_queries

        fallback_query = " ".join(original_query.split())
        return [fallback_query or original_query]

    @staticmethod
    def _completion_to_text(completion: Any) -> str:
        if isinstance(completion, str):
            return completion
        if isinstance(completion, BaseModel):
            return completion.model_dump_json(indent=2)
        try:
            return json.dumps(completion, indent=2)
        except TypeError:
            return str(completion)

    @staticmethod
    def _edge_dedup_key(edge: Edge) -> tuple:
        attributes = getattr(edge, "attributes", None) or {}
        edge_object_id = attributes.get("edge_object_id")
        if edge_object_id:
            return ("edge_object_id", str(edge_object_id))

        relationship_label = (
            attributes.get("relationship_name")
            or attributes.get("edge_text")
            or attributes.get("relationship_type")
        )
        source_id = getattr(getattr(edge, "node1", None), "id", None)
        target_id = getattr(getattr(edge, "node2", None), "id", None)
        directed = bool(getattr(edge, "directed", True))
        return ("edge_shape", str(source_id), str(relationship_label), str(target_id), directed)

    @classmethod
    def _merge_deduplicated_edges(cls, edge_batches: List[List[Edge]]) -> List[Edge]:
        merged_edges: List[Edge] = []
        seen_keys: set[tuple] = set()

        for edge_batch in edge_batches:
            for edge in edge_batch:
                edge_key = cls._edge_dedup_key(edge)
                if edge_key in seen_keys:
                    continue
                merged_edges.append(edge)
                seen_keys.add(edge_key)

        return merged_edges

    @staticmethod
    def _build_subquery_answer_context(state: DecompositionRunState) -> str:
        sections: List[str] = []
        for index, (subquery, answer) in enumerate(
            zip(state.subqueries, state.subquery_answers), start=1
        ):
            sections.append(
                f"Subquery {index}: {subquery}\n"
                f"Subquery {index} Answer:\n{answer}"
            )

        return "Question decomposition results:\n\n" + "\n\n".join(sections)

    def _validate_single_query_input(
        self,
        query: Optional[str],
        query_batch: Optional[List[str]],
    ) -> None:
        if query_batch is not None:
            raise QueryValidationError(
                message=(
                    "GraphCompletionDecompositionRetriever accepts only a single query. "
                    "Decomposition batching is handled internally."
                )
            )
        validate_retriever_input(query, None, self._use_session_cache())

    async def _decompose_query(self, query: str) -> List[str]:
        system_prompt = read_query_prompt(DECOMPOSITION_PROMPT_PATH)
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

        return self._normalize_subqueries(query, getattr(decomposition, "subqueries", None))

    async def _ensure_state(self, query: Optional[str]) -> DecompositionRunState:
        if (
            self._decomposition_state is not None
            and query == self._decomposition_state.original_query
        ):
            return self._decomposition_state

        if query is None:
            raise QueryValidationError(message="A non-empty query is required.")

        await self.get_retrieved_objects(query=query)
        return self._decomposition_state

    async def get_retrieved_objects(
        self, query: Optional[str] = None, query_batch: Optional[List[str]] = None
    ) -> List[Edge]:
        self._validate_single_query_input(query, query_batch)

        self._decomposition_state = None
        self._unified_engine = await get_unified_engine()
        is_empty = await self._unified_engine.graph.is_empty()

        if is_empty:
            logger.warning("Search attempt on an empty knowledge graph")
            self._decomposition_state = DecompositionRunState(
                original_query=query,
                subqueries=[query],
            )
            return []

        subqueries = await self._decompose_query(query)
        state = DecompositionRunState(original_query=query, subqueries=subqueries)
        self._decomposition_state = state

        if self.decomposition_mode == COMBINED_TRIPLETS_CONTEXT_MODE:
            state.subquery_edge_batches = await self.get_triplets_batch(subqueries)
        else:
            for subquery in subqueries:
                triplets = await self.get_triplets(query=subquery)
                state.subquery_edge_batches.append(triplets or [])

        state.merged_edges = self._merge_deduplicated_edges(state.subquery_edge_batches)

        if not state.merged_edges:
            logger.warning("Empty context was provided to the completion")

        return state.merged_edges

    async def get_context_from_objects(
        self,
        query: Optional[str] = None,
        query_batch: Optional[List[str]] = None,
        retrieved_objects=None,
    ) -> str:
        self._validate_single_query_input(query, query_batch)

        state = await self._ensure_state(query)
        if state.final_context is not None:
            return state.final_context

        retrieved_objects = state.merged_edges if retrieved_objects is None else retrieved_objects
        if not retrieved_objects:
            state.final_context = ""
            return state.final_context

        if self.decomposition_mode == COMBINED_TRIPLETS_CONTEXT_MODE:
            state.final_context = await super().get_context_from_objects(
                query=query,
                retrieved_objects=retrieved_objects,
            )
            return state.final_context

        for subquery, edge_batch in zip(state.subqueries, state.subquery_edge_batches):
            subquery_context = await super().get_context_from_objects(
                query=subquery,
                retrieved_objects=edge_batch,
            )
            subquery_context = subquery_context if isinstance(subquery_context, str) else ""
            state.subquery_contexts.append(subquery_context)

            subquery_answer = await generate_completion(
                query=subquery,
                context=subquery_context,
                user_prompt_path=self.user_prompt_path,
                system_prompt_path=self.system_prompt_path,
                system_prompt=self.system_prompt,
                response_model=str,
            )
            state.subquery_answers.append(self._completion_to_text(subquery_answer))

        state.final_context = self._build_subquery_answer_context(state)
        return state.final_context

    async def get_completion_from_context(
        self,
        query: Optional[str] = None,
        query_batch: Optional[List[str]] = None,
        retrieved_objects: Optional[List[Edge]] = None,
        context: str = None,
    ) -> List[Any]:
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
