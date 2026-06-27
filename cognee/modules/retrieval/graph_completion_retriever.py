import asyncio
from typing import Any, Dict, List, Optional, Type, Union
from uuid import UUID

from cognee.context_global_variables import session_user
from cognee.infrastructure.databases.cache.config import CacheConfig
from cognee.infrastructure.databases.unified import get_unified_engine
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.modules.graph.cognee_graph.CogneeGraphElements import Edge
from cognee.modules.graph.utils import resolve_edges_to_text
from cognee.modules.graph.utils.convert_node_to_data_point import get_all_subclasses
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.utils.brute_force_triplet_search import (
    brute_force_triplet_search,
)
from cognee.modules.retrieval.utils.completion import (
    generate_completion,
    generate_completion_batch,
)
from cognee.modules.retrieval.utils.global_context import (
    format_global_context_prelude,
    load_root_text,
    search_top_global_context_summaries,
)
from cognee.modules.retrieval.utils.references import append_answer_grounded_evidence
from cognee.modules.retrieval.utils.used_graph_elements import (
    extract_from_edges,
    is_edge_list,
)
from cognee.modules.retrieval.utils.validate_queries import validate_retriever_input
from cognee.shared.logging_utils import get_logger

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
        node_name_filter_operator: str = "OR",
        wide_search_top_k: Optional[int] = 100,
        triplet_distance_penalty: Optional[float] = 6.5,
        feedback_influence: float = 0.0,
        session_id: Optional[str] = None,
        response_model: Type = str,
        neighborhood_depth: Optional[int] = None,
        neighborhood_seed_top_k: Optional[int] = 10,
        include_global_context_index: bool = False,
        global_context_index_top_k: int = 3,
        include_references: bool = False,
        dataset_ids: Optional[List[UUID]] = None,
        dataset_names: Optional[List[str]] = None,
    ):
        """Initialize retriever with prompt paths and search parameters."""
        self.user_prompt_path = user_prompt_path
        self.system_prompt_path = system_prompt_path
        self.system_prompt = system_prompt
        self.top_k = top_k if top_k is not None else 5
        self.wide_search_top_k = wide_search_top_k
        self.node_type = node_type
        self.node_name = node_name
        self.node_name_filter_operator = node_name_filter_operator
        self.triplet_distance_penalty = triplet_distance_penalty
        self.feedback_influence = feedback_influence
        self.session_id = session_id
        self.response_model = response_model
        self.neighborhood_depth = neighborhood_depth
        self.neighborhood_seed_top_k = neighborhood_seed_top_k
        self.include_global_context_index = include_global_context_index
        self.global_context_index_top_k = global_context_index_top_k
        self.include_references = include_references
        self.dataset_ids = dataset_ids
        self.dataset_names = dataset_names

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
        """
        validate_retriever_input(query, query_batch, self._use_session_cache())

        self._unified_engine = await get_unified_engine()
        is_empty = await self._unified_engine.graph.is_empty()

        if is_empty:
            logger.warning("Search attempt on an empty knowledge graph")
            return []

        triplets = await self.get_triplets(query, query_batch)

        if query_batch and all(len(batched_triplets) == 0 for batched_triplets in triplets):
            logger.warning("Empty context was provided to the completion")
            return []

        if len(triplets) == 0:
            logger.warning("Empty context was provided to the completion")
            return []

        return triplets

    async def resolve_edges_to_text(self, retrieved_edges: list) -> str:
        return await resolve_edges_to_text(retrieved_edges)

    async def get_triplets(
        self,
        query: Optional[str] = None,
        query_batch: Optional[List[str]] = None,
    ) -> Union[List[Edge], List[List[Edge]]]:
        """
        Retrieves relevant graph triplets based on a query string.
        """
        collections = self._get_vector_index_collections()
        unified_engine = getattr(self, "_unified_engine", None)

        return await brute_force_triplet_search(
            query,
            query_batch,
            top_k=self.top_k,
            collections=collections or None,
            node_type=self.node_type,
            node_name=self.node_name,
            node_name_filter_operator=self.node_name_filter_operator,
            wide_search_top_k=self.wide_search_top_k,
            triplet_distance_penalty=self.triplet_distance_penalty,
            feedback_influence=self.feedback_influence,
            unified_engine=unified_engine,
            neighborhood_depth=self.neighborhood_depth,
            neighborhood_seed_top_k=self.neighborhood_seed_top_k,
            dataset_ids=self.dataset_ids,
            dataset_names=self.dataset_names,
        )

    async def get_triplets_batch(
        self,
        queries: List[str],
    ) -> List[List[Edge]]:
        if len(queries) == 1:
            triplets = await self.get_triplets(query=queries[0])
            return [triplets]
        return await self.get_triplets(query_batch=queries)

    async def get_context_from_objects(
        self,
        query: Optional[str] = None,
        query_batch: Optional[List[str]] = None,
        retrieved_objects=None,
    ) -> Union[str, List[str]]:
        triplets = retrieved_objects

        if query_batch:
            if not triplets or all(len(batched_triplets) == 0 for batched_triplets in triplets):
                logger.warning("Empty context was provided to the completion")
                return ["" for _ in query_batch]

            return await asyncio.gather(
                *[self.resolve_edges_to_text(batched_triplets) for batched_triplets in triplets]
            )

        graph_context = await self.resolve_edges_to_text(triplets) if triplets else ""

        if not self.include_global_context_index:
            if not triplets:
                logger.warning("Empty context was provided to the completion")
                return ""
            return graph_context

        prelude = await self._build_global_context_prelude(query)
        if not prelude and not graph_context:
            logger.warning("Empty context was provided to the completion")
            return ""
        if not prelude:
            return graph_context
        if not graph_context:
            return prelude
        return f"{prelude}\n\n{graph_context}"

    async def _build_global_context_prelude(self, query: Optional[str]) -> str:
        if not query:
            return ""
        if getattr(self, "_unified_engine", None) is None:
            self._unified_engine = await get_unified_engine()
        root_text = await load_root_text()
        top_summaries = await search_top_global_context_summaries(
            query, self.global_context_index_top_k, self._unified_engine.vector
        )
        return format_global_context_prelude(root_text, top_summaries)

    def _extract_context_object_ids(self, retrieved_objects: Any) -> Optional[Dict[str, List[str]]]:
        if not isinstance(retrieved_objects, list) or not retrieved_objects:
            return None
        if not is_edge_list(retrieved_objects):
            return None
        return extract_from_edges(retrieved_objects)

    def _completion_kwargs(self, context: str) -> dict:
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
        kwargs = self._completion_kwargs(context)
        if query_batch:
            return await generate_completion_batch(query_batch=query_batch, **kwargs)
        completion = await generate_completion(query=query, **kwargs)
        return [completion]

    async def _append_graph_evidence(self, completions: List[Any]) -> List[Any]:
        return await append_answer_grounded_evidence(
            completions,
            enabled=self.include_references and self.response_model is str,
        )

    async def get_completion_from_context(
        self,
        query: Optional[str] = None,
        query_batch: Optional[List[str]] = None,
        retrieved_objects: Optional[List[Edge]] = None,
        context: str = None,
        effective_query: Optional[str] = None,
        turn_preparation=None,
        persist_trace: bool = False,
    ) -> List[Any]:
        use_session = self._use_session_cache() and not query_batch
        if use_session:
            sm = get_session_manager()
            used_graph_element_ids = self._extract_context_object_ids(retrieved_objects)

            if persist_trace and self.session_id:
                await sm.add_qa(
                    session_id=self.session_id,
                    question=query or effective_query or "Context-only Retrieval",
                    answer="[Context-only Retrieval Trace]",
                    used_graph_element_ids=used_graph_element_ids,
                )
                return [context]

            completion = await sm.generate_completion_with_session(
                session_id=self.session_id,
                query=query,
                context=context,
                user_prompt_path=self.user_prompt_path,
                system_prompt_path=self.system_prompt_path,
                system_prompt=self.system_prompt,
                response_model=self.response_model,
                summarize_context=False,
                used_graph_element_ids=used_graph_element_ids,
                max_context_chars=getattr(self, "max_context_chars", None),
                effective_query=effective_query,
                turn_preparation=turn_preparation,
            )
            completions = [completion]
        else:
            completions = await self._generate_completion_without_session(
                query, query_batch, context
            )

        return await self._append_graph_evidence(completions)

    async def get_completion(
        self, query: Optional[str] = None, query_batch: Optional[List[str]] = None
    ) -> List[Any]:
        validate_retriever_input(query, query_batch)

        effective_query = query
        turn_preparation = None
        if query is not None and not query_batch:
            turn_preparation = await self.prepare_session_turn_for_retrieval(query)
            if not turn_preparation.should_answer:
                return [turn_preparation.response_to_user or "Got it."]
            effective_query = turn_preparation.effective_query or query

        retrieved_objects = await self.get_retrieved_objects(
            query=effective_query,
            query_batch=query_batch,
        )
        context = await self.get_context_from_objects(
            query=effective_query,
            query_batch=query_batch,
            retrieved_objects=retrieved_objects,
        )
        completion = await self.get_completion_from_context(
            query=query,
            query_batch=query_batch,
            retrieved_objects=retrieved_objects,
            context=context,
            effective_query=effective_query,
            turn_preparation=turn_preparation,
        )

        return completion
