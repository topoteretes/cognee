import asyncio
from typing import Any, Dict, List, Optional, Type

from cognee.context_global_variables import current_dataset_id, session_user
from cognee.infrastructure.databases.cache.config import CacheConfig
from cognee.infrastructure.databases.unified import get_unified_engine
from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.exceptions.exceptions import QueryValidationError
from cognee.modules.retrieval.hybrid.chunks import retrieve_hybrid_chunks, search_collection
from cognee.modules.retrieval.hybrid.context import (
    extract_context_object_ids,
    format_hybrid_context,
)
from cognee.modules.retrieval.hybrid.entities import build_entities
from cognee.modules.retrieval.hybrid.facts import edge_rank_by_id, select_facts
from cognee.modules.retrieval.utils.completion import generate_completion
from cognee.modules.retrieval.utils.global_context import (
    format_global_context_prelude,
    load_root_text,
    search_top_global_context_summaries,
)
from cognee.modules.retrieval.utils.validate_queries import validate_retriever_input
from cognee.modules.truth_subspace import align
from cognee.modules.truth_subspace.centroids import load_centroids, pad_coords
from cognee.modules.truth_subspace.constants import DEFAULT_K
from cognee.shared.logging_utils import get_logger

logger = get_logger("HybridRetriever")


class HybridRetriever(BaseRetriever):
    """Completion retriever using chunk, entity, and optional global-context channels."""

    def __init__(
        self,
        chunks_top_k: Optional[int] = 5,
        entities_top_k: Optional[int] = 5,
        max_edges_per_entity: int = 10,
        node_name: Optional[List[str]] = None,
        node_name_filter_operator: str = "OR",
        include_global_context_index: bool = False,
        global_context_index_top_k: int = 3,
        session_id: Optional[str] = None,
        response_model: Type = str,
        user_prompt_path: str = "hybrid_context_for_question.txt",
        system_prompt_path: str = "answer_simple_question.txt",
        system_prompt: Optional[str] = None,
        text_summaries_top_k: Optional[int] = None,
        use_importance_weight: bool = True,
        use_truth_weight: bool = False,
        facts_top_k: Optional[int] = 5,
    ):
        self.chunks_top_k = chunks_top_k if chunks_top_k is not None else 5
        self.entities_top_k = entities_top_k if entities_top_k is not None else 5
        self.max_edges_per_entity = max_edges_per_entity
        self.node_name = node_name
        self.node_name_filter_operator = node_name_filter_operator
        self.include_global_context_index = include_global_context_index
        self.global_context_index_top_k = global_context_index_top_k
        self.session_id = session_id
        self.response_model = response_model
        self.user_prompt_path = user_prompt_path
        self.system_prompt_path = system_prompt_path
        self.system_prompt = system_prompt
        self.text_summaries_top_k = text_summaries_top_k
        self.use_importance_weight = use_importance_weight
        self.use_truth_weight = use_truth_weight
        self.facts_top_k = facts_top_k if facts_top_k is not None else 5

    def _use_session_cache(self) -> bool:
        user = session_user.get()
        user_id = getattr(user, "id", None)
        return bool(user_id and CacheConfig().caching)

    async def get_retrieved_objects(
        self, query: Optional[str] = None, query_batch: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        _reject_query_batch(query_batch)
        validate_retriever_input(query, None, self._use_session_cache())

        self._unified_engine = await get_unified_engine()
        query_embeddings = await self._unified_engine.vector.embedding_engine.embed_text([query])
        query_vector = query_embeddings[0]

        q_coords, current_truth_epoch = await self._build_truth_context(query_vector)

        chunk_objects, (entities, facts) = await asyncio.gather(
            retrieve_hybrid_chunks(
                vector_engine=self._unified_engine.vector,
                graph_engine=self._unified_engine.graph,
                query=query,
                chunks_top_k=self.chunks_top_k,
                text_summaries_top_k=self.text_summaries_top_k,
                node_name=self.node_name,
                node_name_filter_operator=self.node_name_filter_operator,
                use_importance_weight=self.use_importance_weight,
                query_vector=query_vector,
                use_truth_weight=self.use_truth_weight,
                q_coords=q_coords,
                current_truth_epoch=current_truth_epoch,
            ),
            self._retrieve_entities_and_facts(query, query_vector),
        )
        return {**chunk_objects, "entities": entities, "facts": facts}

    async def _build_truth_context(self, query_vector: list[float]) -> tuple:
        """Truth-subspace alignment context for the chunk lane.

        Returns ``(q_coords, current_truth_epoch)``. Both are ``None`` when the
        truth weight is off or centroid slots are absent, so ranking stays at
        exact baseline. Fails open to baseline on any error. The per-candidate
        truth-state map is fetched later, inside the chunk lane, over the full
        assembled candidate set (BM25 + vector + summary), so every channel gets
        a consistent truth factor.
        """
        if not self.use_truth_weight:
            return None, None

        try:
            dataset_id = current_dataset_id.get()
            if dataset_id is None:
                return None, None

            centroids = await load_centroids(self._unified_engine.vector, str(dataset_id))
            if not centroids:
                return None, None

            centroid_vectors = [centroid.centroid for centroid in centroids]
            q_coords = pad_coords(align.query_coords(query_vector, centroid_vectors), DEFAULT_K)
            current_truth_epoch = max(centroid.truth_epoch for centroid in centroids)
            return q_coords, current_truth_epoch
        except Exception as error:
            logger.debug("Truth-subspace lookup failed; using baseline ranking: %s", error)
            return None, None

    async def _retrieve_entities_and_facts(self, query: str, query_vector: list[float]) -> tuple:
        """Entity lane, run concurrently with the chunk lane so the graph round trip for
        edge bullets overlaps the chunk pipeline's ranking and summary loading."""
        max_ranked_bullets = self.entities_top_k * max(0, self.max_edges_per_entity)
        entity_hits, edge_hits = await asyncio.gather(
            search_collection(
                self._unified_engine.vector,
                "Entity_name",
                query,
                self.entities_top_k,
                self.node_name,
                self.node_name_filter_operator,
                query_vector=query_vector,
            ),
            search_collection(
                self._unified_engine.vector,
                "EdgeType_relationship_name",
                query,
                max_ranked_bullets + self.facts_top_k,
                self.node_name,
                self.node_name_filter_operator,
                apply_node_filter=False,
                query_vector=query_vector,
            ),
        )
        entities = await build_entities(
            self._unified_engine.graph,
            entity_hits,
            self.max_edges_per_entity,
            edge_rank_by_id(edge_hits),
        )
        return entities, self._select_facts(edge_hits, entities)

    def _select_facts(self, edge_hits: List[Any], entities: List[dict]) -> List[dict]:
        """Facts are gated off for scoped searches: EdgeType rows carry no node-set fields."""
        if self.facts_top_k <= 0 or self.node_name:
            return []

        bullet_ids = {
            edge["edge_type_id"]
            for entity in entities
            for edge in entity.get("edges", [])
            if edge.get("edge_type_id")
        }
        return select_facts(edge_hits, bullet_ids, self.facts_top_k)

    async def get_context_from_objects(
        self,
        query: Optional[str] = None,
        query_batch: Optional[List[str]] = None,
        retrieved_objects: Any = None,
    ) -> str:
        _reject_query_batch(query_batch)
        global_context = await self._build_global_context_section(query)
        return format_hybrid_context(global_context, retrieved_objects)

    async def _build_global_context_section(self, query: Optional[str]) -> str:
        if not self.include_global_context_index or not query:
            return ""

        if getattr(self, "_unified_engine", None) is None:
            self._unified_engine = await get_unified_engine()

        root_text, top_summaries = await asyncio.gather(
            load_root_text(),
            search_top_global_context_summaries(
                query,
                self.global_context_index_top_k,
                self._unified_engine.vector,
            ),
        )
        prelude = format_global_context_prelude(root_text, top_summaries)
        if not prelude:
            return ""
        return f"## Global context\n{prelude}"

    async def get_completion_from_context(
        self,
        query: Optional[str] = None,
        query_batch: Optional[List[str]] = None,
        retrieved_objects: Any = None,
        context: Optional[str] = None,
        effective_query: Optional[str] = None,
        turn_preparation=None,
    ) -> List[Any]:
        _reject_query_batch(query_batch)

        if self._use_session_cache():
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
                used_graph_element_ids=extract_context_object_ids(retrieved_objects),
                max_context_chars=getattr(self, "max_context_chars", None),
                effective_query=effective_query,
                turn_preparation=turn_preparation,
            )
            return [completion]

        completion = await generate_completion(
            query=query,
            context=context,
            user_prompt_path=self.user_prompt_path,
            system_prompt_path=self.system_prompt_path,
            system_prompt=self.system_prompt,
            response_model=self.response_model,
        )
        return [completion]

    async def get_completion(
        self, query: Optional[str] = None, query_batch: Optional[List[str]] = None
    ) -> List[Any]:
        _reject_query_batch(query_batch)
        validate_retriever_input(query, None, self._use_session_cache())

        retrieved_objects = await self.get_retrieved_objects(query=query)
        context = await self.get_context_from_objects(
            query=query,
            retrieved_objects=retrieved_objects,
        )
        return await self.get_completion_from_context(
            query=query,
            retrieved_objects=retrieved_objects,
            context=context,
        )


def _reject_query_batch(query_batch: Optional[List[str]]) -> None:
    if query_batch is not None:
        raise QueryValidationError("HYBRID_COMPLETION does not support query_batch.")
