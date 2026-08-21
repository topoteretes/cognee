import asyncio
from typing import Any, Dict, List, Optional, Type

from cognee.context_global_variables import session_user
from cognee.infrastructure.databases.cache.config import CacheConfig
from cognee.infrastructure.databases.unified import get_unified_engine
from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.hybrid.chunks import retrieve_hybrid_chunks, search_collection
from cognee.modules.retrieval.hybrid.context import (
    extract_context_object_ids as extract_hybrid_object_ids,
    format_hybrid_context,
    format_hybrid_context_batch,
)
from cognee.modules.retrieval.hybrid.entities import build_entities, search_entities
from cognee.modules.retrieval.hybrid.facts import (
    edge_rank_by_id,
    resolve_facts_top_k,
    select_facts_for_entities,
)
from cognee.modules.retrieval.hybrid.merge import merge_hybrid_results
from cognee.modules.retrieval.hybrid.references import cite_hybrid_completions
from cognee.modules.retrieval.hybrid.results import empty_hybrid_result
from cognee.modules.retrieval.hybrid.truth import build_truth_context
from cognee.modules.retrieval.utils.completion import generate_completion, generate_completion_batch
from cognee.modules.retrieval.utils.global_context import (
    format_global_context_prelude,
    load_root_text,
    search_top_global_context_summaries,
)
from cognee.modules.retrieval.utils.validate_queries import validate_retriever_input
from cognee.shared.logging_utils import get_logger

logger = get_logger("HybridRetriever")

DEFAULT_HYBRID_LANE_TOP_K = 10


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
        include_references: bool = False,
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
        self.include_references = include_references
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
    ) -> Any:
        validate_retriever_input(query, query_batch, self._use_session_cache())
        self._unified_engine = await get_unified_engine()
        if await self._unified_engine.graph.is_empty():
            logger.warning("Search attempt on an empty knowledge graph")
            return (
                [empty_hybrid_result() for _ in query_batch]
                if query_batch
                else empty_hybrid_result()
            )
        if query_batch:
            return list(await asyncio.gather(*[self._retrieve_one(q) for q in query_batch]))
        return await self._retrieve_one(query)

    async def _retrieve_one(self, query: str) -> dict[str, Any]:
        query_embeddings = await self._unified_engine.vector.embedding_engine.embed_text([query])
        query_vector = query_embeddings[0]

        truth = await build_truth_context(
            self._unified_engine,
            query_vector,
            use_truth_weight=self.use_truth_weight,
            chunks_top_k=self.chunks_top_k,
            node_name=self.node_name,
            node_name_filter_operator=self.node_name_filter_operator,
        )

        chunk_objects, (entities, facts) = await asyncio.gather(
            retrieve_hybrid_chunks(
                vector_engine=self._unified_engine.vector,
                query=query,
                chunks_top_k=self.chunks_top_k,
                text_summaries_top_k=self.text_summaries_top_k,
                node_name=self.node_name,
                node_name_filter_operator=self.node_name_filter_operator,
                use_importance_weight=self.use_importance_weight,
                query_vector=query_vector,
                use_truth_weight=self.use_truth_weight,
                q_coords=truth.q_coords,
                truth_state_by_id=truth.truth_state_by_id,
                current_truth_epoch=truth.current_truth_epoch,
            ),
            self._retrieve_entities_and_facts(query, query_vector),
        )
        return {**chunk_objects, "entities": entities, "facts": facts}

    async def _retrieve_entities_and_facts(self, query: str, query_vector: list[float]) -> tuple:
        """Entity lane, run concurrently with the chunk lane so the graph round trip for
        edge bullets overlaps the chunk pipeline's ranking and summary loading."""
        max_ranked_bullets = self.entities_top_k * max(0, self.max_edges_per_entity)
        entity_hits, edge_hits = await asyncio.gather(
            search_entities(
                self._unified_engine.vector,
                query,
                self.entities_top_k,
                self.node_name,
                self.node_name_filter_operator,
                query_vector,
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
        entities, reachable_ids = await build_entities(
            self._unified_engine.graph,
            entity_hits,
            self.max_edges_per_entity,
            edge_rank_by_id(edge_hits),
            self.node_name,
            self.node_name_filter_operator,
        )
        node_scoped = bool(self.node_name)
        return entities, select_facts_for_entities(
            edge_hits,
            entities,
            reachable_ids,
            resolve_facts_top_k(
                entities,
                node_scoped=node_scoped,
                facts_top_k=self.facts_top_k,
                entity_edge_budget=max_ranked_bullets,
            ),
            node_scoped,
        )

    async def get_context_from_objects(
        self,
        query: Optional[str] = None,
        query_batch: Optional[List[str]] = None,
        retrieved_objects: Any = None,
    ) -> Any:
        if query_batch:
            global_contexts = await asyncio.gather(
                *[self._build_global_context_section(q) for q in query_batch]
            )
            return format_hybrid_context_batch(global_contexts, retrieved_objects)
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
        context: Any = None,
        effective_query: Optional[str] = None,
        turn_preparation=None,
    ) -> List[Any]:
        prompts = {
            "user_prompt_path": self.user_prompt_path,
            "system_prompt_path": self.system_prompt_path,
            "system_prompt": self.system_prompt,
            "response_model": self.response_model,
        }
        use_session = self._use_session_cache() and not query_batch
        if use_session:
            sm = get_session_manager()
            completion = await sm.generate_completion_with_session(
                session_id=self.session_id,
                query=query,
                context=context,
                summarize_context=False,
                used_graph_element_ids=extract_hybrid_object_ids(retrieved_objects),
                max_context_chars=getattr(self, "max_context_chars", None),
                effective_query=effective_query,
                turn_preparation=turn_preparation,
                **prompts,
            )
            completions = [completion]
        elif query_batch:
            completions = await generate_completion_batch(
                query_batch=query_batch, context=context, **prompts
            )
        else:
            completions = [await generate_completion(query=query, context=context, **prompts)]
        return await self.append_references(completions, retrieved_objects)

    async def append_references(self, completions: List[Any], retrieved_objects: Any) -> List[Any]:
        return cite_hybrid_completions(
            completions,
            retrieved_objects,
            enabled=self.include_references and self.response_model is str,
        )

    def merge_retrieved_objects(self, primary: Any, secondary: Any) -> Any:
        return merge_hybrid_results(
            primary,
            secondary,
            chunks_limit=self.chunks_top_k,
            entities_limit=self.entities_top_k,
            facts_limit=self.facts_top_k,
        )

    def extract_context_object_ids(self, retrieved_objects: Any) -> Optional[Dict[str, List[str]]]:
        return extract_hybrid_object_ids(retrieved_objects)

    async def get_completion(
        self, query: Optional[str] = None, query_batch: Optional[List[str]] = None
    ) -> List[Any]:
        validate_retriever_input(query, query_batch, self._use_session_cache())

        retrieved_objects = await self.get_retrieved_objects(query=query, query_batch=query_batch)
        context = await self.get_context_from_objects(
            query=query,
            query_batch=query_batch,
            retrieved_objects=retrieved_objects,
        )
        return await self.get_completion_from_context(
            query=query,
            query_batch=query_batch,
            retrieved_objects=retrieved_objects,
            context=context,
        )
