import asyncio
import re
from typing import Any, Dict, List, Optional, Type

from cognee.context_global_variables import current_dataset_id, session_user
from cognee.infrastructure.databases.cache.config import CacheConfig
from cognee.infrastructure.databases.unified import get_unified_engine
from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.graph_completion_retriever import GraphCompletionRetriever
from cognee.modules.retrieval.hybrid.chunks import retrieve_hybrid_chunks, search_collection
from cognee.modules.retrieval.hybrid.context import (
    extract_context_object_ids,
    format_hybrid_context,
)
from cognee.modules.retrieval.hybrid.entities import build_entities
from cognee.modules.retrieval.hybrid.facts import (
    edge_rank_by_id,
    graph_evidence_by_edge_type_id,
    select_facts,
)
from cognee.modules.retrieval.hybrid.results import result_id
from cognee.modules.retrieval.utils.completion import generate_completion, generate_completion_batch
from cognee.modules.retrieval.utils.references import append_chunk_evidence
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

DEFAULT_MAX_CONTEXT_CHARS = 32_000
DEFAULT_MAX_CONTEXT_ITEMS = 30


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
        system_prompt_path: str = "hybrid_answer_guarded.txt",
        system_prompt: Optional[str] = None,
        text_summaries_top_k: Optional[int] = None,
        use_importance_weight: bool = True,
        use_truth_weight: bool = False,
        facts_top_k: Optional[int] = 5,
        include_references: bool = False,
        max_context_chars: Optional[int] = DEFAULT_MAX_CONTEXT_CHARS,
        max_context_items: Optional[int] = DEFAULT_MAX_CONTEXT_ITEMS,
        graph_fallback_enabled: bool = True,
        graph_fallback_top_k: Optional[int] = None,
        graph_fallback_node_type: Optional[Type] = None,
        graph_fallback_wide_search_top_k: Optional[int] = 100,
        graph_fallback_triplet_distance_penalty: Optional[float] = 6.5,
        graph_fallback_feedback_influence: float = 0.0,
        graph_fallback_neighborhood_depth: Optional[int] = None,
        graph_fallback_neighborhood_seed_top_k: Optional[int] = 10,
    ):
        # A caller-supplied ``system_prompt`` intentionally overrides the
        # guarded default, matching the behavior of the other retrievers.
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
        self.include_references = include_references
        self.max_context_chars = max_context_chars
        self.max_context_items = max_context_items
        self.graph_fallback_enabled = graph_fallback_enabled
        self._graph_fallback = GraphCompletionRetriever(
            top_k=(
                graph_fallback_top_k if graph_fallback_top_k is not None else self.entities_top_k
            ),
            node_type=graph_fallback_node_type,
            node_name=node_name,
            node_name_filter_operator=node_name_filter_operator,
            wide_search_top_k=graph_fallback_wide_search_top_k,
            triplet_distance_penalty=graph_fallback_triplet_distance_penalty,
            feedback_influence=graph_fallback_feedback_influence,
            neighborhood_depth=graph_fallback_neighborhood_depth,
            neighborhood_seed_top_k=graph_fallback_neighborhood_seed_top_k,
            session_id=session_id,
            response_model=response_model,
        )

    def _use_session_cache(self) -> bool:
        user = session_user.get()
        user_id = getattr(user, "id", None)
        return bool(user_id and CacheConfig().caching)

    async def get_retrieved_objects(
        self, query: Optional[str] = None, query_batch: Optional[List[str]] = None
    ) -> Any:
        validate_retriever_input(query, query_batch, self._use_session_cache())

        self._unified_engine = await get_unified_engine()
        queries = [query] if query is not None else query_batch
        query_embeddings = await self._unified_engine.vector.embedding_engine.embed_text(queries)
        if len(query_embeddings) != len(queries):
            raise RuntimeError(
                "Embedding provider returned a different number of vectors than queries."
            )
        results = await asyncio.gather(
            *[
                self._get_retrieved_objects_for_query(current_query, query_vector)
                for current_query, query_vector in zip(queries, query_embeddings)
            ]
        )
        return results[0] if query is not None else results

    async def _get_retrieved_objects_for_query(
        self, query: str, query_vector: list[float]
    ) -> Dict[str, Any]:
        status = {
            "chunks": _channel_status("pending"),
            "entities": _channel_status("pending"),
            "facts": _channel_status("pending"),
            "global_context": _channel_status(
                "pending" if self.include_global_context_index else "skipped",
                "disabled" if not self.include_global_context_index else None,
            ),
        }

        q_coords, truth_state_by_id, current_truth_epoch = await self._build_truth_context(
            query_vector
        )

        max_ranked_bullets = self.entities_top_k * max(0, self.max_edges_per_entity)
        chunk_result, entity_result, edge_result = await asyncio.gather(
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
                q_coords=q_coords,
                truth_state_by_id=truth_state_by_id,
                current_truth_epoch=current_truth_epoch,
            ),
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
            return_exceptions=True,
        )

        if isinstance(chunk_result, BaseException):
            logger.warning("Hybrid chunk lane failed; continuing without chunks: %s", chunk_result)
            chunk_objects = {"chunks": [], "chunk_summaries": {}}
            status["chunks"] = _failed_channel_status(chunk_result)
        else:
            chunk_objects = chunk_result
            status["chunks"] = _channel_status("ok", item_count=len(chunk_objects["chunks"]))

        if isinstance(entity_result, BaseException):
            logger.warning(
                "Hybrid entity seed lane failed; continuing without entities: %s", entity_result
            )
            entity_hits = []
            entities = []
            status["entities"] = _failed_channel_status(entity_result)
        else:
            entity_hits = entity_result
            try:
                entities = await build_entities(
                    self._unified_engine.graph,
                    entity_hits,
                    self.max_edges_per_entity,
                    edge_rank_by_id([] if isinstance(edge_result, BaseException) else edge_result),
                    self.node_name,
                    self.node_name_filter_operator,
                )
                status["entities"] = _channel_status("ok", item_count=len(entities))
            except Exception as error:
                logger.warning(
                    "Hybrid entity expansion failed; continuing without entities: %s", error
                )
                entities = []
                status["entities"] = _failed_channel_status(error)

        if self.facts_top_k <= 0:
            facts = []
            status["facts"] = _channel_status("skipped", "disabled")
        elif self.node_name:
            facts = []
            status["facts"] = _channel_status("skipped", "scoped search")
        elif isinstance(edge_result, BaseException):
            logger.warning("Hybrid fact lane failed; continuing without facts: %s", edge_result)
            facts = []
            status["facts"] = _failed_channel_status(edge_result)
        else:
            facts = self._select_facts(edge_result, entities)
            status["facts"] = _channel_status("ok", item_count=len(facts))

        retrieved = {
            **chunk_objects,
            "entities": entities,
            "facts": facts,
            "retrieval_status": status,
        }
        if self.graph_fallback_enabled:
            await self._add_graph_fallback(query, retrieved)
        return retrieved

    async def _build_truth_context(self, query_vector: list[float]) -> tuple:
        """Truth-subspace alignment context for the chunk lane.

        Returns ``(q_coords, truth_state_by_id, current_truth_epoch)``. Values
        are ``None`` when the truth weight is off or centroid slots are absent,
        so ranking stays at exact baseline. Fails open to baseline on any error.
        """
        if not self.use_truth_weight:
            return None, None, None

        try:
            dataset_id = current_dataset_id.get()
            if dataset_id is None:
                return None, None, None

            centroids = await load_centroids(self._unified_engine.vector, str(dataset_id))
            if not centroids:
                return None, None, None

            centroid_vectors = [centroid.centroid for centroid in centroids]
            q_coords = pad_coords(align.query_coords(query_vector, centroid_vectors), DEFAULT_K)
            current_truth_epoch = max(centroid.truth_epoch for centroid in centroids)

            candidate_chunk_ids = await self._candidate_chunk_ids(query_vector)
            if not candidate_chunk_ids:
                return q_coords, {}, current_truth_epoch

            truth_state_by_id = await self._unified_engine.graph.get_node_truth_state(
                candidate_chunk_ids
            )
            return q_coords, truth_state_by_id, current_truth_epoch
        except Exception as error:
            logger.debug("Truth-subspace lookup failed; using baseline ranking: %s", error)
            return None, None, None

    async def _candidate_chunk_ids(self, query_vector: list[float]) -> list[str]:
        """Candidate DocumentChunk ids whose truth alignments we batch-fetch.

        Mirrors the chunk lane's vector candidate window so the truth coords map
        covers the chunks that ranking can surface."""
        candidate_limit = max(0, self.chunks_top_k * 2)
        chunk_hits = await search_collection(
            self._unified_engine.vector,
            "DocumentChunk_text",
            "",
            candidate_limit,
            self.node_name,
            self.node_name_filter_operator,
            query_vector=query_vector,
        )
        ids = []
        for hit in chunk_hits:
            chunk_id = result_id(hit)
            if chunk_id:
                ids.append(str(chunk_id))
        return ids

    async def _add_graph_fallback(self, query: str, retrieved: dict) -> None:
        """Populate graph-only evidence when standard DocumentChunks are unavailable."""
        status = retrieved["retrieval_status"]
        status["graph_fallback"] = _channel_status("pending")
        try:
            if await self._unified_engine.graph.is_empty():
                retrieved["graph_fallback"] = []
                status["graph_fallback"] = _channel_status("skipped", "empty graph")
                return

            self._graph_fallback._unified_engine = self._unified_engine
            triplets = await self._graph_fallback.get_triplets(query=query) or []
            retrieved["graph_fallback"] = triplets
            status["graph_fallback"] = _channel_status("ok", item_count=len(triplets))
        except Exception as error:
            logger.warning("Hybrid graph fallback failed; returning other lanes: %s", error)
            retrieved["graph_fallback"] = []
            status["graph_fallback"] = _failed_channel_status(error)

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
        return select_facts(
            edge_hits,
            bullet_ids,
            self.facts_top_k,
            evidence_by_id=graph_evidence_by_edge_type_id(entities),
            require_evidence=True,
        )

    async def get_context_from_objects(
        self,
        query: Optional[str] = None,
        query_batch: Optional[List[str]] = None,
        retrieved_objects: Any = None,
    ) -> Any:
        validate_retriever_input(query, query_batch, self._use_session_cache())
        if query_batch is not None:
            if not isinstance(retrieved_objects, list) or len(retrieved_objects) != len(
                query_batch
            ):
                raise ValueError("retrieved_objects must align with query_batch")
            return await asyncio.gather(
                *[
                    self._get_context_for_query(current_query, objects)
                    for current_query, objects in zip(query_batch, retrieved_objects)
                ]
            )
        return await self._get_context_for_query(query, retrieved_objects)

    async def _get_context_for_query(self, query: str, retrieved_objects: Any) -> str:
        global_context = await self._build_global_context_section(query, retrieved_objects)
        objects_for_context = dict(retrieved_objects or {})
        fallback_objects = objects_for_context.get("graph_fallback")
        if fallback_objects:
            try:
                objects_for_context[
                    "graph_fallback_context"
                ] = await self._graph_fallback.get_context_from_objects(
                    query=query,
                    retrieved_objects=fallback_objects,
                )
            except Exception as error:
                logger.warning("Hybrid graph fallback formatting failed: %s", error)
                _set_channel_status(
                    retrieved_objects,
                    "graph_fallback",
                    _failed_channel_status(error),
                )
        selected_chunk_ids: set[str] = set()
        context = format_hybrid_context(
            global_context,
            objects_for_context,
            max_context_chars=self.max_context_chars,
            max_context_items=self.max_context_items,
            selected_chunk_ids=selected_chunk_ids,
        )
        if isinstance(retrieved_objects, dict):
            retrieved_objects["context_selection"] = {"chunk_ids": sorted(selected_chunk_ids)}
        return context

    async def _build_global_context_section(
        self, query: Optional[str], retrieved_objects: Any = None
    ) -> str:
        if not self.include_global_context_index or not query:
            return ""

        if getattr(self, "_unified_engine", None) is None:
            self._unified_engine = await get_unified_engine()

        try:
            root_text, top_summaries = await asyncio.gather(
                load_root_text(),
                search_top_global_context_summaries(
                    query,
                    self.global_context_index_top_k,
                    self._unified_engine.vector,
                ),
            )
        except Exception as error:
            logger.warning("Hybrid global-context lane failed; continuing: %s", error)
            _set_channel_status(retrieved_objects, "global_context", _failed_channel_status(error))
            return ""
        prelude = format_global_context_prelude(root_text, top_summaries)
        _set_channel_status(
            retrieved_objects,
            "global_context",
            _channel_status("ok", item_count=len(top_summaries or [])),
        )
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
        validate_retriever_input(query, query_batch, self._use_session_cache())

        if query_batch is not None:
            if not isinstance(context, list) or len(context) != len(query_batch):
                raise ValueError("context must align with query_batch")
            if not isinstance(retrieved_objects, list) or len(retrieved_objects) != len(
                query_batch
            ):
                raise ValueError("retrieved_objects must align with query_batch")
            completions = await generate_completion_batch(
                query_batch=query_batch,
                context=[_escape_context_boundaries(item) for item in context],
                user_prompt_path=self.user_prompt_path,
                system_prompt_path=self.system_prompt_path,
                system_prompt=self.system_prompt,
                response_model=self.response_model,
            )
        elif self._use_session_cache():
            sm = get_session_manager()
            completion = await sm.generate_completion_with_session(
                session_id=self.session_id,
                query=query,
                context=_escape_context_boundaries(context),
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
            completions = [completion]
        else:
            completion = await generate_completion(
                query=query,
                context=_escape_context_boundaries(context),
                user_prompt_path=self.user_prompt_path,
                system_prompt_path=self.system_prompt_path,
                system_prompt=self.system_prompt,
                response_model=self.response_model,
            )
            completions = [completion]

        return self._append_chunk_references(completions, retrieved_objects)

    def _append_chunk_references(self, completions: List[Any], retrieved_objects: Any) -> List[Any]:
        enabled = self.include_references and self.response_model is str
        if isinstance(retrieved_objects, list):
            return [
                append_chunk_evidence(
                    [completion],
                    _context_chunks(objects),
                    enabled=enabled,
                )[0]
                for completion, objects in zip(completions, retrieved_objects)
            ]
        # Graph-only fallback evidence has no DocumentChunk provenance and is
        # therefore never converted into a fabricated chunk citation.
        chunks = _context_chunks(retrieved_objects)
        return append_chunk_evidence(completions, chunks, enabled=enabled)

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


def _channel_status(
    status: str, detail: Optional[str] = None, *, item_count: Optional[int] = None
) -> dict:
    channel = {"status": status}
    if detail:
        channel["detail"] = detail
    if item_count is not None:
        channel["item_count"] = item_count
    return channel


def _failed_channel_status(error: BaseException) -> dict:
    return _channel_status("degraded", type(error).__name__)


def _set_channel_status(retrieved_objects: Any, channel: str, status: dict) -> None:
    if not isinstance(retrieved_objects, dict):
        return
    statuses = retrieved_objects.setdefault("retrieval_status", {})
    if isinstance(statuses, dict):
        statuses[channel] = status


def _context_chunks(retrieved_objects: Any) -> list[Any]:
    if not isinstance(retrieved_objects, dict):
        return []
    chunks = retrieved_objects.get("chunks", [])
    selection = retrieved_objects.get("context_selection")
    selected_ids = selection.get("chunk_ids") if isinstance(selection, dict) else None
    if selected_ids is None:
        # Direct get_completion_from_context callers may provide already-built
        # context without using the Hybrid context builder.
        return chunks
    allowed_ids = {str(chunk_id) for chunk_id in selected_ids}
    return [chunk for chunk in chunks if result_id(chunk) in allowed_ids]


_CONTEXT_BOUNDARY_PATTERN = re.compile(
    r"<\s*/?\s*retrieved_context\b[^>]*>",
    flags=re.IGNORECASE,
)


def _escape_context_boundaries(context: Optional[str]) -> str:
    """Prevent retrieved text from forging the prompt's trusted boundary."""
    if not context:
        return ""
    return _CONTEXT_BOUNDARY_PATTERN.sub(
        lambda match: match.group(0).replace("<", "&lt;").replace(">", "&gt;"),
        context,
    )
