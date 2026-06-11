import asyncio
from typing import Any, Dict, List, Optional, Type
from uuid import UUID, uuid5

from cognee.context_global_variables import session_user
from cognee.infrastructure.databases.cache.config import CacheConfig
from cognee.infrastructure.databases.unified import get_unified_engine
from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.bm25_retriever import BM25ChunksRetriever
from cognee.modules.retrieval.exceptions.exceptions import NoDataError, QueryValidationError
from cognee.modules.retrieval.utils.completion import generate_completion
from cognee.modules.retrieval.utils.global_context import (
    format_global_context_prelude,
    load_root_text,
    search_top_global_context_summaries,
)
from cognee.modules.retrieval.utils.validate_queries import validate_retriever_input
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
        chunk_objects, entity_hits = await asyncio.gather(
            self._build_chunk_objects(query),
            self._search_collection("Entity_name", query, self.entities_top_k),
        )
        entities = await self._build_entities(entity_hits)

        return {**chunk_objects, "entities": entities}

    async def _build_chunk_objects(self, query: str) -> Dict[str, Any]:
        summary_limit = self._summary_candidate_limit()
        if summary_limit <= 0:
            return {"chunks": await self._build_legacy_chunks(query), "chunk_summaries": {}}

        candidate_limit = max(0, self.chunks_top_k * 2)
        bm25_chunks, vector_chunks, summary_hits = await asyncio.gather(
            self._search_bm25_chunks(query, limit=candidate_limit),
            self._search_collection(
                "DocumentChunk_text",
                query,
                candidate_limit,
                required=True,
            ),
            self._search_collection(
                "TextSummary_text",
                query,
                summary_limit,
                apply_node_filter=False,
            ),
        )
        chunks, chunk_summaries = await self._rank_summary_aware_chunks(
            bm25_chunks,
            vector_chunks,
            summary_hits,
        )
        await self._fetch_final_chunk_summaries(chunks, chunk_summaries)
        return {"chunks": chunks, "chunk_summaries": chunk_summaries}

    async def _build_legacy_chunks(self, query: str) -> List[Any]:
        bm25_chunks, vector_chunks = await asyncio.gather(
            self._search_bm25_chunks(query),
            self._search_collection("DocumentChunk_text", query, self.chunks_top_k, required=True),
        )
        return _merge_chunks(bm25_chunks, vector_chunks, self.chunks_top_k)

    def _summary_candidate_limit(self) -> int:
        if self.text_summaries_top_k is None:
            return max(0, self.chunks_top_k)
        return self.text_summaries_top_k

    async def _search_bm25_chunks(self, query: str, limit: Optional[int] = None) -> List[dict]:
        bm25_top_k = limit if limit is not None else _bm25_chunk_limit(self.chunks_top_k)
        if bm25_top_k <= 0:
            return []

        try:
            retriever = BM25ChunksRetriever(top_k=bm25_top_k, with_scores=True)
            scored_chunks = await retriever.get_retrieved_objects(query)
        except NoDataError:
            return []
        except Exception as error:
            logger.warning("BM25 chunk retrieval failed; using vector chunks only: %s", error)
            return []

        chunks = []
        for item in scored_chunks:
            chunk, score = _scored_payload(item)
            if score <= 0:
                continue
            if not isinstance(chunk, dict):
                continue
            if not _payload_matches_node_filter(
                chunk, self.node_name, self.node_name_filter_operator
            ):
                continue
            chunks.append(chunk)
        return chunks

    async def _search_collection(
        self,
        collection_name: str,
        query: str,
        limit: int,
        *,
        required: bool = False,
        apply_node_filter: bool = True,
    ) -> List[Any]:
        if limit <= 0:
            return []

        node_name = self.node_name if apply_node_filter else None
        node_name_filter_operator = self.node_name_filter_operator if apply_node_filter else "OR"
        try:
            return await self._unified_engine.vector.search(
                collection_name,
                query,
                limit=limit,
                include_payload=True,
                node_name=node_name,
                node_name_filter_operator=node_name_filter_operator,
            )
        except CollectionNotFoundError as error:
            if required:
                logger.error("%s collection not found", collection_name)
                raise NoDataError("No data found in the system, please add data first.") from error
            logger.debug("%s collection not found; using empty channel", collection_name)
            return []

    async def _rank_summary_aware_chunks(
        self,
        bm25_chunks: List[Any],
        vector_chunks: List[Any],
        summary_hits: List[Any],
    ) -> tuple[List[Any], Dict[str, str]]:
        summary_ranks, summary_text_by_chunk_id = _summary_signal(summary_hits)

        candidate_by_key = {}
        _add_chunk_candidates(candidate_by_key, bm25_chunks)
        _add_chunk_candidates(candidate_by_key, vector_chunks)

        missing_chunk_ids = [
            key[1] for key in summary_ranks if key not in candidate_by_key and key[0] == "id"
        ]
        if missing_chunk_ids:
            _add_chunk_candidates(
                candidate_by_key,
                await self._fetch_summary_only_chunks(missing_chunk_ids),
            )

        bm25_ranks = _chunk_ranks(bm25_chunks)
        vector_ranks = _chunk_ranks(vector_chunks)
        ranked = _rank_chunk_candidates(
            candidate_by_key,
            [bm25_ranks, vector_ranks, summary_ranks],
            self.chunks_top_k,
            self.use_importance_weight,
        )
        ranked_keys = {_chunk_key(chunk) for chunk in ranked}
        final_summary_text = {
            chunk_id: text
            for chunk_id, text in summary_text_by_chunk_id.items()
            if ("id", chunk_id) in ranked_keys
        }
        return ranked, final_summary_text

    async def _fetch_summary_only_chunks(self, chunk_ids: List[str]) -> List[Any]:
        chunks = await self._unified_engine.vector.retrieve("DocumentChunk_text", chunk_ids)
        return [
            chunk
            for chunk in chunks
            if _payload_matches_node_filter(
                _payload(chunk), self.node_name, self.node_name_filter_operator
            )
        ]

    async def _fetch_final_chunk_summaries(
        self,
        chunks: List[Any],
        summary_text_by_chunk_id: Dict[str, str],
    ) -> None:
        summary_ids_by_chunk_id = {}
        for chunk in chunks:
            chunk_id = _result_id(chunk)
            if not chunk_id or chunk_id in summary_text_by_chunk_id:
                continue
            try:
                chunk_uuid = UUID(chunk_id)
            except (TypeError, ValueError):
                logger.debug("Cannot fetch paired TextSummary for non-UUID chunk id %s", chunk_id)
                continue
            summary_ids_by_chunk_id[chunk_id] = str(uuid5(chunk_uuid, "TextSummary"))

        if not summary_ids_by_chunk_id:
            return

        summaries = await self._unified_engine.vector.retrieve(
            "TextSummary_text",
            list(summary_ids_by_chunk_id.values()),
        )
        summaries_by_id = {_result_id(summary): summary for summary in summaries}
        for chunk_id, summary_id in summary_ids_by_chunk_id.items():
            summary = summaries_by_id.get(summary_id)
            if summary is None:
                logger.debug("No paired TextSummary found for chunk id %s", chunk_id)
                continue
            summary_text = _display_value(_payload(summary).get("text"))
            if summary_text:
                summary_text_by_chunk_id[chunk_id] = summary_text

    async def _build_entities(self, entity_hits: List[Any]) -> List[dict]:
        if not entity_hits:
            return []

        entities = [_entity_from_result(result) for result in entity_hits]
        if await self._unified_engine.graph.is_empty():
            return entities

        connections_by_entity = await asyncio.gather(
            *[self._unified_engine.graph.get_connections(entity["id"]) for entity in entities]
        )
        for entity, connections in zip(entities, connections_by_entity):
            entity["edges"] = _edge_bullets_from_connections(connections, self.max_edges_per_entity)
        return entities

    async def get_context_from_objects(
        self,
        query: Optional[str] = None,
        query_batch: Optional[List[str]] = None,
        retrieved_objects: Any = None,
    ) -> str:
        _reject_query_batch(query_batch)

        retrieved_objects = retrieved_objects or {}
        sections = []

        global_context = await self._build_global_context_section(query)
        if global_context:
            sections.append(global_context)

        passages = _format_passages(
            retrieved_objects.get("chunks", []),
            retrieved_objects.get("chunk_summaries", {}),
        )
        if passages:
            sections.append(passages)

        entities = _format_entities(retrieved_objects.get("entities", []))
        if entities:
            sections.append(entities)

        return "\n\n".join(sections)

    async def _build_global_context_section(self, query: Optional[str]) -> str:
        if not self.include_global_context_index or not query:
            return ""

        if getattr(self, "_unified_engine", None) is None:
            self._unified_engine = await get_unified_engine()

        root_text = await load_root_text()
        top_summaries = await search_top_global_context_summaries(
            query,
            self.global_context_index_top_k,
            self._unified_engine.vector,
        )
        prelude = format_global_context_prelude(root_text, top_summaries)
        if not prelude:
            return ""
        return f"## Global context\n{prelude}"

    def _extract_context_object_ids(self, retrieved_objects: Any) -> Optional[Dict[str, List[str]]]:
        if not isinstance(retrieved_objects, dict):
            return None

        node_ids = set()
        for chunk in retrieved_objects.get("chunks", []):
            result_id = _result_id(chunk)
            if result_id:
                node_ids.add(result_id)

        for entity in retrieved_objects.get("entities", []):
            if not isinstance(entity, dict):
                continue
            entity_id = _display_value(entity.get("id"))
            if entity_id:
                node_ids.add(entity_id)
            for edge in entity.get("edges", []):
                if not isinstance(edge, dict):
                    continue
                for key in ("source_id", "target_id"):
                    edge_node_id = _display_value(edge.get(key))
                    if edge_node_id:
                        node_ids.add(edge_node_id)

        return {"node_ids": sorted(node_ids)} if node_ids else None

    async def get_completion_from_context(
        self,
        query: Optional[str] = None,
        query_batch: Optional[List[str]] = None,
        retrieved_objects: Any = None,
        context: Optional[str] = None,
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
                used_graph_element_ids=self._extract_context_object_ids(retrieved_objects),
                max_context_chars=getattr(self, "max_context_chars", None),
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


def _payload(result: Any) -> dict:
    if isinstance(result, dict):
        return result
    payload = getattr(result, "payload", None)
    return payload if isinstance(payload, dict) else {}


def _display_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool, UUID)):
        text = str(value).strip()
        return text or None
    return None


def _result_id(result: Any) -> Optional[str]:
    payload = _payload(result)
    return _display_value(payload.get("id")) or _display_value(getattr(result, "id", None))


def _scored_payload(item: Any) -> tuple[Any, float]:
    if not isinstance(item, (list, tuple)) or len(item) != 2:
        return item, 0.0
    payload, score = item
    if not isinstance(score, (int, float)):
        return payload, 0.0
    return payload, float(score)


def _bm25_chunk_limit(chunks_top_k: int) -> int:
    if chunks_top_k <= 0:
        return 0
    return max(1, chunks_top_k // 2)


def _merge_chunks(bm25_chunks: List[Any], vector_chunks: List[Any], limit: int) -> List[Any]:
    if limit <= 0:
        return []

    chunks = []
    seen = set()
    for chunk in [*(bm25_chunks or []), *(vector_chunks or [])]:
        key = _chunk_key(chunk)
        if not key or key in seen:
            continue
        seen.add(key)
        chunks.append(chunk)
        if len(chunks) >= limit:
            break
    return chunks


def _add_chunk_candidates(candidate_by_key: dict, chunks: List[Any]) -> None:
    for chunk in chunks or []:
        key = _chunk_key(chunk)
        if key and key not in candidate_by_key:
            candidate_by_key[key] = chunk


def _chunk_ranks(chunks: List[Any]) -> dict:
    ranks = {}
    for rank, chunk in enumerate(chunks or []):
        key = _chunk_key(chunk)
        if key and key not in ranks:
            ranks[key] = rank
    return ranks


def _summary_signal(summary_hits: List[Any]) -> tuple[dict, Dict[str, str]]:
    ranks = {}
    summary_text_by_chunk_id = {}
    for rank, summary in enumerate(summary_hits or []):
        payload = _payload(summary)
        chunk_id = _display_value(payload.get("source_chunk_id"))
        if not chunk_id:
            continue

        key = ("id", chunk_id)
        if key not in ranks:
            ranks[key] = rank

        summary_text = _display_value(payload.get("text"))
        if summary_text and chunk_id not in summary_text_by_chunk_id:
            summary_text_by_chunk_id[chunk_id] = summary_text

    return ranks, summary_text_by_chunk_id


def _rank_chunk_candidates(
    candidate_by_key: dict,
    channel_ranks: List[dict],
    limit: int,
    use_importance_weight: bool,
) -> List[Any]:
    if limit <= 0:
        return []

    rrf_k = _rrf_k(limit)
    ranked = []
    for key, chunk in candidate_by_key.items():
        ranks = [ranks[key] for ranks in channel_ranks if key in ranks]
        if not ranks:
            continue

        rrf_score = sum(1.0 / (rrf_k + rank + 1) for rank in ranks)
        final_score = rrf_score
        if use_importance_weight:
            final_score *= _importance_factor(chunk)

        chunk_id = _result_id(chunk) or ""
        ranked.append((final_score, rrf_score, min(ranks), chunk_id, chunk))

    ranked.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
    return [chunk for *_, chunk in ranked[:limit]]


def _rrf_k(chunks_top_k: int) -> int:
    return max(30, min(60, 20 + 2 * chunks_top_k))


def _importance_factor(chunk: Any) -> float:
    raw_importance = _payload(chunk).get("importance_weight")
    importance = raw_importance if isinstance(raw_importance, (int, float)) else 0.5
    importance = max(0.0, min(1.0, importance))
    return 0.75 + 0.5 * importance


def _chunk_key(chunk: Any) -> Optional[tuple[str, str]]:
    chunk_id = _result_id(chunk)
    if chunk_id:
        return ("id", chunk_id)
    text = _display_value(_payload(chunk).get("text"))
    if text:
        return ("text", text)
    return None


def _payload_matches_node_filter(
    payload: dict, node_name: Optional[List[str]], node_name_filter_operator: str
) -> bool:
    if not node_name:
        return True

    belongs_to_set = payload.get("belongs_to_set")
    if not isinstance(belongs_to_set, list):
        return False

    payload_sets = {str(name) for name in belongs_to_set}
    requested_sets = {str(name) for name in node_name}
    if node_name_filter_operator == "AND":
        return requested_sets.issubset(payload_sets)
    return bool(payload_sets & requested_sets)


def _first_display_value(*values: Any) -> Optional[str]:
    for value in values:
        text = _display_value(value)
        if text:
            return text
    return None


def _entity_from_result(result: Any) -> dict:
    payload = _payload(result)
    entity_id = _result_id(result) or ""
    return {
        "id": entity_id,
        "name": _first_display_value(payload.get("name"), payload.get("text"), entity_id) or "",
        "description": _display_value(payload.get("description")),
        "type": _entity_type(payload),
        "edges": [],
    }


def _entity_type(payload: dict) -> Optional[str]:
    for value in (payload.get("is_a"), payload.get("type")):
        entity_type = _display_value(value)
        if entity_type and entity_type not in {"IndexSchema"}:
            return entity_type
    return None


def _edge_bullets_from_connections(connections: List[Any], max_edges: int) -> List[dict]:
    if max_edges <= 0:
        return []

    edges = []
    seen_keys = set()
    seen_texts = set()
    for connection in connections or []:
        unpacked = _unpack_connection(connection)
        if unpacked is None:
            continue

        source, edge, target = unpacked
        bullet = _edge_bullet(source, edge, target)
        if not bullet:
            continue

        dedupe_key = _edge_dedupe_key(bullet)
        if dedupe_key and dedupe_key in seen_keys:
            continue
        if not dedupe_key and bullet["text"] in seen_texts:
            continue

        if dedupe_key:
            seen_keys.add(dedupe_key)
        else:
            seen_texts.add(bullet["text"])
        edges.append(bullet)
    edges.sort(key=lambda edge: 0 if _is_type_edge(edge) else 1)
    return edges[:max_edges]


def _unpack_connection(connection: Any) -> Optional[tuple[dict, dict, dict]]:
    if not isinstance(connection, (list, tuple)) or len(connection) != 3:
        return None
    source, edge, target = connection
    if not isinstance(source, dict) or not isinstance(edge, dict) or not isinstance(target, dict):
        return None
    return source, edge, target


def _edge_bullet(source: dict, edge: dict, target: dict) -> Optional[dict]:
    source_label = _node_label(source)
    target_label = _node_label(target)
    relationship = _display_value(edge.get("relationship_name"))
    text = _first_display_value(edge.get("edge_text"), _nested_edge_text(edge))
    if not text and source_label and relationship and target_label:
        text = f"{source_label} -- {relationship} -- {target_label}"
    if not text:
        return None

    return {
        "text": text,
        "source": source_label,
        "target": target_label,
        "source_id": _display_value(source.get("id")),
        "relationship": relationship,
        "target_id": _display_value(target.get("id")),
    }


def _edge_dedupe_key(edge: dict) -> Optional[tuple[str, str, str]]:
    source_id = _display_value(edge.get("source_id"))
    relationship = _display_value(edge.get("relationship"))
    target_id = _display_value(edge.get("target_id"))
    if source_id and relationship and target_id:
        return source_id, relationship, target_id
    return None


def _is_type_edge(edge: dict) -> bool:
    relationship = _display_value(edge.get("relationship"))
    if relationship:
        normalized = relationship.lower().replace("_", " ").replace("-", " ").strip()
        if normalized == "is a":
            return True

    text = _display_value(edge.get("text"))
    return bool(text and " is a " in f" {text.lower()} ")


def _nested_edge_text(edge: dict) -> Optional[str]:
    properties = edge.get("properties")
    if not isinstance(properties, dict):
        return None
    return _display_value(properties.get("edge_text"))


def _node_label(node: dict) -> Optional[str]:
    return _first_display_value(node.get("name"), node.get("id"))


def _format_passages(chunks: List[Any], chunk_summaries: Optional[Dict[str, str]] = None) -> str:
    texts = []
    chunk_summaries = chunk_summaries or {}
    for chunk in chunks or []:
        text = _display_value(_payload(chunk).get("text"))
        if not text:
            continue

        chunk_id = _result_id(chunk)
        summary_text = chunk_summaries.get(chunk_id) if chunk_id else None
        if summary_text:
            texts.append(f"[Passage Summary]: {summary_text}\n[Raw Passage]: {text}")
        else:
            texts.append(text)
    if not texts:
        return ""
    return "## Relevant passages\n" + "\n---\n".join(texts)


def _format_entities(entities: List[dict]) -> str:
    blocks = []
    for entity in entities or []:
        block = _format_entity(entity)
        if block:
            blocks.append(block)
    if not blocks:
        return ""
    return "## Relevant entities\n" + "\n\n".join(blocks)


def _format_entity(entity: dict) -> str:
    name = _display_value(entity.get("name"))
    if not name:
        return ""

    entity_type = _entity_type(entity)
    header = f"### {name} ({entity_type})" if entity_type else f"### {name}"

    lines = [header]
    description = _display_value(entity.get("description"))
    if description:
        lines.append(description)

    for edge in entity.get("edges", []):
        edge_text = _display_value(edge.get("text"))
        if edge_text:
            lines.append(f"- {edge_text}")

    return "\n".join(lines)
