from typing import Any, Dict, List, Optional, Type
from uuid import UUID

from cognee.context_global_variables import session_user
from cognee.infrastructure.databases.cache.config import CacheConfig
from cognee.infrastructure.databases.unified import get_unified_engine
from cognee.infrastructure.databases.vector.exceptions import CollectionNotFoundError
from cognee.infrastructure.session.get_session_manager import get_session_manager
from cognee.modules.retrieval.base_retriever import BaseRetriever
from cognee.modules.retrieval.exceptions.exceptions import QueryValidationError
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
        chunks = await self._search_collection("DocumentChunk_text", query, self.chunks_top_k)
        entity_hits = await self._search_collection("Entity_name", query, self.entities_top_k)
        entities = await self._build_entities(entity_hits)

        return {"chunks": chunks, "entities": entities}

    async def _search_collection(self, collection_name: str, query: str, limit: int) -> List[Any]:
        try:
            return await self._unified_engine.vector.search(
                collection_name,
                query,
                limit=limit,
                include_payload=True,
                node_name=self.node_name,
                node_name_filter_operator=self.node_name_filter_operator,
            )
        except CollectionNotFoundError:
            logger.debug("%s collection not found; using empty channel", collection_name)
            return []

    async def _build_entities(self, entity_hits: List[Any]) -> List[dict]:
        if not entity_hits:
            return []

        graph_is_empty = await self._unified_engine.graph.is_empty()
        entities = []
        for result in entity_hits:
            entity = _entity_from_result(result)
            connections = []
            if not graph_is_empty:
                connections = await self._unified_engine.graph.get_connections(entity["id"])
            entity["edges"] = _edge_bullets_from_connections(connections, self.max_edges_per_entity)
            entities.append(entity)
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

        passages = _format_passages(retrieved_objects.get("chunks", []))
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
            entity_id = _display_value(entity.get("id")) if isinstance(entity, dict) else None
            if entity_id:
                node_ids.add(entity_id)
            for edge in entity.get("edges", []):
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
        "type": _first_display_value(payload.get("type"), payload.get("is_a")),
        "edges": [],
    }


def _edge_bullets_from_connections(connections: List[Any], max_edges: int) -> List[dict]:
    if max_edges <= 0:
        return []

    edges = []
    seen_texts = set()
    for connection in connections or []:
        unpacked = _unpack_connection(connection)
        if unpacked is None:
            continue

        source, edge, target = unpacked
        bullet = _edge_bullet(source, edge, target)
        if not bullet or bullet["text"] in seen_texts:
            continue

        seen_texts.add(bullet["text"])
        edges.append(bullet)
        if len(edges) >= max_edges:
            break
    return edges


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
        "target_id": _display_value(target.get("id")),
    }


def _nested_edge_text(edge: dict) -> Optional[str]:
    properties = edge.get("properties")
    if not isinstance(properties, dict):
        return None
    return _display_value(properties.get("edge_text"))


def _node_label(node: dict) -> Optional[str]:
    return _first_display_value(node.get("name"), node.get("id"))


def _format_passages(chunks: List[Any]) -> str:
    texts = []
    for chunk in chunks or []:
        text = _display_value(_payload(chunk).get("text"))
        if text:
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

    entity_type = _display_value(entity.get("type"))
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
