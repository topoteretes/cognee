"""LangChain memory source.

Reads memory held in LangChain components and yields COGX records:

- chat history (``BaseChatMessageHistory``-like, or a list of messages,
  or a JSON list of role/content dicts) -> :class:`COGXEpisode` of
  :class:`COGXTurn`s
- documents (a ``BaseRetriever``/``VectorStore`` ``docstore`` mapping, or a
  list of ``Document``-like objects, or a JSON list of page-content dicts) ->
  :class:`COGXDocument`
- typed relations from ``ConversationKGMemory`` (``NetworkxEntityGraph``
  triples) -> :class:`COGXEntity` + :class:`COGXFact`
- the entity store from ``ConversationEntityMemory`` (name -> summary)
  -> :class:`COGXEntity`

The source has zero hard dependency on ``langchain``: it duck-types every
input. Pass an actual LangChain object (``ConversationBufferMemory``,
``BaseChatMessageHistory``, ``VectorStore`` whose ``docstore.docs`` is a
mapping, etc.) or pre-extracted plain Python structures — both work, and
the unit tests exercise the plain-structure path.

Defaults to ``mode="re-derive"``: LangChain primitives are mostly chunks
and chat turns, lossy compared to Cognee's extracted graph. Pass
``mode="hybrid"`` when ``triples`` or ``entities`` are supplied so the
typed relations are preserved alongside re-derived content.
"""

import json
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Iterable, List, Mapping, Optional, Union

from cognee.modules.migration.cogx import (
    COGXDocument,
    COGXEntity,
    COGXEpisode,
    COGXFact,
    COGXRecord,
    COGXScope,
    COGXTurn,
    parse_timestamp,
)
from cognee.modules.migration.sources.base import MemorySource

_DOCUMENT_CONTENT_KEYS = ("page_content", "content", "text")
_MESSAGE_CONTENT_KEYS = ("content", "text")
_MESSAGE_ROLE_KEYS = ("role", "type")

# LangChain's BaseMessage subclasses report ``.type`` as one of
# {"human", "ai", "system", "tool", "function", "chat"}; normalize to the
# vendor-neutral roles COGX downstream readers expect.
_ROLE_ALIASES = {
    "human": "user",
    "ai": "assistant",
    "chat": "assistant",
}

_SKIPPED_ROLES = {"system", "tool", "function"}


def _coerce_dict(value: Any, fields: Iterable[str]) -> Dict[str, Any]:
    """Project an object (or dict) onto a subset of fields.

    Lets the source consume both dict payloads and live LangChain objects
    without importing the library: every attribute lookup is duck-typed.
    """
    if isinstance(value, Mapping):
        return {key: value.get(key) for key in fields}
    return {key: getattr(value, key, None) for key in fields}


def _first_value(container: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = container.get(key)
        if value not in (None, ""):
            return value
    return None


def _message_text(message: Any) -> str:
    """Extract message text from a dict or a LangChain ``BaseMessage``-like."""
    raw = _coerce_dict(message, _MESSAGE_CONTENT_KEYS + ("additional_kwargs",))
    content = _first_value(raw, *_MESSAGE_CONTENT_KEYS)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, Mapping):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def _message_role(message: Any) -> str:
    raw = _coerce_dict(message, _MESSAGE_ROLE_KEYS)
    role = _first_value(raw, *_MESSAGE_ROLE_KEYS) or "unknown"
    role = str(role).lower()
    return _ROLE_ALIASES.get(role, role)


def _message_timestamp(message: Any) -> Optional[Any]:
    raw = _coerce_dict(message, ("created_at", "timestamp", "occurred_at", "additional_kwargs"))
    direct = _first_value(raw, "created_at", "timestamp", "occurred_at")
    if direct is not None:
        return parse_timestamp(direct)
    extras = raw.get("additional_kwargs")
    if isinstance(extras, Mapping):
        return parse_timestamp(
            extras.get("created_at") or extras.get("timestamp") or extras.get("occurred_at")
        )
    return None


def _iter_messages(payload: Any) -> List[Any]:
    """Normalize a chat-history payload to a flat list of messages.

    Accepts a list of dicts/objects, an object with ``.messages`` (LangChain
    ``BaseChatMessageHistory``), or an object exposing ``chat_memory.messages``
    (LangChain ``ConversationBufferMemory``).
    """
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    messages = getattr(payload, "messages", None)
    if isinstance(messages, list):
        return messages
    chat_memory = getattr(payload, "chat_memory", None)
    if chat_memory is not None:
        nested = getattr(chat_memory, "messages", None)
        if isinstance(nested, list):
            return nested
    return []


def _iter_documents(payload: Any) -> List[Any]:
    """Normalize a documents payload to a flat list of document-like objects.

    Accepts a list of dicts/objects, a mapping of id->doc (e.g. an
    ``InMemoryDocstore.docs`` mapping), or an object exposing ``.docstore``
    (``VectorStore``) or ``.docstore.docs``. Returns a list of (id, doc)
    pairs preserved as tuples when the source carried explicit ids; falls
    back to single objects when no id was paired.
    """
    if payload is None:
        return []
    if isinstance(payload, Mapping):
        return [(str(key), value) for key, value in payload.items()]
    if isinstance(payload, list):
        return list(payload)
    docstore = getattr(payload, "docstore", None)
    if docstore is not None:
        docs = getattr(docstore, "docs", None) or getattr(docstore, "_dict", None)
        if isinstance(docs, Mapping):
            return [(str(key), value) for key, value in docs.items()]
    return []


def _document_content(document: Any) -> str:
    raw = _coerce_dict(document, _DOCUMENT_CONTENT_KEYS)
    content = _first_value(raw, *_DOCUMENT_CONTENT_KEYS)
    return content if isinstance(content, str) else ""


def _document_metadata(document: Any) -> Dict[str, Any]:
    raw = _coerce_dict(document, ("metadata",))
    metadata = raw.get("metadata")
    return dict(metadata) if isinstance(metadata, Mapping) else {}


def _document_id(document: Any, fallback_id: Optional[str], index: int) -> str:
    if fallback_id is not None:
        return fallback_id
    raw = _coerce_dict(document, ("id", "id_", "uuid", "doc_id"))
    explicit = _first_value(raw, "id", "id_", "uuid", "doc_id")
    if explicit is not None:
        return str(explicit)
    metadata = _document_metadata(document)
    for key in ("id", "source", "file_path", "filename"):
        if metadata.get(key):
            return f"{metadata[key]}:{index}"
    return f"langchain-doc-{index}"


def _iter_triples(payload: Any) -> List[Any]:
    """Normalize a KG-memory triples payload.

    Accepts a list of (s, p, o) tuples (``NetworkxEntityGraph.get_triples()``),
    a list of dicts with ``subject``/``predicate``/``object``, or any object
    exposing ``.get_triples()`` (``NetworkxEntityGraph``).
    """
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    if hasattr(payload, "get_triples"):
        try:
            return list(payload.get_triples())
        except Exception:  # pragma: no cover - defensive against partial mocks
            return []
    return []


def _triple_parts(triple: Any) -> Optional[tuple]:
    if isinstance(triple, Mapping):
        subject = triple.get("subject") or triple.get("source") or triple.get("s")
        predicate = triple.get("predicate") or triple.get("relation") or triple.get("p")
        obj = triple.get("object") or triple.get("target") or triple.get("o")
        if subject and predicate and obj:
            return str(subject), str(predicate), str(obj)
        return None
    if isinstance(triple, (list, tuple)) and len(triple) >= 3:
        return str(triple[0]), str(triple[1]), str(triple[2])
    return None


def _iter_entities(payload: Any) -> List[tuple]:
    """Normalize an entity-store payload to a list of (name, description) pairs.

    Accepts a mapping (``ConversationEntityMemory.entity_store.store``: a dict
    of name -> summary), a list of dicts, or any object exposing ``.store``
    (``InMemoryEntityStore``).
    """
    if payload is None:
        return []
    if isinstance(payload, Mapping):
        return [(str(name), value) for name, value in payload.items()]
    if isinstance(payload, list):
        out = []
        for item in payload:
            if isinstance(item, Mapping):
                name = item.get("name") or item.get("entity")
                if name:
                    description = (
                        item.get("description") or item.get("summary") or item.get("value")
                    )
                    out.append((str(name), description))
        return out
    store = getattr(payload, "store", None)
    if isinstance(store, Mapping):
        return [(str(name), value) for name, value in store.items()]
    return []


class LangChainMemorySource(MemorySource):
    source_system = "langchain"

    def __init__(
        self,
        data: Optional[Union[str, Path, Dict[str, Any]]] = None,
        *,
        messages: Any = None,
        documents: Any = None,
        triples: Any = None,
        entities: Any = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        mode: str = "re-derive",
    ):
        super().__init__(mode=mode)
        if data is not None and any(
            value is not None for value in (messages, documents, triples, entities)
        ):
            raise ValueError(
                "Pass either a single `data` payload or per-kind kwargs (messages, "
                "documents, triples, entities), not both."
            )
        self._raw_data = data
        self._messages = messages
        self._documents = documents
        self._triples = triples
        self._entities = entities
        self._scope = COGXScope(session_id=session_id, user_id=user_id, agent_id=agent_id)

    def _resolve_payload(self) -> Dict[str, Any]:
        if self._raw_data is None:
            return {
                "messages": self._messages,
                "documents": self._documents,
                "triples": self._triples,
                "entities": self._entities,
            }
        data = self._raw_data
        if isinstance(data, (str, Path)):
            data = json.loads(Path(data).read_text(encoding="utf-8"))
        if not isinstance(data, Mapping):
            raise ValueError(
                "LangChainMemorySource expects a mapping payload (messages/documents/"
                "triples/entities), a path to a JSON file with that shape, or per-kind kwargs."
            )
        payload = dict(data)
        scope_keys = ("session_id", "user_id", "agent_id")
        for key in scope_keys:
            value = payload.get(key)
            if value is not None and getattr(self._scope, key) in (None, ""):
                setattr(self._scope, key, str(value))
        return {
            "messages": payload.get("messages") or payload.get("chat_memory"),
            "documents": payload.get("documents") or payload.get("docs") or payload.get("docstore"),
            "triples": payload.get("triples")
            or payload.get("kg")
            or payload.get("knowledge_graph"),
            "entities": payload.get("entities") or payload.get("entity_store"),
        }

    async def records(self) -> AsyncIterator[COGXRecord]:
        payload = self._resolve_payload()

        messages = _iter_messages(payload["messages"])
        turns: List[COGXTurn] = []
        first_at = None
        last_at = None
        for message in messages:
            text = _message_text(message).strip()
            role = _message_role(message)
            if not text or role in _SKIPPED_ROLES:
                continue
            occurred_at = _message_timestamp(message)
            if occurred_at is not None:
                first_at = first_at or occurred_at
                last_at = occurred_at
            turns.append(COGXTurn(role=role, content=text, occurred_at=occurred_at))

        if turns:
            external_id = self._scope.session_id or "langchain:conversation"
            yield COGXEpisode(
                external_system=self.source_system,
                external_id=str(external_id),
                title="LangChain conversation history",
                turns=turns,
                created_at=first_at,
                updated_at=last_at,
                scope=self._scope,
            )

        seen_doc_ids: set = set()
        for index, entry in enumerate(_iter_documents(payload["documents"])):
            if isinstance(entry, tuple) and len(entry) == 2:
                fallback_id, document = entry
            else:
                fallback_id, document = None, entry
            content = _document_content(document).strip()
            if not content:
                continue
            external_id = _document_id(document, fallback_id, index)
            if external_id in seen_doc_ids:
                continue
            seen_doc_ids.add(external_id)
            metadata = _document_metadata(document)
            title = metadata.get("title") or metadata.get("source") or metadata.get("filename")
            yield COGXDocument(
                external_system=self.source_system,
                external_id=str(external_id),
                content=content,
                title=str(title) if title else None,
                mime_type=metadata.get("mime_type") or metadata.get("content_type"),
                created_at=parse_timestamp(metadata.get("created_at")),
                scope=self._scope,
                metadata={"langchain_metadata": metadata} if metadata else {},
            )

        entity_names: set = set()
        for name, description in _iter_entities(payload["entities"]):
            description_text = (
                description
                if isinstance(description, str)
                else (description.get("description") if isinstance(description, Mapping) else None)
            )
            yield COGXEntity(
                external_system=self.source_system,
                external_id=f"langchain:entity:{name}",
                name=name,
                description=str(description_text) if description_text else None,
                scope=self._scope,
            )
            entity_names.add(name)

        for index, triple in enumerate(_iter_triples(payload["triples"])):
            parts = _triple_parts(triple)
            if parts is None:
                continue
            subject, predicate, obj = parts
            for endpoint in (subject, obj):
                if endpoint not in entity_names:
                    yield COGXEntity(
                        external_system=self.source_system,
                        external_id=f"langchain:entity:{endpoint}",
                        name=endpoint,
                        scope=self._scope,
                    )
                    entity_names.add(endpoint)
            yield COGXFact(
                external_system=self.source_system,
                external_id=f"langchain:fact:{index}",
                subject_ref=f"langchain:entity:{subject}",
                predicate=predicate,
                object_ref=f"langchain:entity:{obj}",
                fact_text=f"{subject} {predicate} {obj}",
                scope=self._scope,
            )
