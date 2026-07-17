"""LangMem memory source.

Reads a LangMem / LangGraph store export and yields COGX records. Accepts
JSON dumps produced by serializing ``store.search()`` / ``store.get()`` results:

- a plain JSON list of store items
- ``{"items": [...]}`` / ``{"memories": [...]}`` wrappers
- a file path or already-parsed Python list/dict

Each store item is expected to carry ``namespace``, ``key``, ``value``,
``created_at``, and ``updated_at`` (LangGraph ``Item`` shape). The ``value``
payload uses LangMem ``kind`` tags:

- semantic / ``Memory`` → :class:`COGXMemory`
- episodic / ``Episodic`` → :class:`COGXEpisode`
- procedural → :class:`COGXMemoryBlock`
- optional ``entities`` / ``facts`` arrays → :class:`COGXEntity` / :class:`COGXFact`
"""

import json
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from cognee.modules.migration.cogx import (
    COGXDocument,
    COGXEntity,
    COGXEpisode,
    COGXFact,
    COGXMemory,
    COGXMemoryBlock,
    COGXRecord,
    COGXScope,
    COGXTurn,
    parse_timestamp,
)
from cognee.modules.migration.sources.base import MemorySource


def _first_list(container: Dict[str, Any], *keys: str) -> List[Dict[str, Any]]:
    for key in keys:
        value = container.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _namespace_parts(namespace: Any) -> List[str]:
    if isinstance(namespace, tuple):
        namespace = list(namespace)
    if not isinstance(namespace, list):
        return []
    return [str(part) for part in namespace if part is not None and str(part).strip()]


def _message_text(message: Dict[str, Any]) -> str:
    content = message.get("content", message.get("text"))
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return ""


def _extract_semantic_text(content: Any) -> Optional[str]:
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, dict):
        nested = content.get("content")
        if isinstance(nested, str) and nested.strip():
            return nested.strip()
        if nested is None:
            lines = [f"{key}: {value}" for key, value in content.items() if value is not None]
            if lines:
                return "\n".join(lines)
    return None


class LangMemSource(MemorySource):
    source_system = "langmem"
    replayable = True

    def __init__(self, data: Union[str, Path, List[Any], Dict[str, Any]], mode: str = "re-derive"):
        super().__init__(mode=mode)
        self._data = data

    def _load_raw(self) -> List[Dict[str, Any]]:
        data = self._data
        if isinstance(data, (str, Path)):
            data = json.loads(Path(data).read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for key in ("items", "memories", "results"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
            else:
                raise ValueError(
                    "Unrecognized LangMem export: expected a list or a dict with "
                    "'items'/'memories'/'results'."
                )
        if not isinstance(data, list):
            raise ValueError("Unrecognized LangMem export: expected a list of store items.")
        return [item for item in data if isinstance(item, dict)]

    def _scope_from_item(self, item: Dict[str, Any]) -> COGXScope:
        parts = _namespace_parts(item.get("namespace"))
        user_id = parts[0] if parts else item.get("user_id")
        session_id = parts[1] if len(parts) > 1 else item.get("session_id")
        return COGXScope(
            user_id=str(user_id) if user_id is not None else None,
            session_id=str(session_id) if session_id is not None else None,
            agent_id=str(item["agent_id"]) if item.get("agent_id") is not None else None,
        )

    def _value_payload(self, item: Dict[str, Any]) -> Dict[str, Any]:
        value = item.get("value")
        if isinstance(value, dict):
            return value
        if any(key in item for key in ("kind", "content", "entities", "facts")):
            return item
        return {}

    async def records(self) -> AsyncIterator[COGXRecord]:
        for index, item in enumerate(self._load_raw()):
            value = self._value_payload(item)
            if not value:
                continue

            external_id = str(item.get("key") or item.get("id") or f"langmem-{index}")
            scope = self._scope_from_item(item)
            created_at = parse_timestamp(item.get("created_at"))
            updated_at = parse_timestamp(item.get("updated_at"))
            kind = str(value.get("kind") or "Memory").lower()
            content = value.get("content")

            for entity_index, entity in enumerate(
                _first_list(value, "entities", "nodes", "entity_nodes")
            ):
                name = entity.get("name")
                if not isinstance(name, str) or not name.strip():
                    continue
                labels = entity.get("labels") or entity.get("label") or []
                if isinstance(labels, str):
                    labels = [labels]
                entity_type = next((label for label in labels if label != "Entity"), None)
                yield COGXEntity(
                    external_system=self.source_system,
                    external_id=str(
                        entity.get("id")
                        or entity.get("uuid")
                        or f"{external_id}:entity:{entity_index}"
                    ),
                    name=name,
                    entity_type=entity_type,
                    description=entity.get("summary") or entity.get("description"),
                    attributes=entity.get("attributes") or {},
                    scope=scope,
                    created_at=parse_timestamp(entity.get("created_at")) or created_at,
                    updated_at=updated_at,
                )

            for fact_index, fact in enumerate(_first_list(value, "facts", "edges", "entity_edges")):
                subject_ref = (
                    fact.get("subject_ref") or fact.get("source_node_uuid") or fact.get("source")
                )
                object_ref = (
                    fact.get("object_ref") or fact.get("target_node_uuid") or fact.get("target")
                )
                if not subject_ref or not object_ref:
                    continue
                yield COGXFact(
                    external_system=self.source_system,
                    external_id=str(
                        fact.get("id") or fact.get("uuid") or f"{external_id}:fact:{fact_index}"
                    ),
                    subject_ref=str(subject_ref),
                    predicate=str(
                        fact.get("predicate")
                        or fact.get("name")
                        or fact.get("relation")
                        or "relates_to"
                    ),
                    object_ref=str(object_ref),
                    fact_text=fact.get("fact") or fact.get("fact_text"),
                    valid_at=parse_timestamp(fact.get("valid_at")),
                    invalid_at=parse_timestamp(fact.get("invalid_at") or fact.get("expired_at")),
                    scope=scope,
                    created_at=parse_timestamp(fact.get("created_at")) or created_at,
                    updated_at=updated_at,
                )

            if "episodic" in kind and isinstance(content, dict):
                turns = []
                for message in _first_list(content, "messages", "turns"):
                    text = _message_text(message)
                    role = str(message.get("role") or "unknown")
                    if not text.strip() or role in ("system", "tool"):
                        continue
                    turns.append(
                        COGXTurn(
                            role=role,
                            content=text,
                            occurred_at=parse_timestamp(
                                message.get("created_at") or message.get("timestamp")
                            ),
                        )
                    )
                if turns:
                    yield COGXEpisode(
                        external_system=self.source_system,
                        external_id=external_id,
                        turns=turns,
                        scope=scope,
                        created_at=created_at,
                        updated_at=updated_at,
                        metadata={"langmem_kind": value.get("kind")} if value.get("kind") else {},
                    )
                continue

            if "procedural" in kind:
                text = _extract_semantic_text(content) or (
                    content if isinstance(content, str) else None
                )
                if text:
                    yield COGXMemoryBlock(
                        external_system=self.source_system,
                        external_id=external_id,
                        label="procedural",
                        value=text,
                        scope=scope,
                        created_at=created_at,
                        updated_at=updated_at,
                        metadata={"langmem_kind": value.get("kind")} if value.get("kind") else {},
                    )
                continue

            if "profile" in kind and isinstance(content, dict):
                profile_text = _extract_semantic_text(content)
                if profile_text:
                    yield COGXDocument(
                        external_system=self.source_system,
                        external_id=external_id,
                        content=profile_text,
                        title=str(value.get("name") or "LangMem profile"),
                        scope=scope,
                        created_at=created_at,
                        updated_at=updated_at,
                        metadata={"langmem_kind": value.get("kind")} if value.get("kind") else {},
                    )
                continue

            text = _extract_semantic_text(content)
            if not text:
                continue

            categories = value.get("categories") or []
            if isinstance(categories, str):
                categories = [categories]

            yield COGXMemory(
                external_system=self.source_system,
                external_id=external_id,
                content=text,
                categories=[str(category) for category in categories],
                scope=scope,
                created_at=created_at,
                updated_at=updated_at,
                metadata={"langmem_kind": value.get("kind")} if value.get("kind") else {},
            )
