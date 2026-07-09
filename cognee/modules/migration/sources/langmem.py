"""LangMem memory source.

Reads a LangMem store dump (JSON export from a LangGraph BaseStore) and yields
COGX memory records. LangMem stores memories as structured items in namespaces;
each item has a ``key``, a ``value`` dict with ``{kind, content}``, and optional
timestamps. The dump is expected to be one of:

- a JSON list of store items (the ``list(store.search(...))`` shape):
  ``[{"namespace": ["memories", "user123"], "key": "abc", "value": {...}, ...}]``
- a JSON object of ``{namespace: [{key, value, ...}]}`` (grouped dump)
- a JSON object of ``{namespace: {key: value}}`` (simplified key-value map)
- already-parsed Python equivalents (for live-API integration with a
  LangGraph store client)

Each memory's ``content`` field becomes a :class:`COGXMemory`. If the memory
value contains ``entities`` or ``facts`` arrays (from structured schemas like
``PreferenceMemory`` with relationships), those are mapped to
:class:`COGXEntity` and :class:`COGXFact` respectively.
"""

import json
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Union

from cognee.modules.migration.cogx import (
    COGXEntity,
    COGXFact,
    COGXMemory,
    COGXRecord,
    COGXScope,
    parse_timestamp,
)
from cognee.modules.migration.sources.base import MemorySource


def _extract_content(value: Any) -> str:
    """Pull a textual representation from a LangMem memory value.

    LangMem values can be:
    - a plain string
    - ``{"kind": "Memory", "content": {"content": "text"}}`` (default Memory schema)
    - a Pydantic model dumped as a dict (custom schema)
    - ``{"kind": "Memory", "content": <string>}`` (simplified)

    For custom structured schemas the whole value dict is serialized
    as a JSON string so the LLM can reason about it during re-derive.
    """
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        return str(value) if value is not None else ""

    # Default LangMem Memory shape: {"kind": "Memory", "content": {"content": "..."}}
    content = value.get("content")
    if isinstance(content, dict):
        inner = content.get("content")
        if isinstance(inner, str) and inner.strip():
            return inner
        # Empty or whitespace-only content dict: return empty string to signal skip
        if isinstance(inner, str) and not inner.strip():
            return ""
        # Fallback: serialize the inner dict if it has meaningful keys
        non_empty = {k: v for k, v in content.items() if v}
        if non_empty:
            return json.dumps(non_empty)
        return ""
    if isinstance(content, str):
        return content.strip() or ""

    # For custom schemas, serialize the relevant fields
    # Look for common content-carrying key names
    for key in ("content", "text", "preference", "summary", "context", "description"):
        val = value.get(key)
        if isinstance(val, str) and val.strip():
            # If multiple fields carry meaning, include everything
            if key == "content":
                return val
            return json.dumps(value)

    # Fallback: serialize the whole value
    return json.dumps(value)


def _extract_user_id_from_namespace(namespace: Any) -> Union[str, None]:
    """Heuristic: the first path segment after 'memories' is often the user id.

    LangMem default namespace is ``("memories", "{langgraph_user_id}")``.
    This extracts the second segment as user_id.
    """
    if isinstance(namespace, list) and len(namespace) >= 2:
        second = namespace[1]
        if isinstance(second, str) and second.strip() and "{" not in second:
            return second
    return None


def _extract_entities(value: Any) -> List[Dict[str, Any]]:
    """Extract entities from a structured LangMem memory value.

    Some custom LangMem schemas embed entity information alongside facts.
    """
    if not isinstance(value, dict):
        return []
    entities = value.get("entities") or value.get("nodes") or []
    if isinstance(entities, dict):
        entities = [entities]
    if isinstance(entities, list):
        return [e for e in entities if isinstance(e, dict) and e.get("name")]
    return []


def _extract_facts(value: Any) -> List[Dict[str, Any]]:
    """Extract facts/relations from a structured LangMem memory value."""
    if not isinstance(value, dict):
        return []
    facts = value.get("facts") or value.get("relations") or value.get("edges") or []
    if isinstance(facts, dict):
        facts = [facts]
    if isinstance(facts, list):
        return [f for f in facts if isinstance(f, dict)]
    return []


class LangMemSource(MemorySource):
    """Reads a LangMem store dump and yields COGX memory records.

    LangMem is the long-term memory SDK from LangChain/LangGraph. Memories are
    stored in a ``BaseStore`` keyed by ``(namespace, key)`` pairs. A dump
    typically comes from ``list(store.search(...))`` producing a list of
    ``Item`` objects, each with ``namespace``, ``key``, ``value``,
    ``created_at``, and ``updated_at``.

    Args:
        data: Path to a JSON file, a JSON string, or an already-parsed list/dict.
        mode: Import fidelity mode (``"re-derive"``, ``"preserve"``, ``"hybrid"``).

    Example::

        await cognee.remember(
            LangMemSource("langmem_dump.json", mode="preserve")
        )
    """

    source_system = "langmem"

    def __init__(self, data: Union[str, Path, List[Any], Dict[str, Any]], mode: str = "re-derive"):
        super().__init__(mode=mode)
        self._data = data

    def _load_raw(self) -> List[Dict[str, Any]]:
        """Normalise the dump into a flat list of item dicts."""
        data = self._data
        if isinstance(data, (str, Path)):
            data = json.loads(Path(data).read_text(encoding="utf-8"))

        items: List[Dict[str, Any]] = []

        if isinstance(data, list):
            items = [item for item in data if isinstance(item, dict)]
        elif isinstance(data, dict):
            # Detect if this is a collection wrapper ({"items": [...], "memories": [...]})
            # vs. a namespace-keyed dump ({"namespace1": [items], "namespace2": {key: value}})
            is_collection = any(
                isinstance(data.get(k), list) for k in ("items", "memories", "results")
            )

            if is_collection:
                for coll_key in ("items", "memories", "results"):
                    if isinstance(data.get(coll_key), list):
                        for item in data[coll_key]:
                            if isinstance(item, dict):
                                items.append(item)
            else:
                # Namespace-keyed dump: {"namespace_string": [items]} or
                # {"namespace_string": {user: [items]}} or
                # {"namespace_string": {key: value}}
                for key, value in data.items():
                    if isinstance(value, list):
                        for item in value:
                            if isinstance(item, dict):
                                if "namespace" not in item:
                                    item = {**item, "namespace": [key]}
                                items.append(item)
                    elif isinstance(value, dict):
                        for inner_key, inner_value in value.items():
                            if isinstance(inner_value, list):
                                # {"namespace": {"user": [items]}}
                                for sub_item in inner_value:
                                    if isinstance(sub_item, dict):
                                        if "namespace" not in sub_item:
                                            sub_item = {**sub_item, "namespace": [key, inner_key]}
                                        items.append(sub_item)
                            elif isinstance(inner_value, dict):
                                # {"namespace": {"key": value}}
                                items.append(
                                    {
                                        "namespace": [key],
                                        "key": inner_key,
                                        "value": inner_value,
                                    }
                                )
        return items

    async def records(self) -> AsyncIterator[COGXRecord]:
        for index, item in enumerate(self._load_raw()):
            namespace = item.get("namespace") or []
            key = item.get("key") or f"langmem-{index}"
            value = item.get("value") or item
            created_at = parse_timestamp(item.get("created_at"))
            updated_at = parse_timestamp(item.get("updated_at"))

            user_id: Union[str, None] = (
                item.get("user_id") or _extract_user_id_from_namespace(namespace)
            )

            # If value is logically nested (e.g. {"value": {...}}), unwrap once.
            if isinstance(value, dict) and "value" in value and isinstance(value["value"], dict):
                value = value["value"]

            scope = COGXScope(
                user_id=user_id,
                agent_id=item.get("agent_id"),
                session_id=item.get("session_id") or item.get("group_id"),
                run_id=item.get("run_id"),
            )

            content = _extract_content(value)
            if content:
                categories = item.get("categories") or []
                if isinstance(categories, str):
                    categories = [categories]
                yield COGXMemory(
                    external_system=self.source_system,
                    external_id=str(key),
                    content=content,
                    categories=[str(c) for c in categories],
                    scope=scope,
                    created_at=created_at,
                    updated_at=updated_at,
                    metadata={
                        "langmem_namespace": namespace if isinstance(namespace, list) else [str(namespace)],
                        "langmem_key": str(key),
                    },
                )

            # Yield entities if the value carries them (structured schemas)
            for entity in _extract_entities(value):
                name = entity.get("name")
                if not isinstance(name, str) or not name.strip():
                    continue
                yield COGXEntity(
                    external_system=self.source_system,
                    external_id=str(entity.get("id") or entity.get("external_id") or f"{key}-entity-{name}"),
                    name=name,
                    entity_type=entity.get("entity_type") or entity.get("type"),
                    description=entity.get("description") or entity.get("summary"),
                    aliases=entity.get("aliases") or [],
                    attributes=entity.get("attributes") or {},
                    scope=scope,
                    created_at=parse_timestamp(entity.get("created_at")),
                    metadata=entity.get("metadata") or {},
                )

            # Yield facts if the value carries them
            for fact in _extract_facts(value):
                subject = fact.get("subject_ref") or fact.get("subject") or fact.get("source")
                obj_ref = fact.get("object_ref") or fact.get("object") or fact.get("target")
                if not subject or not obj_ref:
                    continue
                yield COGXFact(
                    external_system=self.source_system,
                    external_id=str(fact.get("id") or fact.get("external_id") or f"{key}-fact-{subject}-{obj_ref}"),
                    subject_ref=str(subject),
                    predicate=str(fact.get("predicate") or fact.get("relation") or "relates_to"),
                    object_ref=str(obj_ref),
                    fact_text=fact.get("fact_text") or fact.get("fact"),
                    valid_at=parse_timestamp(fact.get("valid_at")),
                    invalid_at=parse_timestamp(fact.get("invalid_at") or fact.get("expired_at")),
                    confidence=fact.get("confidence"),
                    provenance=fact.get("provenance") or [],
                    scope=scope,
                    created_at=parse_timestamp(fact.get("created_at")),
                )
