"""LangMem memory source.

Reads a LangMem export and yields COGX records. LangMem stores memories as
short derived facts (similar to mem0) and optionally extracts entities and
relations into a graph. This importer handles both.

Accepted shapes
---------------
A plain list of memory objects::

    [{"id": "...", "content": "...", "user_id": "...", ...}]

A dict with a wrapper key (``memories``, ``results``, or ``items``)::

    {"memories": [...]}

A dict with an optional graph section alongside memories::

    {
        "memories":  [{"id": "...", "content": "..."}],
        "entities":  [{"id": "...", "name": "...", "type": "..."}],
        "relations": [{"id": "...", "source": "...", "target": "...", "relation": "..."}]
    }

A file path (str or Path) pointing to any of the above as JSON.

Each memory becomes a :class:`COGXMemory`. Entities become :class:`COGXEntity`
and relations become :class:`COGXFact`, so the full graph is preserved when
running in ``preserve`` or ``hybrid`` mode.
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

_CONTENT_KEYS = ("content", "memory", "text", "data")


def _first_list(container: Dict[str, Any], *keys: str) -> List[Dict[str, Any]]:
    for key in keys:
        value = container.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


class LangMemSource(MemorySource):
    """Import memories from a LangMem export into Cognee.

    Args:
        data: A file path, a list of memory dicts, or a dict export from LangMem.
        mode: Import fidelity mode — ``"re-derive"`` (default), ``"preserve"``,
              or ``"hybrid"``. See :class:`MemorySource` for details.
    """

    source_system = "langmem"

    def __init__(
        self,
        data: Union[str, Path, List[Any], Dict[str, Any]],
        mode: str = "re-derive",
    ):
        super().__init__(mode=mode)
        self._data = data

    def _load_raw(self) -> Dict[str, Any]:
        data = self._data
        if isinstance(data, (str, Path)):
            data = json.loads(Path(data).read_text(encoding="utf-8"))

        if isinstance(data, list):
            return {"memories": [item for item in data if isinstance(item, dict)]}

        if isinstance(data, dict):
            # If the dict already has a graph section, return as-is.
            if any(key in data for key in ("entities", "relations")):
                return data
            # Otherwise look for a flat memory list under a known wrapper key.
            for key in ("memories", "results", "items"):
                if isinstance(data.get(key), list):
                    return data
            raise ValueError(
                "Unrecognized LangMem export shape: expected a list or a dict "
                "with a 'memories', 'results', or 'items' key."
            )

        raise ValueError("Unrecognized LangMem export: expected a list or dict.")

    async def records(self) -> AsyncIterator[COGXRecord]:
        data = self._load_raw()
        memories = _first_list(data, "memories", "results", "items")
        entities = _first_list(data, "entities", "nodes")
        relations = _first_list(data, "relations", "edges", "facts")

        # Memories → COGXMemory
        for index, item in enumerate(memories):
            content = next(
                (item[key] for key in _CONTENT_KEYS if isinstance(item.get(key), str)),
                None,
            )
            if not content or not content.strip():
                continue

            categories = item.get("categories") or item.get("tags") or []
            if isinstance(categories, str):
                categories = [categories]

            context = item.get("context") or item.get("namespace") or ""
            if context and context not in categories:
                categories = [context, *categories]

            yield COGXMemory(
                external_system=self.source_system,
                external_id=str(item.get("id") or f"langmem-{index}"),
                content=content,
                categories=[str(c) for c in categories],
                scope=COGXScope(
                    user_id=item.get("user_id"),
                    agent_id=item.get("agent_id"),
                    run_id=item.get("run_id"),
                    session_id=item.get("thread_id") or item.get("session_id"),
                ),
                created_at=parse_timestamp(item.get("created_at")),
                updated_at=parse_timestamp(item.get("updated_at")),
                metadata={"langmem_metadata": item.get("metadata")} if item.get("metadata") else {},
            )

        # Entities → COGXEntity
        for index, entity in enumerate(entities):
            name = entity.get("name")
            if not isinstance(name, str) or not name.strip():
                continue
            yield COGXEntity(
                external_system=self.source_system,
                external_id=str(entity.get("id") or f"langmem-entity-{index}"),
                name=name,
                entity_type=entity.get("type") or entity.get("label"),
                description=entity.get("description") or entity.get("summary"),
                attributes=entity.get("attributes") or {},
                created_at=parse_timestamp(entity.get("created_at")),
                scope=COGXScope(user_id=entity.get("user_id")),
            )

        # Relations → COGXFact
        for index, relation in enumerate(relations):
            subject = relation.get("source") or relation.get("source_id")
            obj = relation.get("target") or relation.get("target_id")
            predicate = relation.get("relation") or relation.get("type") or "relates_to"
            if not subject or not obj:
                continue
            yield COGXFact(
                external_system=self.source_system,
                external_id=str(relation.get("id") or f"langmem-relation-{index}"),
                subject_ref=str(subject),
                predicate=str(predicate),
                object_ref=str(obj),
                fact_text=relation.get("description") or relation.get("fact"),
                created_at=parse_timestamp(relation.get("created_at")),
                scope=COGXScope(user_id=relation.get("user_id")),
            )
