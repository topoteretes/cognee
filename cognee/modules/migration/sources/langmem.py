"""LangMem memory source.

Reads a LangMem export (or LangGraph BaseStore dump) and yields COGX records.
Accepts a JSON list of items or a dict wrapping items.

Expected item shapes:
- LangGraph store style: {"namespace": [...], "key": "...", "value": {...}, "created_at": ...}
- Flat structure: {"id": "...", "content": "...", "entities": [...], "relations": [...]}

Each memory block maps to a COGXMemory. If entities or relations/facts are present,
they map to COGXEntity and COGXFact respectively.
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

_CONTENT_KEYS = ("text", "content", "memory", "data")


class LangMemSource(MemorySource):
    source_system = "langmem"

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
                    "Unrecognized LangMem export shape: expected a list or a dict "
                    "with an 'items' or 'memories' key."
                )
        if not isinstance(data, list):
            raise ValueError("Unrecognized LangMem export shape: expected a list of memories.")
        
        return [item for item in data if isinstance(item, dict)]

    async def records(self) -> AsyncIterator[COGXRecord]:
        for index, item in enumerate(self._load_raw()):
            # Handle LangGraph store item structure
            if "key" in item and "value" in item and isinstance(item["value"], dict):
                key = item.get("key")
                value = item["value"]
                namespace = item.get("namespace", [])
                
                # Try to extract user_id from namespace if possible
                user_id = None
                if isinstance(namespace, list) and len(namespace) > 0:
                    user_id = namespace[-1]  # Common pattern: ["memories", "user_123"]
                
                content = next((value[k] for k in _CONTENT_KEYS if isinstance(value.get(k), str)), None)
                entities = value.get("entities", [])
                relations = value.get("relations", []) or value.get("facts", [])
                
                created_at = parse_timestamp(item.get("created_at"))
                updated_at = parse_timestamp(item.get("updated_at"))
                scope = COGXScope(user_id=user_id)
                external_id_base = str(key or f"langmem-{index}")
            else:
                # Flat structure
                value = item
                content = next((item[k] for k in _CONTENT_KEYS if isinstance(item.get(k), str)), None)
                entities = item.get("entities", [])
                relations = item.get("relations", []) or item.get("facts", [])
                
                created_at = parse_timestamp(item.get("created_at"))
                updated_at = parse_timestamp(item.get("updated_at"))
                scope = COGXScope(
                    user_id=item.get("user_id"),
                    agent_id=item.get("agent_id"),
                    run_id=item.get("run_id"),
                    session_id=item.get("session_id") or item.get("thread_id"),
                )
                external_id_base = str(item.get("id") or item.get("uuid") or f"langmem-{index}")

            # 1. Yield Memory
            if content:
                yield COGXMemory(
                    external_system=self.source_system,
                    external_id=external_id_base,
                    content=content,
                    scope=scope,
                    created_at=created_at,
                    updated_at=updated_at,
                    metadata={"langmem_metadata": value.get("metadata")} if value.get("metadata") else {},
                )

            # 2. Yield Entities
            if isinstance(entities, list):
                for e_idx, entity in enumerate(entities):
                    if not isinstance(entity, dict):
                        continue
                    name = entity.get("name")
                    if not name:
                        continue
                        
                    e_id = entity.get("id") or f"{external_id_base}-ent-{e_idx}"
                    yield COGXEntity(
                        external_system=self.source_system,
                        external_id=str(e_id),
                        name=str(name),
                        entity_type=entity.get("type") or entity.get("entity_type"),
                        description=entity.get("description") or entity.get("summary"),
                        attributes=entity.get("attributes") or {},
                        scope=scope,
                        created_at=created_at,
                    )

            # 3. Yield Relations/Facts
            if isinstance(relations, list):
                for r_idx, relation in enumerate(relations):
                    if not isinstance(relation, dict):
                        continue
                    
                    subject = relation.get("subject") or relation.get("source")
                    predicate = relation.get("predicate") or relation.get("relation")
                    obj = relation.get("object") or relation.get("target")
                    
                    if not subject or not predicate or not obj:
                        continue
                        
                    r_id = relation.get("id") or f"{external_id_base}-rel-{r_idx}"
                    yield COGXFact(
                        external_system=self.source_system,
                        external_id=str(r_id),
                        subject_ref=str(subject),
                        predicate=str(predicate),
                        object_ref=str(obj),
                        fact_text=relation.get("fact_text") or relation.get("description"),
                        valid_at=parse_timestamp(relation.get("valid_at") or created_at),
                        scope=scope,
                        created_at=created_at,
                    )
