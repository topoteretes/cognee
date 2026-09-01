"""Mem0 memory source.

Reads a Mem0 export and yields COGX memory records. Accepts the shapes
produced by the Mem0 platform export API and by the OSS ``get_all()`` call:

- a plain JSON list of memory objects
- ``{"results": [...]}`` / ``{"memories": [...]}`` wrappers
- already-parsed Python lists/dicts (for live-API integration: fetch with the
  ``mem0ai`` client yourself and pass the response in)

Each memory becomes a :class:`COGXMemory` with scope taken from
``user_id``/``agent_id``/``run_id`` and timestamps preserved.
"""

import json
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Union

from cognee.modules.migration.cogx import COGXMemory, COGXRecord, COGXScope, parse_timestamp
from cognee.modules.migration.sources.base import MemorySource, read_export_file

_CONTENT_KEYS = ("memory", "text", "data", "content")


class Mem0Source(MemorySource):
    source_system = "mem0"

    def __init__(self, data: Union[str, Path, List[Any], Dict[str, Any]], mode: str = "re-derive"):
        super().__init__(mode=mode)
        self._data = data

    def _load_raw(self) -> List[Dict[str, Any]]:
        data = self._data
        if isinstance(data, (str, Path)):
            data = json.loads(read_export_file(data))
        if isinstance(data, dict):
            recognized = False
            for key in ("results", "memories", "items"):
                value = data.get(key)
                if not isinstance(value, list):
                    continue
                recognized = True
                # An accepted alias that is present but empty must not shadow a
                # populated one later in the list -- unwrapping on it imports
                # nothing at all. Same rule as sources/zep.py's _first_list.
                records = [item for item in value if isinstance(item, dict)]
                if records:
                    return records
            if not recognized:
                raise ValueError(
                    "Unrecognized Mem0 export shape: expected a list or a dict "
                    "with a 'results'/'memories' key."
                )
            return []
        if not isinstance(data, list):
            raise ValueError("Unrecognized Mem0 export shape: expected a list of memories.")
        return [item for item in data if isinstance(item, dict)]

    async def records(self) -> AsyncIterator[COGXRecord]:
        for index, item in enumerate(self._load_raw()):
            content = next(
                (item[key] for key in _CONTENT_KEYS if isinstance(item.get(key), str)), None
            )
            if not content:
                continue
            categories = item.get("categories") or []
            if isinstance(categories, str):
                categories = [categories]
            yield COGXMemory(
                external_system=self.source_system,
                external_id=str(item.get("id") or f"mem0-{index}"),
                content=content,
                categories=[str(category) for category in categories],
                scope=COGXScope(
                    user_id=item.get("user_id"),
                    agent_id=item.get("agent_id"),
                    run_id=item.get("run_id"),
                ),
                created_at=parse_timestamp(item.get("created_at")),
                updated_at=parse_timestamp(item.get("updated_at")),
                metadata={"mem0_metadata": item.get("metadata")} if item.get("metadata") else {},
            )
