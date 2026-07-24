"""LangMem memory source.

Reads a LangMem export/dump and yields COGX memory records. Accepts the shapes
produced by LangMem memory storage:

- a plain JSON list of memory objects
- ``{"memories": [...]}`` / ``{"results": [...]}`` / ``{"data": [...]}`` wrappers
- already-parsed Python lists/dicts (for live-API integration: fetch with the
  LangMem client yourself and pass the response in)

Each memory becomes a :class:`COGXMemory`. LangMem memories are free-form text
with optional metadata, so they map cleanly onto COGX's atomic memory record.
"""

import json
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Union

from cognee.modules.migration.cogx import (
    COGXMemory,
    COGXRecord,
    COGXScope,
    parse_timestamp,
)
from cognee.modules.migration.sources.base import MemorySource

_CONTENT_KEYS = ("content", "text", "memory", "data", "message")


class LangMemSource(MemorySource):
    source_system = "langmem"

    def __init__(
        self,
        data: Union[str, Path, List[Any], Dict[str, Any]],
        mode: str = "re-derive",
    ):
        super().__init__(mode=mode)
        self._data = data

    def _load_raw(self) -> List[Dict[str, Any]]:
        data = self._data
        if isinstance(data, (str, Path)):
            data = json.loads(Path(data).read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for key in ("memories", "results", "items", "data"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
            else:
                raise ValueError(
                    "Unrecognized LangMem export shape: expected a list or a dict "
                    "with a 'memories'/'results' key."
                )
        if not isinstance(data, list):
            raise ValueError("Unrecognized LangMem export shape: expected a list of memories.")
        return [item for item in data if isinstance(item, dict)]

    async def records(self) -> AsyncIterator[COGXRecord]:
        for index, item in enumerate(self._load_raw()):
            content = next(
                (item[key] for key in _CONTENT_KEYS if isinstance(item.get(key), str)),
                None,
            )
            if not content:
                continue
            categories = item.get("categories") or []
            if isinstance(categories, str):
                categories = [categories]
            scope = COGXScope(
                user_id=item.get("user_id") or item.get("namespace"),
                agent_id=item.get("agent_id"),
                session_id=item.get("session_id"),
                run_id=item.get("run_id"),
            )
            yield COGXMemory(
                external_system=self.source_system,
                external_id=str(item.get("id") or f"langmem-{index}"),
                content=content,
                categories=[str(category) for category in categories],
                scope=scope,
                created_at=parse_timestamp(
                    item.get("created_at") or item.get("createdAt") or item.get("timestamp")
                ),
                updated_at=parse_timestamp(item.get("updated_at") or item.get("updatedAt")),
                metadata={"langmem_metadata": item.get("metadata")}
                if item.get("metadata")
                else {},
            )
