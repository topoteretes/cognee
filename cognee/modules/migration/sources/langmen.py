"""LangMem memory source.

Reads a LangMem export and yields COGX memory records. Accepts the shapes
produced by ``store.search()`` / ``store.list_namespaces()`` dumps and by
manual JSON exports:

- a plain JSON list of memory items, each with ``key``/``value``/``namespace``
- ``{"memories": [...]}`` / ``{"items": [...]}`` wrappers
- already-parsed Python lists/dicts (for live-API integration: fetch with the
  ``langmem`` store client yourself and pass the response in)

Each memory becomes a :class:`COGXMemory` with scope taken from the first
element of ``namespace`` (conventionally the user/thread id) and timestamps
preserved. The memory content itself is read from the ``value`` dict, trying
common keys such as ``content``/``text``/``memory``/``data``.
"""

import json
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Union

from cognee.modules.migration.cogx import COGXMemory, COGXRecord, COGXScope, parse_timestamp
from cognee.modules.migration.sources.base import MemorySource

_CONTENT_KEYS = ("content", "text", "memory", "data")


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
            for key in ("memories", "items", "results"):
                if isinstance(data.get(key), list):
                    data = data[key]
                    break
            else:
                raise ValueError(
                    "Unrecognized LangMem export shape: expected a list or a dict "
                    "with a 'memories'/'items' key."
                )
        if not isinstance(data, list):
            raise ValueError("Unrecognized LangMem export shape: expected a list of memories.")
        return [item for item in data if isinstance(item, dict)]

    @staticmethod
    def _extract_content(value: Dict[str, Any]) -> Union[str, None]:
        return next(
            (value[key] for key in _CONTENT_KEYS if isinstance(value.get(key), str)), None
        )

    @staticmethod
    def _extract_scope_id(namespace: Any) -> Union[str, None]:
        if isinstance(namespace, (list, tuple)) and namespace:
            return str(namespace[0])
        if isinstance(namespace, str) and namespace:
            return namespace
        return None

    async def records(self) -> AsyncIterator[COGXRecord]:
        for index, item in enumerate(self._load_raw()):
            value = item.get("value")
            if not isinstance(value, dict):
                continue
            content = self._extract_content(value)
            if not content:
                continue

            namespace = item.get("namespace")
            scope_id = self._extract_scope_id(namespace)

            yield COGXMemory(
                external_system=self.source_system,
                external_id=str(item.get("key") or f"langmem-{index}"),
                content=content,
                categories=[str(c) for c in (value.get("categories") or [])],
                scope=COGXScope(
                    user_id=scope_id,
                    agent_id=None,
                    run_id=None,
                ),
                created_at=parse_timestamp(item.get("created_at")),
                updated_at=parse_timestamp(item.get("updated_at")),
                metadata={"langmem_namespace": namespace} if namespace else {},
            )
