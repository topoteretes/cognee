"""Google ADK memory source.

Reads serialized Google ADK ``MemoryEntry`` JSON and yields COGX memory records.

Pinned spec: ADK ``MemoryEntry`` + ``SearchMemoryResponse``
(https://github.com/google/adk-python/blob/main/src/google/adk/memory/memory_entry.py).

Accepted input shapes:

- a plain JSON list of MemoryEntry objects
- ``{"memories": [...]}`` wrapper (``SearchMemoryResponse`` serialization)
- a file path or already-parsed Python list/dict

Mapping: one ``MemoryEntry`` → one :class:`COGXMemory`. Text is taken from
``content.parts[].text``. Scope is taken from ``custom_metadata``; generic
``author`` role labels (``"user"``, ``"model"``) are not mapped to ``user_id``.
"""

import json
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Union

from cognee.modules.migration.cogx import COGXMemory, COGXRecord, COGXScope, parse_timestamp
from cognee.modules.migration.sources.base import MemorySource

_GENERIC_AUTHORS = frozenset({"user", "model"})


class GoogleMemorySource(MemorySource):
    source_system = "google_adk"

    def __init__(self, data: Union[str, Path, List[Any], Dict[str, Any]], mode: str = "re-derive"):
        super().__init__(mode=mode)
        self._data = data

    def _load_raw(self) -> List[Dict[str, Any]]:
        data = self._data
        if isinstance(data, (str, Path)):
            data = json.loads(Path(data).read_text(encoding="utf-8"))
        if isinstance(data, dict):
            memories = data.get("memories")
            if isinstance(memories, list):
                data = memories
            else:
                raise ValueError(
                    "Unrecognized Google ADK export shape: expected a list or a dict "
                    "with a 'memories' key."
                )
        if not isinstance(data, list):
            raise ValueError(
                "Unrecognized Google ADK export shape: expected a list of MemoryEntry objects."
            )
        return [item for item in data if isinstance(item, dict)]

    @staticmethod
    def _extract_text_from_content(content: Any) -> str:
        if not isinstance(content, dict):
            return ""
        parts = content.get("parts")
        if not isinstance(parts, list):
            return ""
        texts = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                texts.append(text)
        return "\n".join(texts)

    @classmethod
    def _scope_from_entry(cls, entry: Dict[str, Any]) -> COGXScope:
        metadata = entry.get("custom_metadata")
        if not isinstance(metadata, dict):
            metadata = {}

        user_id = metadata.get("user_id")
        if not user_id:
            author = entry.get("author")
            if isinstance(author, str) and author and author not in _GENERIC_AUTHORS:
                user_id = author

        return COGXScope(
            user_id=user_id,
            agent_id=metadata.get("app_name"),
            session_id=metadata.get("session_id"),
        )

    @staticmethod
    def _metadata_from_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
        metadata = entry.get("custom_metadata")
        if isinstance(metadata, dict) and metadata:
            return dict(metadata)
        return {}

    async def records(self) -> AsyncIterator[COGXRecord]:
        for index, entry in enumerate(self._load_raw()):
            content = self._extract_text_from_content(entry.get("content"))
            if not content:
                continue
            yield COGXMemory(
                external_system=self.source_system,
                external_id=str(entry.get("id") or f"google-adk-{index}"),
                content=content,
                scope=self._scope_from_entry(entry),
                created_at=parse_timestamp(entry.get("timestamp")),
                metadata=self._metadata_from_entry(entry),
            )
