"""Live Mem0 API memory source."""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, List, Optional

from cognee.modules.migration.cogx import COGXRecord
from cognee.modules.migration.sources.base import MemorySource
from cognee.modules.migration.sources.live._utils import call_maybe_async, require_extra
from cognee.modules.migration.sources.mem0 import iter_mem0_records, normalize_mem0_items

_ENTITY_FILTER_KEYS = ("user_id", "agent_id", "app_id", "run_id")


class Mem0LiveSource(MemorySource):
    """Fetch memories from a running Mem0 instance via an injected client.

    Credentials and API keys must be configured on the client by the caller
    (for example ``MemoryClient(api_key=...)``). This source only paginates
    ``get_all()`` and maps results into COGX.

    The snapshot is taken on the first ``records()`` call and cached; subsequent
    calls replay the same point-in-time data (``replayable=True``).

    Args:
        client: ``mem0.MemoryClient``, ``mem0.AsyncMemoryClient``, or OSS ``Memory``.
        filters: Mem0 v3 filter dict with at least one entity id, unless
            ``discover_entities=True``.
        mode: Import fidelity mode (default ``re-derive``).
        page_size: Page size for platform ``get_all`` pagination.
        discover_entities: When True, call ``users()`` and fetch memories per
            discovered entity scope.
    """

    source_system = "mem0"
    replayable = True

    def __init__(
        self,
        client: Any,
        filters: Optional[Dict[str, Any]] = None,
        mode: str = "re-derive",
        page_size: int = 100,
        discover_entities: bool = False,
    ):
        super().__init__(mode=mode)
        self._client = client
        self._filters = filters or {}
        self._page_size = page_size
        self._discover_entities = discover_entities
        self._snapshot: Optional[List[Dict[str, Any]]] = None

    async def _fetch_all(self) -> List[Dict[str, Any]]:
        require_extra("mem0", "mem0")

        if self._discover_entities:
            return await self._fetch_discovered()
        if not any(self._filters.get(key) for key in _ENTITY_FILTER_KEYS):
            raise ValueError(
                "Mem0LiveSource requires entity-scoped filters (user_id, agent_id, "
                "app_id, or run_id) or discover_entities=True."
            )
        return await self._paginate_get_all(self._filters)

    async def _fetch_discovered(self) -> List[Dict[str, Any]]:
        users_fn = getattr(self._client, "users", None)
        if users_fn is None:
            raise ValueError("discover_entities=True requires a client with users().")
        users = await call_maybe_async(users_fn)
        if not isinstance(users, list):
            users = users.get("results", []) if isinstance(users, dict) else []

        memories: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()
        for entry in users:
            if not isinstance(entry, dict):
                continue
            entity_filters = {
                key: entry[key] for key in _ENTITY_FILTER_KEYS if entry.get(key) is not None
            }
            if not entity_filters:
                continue
            for item in await self._paginate_get_all(entity_filters):
                memory_id = str(item.get("id", ""))
                if memory_id and memory_id in seen_ids:
                    continue
                if memory_id:
                    seen_ids.add(memory_id)
                memories.append(item)
        return memories

    async def _paginate_get_all(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        get_all = getattr(self._client, "get_all", None)
        if get_all is None:
            raise ValueError(
                "Mem0 client must expose get_all(filters=..., page=..., page_size=...)."
            )

        memories: List[Dict[str, Any]] = []
        page = 1
        while True:
            response = await call_maybe_async(
                get_all, filters=filters, page=page, page_size=self._page_size
            )
            if isinstance(response, list):
                memories.extend(response)
                break
            batch = response.get("results", []) if isinstance(response, dict) else []
            memories.extend(batch)
            if not isinstance(response, dict) or not response.get("next"):
                break
            page += 1
        return memories

    async def _ensure_snapshot(self) -> List[Dict[str, Any]]:
        if self._snapshot is None:
            self._snapshot = await self._fetch_all()
        return self._snapshot

    async def records(self) -> AsyncIterator[COGXRecord]:
        items = normalize_mem0_items(await self._ensure_snapshot())
        async for record in iter_mem0_records(items, source_system=self.source_system):
            yield record
